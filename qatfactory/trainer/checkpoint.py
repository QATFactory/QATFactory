"""Shared quantization-aware checkpoint saving for the distillation and RL trainers.

Both trainers produce the *same* artifact: BF16 latent weights, the original (deployment)
config annotated with ``qatfactory_qad`` metadata, the tokenizer assets, and — for
fused-runtime-scale formats — the recorded activation scales. Keeping one implementation
means a QAD checkpoint and a QARL checkpoint are interchangeable inputs to the exporters.

A host trainer supplies the quantization context through six attributes:
``quant_format``, ``fused_runtime_scales``, ``converted``, ``original_config_path``,
``fused_process_group`` and ``lora_metadata``.
"""

import contextlib
import json
import logging
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file

from ..quant.export.nvfp4_vllm import ShardedSafetensorWriter
from ..quant.formats import get_quant_format
from ..quant.formats.nvfp4_experts import EXPERT_PROJECTIONS


logger = logging.getLogger(__name__)

# Raw per-module running input maxima, persisted per checkpoint so a resumed run
# continues them (W4A4 additionally derives its step-static scales from this state).
RUNTIME_STATE_FILENAME = "nvfp4_runtime_state.pt"
# LoRA runs save their adapters here instead of a full copy of the frozen backbone.
ADAPTER_FILENAME = "lora_adapters.safetensors"
# Dtypes worth narrowing to BF16 on save; a frozen FP8 base is already narrower and is kept.
_WIDE_FLOAT_DTYPES = (torch.float32, torch.float64, torch.float16, torch.bfloat16)
# Released-checkpoint prefixes with no in-memory counterpart (matches the exporters' carry-drop list).
_UNTRAINED_ONDISK_PREFIXES = ("mtp", "model.mtp")
# Shard size when a checkpoint's BF16 weights are written while they are gathered.
_STREAMED_SHARD_BYTES = 5 * 1024**3


class QuantCheckpointMixin:
    """``save_model`` for a fake-quantized student, plus checkpoint-resume of scale state."""

    # Tokenizer/vocab assets the checkpoint needs to be self-contained for GGUF conversion
    # (convert_hf_to_gguf.py requires these to build the BPE vocab + pre-tokenizer hash).
    _TOKENIZER_ASSET_FILES = (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    )

    def _qad_metadata(self, *, target_modules: list[str], global_step: int) -> dict:
        metadata = {
            "format_version": 1,
            "quant_format": self.quant_format,
            "quantizer": get_quant_format(self.quant_format).quantizer_id,
            "fused_runtime_scales": bool(self.fused_runtime_scales),
            "input_scale_source": "training_running_max" if self.fused_runtime_scales else None,
            # W4A4 provenance (derived from the live modules, not config): the exported
            # artifact is schema-identical either way, so this is the only durable record
            # of whether activations were fake-quantized during training.
            "quantize_activations": any(
                getattr(module, "_quantize_activations", False) for module in self.model.modules()
            ),
            "target_modules": target_modules,
            "global_step": global_step,
        }
        if self.converted.lora:
            # Adapters are saved unmerged inside the checkpoint; the exporter folds them into
            # the base weight with this scaling before packing. This is the durable record of
            # how to reconstruct the deployed (merged) weight.
            metadata["lora"] = dict(self.lora_metadata)
        return metadata

    @staticmethod
    def _translate_to_ondisk(inmem_module: str, ondisk_keys: set[str]) -> str:
        """Map an in-memory module FQN to the on-disk one (HF may rename on save).

        e.g. ``model.layers.3.self_attn.q_proj`` -> ``model.language_model.layers.3.self_attn.q_proj``.
        Matched by the unique layer-qualified suffix so it is convention-agnostic. Routed
        expert targets (``...experts.7.w1``) have no tensor of their own — they are slices of
        the fused ``gate_up_proj`` / ``down_proj`` — so the experts *module* is located instead
        and the per-expert suffix re-appended.
        """

        def locate(module: str, leaf: str) -> str | None:
            if f"{module}.{leaf}" in ondisk_keys:
                return module
            core = module.split(".", 1)[1] if "." in module else module
            matches = [key[: -len(f".{leaf}")] for key in ondisk_keys if key.endswith(f".{core}.{leaf}")]
            if len(matches) == 1:
                return matches[0]
            if matches:
                raise RuntimeError(f"Could not uniquely map {module!r} to an on-disk key; matches={matches[:3]}.")
            return None

        marker = ".experts."
        if marker in inmem_module:
            root, _, suffix = inmem_module.partition(marker)
            parts = suffix.split(".")
            if len(parts) == 2 and parts[0].isdigit() and parts[1] in EXPERT_PROJECTIONS:
                experts_module = f"{root}{marker[:-1]}"
                leaf = "down_proj" if parts[1] == "w2" else "gate_up_proj"
                located = locate(experts_module, leaf)
                if located is None:
                    raise RuntimeError(f"Could not map fused experts module {experts_module!r} to an on-disk key.")
                return f"{located}.{parts[0]}.{parts[1]}"
        if inmem_module.endswith(marker[:-1]):
            # The fused experts module itself (owner of the per-layer activation scales).
            located = locate(inmem_module, "gate_up_proj")
            if located is None:
                raise RuntimeError(f"Could not map fused experts module {inmem_module!r} to an on-disk key.")
            return located
        located = locate(inmem_module, "weight")
        if located is None:
            raise RuntimeError(f"Could not uniquely map {inmem_module!r} to an on-disk key; matches=[].")
        return located

    @classmethod
    def _adapter_translator(cls, ondisk_keys: set[str]):
        """Name map for an adapters-only save: dense targets take the on-disk spelling of the
        original checkpoint; routed experts (``...experts`` and ``...experts.<i>.<w>``) stay in
        the in-memory fused layout, which has no per-tensor counterpart in a Hub checkpoint."""

        # The original checkpoint also holds the multi-token-prediction head (Qwen3.5 ``mtp.*``),
        # which is neither trained nor exported (the exporters drop the same prefixes) and
        # whose layers share the layer-qualified suffixes of the backbone.
        backbone_keys = {
            key for key in ondisk_keys if not any(key == p or key.startswith(f"{p}.") for p in _UNTRAINED_ONDISK_PREFIXES)
        }

        def translate(name: str) -> str:
            if ".experts." in name or name.endswith(".experts"):
                return name
            return cls._translate_to_ondisk(name, backbone_keys)

        return translate

    @classmethod
    def _rename_under_owner(cls, key: str, owners: dict[str, str]) -> str:
        """Re-prefix a tensor key (``<module>.lora_A`` ...) with its owning module's new name."""

        try:
            module = cls._owning_converted_module(key, owners)
        except KeyError:
            return key
        return f"{owners[module]}{key[len(module):]}"

    @staticmethod
    def _owning_converted_module(scale_key: str, name_map: dict[str, str]) -> str:
        """Longest converted-module FQN that prefixes an input-scale key."""
        module = scale_key
        while module and module not in name_map:
            module = module.rpartition(".")[0]
        if not module:
            raise KeyError(f"Input scale {scale_key!r} does not belong to any converted module.")
        return module

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        # finalize_input_scales and get_state_dict are collective -> run on every rank.
        input_scales = (
            get_quant_format(self.quant_format).finalize_input_scales(self.model, self.fused_process_group)
            if self.fused_runtime_scales
            else {}
        )
        # A LoRA run never changes the base weights, so persisting them would copy the whole
        # backbone on every save — hundreds of gigabytes for a large MoE. Save the adapters
        # alone; the exporter reads the untouched base from the original model assets.
        adapters_only = bool(self.converted.lora)
        # Routed-expert checkpoints keep the in-memory layout (no Hub-format rename, see below), so
        # their BF16 weights are written shard by shard as they are gathered: the saving host then
        # holds one shard rather than the whole model (about 460 GB for MiniMax-M2.7).
        stream = bool(self.converted.experts) and not adapters_only and get_quant_format(self.quant_format).pack_weight is None
        writer = None
        if stream and self.args.should_save:
            os.makedirs(output_dir, exist_ok=True)
            writer = ShardedSafetensorWriter(Path(output_dir), _STREAMED_SHARD_BYTES)
        drop_head = self._deployment_is_tied()

        def write(key, value):
            if not (drop_head and (key == "lm_head.weight" or key.endswith(".lm_head.weight"))):
                writer.add(key, value)

        bf16_state = self._gathered_bf16_state(
            self.converted.trainable_parameters if adapters_only else None, sink=write if stream else None
        )
        if writer is not None:
            writer.finalize()

        if self.args.should_save:
            os.makedirs(output_dir, exist_ok=True)
            # Ephemeral untie: QAD unties lm_head from the input embedding only so FSDP2 can
            # shard two distinct storages (see models.llm._untie_frozen_lm_head). When the
            # deployment config keeps embeddings tied, the head is redundant — drop the cloned
            # lm_head.weight so the saved checkpoint matches the tied layout (tie_word_embeddings
            # stays True, no lm_head tensor) instead of persisting a contradictory copy.
            if self._deployment_is_tied():
                bf16_state = {
                    key: value
                    for key, value in bf16_state.items()
                    if key != "lm_head.weight" and not key.endswith(".lm_head.weight")
                }
            unwrapped = self.accelerator.unwrap_model(self.model)
            if adapters_only:
                # `save_file` writes the in-memory FQNs verbatim (no Hub-format rename), so give
                # dense targets the released checkpoint's spelling, looked up in the original
                # model's weights: the exporter and scripts/merge_lora_bf16.py read the adapters
                # next to that checkpoint (Qwen3.5 keeps its text tower under
                # model.language_model.*). Routed experts keep the in-memory fused layout, which
                # is what merge_lora_bf16.py consumes for MoE adapters.
                translate = self._adapter_translator(
                    self._read_ondisk_weight_keys(str(Path(self.original_config_path).parent))
                )
                owners = {name: translate(name) for name in (*self.converted.all, *self.converted.experts)}
                save_file(
                    {self._rename_under_owner(key, owners): value for key, value in bf16_state.items()},
                    os.path.join(output_dir, ADAPTER_FILENAME),
                )
                unwrapped.config.save_pretrained(output_dir)
            elif stream:
                unwrapped.config.save_pretrained(output_dir)
                if getattr(unwrapped, "generation_config", None) is not None:
                    unwrapped.generation_config.save_pretrained(output_dir)
            else:
                unwrapped.save_pretrained(
                    output_dir,
                    state_dict=bf16_state,
                    safe_serialization=True,
                    # Keep the instrumented in-memory layout for routed experts: the Hub format
                    # splits the fused tensors back into per-expert w1/w3/w2, which would strand
                    # the adapters. The exporter performs the serving-layout split after
                    # folding LoRA instead.
                    save_original_format=not self.converted.experts,
                )

            if not adapters_only:
                ondisk_keys = self._read_ondisk_weight_keys(output_dir)
                translate = lambda name: self._translate_to_ondisk(name, ondisk_keys)  # noqa: E731
            # Deployed targets (what the QAD metadata and packed weights speak in).
            name_map = {name: translate(name) for name in self.converted.all}
            self._write_original_config(output_dir, sorted(name_map.values()))
            self._save_tokenizer_assets(output_dir)

            if self.fused_runtime_scales:
                # Linears emit `<module>.input_scale`; fused experts emit one scale per serving
                # weight keyed by the experts *module* (`<experts>.w13.input_scale`,
                # `<experts>.w2.input_scale`), which is not itself a deployed target.
                scale_owners = {**name_map, **{name: translate(name) for name in self.converted.experts}}
                translated = {}
                for key, value in input_scales.items():
                    module = self._owning_converted_module(key, scale_owners)
                    translated[f"{scale_owners[module]}{key[len(module):]}"] = value.detach().cpu().float()
                path = os.path.join(output_dir, "input_scales.safetensors")
                save_file(translated, path)
                logger.info("Saved %d training activation scales to %s.", len(translated), path)
                # Persist raw running maxima so resume_from_checkpoint continues them
                # (finalize_input_scales just all-reduced them, so this is rank-consistent).
                runtime_state_dict = get_quant_format(self.quant_format).runtime_state_dict
                if runtime_state_dict is not None:
                    torch.save(runtime_state_dict(self.model), os.path.join(output_dir, RUNTIME_STATE_FILENAME))

            self._save_packed_weights(output_dir, bf16_state, name_map)

        if dist.is_initialized():
            dist.barrier()

    def _gathered_bf16_state(self, keep=None, sink=None) -> dict:
        """The full model state dict on the saving rank, with wide floats narrowed to BF16.

        The trainable latent weights hold an FP32 master; deployment (and the exporter) use BF16.
        Already-narrow dtypes (a block-FP8 frozen base) are left exactly as they are. ``keep``
        restricts the result to those keys.

        Under FSDP2, accelerate gathers the whole FP32 state onto the main rank's host before
        anything can be narrowed, which needs four bytes per parameter on one node (about 0.9 TB
        for MiniMax-M2.7) on top of the BF16 copy. Here each tensor is narrowed on its shard and
        gathered one at a time instead, so the main rank holds a single BF16 copy. Collective:
        every rank walks the same keys in the same order. With ``sink``, the saving rank hands each
        gathered tensor to ``sink(key, tensor)`` instead of keeping it, and the result is empty.
        """

        def narrow(value):
            return value.to(torch.bfloat16) if value.dtype in _WIDE_FLOAT_DTYPES else value

        if not getattr(self.accelerator, "is_fsdp2", False):
            state_dict = self.accelerator.get_state_dict(self.model)
            if not self.args.should_save:
                return {}
            kept = {key: narrow(value) for key, value in state_dict.items() if keep is None or key in keep}
            if sink is None:
                return kept
            for key, value in kept.items():
                sink(key, value)
            return {}

        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
        from torch.distributed.tensor import DTensor

        sharded = get_model_state_dict(self.model, options=StateDictOptions(full_state_dict=False))
        gathered = {}
        for key, value in sharded.items():
            if keep is not None and key not in keep:
                continue
            value = value.detach()
            if value.is_cpu:  # parameters offloaded by FSDP2 are gathered through the device
                value = value.to(self.accelerator.device)
            value = narrow(value)
            if isinstance(value, DTensor):
                value = value.full_tensor()
            if self.args.should_save:
                if sink is None:
                    gathered[key] = value.cpu()
                else:
                    sink(key, value.cpu())
            del value
        return gathered

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        with _version_only_checkpoint_config():
            super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        load_runtime_state_dict = get_quant_format(self.quant_format).load_runtime_state_dict
        if load_runtime_state_dict is None or not self.fused_runtime_scales:
            return
        path = os.path.join(resume_from_checkpoint, RUNTIME_STATE_FILENAME)
        if os.path.isfile(path):
            load_runtime_state_dict(self.model, torch.load(path, map_location="cpu"))
            logger.info("Restored %s running input maxima from %s.", self.quant_format, path)
        else:
            logger.warning(
                "Checkpoint %s has no %s; input running maxima restart from this point.",
                resume_from_checkpoint,
                RUNTIME_STATE_FILENAME,
            )

    def _save_tokenizer_assets(self, output_dir: str) -> None:
        """Copy tokenizer/vocab files from the source model dir into the checkpoint.

        The student loads as a causal LM and ``save_pretrained`` does not emit tokenizer
        files, so without this the exported checkpoint has no vocab and ``convert_hf_to_gguf.py``
        fails ("BPE pre-tokenizer was not recognized"). Copy them so the checkpoint is
        self-contained for both the Q4_K GGUF export and normal Hugging Face loading.
        """

        source_dir = Path(self.original_config_path).parent
        copied = 0
        for filename in self._TOKENIZER_ASSET_FILES:
            src = source_dir / filename
            dst = Path(output_dir) / filename
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
        logger.info("Copied %d tokenizer asset files into %s.", copied, output_dir)

    def _save_packed_weights(self, output_dir: str, bf16_state: dict, name_map: dict[str, str]) -> None:
        """Persist the exact low-bit blocks for save-time-packing formats (e.g. Q4_K).

        The deployable blocks are computed once, here, on the training device from the final
        latent weights, and carried through export unchanged. This makes the export a pure,
        device-independent splice and guarantees the deployed GGUF dequantizes value-for-value
        to the trained student.
        """

        quant = get_quant_format(self.quant_format)
        if quant.pack_weight is None:
            return
        device = self.accelerator.device
        unwrapped = self.accelerator.unwrap_model(self.model)
        packed = {}
        for inmem_name in self.converted.linears:
            weight = bf16_state[f"{inmem_name}.weight"].to(device)
            try:
                module = unwrapped.get_submodule(inmem_name)
            except AttributeError:
                module = None
            packed[name_map[inmem_name]] = quant.pack_weight(weight, module).detach().cpu().contiguous()
            del weight
        path = os.path.join(output_dir, quant.packed_weight_filename)
        save_file(packed, path)
        logger.info("Saved %d packed %s weight blocks to %s.", len(packed), self.quant_format, path)

    @staticmethod
    def _read_ondisk_weight_keys(output_dir: str) -> set[str]:
        """Saved weight keys: linear ``.weight`` tensors and fused 3D expert tensors.

        Fused experts are stored as bare ``<experts>.gate_up_proj`` / ``<experts>.down_proj``
        parameters (no ``.weight`` leaf); without them a full-parameter MoE checkpoint cannot
        map its expert targets to on-disk names.
        """

        suffixes = (".weight", ".gate_up_proj", ".down_proj")
        index = Path(output_dir) / "model.safetensors.index.json"
        if index.is_file():
            with index.open(encoding="utf-8") as handle:
                return {key for key in json.load(handle)["weight_map"] if key.endswith(suffixes)}
        keys: set[str] = set()
        for shard in Path(output_dir).glob("*.safetensors"):
            if shard.name == "input_scales.safetensors":
                continue
            with safe_open(shard, framework="pt") as handle:
                keys.update(key for key in handle.keys() if key.endswith(suffixes))
        return keys

    def _deployment_is_tied(self) -> bool:
        """True if the original (deployment) config keeps input embeddings and lm_head tied.

        The served config is the original one (see :meth:`_write_original_config`), so its
        ``tie_word_embeddings`` — not the in-memory value QAD flips to untie for FSDP2 — decides
        whether a standalone ``lm_head`` belongs in the checkpoint.
        """

        with open(self.original_config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        text_config = config.get("text_config")
        return bool(
            config.get("tie_word_embeddings")
            or (isinstance(text_config, dict) and text_config.get("tie_word_embeddings"))
        )

    def _write_original_config(self, output_dir: str, ondisk_target_modules: list[str]) -> None:
        """Overwrite the saved (flattened) config with the original full config + QAD metadata.

        This keeps parity with the deployment checkpoint the exporter/vLLM expect (full
        multimodal ``qwen3_5`` config with ``text_config``), rather than the text-only config
        the causal-LM head emits.
        """

        with open(self.original_config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        config["qatfactory_qad"] = self._qad_metadata(
            target_modules=ondisk_target_modules,
            global_step=int(self.state.global_step),
        )
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")


@contextlib.contextmanager
def _version_only_checkpoint_config():
    """Let ``Trainer._load_from_checkpoint`` read the checkpoint's transformers version without
    parsing the whole config.

    The base method builds a generic ``PreTrainedConfig`` from ``config.json`` only to warn when
    the checkpoint was written by another transformers version. Under transformers 5.x that
    generic parse standardizes the RoPE parameters, and for ``llama3``/``yarn``/``longrope``
    scaling it reads ``max_position_embeddings``, which the base class does not have, so
    resuming a DeepSeek-R1-Distill-Llama run raised ``AttributeError`` before anything loaded.
    """
    import transformers.trainer as hf_trainer

    name = "PreTrainedConfig" if hasattr(hf_trainer, "PreTrainedConfig") else "PretrainedConfig"
    original = getattr(hf_trainer, name, None)
    if original is None:
        yield
        return

    class _VersionOnlyConfig:
        @staticmethod
        def from_json_file(json_file):
            with open(json_file, encoding="utf-8") as handle:
                return SimpleNamespace(transformers_version=json.load(handle).get("transformers_version"))

    setattr(hf_trainer, name, _VersionOnlyConfig)
    try:
        yield
    finally:
        setattr(hf_trainer, name, original)
