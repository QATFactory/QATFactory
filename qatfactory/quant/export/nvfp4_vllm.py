"""Pack a fused-scale QAD checkpoint into vLLM's compressed-tensors NVFP4 layout.

This is a direct, dependency-light packer (no ModelOpt / llm-compressor): it reads the
trained BF16 latent weights + recorded training input scales and emits
``nvfp4-pack-quantized`` tensors (``weight_packed`` u8, ``weight_scale`` E4M3,
``weight_global_scale`` and ``input_global_scale`` reciprocal FP32) with a matching
``quantization_config`` in ``config.json``.
"""

from __future__ import annotations

import json
import re
import os
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ... import __version__
from ..formats.lora import merge_lora_weight
from ..formats.nvfp4 import build_nvfp4_fused_family_names, dequantize_nvfp4, quantize_nvfp4


# Non-target weights under these prefixes are not carried into the served checkpoint
# (multi-token-prediction head is unused by vLLM); everything else frozen is kept in BF16.
_CARRY_DROP_PREFIXES = ("mtp", "model.mtp")


_ASSET_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SafetensorDirectory:
    """Random-access reader for a sharded or unsharded HF checkpoint."""

    def __init__(self, directory: str | os.PathLike):
        self.directory = Path(directory)
        index_path = self.directory / "model.safetensors.index.json"
        if index_path.is_file():
            with index_path.open(encoding="utf-8") as file:
                self.weight_map = json.load(file)["weight_map"]
        else:
            self.weight_map = {}
            for path in sorted(self.directory.glob("*.safetensors")):
                if path.name == "input_scales.safetensors":
                    continue
                with safe_open(path, framework="pt") as file:
                    for key in file.keys():
                        if key in self.weight_map:
                            raise ValueError(f"Duplicate tensor {key!r} in {self.directory}.")
                        self.weight_map[key] = path.name
        if not self.weight_map:
            raise FileNotFoundError(f"No model safetensors found under {self.directory}.")

    def keys(self) -> set[str]:
        return set(self.weight_map)

    def get_tensor(self, key: str) -> torch.Tensor:
        try:
            path = self.directory / self.weight_map[key]
        except KeyError as exc:
            raise KeyError(f"Tensor {key!r} is missing from {self.directory}.") from exc
        with safe_open(path, framework="pt") as file:
            return file.get_tensor(key)

    def get_dtype_shape(self, key: str) -> tuple[str, tuple[int, ...]]:
        try:
            path = self.directory / self.weight_map[key]
        except KeyError as exc:
            raise KeyError(f"Tensor {key!r} is missing from {self.directory}.") from exc
        with safe_open(path, framework="pt") as file:
            tensor_slice = file.get_slice(key)
            return tensor_slice.get_dtype(), tuple(tensor_slice.get_shape())


class _BaseWeightOverlay:
    """Source checkpoint whose target base weights live in the original model assets.

    A LoRA-QAD run saves only its adapters (the base never changes), so the frozen base weight
    of every target is read from the released checkpoint and the adapters from the run.
    """

    def __init__(self, source: SafetensorDirectory, assets: SafetensorDirectory, base_keys: set[str]):
        self.source = source
        self.assets = assets
        self.base_keys = set(base_keys)

    def keys(self) -> set[str]:
        return self.source.keys() | self.base_keys

    def get_tensor(self, key: str) -> torch.Tensor:
        return (self.assets if key in self.base_keys else self.source).get_tensor(key)

    def get_dtype_shape(self, key: str) -> tuple[str, tuple[int, ...]]:
        return (self.assets if key in self.base_keys else self.source).get_dtype_shape(key)


class ShardedSafetensorWriter:
    """Incrementally write canonical Hugging Face safetensors shards."""

    def __init__(self, directory: Path, shard_size: int):
        if shard_size <= 0:
            raise ValueError("shard_size must be positive.")
        self.directory = directory
        self.shard_size = shard_size
        self.current: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.current_size = 0
        self.parts: list[tuple[Path, list[str]]] = []
        self.total_size = 0
        self.tensor_count = 0
        self.seen_keys: set[str] = set()

    def add(self, key: str, tensor: torch.Tensor) -> None:
        if key in self.seen_keys:
            raise ValueError(f"Duplicate output tensor {key!r}.")
        tensor = tensor.detach().cpu().contiguous()
        tensor_size = tensor.numel() * tensor.element_size()
        if self.current and self.current_size + tensor_size > self.shard_size:
            self._flush()
        self.current[key] = tensor
        self.current_size += tensor_size
        self.total_size += tensor_size
        self.tensor_count += 1
        self.seen_keys.add(key)

    def _flush(self) -> None:
        if not self.current:
            return
        part_path = self.directory / f".part-{len(self.parts) + 1:05d}.safetensors"
        keys = list(self.current)
        save_file(self.current, part_path, metadata={"format": "pt"})
        self.parts.append((part_path, keys))
        self.current = OrderedDict()
        self.current_size = 0

    def finalize(self) -> None:
        self._flush()
        if not self.parts:
            raise ValueError("Cannot finalize an empty checkpoint.")
        if len(self.parts) == 1:
            os.replace(self.parts[0][0], self.directory / "model.safetensors")
            return
        weight_map = {}
        shard_count = len(self.parts)
        for index, (part_path, keys) in enumerate(self.parts, start=1):
            filename = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
            os.replace(part_path, self.directory / filename)
            weight_map.update(dict.fromkeys(keys, filename))
        with (self.directory / "model.safetensors.index.json").open("w", encoding="utf-8") as file:
            json.dump(
                {"metadata": {"total_size": self.total_size}, "weight_map": weight_map},
                file,
                indent=2,
                sort_keys=True,
            )
            file.write("\n")


def _module_key(module: str, leaf: str) -> str:
    return f"{module}.{leaf}" if module else leaf


def _read_target_weight(
    store: "SafetensorDirectory",
    module: str,
    lora_scaling: float | None,
) -> torch.Tensor:
    """Read a target's latent weight, folding in a LoRA adapter when the run trained one.

    LoRA-QAD keeps the adapter unmerged, exactly matching what the student quantized during
    training (``fake_quant(W + scaling * B @ A)``); merging here makes the packer identical to
    the plain-QAD path from that point on. For an adapters-only save, the frozen base weight
    ``W`` is read from the original model assets (see ``_BaseWeightOverlay``).
    """

    weight = store.get_tensor(_module_key(module, "weight"))
    if lora_scaling is None:
        return weight
    lora_a = store.get_tensor(_module_key(module, "lora_A"))
    lora_b = store.get_tensor(_module_key(module, "lora_B"))
    return merge_lora_weight(weight, lora_a, lora_b, lora_scaling)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


# Routed-expert targets are recorded per expert in serving naming: ``<experts>.<e>.w1``
# (gate), ``w3`` (up) and ``w2`` (down), sliced from the fused ``gate_up_proj`` /
# ``down_proj`` tensors the trainer saves.
EXPERT_PROJECTIONS = ("w1", "w3", "w2")
_EXPERT_TARGET_RE = re.compile(r"^(?P<module>.+\.experts)\.(?P<expert>\d+)\.(?P<projection>w[123])$")
# HF per-expert projection names, which vLLM's qwen3_moe loader reads.
_HF_EXPERT_PROJECTIONS = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
_MOE_MODEL_TYPES = ("qwen3_moe", "minimax_m2")


def _expert_targets(module: str, num_experts: int) -> set[str]:
    return {f"{module}.{expert}.{projection}" for expert in range(num_experts) for projection in EXPERT_PROJECTIONS}


def _split_expert_targets(targets) -> tuple[list[str], dict[str, int]]:
    """Separate linear targets from per-expert ones; return ``(linears, {experts module: count})``."""

    linears, experts = [], {}
    for name in targets:
        match = _EXPERT_TARGET_RE.match(name)
        if match is None:
            linears.append(name)
        else:
            module = match["module"]
            experts[module] = max(experts.get(module, 0), int(match["expert"]) + 1)
    return sorted(linears), experts


def _to_minimax_serving_key(key: str) -> str:
    """transformers ``minimax_m2`` namespace -> vLLM's serving-checkpoint namespace.

    MoE-block keys under ``model.layers.N.mlp.`` (router ``gate``, ``experts``,
    ``e_score_correction_bias``) move under ``block_sparse_moe``, and per-expert projections
    use the released ``w1``/``w3``/``w2`` names. Every other key passes through.
    """

    head, sep, rest = key.partition(".mlp.")
    if not sep:
        return key
    parts = rest.split(".")
    reverse = {hf: serving for serving, hf in _HF_EXPERT_PROJECTIONS.items()}
    if parts[0] == "experts" and len(parts) > 2 and parts[2] in reverse:
        parts[2] = reverse[parts[2]]
    return f"{head}.block_sparse_moe.{'.'.join(parts)}"


def _expected_targets(config: dict[str, Any]) -> set[str]:
    model_type = config.get("model_type")
    if model_type in _MOE_MODEL_TYPES:
        # Uniform full-attention MoE decoders. Qwen3-MoE may interleave dense MLP layers
        # (``mlp_only_layers`` / ``decoder_sparse_step``); MiniMax-M2 has none.
        text_config = config
        num_experts = int(text_config.get("num_experts") or text_config.get("num_local_experts") or 0)
        mlp_only = set(text_config.get("mlp_only_layers") or [])
        sparse_step = int(text_config.get("decoder_sparse_step") or 1)
        targets = set()
        for index in range(int(text_config["num_hidden_layers"])):
            layer = f"model.layers.{index}"
            targets.update(f"{layer}.self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
            if num_experts and index not in mlp_only and (index + 1) % sparse_step == 0:
                targets |= _expert_targets(f"{layer}.mlp.experts", num_experts)
            else:
                targets.update(f"{layer}.mlp.{name}" for name in ("gate_proj", "up_proj", "down_proj"))
        return targets
    if model_type in ("qwen3", "llama"):
        text_config = config
        prefix = "model.layers"
        layer_types = ["full_attention"] * int(text_config["num_hidden_layers"])
    elif model_type == "qwen3_5":
        text_config = config["text_config"]
        prefix = "model.language_model.layers"
        layer_types = text_config["layer_types"]
    elif model_type == "gemma4_unified":
        text_config = config["text_config"]
        prefix = "model.language_model.layers"
        layer_types = text_config["layer_types"]
    else:
        raise ValueError(f"Native NVFP4 export does not support model_type={model_type!r}.")

    # Gemma-4's `full_attention` (global) layers reuse the key projection as the value
    # (`attention_k_eq_v`, one global KV head), so they carry q/k/o but no `v_proj`; the
    # `sliding_attention` layers are ordinary q/k/v/o. Every other backbone uses q/k/v/o for
    # its full-attention layers.
    is_gemma4 = model_type == "gemma4_unified"
    targets = set()
    for index, layer_type in enumerate(layer_types):
        layer = f"{prefix}.{index}"
        if layer_type in ("full_attention", "sliding_attention"):
            attn_names = ("q_proj", "k_proj", "o_proj") if (is_gemma4 and layer_type == "full_attention") else (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            )
            targets.update(f"{layer}.self_attn.{name}" for name in attn_names)
        elif layer_type == "linear_attention":
            targets.update(
                f"{layer}.linear_attn.{name}"
                for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
            )
        else:
            raise ValueError(f"Unsupported layer type {layer_type!r} at layer {index}.")
        targets.update(f"{layer}.mlp.{name}" for name in ("gate_proj", "up_proj", "down_proj"))
    return targets


def unquantized_ignore_entries(unquantized: list[str], quantizable: set[str]) -> list[str]:
    r"""compressed-tensors ``ignore`` entries for quantizable linears a run left in BF16.

    vLLM checks the ignore list per *shard* of its fused layers (``qkv_proj`` -> q/k/v,
    ``gate_up_proj`` -> gate/up) and requires one answer for the whole fused layer, and its
    fused shard list can name projections the checkpoint does not have (Gemma-4's k_eq_v
    global layers carry no ``v_proj``). So when every quantizable linear of a block is
    unquantized, the block is ignored by a name-agnostic regex over its parent
    (``re:.*\.layers\.5\.self_attn\..*``), which also survives the HF -> vLLM prefix
    remap; a partially quantized block falls back to explicit module names.
    """

    skipped = set(unquantized)
    by_parent: dict[str, set[str]] = {}
    for module in quantizable:
        by_parent.setdefault(module.rpartition(".")[0], set()).add(module)
    entries: list[str] = []
    for parent, members in sorted(by_parent.items()):
        block = members & skipped
        if not block:
            continue
        if block == members and ".layers." in parent:
            tail = parent.split(".layers.", 1)[1]
            entries.append(f"re:.*\\.layers\\.{re.escape(tail)}\\..*")
        else:
            entries.extend(sorted(block))
    return entries


def _moe_serving_overrides(config: dict[str, Any], has_experts: bool) -> tuple[list[str], list[str]]:
    """Extra ``ignore`` entries and quantization ``targets`` for MoE serving checkpoints."""

    if not has_experts:
        return [], ["Linear"]
    if config.get("model_type") == "minimax_m2":
        # The router and the routing-correction bias serve unquantized, under both the
        # checkpoint names (block_sparse_moe.*) and vLLM's synthetic FusedMoE probe names
        # (mlp.*); the expert targets likewise name both spellings.
        return (
            ["re:.*block_sparse_moe\\.gate$", "re:.*e_score_correction_bias$", "re:.*mlp\\.gate$"],
            [
                "Linear",
                "re:.*block_sparse_moe\\.experts\\.\\d+\\.(w1|w2|w3)$",
                "re:.*mlp\\.experts\\.\\d+\\.(gate|up|gate_up|down)_proj$",
            ],
        )
    return ["re:.*mlp\\.gate$"], ["Linear"]


def _compressed_tensors_config(
    unquantized_modules: list[str] | None = None,
    *,
    extra_ignore: list[str] | None = None,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    shared_spec = {
        "actorder": None,
        "block_structure": None,
        "group_size": 16,
        "num_bits": 4,
        "observer_kwargs": {},
        "scale_dtype": "torch.float8_e4m3fn",
        "strategy": "tensor_group",
        "symmetric": True,
        "type": "float",
        "zp_dtype": None,
    }
    return {
        "config_groups": {
            "group_0": {
                "format": "nvfp4-pack-quantized",
                "input_activations": {
                    **shared_spec,
                    "dynamic": "local",
                    "observer": "static_minmax",
                },
                "output_activations": None,
                "targets": list(targets or ["Linear"]),
                "weights": {
                    **shared_spec,
                    "dynamic": False,
                    "observer": "memoryless_minmax",
                },
            }
        },
        "format": "nvfp4-pack-quantized",
        "global_compression_ratio": None,
        # Quantizable linears the run left in BF16 (e.g. an MLP-only recipe), see
        # unquantized_ignore_entries: the engine loads them as plain Linear.
        "ignore": [
            "lm_head",
            "re:.*visual.*",
            "re:.*mtp.*",
            "re:.*embed_vision.*",
            "re:.*embed_audio.*",
            "re:.*vision_embedder.*",
            *(extra_ignore or ()),
            *sorted(unquantized_modules or ()),
        ],
        "kv_cache_scheme": None,
        "producer": {"name": "qatfactory", "version": __version__},
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
    }


def _copy_assets(source: Path, model_assets: Path, output: Path) -> None:
    for filename in _ASSET_FILENAMES:
        for directory in (source, model_assets):
            path = directory / filename
            destination = output / filename
            if path.is_file() and not destination.exists():
                shutil.copy2(path, destination)
                break
    _repair_declared_tokenizer_class(output)


def _repair_declared_tokenizer_class(output: Path) -> None:
    """Make the exported ``tokenizer_config.json`` load correctly from a local directory.

    The serving engine loads the tokenizer from the export directory with
    ``AutoTokenizer.from_pretrained(<dir>)``, which honors the ``tokenizer_class`` declared in
    ``tokenizer_config.json``. DeepSeek-R1-Distill declares ``LlamaTokenizerFast`` over a
    byte-level BPE ``tokenizer.json``; from a local directory that builds a sentencepiece-style
    tokenizer that drops every space and non-ASCII character (a Hub id takes a different code
    path and is unaffected), so a bit-exact export serves garbled text. Training already guards
    against this in :func:`qatfactory.data.load_tokenizer` by probing an encode/decode
    round-trip and falling back to the generic fast backend; apply the same probe here and,
    when the fallback was needed, declare that backend in the exported config.
    """

    config_path = output / "tokenizer_config.json"
    if not config_path.is_file() or not (output / "tokenizer.json").is_file():
        return
    config = _load_json(config_path)
    declared = config.get("tokenizer_class")
    if declared is None:
        return
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    from ...data.text import tokenizer_roundtrips  # local import: keeps the export module light

    if tokenizer_roundtrips(AutoTokenizer.from_pretrained(str(output))):
        return
    if not tokenizer_roundtrips(PreTrainedTokenizerFast.from_pretrained(str(output))):
        return  # nothing better to offer; leave the declaration alone
    # PreTrainedTokenizerFast is the stable public name of the generic fast backend that loads
    # tokenizer.json verbatim (transformers 4.x and 5.x alike).
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[assets] tokenizer_config.json declared {declared!r}, which does not round-trip from a "
        f"local directory; exported as 'PreTrainedTokenizerFast' (tokenizer.json unchanged)."
    )


def _verify_reference_schema(output: Path, reference: Path) -> None:
    output_store = SafetensorDirectory(output)
    reference_store = SafetensorDirectory(reference)
    reference_modules = {
        key.removesuffix(".weight_packed") for key in reference_store.keys() if key.endswith(".weight_packed")
    }
    output_modules = {
        key.removesuffix(".weight_packed") for key in output_store.keys() if key.endswith(".weight_packed")
    }
    if not reference_modules <= output_modules:
        raise ValueError(
            f"Reference quantized modules are not a subset of output: "
            f"{sorted(reference_modules - output_modules)[:5]}."
        )
    suffixes = ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")
    for module in sorted(reference_modules):
        for suffix in suffixes:
            key = _module_key(module, suffix)
            if output_store.get_dtype_shape(key) != reference_store.get_dtype_shape(key):
                raise ValueError(f"Reference dtype/shape mismatch for {key}.")
    output_quant = _load_json(output / "config.json")["quantization_config"]
    reference_quant = _load_json(reference / "config.json")["quantization_config"]
    for key in ("format", "quant_method", "quantization_status"):
        if output_quant.get(key) != reference_quant.get(key):
            raise ValueError(f"Reference quantization config mismatch for {key!r}.")


def _trained_targets(metadata: dict[str, Any], quantizable: set[str], what: str) -> list[str]:
    """The linears the run actually quantized: recorded by the trainer, checked against the
    architecture's quantizable set. A recipe may quantize a subset (e.g. MLP only); the
    remaining quantizable linears stay BF16 and are declared in the engine's ``ignore`` list."""

    targets = metadata.get("target_modules")
    if not isinstance(targets, list) or not targets or not all(isinstance(name, str) for name in targets):
        raise ValueError(f"Source checkpoint is not eligible for {what}: qatfactory_qad.target_modules is missing.")
    unknown = sorted(set(targets) - quantizable)
    if unknown:
        raise ValueError(f"Source checkpoint is not eligible for {what}: unknown target modules {unknown[:5]}.")
    return sorted(set(targets))


def _validate_qad_metadata(config: dict[str, Any], quantizable: set[str]) -> dict[str, Any]:
    metadata = config.get("qatfactory_qad")
    if not isinstance(metadata, dict):
        raise ValueError("Source checkpoint is missing qatfactory_qad export metadata.")
    expected = {
        "format_version": 1,
        "quantizer": "standard_max",
        "fused_runtime_scales": True,
        "input_scale_source": "training_running_max",
    }
    mismatched = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatched:
        raise ValueError(f"Source checkpoint is not eligible for native NVFP4 export: {mismatched}.")
    _trained_targets(metadata, quantizable, "native NVFP4 export")
    if not isinstance(metadata.get("global_step"), int) or metadata["global_step"] <= 0:
        raise ValueError("Source checkpoint metadata has no positive training global_step.")
    return metadata


def _load_input_scales(
    source: Path,
    target_list: list[str],
    fused_families: list[tuple[str, ...]],
    experts_modules: list[str] = (),
) -> dict[str, torch.Tensor]:
    """Training activation scales keyed by linear FQN, and by ``<experts>.w13`` / ``<experts>.w2``
    for fused expert modules (one per serving weight group, shared by every expert)."""

    path = source / "input_scales.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Source checkpoint is missing training activation scales: {path}.")
    owners = list(target_list) + [f"{module}.{group}" for module in experts_modules for group in ("w13", "w2")]
    expected_keys = {_module_key(owner, "input_scale") for owner in owners}
    with safe_open(path, framework="pt") as file:
        actual_keys = set(file.keys())
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(f"Input-scale key mismatch; missing={missing[:5]}, extra={extra[:5]}.")
        scales = {owner: file.get_tensor(_module_key(owner, "input_scale")).reshape(()) for owner in owners}
    for module, scale in scales.items():
        if scale.dtype != torch.float32 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
            raise ValueError(f"Invalid FP32 training input scale for {module}.")
    for family in fused_families:
        reference = scales[family[0]]
        if any(not torch.equal(scales[module], reference) for module in family[1:]):
            raise ValueError(f"Fused family {family} has unequal training input scales.")
    return scales


def export_nvfp4_vllm_checkpoint(
    *,
    source: str | os.PathLike,
    model_assets: str | os.PathLike,
    output: str | os.PathLike,
    device: str = "cuda:0",
    shard_size: int = 5_000_000_000,
    reference: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Export a fused-scale QAD HF checkpoint as compressed-tensors NVFP4."""

    source = Path(source)
    model_assets = Path(model_assets)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing source config: {config_path}.")
    config = _load_json(config_path)
    quantizable = _expected_targets(config)
    qad_metadata = _validate_qad_metadata(config, quantizable)
    all_targets = _trained_targets(qad_metadata, quantizable, "native NVFP4 export")
    target_list, experts_counts = _split_expert_targets(all_targets)
    experts_modules = sorted(experts_counts)
    for module, count in experts_counts.items():
        if not _expert_targets(module, count) <= set(all_targets):
            raise ValueError(f"Source checkpoint quantized only some experts of {module}; export needs all of them.")
    targets = set(target_list)
    # Unquantized *linears* (e.g. an experts-only MoE recipe) go to the ignore list; routed
    # experts are either all quantized (checked above) or the module is not an MoE target.
    unquantized = sorted(name for name in quantizable - set(all_targets) if _EXPERT_TARGET_RE.match(name) is None)
    fused_families = build_nvfp4_fused_family_names(target_list)
    input_scales = _load_input_scales(source, target_list, fused_families, experts_modules)
    source_store = SafetensorDirectory(source)
    target_weight_keys = {_module_key(module, "weight") for module in targets}
    missing = sorted(target_weight_keys - source_store.keys())
    if missing and qad_metadata.get("lora"):
        # Adapters-only LoRA save: the frozen base weights come from the original checkpoint.
        assets_store = SafetensorDirectory(model_assets)
        if not set(missing) - assets_store.keys():
            source_store = _BaseWeightOverlay(source_store, assets_store, set(missing))
            missing = []
    if missing:
        raise ValueError(f"Source checkpoint is missing target weights: {missing[:5]}.")
    if experts_modules and qad_metadata.get("lora"):
        raise ValueError(
            "Packed NVFP4 export of LoRA-QAD routed experts is not supported; merge the adapters "
            "with scripts/merge_lora_bf16.py and serve the merged BF16 checkpoint."
        )
    experts_keys = {f"{module}.{leaf}" for module in experts_modules for leaf in ("gate_up_proj", "down_proj")}
    missing = sorted(experts_keys - source_store.keys())
    if missing:
        raise ValueError(f"Source checkpoint is missing fused expert weights: {missing[:5]}.")
    target_weight_keys |= experts_keys
    # minimax_m2 tensors are written under vLLM's serving namespace (block_sparse_moe,
    # per-expert w1/w3/w2); the original snapshot already stores every weight there
    # (possibly as FP8), so nothing is carried from it.
    is_minimax = config.get("model_type") == "minimax_m2"
    rename = _to_minimax_serving_key if is_minimax else (lambda key: key)

    # LoRA-QAD checkpoints carry unmerged adapters folded into the target weights below and
    # dropped from the passthrough set; the merge scaling is recorded in the QAD metadata.
    lora_meta = qad_metadata.get("lora")
    lora_scaling = float(lora_meta["scaling"]) if lora_meta else None
    lora_keys = {key for key in source_store.keys() if key.endswith((".lora_A", ".lora_B"))}
    if lora_scaling is not None:
        missing_adapters = sorted(
            _module_key(module, leaf)
            for module in targets
            for leaf in ("lora_A", "lora_B")
            if _module_key(module, leaf) not in source_store.keys()
        )
        if missing_adapters:
            raise ValueError(f"LoRA checkpoint is missing target adapters: {missing_adapters[:5]}.")
    elif lora_keys:
        raise ValueError("Checkpoint has LoRA adapters but no qatfactory_qad.lora metadata to merge them.")

    device = torch.device(device)
    weight_amax = {}
    for family in fused_families:
        maxima = []
        for module in family:
            weight = _read_target_weight(source_store, module, lora_scaling)
            if weight.dtype != torch.bfloat16 or not bool(torch.isfinite(weight).all()):
                raise ValueError(f"Source target weight for {module} must be finite BF16, got {weight.dtype}.")
            maxima.append(weight.abs().amax().float())
        family_amax = torch.stack(maxima).amax()
        for module in family:
            weight_amax[module] = family_amax

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    writer = ShardedSafetensorWriter(temporary, shard_size)
    try:
        # With tied embeddings the head is derived from the input embedding at serve time, so
        # drop the redundant lm_head.weight (QAD stored a frozen clone only to untie for FSDP2).
        # This matches the tied deployment layout (tie_word_embeddings stays True, no lm_head
        # tensor). Untied backbones (Qwen) keep their standalone head.
        tied_head = bool(
            config.get("tie_word_embeddings")
            or (isinstance(config.get("text_config"), dict) and config["text_config"].get("tie_word_embeddings"))
        )
        nonquantized = 0
        for key in sorted(source_store.keys() - target_weight_keys - lora_keys):
            if tied_head and (key == "lm_head.weight" or key.endswith(".lm_head.weight")):
                continue
            tensor = source_store.get_tensor(key)
            if key.endswith(".e_score_correction_bias"):
                # vLLM builds the MiniMax routing-correction bias in FP32, as released.
                tensor = tensor.float()
            elif tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
                raise ValueError(f"Non-target tensor {key!r} is not BF16: {tensor.dtype}.")
            writer.add(rename(key), tensor)
            nonquantized += 1

        # A text-only causal-LM checkpoint omits the frozen vision tower that the
        # multimodal serving model still needs. Carry those (and any other untrained,
        # non-target) weights over from the original assets in BF16, matching the
        # deployment layout. MTP (speculative) weights are dropped, as vLLM does not use them.
        carried = 0
        try:
            assets_store = None if is_minimax else SafetensorDirectory(model_assets)
        except FileNotFoundError:
            assets_store = None
        if assets_store is not None:
            # Expert weights may be stored per expert in the original snapshot; never carry
            # raw copies of anything that is emitted quantized.
            assets_experts = {
                key for key in assets_store.keys() if any(key.startswith(f"{module}.") for module in experts_modules)
            }
            for key in sorted(assets_store.keys() - source_store.keys() - target_weight_keys - assets_experts):
                if any(key == prefix or key.startswith(prefix + ".") for prefix in _CARRY_DROP_PREFIXES):
                    continue
                tensor = assets_store.get_tensor(key)
                if tensor.is_floating_point():
                    if not bool(torch.isfinite(tensor).all()):
                        raise ValueError(f"Carried asset tensor {key!r} is not finite.")
                    tensor = tensor.to(torch.bfloat16)
                writer.add(key, tensor)
                carried += 1

        emitted_scale = {}
        for index, module in enumerate(target_list, start=1):
            weight = _read_target_weight(source_store, module, lora_scaling).to(device)
            result = quantize_nvfp4(weight, tensor_amax=weight_amax[module].to(device))
            dequantized = dequantize_nvfp4(
                result.packed_weight,
                result.weight_scale,
                result.weight_scale_2,
            )
            if not torch.equal(dequantized.to(result.reconstruction.dtype), result.reconstruction):
                raise ValueError(f"Packed NVFP4 dequantization is not bit-equal for {module}.")
            writer.add(_module_key(module, "weight_packed"), result.packed_weight)
            writer.add(_module_key(module, "weight_scale"), result.weight_scale)
            writer.add(
                _module_key(module, "weight_global_scale"),
                result.weight_scale_2.reshape(1).float().reciprocal(),
            )
            writer.add(
                _module_key(module, "input_global_scale"),
                input_scales[module].reshape(1).reciprocal(),
            )
            emitted_scale[module] = result.weight_scale_2.cpu()
            del weight, result, dequantized
            _empty_cache()
            print(f"[packing] {index}/{len(target_list)} {module}", flush=True)

        for family in fused_families:
            reference_scale = emitted_scale[family[0]]
            if any(not torch.equal(emitted_scale[module], reference_scale) for module in family[1:]):
                raise ValueError(f"Fused family {family} emitted unequal weight scales.")

        expert_linears = 0
        for module_index, module in enumerate(experts_modules, start=1):
            gate_up_all = source_store.get_tensor(f"{module}.gate_up_proj")
            down_all = source_store.get_tensor(f"{module}.down_proj")
            num_experts = experts_counts[module]
            if gate_up_all.ndim != 3 or gate_up_all.shape[0] != num_experts or down_all.shape[0] != num_experts:
                raise ValueError(
                    f"Fused experts {module} have shapes {tuple(gate_up_all.shape)} / {tuple(down_all.shape)}, "
                    f"expected {num_experts} experts."
                )
            # One layer-level activation scale per serving weight group, identical for every
            # expert (vLLM reduces per-expert input scales with a max, which is then exact).
            w13_input = input_scales[f"{module}.w13"].reshape(1).reciprocal()
            w2_input = input_scales[f"{module}.w2"].reshape(1).reciprocal()
            for expert in range(num_experts):
                gate, up = gate_up_all[expert].chunk(2, dim=0)
                down = down_all[expert]
                for name, weight in (("w1", gate), ("w3", up), ("w2", down)):
                    if weight.dtype != torch.bfloat16 or not bool(torch.isfinite(weight).all()):
                        raise ValueError(f"Expert weight {module}.{expert}.{name} must be finite BF16, got {weight.dtype}.")
                gate, up, down = (weight.to(device) for weight in (gate, up, down))
                # gate and up share one amax, so their global scales are equal: the serving
                # w13 kernel applies one scale to both. Matches NVFP4Experts in training.
                w13_amax = torch.stack([gate.abs().amax().float(), up.abs().amax().float()]).amax()
                emitted = {}
                for name, weight, amax, input_scale in (
                    ("w1", gate, w13_amax, w13_input),
                    ("w3", up, w13_amax, w13_input),
                    ("w2", down, down.abs().amax().float(), w2_input),
                ):
                    expert_module = f"{module}.{expert}.{_HF_EXPERT_PROJECTIONS[name]}"
                    result = quantize_nvfp4(weight, tensor_amax=amax)
                    dequantized = dequantize_nvfp4(result.packed_weight, result.weight_scale, result.weight_scale_2)
                    if not torch.equal(dequantized.to(result.reconstruction.dtype), result.reconstruction):
                        raise ValueError(f"Packed NVFP4 dequantization is not bit-equal for {expert_module}.")
                    writer.add(rename(_module_key(expert_module, "weight_packed")), result.packed_weight)
                    writer.add(rename(_module_key(expert_module, "weight_scale")), result.weight_scale)
                    writer.add(
                        rename(_module_key(expert_module, "weight_global_scale")),
                        result.weight_scale_2.reshape(1).float().reciprocal(),
                    )
                    # clone: safetensors rejects tensors that share storage.
                    writer.add(rename(_module_key(expert_module, "input_global_scale")), input_scale.clone())
                    emitted[name] = result.weight_scale_2.cpu()
                    expert_linears += 1
                    del result, dequantized
                if not torch.equal(emitted["w1"], emitted["w3"]):
                    raise ValueError(f"Expert {module}.{expert} emitted unequal gate/up global scales.")
                del gate, up, down
            del gate_up_all, down_all
            _empty_cache()
            print(f"[packing experts] {module_index}/{len(experts_modules)} {module}", flush=True)
        writer.finalize()

        config["dtype"] = "bfloat16"
        config["torch_dtype"] = "bfloat16"
        for section in ("text_config", "vision_config", "audio_config"):
            if section in config:
                config[section]["dtype"] = "bfloat16"
        extra_ignore, ct_targets = _moe_serving_overrides(config, bool(experts_modules))
        ignore_entries = unquantized_ignore_entries(unquantized, quantizable - {n for n in quantizable if _EXPERT_TARGET_RE.match(n)})
        if is_minimax:
            ignore_entries = [entry if entry.startswith("re:") else rename(entry) for entry in ignore_entries]
        config["quantization_config"] = _compressed_tensors_config(
            ignore_entries, extra_ignore=extra_ignore, targets=ct_targets
        )
        with (temporary / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, sort_keys=True)
            file.write("\n")
        with (temporary / "nvfp4_export.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "format": "packed_compressed_tensors_nvfp4",
                    "format_version": 1,
                    "quantizer": "standard_max",
                    "weight_scale_scope": "fused_family",
                    "source_global_step": qad_metadata["global_step"],
                    "target_modules": target_list,
                    "experts_modules": experts_modules,
                    "unquantized_modules": unquantized,
                },
                file,
                indent=2,
                sort_keys=True,
            )
            file.write("\n")
        _copy_assets(source, model_assets, temporary)
        if reference is not None:
            _verify_reference_schema(temporary, Path(reference))
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "output": str(output),
        "quantized_linears": len(target_list),
        "quantized_expert_linears": expert_linears,
        "nonquantized_tensors": nonquantized,
        "carried_asset_tensors": carried,
        "total_tensors": writer.tensor_count,
        "total_bytes": writer.total_size,
    }
