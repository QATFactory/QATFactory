"""Pack an MXFP4 QAD checkpoint into vLLM's compressed-tensors MXFP4 layout.

This is a direct, dependency-light packer (no ModelOpt / llm-compressor): it re-quantizes
the trained BF16 latent weights with the same EVEN-rule encoder used during fake-quant
training and emits ``mxfp4-pack-quantized`` tensors (``weight_packed`` u8 ``[N, K/2]``,
``weight_scale`` u8 biased exponents ``[N, K/32]``) with a matching
``quantization_config`` in ``config.json``. MXFP4 has no global or activation scales, so
unlike the NVFP4 exporter there are no fused families, no ``weight_global_scale``, and no
``input_global_scale``; W4A4 vs W4A16 serving is decided by the hardware at load time.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from ... import __version__
from ..formats.mxfp4 import MXFP4_BLOCK_SIZE, dequantize_mxfp4, quantize_mxfp4
from .nvfp4_vllm import (
    _BaseWeightOverlay,
    _CARRY_DROP_PREFIXES,
    SafetensorDirectory,
    ShardedSafetensorWriter,
    _copy_assets,
    _empty_cache,
    _expected_targets,
    _trained_targets,
    unquantized_ignore_entries,
    _load_json,
    _module_key,
    _read_target_weight,
)


def _compressed_tensors_config(unquantized_modules: list[str] | None = None) -> dict[str, Any]:
    """Quantization config mirroring the compressed-tensors ``MXFP4`` preset scheme.

    vLLM's ``_is_mxfp4`` dispatch requires the weight args to be 4-bit float, symmetric,
    group strategy, group_size 32; the dynamic group-32 ``input_activations`` section
    declares W4A4 intent exactly as llm-compressor output does (the loader falls back to
    Marlin W4A16 on hardware without FP4 tensor cores).
    """

    shared_spec = {
        "actorder": None,
        "block_structure": None,
        "group_size": MXFP4_BLOCK_SIZE,
        "num_bits": 4,
        "observer_kwargs": {},
        "scale_dtype": "torch.uint8",
        "strategy": "group",
        "symmetric": True,
        "type": "float",
        "zp_dtype": None,
    }
    return {
        "config_groups": {
            "group_0": {
                "format": "mxfp4-pack-quantized",
                "input_activations": {
                    **shared_spec,
                    "dynamic": True,
                    "observer": None,
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    **shared_spec,
                    "dynamic": False,
                    "observer": "memoryless_minmax",
                },
            }
        },
        "format": "mxfp4-pack-quantized",
        "global_compression_ratio": None,
        "ignore": [
            "lm_head",
            "re:.*visual.*",
            "re:.*mtp.*",
            "re:.*embed_vision.*",
            "re:.*embed_audio.*",
            "re:.*vision_embedder.*",
            *sorted(unquantized_modules or ()),
        ],
        "kv_cache_scheme": None,
        "producer": {"name": "qatfactory", "version": __version__},
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
    }


def _validate_qad_metadata(config: dict[str, Any], quantizable: set[str]) -> dict[str, Any]:
    metadata = config.get("qatfactory_qad")
    if not isinstance(metadata, dict):
        raise ValueError("Source checkpoint is missing qatfactory_qad export metadata.")
    expected = {
        "format_version": 1,
        "quant_format": "mxfp4",
        "quantizer": "mx_even",
        "fused_runtime_scales": False,
        "input_scale_source": None,
    }
    mismatched = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatched:
        raise ValueError(f"Source checkpoint is not eligible for MXFP4 export: {mismatched}.")
    _trained_targets(metadata, quantizable, "MXFP4 export")
    if not isinstance(metadata.get("global_step"), int) or metadata["global_step"] < 0:
        raise ValueError("Source checkpoint metadata has no non-negative training global_step.")
    return metadata


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
    for module in sorted(reference_modules):
        for suffix in ("weight_packed", "weight_scale"):
            key = _module_key(module, suffix)
            if output_store.get_dtype_shape(key) != reference_store.get_dtype_shape(key):
                raise ValueError(f"Reference dtype/shape mismatch for {key}.")
    output_quant = _load_json(output / "config.json")["quantization_config"]
    reference_quant = _load_json(reference / "config.json")["quantization_config"]
    for key in ("format", "quant_method", "quantization_status"):
        if output_quant.get(key) != reference_quant.get(key):
            raise ValueError(f"Reference quantization config mismatch for {key!r}.")


def export_mxfp4_vllm_checkpoint(
    *,
    source: str | os.PathLike,
    model_assets: str | os.PathLike,
    output: str | os.PathLike,
    device: str = "cuda:0",
    shard_size: int = 5_000_000_000,
    reference: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Export an MXFP4 QAD HF checkpoint as compressed-tensors ``mxfp4-pack-quantized``."""

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
    target_list = _trained_targets(qad_metadata, quantizable, "MXFP4 export")
    targets = set(target_list)
    unquantized = sorted(quantizable - targets)
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

    # LoRA-QAD checkpoints carry unmerged adapters, folded into the target weights below
    # (see nvfp4_vllm._read_target_weight) and dropped from the passthrough set.
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
    for module in target_list:
        _, shape = source_store.get_dtype_shape(_module_key(module, "weight"))
        if len(shape) != 2 or shape[-1] % MXFP4_BLOCK_SIZE:
            raise ValueError(
                f"MXFP4 export requires 2D targets with in_features divisible by {MXFP4_BLOCK_SIZE}; "
                f"{module} has shape {shape}."
            )

    device = torch.device(device)
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
            if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
                raise ValueError(f"Non-target tensor {key!r} is not BF16: {tensor.dtype}.")
            writer.add(key, tensor)
            nonquantized += 1

        # A text-only causal-LM checkpoint omits the frozen vision tower that the
        # multimodal serving model still needs. Carry those (and any other untrained,
        # non-target) weights over from the original assets in BF16, matching the
        # deployment layout. MTP (speculative) weights are dropped, as vLLM does not use them.
        carried = 0
        try:
            assets_store = SafetensorDirectory(model_assets)
        except FileNotFoundError:
            assets_store = None
        if assets_store is not None:
            for key in sorted(assets_store.keys() - source_store.keys() - target_weight_keys):
                if any(key == prefix or key.startswith(prefix + ".") for prefix in _CARRY_DROP_PREFIXES):
                    continue
                tensor = assets_store.get_tensor(key)
                if tensor.is_floating_point():
                    if not bool(torch.isfinite(tensor).all()):
                        raise ValueError(f"Carried asset tensor {key!r} is not finite.")
                    tensor = tensor.to(torch.bfloat16)
                writer.add(key, tensor)
                carried += 1

        for index, module in enumerate(target_list, start=1):
            weight = _read_target_weight(source_store, module, lora_scaling)
            if weight.dtype != torch.bfloat16 or not bool(torch.isfinite(weight).all()):
                raise ValueError(f"Source target weight for {module} must be finite BF16, got {weight.dtype}.")
            result = quantize_mxfp4(weight.to(device))
            dequantized = dequantize_mxfp4(result.packed_weight, result.weight_scale)
            if not torch.equal(dequantized.to(result.reconstruction.dtype), result.reconstruction):
                raise ValueError(f"Packed MXFP4 dequantization is not bit-equal for {module}.")
            writer.add(_module_key(module, "weight_packed"), result.packed_weight)
            writer.add(_module_key(module, "weight_scale"), result.weight_scale)
            del weight, result, dequantized
            _empty_cache()
            print(f"[packing] {index}/{len(target_list)} {module}", flush=True)
        writer.finalize()

        config["dtype"] = "bfloat16"
        config["torch_dtype"] = "bfloat16"
        for section in ("text_config", "vision_config", "audio_config"):
            if section in config:
                config[section]["dtype"] = "bfloat16"
        config["quantization_config"] = _compressed_tensors_config(unquantized_ignore_entries(unquantized, quantizable))
        with (temporary / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, sort_keys=True)
            file.write("\n")
        with (temporary / "mxfp4_export.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "format": "packed_compressed_tensors_mxfp4",
                    "format_version": 1,
                    "quantizer": "mx_even",
                    "weight_scale_scope": "block_32_e8m0",
                    "source_global_step": qad_metadata["global_step"],
                    "target_modules": target_list,
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
        "nonquantized_tensors": nonquantized,
        "carried_asset_tensors": carried,
        "total_tensors": writer.tensor_count,
        "total_bytes": writer.total_size,
    }
