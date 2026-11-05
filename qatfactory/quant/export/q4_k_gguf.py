"""Export a Q4_K QAD Hugging Face checkpoint to a GGUF with lossless Q4_K target weights.

The deployable Q4_K blocks are computed once, at training-checkpoint save time, on the
training device (``q4k_weight.safetensors``). Export is therefore a pure, device-independent
*splice*: build the GGUF container / metadata / tokenizer / architecture reorders with the
trusted ``convert_hf_to_gguf.py`` (BF16 out-type), then overwrite the target tensors' data
with the pre-computed Q4_K blocks. Because ggml's ``dequantize_row_q4_K`` is bit-identical to
:func:`qatfactory.quant.formats.q4_k.dequantize_q4_k`, the deployed model dequantizes
value-for-value to what the student trained/eval'd with.

We deliberately do not re-run llama.cpp's Q4_K *encoder* (its grid-search is not reproducible
bit-for-bit across float backends); we only reuse llama.cpp's decoder-compatible container.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from ... import __version__
from ..formats.q4_k import QK_K, Q4K_BLOCK_BYTES, dequantize_q4_k


logger = logging.getLogger(__name__)


# GGUF tensor roles for each dense/hybrid QAD target leaf (verified against gguf's
# TensorNameMap for QWEN3 / QWEN35). The Qwen3.5 linear-attn tensors are renamed to fused
# attn_qkv / attn_gate / ssm_* forms.
_LEAF_TO_GGML = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.out_proj": "ssm_out",
}

# Closeness bound for the export sanity check: dequant(spliced block) vs the scaffold's BF16
# tensor. Q4_K error is a small fraction of the tensor's own amax; a wrong name/reorder gives
# a diff on the order of the amax itself, which this catches.
_CLOSENESS_FRACTION_OF_AMAX = 0.25

# Frozen (non-QAD) vocab tensors we optionally down-cast from the BF16 scaffold. These are
# never trained by QAD, so they carry no bit-exactness requirement; Q8_0 is a deterministic
# round-to-nearest re-encode (block of 32) that roughly halves their footprint vs BF16.
_EMBED_OUTPUT_TENSORS = ("token_embd.weight", "output.weight")
_EMBED_OUTPUT_DTYPES = ("q8_0", "bf16")


def _require_gguf():
    """Import the ggml/llama.cpp ``gguf`` package with an actionable error if missing."""

    try:
        import gguf
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "Q4_K GGUF export needs the `gguf` package (ggml/llama.cpp, MIT). "
            "Install it with `pip install -e .[q4k]` or `pip install gguf`."
        ) from exc
    return gguf


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


# Both Q4_K variants emit identical block_q4_K bytes; the export splice + lossless verify is
# the same regardless of which level-1 encoder trained the checkpoint. ("q4_k_ref" is the
# legacy id for the grid-search encoder, kept for older checkpoints.)
_Q4K_QUANT_FORMATS = {"q4_k", "q4_k_grid_search"}
_Q4K_QUANTIZERS = {"q4_k_ref", "q4_k_grid_search", "q4_k_minmax"}


def _validate_q4k_metadata(config: dict[str, Any]) -> dict[str, Any]:
    metadata = config.get("qatfactory_qad")
    if not isinstance(metadata, dict):
        raise ValueError("Source checkpoint is missing qatfactory_qad export metadata.")
    if metadata.get("quant_format") not in _Q4K_QUANT_FORMATS or metadata.get("quantizer") not in _Q4K_QUANTIZERS:
        raise ValueError(
            f"Source checkpoint is not a Q4_K QAD run: quant_format={metadata.get('quant_format')!r}, "
            f"quantizer={metadata.get('quantizer')!r}."
        )
    if metadata.get("fused_runtime_scales"):
        raise ValueError("Q4_K QAD checkpoints must not use fused_runtime_scales.")
    targets = metadata.get("target_modules")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Source checkpoint metadata has no target_modules list.")
    if not isinstance(metadata.get("global_step"), int) or metadata["global_step"] <= 0:
        raise ValueError("Source checkpoint metadata has no positive training global_step.")
    return metadata


def _load_packed_weights(source: Path, target_list: list[str]) -> dict[str, torch.Tensor]:
    path = source / "q4k_weight.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Source checkpoint is missing packed Q4_K blocks: {path}.")
    with safe_open(path, framework="pt") as file:
        actual = set(file.keys())
        expected = set(target_list)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"Packed-weight key mismatch; missing={missing[:5]}, extra={extra[:5]}.")
        packed = {name: file.get_tensor(name) for name in target_list}
    for name, tensor in packed.items():
        if tensor.dtype != torch.uint8 or tensor.ndim != 2 or tensor.shape[1] % Q4K_BLOCK_BYTES:
            raise ValueError(f"Packed Q4_K block for {name!r} is not [out, n_super*144] uint8.")
    return packed


def _layer_and_leaf(module: str) -> tuple[int, str]:
    """Split ``...layers.<i>.<leaf...>`` into ``(layer_index, leaf)`` (arch-prefix agnostic)."""

    parts = module.split(".")
    for index in range(len(parts) - 1):
        if parts[index] == "layers":
            return int(parts[index + 1]), ".".join(parts[index + 2 :])
    raise ValueError(f"Cannot locate a layer index in module name {module!r}.")


def _map_targets_to_ggml(target_list: list[str]) -> dict[str, str]:
    """Map HF on-disk target module names to GGUF tensor names (``blk.<i>.<role>.weight``)."""

    mapping = {}
    for module in target_list:
        layer, leaf = _layer_and_leaf(module)
        role = _LEAF_TO_GGML.get(leaf)
        if role is None:
            raise ValueError(f"No GGUF tensor-name mapping for target leaf {leaf!r} (module {module!r}).")
        mapping[module] = f"blk.{layer}.{role}.weight"
    return mapping


def _config_has_mtp(config: dict[str, Any]) -> bool:
    """True if the checkpoint config declares MTP / multi-token-prediction (nextn) layers.

    The QAD student is loaded as a text-only causal LM, which drops the MTP head, so its
    checkpoint has no MTP weights. The converter, however, sets ``block_count`` to
    ``num_hidden_layers + nextn`` and expects the extra ``blk.<n>`` tensors; without ``--no-mtp``
    the resulting GGUF declares one block too many and fails to load in llama.cpp.
    """

    scopes = [config, config.get("text_config", {}) if isinstance(config.get("text_config"), dict) else {}]
    keys = ("num_nextn_predict_layers", "mtp_num_hidden_layers", "nextn_predict_layers")
    return any(bool(scope.get(key)) for scope in scopes for key in keys)


def _run_convert(convert_script: Path, python_exe: str, source: Path, scaffold: Path, no_mtp: bool = False) -> None:
    command = [python_exe, str(convert_script), "--outfile", str(scaffold), "--outtype", "bf16"]
    if no_mtp:
        command.append("--no-mtp")
    command.append(str(source))
    subprocess.run(command, check=True)


def splice_q4k_into_gguf(
    scaffold_path: Path,
    output_path: Path,
    ggml_packed: dict[str, np.ndarray],
    embed_output_dtype: str = "q8_0",
) -> dict[str, tuple[int, int]]:
    """Copy ``scaffold_path`` to ``output_path``, replacing tensors in ``ggml_packed`` with Q4_K.

    ``ggml_packed`` maps GGUF tensor names to ``[out, n_super*144]`` uint8 block buffers.
    ``embed_output_dtype`` controls the storage of the frozen ``token_embd``/``output``
    tensors: ``"q8_0"`` (default) re-encodes them from the BF16 scaffold to Q8_0, ``"bf16"``
    copies them through unchanged. Returns a map from spliced tensor name to its
    ``(in_features, out_features)``.
    """

    if embed_output_dtype not in _EMBED_OUTPUT_DTYPES:
        raise ValueError(f"embed_output_dtype must be one of {_EMBED_OUTPUT_DTYPES}, got {embed_output_dtype!r}.")

    gguf = _require_gguf()

    reader = gguf.GGUFReader(scaffold_path)
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        raise ValueError("Scaffold GGUF has no general.architecture field.")
    arch = arch_field.contents()
    writer = gguf.GGUFWriter(output_path, arch)

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if value is not None:
            writer.add_key_value(field.name, value, value_type, sub_type=sub_type)

    from gguf.quants import dequantize as _gguf_dequantize
    from gguf.quants import quantize as _gguf_quantize

    spliced = {}
    reencoded: dict[str, np.ndarray] = {}
    max_closeness_ratio = 0.0
    remaining = dict(ggml_packed)
    for tensor in reader.tensors:
        block = remaining.pop(tensor.name, None)
        if block is None:
            if embed_output_dtype == "q8_0" and tensor.name in _EMBED_OUTPUT_TENSORS:
                # Re-encode the frozen vocab tensor from the BF16 scaffold to Q8_0. ne is
                # reversed (ne=[in, out]); the logical row-major matrix is [out, in] and Q8_0
                # blocks tile the trailing in-features axis (must be a multiple of 32).
                in_features = int(tensor.shape[0])
                out_features = int(tensor.shape[1])
                f32 = (
                    _gguf_dequantize(np.ascontiguousarray(tensor.data), tensor.tensor_type)
                    .reshape(out_features, in_features)
                    .astype(np.float32)
                )
                q8_blocks = _gguf_quantize(f32, gguf.GGMLQuantizationType.Q8_0)
                reencoded[tensor.name] = q8_blocks
                writer.add_tensor_info(
                    tensor.name, q8_blocks.shape, q8_blocks.dtype, q8_blocks.nbytes, gguf.GGMLQuantizationType.Q8_0
                )
                continue
            writer.add_tensor_info(
                tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type
            )
            continue
        out_features = int(tensor.shape[1])
        in_features = int(tensor.shape[0])
        if block.shape != (out_features, (in_features // QK_K) * Q4K_BLOCK_BYTES):
            raise ValueError(
                f"Packed block for {tensor.name!r} has shape {block.shape}, expected "
                f"{(out_features, (in_features // QK_K) * Q4K_BLOCK_BYTES)} for ggml ne={list(tensor.shape)}."
            )
        # Sanity: the spliced blocks must dequantize close to the scaffold's (deploy-layout)
        # BF16 tensor. This validates the HF->GGUF name mapping and any architecture reorder;
        # a mismatch would otherwise silently ship a broken "lossless" checkpoint.
        # gguf's GGUFReader returns BF16 (and other non-F32) tensor data as raw bytes, so a
        # plain numpy reshape to logical [out, in] would see 2x the element count. Decode via
        # gguf's own dequantizer, which is correct regardless of how the reader represents BF16.
        reference = (
            _gguf_dequantize(np.ascontiguousarray(tensor.data), tensor.tensor_type)
            .reshape(out_features, in_features)
            .astype(np.float32)
        )
        our_recon = dequantize_q4_k(torch.from_numpy(block.copy()), in_features).numpy()
        amax = float(np.abs(reference).max())
        max_diff = float(np.abs(our_recon - reference).max())
        ratio = max_diff / amax if amax > 0 else 0.0
        if ratio > _CLOSENESS_FRACTION_OF_AMAX:
            raise ValueError(
                f"Spliced Q4_K for {tensor.name!r} is not close to the scaffold weight "
                f"(max diff {max_diff:.4g} vs amax {amax:.4g}); likely a name/reorder mismatch."
            )
        max_closeness_ratio = max(max_closeness_ratio, ratio)
        writer.add_tensor_info(
            tensor.name, block.shape, block.dtype, block.nbytes, gguf.GGMLQuantizationType.Q4_K
        )
        spliced[tensor.name] = (in_features, out_features)

    if remaining:
        raise ValueError(f"Packed blocks did not match any scaffold tensor: {sorted(remaining)[:5]}.")

    logger.info(
        "Spliced %d Q4_K target tensors; max scaffold-closeness ratio %.4g (bound %.2g). "
        "Vocab tensors (token_embd/output) stored as %s%s.",
        len(spliced),
        max_closeness_ratio,
        _CLOSENESS_FRACTION_OF_AMAX,
        embed_output_dtype.upper(),
        f" (re-encoded {sorted(reencoded)})" if reencoded else "",
    )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        block = ggml_packed.get(tensor.name)
        if block is not None:
            data = block
        elif tensor.name in reencoded:
            data = reencoded[tensor.name]
        else:
            data = tensor.data
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()
    return spliced


def _verify_lossless(output_path: Path, ggml_packed: dict[str, np.ndarray], spliced: dict[str, tuple[int, int]]) -> int:
    """Assert the written GGUF dequantizes bit-exactly to our Q4_K reconstruction.

    Reads the tensors back from the *output* file (not the in-memory blocks) and checks that
    ggml's own decoder reproduces our decoder value-for-value, proving the deployed weights
    equal the trained student. Returns the number of tensors verified.
    """

    gguf = _require_gguf()
    from gguf.quants import Q4_K as GGUFQ4K

    reader = gguf.GGUFReader(output_path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    for name, (in_features, out_features) in spliced.items():
        tensor = tensors[name]
        if tensor.tensor_type != gguf.GGMLQuantizationType.Q4_K:
            raise ValueError(f"Spliced tensor {name!r} is not Q4_K in the output.")
        blocks = tensor.data.reshape(-1, Q4K_BLOCK_BYTES)
        gguf_recon = GGUFQ4K.dequantize_blocks(blocks).reshape(out_features, in_features)
        our_recon = dequantize_q4_k(
            torch.from_numpy(ggml_packed[name]), in_features
        ).numpy()
        max_diff = float(np.abs(gguf_recon - our_recon).max())
        if max_diff != 0.0:
            raise ValueError(f"Spliced Q4_K for {name!r} is not bit-exact with our decoder (max diff {max_diff}).")
    logger.info(
        "Q4_K lossless verify PASSED: %d/%d target tensors dequantize bit-exactly (max diff 0.0).",
        len(spliced),
        len(spliced),
    )
    return len(spliced)


def export_q4k_gguf_checkpoint(
    *,
    source: str | os.PathLike,
    output: str | os.PathLike,
    convert_script: str | os.PathLike,
    python_exe: str = sys.executable,
    keep_scaffold: bool = False,
    embed_output_dtype: str = "q8_0",
) -> dict[str, Any]:
    """Export a Q4_K QAD Hugging Face checkpoint to a GGUF (targets losslessly Q4_K).

    ``embed_output_dtype`` sets the storage of the frozen ``token_embd``/``output`` tensors:
    ``"q8_0"`` (default) or ``"bf16"``.
    """

    if embed_output_dtype not in _EMBED_OUTPUT_DTYPES:
        raise ValueError(f"embed_output_dtype must be one of {_EMBED_OUTPUT_DTYPES}, got {embed_output_dtype!r}.")

    source = Path(source)
    output = Path(output)
    convert_script = Path(convert_script)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    if not convert_script.is_file():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {convert_script}.")

    config = _load_json(source / "config.json")
    metadata = _validate_q4k_metadata(config)
    target_list = sorted(metadata["target_modules"])
    packed = _load_packed_weights(source, target_list)
    name_map = _map_targets_to_ggml(target_list)
    ggml_packed = {name_map[module]: packed[module].numpy() for module in target_list}

    output.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    scaffold = workdir / "scaffold.bf16.gguf"
    try:
        _run_convert(convert_script, python_exe, source, scaffold, no_mtp=_config_has_mtp(config))
        spliced = splice_q4k_into_gguf(scaffold, output, ggml_packed, embed_output_dtype=embed_output_dtype)
        verified = _verify_lossless(output, ggml_packed, spliced)
        if keep_scaffold:
            shutil.copy2(scaffold, output.parent / f"{output.name}.scaffold.bf16.gguf")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    total_bytes = output.stat().st_size
    logger.info(
        "Q4_K export complete: %s (%d quantized linears, lossless_verified=%s, %.2f GB).",
        output,
        len(spliced),
        verified == len(spliced),
        total_bytes / 1e9,
    )
    return {
        "output": str(output),
        "quantized_linears": len(spliced),
        "lossless_verified": verified == len(spliced),
        "verified_tensors": verified,
        "embed_output_dtype": embed_output_dtype,
        "source_global_step": metadata["global_step"],
        "total_bytes": total_bytes,
        "producer": {"name": "qatfactory", "version": __version__},
    }
