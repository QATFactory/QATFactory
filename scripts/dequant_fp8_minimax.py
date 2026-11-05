#!/usr/bin/env python3
"""Streaming FP8 -> BF16 dequantizer for the MiniMaxAI/MiniMax-M2.7 snapshot.

The official repo ships F8_E4M3 block-quantized weights (``quant_method="fp8"``,
``weight_block_size=[128, 128]``): every FP8 weight ``<name>`` of shape [..., out, in]
pairs with an FP32 ``<name>_scale_inv`` of shape [..., ceil(out/128), ceil(in/128)],
and dequantization multiplies each 128x128 block by its scale. QAD training needs a
clean BF16 base, so this script rewrites the checkpoint shard-by-shard:

  * FP8 weights become BF16 (``weight.float() * expanded_scales``, block scales gathered
    per element so edge blocks of non-divisible dims are exact); ``*_scale_inv`` drop.
  * Non-converted tensors (embeddings, lm_head, norms, router ``gate`` weights,
    ``e_score_correction_bias``) pass through with their original dtypes.
  * ``config.json`` is rewritten without ``quantization_config`` and with
    ``torch_dtype=bfloat16``; tokenizer / chat-template / generation assets copy verbatim.
  * ``model.safetensors.index.json`` is rebuilt with the correct BF16 total_size.

Streaming: tensors are planned into output shards up front from headers only, then each
shard is materialized and written independently, so peak RAM stays around one output
shard. Resumable: shards are written to a temporary name and os.replace'd, and a rerun
skips any existing output shard whose size/header already matches the planned tensor
set (key/dtype/shape). A built-in verification pass then asserts every output tensor's
dtype/shape, recomputes k random blocks per shard against a reference einsum, checks
that no source key (minus the consumed ``*_scale_inv`` tensors) is missing, and prints
a tensor-count + dtype census as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


_SCALE_SUFFIX = "_scale_inv"

_ASSET_FILENAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
)

# safetensors header dtype string -> bytes per element (output-size planning + census).
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


@dataclass(frozen=True)
class TensorPlan:
    """One output tensor: where it comes from, what it becomes, and its paired scale."""

    key: str
    filename: str  # source shard
    source_dtype: str
    output_dtype: str
    shape: tuple[int, ...]
    nbytes: int  # output bytes
    scale_key: str | None  # set iff the source tensor is FP8 block-quantized
    scale_filename: str | None


class _ShardReader:
    """Header/tensor reader over a sharded checkpoint with cached safe_open handles."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._handles: dict[str, object] = {}

    def handle(self, filename: str):
        if filename not in self._handles:
            self._handles[filename] = safe_open(self.directory / filename, framework="pt")
        return self._handles[filename]

    def get_tensor(self, filename: str, key: str) -> torch.Tensor:
        return self.handle(filename).get_tensor(key)


def dequantize_fp8_block(
    weight: torch.Tensor, scale_inv: torch.Tensor, block_size: tuple[int, int] = (128, 128)
) -> torch.Tensor:
    """Dequantize a block-quantized FP8 tensor to BF16 over its last two dims.

    ``scale_inv`` carries one FP32 multiplier per block on a ceil-grid; each element
    gathers its block's scale through integer division of its coordinates, so edge
    blocks of non-divisible dims are handled without any divisibility assumption.
    Leading dims (packed 3D expert tensors) are processed one matrix at a time to
    bound the FP32 transient.
    """

    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"Expected a float8_e4m3fn weight, got {weight.dtype}.")
    if weight.ndim < 2:
        raise ValueError(f"Block-quantized weight must have >=2 dims, got shape {tuple(weight.shape)}.")
    if scale_inv.dtype != torch.float32:
        raise ValueError(f"Expected a float32 weight_scale_inv, got {scale_inv.dtype}.")
    rows, cols = weight.shape[-2:]
    block_rows, block_cols = block_size
    grid_rows, grid_cols = _ceil_div(rows, block_rows), _ceil_div(cols, block_cols)
    expected = tuple(weight.shape[:-2]) + (grid_rows, grid_cols)
    if tuple(scale_inv.shape) != expected:
        raise ValueError(
            f"weight_scale_inv shape {tuple(scale_inv.shape)} does not match the ceil-grid {expected} "
            f"of weight shape {tuple(weight.shape)} at block size {block_size}."
        )
    row_block = torch.arange(rows) // block_rows
    col_block = torch.arange(cols) // block_cols
    flat_weight = weight.reshape(-1, rows, cols)
    flat_scale = scale_inv.reshape(-1, grid_rows, grid_cols)
    output = torch.empty(flat_weight.shape, dtype=torch.bfloat16)
    for index in range(flat_weight.shape[0]):
        expanded = flat_scale[index][row_block[:, None], col_block[None, :]]
        output[index] = (flat_weight[index].to(torch.float32) * expanded).to(torch.bfloat16)
    return output.reshape(weight.shape)


def _read_weight_map(source: Path) -> dict[str, str]:
    """Tensor-name -> shard-filename map from the index (or by scanning unsharded files)."""

    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as file:
            weight_map = json.load(file)["weight_map"]
    else:
        weight_map = {}
        for path in sorted(source.glob("*.safetensors")):
            with safe_open(path, framework="pt") as file:
                for key in file.keys():
                    if key in weight_map:
                        raise ValueError(f"Duplicate tensor {key!r} in {source}.")
                    weight_map[key] = path.name
    if not weight_map:
        raise FileNotFoundError(f"No model safetensors found under {source}.")
    return weight_map


def _build_plan(reader: _ShardReader, weight_map: dict[str, str], block_size: tuple[int, int]) -> list[TensorPlan]:
    """One deterministic TensorPlan per output tensor; FP8 weights paired with their scales."""

    by_file: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        by_file.setdefault(filename, []).append(key)
    scale_keys = {key for key in weight_map if key.endswith(_SCALE_SUFFIX)}
    ordered = [key for filename in sorted(by_file) for key in sorted(by_file[filename]) if key not in scale_keys]
    entries = []
    claimed = set()
    for key in tqdm(ordered, desc="planning", unit="tensor"):
        filename = weight_map[key]
        tensor_slice = reader.handle(filename).get_slice(key)
        dtype = tensor_slice.get_dtype()
        shape = tuple(tensor_slice.get_shape())
        scale_key = scale_filename = None
        if dtype == "F8_E4M3":
            if len(shape) < 2:
                raise ValueError(f"FP8 weight {key!r} must have >=2 dims, got shape {shape}.")
            scale_key = key + _SCALE_SUFFIX
            if scale_key not in weight_map:
                raise ValueError(f"FP8 weight {key!r} is missing its {scale_key!r} partner.")
            scale_filename = weight_map[scale_key]
            scale_slice = reader.handle(scale_filename).get_slice(scale_key)
            if scale_slice.get_dtype() != "F32":
                raise ValueError(f"{scale_key!r} must be F32, got {scale_slice.get_dtype()!r}.")
            expected = shape[:-2] + (_ceil_div(shape[-2], block_size[0]), _ceil_div(shape[-1], block_size[1]))
            if tuple(scale_slice.get_shape()) != expected:
                raise ValueError(
                    f"{scale_key!r} shape {tuple(scale_slice.get_shape())} does not match the ceil-grid "
                    f"{expected} of {key!r} {shape} at block size {block_size}."
                )
            claimed.add(scale_key)
            output_dtype = "BF16"
        elif dtype not in _DTYPE_BYTES:
            raise ValueError(f"Unsupported source dtype {dtype!r} for {key!r}.")
        else:
            output_dtype = dtype
        entries.append(
            TensorPlan(
                key=key,
                filename=filename,
                source_dtype=dtype,
                output_dtype=output_dtype,
                shape=shape,
                nbytes=math.prod(shape) * _DTYPE_BYTES[output_dtype],
                scale_key=scale_key,
                scale_filename=scale_filename,
            )
        )
    orphans = sorted(scale_keys - claimed)
    if orphans:
        raise ValueError(f"Orphan weight_scale_inv tensors with no FP8 weight: {orphans[:5]}.")
    return entries


def _assign_shards(entries: list[TensorPlan], shard_size: int) -> list[list[TensorPlan]]:
    """Greedy deterministic assignment of planned tensors to output shards."""

    shards: list[list[TensorPlan]] = []
    current: list[TensorPlan] = []
    current_size = 0
    for entry in entries:
        if current and current_size + entry.nbytes > shard_size:
            shards.append(current)
            current = []
            current_size = 0
        current.append(entry)
        current_size += entry.nbytes
    if current:
        shards.append(current)
    if not shards:
        raise ValueError("Cannot write an empty checkpoint.")
    return shards


def _shard_matches_plan(path: Path, entries: list[TensorPlan]) -> bool:
    """True when an existing output shard already carries exactly the planned tensors.

    Shards are written via a temporary name and os.replace, so a file existing under
    its final name is complete; the header check (key set + per-tensor dtype/shape,
    which fixes the file size) additionally guards against plan drift between runs.
    """

    if not path.is_file():
        return False
    try:
        with safe_open(path, framework="pt") as file:
            if set(file.keys()) != {entry.key for entry in entries}:
                return False
            for entry in entries:
                tensor_slice = file.get_slice(entry.key)
                if tensor_slice.get_dtype() != entry.output_dtype or tuple(tensor_slice.get_shape()) != entry.shape:
                    return False
    except Exception:
        return False
    return True


def _write_config(config: dict, output: Path) -> None:
    del config["quantization_config"]
    config["torch_dtype"] = "bfloat16"
    if "dtype" in config:
        config["dtype"] = "bfloat16"
    with (output / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")


def _copy_assets(source: Path, output: Path) -> None:
    for filename in _ASSET_FILENAMES:
        path = source / filename
        if path.is_file():
            shutil.copy2(path, output / filename)


def _verify_output(
    reader: _ShardReader,
    output: Path,
    filenames: list[str],
    shards: list[list[TensorPlan]],
    block_size: tuple[int, int],
    verify_blocks: int,
    seed: int,
) -> int:
    """Assert every output tensor's dtype/shape and recompute random blocks via a reference einsum."""

    block_rows, block_cols = block_size
    verified = 0
    for filename, shard in tqdm(list(zip(filenames, shards)), desc="verifying", unit="shard"):
        with safe_open(output / filename, framework="pt") as file:
            keys = set(file.keys())
            planned = {entry.key for entry in shard}
            if keys != planned:
                raise ValueError(
                    f"{filename} tensor set diverges from the plan: "
                    f"missing={sorted(planned - keys)[:3]}, extra={sorted(keys - planned)[:3]}."
                )
            for entry in shard:
                tensor_slice = file.get_slice(entry.key)
                if tensor_slice.get_dtype() != entry.output_dtype or tuple(tensor_slice.get_shape()) != entry.shape:
                    raise ValueError(
                        f"Output tensor {entry.key!r} is {tensor_slice.get_dtype()}"
                        f"/{tuple(tensor_slice.get_shape())}, expected {entry.output_dtype}/{entry.shape}."
                    )
            fp8_entries = [entry for entry in shard if entry.scale_key is not None]
            if not fp8_entries:
                continue
            rng = random.Random(f"{seed}:{filename}")
            for _ in range(verify_blocks):
                entry = rng.choice(fp8_entries)
                rows, cols = entry.shape[-2:]
                leading = tuple(rng.randrange(dim) for dim in entry.shape[:-2])
                block_row = rng.randrange(_ceil_div(rows, block_rows))
                block_col = rng.randrange(_ceil_div(cols, block_cols))
                row0, row1 = block_row * block_rows, min(rows, (block_row + 1) * block_rows)
                col0, col1 = block_col * block_cols, min(cols, (block_col + 1) * block_cols)
                index = tuple(slice(v, v + 1) for v in leading) + (slice(row0, row1), slice(col0, col1))
                weight_block = (
                    reader.handle(entry.filename).get_slice(entry.key)[index].reshape(row1 - row0, col1 - col0)
                )
                scale_index = tuple(slice(v, v + 1) for v in leading) + (
                    slice(block_row, block_row + 1),
                    slice(block_col, block_col + 1),
                )
                scale = reader.handle(entry.scale_filename).get_slice(entry.scale_key)[scale_index].reshape(())
                reference = torch.einsum("rc,->rc", weight_block.to(torch.float32), scale).to(torch.bfloat16)
                produced = file.get_slice(entry.key)[index].reshape(row1 - row0, col1 - col0)
                if not torch.equal(produced, reference):
                    raise ValueError(
                        f"Block recompute mismatch for {entry.key!r} at leading={leading}, "
                        f"block=({block_row}, {block_col})."
                    )
                verified += 1
    return verified


def dequant_fp8_checkpoint(
    *,
    source: str | os.PathLike,
    output: str | os.PathLike,
    shard_size: int = 5_000_000_000,
    verify_blocks: int = 8,
    seed: int = 0,
) -> dict:
    """Stream-dequantize an FP8 block-quantized HF snapshot into a BF16 checkpoint."""

    source = Path(source)
    output = Path(output)
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")
    if verify_blocks < 0:
        raise ValueError("verify_blocks must be non-negative.")
    if source.resolve() == output.resolve():
        raise ValueError("Output directory must differ from the source snapshot.")
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing source config: {config_path}.")
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict) or quantization.get("quant_method") != "fp8":
        raise ValueError(f"Source is not an FP8-quantized checkpoint: quantization_config={quantization!r}.")
    if quantization.get("fmt", "float8_e4m3fn") != "float8_e4m3fn":
        raise ValueError(f"Unsupported FP8 format {quantization.get('fmt')!r}.")
    block_size = tuple(int(value) for value in (quantization.get("weight_block_size") or (128, 128)))
    if len(block_size) != 2 or min(block_size) <= 0:
        raise ValueError(f"Unsupported weight_block_size {block_size!r}.")

    reader = _ShardReader(source)
    weight_map = _read_weight_map(source)
    entries = _build_plan(reader, weight_map, block_size)
    scale_keys = {key for key in weight_map if key.endswith(_SCALE_SUFFIX)}
    missing = sorted(set(weight_map) - scale_keys - {entry.key for entry in entries})
    if missing:
        raise ValueError(f"Plan is missing source tensors: {missing[:5]}.")
    shards = _assign_shards(entries, shard_size)
    filenames = [f"model-{index:05d}-of-{len(shards):05d}.safetensors" for index in range(1, len(shards) + 1)]

    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob(".*.safetensors.tmp"):
        stale.unlink()
    written = skipped = 0
    with tqdm(total=len(entries), desc="dequantizing", unit="tensor") as progress:
        for filename, shard in zip(filenames, shards):
            path = output / filename
            if _shard_matches_plan(path, shard):
                skipped += 1
                progress.update(len(shard))
                continue
            tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
            for entry in shard:
                if entry.scale_key is not None:
                    weight = reader.get_tensor(entry.filename, entry.key)
                    scale_inv = reader.get_tensor(entry.scale_filename, entry.scale_key)
                    tensors[entry.key] = dequantize_fp8_block(weight, scale_inv, block_size)
                    del weight, scale_inv
                else:
                    tensors[entry.key] = reader.get_tensor(entry.filename, entry.key)
                progress.update(1)
            temporary = output / f".{filename}.tmp"
            save_file(tensors, temporary, metadata={"format": "pt"})
            os.replace(temporary, path)
            del tensors
            written += 1

    total_size = sum(entry.nbytes for entry in entries)
    output_weight_map: dict[str, str] = {}
    for filename, shard in zip(filenames, shards):
        output_weight_map.update(dict.fromkeys((entry.key for entry in shard), filename))
    with (output / "model.safetensors.index.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": output_weight_map},
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    _write_config(config, output)
    _copy_assets(source, output)

    extras = sorted({path.name for path in output.glob("*.safetensors")} - set(filenames))
    if extras:
        raise ValueError(f"Output directory holds unplanned safetensors files (stale run?): {extras[:5]}.")
    verified_blocks = _verify_output(reader, output, filenames, shards, block_size, verify_blocks, seed)

    source_census = Counter(entry.source_dtype for entry in entries)
    source_census["F32"] += len(scale_keys)  # every consumed scale was validated F32
    output_census = Counter(entry.output_dtype for entry in entries)
    return {
        "output": str(output),
        "block_size": list(block_size),
        "shard_count": len(shards),
        "shards_written": written,
        "shards_skipped": skipped,
        "tensors_total": len(entries),
        "tensors_dequantized": sum(1 for entry in entries if entry.scale_key is not None),
        "tensors_passthrough": sum(1 for entry in entries if entry.scale_key is None),
        "scale_tensors_consumed": len(scale_keys),
        "missing_output_keys": 0,
        "source_dtype_census": dict(sorted(source_census.items())),
        "output_dtype_census": dict(sorted(output_census.items())),
        "verified_blocks": verified_blocks,
        "total_size": total_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream-dequantize the FP8 block-quantized MiniMax-M2.7 snapshot to a BF16 checkpoint."
    )
    parser.add_argument("--source", required=True, help="Local snapshot directory of the FP8 repo.")
    parser.add_argument("--output", required=True, help="Fresh output directory (resumable across reruns).")
    parser.add_argument("--shard-size", type=int, default=5_000_000_000)
    parser.add_argument("--verify-blocks", type=int, default=8, help="Random blocks recomputed per output shard.")
    parser.add_argument("--seed", type=int, default=0, help="Verification sampling seed.")
    args = parser.parse_args()

    summary = dequant_fp8_checkpoint(
        source=args.source,
        output=args.output,
        shard_size=args.shard_size,
        verify_blocks=args.verify_blocks,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
