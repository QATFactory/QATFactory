#!/usr/bin/env python3
"""Benchmark PyTorch and fused Triton FP4 fake-quantization forward paths."""

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

import torch

from qatfactory.quant.formats.fp4_triton import (
    HAS_TRITON,
    triton_fake_quantize_mxfp4,
    triton_fake_quantize_mxfp4_activation,
    triton_fake_quantize_nvfp4,
)
from qatfactory.quant.formats.mxfp4 import (
    _fake_quantize_mxfp4_activation_torch,
    _fake_quantize_mxfp4_torch,
)
from qatfactory.quant.formats.nvfp4 import NVFP4_SCALE_DENOMINATOR, _fake_quantize_nvfp4_torch

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
_WEIGHT_SHAPES = (
    ("q_o_proj", 4096, 4096),
    ("k_v_proj", 1024, 4096),
    ("gate_up_proj", 12288, 4096),
    ("down_proj", 4096, 12288),
)
_ACTIVATION_SHAPES = tuple(
    (f"tokens{tokens}_hidden{hidden}", tokens, hidden)
    for tokens in (2048, 8192, 32768)
    for hidden in (4096, 12288)
)


def _elapsed_us(function: Callable[[], object], *, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def _check_pair(actual, expected) -> None:
    actual_quantized, actual_mask = actual
    expected_quantized, expected_mask = expected
    if not torch.equal(actual_quantized, expected_quantized) or not torch.equal(actual_mask, expected_mask):
        raise AssertionError("Triton output does not bit-match the PyTorch reference.")


def _record(
    rows: list[dict[str, object]],
    *,
    operation: str,
    name: str,
    shape: tuple[int, int],
    dtype: torch.dtype,
    torch_function: Callable[[], object],
    triton_function: Callable[[], object],
    mask_bytes: int,
    iters: int,
) -> None:
    with torch.no_grad():
        _check_pair(triton_function(), torch_function())
        torch_us = _elapsed_us(torch_function, iters=iters)
        triton_us = _elapsed_us(triton_function, iters=iters)

    elements = shape[0] * shape[1]
    tensor_bytes = elements * torch.tensor([], dtype=dtype).element_size()
    traffic_bytes = tensor_bytes * 2 + elements * mask_bytes
    rows.append(
        {
            "operation": operation,
            "name": name,
            "shape": f"{shape[0]}x{shape[1]}",
            "dtype": str(dtype).removeprefix("torch."),
            "torch_us": torch_us,
            "triton_us": triton_us,
            "speedup": torch_us / triton_us,
            "effective_gbps": traffic_bytes / triton_us / 1e3,
        }
    )


def _print_table(rows: list[dict[str, object]]) -> None:
    header = (
        f"{'operation':<18} {'shape':>15} {'torch us':>11} "
        f"{'triton us':>11} {'speedup':>9} {'GB/s':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['operation']:<18} {row['shape']:>15} "
            f"{row['torch_us']:>11.1f} {row['triton_us']:>11.1f} "
            f"{row['speedup']:>8.2f}x {row['effective_gbps']:>10.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:1")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bf16")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--csv", type=Path, help="optional output CSV path")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not HAS_TRITON:
        raise RuntimeError("This benchmark requires CUDA and Triton.")
    if args.iters < 1:
        raise ValueError("--iters must be positive.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must name a CUDA device.")
    torch.cuda.set_device(device)
    dtype = _DTYPES[args.dtype]
    rows: list[dict[str, object]] = []

    for name, out_features, in_features in _WEIGHT_SHAPES:
        tensor = torch.randn(out_features, in_features, dtype=dtype, device=device)
        tensor_amax = tensor.float().abs().amax()
        global_scale = (tensor_amax / NVFP4_SCALE_DENOMINATOR).reshape(())
        _record(
            rows,
            operation="nvfp4_weight",
            name=name,
            shape=(out_features, in_features),
            dtype=dtype,
            torch_function=lambda x=tensor, amax=tensor_amax: _fake_quantize_nvfp4_torch(
                x, tensor_amax=amax
            ),
            triton_function=lambda x=tensor, scale=global_scale: triton_fake_quantize_nvfp4(x, scale),
            mask_bytes=1,
            iters=args.iters,
        )
        _record(
            rows,
            operation="mxfp4_weight",
            name=name,
            shape=(out_features, in_features),
            dtype=dtype,
            torch_function=lambda x=tensor: _fake_quantize_mxfp4_torch(x),
            triton_function=lambda x=tensor: triton_fake_quantize_mxfp4(x),
            mask_bytes=1,
            iters=args.iters,
        )
        del tensor

    for name, tokens, hidden in _ACTIVATION_SHAPES:
        tensor = torch.randn(tokens, hidden, dtype=dtype, device=device)
        input_amax = tensor.float().abs().amax()
        global_scale = (input_amax / NVFP4_SCALE_DENOMINATOR).reshape(())
        _record(
            rows,
            operation="nvfp4_activation",
            name=name,
            shape=(tokens, hidden),
            dtype=dtype,
            torch_function=lambda x=tensor, amax=input_amax: _fake_quantize_nvfp4_torch(
                x, tensor_amax=amax
            ),
            triton_function=lambda x=tensor, scale=global_scale: triton_fake_quantize_nvfp4(x, scale),
            mask_bytes=1,
            iters=args.iters,
        )

        def torch_mx_activation(x=tensor):
            return _fake_quantize_mxfp4_activation_torch(x)

        def triton_mx_activation(x=tensor):
            quantized = triton_fake_quantize_mxfp4_activation(x)
            return quantized, torch.ones_like(quantized, dtype=torch.bool)

        _record(
            rows,
            operation="mxfp4_activation",
            name=name,
            shape=(tokens, hidden),
            dtype=dtype,
            torch_function=torch_mx_activation,
            triton_function=triton_mx_activation,
            mask_bytes=1,
            iters=args.iters,
        )
        del tensor

    _print_table(rows)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
