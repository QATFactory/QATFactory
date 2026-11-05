"""Block-scaled FP8 frozen bases for LoRA-QAD.

Some released checkpoints (MiniMax-M2.7) ship their weights as fine-grained FP8: an
``float8_e4m3fn`` code tensor plus a per-block ``weight_scale_inv`` grid. LoRA-QAD only ever
*reads* that base, so keeping the FP8 codes as the frozen parameter and dequantizing a single
weight at a time is what makes a 230B backbone trainable — a permanent high-precision copy
would be roughly twice the parameter memory.

The scales live as non-persistent buffers rather than parameters. As parameters they would be
sharded alongside the weight and rounded by an FSDP2 mixed-precision cast; as *persistent*
buffers they would appear in ``state_dict()``, where FSDP2 full-state-dict loading expects
every entry to be a sharded ``DTensor``. Nothing is lost by leaving them out of the state
dict: they are a fixed property of the frozen base, so reloading that base reconstructs them.
"""

import torch
from torch import nn


FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def is_fp8_tensor(tensor: torch.Tensor | None) -> bool:
    return tensor is not None and tensor.dtype in FP8_DTYPES


def dequantize_fp8_blocks(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Expand a per-block ``scale_inv`` grid over ``weight`` and return ``dtype`` values.

    ``weight`` is ``[..., rows, cols]`` and ``scale_inv`` is ``[..., scale_rows, scale_cols]``;
    each scale covers a ``rows // scale_rows`` by ``cols // scale_cols`` block. Leading
    dimensions (the expert axis) are broadcast, so one call dequantizes a whole fused expert
    tensor as well as a single dense weight.
    """

    rows, cols = weight.shape[-2:]
    scale_rows, scale_cols = scale_inv.shape[-2:]
    if rows % scale_rows or cols % scale_cols:
        raise ValueError(
            f"FP8 weight shape ({rows}, {cols}) is not divisible by its scale grid "
            f"({scale_rows}, {scale_cols})."
        )
    block_rows = rows // scale_rows
    block_cols = cols // scale_cols

    blocked = weight.to(torch.float32).reshape(
        *weight.shape[:-2], scale_rows, block_rows, scale_cols, block_cols
    )
    scales = scale_inv.to(torch.float32).reshape(
        *scale_inv.shape[:-2], scale_rows, 1, scale_cols, 1
    )
    return (blocked * scales).reshape(weight.shape).to(dtype)


def fp8_linear_state(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``(weight, weight_scale_inv)`` when ``module`` is a block-FP8 linear."""

    weight = getattr(module, "weight", None)
    scale_inv = getattr(module, "weight_scale_inv", None)
    if not is_fp8_tensor(weight) or scale_inv is None:
        return None
    return weight, scale_inv


def fp8_experts_state(module: nn.Module) -> dict[str, torch.Tensor] | None:
    """Return the fused FP8 expert tensors of ``module``, or None if it is not one."""

    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    if not is_fp8_tensor(gate_up) or not is_fp8_tensor(down):
        return None
    gate_up_scale = getattr(module, "gate_up_proj_scale_inv", None)
    down_scale = getattr(module, "down_proj_scale_inv", None)
    if gate_up_scale is None or down_scale is None:
        raise ValueError(
            "FP8 fused experts are missing their per-block scales "
            "(gate_up_proj_scale_inv / down_proj_scale_inv)."
        )
    return {
        "gate_up_proj": gate_up,
        "down_proj": down,
        "gate_up_proj_scale_inv": gate_up_scale,
        "down_proj_scale_inv": down_scale,
    }
