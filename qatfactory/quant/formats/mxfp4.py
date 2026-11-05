"""MXFP4 fake quantization (weights and, for W4A4, input activations).

The numerics here must stay bit-exact with the deployment (vLLM compressed-tensors
``mxfp4-pack-quantized``) decode: E2M1 values, groups of 32 along the input dimension,
E8M0 power-of-two block scales stored as biased uint8 exponents, no global scale, and
deterministic round-to-nearest-even ties.

MXFP4 uses TWO E8M0 scale rules, by tensor role, matching what vLLM actually serves:

- **Weights** use the OCP "EVEN" rule shared by compressed-tensors
  ``generate_mx_scales``, Quark's EVEN mode, and vLLM's
  ``tests/quantization/reference_mxfp4.py``: threshold-round the block amax to a power
  of two (carry iff its mantissa fraction >= 1.75), subtract the E2M1 max exponent, and
  clamp the biased exponent to [0, 254]. Blocks whose amax mantissa lies in
  (1.5, 1.75) are clipped (up to 16.7%).
- **Activations** (W4A4 serving, flashinfer ``mxfp4_quantize`` cute-dsl backend =
  ``__nv_cvt_float_to_e8m0(amax * rcp(6), SATFINITE, RoundPosInf)``) use the CEIL rule:
  the smallest power of two >= amax/6, so ``6 * scale >= amax`` always holds and
  activations are never clipped. Fully dynamic — no calibration state anywhere.

Do not "clean up" the rounding/scale math without a matching bit-exactness test.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .lora import LoRAMixin, shared_base_teacher_enabled
from .nvfp4 import (
    _e2m1_values_to_indices,
    _MaskedStraightThrough,
    _round_to_e2m1,
    pack_e2m1_rows,
    unpack_e2m1_rows,
)

MXFP4_BLOCK_SIZE = 32
MXFP4_MAX_VALUE = 6.0
E2M1_MAX_EXP = 2  # floor(log2(MXFP4_MAX_VALUE))
E8M0_BIAS = 127
E8M0_MAX_EXP = 127
E8M0_MIN_EXP = -127  # biased byte 0; 255 (NaN) is never emitted


def _e8m0_exponent_even(block_amax: torch.Tensor) -> torch.Tensor:
    """EVEN-mode E8M0 scale exponent ``k`` (``scale = 2**k``) from FP32 block maxima.

    Adding ``0x200000`` (``1 << (23 - 2)``) to the FP32 bit pattern carries into the
    exponent iff the mantissa fraction is >= 1.75, so masking the mantissa afterwards
    threshold-rounds the amax to a power of two. All-zero blocks are pinned to the
    minimum exponent instead of propagating ``log2(0) = -inf`` into the stored byte.
    """

    bits = block_amax.float().contiguous().view(torch.int32)
    rounded = ((bits + 0x200000) & 0x7F800000).view(torch.float32)
    rounded = torch.where(rounded == 0, torch.full_like(rounded, 2.0**-126), rounded)
    return (torch.log2(rounded).floor() - E2M1_MAX_EXP).clamp(E8M0_MIN_EXP, E8M0_MAX_EXP)


# FP32 nearest neighbour of 1/6 (0x3E2AAAAB) — the torch stand-in for the kernel's
# ``rcp_approx_ftz(6.0)``. At boundary amaxes ({6, 3, 1.5} * 2^k) the FP32 product
# ``amax * rcp`` is within half an ulp of the exact power of two and rounds back to it
# (RNE) for either 1-ulp rcp variant, so no bump happens there; a disagreement with the
# hardware would need ``amax/6`` within ~2^-25 relative of a power-of-two from an
# irregular amax. The GPU fuzz vs ``flashinfer.mxfp4_quantize`` pins (and, if needed,
# corrects) this constant.
_RCP_SIX = 0.16666667163372039794921875


def _e8m0_exponent_ceil(block_amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ceil-mode E8M0 scale exponent ``k`` (``scale = 2**k``) from FP32 block maxima.

    Replicates the serving-side activation quantizer (flashinfer ``mxfp4_quantize``,
    cute-dsl backend): ``__nv_cvt_float_to_e8m0(amax * rcp(6), SATFINITE, RoundPosInf)``
    — take the FP32 exponent of ``amax/6`` and add 1 when the mantissa is nonzero
    (round towards +inf), clamp the biased result to [0, 254]. The scale is therefore
    the smallest power of two >= amax/6, so ``6 * scale >= amax`` always holds and no
    activation value is ever clipped.

    Returns ``(exponent, zero_block)``: ``zero_block`` marks blocks the kernel maps to
    scale byte 0 (amax <= 6 * 2^-127, incl. all-zero blocks), whose decoded inverse
    scale is exactly 0 — the kernel emits all-zero codes for them.
    """

    q = block_amax.float().contiguous() * _RCP_SIX
    bits = q.view(torch.int32)
    exponent_biased = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    # Hardware corner: for subnormal q with mantissa <= 0x400000 (i.e. q <= 2^-127) the
    # cvt suppresses the bump, landing on biased byte 0 — the "zero scale" encoding.
    tiny = (exponent_biased == 0) & (mantissa <= 0x400000)
    bump = ((mantissa != 0) & ~tiny).to(exponent_biased.dtype)
    biased = (exponent_biased + bump).clamp(max=E8M0_BIAS + E8M0_MAX_EXP)
    zero_block = biased == 0
    return (biased - E8M0_BIAS).to(torch.float32), zero_block


def dequantize_mxfp4(packed_weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """Dequantize packed MXFP4 tensors to FP32."""

    if weight_scale.dtype != torch.uint8:
        raise ValueError(f"MXFP4 scales must be biased uint8 exponents, got {weight_scale.dtype}.")
    indices = unpack_e2m1_rows(packed_weight)
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=packed_weight.device)
    values = levels[(indices & 0x07).long()]
    values = torch.where((indices & 0x08) != 0, -values, values)
    if values.shape[-1] != weight_scale.shape[-1] * MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"MXFP4 scale shape {tuple(weight_scale.shape)} is incompatible with "
            f"weight shape {tuple(packed_weight.shape)}."
        )
    scale = torch.exp2(weight_scale.float() - E8M0_BIAS)
    return values * scale.repeat_interleave(MXFP4_BLOCK_SIZE, dim=-1)


@dataclass(frozen=True)
class MXFP4QuantizationResult:
    packed_weight: torch.Tensor
    weight_scale: torch.Tensor
    reconstruction: torch.Tensor


@torch.no_grad()
def quantize_mxfp4(weight: torch.Tensor) -> MXFP4QuantizationResult:
    """Pack a 2D weight on the EVEN-rule MXFP4 lattice."""

    if weight.ndim != 2:
        raise ValueError(f"MXFP4 packing expects a 2D weight, got shape {tuple(weight.shape)}.")
    if weight.shape[-1] % MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"MXFP4 packing requires in_features divisible by {MXFP4_BLOCK_SIZE}, got {weight.shape[-1]}."
        )
    if weight.is_meta:
        raise ValueError("Cannot pack an MXFP4 lattice from a meta tensor.")

    original_dtype = weight.dtype
    blocks = weight.float().reshape(weight.shape[0], -1, MXFP4_BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    exponent = _e8m0_exponent_even(block_amax)
    scale = torch.exp2(exponent)
    code_values = _round_to_e2m1((blocks / scale).clamp(-MXFP4_MAX_VALUE, MXFP4_MAX_VALUE))
    reconstruction = (code_values * scale).reshape_as(weight).to(original_dtype)
    code_indices = _e2m1_values_to_indices(code_values.reshape_as(weight))
    return MXFP4QuantizationResult(
        packed_weight=pack_e2m1_rows(code_indices),
        weight_scale=(exponent.squeeze(-1) + E8M0_BIAS).to(torch.uint8),
        reconstruction=reconstruction,
    )


def _fake_quantize_mxfp4_torch(
    weight: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference PyTorch MXFP4 weight QDQ, returning reconstruction and STE mask."""

    original_dtype = weight.dtype
    latent = weight.float()
    padding = (-latent.shape[-1]) % block_size
    padded = F.pad(latent, (0, padding)) if padding else latent
    blocks = padded.reshape(*padded.shape[:-1], -1, block_size)

    with torch.no_grad():
        block_amax = blocks.detach().abs().amax(dim=-1, keepdim=True)
        scale = torch.exp2(_e8m0_exponent_even(block_amax))

    normalized = (blocks / scale).clamp(-MXFP4_MAX_VALUE, MXFP4_MAX_VALUE)
    quantized = _round_to_e2m1(normalized) * scale
    in_range = blocks.detach().abs() <= scale * MXFP4_MAX_VALUE
    quantized = quantized.reshape_as(padded)[..., : latent.shape[-1]].to(original_dtype)
    in_range = in_range.reshape_as(padded)[..., : latent.shape[-1]]
    return quantized, in_range


def fake_quantize_mxfp4(weight: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE) -> torch.Tensor:
    """Fake-quantize a linear weight with E2M1 values and EVEN-rule E8M0 block scales.

    The final dimension is grouped in blocks of 32 (zero-padded when not divisible;
    the padding never widens a block's amax). Quantization statistics are detached,
    while the masked straight-through estimator sends gradients to the high-precision
    latent weight; elements clipped by the EVEN scale rule get zero gradient. CUDA
    tensors use Triton unless ``QATFACTORY_FP4_DISABLE_TRITON`` is set.
    """

    if block_size != MXFP4_BLOCK_SIZE:
        raise ValueError(f"MXFP4 requires block_size={MXFP4_BLOCK_SIZE}, got {block_size}.")
    if weight.ndim != 2:
        raise ValueError(f"MXFP4Linear expects a 2D weight, got shape {tuple(weight.shape)}.")
    if weight.is_meta:
        return weight

    from .fp4_triton import HAS_TRITON, triton_fake_quantize_mxfp4

    disabled = os.environ.get("QATFACTORY_FP4_DISABLE_TRITON", "") not in ("", "0", "false", "False")
    with torch.no_grad():
        if weight.is_cuda and HAS_TRITON and not disabled:
            quantized, in_range = triton_fake_quantize_mxfp4(weight)
        else:
            quantized, in_range = _fake_quantize_mxfp4_torch(weight, block_size=block_size)
    return _MaskedStraightThrough.apply(weight, quantized, in_range)


def _fake_quantize_mxfp4_activation_torch(activation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference PyTorch MXFP4 activation QDQ and its all-true STE mask."""

    shape = activation.shape
    original_dtype = activation.dtype
    latent = activation.reshape(-1, shape[-1]).float()
    blocks = latent.reshape(latent.shape[0], -1, MXFP4_BLOCK_SIZE)

    with torch.no_grad():
        block_amax = blocks.detach().abs().amax(dim=-1, keepdim=True)
        exponent, zero_block = _e8m0_exponent_ceil(block_amax)
        scale = torch.exp2(exponent)

    quantized = _round_to_e2m1((blocks / scale).clamp(-MXFP4_MAX_VALUE, MXFP4_MAX_VALUE)) * scale
    quantized = torch.where(zero_block, torch.zeros_like(quantized), quantized)
    quantized = quantized.reshape_as(latent).to(original_dtype).reshape(shape)
    in_range = torch.ones_like(quantized, dtype=torch.bool)
    return quantized, in_range


def fake_quantize_mxfp4_activation(activation: torch.Tensor) -> torch.Tensor:
    """Fake-quantize an input activation with E2M1 values on ceil-rule E8M0 scales.

    Matches vLLM W4A4 serving (flashinfer ``mxfp4_quantize``, cute-dsl backend): fully
    dynamic per-token block-32 scales along the feature dimension — the smallest power
    of two >= block_amax/6 — so no value is ever clipped and there is no calibration
    state anywhere (same input, bit-identical output; gradient-checkpoint recompute is
    deterministic by construction). The straight-through estimator's in-range mask is
    all-true under this rule, so gradients pass through to ``activation`` unmasked and
    keep flowing to the upstream latent weights.
    """

    if activation.is_meta:
        return activation
    shape = activation.shape
    if shape[-1] % MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"MXFP4 W4A4 requires in_features divisible by {MXFP4_BLOCK_SIZE}, got shape {tuple(shape)}."
        )

    from .fp4_triton import HAS_TRITON, triton_fake_quantize_mxfp4_activation

    disabled = os.environ.get("QATFACTORY_FP4_DISABLE_TRITON", "") not in ("", "0", "false", "False")
    with torch.no_grad():
        if activation.is_cuda and HAS_TRITON and not disabled:
            latent = activation.reshape(-1, shape[-1]).contiguous()
            quantized = triton_fake_quantize_mxfp4_activation(latent).reshape(shape)
            in_range = torch.ones_like(quantized, dtype=torch.bool)
        else:
            quantized, in_range = _fake_quantize_mxfp4_activation_torch(activation)
    return _MaskedStraightThrough.apply(activation, quantized, in_range)


class MXFP4Linear(LoRAMixin, nn.Linear):
    """Linear layer whose weight is MXFP4 fake-quantized on every forward.

    With ``_quantize_activations`` set (W4A4), the input activation is also MXFP4
    fake-quantized — dynamically, exactly as vLLM's native W4A4 kernel does at serve
    time — so training matches the deployed W4A4 numerics instead of the weight-only
    (W4A16) approximation. No scale state is involved (contrast NVFP4Linear).

    With a LoRA adapter attached (:meth:`init_lora`), the frozen base weight and the
    adapter are merged before quantization; inside ``shared_base_teacher()`` the layer bypasses both
    to serve the shared frozen base as the full-precision teacher.
    """

    @classmethod
    def from_linear(cls, module: nn.Linear) -> "MXFP4Linear":
        converted = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        converted.weight = module.weight
        converted.bias = module.bias
        converted._quantize_activations = False
        return converted

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if shared_base_teacher_enabled():
            return F.linear(input, self.weight, self.bias)
        if self._quantize_activations:
            input = fake_quantize_mxfp4_activation(input)
        return F.linear(input, fake_quantize_mxfp4(self._effective_weight()), self.bias)
