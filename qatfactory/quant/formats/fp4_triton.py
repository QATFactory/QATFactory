"""Fused Triton fake-quantization kernels for NVFP4 and MXFP4.

Each program handles eight adjacent quantization blocks. Inputs are loaded once in
their native dtype, promoted to FP32 in registers, reduced for the block scale, and
written once as reconstructed values (plus the STE mask where required).
"""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - triton is a soft dependency
    HAS_TRITON = False


_MODE_NVFP4 = 0
_MODE_MXFP4_EVEN = 1
_MODE_MXFP4_CEIL = 2


if HAS_TRITON:
    @triton.jit
    def _fdiv_rn(numerator, denominator):
        """IEEE FP32 division, avoiding reciprocal-approximation lowering."""

        return tl.inline_asm_elementwise(
            asm="div.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[numerator, denominator],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _fmul_rn(lhs, rhs):
        """IEEE FP32 multiplication with gradual underflow and no contraction."""

        return tl.inline_asm_elementwise(
            asm="mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[lhs, rhs],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _e4m3fn_rne_positive(value):
        """Round non-negative FP32 to E4M3FN and return the exact FP32 value.

        This integer implementation is used instead of ``tl.float8e4nv`` so the
        RNE and subnormal behavior do not depend on Triton's conversion lowering.
        Normal values discard 20 FP32 mantissa bits with an explicit ties-to-even
        carry. Values below 2^-6 are rounded in integer units of 2^-9, including
        the tie between zero and the minimum E4M3 subnormal.
        """

        is_nan = value != value  # noqa: PLR0124 - portable NaN test in Triton JIT
        clamped = tl.minimum(tl.maximum(value, 0.0), 448.0)
        bits = clamped.to(tl.int32, bitcast=True)
        exponent_biased = (bits >> 23) & 0xFF

        mantissa_24 = (bits & 0x7FFFFF) | 0x800000
        shift = tl.minimum(tl.maximum(141 - exponent_biased, 1), 24)
        truncated = mantissa_24 >> shift
        remainder = mantissa_24 & ((1 << shift) - 1)
        halfway = 1 << (shift - 1)
        increment = (remainder > halfway) | ((remainder == halfway) & ((truncated & 1) != 0))
        subnormal_code = tl.where(exponent_biased >= 117, truncated + increment.to(tl.int32), 0)
        subnormal = _fmul_rn(subnormal_code.to(tl.float32), 0.001953125)

        # Keep three mantissa bits; 0x7ffff implements RNE and the retained LSB
        # supplies the tie parity. The explicit [0, 448] clamp prevents overflow.
        rounded_bits = (bits + 0x7FFFF + ((bits >> 20) & 1)) & -0x100000
        normal = rounded_bits.to(tl.float32, bitcast=True)
        converted = tl.where(clamped < 0.015625, subnormal, normal)
        return tl.where(is_nan, value, converted)

    @triton.jit
    def _power_of_two_f32(exponent):
        """Construct 2**exponent exactly for integer exponent in [-127, 127]."""

        normal_bits = (exponent + 127) << 23
        bits = tl.where(exponent == -127, 0x400000, normal_bits)
        return bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _e2m1_rne(value):
        """Round clamped FP32 values to the E2M1 codebook, ties to even."""

        magnitude = tl.abs(value)
        index = (
            (magnitude > 0.25).to(tl.int32)
            + (magnitude > 0.75).to(tl.int32)
            + (magnitude > 1.25).to(tl.int32)
            + (magnitude > 1.75).to(tl.int32)
            + (magnitude > 2.5).to(tl.int32)
            + (magnitude > 3.5).to(tl.int32)
            + (magnitude > 5.0).to(tl.int32)
        )
        tie = (
            (magnitude == 0.25)
            | (magnitude == 0.75)
            | (magnitude == 1.25)
            | (magnitude == 1.75)
            | (magnitude == 2.5)
            | (magnitude == 3.5)
            | (magnitude == 5.0)
        )
        index += (tie & ((index & 1) != 0)).to(tl.int32)

        level = tl.where(
            index == 0,
            0.0,
            tl.where(
                index == 1,
                0.5,
                tl.where(
                    index == 2,
                    1.0,
                    tl.where(
                        index == 3,
                        1.5,
                        tl.where(
                            index == 4,
                            2.0,
                            tl.where(index == 5, 3.0, tl.where(index == 6, 4.0, 6.0)),
                        ),
                    ),
                ),
            ),
        )
        return tl.where(value < 0.0, -level, level)

    @triton.jit
    def _fp4_fake_quant_kernel(
        x_ptr,
        output_ptr,
        in_range_ptr,
        global_scale_ptr,
        rows,
        cols,
        n_blocks,
        BLOCK_SIZE: tl.constexpr,
        BLOCKS_PER_PROGRAM: tl.constexpr,
        MODE: tl.constexpr,
        STORE_MASK: tl.constexpr,
    ):
        block_ids = tl.program_id(0) * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
        lanes = tl.arange(0, BLOCK_SIZE)
        row = block_ids[:, None] // n_blocks
        block_in_row = block_ids[:, None] - row * n_blocks
        col = block_in_row * BLOCK_SIZE + lanes[None, :]
        offsets = row * cols + col
        valid = (row < rows) & (col < cols)

        # The pointer load converts fp16/bf16/fp32 to FP32 in registers. No full
        # FP32 input tensor is materialized.
        values = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        block_amax = tl.max(tl.abs(values), axis=1)[:, None]

        if MODE == 0:
            global_scale = tl.load(global_scale_ptr).to(tl.float32)
            denominator = _fmul_rn(global_scale, 6.0)
            scale_input = _fdiv_rn(block_amax, denominator)
            block_scale = _e4m3fn_rne_positive(scale_input)
            block_scale = tl.where(block_scale == 0.0, 0.001953125, block_scale)
            scale = _fmul_rn(block_scale, global_scale)
            # The global scale is the tensor amax over 6, so it is zero only when every
            # value in the tensor is zero. Handling that here rather than with a host-side
            # branch keeps the caller from synchronizing the device on every call; any
            # positive substitute reproduces the reference zeros exactly.
            scale = tl.where(global_scale == 0.0, 1.0, scale)
        elif MODE == 1:
            amax_bits = block_amax.to(tl.int32, bitcast=True)
            rounded_exponent = (((amax_bits + 0x200000) & 0x7F800000) >> 23) & 0xFF
            exponent = tl.maximum(rounded_exponent - 129, -127)
            exponent = tl.where(rounded_exponent == 0xFF, 127, exponent)
            exponent = tl.minimum(exponent, 127)
            scale = _power_of_two_f32(exponent)
        else:
            q = _fmul_rn(block_amax, 0.16666667163372039794921875)
            q_bits = q.to(tl.int32, bitcast=True)
            exponent_biased = (q_bits >> 23) & 0xFF
            mantissa = q_bits & 0x7FFFFF
            tiny = (exponent_biased == 0) & (mantissa <= 0x400000)
            bump = (mantissa != 0) & ~tiny
            biased = tl.minimum(exponent_biased + bump.to(tl.int32), 254)
            zero_block = biased == 0
            scale = _power_of_two_f32(biased - 127)

        raw_normalized = _fdiv_rn(values, scale)
        normalized = tl.minimum(tl.maximum(raw_normalized, -6.0), 6.0)
        code = _e2m1_rne(normalized)
        reconstructed = _fmul_rn(code, scale)
        # PyTorch propagates a 0/0 NaN through clamp -> E2M1 sign -> multiply.
        normalized_is_nan = raw_normalized != raw_normalized  # noqa: PLR0124 - Triton JIT NaN test
        reconstructed = tl.where(normalized_is_nan, raw_normalized, reconstructed)

        if MODE == 2:
            reconstructed = tl.where(zero_block, 0.0, reconstructed)

        # A typed tl.store performs the same RNE FP32 -> fp16/bf16 cast used by
        # Tensor.to(original_dtype); this is covered by the GPU bit-exact tests.
        tl.store(output_ptr + offsets, reconstructed, mask=valid)
        if STORE_MASK:
            limit = _fmul_rn(scale, 6.0)
            in_range = tl.abs(values) <= limit
            tl.store(in_range_ptr + offsets, in_range.to(tl.int8), mask=valid)


_BLOCKS_PER_PROGRAM = 8


def _launch_fp4(
    tensor: torch.Tensor,
    *,
    block_size: int,
    mode: int,
    global_scale: torch.Tensor | None = None,
    store_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not HAS_TRITON:  # pragma: no cover - guarded by caller
        raise RuntimeError("Triton is not available.")
    if tensor.ndim != 2:
        raise ValueError(f"Triton FP4 fake quantization expects a 2D tensor, got {tuple(tensor.shape)}.")

    source = tensor.contiguous()
    rows, cols = source.shape
    output = torch.empty_like(source)
    mask_storage = torch.empty(source.shape, dtype=torch.int8, device=source.device) if store_mask else None
    if source.numel() == 0:
        return output, mask_storage

    n_blocks = triton.cdiv(cols, block_size)
    total_blocks = rows * n_blocks
    scale_arg = global_scale if global_scale is not None else source
    _fp4_fake_quant_kernel[(triton.cdiv(total_blocks, _BLOCKS_PER_PROGRAM),)](
        source,
        output,
        mask_storage if mask_storage is not None else source,
        scale_arg,
        rows,
        cols,
        n_blocks,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROGRAM=_BLOCKS_PER_PROGRAM,
        MODE=mode,
        STORE_MASK=store_mask,
        num_warps=4,
    )
    return output, mask_storage


def triton_fake_quantize_nvfp4(
    tensor: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused NVFP4 QDQ with group-16 E4M3 scales and an FP32 global scale."""

    quantized, mask_storage = _launch_fp4(
        tensor,
        block_size=16,
        mode=_MODE_NVFP4,
        global_scale=global_scale,
        store_mask=True,
    )
    assert mask_storage is not None
    return quantized, mask_storage.bool()


def triton_fake_quantize_mxfp4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused MXFP4 weight QDQ with group-32 EVEN-rule E8M0 scales."""

    quantized, mask_storage = _launch_fp4(
        tensor,
        block_size=32,
        mode=_MODE_MXFP4_EVEN,
        store_mask=True,
    )
    assert mask_storage is not None
    return quantized, mask_storage.bool()


def triton_fake_quantize_mxfp4_activation(tensor: torch.Tensor) -> torch.Tensor:
    """Fused MXFP4 activation QDQ with group-32 CEIL-rule E8M0 scales."""

    quantized, _ = _launch_fp4(
        tensor,
        block_size=32,
        mode=_MODE_MXFP4_CEIL,
        store_mask=False,
    )
    return quantized
