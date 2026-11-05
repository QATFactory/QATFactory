"""Q4_K two-level weight-only fake quantization with a masked straight-through estimator.

Q4_K (llama.cpp / GGUF ``block_q4_K``) is a two-level affine format: super-blocks of
``QK_K = 256`` weights split into 8 sub-blocks of 32. Each sub-block has an affine
``(scale_j >= 0, min_j >= 0)`` fit; the 8 scales and 8 mins are themselves quantized to
6-bit integers ``ls_j``/``lm_j`` against two fp16 super-block scales ``d``/``dmin``.
Reconstruction is ``x ~= d*ls_j*q_i - dmin*lm_j`` with ``q_i in {0..15}``.

This module is the single source of truth for the Q4_K lattice: the QAT fake-quant forward
and the GGUF exporter call the *same* deterministic encoder, so the deployed GGUF blocks
dequantize value-for-value to what the student trained/eval'd with. The reconstruction here
matches ggml's ``dequantize_row_q4_K`` exactly (it is the plain affine formula with the same
fp16 ``d``/``dmin`` bits and no summation/rounding ambiguity). The *encoder* follows the
no-imatrix reference path (``quantize_row_q4_K_ref`` -> ``make_qkx2_quants``); it does not
need to be byte-identical to llama.cpp's grid search because losslessness comes from using
this encoder on both sides, not from matching llama.cpp's encoder.
"""

import logging
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


logger = logging.getLogger(__name__)


QK_K = 256
Q4K_GROUP = 32
Q4K_NSUB = QK_K // Q4K_GROUP  # 8 sub-blocks per super-block
Q4K_NMAX = 15  # 4-bit weight codes
Q4K_L2MAX = 63  # 6-bit level-2 scale/min codes
Q4K_BLOCK_BYTES = 2 + 2 + 12 + 128  # d, dmin, packed scales, packed qs -> 144

# make_qkx2_quants parameters for the no-imatrix reference path (quantize_row_q4_K_ref).
_RMIN = -1.0
_RDELTA = 0.1
_NSTEP = 20


def _round(x: torch.Tensor) -> torch.Tensor:
    """Round half to even, matching ggml ``nearest_int`` (the magic-number trick)."""

    return torch.round(x)


def _fp16(x: torch.Tensor) -> torch.Tensor:
    """Round-trip through fp16 (round-to-nearest-even), matching GGML_FP32_TO_FP16."""

    return x.to(torch.float16).to(torch.float32)


class _MaskedStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, latent: torch.Tensor, quantized: torch.Tensor, in_range: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(in_range)
        return quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (in_range,) = ctx.saved_tensors
        return grad_output * in_range, None, None


def _make_qkx2(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized ``make_qkx2_quants`` over the leading dims; last dim is the 32-weight sub-block.

    Returns ``(scale, the_min, codes)`` with ``scale >= 0``, ``the_min >= 0`` (the subtracted
    offset), and integer ``codes in [0, 15]``.
    """

    nmax = Q4K_NMAX
    zero = x.new_zeros(())
    xmax = x.max(dim=-1).values
    vmin = torch.minimum(x.min(dim=-1).values, zero)  # min(min, 0)
    degenerate = xmax == vmin
    denom = torch.where(degenerate, torch.ones_like(xmax), xmax - vmin)

    iscale = nmax / denom
    scale = 1.0 / iscale
    vmin_e = vmin.unsqueeze(-1)
    codes = torch.clamp(_round(iscale.unsqueeze(-1) * (x - vmin_e)), 0, nmax)
    diff = scale.unsqueeze(-1) * codes + vmin_e - x
    best_error = (w * diff * diff).sum(dim=-1)

    sum_w = w.sum(dim=-1)
    sum_x = (w * x).sum(dim=-1)
    cur_min = vmin.clone()
    for step in range(_NSTEP + 1):
        trial_iscale = (_RMIN + _RDELTA * step + nmax) / denom
        aux = torch.clamp(_round(trial_iscale.unsqueeze(-1) * (x - vmin_e)), 0, nmax)
        sum_l = (w * aux).sum(dim=-1)
        sum_l2 = (w * aux * aux).sum(dim=-1)
        sum_xl = (w * aux * x).sum(dim=-1)
        determinant = sum_w * sum_l2 - sum_l * sum_l
        safe_det = torch.where(determinant > 0, determinant, torch.ones_like(determinant))
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / safe_det
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / safe_det
        positive_min = this_min > 0
        safe_l2 = torch.where(sum_l2 > 0, sum_l2, torch.ones_like(sum_l2))
        this_scale = torch.where(positive_min, sum_xl / safe_l2, this_scale)
        this_min = torch.where(positive_min, torch.zeros_like(this_min), this_min)
        trial_diff = this_scale.unsqueeze(-1) * aux + this_min.unsqueeze(-1) - x
        cur_error = (w * trial_diff * trial_diff).sum(dim=-1)

        better = (determinant > 0) & (cur_error < best_error)
        better_e = better.unsqueeze(-1)
        codes = torch.where(better_e, aux, codes)
        best_error = torch.where(better, cur_error, best_error)
        scale = torch.where(better, this_scale, scale)
        cur_min = torch.where(better, this_min, cur_min)

    scale = torch.where(degenerate, torch.zeros_like(scale), scale)
    the_min = -torch.where(degenerate, vmin, cur_min)
    codes = torch.where(degenerate.unsqueeze(-1), torch.zeros_like(codes), codes)
    return scale, the_min, codes


@dataclass(frozen=True)
class _Q4KEncoding:
    d: torch.Tensor  # fp16-rounded super-block scale, [K, 1] float32
    dmin: torch.Tensor  # fp16-rounded super-block min, [K, 1] float32
    ls: torch.Tensor  # 6-bit sub-block scale codes, [K, 8] int64
    lm: torch.Tensor  # 6-bit sub-block min codes, [K, 8] int64
    q: torch.Tensor  # 4-bit weight codes, [K, 8, 32] float32
    reconstruction: torch.Tensor  # [K, 8, 32] float32
    in_range: torch.Tensor  # STE mask, [K, 8, 32] bool


# Supported level-1 (per-sub-block) encoders. ``grid_search`` is the llama.cpp
# ``make_qkx2_quants`` weighted-least-squares grid search; ``minmax`` is the naive asymmetric
# affine fit (llama.cpp's own seed, no search). Both share the level-2 6-bit scale/min quant.
Q4K_METHODS = ("grid_search", "minmax")


def _level1_grid_search_torch(sub: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted-LS grid-search level-1 fit; returns ``(scale, the_min)`` of shape ``[K, 8]``."""

    n_super = sub.shape[0]
    x = sub.reshape(n_super * Q4K_NSUB, Q4K_GROUP)
    av_x = torch.sqrt((x * x).sum(dim=-1, keepdim=True) / Q4K_GROUP)
    weights = av_x + x.abs()
    scale, the_min, _ = _make_qkx2(x, weights)
    return scale.reshape(n_super, Q4K_NSUB), the_min.reshape(n_super, Q4K_NSUB)


def _level1_minmax_torch(sub: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive asymmetric min-max level-1 fit; returns ``(scale, the_min)`` of shape ``[K, 8]``.

    Per sub-block: force the signed min non-positive (``vmin = min(min(x), 0)``), spread the
    ``[vmin, max]`` range across the 16 codes (``scale = (max - vmin) / 15``), and subtract the
    offset ``the_min = -vmin >= 0``. This is exactly llama.cpp's ``make_qkx2_quants`` seed
    before its grid search; ``scale`` degenerates to 0 when ``max == vmin`` (handled downstream).
    """

    xmax = sub.max(dim=-1).values
    vmin = sub.min(dim=-1).values.clamp(max=0.0)  # min(min, 0)
    scale = (xmax - vmin) / Q4K_NMAX
    the_min = -vmin
    return scale, the_min


def _level2_encode_torch(sub: torch.Tensor, scale: torch.Tensor, the_min: torch.Tensor) -> _Q4KEncoding:
    """Shared level-2 6-bit scale/min quant + re-encode the 4-bit codes from fp16 d/dmin.

    ``sub`` is ``[K, 8, 32]``; ``scale``/``the_min`` are the level-1 fits ``[K, 8]``. The final
    codes are recomputed against the fp16-rounded ``d``/``dmin`` so encode matches the decoder.
    """

    max_scale = scale.max(dim=-1, keepdim=True).values
    max_min = the_min.max(dim=-1, keepdim=True).values
    inv_scale = torch.where(max_scale > 0, 63.0 / max_scale, torch.zeros_like(max_scale))
    inv_min = torch.where(max_min > 0, 63.0 / max_min, torch.zeros_like(max_min))
    ls = torch.clamp(_round(inv_scale * scale), 0, Q4K_L2MAX).to(torch.int64)
    lm = torch.clamp(_round(inv_min * the_min), 0, Q4K_L2MAX).to(torch.int64)

    d = _fp16(max_scale / 63.0)
    dmin = _fp16(max_min / 63.0)
    deff = (d * ls.to(torch.float32)).unsqueeze(-1)  # [K, 8, 1]
    dm = (dmin * lm.to(torch.float32)).unsqueeze(-1)

    safe = deff != 0
    raw = torch.where(safe, (sub + dm) / torch.where(safe, deff, torch.ones_like(deff)), torch.zeros_like(sub))
    q = torch.where(safe, torch.clamp(_round(raw), 0, Q4K_NMAX), torch.zeros_like(sub))
    reconstruction = deff * q - dm
    in_range = (raw >= 0) & (raw <= Q4K_NMAX)
    return _Q4KEncoding(d=d, dmin=dmin, ls=ls, lm=lm, q=q, reconstruction=reconstruction, in_range=in_range)


def _encode_superblocks_torch(sub: torch.Tensor, method: str = "grid_search") -> _Q4KEncoding:
    """Encode contiguous super-blocks ``sub`` of shape ``[K, 8, 32]`` (float32).

    Reference PyTorch encoder (CPU fallback and equivalence ground truth). ``method`` selects
    the level-1 fit (``grid_search`` or ``minmax``); the CUDA hot path runs the same algorithm
    in a fused Triton kernel; see :func:`_encode_superblocks`.
    """

    if method == "grid_search":
        scale, the_min = _level1_grid_search_torch(sub)
    elif method == "minmax":
        scale, the_min = _level1_minmax_torch(sub)
    else:
        raise ValueError(f"Unknown Q4_K method {method!r}; supported: {Q4K_METHODS}.")
    return _level2_encode_torch(sub, scale, the_min)


def _encode_superblocks(sub: torch.Tensor, *, pack: bool = True, method: str = "grid_search") -> _Q4KEncoding:
    """Encode super-blocks ``sub`` of shape ``[K, 8, 32]`` (float32).

    Dispatches to the fused Triton kernel on CUDA (single encoder for both the training
    fake-quant and the save-time pack, so they stay bit-identical) and to the reference
    PyTorch encoder on CPU / when Triton is unavailable. ``pack=False`` skips the code-tensor
    outputs on the fake-quant hot path (only ``reconstruction``/``in_range`` are needed there).
    ``method`` selects the level-1 fit (``grid_search`` or ``minmax``).
    """

    if method not in Q4K_METHODS:
        raise ValueError(f"Unknown Q4_K method {method!r}; supported: {Q4K_METHODS}.")

    from .q4_k_triton import HAS_TRITON, triton_encode_superblocks

    disabled = os.environ.get("QATFACTORY_Q4K_DISABLE_TRITON", "") not in ("", "0", "false", "False")
    if disabled or not (sub.is_cuda and HAS_TRITON):
        return _encode_superblocks_torch(sub, method=method)

    recon, in_range, q, ls, lm, d, dmin = triton_encode_superblocks(sub, pack=pack, method=method)
    return _Q4KEncoding(d=d, dmin=dmin, ls=ls, lm=lm, q=q, reconstruction=recon, in_range=in_range)


def _as_superblocks(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"Q4_K expects a 2D weight, got shape {tuple(weight.shape)}.")
    if weight.shape[-1] % QK_K:
        raise ValueError(
            f"Q4_K requires in_features divisible by {QK_K}, got {weight.shape[-1]}. "
            "llama.cpp falls back to another type for such tensors; exclude them from the target set."
        )
    out_features, in_features = weight.shape
    return weight.float().reshape(out_features, in_features // QK_K, Q4K_NSUB, Q4K_GROUP).reshape(-1, Q4K_NSUB, Q4K_GROUP)


def fake_quantize_q4_k(
    weight: torch.Tensor, *, col_perm: torch.Tensor | None = None, method: str = "grid_search"
) -> torch.Tensor:
    """Fake-quantize a linear weight on the Q4_K lattice with a masked STE.

    In-features are grouped into super-blocks of 256 (8 sub-blocks of 32). Quantization
    statistics are detached; the straight-through estimator sends gradients to the
    high-precision latent weight where the 4-bit code is not saturated. ``method`` selects the
    level-1 fit (``grid_search`` or naive ``minmax``).

    ``col_perm`` handles Qwen3.5 ``out_proj``: GGUF conversion reorders its input columns in
    16-wide groups, which changes the Q4_K super-block groupings. Encoding the column-permuted
    weight and un-permuting the reconstruction makes the training reconstruction match the
    deployed (reordered) GGUF value-for-value.
    """

    if weight.is_meta:
        return weight
    original_dtype = weight.dtype
    out_features, in_features = weight.shape

    permuted = weight if col_perm is None else weight.index_select(1, col_perm.to(weight.device))
    with torch.no_grad():
        encoding = _encode_superblocks(_as_superblocks(permuted), pack=False, method=method)
    reconstruction = encoding.reconstruction.reshape(out_features, in_features).to(original_dtype)
    in_range = encoding.in_range.reshape(out_features, in_features)
    if col_perm is not None:
        inverse = torch.empty_like(col_perm)
        inverse[col_perm] = torch.arange(col_perm.numel(), device=col_perm.device)
        inverse = inverse.to(weight.device)
        reconstruction = reconstruction.index_select(1, inverse)
        in_range = in_range.index_select(1, inverse)
    return _MaskedStraightThrough.apply(weight, reconstruction, in_range)


@dataclass(frozen=True)
class Q4KQuantizationResult:
    packed_weight: torch.Tensor  # [out_features, n_super * 144] uint8 (ggml block_q4_K bytes)
    reconstruction: torch.Tensor  # [out_features, in_features], weight dtype


def _pack_scales(ls: torch.Tensor, lm: torch.Tensor) -> torch.Tensor:
    """Pack 8 6-bit ``ls`` and 8 6-bit ``lm`` into the 12-byte ggml scales layout."""

    n_super = ls.shape[0]
    scales = torch.zeros(n_super, 12, dtype=torch.uint8, device=ls.device)
    ls = ls.to(torch.int64)
    lm = lm.to(torch.int64)
    for j in range(Q4K_NSUB):
        if j < 4:
            scales[:, j] = (ls[:, j] & 0x3F).to(torch.uint8)
            scales[:, j + 4] = (lm[:, j] & 0x3F).to(torch.uint8)
        else:
            scales[:, j + 4] = ((ls[:, j] & 0xF) | ((lm[:, j] & 0xF) << 4)).to(torch.uint8)
            scales[:, j - 4] |= ((ls[:, j] >> 4) << 6).to(torch.uint8)
            scales[:, j] |= ((lm[:, j] >> 4) << 6).to(torch.uint8)
    return scales


def _pack_qs(q: torch.Tensor) -> torch.Tensor:
    """Pack 256 4-bit codes ``q`` ([K, 8, 32]) into the 128-byte ggml qs layout."""

    n_super = q.shape[0]
    codes = q.reshape(n_super, QK_K).to(torch.int64)
    qs = torch.zeros(n_super, 128, dtype=torch.uint8, device=q.device)
    for group in range(4):
        base = group * 64
        low = codes[:, base : base + 32]
        high = codes[:, base + 32 : base + 64]
        qs[:, group * 32 : (group + 1) * 32] = (low | (high << 4)).to(torch.uint8)
    return qs


@torch.no_grad()
def quantize_q4_k(weight: torch.Tensor, *, method: str = "grid_search") -> Q4KQuantizationResult:
    """Encode a 2D weight into ggml ``block_q4_K`` bytes plus the FP32 reconstruction.

    ``method`` selects the level-1 fit (``grid_search`` or naive ``minmax``); it must match the
    method used by the training fake-quant so the deployed blocks are lossless by construction.
    """

    if weight.is_meta:
        raise ValueError("Cannot pack a Q4_K lattice from a meta tensor.")
    out_features, in_features = weight.shape
    n_super_per_row = in_features // QK_K
    encoding = _encode_superblocks(_as_superblocks(weight), method=method)

    d_bytes = (encoding.d.reshape(-1, 1) / 1.0).to(torch.float16).contiguous().view(torch.uint8)
    dmin_bytes = encoding.dmin.reshape(-1, 1).to(torch.float16).contiguous().view(torch.uint8)
    scales = _pack_scales(encoding.ls, encoding.lm)
    qs = _pack_qs(encoding.q)
    blocks = torch.cat([d_bytes, dmin_bytes, scales, qs], dim=-1)  # [K, 144]
    packed = blocks.reshape(out_features, n_super_per_row * Q4K_BLOCK_BYTES).contiguous()

    reconstruction = encoding.reconstruction.reshape(out_features, in_features).to(weight.dtype)
    return Q4KQuantizationResult(packed_weight=packed, reconstruction=reconstruction)


def _unpack_scale_min(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert :func:`_pack_scales`; returns ``(ls, lm)`` int64 of shape ``[K, 8]``."""

    s = scales.to(torch.int64)
    ls = torch.zeros(s.shape[0], Q4K_NSUB, dtype=torch.int64, device=s.device)
    lm = torch.zeros_like(ls)
    for j in range(Q4K_NSUB):
        if j < 4:
            ls[:, j] = s[:, j] & 0x3F
            lm[:, j] = s[:, j + 4] & 0x3F
        else:
            ls[:, j] = (s[:, j + 4] & 0xF) | (((s[:, j - 4] >> 6) & 0x3) << 4)
            lm[:, j] = (s[:, j + 4] >> 4) | (((s[:, j] >> 6) & 0x3) << 4)
    return ls, lm


def dequantize_q4_k(packed_weight: torch.Tensor, in_features: int) -> torch.Tensor:
    """Dequantize ggml ``block_q4_K`` bytes to FP32, matching ``dequantize_row_q4_K``."""

    if packed_weight.dtype != torch.uint8 or packed_weight.ndim != 2:
        raise ValueError(f"Q4_K blocks must be 2D uint8, got {packed_weight.dtype} {tuple(packed_weight.shape)}.")
    out_features = packed_weight.shape[0]
    n_super_per_row = in_features // QK_K
    blocks = packed_weight.reshape(out_features * n_super_per_row, Q4K_BLOCK_BYTES)

    d = blocks[:, 0:2].contiguous().view(torch.float16).float()  # [K, 1]
    dmin = blocks[:, 2:4].contiguous().view(torch.float16).float()
    ls, lm = _unpack_scale_min(blocks[:, 4:16])
    qs = blocks[:, 16:144]

    low = (qs & 0x0F).to(torch.int64)
    high = ((qs >> 4) & 0x0F).to(torch.int64)
    codes = torch.zeros(qs.shape[0], QK_K, dtype=torch.int64, device=qs.device)
    for group in range(4):
        base = group * 64
        codes[:, base : base + 32] = low[:, group * 32 : (group + 1) * 32]
        codes[:, base + 32 : base + 64] = high[:, group * 32 : (group + 1) * 32]
    codes = codes.reshape(-1, Q4K_NSUB, Q4K_GROUP).float()

    deff = (d * ls.to(torch.float32)).unsqueeze(-1)
    dm = (dmin * lm.to(torch.float32)).unsqueeze(-1)
    values = deff * codes - dm
    return values.reshape(out_features, n_super_per_row * QK_K)


@torch.no_grad()
def pack_deploy_weight(
    weight: torch.Tensor,
    *,
    row_perm: torch.Tensor | None = None,
    col_perm: torch.Tensor | None = None,
    method: str = "grid_search",
) -> torch.Tensor:
    """Pack the deployed-layout Q4_K blocks for a target weight.

    ``row_perm``/``col_perm`` apply the GGUF architecture reorders (e.g. Qwen3.5 linear-attn
    V-head grouped->tiled row reorder, or ``out_proj`` column reorder) so the stored blocks
    match the layout ``convert_hf_to_gguf.py`` produces and can be spliced in unchanged.
    ``method`` selects the level-1 fit and must match the training fake-quant.
    """

    if row_perm is not None:
        weight = weight.index_select(0, row_perm.to(weight.device))
    if col_perm is not None:
        weight = weight.index_select(1, col_perm.to(weight.device))
    return quantize_q4_k(weight, method=method).packed_weight


def v_head_reorder_perm(num_k_heads: int, num_v_per_k: int, head_dim: int) -> torch.Tensor:
    """Grouped-by-K -> tiled V-head permutation ``P`` with ``deployed[i] = hf[P[i]]``.

    Mirrors llama.cpp's ``_reorder_v_heads`` (reshape ``[num_k, num_v_per_k, head_dim]``,
    swap the first two axes, flatten).
    """

    total = num_k_heads * num_v_per_k * head_dim
    index = torch.arange(total, dtype=torch.long).reshape(num_k_heads, num_v_per_k, head_dim)
    return index.permute(1, 0, 2).reshape(total).contiguous()


def configure_q4k_reorders(model: nn.Module) -> int:
    """Attach GGUF reorder permutations to Qwen3.5 linear-attn ``Q4KLinear`` targets.

    Sets ``q4k_row_perm`` (V-head row reorder) / ``q4k_col_perm`` (``out_proj`` column reorder)
    so both the fake-quant reconstruction and the save-time packed blocks match the deployed
    GGUF layout. No-op for architectures without a mismatched linear-attn head count.
    """

    config = model.config
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    num_k = int(getattr(text_config, "linear_num_key_heads", 0) or 0)
    num_v = int(getattr(text_config, "linear_num_value_heads", 0) or 0)
    if not num_k or not num_v or num_k == num_v:
        return 0
    num_v_per_k = num_v // num_k
    head_v = int(text_config.linear_value_head_dim)
    head_k = int(text_config.linear_key_head_dim)

    v_perm = v_head_reorder_perm(num_k, num_v_per_k, head_v)
    scalar_perm = v_head_reorder_perm(num_k, num_v_per_k, 1)
    qk_rows = head_k * num_k + head_k * num_k
    qkv_perm = torch.cat([torch.arange(qk_rows, dtype=torch.long), qk_rows + v_perm])

    configured = 0
    for name, module in model.named_modules():
        if not isinstance(module, Q4KLinear):
            continue
        if name.endswith(".linear_attn.in_proj_qkv"):
            module.q4k_row_perm = qkv_perm
        elif name.endswith(".linear_attn.in_proj_z"):
            module.q4k_row_perm = v_perm
        elif name.endswith((".linear_attn.in_proj_a", ".linear_attn.in_proj_b")):
            module.q4k_row_perm = scalar_perm
        elif name.endswith(".linear_attn.out_proj"):
            module.q4k_col_perm = v_perm
        else:
            continue
        configured += 1
    return configured


class Q4KLinear(nn.Linear):
    """Linear layer whose weight is Q4_K fake-quantized on every forward.

    ``quant_method`` selects the level-1 encoder (``grid_search`` or naive ``minmax``); the
    quant format sets it at conversion so the fake-quant and save-time pack stay consistent.
    """

    @classmethod
    def from_linear(cls, module: nn.Linear, *, method: str = "grid_search") -> "Q4KLinear":
        if module.in_features % QK_K:
            raise ValueError(
                f"Q4_K requires in_features divisible by {QK_K}, got {module.in_features}; "
                "exclude this module from the target set."
            )
        if method not in Q4K_METHODS:
            raise ValueError(f"Unknown Q4_K method {method!r}; supported: {Q4K_METHODS}.")
        converted = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        converted.weight = module.weight
        converted.bias = module.bias
        converted.quant_method = method
        # Optional GGUF reorder permutations (Qwen3.5 linear-attn); plain (non-persistent)
        # attributes so they never enter the checkpoint state_dict.
        converted.q4k_row_perm = None
        converted.q4k_col_perm = None
        return converted

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        method = getattr(self, "quant_method", "grid_search")
        return F.linear(input, fake_quantize_q4_k(self.weight, col_perm=self.q4k_col_perm, method=method), self.bias)
