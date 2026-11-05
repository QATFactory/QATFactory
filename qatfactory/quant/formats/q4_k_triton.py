"""Fused Triton implementation of the Q4_K two-level encoder.

This is a drop-in accelerator for the reference PyTorch encoder in :mod:`q4_k`
(``_encode_superblocks_torch``). It computes the *identical* algorithm for either level-1
method (``grid_search`` = ``make_qkx2_quants`` search, or naive ``minmax``) plus the level-2
6-bit scale/min quantization, but runs the whole per-super-block encode in-register with
**one program per 256-weight super-block**, turning the memory-bound elementwise loop into a
single pass over the weight.

Equivalence contract: this kernel is the single encoder for both the training fake-quant and
the save-time GGUF pack (both dispatch here on CUDA), so the deployed blocks dequantize
value-for-value to the trained student — lossless *by construction*, exactly as with the
PyTorch reference. Against that reference it agrees on ~99.4% of weights bit-for-bit; the rest
differ by at most one 4-bit code step, a float32 reduction-order tie-break in the grid search
(both selections are equally valid Q4_K encodings). See ``docs/q4_k_qad_plan.md``.
"""

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:  # pragma: no cover - triton is a soft dependency
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _q4k_encode_kernel(
        x_ptr,
        recon_ptr,
        inrange_ptr,
        q_ptr,
        ls_ptr,
        lm_ptr,
        d_ptr,
        dmin_ptr,
        n_super,
        RMIN: tl.constexpr,
        RDELTA: tl.constexpr,
        NSTEP: tl.constexpr,
        NMAX: tl.constexpr,
        L2MAX: tl.constexpr,
        PACK: tl.constexpr,
        METHOD: tl.constexpr,
    ):
        """Encode one Q4_K super-block (8 sub-blocks x 32 weights) per program.

        ``METHOD == 0`` runs the ``make_qkx2_quants`` grid search; ``METHOD == 1`` runs the
        naive asymmetric min-max fit. Both feed the shared level-2 quant below.
        """

        pid = tl.program_id(0)
        if pid >= n_super:
            return

        sub = tl.arange(0, 8)[:, None]  # [8,1]
        lane = tl.arange(0, 32)[None, :]  # [1,32]
        off = pid * 256 + sub * 32 + lane  # [8,32]
        x = tl.load(x_ptr + off).to(tl.float32)

        xmax = tl.max(x, axis=1)[:, None]  # [8,1]
        vmin = tl.minimum(tl.min(x, axis=1)[:, None], 0.0)  # min(min, 0)
        degenerate = xmax == vmin

        if METHOD == 0:  # grid_search
            # make_qkx2_quants importance weights: av_x + |x| per sub-block.
            av_x = tl.sqrt(tl.sum(x * x, axis=1) / 32.0)[:, None]  # [8,1]
            w = av_x + tl.abs(x)
            denom = tl.where(degenerate, 1.0, xmax - vmin)
            iscale = NMAX / denom
            scale = 1.0 / iscale
            codes = tl.minimum(tl.maximum(libdevice.rint(iscale * (x - vmin)), 0.0), float(NMAX))
            diff = scale * codes + vmin - x
            best_error = tl.sum(w * diff * diff, axis=1)[:, None]

            sum_w = tl.sum(w, axis=1)[:, None]
            sum_x = tl.sum(w * x, axis=1)[:, None]
            cur_min = vmin

            for step in range(NSTEP + 1):
                trial_iscale = (RMIN + RDELTA * step + NMAX) / denom
                aux = tl.minimum(tl.maximum(libdevice.rint(trial_iscale * (x - vmin)), 0.0), float(NMAX))
                sum_l = tl.sum(w * aux, axis=1)[:, None]
                sum_l2 = tl.sum(w * aux * aux, axis=1)[:, None]
                sum_xl = tl.sum(w * aux * x, axis=1)[:, None]
                det = sum_w * sum_l2 - sum_l * sum_l
                safe_det = tl.where(det > 0, det, 1.0)
                this_scale = (sum_w * sum_xl - sum_x * sum_l) / safe_det
                this_min = (sum_l2 * sum_x - sum_l * sum_xl) / safe_det
                positive_min = this_min > 0
                safe_l2 = tl.where(sum_l2 > 0, sum_l2, 1.0)
                this_scale = tl.where(positive_min, sum_xl / safe_l2, this_scale)
                this_min = tl.where(positive_min, 0.0, this_min)
                tdiff = this_scale * aux + this_min - x
                cur_error = tl.sum(w * tdiff * tdiff, axis=1)[:, None]
                better = (det > 0) & (cur_error < best_error)
                best_error = tl.where(better, cur_error, best_error)
                scale = tl.where(better, this_scale, scale)
                cur_min = tl.where(better, this_min, cur_min)

            scale = tl.where(degenerate, 0.0, scale)  # [8,1], >= 0
            the_min = -tl.where(degenerate, vmin, cur_min)  # [8,1], >= 0 (subtracted offset)
        else:  # minmax: naive asymmetric affine fit (no search)
            scale = (xmax - vmin) / float(NMAX)  # [8,1], 0 when degenerate
            the_min = -vmin  # [8,1], >= 0

        # Level-2: quantize the 8 sub-block scales/mins to 6-bit against fp16 d/dmin.
        max_scale = tl.max(scale, axis=0)  # [1]
        max_min = tl.max(the_min, axis=0)  # [1]
        inv_scale = tl.where(max_scale > 0, float(L2MAX) / max_scale, 0.0)
        inv_min = tl.where(max_min > 0, float(L2MAX) / max_min, 0.0)
        ls = tl.minimum(tl.maximum(libdevice.rint(inv_scale * scale), 0.0), float(L2MAX))  # [8,1]
        lm = tl.minimum(tl.maximum(libdevice.rint(inv_min * the_min), 0.0), float(L2MAX))

        d = (max_scale / float(L2MAX)).to(tl.float16).to(tl.float32)  # [1], fp16-rounded
        dmin = (max_min / float(L2MAX)).to(tl.float16).to(tl.float32)
        deff = d * ls  # [8,1]
        dm = dmin * lm  # [8,1]

        safe = deff != 0
        raw = tl.where(safe, (x + dm) / tl.where(safe, deff, 1.0), 0.0)
        q = tl.where(safe, tl.minimum(tl.maximum(libdevice.rint(raw), 0.0), float(NMAX)), 0.0)
        recon = deff * q - dm
        in_range = (raw >= 0.0) & (raw <= float(NMAX))

        tl.store(recon_ptr + off, recon)
        tl.store(inrange_ptr + off, in_range.to(tl.int8))
        if PACK:
            tl.store(q_ptr + off, q)
            sb = pid * 8 + tl.arange(0, 8)[:, None]
            tl.store(ls_ptr + sb, ls.to(tl.int32))
            tl.store(lm_ptr + sb, lm.to(tl.int32))
            one = tl.arange(0, 1)
            tl.store(d_ptr + pid + one, d)
            tl.store(dmin_ptr + pid + one, dmin)


_METHOD_CODES = {"grid_search": 0, "minmax": 1}


def triton_encode_superblocks(sub: torch.Tensor, *, pack: bool, method: str = "grid_search"):
    """Encode ``sub`` of shape ``[n_super, 8, 32]`` (float32, CUDA) with the fused kernel.

    ``method`` selects the level-1 fit (``grid_search`` or naive ``minmax``). Returns
    ``(reconstruction, in_range, q, ls, lm, d, dmin)`` where the code tensors are only
    materialized when ``pack`` is True (they are placeholders otherwise, to keep the training
    forward from writing the extra ~1x output bandwidth).
    """

    from .q4_k import Q4K_L2MAX, Q4K_NMAX, Q4K_NSUB, _NSTEP, _RDELTA, _RMIN

    if not HAS_TRITON:  # pragma: no cover - guarded by caller
        raise RuntimeError("Triton is not available.")
    if method not in _METHOD_CODES:
        raise ValueError(f"Unknown Q4_K method {method!r}; supported: {tuple(_METHOD_CODES)}.")
    n_super = sub.shape[0]
    x = sub.reshape(-1).contiguous()
    recon = torch.empty_like(x)
    in_range = torch.empty(x.numel(), dtype=torch.int8, device=x.device)

    if pack:
        q = torch.empty_like(x)
        ls = torch.empty(n_super, Q4K_NSUB, dtype=torch.int32, device=x.device)
        lm = torch.empty(n_super, Q4K_NSUB, dtype=torch.int32, device=x.device)
        d = torch.empty(n_super, dtype=torch.float32, device=x.device)
        dmin = torch.empty(n_super, dtype=torch.float32, device=x.device)
    else:
        placeholder = torch.empty(0, dtype=torch.float32, device=x.device)
        q = ls = lm = d = dmin = placeholder

    _q4k_encode_kernel[(n_super,)](
        x,
        recon,
        in_range,
        q,
        ls,
        lm,
        d,
        dmin,
        n_super,
        RMIN=float(_RMIN),
        RDELTA=float(_RDELTA),
        NSTEP=int(_NSTEP),
        NMAX=int(Q4K_NMAX),
        L2MAX=int(Q4K_L2MAX),
        PACK=bool(pack),
        METHOD=_METHOD_CODES[method],
        num_warps=1,
    )

    recon = recon.view(n_super, Q4K_NSUB, 32)
    in_range = in_range.view(n_super, Q4K_NSUB, 32).bool()
    if pack:
        q = q.view(n_super, Q4K_NSUB, 32)
        ls = ls.to(torch.int64)
        lm = lm.to(torch.int64)
        d = d.view(n_super, 1)
        dmin = dmin.view(n_super, 1)
    return recon, in_range, q, ls, lm, d, dmin
