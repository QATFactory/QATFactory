"""Per-format fake-quant plugins and a small registry.

Each quantization format exposes a :class:`QuantFormat` descriptor so the model
instrumentation (``apply_fake_quant``), the trainer, and the entry-point scripts can be
written once and dispatch by ``quant_format`` name. NVFP4 is the fused-runtime-scale format
(shared per-family tensor scales + recorded activation scales); Q4_K is a self-contained
two-level GGUF format with no shared scales and no activation statistics; MXFP4 is a
self-contained block-32 E8M0 microscaling format, also with no shared scales and no
activation statistics.
"""

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class QuantFormat:
    """Describes one fake-quant format and its optional lifecycle hooks."""

    name: str
    linear_cls: type[nn.Module]
    from_linear: Callable[[nn.Linear], nn.Module]
    quantizer_id: str  # recorded in the qatfactory_qad checkpoint metadata
    supports_fused_runtime_scales: bool
    # LoRA-QAD support: the format merges the frozen base weight with a trained low-rank
    # adapter before fake-quantization (``LoRAMixin``). Formats that pack the deployable
    # blocks at save time from the raw latent weight (Q4_K) don't yet participate.
    supports_lora: bool = True
    # Input-activation fake-quant (W4A4 training) support.
    supports_activation_quant: bool = False
    # True when the format's W4A4 quantizes activations with a static global scale that
    # rides on the running-max infrastructure (NVFP4); formats with fully dynamic
    # activation quantization (MXFP4) need no runtime scales for W4A4.
    activation_quant_requires_runtime_scales: bool = False
    # Called by the trainer once per optimizer step (before its first forward), as
    # ``commit_input_scales(model, process_group=...)``, to publish the running input
    # maxima into the step-static W4A4 scales — all-reduced (MAX) across ranks so every
    # rank quantizes with the same global scale. None for formats whose activation
    # quantization holds no state.
    commit_input_scales: Callable | None = None
    # Pre-quantized block-FP8 checkpoints: wrap the FP8 codes as the frozen LoRA base rather
    # than requiring a full-precision copy of the backbone. None for formats that have no
    # FP8-aware linear.
    from_fp8_linear: Callable[[nn.Module], nn.Module] | None = None
    # Packed-3D MoE experts support: ``from_experts(module, rank, alpha, ...)`` returns the
    # LoRA-QAD replacement, ``from_experts_full(module)`` the full-parameter one, and
    # ``is_fused_experts`` recognizes convertible modules.
    experts_cls: type[nn.Module] | None = None
    from_experts: Callable[..., nn.Module] | None = None
    from_experts_full: Callable[[nn.Module], nn.Module] | None = None
    is_fused_experts: Callable[[nn.Module], bool] = lambda module: False
    # Fused-runtime-scale lifecycle (populated only for formats that share family scales).
    attach_amax_groups: Callable | None = None
    refresh_weight_amax: Callable | None = None
    finalize_input_scales: Callable | None = None
    runtime_state_dict: Callable | None = None
    load_runtime_state_dict: Callable | None = None
    # Save-time packing: for formats whose deployable low-bit blocks are computed once at
    # checkpoint save (on the training device) and carried through export unchanged, this
    # returns the packed byte tensor for a single 2D target weight, given ``(weight, module)``
    # (the module may carry architecture reorder permutations). ``None`` for formats that
    # (re)quantize at export time instead (e.g. NVFP4).
    pack_weight: Callable | None = None
    packed_weight_filename: str | None = None


def _nvfp4_format() -> QuantFormat:
    from .nvfp4 import (
        NVFP4Linear,
        attach_nvfp4_amax_groups,
        commit_nvfp4_input_scales,
        finalize_nvfp4_input_scales,
        load_nvfp4_runtime_state_dict,
        nvfp4_runtime_state_dict,
        refresh_nvfp4_weight_amax,
    )
    from .nvfp4_experts import NVFP4ExpertLoRA, NVFP4Experts, is_fused_experts

    return QuantFormat(
        name="nvfp4",
        linear_cls=NVFP4Linear,
        from_linear=NVFP4Linear.from_linear,
        quantizer_id="standard_max",
        supports_fused_runtime_scales=True,
        supports_activation_quant=True,
        activation_quant_requires_runtime_scales=True,
        commit_input_scales=commit_nvfp4_input_scales,
        from_fp8_linear=NVFP4Linear.from_fp8_linear,
        experts_cls=NVFP4ExpertLoRA,
        from_experts=NVFP4ExpertLoRA,
        from_experts_full=NVFP4Experts.from_experts,
        is_fused_experts=is_fused_experts,
        attach_amax_groups=attach_nvfp4_amax_groups,
        refresh_weight_amax=refresh_nvfp4_weight_amax,
        finalize_input_scales=finalize_nvfp4_input_scales,
        runtime_state_dict=nvfp4_runtime_state_dict,
        load_runtime_state_dict=load_nvfp4_runtime_state_dict,
    )


def _make_q4_k_pack_weight(default_method: str):
    def _q4_k_pack_weight(weight, module=None):
        from .q4_k import pack_deploy_weight

        row_perm = getattr(module, "q4k_row_perm", None) if module is not None else None
        col_perm = getattr(module, "q4k_col_perm", None) if module is not None else None
        method = getattr(module, "quant_method", default_method) if module is not None else default_method
        return pack_deploy_weight(weight, row_perm=row_perm, col_perm=col_perm, method=method)

    return _q4_k_pack_weight


def _q4_k_variant(name: str, quantizer_id: str, method: str) -> QuantFormat:
    """Build a Q4_K format descriptor for a given level-1 encoder ``method``.

    Both variants emit the identical ``block_q4_K`` bytes (same decoder, same GGUF export);
    they differ only in how each sub-block's affine fit is found: ``grid_search`` runs
    llama.cpp's weighted-LS ``make_qkx2_quants`` search, ``minmax`` uses the naive asymmetric
    min-max fit (llama.cpp's seed, no search).
    """

    from functools import partial

    from .q4_k import Q4KLinear

    return QuantFormat(
        name=name,
        linear_cls=Q4KLinear,
        from_linear=partial(Q4KLinear.from_linear, method=method),
        quantizer_id=quantizer_id,
        supports_fused_runtime_scales=False,
        # Q4_K packs its deploy blocks from the raw latent weight at save time; folding a
        # LoRA adapter into that path is not wired up yet, so reject Q4_K + LoRA explicitly.
        supports_lora=False,
        pack_weight=_make_q4_k_pack_weight(method),
        packed_weight_filename="q4k_weight.safetensors",
    )


def _mxfp4_format() -> QuantFormat:
    """MXFP4: E2M1 elements, EVEN-rule E8M0 block-32 weight scales, no global scales.

    Like Q4_K it has no fused-scale lifecycle; unlike Q4_K the encoder is closed-form and
    cheap, so weights are re-quantized at export time instead of packed at save time.
    W4A4 training is supported with fully dynamic (ceil-rule) activation fake-quant —
    stateless, so no runtime scales, no commit hook, and no export/checkpoint changes.
    """

    from .mxfp4 import MXFP4Linear

    return QuantFormat(
        name="mxfp4",
        linear_cls=MXFP4Linear,
        from_linear=MXFP4Linear.from_linear,
        quantizer_id="mx_even",
        supports_fused_runtime_scales=False,
        supports_activation_quant=True,
    )


def _q4_k_format() -> QuantFormat:
    return _q4_k_variant("q4_k", quantizer_id="q4_k_minmax", method="minmax")


def _q4_k_grid_search_format() -> QuantFormat:
    return _q4_k_variant("q4_k_grid_search", quantizer_id="q4_k_grid_search", method="grid_search")


_FORMAT_BUILDERS = {
    "mxfp4": _mxfp4_format,
    "nvfp4": _nvfp4_format,
    "q4_k": _q4_k_format,
    "q4_k_grid_search": _q4_k_grid_search_format,
}


def get_quant_format(name: str) -> QuantFormat:
    """Return the :class:`QuantFormat` descriptor for ``name`` (e.g. ``"nvfp4"``, ``"q4_k"``)."""

    try:
        builder = _FORMAT_BUILDERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown quant_format {name!r}; supported formats: {sorted(_FORMAT_BUILDERS)}."
        ) from exc
    return builder()


def available_quant_formats() -> list[str]:
    return sorted(_FORMAT_BUILDERS)
