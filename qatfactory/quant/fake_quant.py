"""Inject fake-quant modules into a stock Hugging Face model and freeze the rest.

This is the "patch, not fork" utility: walk the module tree, swap selected
``nn.Linear`` leaves for the fake-quant variant while preserving parameter FQNs, then
train only the quantized latent weights. It is model-agnostic (config-driven suffix
matching) and format-generic (``quant_format`` registry dispatch), unlike a
per-architecture rewrite.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from .formats import get_quant_format
from .formats.fp8_base import dequantize_fp8_blocks, fp8_linear_state


def is_fp8_linear(module: nn.Module) -> bool:
    return fp8_linear_state(module) is not None


def dequantize_remaining_fp8_linears(model: nn.Module) -> list[str]:
    """Replace every still-FP8 linear with a frozen BF16 copy of the same weight.

    A pre-quantized checkpoint leaves its non-target linears as the library's FP8 modules,
    which call vendor FP8 matmul kernels. Those kernels are not universally available — on
    SM100 with a CUDA runtime below 12.9 they silently fall back to a path that destroys the
    model's output — and a broken forward is invisible here, because the shared-base teacher
    is computed through the very same layers: teacher and student degrade together, the
    distillation loss stays near zero, and training has nothing to learn from.

    Dequantizing to BF16 removes the dependency entirely and is what the frozen base is
    conceptually anyway: full precision. It costs one BF16 copy of the *unquantized* linears
    only, which for a MoE backbone is a rounding error next to the routed experts.
    """

    dequantized = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            state = fp8_linear_state(child)
            if state is None:
                continue
            weight, scale_inv = state
            replacement = nn.Linear(
                child.in_features, child.out_features, bias=child.bias is not None, device="meta"
            )
            replacement.weight = nn.Parameter(
                dequantize_fp8_blocks(weight, scale_inv, torch.bfloat16), requires_grad=False
            )
            replacement.bias = child.bias
            setattr(parent, child_name, replacement)
            dequantized.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    return dequantized


def _mark_quantized_base_as_qat_trainable(model: nn.Module) -> None:
    """Record that a pre-quantized model now carries trainable adapters over a frozen base.

    ``Trainer`` refuses to train a model that was loaded through a quantizer unless it
    advertises either PEFT adapters or QAT support, because the usual failure it guards
    against is someone trying to optimize packed low-bit weights directly. That is not what
    happens here: the quantized linears have been replaced by fake-quant modules whose frozen
    base is the loaded low-bit weight and whose only trainable parameters are the LoRA
    adapters — the QAT case the flags describe, and one this function is only reached after
    :func:`apply_fake_quant` has verified.

    Both flags are read-only properties, so they are overridden on a per-model subclass
    rather than on the shared quantizer class.
    """

    quantizer = getattr(model, "hf_quantizer", None)
    if quantizer is None or (quantizer.is_qat_trainable and quantizer.is_trainable):
        return
    quantizer.__class__ = type(
        f"QATTrainable{type(quantizer).__name__}",
        (type(quantizer),),
        {
            "is_qat_trainable": property(lambda self: True),
            "is_trainable": property(lambda self: True),
        },
    )


logger = logging.getLogger(__name__)

# Modules that stay in high precision at inference and must not be quantized. Vision towers
# are spelled several ways across multimodal checkpoints, and their attention leaves reuse the
# text-side names (`q_proj`, `k_proj`, ...), so a missing prefix here silently quantizes and
# trains the image encoder instead of leaving it frozen.
DEFAULT_EXCLUDE_PREFIXES = (
    "model.visual",
    "visual",
    "model.vision_tower",
    "vision_tower",
    "model.vision_adapter",
    "model.vision_projection",
    "mtp",
    "lm_head",
)


@dataclass(frozen=True)
class ConvertedModules:
    """Module FQNs converted to fake-quant variants, by kind.

    ``linears`` holds real ``nn.Linear`` FQNs. ``experts`` holds the packed-3D routed-expert
    module FQNs, and ``expert_targets`` their logical per-expert projections — one deployed
    matrix each, which is the granularity the exporter and the QAD metadata speak in.
    Iterating yields every deployed target name.
    """

    linears: tuple[str, ...]
    experts: tuple[str, ...] = ()
    expert_targets: tuple[str, ...] = ()
    # LoRA-QAD: the base latent weights stay frozen and only the injected low-rank
    # adapters (``lora_A``/``lora_B``) train.
    lora: bool = False

    @property
    def all(self) -> list[str]:
        return sorted((*self.linears, *self.expert_targets))

    @property
    def trainable_parameters(self) -> set[str]:
        if self.lora:
            from .formats.nvfp4_experts import LORA_LEAVES

            parameters: set[str] = set()
            for name in self.linears:
                parameters.update((f"{name}.lora_A", f"{name}.lora_B"))
            for name in self.experts:
                parameters.update(f"{name}.{leaf}" for leaf in LORA_LEAVES)
            return parameters
        parameters = {f"{name}.weight" for name in self.linears}
        for name in self.experts:
            parameters.update((f"{name}.gate_up_proj", f"{name}.down_proj"))
        return parameters

    def __iter__(self):
        return iter(self.all)

    def __len__(self) -> int:
        return len(self.all)


# MoE shared-expert projections live under one of these path segments (Qwen3.5-MoE uses the
# singular, DeepSeek-style models the plural).
_SHARED_EXPERT_SEGMENTS = frozenset({"shared_expert", "shared_experts"})


def _is_excluded(module_name: str, exclude_prefixes: tuple[str, ...]) -> bool:
    return any(module_name == prefix or module_name.startswith(prefix + ".") for prefix in exclude_prefixes)


def apply_fake_quant(
    model: nn.Module,
    target_modules: Iterable[str],
    *,
    quant_format: str = "nvfp4",
    exclude_prefixes: Iterable[str] = DEFAULT_EXCLUDE_PREFIXES,
    fused_runtime_scales: bool = False,
    quantize_activations: bool = False,
    lora_rank: int = 0,
    lora_alpha: float = 0.0,
    lora_moe_hybrid_shared: bool = True,
    expert_checkpointing: bool = True,
    quantize_shared_expert: bool = True,
) -> ConvertedModules:
    """Replace target ``nn.Linear`` leaves with the format's fake-quant variant and freeze the rest.

    Args:
        model: a loaded Hugging Face model (mutated in place).
        target_modules: leaf-name suffixes to quantize (e.g. ``q_proj``, ``gate_proj``).
        quant_format: registered format name (``"nvfp4"``, ``"mxfp4"`` or ``"q4_k"``).
        exclude_prefixes: module-name prefixes to skip entirely (vision, MTP, lm_head).
        fused_runtime_scales: when True, attach fused-family amax groups so QKV / gate-up /
            delta-net input projections share one max-based tensor scale, matching the vLLM
            fused runtime. Required for an exportable NVFP4 checkpoint; unsupported for
            formats without shared runtime scales (Q4_K, MXFP4).
        quantize_activations: when True (W4A4), also fake-quantize the input activation of
            every quantized linear, matching deployed W4A4 numerics instead of weight-only
            W4A16. Requires a format with activation-quant support; formats whose W4A4
            uses a static global scale (NVFP4) additionally require
            ``fused_runtime_scales`` for the running-max tracking, while fully dynamic
            formats (MXFP4) need no scale state at all.
        lora_rank: when > 0, attach a rank-``lora_rank`` LoRA adapter to every quantized
            linear and train only those adapters — the base latent weights stay frozen and
            the student quantizes the merged weight ``W + (lora_alpha/lora_rank) * B @ A``.
            Because the base is frozen it can be shared as the full-precision teacher.
        lora_alpha: LoRA scaling numerator (scaling = ``lora_alpha / lora_rank``).
        lora_moe_hybrid_shared: for fused routed experts, share the gate/up ``A`` factor and
            the down ``B`` factor across experts. False gives every expert independent
            factors (far more trainable parameters, no cross-expert coupling).
        expert_checkpointing: recompute each routed expert's merged + fake-quantized weight
            in its own backward instead of keeping every expert's alive (LoRA-QAD only).
        quantize_shared_expert: when False, MoE shared-expert projections stay in high
            precision (their exclusion must then be mirrored in the export ignore list).

    MoE models: fused routed experts are converted automatically. With LoRA they become
    :class:`NVFP4ExpertLoRA` (frozen, possibly FP8, base shared with the teacher); without
    LoRA they become :class:`NVFP4Experts`, whose 3D latents are trained in full and which
    needs a separate teacher model.

    Returns:
        :class:`ConvertedModules` with the converted module FQNs (iterable of names).
    """

    quant = get_quant_format(quant_format)
    target_modules = set(target_modules)
    exclude_prefixes = tuple(exclude_prefixes)
    if fused_runtime_scales and not quant.supports_fused_runtime_scales:
        raise ValueError(f"quant_format={quant_format!r} does not support fused_runtime_scales.")
    if quantize_activations and not quant.supports_activation_quant:
        raise ValueError(f"quant_format={quant_format!r} does not support input-activation fake-quant (W4A4).")
    if quantize_activations and quant.activation_quant_requires_runtime_scales and not fused_runtime_scales:
        raise ValueError(
            f"quant_format={quant_format!r} W4A4 requires fused_runtime_scales=True to track the "
            "activation running-max used as the static global scale."
        )
    lora_enabled = lora_rank and lora_rank > 0
    if lora_enabled and not quant.supports_lora:
        raise ValueError(f"quant_format={quant_format!r} does not support LoRA-QAD.")

    text_config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    is_moe = bool(getattr(text_config, "num_experts", None))
    if not target_modules and not is_moe:
        raise ValueError("apply_fake_quant requires a non-empty target_modules set.")
    if is_moe and (quant.from_experts if lora_enabled else quant.from_experts_full) is None:
        raise ValueError(
            f"quant_format={quant_format!r} does not support "
            f"{'LoRA' if lora_enabled else 'full-parameter'} QAD of fused routed experts."
        )
    tied_embeddings = bool(
        getattr(model.config, "tie_word_embeddings", False) or getattr(text_config, "tie_word_embeddings", False)
    )
    if tied_embeddings:
        # Tied input embeddings / lm_head are safe for QAD as long as the shared tensor is
        # never quantized or trained. It is excluded from quantization (lm_head is in
        # DEFAULT_EXCLUDE_PREFIXES) and is not a quantization target, so it stays frozen in
        # full precision and the tie is preserved on save (weight-only and W4A4 both only
        # touch the quantized-linear latent weights). Reject only the unsafe case where the
        # shared tensor would itself be a quantization target.
        if target_modules & {"embed_tokens", "lm_head"}:
            raise ValueError(
                "Tied embeddings cannot be a quantization target: remove embed_tokens/lm_head "
                "from target_modules (the shared tensor must stay frozen and unquantized)."
            )
        logger.info("Tied input embeddings / lm_head detected: kept frozen and unquantized (QAD-safe).")

    model.requires_grad_(False)
    converted_names = []
    converted_experts = []
    expert_targets = []
    skipped_shared_expert = 0
    for parent_name, parent in list(model.named_modules()):
        if _is_excluded(parent_name, exclude_prefixes):
            continue
        shared_expert_path = not _SHARED_EXPERT_SEGMENTS.isdisjoint(parent_name.split("."))
        for child_name, child in list(parent.named_children()):
            module_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if is_moe and quant.is_fused_experts(child):
                if lora_enabled:
                    converted = quant.from_experts(
                        child,
                        int(lora_rank),
                        float(lora_alpha),
                        hybrid_shared=lora_moe_hybrid_shared,
                        expert_checkpointing=expert_checkpointing,
                    )
                else:
                    converted = quant.from_experts_full(child)
                setattr(parent, child_name, converted)
                converted_experts.append(module_name)
                expert_targets.extend(converted.target_names(module_name))
                continue
            if child_name not in target_modules or not isinstance(child, nn.Linear):
                continue
            if shared_expert_path and not quantize_shared_expert:
                skipped_shared_expert += 1
                continue
            # A pre-quantized checkpoint hands us block-FP8 linears; keep the FP8 codes as
            # the frozen base instead of forcing a full-precision copy of the backbone.
            replacement = (
                quant.from_fp8_linear(child)
                if quant.from_fp8_linear is not None and is_fp8_linear(child)
                else quant.from_linear(child)
            )
            setattr(parent, child_name, replacement)
            converted_names.append(module_name)

    if not converted_names and not converted_experts:
        raise ValueError(f"No modules matched target suffixes {sorted(target_modules)}.")
    if is_moe and not converted_experts:
        raise ValueError(
            "MoE model (num_experts set) but no fused routed-expert modules were found to convert; "
            "only packed 3D gate_up_proj/down_proj experts (transformers>=5) are supported."
        )
    if not quantize_shared_expert and not skipped_shared_expert:
        raise ValueError(
            "quantize_shared_expert=False but no shared-expert Linear modules were found to skip; "
            "check the model's shared-expert module naming."
        )
    if target_modules and not converted_names:
        raise ValueError(f"No linear modules matched target suffixes {sorted(target_modules)}.")

    if lora_enabled:
        for module in model.modules():
            if isinstance(module, quant.linear_cls):
                module.init_lora(int(lora_rank), float(lora_alpha))

    for module in model.modules():
        if isinstance(module, quant.linear_cls):
            if lora_enabled:
                # Freeze the base latent weight; train only the low-rank adapter. Keeping the
                # base frozen is what lets the teacher reuse these same weights.
                module.weight.requires_grad_(False)
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)
            else:
                module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(False)
        elif not lora_enabled and getattr(module, "_is_nvfp4_experts", False):
            module.gate_up_proj.requires_grad_(True)
            module.down_proj.requires_grad_(True)
    if fused_runtime_scales:
        quant.attach_amax_groups(model)
        for module in model.modules():
            if getattr(module, "_is_nvfp4_experts", False):
                module._track_input_amax = True
    if quantize_activations:
        for module in model.modules():
            if isinstance(module, quant.linear_cls) or getattr(module, "_is_nvfp4_experts", False):
                module._quantize_activations = True

    converted = ConvertedModules(
        linears=tuple(sorted(converted_names)),
        experts=tuple(sorted(converted_experts)),
        expert_targets=tuple(sorted(expert_targets)),
        lora=bool(lora_enabled),
    )
    # Enforce the QAD invariant: only the intended latent parameters train (the quantized
    # weights, or — under LoRA — only the injected adapters); every other module
    # (embeddings, lm_head, norms, biases, conv, gates, routers, and the frozen base under
    # LoRA) stays frozen.
    actual_trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    unexpected = actual_trainable - converted.trainable_parameters
    missing = converted.trainable_parameters - actual_trainable
    if missing:
        raise RuntimeError(
            f"Expected trainable parameters (quantized latents or LoRA adapters) are frozen: {sorted(missing)[:8]}."
        )
    if unexpected:
        raise RuntimeError(
            f"Non-quantized parameters would be trained during distillation: {sorted(unexpected)[:8]}. "
            "By default only quantized-linear latent weights (or LoRA adapters) are trainable."
        )
    leftover_fp8 = dequantize_remaining_fp8_linears(model)
    if leftover_fp8:
        logger.info(
            "Dequantized %d non-target FP8 linears to BF16 (e.g. %s) so no vendor FP8 kernel "
            "runs in the forward.",
            len(leftover_fp8),
            leftover_fp8[0],
        )
    if lora_enabled:
        _mark_quantized_base_as_qat_trainable(model)
    frozen = sum(1 for _, parameter in model.named_parameters() if not parameter.requires_grad)
    logger.info(
        "Enabled %s %s QAD%s: %d quantized linears, %d fused expert modules, %d frozen parameters.",
        quant_format,
        "W4A4 (weight + input-activation)" if quantize_activations else "weight-only",
        f" + LoRA(rank={int(lora_rank)}, alpha={float(lora_alpha):g})" if lora_enabled else "",
        len(converted.linears),
        len(converted.experts),
        frozen,
    )
    return converted
