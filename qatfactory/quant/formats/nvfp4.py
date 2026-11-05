"""NVFP4 weight-only fake quantization with a masked straight-through estimator.

The numerics here must stay bit-exact with the deployment (vLLM compressed-tensors
``nvfp4-pack-quantized``) kernel: E2M1 values, groups of 16 along the input dimension,
FP8 E4M3 block scales, and an FP32 tensor scale, with deterministic
round-to-nearest-even ties. Do not "clean up" the rounding/scale math without a
matching bit-exactness test.
"""

import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .fp8_base import dequantize_fp8_blocks, fp8_linear_state
from .lora import LoRAMixin, shared_base_teacher_enabled

logger = logging.getLogger(__name__)


NVFP4_BLOCK_SIZE = 16
NVFP4_MAX_VALUE = 6.0
E4M3_MAX_VALUE = 448.0
E4M3_MIN_SUBNORMAL = 2**-9
NVFP4_SCALE_DENOMINATOR = NVFP4_MAX_VALUE * E4M3_MAX_VALUE


class _MaskedStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, latent: torch.Tensor, quantized: torch.Tensor, in_range: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(in_range)
        return quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (in_range,) = ctx.saved_tensors
        return grad_output * in_range, None, None


def _round_to_e2m1(values: torch.Tensor) -> torch.Tensor:
    codebook = values.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    boundaries = values.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    magnitude = values.abs()
    indices = torch.bucketize(magnitude, boundaries, right=False)
    boundary_indices = indices.clamp_max(boundaries.numel() - 1)
    ties = (indices < codebook.numel() - 1) & (magnitude == boundaries[boundary_indices])
    indices = indices + (ties & (indices % 2 == 1)).to(indices.dtype)
    return codebook[indices] * values.sign()


def _cast_e4m3(values: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("NVFP4 fake quantization requires torch.float8_e4m3fn.")
    return values.clamp(min=0.0, max=E4M3_MAX_VALUE).to(torch.float8_e4m3fn).float()


def _cast_e4m3_storage(values: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("NVFP4 quantization requires torch.float8_e4m3fn.")
    return values.clamp(min=0.0, max=E4M3_MAX_VALUE).to(torch.float8_e4m3fn)


def _e2m1_values_to_indices(values: torch.Tensor) -> torch.Tensor:
    """Encode exact E2M1 values as canonical sign-magnitude nibbles."""

    levels = values.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    magnitude_indices = torch.searchsorted(levels, values.abs()).to(torch.uint8)
    negative = (values < 0) & (magnitude_indices != 0)
    return (negative.to(torch.uint8) << 3) | magnitude_indices


def pack_e2m1_rows(indices: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 nibbles with even columns in the low nibble."""

    if indices.ndim != 2:
        raise ValueError(f"E2M1 packing expects a 2D tensor, got shape {tuple(indices.shape)}.")
    if indices.shape[-1] % 2:
        raise ValueError("E2M1 packing requires an even final dimension.")
    low = indices[:, 0::2] & 0x0F
    high = indices[:, 1::2] & 0x0F
    return (low | (high << 4)).to(torch.uint8)


def unpack_e2m1_rows(packed: torch.Tensor) -> torch.Tensor:
    """Unpack bytes into canonical E2M1 nibbles."""

    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError(f"Packed E2M1 weights must be 2D uint8, got {packed.dtype} {tuple(packed.shape)}.")
    indices = torch.empty(
        packed.shape[0],
        packed.shape[1] * 2,
        dtype=torch.uint8,
        device=packed.device,
    )
    indices[:, 0::2] = packed & 0x0F
    indices[:, 1::2] = (packed >> 4) & 0x0F
    return indices


def dequantize_nvfp4(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    """Dequantize packed NVFP4 tensors to FP32."""

    indices = unpack_e2m1_rows(packed_weight)
    levels = weight_scale.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    values = levels[(indices & 0x07).long()]
    values = torch.where((indices & 0x08) != 0, -values, values)
    if values.shape[-1] != weight_scale.shape[-1] * NVFP4_BLOCK_SIZE:
        raise ValueError(
            f"NVFP4 scale shape {tuple(weight_scale.shape)} is incompatible with "
            f"weight shape {tuple(packed_weight.shape)}."
        )
    effective_scale = weight_scale.float() * weight_scale_2.float()
    return values * effective_scale.repeat_interleave(NVFP4_BLOCK_SIZE, dim=-1)


@dataclass(frozen=True)
class NVFP4QuantizationResult:
    packed_weight: torch.Tensor
    weight_scale: torch.Tensor
    weight_scale_2: torch.Tensor
    # None when packed with ``compute_reconstruction=False`` (the RL weight-sync path packs
    # every target on every optimizer step and never reads the dequantized weight).
    reconstruction: torch.Tensor | None


@dataclass
class NVFP4AmaxGroup:
    member_names: tuple[str, ...]
    members: tuple["NVFP4Linear", ...]
    tensor_amax_value: torch.Tensor | None = None

    def tensor_amax(self) -> torch.Tensor:
        if self.tensor_amax_value is not None:
            return self.tensor_amax_value
        if any(hasattr(module.weight, "to_local") for module in self.members):
            raise RuntimeError("Fused NVFP4 weight amax must be refreshed after FSDP parallelization.")
        # The scale must track the *merged* (LoRA-folded) weight the forward quantizes.
        maxima = [module._effective_weight().detach().abs().amax().float() for module in self.members]
        return torch.stack(maxima).amax()


def build_nvfp4_fused_family_names(module_names: list[str]) -> list[tuple[str, ...]]:
    """Group projections exactly as the serving runtime fuses them."""

    name_set = set(module_names)
    consumed = set()
    families = []
    for name in module_names:
        if name in consumed:
            continue
        parent, _, leaf = name.rpartition(".")
        # QKV may be a partial set: Gemma-4's `full_attention` global layers reuse the key as
        # the value (`attention_k_eq_v`, one global KV head) and so have no `v_proj`. The serving
        # runtime fuses whichever of q/k/v exist into one packed QKV projection, so group the
        # present subset instead of rejecting it. Every other fused family (gate/up, delta-net
        # in_proj) must still be complete.
        allow_partial = False
        if leaf in {"q_proj", "k_proj", "v_proj"} and parent.endswith("self_attn"):
            candidates = tuple(f"{parent}.{sibling}" for sibling in ("q_proj", "k_proj", "v_proj"))
            allow_partial = True
        elif leaf in {"gate_proj", "up_proj"} and (parent.endswith("mlp") or parent.endswith("shared_expert")):
            candidates = tuple(f"{parent}.{sibling}" for sibling in ("gate_proj", "up_proj"))
        elif leaf in {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"} and parent.endswith("linear_attn"):
            candidates = tuple(
                f"{parent}.{sibling}" for sibling in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
            )
        else:
            candidates = (name,)
        present = tuple(candidate for candidate in candidates if candidate in name_set)
        if not allow_partial and present and len(present) < len(candidates):
            raise ValueError(f"Incomplete fused NVFP4 family for {name!r}: {present}.")
        family = present if present else (name,)
        families.append(family)
        consumed.update(family)
    if consumed != name_set:
        raise ValueError(f"Failed to assign NVFP4 modules to fused families: {sorted(name_set - consumed)}.")
    return families


def attach_nvfp4_amax_groups(model: nn.Module) -> list[NVFP4AmaxGroup]:
    named_modules = {name: module for name, module in model.named_modules() if isinstance(module, NVFP4Linear)}
    groups = []
    for member_names in build_nvfp4_fused_family_names(list(named_modules)):
        members = tuple(named_modules[name] for name in member_names)
        group = NVFP4AmaxGroup(member_names, members)
        for module in members:
            module._amax_group = group
            module._track_input_amax = True
        groups.append(group)
    model._nvfp4_amax_groups = groups
    return groups


def _nvfp4_amax_groups(model: nn.Module) -> list[NVFP4AmaxGroup]:
    groups = getattr(model, "_nvfp4_amax_groups", None)
    if groups is None:
        raise RuntimeError("NVFP4 fused amax groups have not been attached to the model.")
    return groups


def _local_weight(module: "NVFP4Linear") -> torch.Tensor:
    weight = module.weight
    return weight.to_local() if hasattr(weight, "to_local") else weight


def _merged_local_weight(module: "NVFP4Linear") -> torch.Tensor:
    """Local FSDP shard of the merged weight ``W + scaling * B @ A``.

    The fused-family tensor scale must reflect the weight the forward actually
    quantizes (the LoRA-merged weight), yet the refresh runs between optimizer steps on
    *sharded* parameters. ``lora_B`` is sharded on dim-0 (rows) exactly like ``weight``
    (same ``out_features``, same FSDP2 chunking), so ``B.to_local() @ A_full`` yields the
    delta rows aligned to ``weight.to_local()`` — no full weight is ever materialized.
    ``lora_A`` (``[rank, in]``) is tiny, so gathering it with ``full_tensor`` is cheap. A
    robust fallback reconstructs the full delta and reshards it should the shard rows ever
    fail to align.
    """

    weight = module.weight
    if getattr(module, "_fp8_scale_inv", None) is not None:
        # A block-FP8 base cannot be dequantized from a row shard alone: the shard boundary
        # need not align with the 128-row scale blocks. Gather the (frozen, small) weight
        # once instead of reconstructing the block mapping.
        full = weight.full_tensor() if hasattr(weight, "full_tensor") else weight
        base = dequantize_fp8_blocks(full, module._fp8_scale_inv, module._compute_dtype)
        if getattr(module, "lora_A", None) is None:
            return base
        lora_a = module.lora_A.full_tensor() if hasattr(module.lora_A, "full_tensor") else module.lora_A
        lora_b = module.lora_B.full_tensor() if hasattr(module.lora_B, "full_tensor") else module.lora_B
        return base + (module.lora_scaling * (lora_b @ lora_a)).to(base.dtype)
    if getattr(module, "lora_A", None) is None:
        return _local_weight(module)
    scaling = module.lora_scaling
    if not hasattr(weight, "to_local"):
        return weight + (scaling * (module.lora_B @ module.lora_A)).to(weight.dtype)

    w_local = weight.to_local()
    a_full = module.lora_A.full_tensor() if hasattr(module.lora_A, "full_tensor") else module.lora_A
    b_local = module.lora_B.to_local() if hasattr(module.lora_B, "to_local") else module.lora_B
    if b_local.shape[0] == w_local.shape[0]:
        delta_local = scaling * (b_local @ a_full)
    else:
        from torch.distributed.tensor import distribute_tensor

        b_full = module.lora_B.full_tensor()
        delta_full = scaling * (b_full @ a_full)
        delta_local = distribute_tensor(delta_full, weight.device_mesh, weight.placements).to_local()
    return w_local + delta_local.to(w_local.dtype)


@torch.no_grad()
def refresh_nvfp4_weight_amax(model: nn.Module, process_group=None) -> None:
    """Refresh globally reduced fused-family maxima from local FSDP shards."""

    groups = _nvfp4_amax_groups(model)
    if not groups:
        # An experts-only MoE run has no dense quantized linear to fuse; the routed experts
        # take their tensor scale from the merged weight inside the forward.
        return
    local_maxima = []
    for group in groups:
        member_maxima = [
            _merged_local_weight(module).detach().to(torch.bfloat16).abs().amax().float() for module in group.members
        ]
        local_maxima.append(torch.stack(member_maxima).amax())
    reduced_maxima = torch.stack(local_maxima)
    if dist.is_initialized():
        dist.all_reduce(
            reduced_maxima,
            op=dist.ReduceOp.MAX,
            group=process_group,
        )
    for group, tensor_amax in zip(groups, reduced_maxima.unbind()):
        group.tensor_amax_value = tensor_amax


@torch.no_grad()
def finalize_nvfp4_input_scales(model: nn.Module, process_group=None) -> dict[str, torch.Tensor]:
    """Reduce running training-input maxima and return fused-family input scales."""

    input_scales = {}
    for name, module in model.named_modules():
        if getattr(module, "_is_nvfp4_experts", False):
            input_scales.update(module.finalize_input_scales(name, process_group))

    groups = _nvfp4_amax_groups(model)
    if not groups:
        return input_scales
    local_maxima = []
    for group in groups:
        missing = [name for name, module in zip(group.member_names, group.members) if module._input_amax is None]
        if missing:
            raise RuntimeError(f"Missing training input-amax observations for {missing}.")
        local_maxima.append(torch.stack([module._input_amax for module in group.members]).amax())
    # Reduce where the parameters live: scale state seeded outside a forward (QARL pins it
    # from the rollout export before the model is placed) can still be on CPU, which NCCL
    # cannot reduce.
    reduced_maxima = torch.stack(local_maxima).to(_local_weight(groups[0].members[0]).device)
    if dist.is_initialized():
        dist.all_reduce(
            reduced_maxima,
            op=dist.ReduceOp.MAX,
            group=process_group,
        )
    for group, maximum in zip(groups, reduced_maxima.unbind()):
        if not bool(torch.isfinite(maximum)) or not bool(maximum > 0):
            raise RuntimeError(f"Invalid training input amax for fused family {group.member_names}.")
        for name, module in zip(group.member_names, group.members):
            module._input_amax = maximum.clone()
            input_scales[f"{name}.input_scale"] = (maximum / NVFP4_SCALE_DENOMINATOR).reshape(())
    return input_scales


def nvfp4_runtime_state_dict(model: nn.Module) -> dict:
    modules = {}
    experts = {}
    for name, module in model.named_modules():
        if isinstance(module, NVFP4Linear) and module._track_input_amax and module._input_amax is not None:
            modules[name] = module._input_amax.detach().cpu()
        elif getattr(module, "_is_nvfp4_experts", False):
            state = module.runtime_state_dict()
            if state:
                experts[name] = state
    return {"version": 1, "modules": modules, "experts": experts}


def load_nvfp4_runtime_state_dict(model: nn.Module, state_dict: dict) -> None:
    if state_dict.get("version") != 1:
        raise ValueError("Unsupported NVFP4 QAD runtime-state version.")
    saved_modules = state_dict.get("modules", {})
    expected = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, NVFP4Linear) and module._track_input_amax
    }
    if set(saved_modules) != set(expected):
        missing = sorted(set(expected) - set(saved_modules))
        extra = sorted(set(saved_modules) - set(expected))
        raise RuntimeError(f"NVFP4 runtime-state module mismatch; missing={missing[:5]}, extra={extra[:5]}.")
    for name, module in expected.items():
        module._input_amax = saved_modules[name].to(_local_weight(module).device).reshape(())
        # W4A4 resume: the restored running max is the best available static scale until
        # the first commit of the resumed run (a no-op for weight-only training, where
        # the static is never read).
        module._input_amax_static = module._input_amax.clone()

    saved_experts = state_dict.get("experts", {})
    for name, module in model.named_modules():
        if getattr(module, "_is_nvfp4_experts", False) and name in saved_experts:
            module.load_runtime_state_dict(saved_experts[name], module.gate_up_proj.device)


@torch.no_grad()
def commit_nvfp4_input_scales(model: nn.Module, process_group=None) -> None:
    """Publish the running input maxima as the static W4A4 activation-quant scales.

    Must only be called between optimizer steps (QADTrainer does so at the first
    micro-batch of each new global step): within a step — including the gradient-
    checkpoint recompute during backward — every forward then quantizes with an
    immutable scale, keeping recomputes bit-identical to the original forward.

    The maxima are all-reduced (MAX) across ranks before publishing, so every rank
    quantizes every module with the same static scale — the GLOBAL running max, the
    same statistic the exported ``input_global_scale`` is finalized from. The reduced
    value is also written back into the running max, keeping it rank-consistent.
    """

    for module in model.modules():
        if getattr(module, "_is_nvfp4_experts", False):
            module.commit_input_scales(process_group)

    modules = [
        module
        for module in model.modules()
        if isinstance(module, NVFP4Linear)
        and getattr(module, "_quantize_activations", False)
        and module._input_amax is not None
    ]
    if not modules:
        return
    maxima = torch.stack([module._input_amax for module in modules]).to(_local_weight(modules[0]).device)
    if dist.is_initialized():
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=process_group)
    for module, maximum in zip(modules, maxima.unbind()):
        # clone(): the static must never alias the running max — a later in-place
        # mutation of the running max would otherwise silently shift the static
        # mid-step and reintroduce the recompute-nondeterminism bug.
        module._input_amax = maximum.clone()
        module._input_amax_static = maximum.clone()


@torch.no_grad()
def quantize_nvfp4(
    weight: torch.Tensor,
    *,
    tensor_amax: torch.Tensor | None = None,
    compute_reconstruction: bool = True,
) -> NVFP4QuantizationResult:
    """Pack a 2D weight on the standard max-based NVFP4 lattice.

    ``compute_reconstruction=False`` skips materializing the dequantized weight (a
    full-size tensor callers that only need the packed lattice never read).
    """

    if weight.ndim != 2:
        raise ValueError(f"NVFP4 packing expects a 2D weight, got shape {tuple(weight.shape)}.")
    if weight.shape[-1] % NVFP4_BLOCK_SIZE:
        raise ValueError(
            f"NVFP4 packing requires in_features divisible by {NVFP4_BLOCK_SIZE}, got {weight.shape[-1]}."
        )
    if weight.is_meta:
        raise ValueError("Cannot pack an NVFP4 lattice from a meta tensor.")

    original_dtype = weight.dtype
    blocks = weight.float().reshape(weight.shape[0], -1, NVFP4_BLOCK_SIZE)
    tensor_amax = weight.detach().abs().amax() if tensor_amax is None else tensor_amax.detach()
    if tensor_amax.numel() != 1 or not bool(torch.isfinite(tensor_amax)) or bool(tensor_amax < 0):
        raise ValueError("NVFP4 tensor_amax must be one finite non-negative value.")
    if bool(tensor_amax == 0):
        code_values = torch.zeros_like(weight, dtype=torch.float32)
        weight_scale = torch.full(
            (weight.shape[0], weight.shape[1] // NVFP4_BLOCK_SIZE),
            E4M3_MIN_SUBNORMAL,
            device=weight.device,
        ).to(torch.float8_e4m3fn)
        weight_scale_2 = torch.ones((), dtype=torch.float32, device=weight.device)
        reconstruction = torch.zeros_like(weight) if compute_reconstruction else None
    else:
        weight_scale_2 = (tensor_amax.float() / NVFP4_SCALE_DENOMINATOR).reshape(())
        block_amax = blocks.abs().amax(dim=-1, keepdim=True)
        stored_group_scale = _cast_e4m3_storage(block_amax / (weight_scale_2 * NVFP4_MAX_VALUE)).float()
        stored_group_scale = torch.where(
            stored_group_scale == 0,
            stored_group_scale.new_full((), E4M3_MIN_SUBNORMAL),
            stored_group_scale,
        )
        effective_scale = stored_group_scale * weight_scale_2
        code_values = _round_to_e2m1((blocks / effective_scale).clamp(-NVFP4_MAX_VALUE, NVFP4_MAX_VALUE)).reshape_as(
            weight
        )
        weight_scale = _cast_e4m3_storage(stored_group_scale.squeeze(-1))
        reconstruction = (
            (code_values.reshape_as(blocks) * effective_scale).reshape_as(weight).to(original_dtype)
            if compute_reconstruction
            else None
        )

    code_indices = _e2m1_values_to_indices(code_values)
    return NVFP4QuantizationResult(
        packed_weight=pack_e2m1_rows(code_indices),
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2.to(torch.float32),
        reconstruction=reconstruction,
    )


def _fake_quantize_nvfp4_torch(
    weight: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
    *,
    tensor_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference PyTorch NVFP4 QDQ, returning reconstruction and STE mask."""

    original_dtype = weight.dtype
    latent = weight.float()
    padding = (-latent.shape[-1]) % block_size
    padded = F.pad(latent, (0, padding)) if padding else latent
    blocks = padded.reshape(*padded.shape[:-1], -1, block_size)

    with torch.no_grad():
        tensor_amax = latent.detach().abs().amax() if tensor_amax is None else tensor_amax.detach()
        is_zero = bool(tensor_amax == 0)
        if not is_zero:
            global_scale = tensor_amax / NVFP4_SCALE_DENOMINATOR
            block_amax = blocks.detach().abs().amax(dim=-1, keepdim=True)
            block_scale = _cast_e4m3(block_amax / (global_scale * NVFP4_MAX_VALUE))
            block_scale = torch.where(
                block_scale == 0,
                block_scale.new_full((), E4M3_MIN_SUBNORMAL),
                block_scale,
            )
            dequant_scale = block_scale * global_scale

    if is_zero:
        return torch.zeros_like(weight), torch.ones_like(weight, dtype=torch.bool)

    normalized = (blocks / dequant_scale).clamp(-NVFP4_MAX_VALUE, NVFP4_MAX_VALUE)
    quantized = _round_to_e2m1(normalized) * dequant_scale
    in_range = blocks.detach().abs() <= dequant_scale * NVFP4_MAX_VALUE
    quantized = quantized.reshape_as(padded)[..., : latent.shape[-1]].to(original_dtype)
    in_range = in_range.reshape_as(padded)[..., : latent.shape[-1]]
    return quantized, in_range


def fake_quantize_nvfp4(
    weight: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
    *,
    tensor_amax: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fake-quantize a linear weight with NVFP4 E2M1 values and E4M3 block scales.

    The final dimension is grouped in blocks of 16. Quantization statistics are
    detached, while the masked straight-through estimator sends gradients to
    the high-precision latent weight. CUDA tensors use the fused Triton path
    unless ``QATFACTORY_FP4_DISABLE_TRITON`` is set.
    """

    if block_size != NVFP4_BLOCK_SIZE:
        raise ValueError(f"NVFP4 requires block_size={NVFP4_BLOCK_SIZE}, got {block_size}.")
    if weight.ndim != 2:
        raise ValueError(f"NVFP4Linear expects a 2D weight, got shape {tuple(weight.shape)}.")
    if weight.is_meta:
        return weight

    from .fp4_triton import HAS_TRITON, triton_fake_quantize_nvfp4

    disabled = os.environ.get("QATFACTORY_FP4_DISABLE_TRITON", "") not in ("", "0", "false", "False")
    with torch.no_grad():
        # Max selection is unchanged by the exact native-dtype -> FP32 promotion,
        # and this avoids materializing weight.float() solely for the reduction.
        tensor_amax = weight.detach().abs().amax().float() if tensor_amax is None else tensor_amax.detach().float()
        if weight.is_cuda and HAS_TRITON and not disabled:
            # Deliberately no host-side test for an all-zero tensor: reading `tensor_amax`
            # would synchronize the device on every call, and a 256-expert MoE layer calls
            # this tens of thousands of times per step. The kernel handles that case.
            global_scale = (tensor_amax / NVFP4_SCALE_DENOMINATOR).to(
                device=weight.device, dtype=torch.float32
            ).reshape(())
            quantized, in_range = triton_fake_quantize_nvfp4(weight, global_scale)
        elif bool(tensor_amax == 0):
            # Host-side on CPU only, where reading a scalar costs nothing.
            quantized = torch.zeros_like(weight)
            in_range = torch.ones_like(weight, dtype=torch.bool)
        else:
            quantized, in_range = _fake_quantize_nvfp4_torch(
                weight,
                block_size=block_size,
                tensor_amax=tensor_amax,
            )
    return _MaskedStraightThrough.apply(weight, quantized, in_range)


def fake_quantize_nvfp4_activation(activation: torch.Tensor, input_amax: torch.Tensor) -> torch.Tensor:
    """Fake-quantize an activation tensor with NVFP4, matching vLLM W4A4 serving.

    Serving quantizes the linear input to NVFP4 with a *static* per-tensor global scale
    (the recorded training running-max, passed as ``input_amax``) and *dynamic*
    per-16-group block scales computed from the actual activation — exactly what
    :func:`fake_quantize_nvfp4` does given a fixed ``tensor_amax``. Flattens leading
    dims to the 2D layout the quantizer expects (grouping along the last / feature
    dim). The masked STE keeps gradients flowing to the upstream latent weights;
    values beyond the static scale saturate and get zero gradient.
    """

    if activation.is_meta:
        return activation
    shape = activation.shape
    quantized = fake_quantize_nvfp4(activation.reshape(-1, shape[-1]), tensor_amax=input_amax.detach())
    return quantized.reshape(shape)


class NVFP4Linear(LoRAMixin, nn.Linear):
    """Linear layer whose weight is NVFP4 fake-quantized on every forward.

    With ``_quantize_activations`` set (W4A4), the input activation is also NVFP4
    fake-quantized using the training running-max as a static global scale, so training
    matches the deployed W4A4 numerics instead of the weight-only (W4A16) approximation.

    With a LoRA adapter attached (:meth:`init_lora`), the frozen base weight and the
    adapter are merged before quantization; inside ``shared_base_teacher()`` the layer
    bypasses both to serve the shared frozen base as the full-precision teacher.

    The frozen base may itself be block-scaled FP8 (:meth:`from_fp8_linear`), in which case
    :meth:`base_weight` dequantizes it transiently and nothing else in the layer changes.
    """

    _compute_dtype: torch.dtype = torch.bfloat16

    @classmethod
    def _empty_like(cls, module: nn.Module, in_features: int, out_features: int) -> "NVFP4Linear":
        # Allocate on meta and rebind the real tensors: the source weight can be several
        # hundred megabytes and a second allocation would be pure waste.
        converted = cls(in_features, out_features, bias=module.bias is not None, device="meta")
        converted.bias = module.bias
        converted.register_buffer("_fp8_scale_inv", None, persistent=False)
        converted._amax_group = None
        converted._track_input_amax = False
        converted._input_amax = None
        converted._input_amax_static = None
        converted._quantize_activations = False
        return converted

    @classmethod
    def from_linear(cls, module: nn.Linear) -> "NVFP4Linear":
        converted = cls._empty_like(module, module.in_features, module.out_features)
        converted.weight = module.weight
        return converted

    @classmethod
    def from_fp8_linear(cls, module: nn.Module, compute_dtype: torch.dtype = torch.bfloat16) -> "NVFP4Linear":
        """Wrap a block-scaled FP8 linear, keeping the FP8 codes as the frozen base."""

        state = fp8_linear_state(module)
        if state is None:
            raise ValueError(f"{type(module).__name__} is not a block-scaled FP8 linear.")
        weight, scale_inv = state
        out_features, in_features = weight.shape
        converted = cls._empty_like(module, in_features, out_features)
        converted.weight = nn.Parameter(weight.detach(), requires_grad=False)
        converted._fp8_scale_inv = scale_inv.detach().float()
        converted._compute_dtype = compute_dtype
        return converted

    def base_weight(self) -> torch.Tensor:
        if getattr(self, "_fp8_scale_inv", None) is None:
            return self.weight
        return dequantize_fp8_blocks(self.weight, self._fp8_scale_inv, self._compute_dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if shared_base_teacher_enabled():
            # Shared frozen base as the full-precision teacher: no quant, no adapter, and
            # no activation statistics (those belong to the student path only).
            return F.linear(input, self.base_weight(), self.bias)
        if self._quantize_activations:
            current = input.detach().abs().amax().float()
            if self._input_amax_static is None:
                # Bootstrap the static global scale from the first observed batch,
                # all-reduced so every rank starts from the same (global) scale. The
                # collective stays matched across ranks: each rank bootstraps this
                # module exactly once, in the same first forward, and the gradient-
                # checkpoint recompute never re-enters here (the static is set by then).
                if dist.is_initialized():
                    dist.all_reduce(current, op=dist.ReduceOp.MAX)
                self._input_amax_static = current.clone()
            if self._track_input_amax and self.training:  # grow the running max for export
                self._input_amax = (
                    current if self._input_amax is None else torch.maximum(self._input_amax, current)
                )
            # Quantize with the step-static scale, never the just-updated running max:
            # the running max is published to ``_input_amax_static`` only between
            # optimizer steps (commit_nvfp4_input_scales), so a gradient-checkpoint
            # recompute reads exactly the scale the original forward used.
            input = fake_quantize_nvfp4_activation(input, self._input_amax_static)
        elif self._track_input_amax and self.training:
            input_amax = input.detach().abs().amax().float()
            self._input_amax = input_amax if self._input_amax is None else torch.maximum(self._input_amax, input_amax)
        tensor_amax = self._amax_group.tensor_amax() if self._amax_group is not None else None
        return F.linear(input, fake_quantize_nvfp4(self._effective_weight(), tensor_amax=tensor_amax), self.bias)
