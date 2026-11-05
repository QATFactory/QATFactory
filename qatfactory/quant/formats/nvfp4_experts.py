"""NVFP4 LoRA-QAD for fused routed-expert tensors.

Recent Hugging Face MoE backbones (MiniMax-M2, Qwen3.5-MoE) keep every routed expert in two
fused 3D parameters rather than per-expert ``nn.Linear`` modules::

    gate_up_proj  [num_experts, 2 * intermediate, hidden]   # w1 stacked over w3
    down_proj     [num_experts, hidden, intermediate]       # w2

This wrapper keeps those parameter names and shapes untouched, adds LoRA factors, and
fake-quantizes each *active* expert's merged weight. ``gate_up_proj`` is quantized as one
tensor so w1 and w3 share a single NVFP4 tensor scale, matching how the serving runtime fuses
them.

Two adapter layouts are supported. ``hybrid_shared`` shares the gate/up ``A`` factor and the
down ``B`` factor across all experts (cheap; the XoRL layout), while the fully unshared layout
gives every expert its own ``A`` and ``B``. Both keep ``B`` zero-initialized, so step 0 is
exactly the post-training-quantization point.
"""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .fp8_base import dequantize_fp8_blocks, fp8_experts_state, is_fp8_tensor
from .lora import shared_base_teacher_enabled
from .nvfp4 import (
    E4M3_MIN_SUBNORMAL,
    NVFP4_BLOCK_SIZE,
    NVFP4_MAX_VALUE,
    NVFP4_SCALE_DENOMINATOR,
    _cast_e4m3,
    _MaskedStraightThrough,
    _round_to_e2m1,
    fake_quantize_nvfp4,
    fake_quantize_nvfp4_activation,
)


LORA_LEAVES = ("gate_up_lora_A", "gate_up_lora_B", "down_lora_A", "down_lora_B")
EXPERT_PROJECTIONS = ("w1", "w3", "w2")


def fused_experts_shapes(module: nn.Module) -> tuple[int, int, int] | None:
    """Return ``(num_experts, hidden, intermediate)`` for a fused-3D experts module."""

    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    if not isinstance(gate_up, nn.Parameter) or not isinstance(down, nn.Parameter):
        return None
    if gate_up.ndim != 3 or down.ndim != 3 or gate_up.shape[0] != down.shape[0]:
        return None
    num_experts, gate_up_rows, hidden = gate_up.shape
    if gate_up_rows % 2:
        return None
    intermediate = gate_up_rows // 2
    if tuple(down.shape[1:]) != (hidden, intermediate):
        return None
    return num_experts, hidden, intermediate


class NVFP4ExpertLoRA(nn.Module):
    """LoRA-QAD wrapper for fused ``[experts, out, in]`` routed-expert weights."""

    _is_nvfp4_expert_lora = True
    # Shared expert-module protocol (see NVFP4Experts): the NVFP4 scale lifecycle in
    # ``nvfp4.py`` drives both wrappers through the same four methods.
    _is_nvfp4_experts = True

    def __init__(
        self,
        source: nn.Module,
        rank: int,
        alpha: float,
        *,
        hybrid_shared: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
        expert_checkpointing: bool = True,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        shapes = fused_experts_shapes(source)
        if shapes is None:
            raise ValueError(
                f"{type(source).__name__} does not expose fused 3D gate_up_proj / down_proj experts."
            )
        self.num_experts, self.hidden_dim, self.intermediate_dim = shapes
        if getattr(source, "has_bias", False) or getattr(source, "is_transposed", False):
            raise ValueError("NVFP4 expert LoRA supports only non-transposed, bias-free expert weights.")

        fp8_state = fp8_experts_state(source)
        self.gate_up_proj = nn.Parameter(source.gate_up_proj.detach(), requires_grad=False)
        self.down_proj = nn.Parameter(source.down_proj.detach(), requires_grad=False)
        self.register_buffer(
            "gate_up_proj_scale_inv",
            fp8_state["gate_up_proj_scale_inv"].detach().float() if fp8_state else None,
            persistent=False,
        )
        self.register_buffer(
            "down_proj_scale_inv",
            fp8_state["down_proj_scale_inv"].detach().float() if fp8_state else None,
            persistent=False,
        )
        self.act_fn = getattr(source, "act_fn", F.silu)
        self._compute_dtype = compute_dtype
        self.expert_checkpointing = bool(expert_checkpointing)

        self.rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_scaling = self.lora_alpha / self.rank
        self.hybrid_shared = bool(hybrid_shared)

        device = self.gate_up_proj.device
        factor_dtype = compute_dtype
        experts, two = self.num_experts, 2
        gate_up_a_shape = (two, rank, self.hidden_dim) if self.hybrid_shared else (experts, two, rank, self.hidden_dim)
        down_b_shape = (self.hidden_dim, rank) if self.hybrid_shared else (experts, self.hidden_dim, rank)
        self.gate_up_lora_A = nn.Parameter(torch.empty(*gate_up_a_shape, device=device, dtype=factor_dtype))
        self.gate_up_lora_B = nn.Parameter(
            torch.zeros(experts, two, self.intermediate_dim, rank, device=device, dtype=factor_dtype)
        )
        self.down_lora_A = nn.Parameter(
            torch.empty(experts, rank, self.intermediate_dim, device=device, dtype=factor_dtype)
        )
        self.down_lora_B = nn.Parameter(torch.zeros(*down_b_shape, device=device, dtype=factor_dtype))
        self.reset_lora_parameters()

        # W4A4 state. The serving runtime fuses all experts into one kernel per projection
        # group, so there is one activation scale for the gate/up input and one for the down
        # input per layer — not one per expert.
        self._quantize_activations = False
        self._track_input_amax = False
        self._gate_up_input_amax = None
        self._gate_up_input_amax_static = None
        self._down_input_amax = None
        self._down_input_amax_static = None

    # The two activation scales a serving FusedMoE layer carries, in export order.
    INPUT_SCALE_GROUPS = (("w13", "_gate_up_input_amax"), ("w2", "_down_input_amax"))

    @torch.no_grad()
    def commit_input_scales(self, process_group=None) -> None:
        """Publish the running input maxima as the next step's static W4A4 scales."""

        if not self._quantize_activations or self._gate_up_input_amax is None:
            return
        maxima = torch.stack([self._gate_up_input_amax, self._down_input_amax])
        if dist.is_initialized():
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=process_group)
        self._gate_up_input_amax, self._down_input_amax = (value.clone() for value in maxima.unbind())
        self._gate_up_input_amax_static, self._down_input_amax_static = maxima.unbind()

    @torch.no_grad()
    def finalize_input_scales(self, prefix: str, process_group=None) -> dict[str, torch.Tensor]:
        """Reduce the running maxima and return this layer's exported activation scales."""

        if self._gate_up_input_amax is None:
            raise RuntimeError(f"Missing training input-amax observations for {prefix}.")
        maxima = torch.stack([self._gate_up_input_amax, self._down_input_amax])
        if dist.is_initialized():
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=process_group)
        self._gate_up_input_amax, self._down_input_amax = (value.clone() for value in maxima.unbind())
        scales = {}
        for (name, _), maximum in zip(self.INPUT_SCALE_GROUPS, maxima.unbind(), strict=True):
            if not bool(torch.isfinite(maximum)) or not bool(maximum > 0):
                raise RuntimeError(f"Invalid training input amax for {prefix}.{name}.")
            scales[f"{prefix}.{name}.input_scale"] = (maximum / NVFP4_SCALE_DENOMINATOR).reshape(())
        return scales

    def runtime_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, attribute).detach().cpu()
            for name, attribute in self.INPUT_SCALE_GROUPS
            if getattr(self, attribute) is not None
        }

    def load_runtime_state_dict(self, state: dict[str, torch.Tensor], device) -> None:
        for name, attribute in self.INPUT_SCALE_GROUPS:
            if name not in state:
                continue
            value = state[name].to(device).reshape(())
            setattr(self, attribute, value)
            setattr(self, f"{attribute}_static", value.clone())

    def reset_lora_parameters(self) -> None:
        for projection in self.gate_up_lora_A.reshape(-1, self.rank, self.hidden_dim):
            nn.init.kaiming_uniform_(projection, a=math.sqrt(5))
        for projection in self.down_lora_A.reshape(-1, self.rank, self.intermediate_dim):
            nn.init.kaiming_uniform_(projection, a=math.sqrt(5))
        nn.init.zeros_(self.gate_up_lora_B)
        nn.init.zeros_(self.down_lora_B)

    @property
    def is_fp8_base(self) -> bool:
        return self.gate_up_proj_scale_inv is not None

    def target_names(self, prefix: str) -> list[str]:
        """Per-expert projection FQNs in serving (``w1``/``w3``/``w2``) naming."""

        return [
            f"{prefix}.{expert}.{projection}"
            for expert in range(self.num_experts)
            for projection in EXPERT_PROJECTIONS
        ]

    def trainable_parameter_names(self, prefix: str) -> set[str]:
        return {f"{prefix}.{leaf}" for leaf in LORA_LEAVES}

    def base_gate_up(self, expert: int | torch.Tensor) -> torch.Tensor:
        weight = self.gate_up_proj[expert]
        if self.gate_up_proj_scale_inv is None:
            return weight
        return dequantize_fp8_blocks(weight, self.gate_up_proj_scale_inv[expert], self._compute_dtype)

    def base_down(self, expert: int | torch.Tensor) -> torch.Tensor:
        weight = self.down_proj[expert]
        if self.down_proj_scale_inv is None:
            return weight
        return dequantize_fp8_blocks(weight, self.down_proj_scale_inv[expert], self._compute_dtype)

    def effective_gate_up(self, expert: int | torch.Tensor) -> torch.Tensor:
        """Merged ``[2 * intermediate, hidden]`` weight for one expert (w1 over w3)."""

        base = self.base_gate_up(expert)
        lora_a = self.gate_up_lora_A if self.hybrid_shared else self.gate_up_lora_A[expert]
        delta = torch.matmul(self.gate_up_lora_B[expert], lora_a) * self.lora_scaling
        return base + delta.reshape_as(base).to(base.dtype)

    def effective_down(self, expert: int | torch.Tensor) -> torch.Tensor:
        base = self.base_down(expert)
        lora_b = self.down_lora_B if self.hybrid_shared else self.down_lora_B[expert]
        delta = torch.matmul(lora_b, self.down_lora_A[expert]) * self.lora_scaling
        return base + delta.to(base.dtype)

    def _gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def _teacher_expert(self, tokens: torch.Tensor, expert: int) -> torch.Tensor:
        return F.linear(self._gate(F.linear(tokens, self.base_gate_up(expert))), self.base_down(expert))

    @staticmethod
    def _grow(running: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
        return value if running is None else torch.maximum(running, value)

    def _quantize_input(self, value: torch.Tensor, static: torch.Tensor | None) -> torch.Tensor:
        """Fake-quantize one matmul input with the step-static global scale.

        The static scale is only republished between optimizer steps, so a gradient
        checkpoint recompute reads exactly the scale the original forward used.
        """

        if not self._quantize_activations or static is None:
            return value
        return fake_quantize_nvfp4_activation(value, static)

    def _student_expert(self, tokens: torch.Tensor, expert: int) -> torch.Tensor:
        # w1 and w3 are quantized as one fused tensor so they share a single NVFP4 tensor
        # scale, exactly as the serving runtime packs them.
        tokens = self._quantize_input(tokens, self._gate_up_input_amax_static)
        gate_up = F.linear(tokens, fake_quantize_nvfp4(self.effective_gate_up(expert)))
        gated = self._gate(gate_up)
        if self._track_input_amax and self.training:
            # Growing a maximum is idempotent, so a gradient-checkpoint recompute of this
            # same expert cannot corrupt the statistic.
            with torch.no_grad():
                self._down_input_amax = self._grow(
                    self._down_input_amax, gated.detach().abs().amax().float()
                )
        return F.linear(
            self._quantize_input(gated, self._down_input_amax_static),
            fake_quantize_nvfp4(self.effective_down(expert)),
        )

    @torch.no_grad()
    def _bootstrap_input_scales(self, hidden_states, assignments, counts) -> None:
        """Set both static activation scales from the first observed batch.

        The down-projection input only exists per expert, so its maximum is taken over every
        active expert's gated intermediate rather than whichever expert happened to be routed
        first. That costs one extra pass, which is why it runs only once. The all-reduce makes
        every rank start from the same global scale and is matched: each rank bootstraps this
        module exactly once.
        """

        gate_up_max = hidden_states.detach().abs().amax().float()
        down_max = gate_up_max.new_zeros(())
        start = 0
        for expert, count in enumerate(counts):
            if count == 0:
                continue
            token_idx = assignments[start : start + count, 2]
            start += count
            gated = self._gate(F.linear(hidden_states[token_idx], self.base_gate_up(expert)))
            down_max = torch.maximum(down_max, gated.abs().amax().float())

        if dist.is_initialized():
            observed = torch.stack([gate_up_max, down_max])
            dist.all_reduce(observed, op=dist.ReduceOp.MAX)
            gate_up_max, down_max = observed.unbind()
        self._gate_up_input_amax_static = gate_up_max.clone()
        self._down_input_amax_static = down_max.clone()

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        teacher = shared_base_teacher_enabled()
        expert_fn = self._teacher_expert if teacher else self._student_expert
        # Recompute each expert's merged + fake-quantized weight in its own backward instead
        # of keeping all of them alive: at 256 experts the saved weights would otherwise be
        # tens of gigabytes per layer.
        recompute = self.expert_checkpointing and not teacher and torch.is_grad_enabled()

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            # One host transfer for the whole layer. Asking per expert — via `int(index)` or
            # `torch.where` on that expert's mask — would synchronize the device once per
            # expert, and at 256 experts that dominates the step. `nonzero` returns rows in
            # row-major order, so each expert's rows are contiguous and can be sliced with
            # bounds known on the host while the indices themselves stay on the device.
            assignments = expert_mask.nonzero()
            counts = torch.bincount(assignments[:, 0], minlength=self.num_experts).tolist()
            if not teacher and self._track_input_amax:
                if self._quantize_activations and self._gate_up_input_amax_static is None:
                    self._bootstrap_input_scales(hidden_states, assignments, counts)
                self._gate_up_input_amax = self._grow(
                    self._gate_up_input_amax, hidden_states.detach().abs().amax().float()
                )

        start = 0
        for expert, count in enumerate(counts):
            if count == 0:
                continue
            rows = assignments[start : start + count]
            start += count
            top_k_pos, token_idx = rows[:, 1], rows[:, 2]
            tokens = hidden_states[token_idx]
            if recompute:
                current = checkpoint(expert_fn, tokens, expert, use_reentrant=False)
            else:
                current = expert_fn(tokens, expert)
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
        return final_hidden_states

    @torch.no_grad()
    def merged_projection(self, expert: int, projection: str) -> torch.Tensor:
        """One merged 2D expert projection in serving naming, for export."""

        if projection == "w2":
            return self.effective_down(expert)
        gate_up = self.effective_gate_up(expert)
        gate, up = gate_up.chunk(2, dim=0)
        if projection == "w1":
            return gate
        if projection == "w3":
            return up
        raise ValueError(f"Unknown expert projection {projection!r}; expected one of {EXPERT_PROJECTIONS}.")


def is_fused_experts(module: nn.Module) -> bool:
    """Whether ``module`` holds convertible fused routed-expert weights."""

    if isinstance(module, (NVFP4ExpertLoRA, NVFP4Experts)):
        return False
    if fused_experts_shapes(module) is None:
        return False
    gate_up = module.gate_up_proj
    return gate_up.is_floating_point() or is_fp8_tensor(gate_up)


# Cap on elements per expert fake-quant slab. The elementwise chain materializes several
# fp32/int64 intermediates per element; on a full MiniMax-M2.7 gate_up layer (256 experts,
# 2.4B elements) that is ~45 GB of transients. 2**27 elements keeps it at ~2-3 GB.
_EXPERT_FAKE_QUANT_CHUNK_ELEMENTS = 1 << 27


def fake_quantize_nvfp4_experts(weight: torch.Tensor, tensor_amax: torch.Tensor) -> torch.Tensor:
    """Fake-quantize a packed 3D expert weight ``[E, rows, in_features]`` per expert.

    ``tensor_amax`` holds one amax per expert (``[E]``). Each expert slice gets exactly the
    2D :func:`fake_quantize_nvfp4` math (every operation is elementwise or reduces over the
    block dimension only), so the result is bit-identical to quantizing the experts one by
    one. Experts with a zero amax quantize to zero with a pass-through gradient, as in the
    2D path. Large tensors are processed in expert-dimension slabs to bound transient memory;
    per-expert separability keeps that bit-identical too.
    """

    if weight.ndim != 3:
        raise ValueError(f"Expert fake quantization expects a 3D weight, got shape {tuple(weight.shape)}.")
    if weight.shape[-1] % NVFP4_BLOCK_SIZE:
        raise ValueError(
            f"NVFP4 expert weights require in_features divisible by {NVFP4_BLOCK_SIZE}, "
            f"got shape {tuple(weight.shape)}."
        )
    if tensor_amax.shape != (weight.shape[0],):
        raise ValueError(f"Expert tensor_amax must have shape ({weight.shape[0]},), got {tuple(tensor_amax.shape)}.")
    if weight.is_meta:
        return weight

    experts_per_chunk = max(1, _EXPERT_FAKE_QUANT_CHUNK_ELEMENTS // max(weight.shape[1] * weight.shape[2], 1))
    if experts_per_chunk < weight.shape[0]:
        return torch.cat(
            [
                _fake_quantize_nvfp4_experts_whole(
                    weight[start : start + experts_per_chunk], tensor_amax[start : start + experts_per_chunk]
                )
                for start in range(0, weight.shape[0], experts_per_chunk)
            ],
            dim=0,
        )
    return _fake_quantize_nvfp4_experts_whole(weight, tensor_amax)


def _fake_quantize_nvfp4_experts_whole(weight: torch.Tensor, tensor_amax: torch.Tensor) -> torch.Tensor:
    # The whole lattice computation runs without autograd, as in the 2D path: the masked
    # straight-through estimator only needs `in_range`, and a tracked graph would keep every
    # FP32 intermediate of the layer's experts alive until backward (tens of GB per MiniMax
    # layer on a gradient-checkpoint recompute).
    with torch.no_grad():
        quantized, in_range = _fake_quantize_nvfp4_experts_values(weight.detach(), tensor_amax)
    return _MaskedStraightThrough.apply(weight, quantized, in_range)


def _check_expert_amax(tensor_amax: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor_amax).all()) or bool((tensor_amax < 0).any()):
        raise ValueError("Expert tensor_amax values must be finite and non-negative.")


def _fake_quantize_nvfp4_experts_values(weight: torch.Tensor, tensor_amax: torch.Tensor, validate: bool = True):
    original_dtype = weight.dtype
    latent = weight.float()
    blocks = latent.reshape(*latent.shape[:-1], -1, NVFP4_BLOCK_SIZE)

    with torch.no_grad():
        tensor_amax = tensor_amax.detach().float()
        if validate:  # a host sync; per-expert callers validate the whole amax vector once instead
            _check_expert_amax(tensor_amax)
        zero_expert = (tensor_amax == 0).view(-1, 1, 1, 1)
        global_scale = torch.where(
            tensor_amax > 0, tensor_amax / NVFP4_SCALE_DENOMINATOR, torch.ones_like(tensor_amax)
        ).view(-1, 1, 1, 1)
        block_amax = blocks.detach().abs().amax(dim=-1, keepdim=True)
        block_scale = _cast_e4m3(block_amax / (global_scale * NVFP4_MAX_VALUE))
        block_scale = torch.where(block_scale == 0, block_scale.new_full((), E4M3_MIN_SUBNORMAL), block_scale)
        dequant_scale = block_scale * global_scale

    normalized = (blocks / dequant_scale).clamp(-NVFP4_MAX_VALUE, NVFP4_MAX_VALUE)
    quantized = _round_to_e2m1(normalized) * dequant_scale
    in_range = (blocks.detach().abs() <= dequant_scale * NVFP4_MAX_VALUE) | zero_expert
    quantized = torch.where(zero_expert, torch.zeros_like(quantized), quantized)
    quantized = quantized.reshape_as(latent).to(original_dtype)
    in_range = in_range.reshape_as(latent)
    return quantized, in_range


class _ExpertFakeQuantLinear(torch.autograd.Function):
    """``F.linear(x, fake_quant(w))`` for one expert, with the masked straight-through gradient to ``w``.

    The fake-quantized weight and its in-range mask are recomputed in backward from the latent
    slice ``w`` (a view of the gathered expert tensor that FSDP holds for the layer anyway)
    instead of being kept from the forward. Keeping them meant one fake-quantized copy, one
    mask and one masked gradient of the layer's entire expert tensor alive at the backward
    peak, about 17 GiB per GPU for a MiniMax-M2.7 layer. Values and gradients are unchanged:
    the per-expert lattice is separable, and the products are the ones ``F.linear`` computes.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            quantized, _ = _fake_quantize_nvfp4_experts_values(w.unsqueeze(0), amax.reshape(1), validate=False)
        ctx.save_for_backward(x, w, amax)
        return F.linear(x, quantized[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, w, amax = ctx.saved_tensors
        with torch.no_grad():
            quantized, in_range = _fake_quantize_nvfp4_experts_values(w.unsqueeze(0), amax.reshape(1), validate=False)
        grad_x = grad_output.matmul(quantized[0]) if ctx.needs_input_grad[0] else None
        grad_w = None
        if ctx.needs_input_grad[1]:
            grad_w = grad_output.reshape(-1, grad_output.shape[-1]).t().mm(x.reshape(-1, x.shape[-1]))
            grad_w = grad_w * in_range[0]
        return grad_x, grad_w, None


class NVFP4Experts(nn.Module):
    """Full-parameter NVFP4 QAD for fused ``[experts, out, in]`` routed-expert weights.

    Drop-in replacement for the transformers>=5.x fused experts module (contract:
    ``forward(hidden_states, top_k_index, top_k_weights)`` over ``gate_up_proj``
    ``[E, 2 * intermediate, hidden]`` and ``down_proj`` ``[E, hidden, intermediate]``). Both
    3D latents are trained; every expert is fake-quantized on every forward.

    Each expert's whole ``gate_up_proj[e]`` slice shares one tensor amax, so the exported
    gate and up global scales of an expert are equal by construction, which is what the
    serving fused-MoE runtime (w13) requires. Activation scales mirror the serving
    quantizers: one scalar for the block input (w13, shared by every expert) and one for the
    expert intermediates (w2, the max over all routed experts).

    Unlike :class:`NVFP4ExpertLoRA` this needs a separate teacher model: the latents are the
    trained weights, so the frozen full-precision reference cannot share them.
    """

    _is_nvfp4_experts = True
    INPUT_SCALE_GROUPS = NVFP4ExpertLoRA.INPUT_SCALE_GROUPS

    def __init__(self, gate_up_proj: nn.Parameter, down_proj: nn.Parameter, act_fn):
        super().__init__()
        if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
            raise ValueError("NVFP4Experts requires fused 3D gate_up_proj and down_proj parameters.")
        num_experts, two_intermediate, hidden = gate_up_proj.shape
        if two_intermediate % 2 or down_proj.shape != (num_experts, hidden, two_intermediate // 2):
            raise ValueError(
                f"Inconsistent expert shapes: gate_up_proj {tuple(gate_up_proj.shape)}, "
                f"down_proj {tuple(down_proj.shape)}."
            )
        for name, parameter in (("gate_up_proj", gate_up_proj), ("down_proj", down_proj)):
            if parameter.shape[-1] % NVFP4_BLOCK_SIZE:
                raise ValueError(
                    f"NVFP4 requires {name} in_features divisible by {NVFP4_BLOCK_SIZE}, got {tuple(parameter.shape)}."
                )
        self.num_experts = num_experts
        self.hidden_dim = hidden
        self.intermediate_dim = two_intermediate // 2
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn
        self._track_input_amax = False
        self._quantize_activations = False
        self._gate_up_input_amax = None
        self._gate_up_input_amax_static = None
        self._down_input_amax = None
        self._down_input_amax_static = None

    @classmethod
    def from_experts(cls, module: nn.Module) -> "NVFP4Experts":
        for attribute in ("gate_up_proj", "down_proj", "act_fn"):
            if not hasattr(module, attribute):
                raise ValueError(f"Cannot convert {type(module).__name__}: missing {attribute!r}.")
        if is_fp8_tensor(module.gate_up_proj) or is_fp8_tensor(module.down_proj):
            raise ValueError(
                "Full-parameter expert QAD trains the expert latents and needs a BF16 base; "
                "dequantize the FP8 checkpoint first, or use LoRA-QAD (qad.lora.enable), which "
                "keeps the FP8 base frozen."
            )
        if getattr(module, "has_bias", False) or getattr(module, "is_transposed", False):
            raise ValueError("NVFP4 expert QAD supports only non-transposed, bias-free expert weights.")
        for name, _ in module.named_parameters():
            if name not in {"gate_up_proj", "down_proj"}:
                raise ValueError(f"Unsupported extra expert parameter {name!r} (biased experts are not supported).")
        return cls(module.gate_up_proj, module.down_proj, module.act_fn)

    def target_names(self, prefix: str) -> list[str]:
        """Per-expert projection FQNs in serving (``w1``/``w3``/``w2``) naming."""

        return [
            f"{prefix}.{expert}.{projection}" for expert in range(self.num_experts) for projection in EXPERT_PROJECTIONS
        ]

    def trainable_parameter_names(self, prefix: str) -> set[str]:
        return {f"{prefix}.gate_up_proj", f"{prefix}.down_proj"}

    # Scale lifecycle: identical to NVFP4ExpertLoRA, which owns the implementation.
    commit_input_scales = NVFP4ExpertLoRA.commit_input_scales
    finalize_input_scales = NVFP4ExpertLoRA.finalize_input_scales
    runtime_state_dict = NVFP4ExpertLoRA.runtime_state_dict
    load_runtime_state_dict = NVFP4ExpertLoRA.load_runtime_state_dict

    @staticmethod
    def _grow(running: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
        amax = value.detach().abs().amax().float()
        return amax if running is None else torch.maximum(running, amax)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        tracking = self._track_input_amax and self.training
        quantize = self._quantize_activations
        if quantize:
            # W4A4: the block input is quantized once with the step-static scale (shared by
            # all experts, like the serving w13 scale). The running max is only published
            # between optimizer steps, so a gradient-checkpoint recompute is bit-identical;
            # the very first forward bootstraps the static scale.
            if self._gate_up_input_amax_static is None:
                self._gate_up_input_amax_static = hidden_states.detach().abs().amax().float()
            if tracking:
                self._gate_up_input_amax = self._grow(self._gate_up_input_amax, hidden_states)
            hidden_states = fake_quantize_nvfp4_activation(hidden_states, self._gate_up_input_amax_static)
        elif tracking:
            self._gate_up_input_amax = self._grow(self._gate_up_input_amax, hidden_states)

        gate_up_amax = self.gate_up_proj.detach().abs().amax(dim=(1, 2)).float()
        down_amax = self.down_proj.detach().abs().amax(dim=(1, 2)).float()
        _check_expert_amax(gate_up_amax)
        _check_expert_amax(down_amax)
        # Each expert is fake-quantized inside its own matmul (see _ExpertFakeQuantLinear), so no
        # fake-quantized copy of the whole expert tensor is ever held for backward. unbind
        # (rather than indexing per expert) lets backward stack all slice grads in one op.
        gate_up = self.gate_up_proj.unbind(0)
        down = self.down_proj.unbind(0)

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            hit_experts = torch.unique(top_k_index)
        for expert_index in hit_experts.tolist():
            if not 0 <= expert_index < self.num_experts:  # expert-parallel padding sentinel
                continue
            token_index, k_position = torch.where(top_k_index == expert_index)
            gate, up = _ExpertFakeQuantLinear.apply(
                hidden_states[token_index], gate_up[expert_index], gate_up_amax[expert_index]
            ).chunk(2, dim=-1)
            intermediate = self.act_fn(gate) * up
            if quantize:
                if self._down_input_amax_static is None:  # first-ever forward: bootstrap from this expert
                    self._down_input_amax_static = intermediate.detach().abs().amax().float()
                if tracking:
                    self._down_input_amax = self._grow(self._down_input_amax, intermediate)
                intermediate = fake_quantize_nvfp4_activation(intermediate, self._down_input_amax_static)
            elif tracking:
                self._down_input_amax = self._grow(self._down_input_amax, intermediate)
            expert_output = _ExpertFakeQuantLinear.apply(intermediate, down[expert_index], down_amax[expert_index])
            expert_output = expert_output * top_k_weights[token_index, k_position, None]
            final_hidden_states.index_add_(0, token_index, expert_output.to(final_hidden_states.dtype))
        return final_hidden_states

    @torch.no_grad()
    def merged_projection(self, expert: int, projection: str) -> torch.Tensor:
        """One 2D expert latent in serving naming, for export."""

        if projection == "w2":
            return self.down_proj[expert]
        gate, up = self.gate_up_proj[expert].chunk(2, dim=0)
        if projection == "w1":
            return gate
        if projection == "w3":
            return up
        raise ValueError(f"Unknown expert projection {projection!r}; expected one of {EXPERT_PROJECTIONS}.")
