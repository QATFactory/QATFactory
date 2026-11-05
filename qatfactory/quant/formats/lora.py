"""Format-agnostic LoRA and the shared-base teacher context.

The base weight of every quantized linear stays frozen; a low-rank adapter
``scaling * (B @ A)`` is folded into it *before* fake-quantization, so the student
trains ``fake_quant(W + scaling * B @ A)`` — the exact merged weight the deployment
kernel quantizes (LoRA is merged, then quantized, at serve time). The masked
straight-through estimator therefore routes gradients only into ``A``/``B``.

Because the base ``W`` is frozen, one model serves as both teacher and student. Inside
:func:`shared_base_teacher` every quantized module returns its plain full-precision base
result — no quantization, no adapter, no activation statistics. The switch is a
:class:`~contextvars.ContextVar`, so it is task-local and exception-safe: a failed teacher
forward can never leave the model stuck in teacher mode.

``base_weight`` is the single extension point for frozen bases that are not already the
compute dtype. Checkpoints stored as block-scaled FP8 keep their FP8 codes as the frozen
parameter and dequantize transiently here, which is what lets a 230B FP8 backbone train
without ever materializing a full high-precision copy.
"""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn


_SHARED_BASE_TEACHER = ContextVar("qatfactory_shared_base_teacher", default=False)


def shared_base_teacher_enabled() -> bool:
    """Whether the current forward must use only the frozen full-precision base."""

    return _SHARED_BASE_TEACHER.get()


@contextmanager
def shared_base_teacher() -> Iterator[None]:
    """Temporarily turn the instrumented student into its own shared-base teacher."""

    token = _SHARED_BASE_TEACHER.set(True)
    try:
        yield
    finally:
        _SHARED_BASE_TEACHER.reset(token)


class LoRAMixin:
    """Adds an optional low-rank adapter to a quantized linear.

    Plain-value class defaults keep every non-LoRA instance a no-op until
    :meth:`init_lora` runs. ``lora_A``/``lora_B`` are declared as annotations only (no class
    attribute) so ``nn.Module.register_parameter`` doesn't treat the name as already-taken
    when the real ``nn.Parameter`` is assigned; readers use ``getattr(self, "lora_A", None)``
    before then.
    """

    lora_A: "nn.Parameter | None"
    lora_B: "nn.Parameter | None"
    lora_rank: int = 0
    lora_scaling: float = 1.0

    def init_lora(self, rank: int, alpha: float) -> None:
        """Attach a rank-``r`` adapter: ``A`` Kaiming-init, ``B`` zero (delta starts at 0).

        Zero ``B`` means the initial effective weight is exactly ``W``, so training
        starts from the pure post-training-quantization point ``fake_quant(W)`` and the
        adapter only ever *adds* learned recovery.
        """

        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        base = self.base_weight()
        self.lora_rank = int(rank)
        self.lora_scaling = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, device=base.device, dtype=base.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device=base.device, dtype=base.dtype)
        )
        # Kaiming-uniform A / zero B is the standard LoRA init; the product B @ A is 0 at
        # step 0 so the student == fake_quant(W) before any optimization.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def has_lora(self) -> bool:
        return getattr(self, "lora_A", None) is not None

    def base_weight(self) -> torch.Tensor:
        """The frozen base weight in the compute dtype (identity for unquantized bases)."""

        return self.weight

    def lora_delta(self) -> torch.Tensor:
        """The dense weight delta ``scaling * (B @ A)``."""

        return self.lora_scaling * (self.lora_B @ self.lora_A)

    def _effective_weight(self) -> torch.Tensor:
        """The merged latent weight ``W + scaling * B @ A`` (or ``W`` when no adapter).

        Under FSDP2 this runs inside the module forward, where the base and the adapters are
        already all-gathered to full tensors, so the product is dense and correct. Gradients
        flow to ``A``/``B`` (the base is frozen).
        """

        base = self.base_weight()
        if getattr(self, "lora_A", None) is None:
            return base
        return base + self.lora_delta().to(base.dtype)


def is_lora_quantized(module: nn.Module) -> bool:
    return isinstance(module, LoRAMixin) and getattr(module, "lora_A", None) is not None


def merge_lora_weight(
    weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Fold a saved adapter into a base weight for export (``W + scaling * B @ A``).

    The factors keep their saved dtype so the product and scaling round exactly where the
    training forward did; the exporter then (re)quantizes the merged weight just as it does
    a plain QAD latent weight.
    """

    delta = lora_B @ lora_A
    return weight + (scaling * delta).to(weight.dtype)
