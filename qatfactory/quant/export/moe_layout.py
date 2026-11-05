"""How released MoE checkpoints spell their routed experts.

``transformers`` keeps two fused 3D parameters per routed-experts module
(``experts.gate_up_proj`` with the gate rows above the up rows, and ``experts.down_proj``),
but released checkpoints store one 2D tensor per expert and projection under family-specific
names. The offline fuse (``scripts/fuse_moe_checkpoint.py``) and the merged-BF16 export
(``scripts/merge_lora_bf16.py --released-layout``) translate between the two, so they share
the naming table here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ExpertLayout:
    name: str
    # Released per-expert projection names whose rows form ``gate_up_proj`` in this order.
    gate_up: tuple[str, str]
    # Released per-expert projection name that becomes ``down_proj``.
    down: str
    # (released, in-memory) spelling of the MoE block, when the two differ.
    module_rename: tuple[str, str] | None = None
    model_types: tuple[str, ...] = ()

    @property
    def projections(self) -> tuple[str, str, str]:
        return (*self.gate_up, self.down)

    def to_fused_prefix(self, released_prefix: str) -> str:
        if self.module_rename is None:
            return released_prefix
        released, in_memory = self.module_rename
        return released_prefix.replace(released, in_memory)

    def to_released_prefix(self, fused_prefix: str) -> str:
        if self.module_rename is None:
            return fused_prefix
        released, in_memory = self.module_rename
        return fused_prefix.replace(in_memory, released)

    def released_keys(self, fused_prefix: str, leaf: str, expert: int) -> tuple[str, ...]:
        """Per-expert weight keys a serving stack loads for one fused tensor of one expert."""

        stem = f"{self.to_released_prefix(fused_prefix)}.{expert}"
        names = self.gate_up if leaf == "gate_up_proj" else (self.down,)
        return tuple(f"{stem}.{name}.weight" for name in names)


# Mixtral-style: `block_sparse_moe.experts.{e}.w1/w3/w2` (MiniMax-M2 among them).
MIXTRAL = ExpertLayout(
    "mixtral",
    gate_up=("w1", "w3"),
    down="w2",
    module_rename=(".block_sparse_moe.", ".mlp."),
    model_types=("mixtral", "minimax", "minimax_m2"),
)
# Qwen-style: `mlp.experts.{e}.gate_proj/up_proj/down_proj` (Qwen2-MoE, Qwen3-MoE, Qwen3-Next).
QWEN_MOE = ExpertLayout(
    "qwen_moe",
    gate_up=("gate_proj", "up_proj"),
    down="down_proj",
    model_types=("qwen2_moe", "qwen3_moe", "qwen3_next", "qwen3_5_moe"),
)
LAYOUTS: tuple[ExpertLayout, ...] = (MIXTRAL, QWEN_MOE)

# `...experts.<index>.<projection>.<suffix>`, e.g. `...experts.7.w1.weight_scale_inv`.
EXPERT_KEY = re.compile(
    r"^(?P<prefix>.*\.experts)\.(?P<expert>\d+)\.(?P<projection>"
    + "|".join(sorted({p for layout in LAYOUTS for p in layout.projections}, key=len, reverse=True))
    + r")\.(?P<suffix>.+)$"
)


def detect_layout(keys: Iterable[str]) -> ExpertLayout:
    """Pick the layout whose per-expert projection names appear in a released checkpoint."""

    seen = {match["projection"] for key in keys if (match := EXPERT_KEY.match(key))}
    if not seen:
        raise ValueError("No per-expert routed-expert tensors found; is this already a fused checkpoint?")
    matches = [layout for layout in LAYOUTS if seen <= set(layout.projections)]
    if len(matches) != 1:
        raise ValueError(f"Per-expert projections {sorted(seen)} do not match exactly one known MoE layout.")
    return matches[0]


def layout_for_model_type(model_type: str) -> ExpertLayout:
    for layout in LAYOUTS:
        if model_type in layout.model_types:
            return layout
    known = sorted(t for layout in LAYOUTS for t in layout.model_types)
    raise ValueError(f"No released-expert layout registered for model_type {model_type!r}; known: {known}.")
