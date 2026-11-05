"""Teacher routing replay for MoE QAD, adapted from R3 (arXiv:2510.11370).

In self-distillation QAD the teacher and student share identical, frozen router
weights, so top-k expert flips between them arise purely from quantization-induced
drift of the router's input hidden states. Replaying the teacher's expert selection in
the student forward removes the expert-identity mismatch from the KD loss during the
early, high-flip phase of training (the replay schedule then anneals to free routing so
late training optimizes the deployed objective).

Mechanism (R3-faithful "mask" mode): record the teacher's top-k indices per (layer,
token) with a forward hook on each router (``mlp.gate``); in the student's router,
keep its own logits in the graph but replace the selected experts with the teacher's
and renormalize the student's softmax over that set — gradients still flow to the
upstream (quantized) layers through the gating weights. "full" mode replays the
teacher's gating weights too, removing the gating mismatch from the loss entirely at
the cost of severing that gradient path.

Works with both transformers router output conventions (<=5.5 returns full softmax
probabilities first, >=5.9 returns raw logits first) and with both scoring conventions:
softmax (qwen3_moe, qwen3_5_moe) and sigmoid (minimax_m2, whose gate weights are
gathered sigmoids renormalized by their sum, and whose ``e_score_correction_bias``
shifts top-k SELECTION only — never the weights). The scoring convention is read from
the model config when it is explicit and probed at attach time otherwise; the output
convention is always probed. Replay state is keyed per module and only reset when a new
batch begins, so gradient-checkpointing recomputation replays identical values.
"""

import inspect
import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)

REPLAY_MODES = ("mask", "full")

# Sigmoid-scored MoE archs whose configs don't declare a ``scoring_func``.
_SIGMOID_SCORED_MODEL_TYPES = ("minimax_m2",)


def _find_routers(model: nn.Module) -> dict[str, nn.Module]:
    """Map router FQN (``<block>.gate``) -> router module (the ``gate`` sibling of ``experts``)."""

    routers = {}
    for name, module in model.named_modules():
        children = dict(module.named_children())
        if "gate" in children and "experts" in children:
            routers[f"{name}.gate" if name else "gate"] = children["gate"]
    return routers


def _sigmoid_scoring_from_config(model: nn.Module) -> bool | None:
    """Read the router scoring convention from the model config when it is explicit.

    Returns True/False for a definitive answer, None when the config is absent (bare
    blocks in tests) or silent about scoring, so the caller falls back to probing.
    """

    config = getattr(model, "config", None)
    if config is None:
        return None
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    scoring = getattr(text_config, "scoring_func", None)
    if scoring is not None:
        return scoring == "sigmoid"
    if getattr(text_config, "model_type", None) in _SIGMOID_SCORED_MODEL_TYPES:
        return True
    return None


@torch.no_grad()
def _probe_router_conventions(router: nn.Module) -> tuple[bool, bool]:
    """Probe (first output is probabilities, scoring is sigmoid) from one router forward.

    Routers with a two-argument forward (MiniMaxM2TopKRouter takes an
    ``e_score_correction_bias``) are probed with a zero bias, which leaves both the
    selection and the weights at their bias-free values. The scoring convention is
    decided by reconstructing the router's own top-k weights from its first output
    under each hypothesis and keeping the one that reproduces them.
    """

    weight = getattr(router, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError(f"Unsupported router module {type(router).__name__}: expected a 2D weight.")
    probe = torch.randn(4, weight.shape[1], dtype=weight.dtype, device=weight.device)
    required = [
        parameter
        for parameter in inspect.signature(router.forward).parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(required) >= 2:
        first, scores, indices = router(probe, torch.zeros(weight.shape[0], device=weight.device))
    else:
        first, scores, indices = router(probe)
    first = first.float()
    row_sums = first.sum(dim=-1)
    if bool((first >= 0).all()) and bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)):
        return True, False  # <=5.5 full-probabilities convention is softmax-only
    errors = []
    for candidate in (torch.softmax(first, dim=-1), torch.sigmoid(first)):
        gathered = candidate.gather(-1, indices)
        if getattr(router, "norm_topk_prob", True):
            gathered = gathered / gathered.sum(dim=-1, keepdim=True)
        errors.append((gathered - scores.float()).abs().max())
    return False, bool(errors[1] < errors[0])


class RoutingReplay:
    """Records teacher top-k routing per batch and replays it in the student forward."""

    def __init__(self, teacher: nn.Module, student: nn.Module, *, mode: str = "mask", eps: float = 1e-9):
        if mode not in REPLAY_MODES:
            raise ValueError(f"routing replay mode must be one of {REPLAY_MODES}, got {mode!r}.")
        teacher_routers = _find_routers(teacher)
        student_routers = _find_routers(student)
        if not teacher_routers:
            raise ValueError("No MoE routers (modules with 'gate' and 'experts' children) found.")
        if set(teacher_routers) != set(student_routers):
            raise ValueError(
                f"Teacher/student router mismatch; teacher-only="
                f"{sorted(set(teacher_routers) - set(student_routers))[:3]}, student-only="
                f"{sorted(set(student_routers) - set(teacher_routers))[:3]}."
            )
        self.mode = mode
        self.eps = eps
        self.router_names = tuple(sorted(teacher_routers))
        # Probe the TEACHER router: it shares the student's class/convention but may still
        # be unsharded (calling a router directly on FSDP2-sharded weights is invalid).
        # Under cpu-ram-efficient loading the teacher weights are meta everywhere except
        # global rank 0 at attach time, so the probe can only run where weights are real;
        # probe there and broadcast the (rank-0-authoritative) result to every rank.
        probe_router = teacher_routers[self.router_names[0]]
        probe_weight = getattr(probe_router, "weight", None)
        if probe_weight is not None and getattr(probe_weight, "is_meta", False):
            probed = [None, None]
        else:
            probed = list(_probe_router_conventions(probe_router))
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            payload = [probed]
            torch.distributed.broadcast_object_list(payload, src=0)
            probed = payload[0]
        self._first_is_probs, probed_sigmoid = probed
        # Scoring convention: trust the model config when it is explicit, else the probe.
        config_sigmoid = _sigmoid_scoring_from_config(teacher)
        self._sigmoid_scoring = probed_sigmoid if config_sigmoid is None else config_sigmoid
        self._active = False
        self._supervision_mask = None
        self._recorded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._self_recorded: dict[str, torch.Tensor] = {}
        self._stats: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._hook_handles = []
        for name in self.router_names:
            self._hook_handles.append(teacher_routers[name].register_forward_hook(self._record_hook(name)))
            self._hook_handles.append(student_routers[name].register_forward_hook(self._replay_hook(name)))
        logger.info(
            "Attached routing replay to %d MoE routers (mode=%s, router first output: %s, scoring: %s).",
            len(self.router_names),
            mode,
            "probabilities" if self._first_is_probs else "logits",
            "sigmoid" if self._sigmoid_scoring else "softmax",
        )

    def begin_batch(self, *, replay_active: bool, supervision_mask: torch.Tensor | None = None) -> None:
        """Start a new teacher->student batch; must run before the teacher forward.

        ``supervision_mask`` is a bool tensor over token positions (any shape that
        flattens to the router's token count, e.g. ``labels != IGNORE_INDEX``) used to
        restrict flip metrics to loss-active tokens.
        """

        self._active = replay_active
        self._supervision_mask = None if supervision_mask is None else supervision_mask.reshape(-1)
        self._recorded.clear()
        self._self_recorded.clear()
        self._stats.clear()

    def metrics(self) -> dict[str, torch.Tensor]:
        """Aggregate flip statistics over the routers seen since ``begin_batch``."""

        if not self._stats:
            return {}
        flipped = torch.stack([stats[0] for stats in self._stats.values()])
        jaccard = torch.stack([stats[1] for stats in self._stats.values()])
        valid = torch.stack([stats[2] for stats in self._stats.values()])
        total = valid.sum().clamp_min(1)
        return {
            "routing_flip_rate": flipped.sum() / total,
            "routing_jaccard": jaccard.sum() / total,
            "routing_flipped_layers_per_token": flipped.sum() / valid[0].clamp_min(1),
        }

    def remove(self) -> None:
        """Detach all hooks and clear state; call after training before any free forward."""

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._active = False
        self._supervision_mask = None
        self._recorded.clear()
        self._self_recorded.clear()
        self._stats.clear()

    def _record_hook(self, name: str):
        def hook(module, args, output):
            _, scores, indices = output
            self._recorded[name] = (indices.detach(), scores.detach())

        return hook

    def _replay_hook(self, name: str):
        def hook(module, args, output):
            first, own_scores, own_indices = output
            recorded = self._recorded.get(name)
            if recorded is not None:
                teacher_indices, teacher_scores = recorded
                if teacher_indices.shape != own_indices.shape:
                    raise RuntimeError(
                        f"Teacher/student routing shape mismatch for {name}: "
                        f"{tuple(teacher_indices.shape)} vs {tuple(own_indices.shape)} — "
                        "teacher and student must see the identical batch."
                    )
                self._update_stats(name, own_indices, teacher_indices)
            elif self._active:
                raise RuntimeError(
                    f"Routing replay is active but no teacher routing was recorded for {name}; "
                    "the teacher forward must precede the student forward each batch."
                )
            if self._active:
                # recorded is non-None here: the elif above raised when active without it.
                teacher_indices, teacher_scores = recorded
                if self.mode == "full":
                    return (first, teacher_scores.to(own_scores.dtype), teacher_indices)
                indices = teacher_indices
            else:
                # Self-pinning: even under free routing, the expert-dispatch SHAPES must be
                # identical across a gradient-checkpoint recompute — kernel-level float
                # nondeterminism (atomic index_add_ expert accumulation, Triton scans) can
                # flip a top-k boundary on recompute, changing dispatch shapes and tripping
                # torch's saved-tensor metadata check and failing the step.
                # Pin the step's first-pass indices and route BOTH passes through the same
                # gather+renorm ops so the recomputed op sequence matches the saved one.
                # Plain no-grad forwards (eval) are never recomputed: keep native output.
                if not torch.is_grad_enabled():
                    return None
                pinned = self._self_recorded.get(name)
                if pinned is None or pinned.shape != own_indices.shape:
                    pinned = own_indices.detach()
                    self._self_recorded[name] = pinned
                indices = pinned
            # Reconstruct gate weights from the router's own first output; the router is
            # never re-invoked, so its e_score_correction_bias (sigmoid archs) needs no
            # handling here: the bias shifts top-k SELECTION only, and every index used
            # below (recorded teacher, own, or pinned) came from a native router forward
            # that already applied it — the WEIGHTS are bias-free by the router's math.
            if self._first_is_probs:
                scores = first.float()
            elif self._sigmoid_scoring:
                # Sigmoid archs (minimax_m2): gathered sigmoids renormalized by their sum.
                scores = torch.sigmoid(first.float())
            else:
                scores = torch.softmax(first.float(), dim=-1)
            gathered = scores.gather(-1, indices)
            if getattr(module, "norm_topk_prob", True):
                gathered = gathered / gathered.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            return (first, gathered.to(own_scores.dtype), indices)

        return hook

    @torch.no_grad()
    def _update_stats(self, name: str, student_indices: torch.Tensor, teacher_indices: torch.Tensor) -> None:
        flips = (student_indices.sort(dim=-1).values != teacher_indices.sort(dim=-1).values).any(dim=-1)
        intersection = (
            (student_indices.unsqueeze(-1) == teacher_indices.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float()
        )
        top_k = student_indices.shape[-1]
        jaccard = intersection / (2 * top_k - intersection)
        if self._supervision_mask is not None:
            if self._supervision_mask.numel() != flips.numel():
                raise RuntimeError(
                    f"Supervision mask has {self._supervision_mask.numel()} positions but the router "
                    f"saw {flips.numel()} tokens."
                )
            mask = self._supervision_mask.to(flips.device)
            valid = mask.sum().float()
            flipped = flips[mask].sum().float()
            jaccard_sum = jaccard[mask].sum()
        else:
            valid = torch.tensor(float(flips.numel()), device=flips.device)
            flipped = flips.sum().float()
            jaccard_sum = jaccard.sum()
        # Keyed overwrite keeps gradient-checkpointing recomputation idempotent.
        self._stats[name] = (flipped, jaccard_sum, valid)
