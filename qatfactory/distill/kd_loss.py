"""Self-distillation loss for QAD: temperature-scaled forward KL + optional hard CE.

The teacher (frozen, full precision) and student (fake-quantized) produce logits over
the same batch. We match the student to the teacher over the supervised, next-token
positions only (``labels != IGNORE_INDEX``). Softmax/KL run in FP32 and are chunked over
tokens to bound peak memory on the large vocabulary.

Optionally (``compute_metrics=True``) the same chunk loop also emits cheap teacher/student
diagnostics (forward/reverse KL, entropies, top-k agreement) as token-sums, reusing the
already-materialized logits so there is no extra model forward.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


IGNORE_INDEX = -100


@dataclass
class DistillLossOutput:
    loss: torch.Tensor  # combined, per-token mean (differentiable)
    distill_loss: torch.Tensor  # temperature-KL component (detached)
    ce_loss: torch.Tensor  # hard-label component (detached)
    num_tokens: torch.Tensor  # supervised token count
    metrics: dict[str, torch.Tensor]  # optional diagnostics as token-SUMS (detached)


# Row-block size for the metric reductions. Metrics are exact token-SUMS, so they are
# accumulated over small row blocks; this bounds the peak of the full-vocab FP32 temporaries
# (log-softmax/prob tensors) that would otherwise scale with the whole loss chunk and, at long
# sequence lengths, dominate the logging-step memory peak and OOM.
_METRICS_ROW_BLOCK = 512


# Streamed Qwen3.5 loss operates on a 248K vocabulary while the model activations are
# already near the long-context memory limit. Keep its diagnostic temporaries to ~32 rows
# at a time even when the output-head loss chunk is larger.
_STREAMED_METRICS_ROW_BLOCK = 32


def _agreement_block(
    student_rows: torch.Tensor,
    teacher_rows: torch.Tensor,
    ks: list[int],
    max_k: int,
) -> dict[str, torch.Tensor]:
    """Per-row-block token-SUMS for the diagnostics (see :func:`_agreement_metrics`)."""

    teacher_logp = F.log_softmax(teacher_rows, dim=-1)
    student_logp = F.log_softmax(student_rows, dim=-1)
    teacher_p = teacher_logp.exp()
    student_p = student_logp.exp()

    metrics = {
        "fwd_kl": (teacher_p * (teacher_logp - student_logp)).sum(),
        "rev_kl": (student_p * (student_logp - teacher_logp)).sum(),
        "teacher_entropy": -(teacher_p * teacher_logp).sum(),
        "student_entropy": -(student_p * student_logp).sum(),
    }

    student_top1 = student_rows.argmax(dim=-1)
    teacher_top1 = teacher_rows.argmax(dim=-1)
    metrics["top1_agree"] = (student_top1 == teacher_top1).sum().float()

    if ks:
        teacher_topk = teacher_rows.topk(max_k, dim=-1).indices
        student_topk = student_rows.topk(max_k, dim=-1).indices
        for k in ks:
            s_in_t = (teacher_topk[:, :k] == student_top1.unsqueeze(-1)).any(dim=-1)
            t_in_s = (student_topk[:, :k] == teacher_top1.unsqueeze(-1)).any(dim=-1)
            metrics[f"s_in_t_top{k}"] = s_in_t.sum().float()
            metrics[f"t_in_s_top{k}"] = t_in_s.sum().float()
            # Top-k set Jaccard: with |S|=|T|=k, |S∪T| = 2k - |S∩T|, so we only need the
            # per-token intersection count. O(N*k^2) index comparisons, reusing the top-k
            # indices already materialized above (no extra top-k / forward).
            intersection = (
                (student_topk[:, :k].unsqueeze(2) == teacher_topk[:, :k].unsqueeze(1))
                .any(dim=2)
                .sum(dim=1)
                .float()
            )
            metrics[f"jaccard_top{k}"] = (intersection / (2 * k - intersection)).sum()
    return metrics


def _agreement_metrics(
    student_chunk: torch.Tensor,
    teacher_chunk: torch.Tensor,
    top_k: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Cheap, T=1 teacher/student diagnostics for one FP32 logits chunk.

    Everything is a token-SUM (so callers can token-weight across chunks/batches and
    all-reduce across ranks before taking a single mean). Computed under no-grad from the
    same materialized logits, so the cost is a handful of vocab reductions with no extra
    model forward. The reductions run over small row blocks (:data:`_METRICS_ROW_BLOCK`) and
    are summed, which is exact for token-sums and caps the extra full-vocab FP32 memory peak.
    Metrics:

    - ``fwd_kl`` / ``rev_kl``: KL(teacher||student) / KL(student||teacher), nats.
    - ``teacher_entropy`` / ``student_entropy``: predictive entropy, nats.
    - ``top1_agree``: argmax match rate (symmetric, so reported once).
    - ``s_in_t_top{k}`` / ``t_in_s_top{k}`` for each k>1: student top-1 in teacher's top-k
      (and the reverse). Direction matters for k>1, so both are reported.
    - ``jaccard_top{k}`` for each k>1: Jaccard index of the student/teacher top-k token
      sets, ``|S∩T| / |S∪T|`` (symmetric, reported once). Jaccard@1 is identical to
      ``top1_agree``, so k=1 is skipped.
    """

    ks = sorted({k for k in top_k if k > 1})
    max_k = ks[-1] if ks else 1
    totals: dict[str, torch.Tensor] = {}
    for start in range(0, student_chunk.shape[0], _METRICS_ROW_BLOCK):
        block = _agreement_block(
            student_chunk[start : start + _METRICS_ROW_BLOCK],
            teacher_chunk[start : start + _METRICS_ROW_BLOCK],
            ks,
            max_k,
        )
        for key, value in block.items():
            totals[key] = totals.get(key, 0) + value
    return totals


def _shift_and_select(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop the last position, shift labels, and gather supervised tokens."""

    student_logits = student_logits[..., :-1, :]
    teacher_logits = teacher_logits[..., :-1, :]
    labels = labels[..., 1:]
    flat_labels = labels.reshape(-1)
    valid = flat_labels != IGNORE_INDEX
    student = student_logits.reshape(-1, student_logits.shape[-1])[valid]
    teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1])[valid]
    targets = flat_labels[valid]
    return student, teacher, targets, valid.sum()


def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    distill_weight: float = 1.0,
    hard_label_weight: float = 0.0,
    logits_chunk_size: int = 2048,
    compute_metrics: bool = False,
    metrics_top_k: tuple[int, ...] = (1, 5, 10),
    loss_denominator: torch.Tensor | int | None = None,
) -> DistillLossOutput:
    student, teacher, targets, token_count = _shift_and_select(student_logits, teacher_logits, labels)
    if token_count == 0:
        zero = student_logits.sum() * 0
        return DistillLossOutput(zero, zero.detach(), zero.detach(), token_count, {})

    chunk_size = logits_chunk_size or student.shape[0]
    distill_sum = student.new_zeros((), dtype=torch.float32)
    ce_sum = student.new_zeros((), dtype=torch.float32)
    metric_sums: dict[str, torch.Tensor] = {}
    for start in range(0, student.shape[0], chunk_size):
        student_chunk = student[start : start + chunk_size].float()
        teacher_chunk = teacher[start : start + chunk_size].float()
        teacher_prob = F.softmax(teacher_chunk / temperature, dim=-1)
        student_log_prob = F.log_softmax(student_chunk / temperature, dim=-1)
        distill_sum = distill_sum + F.kl_div(student_log_prob, teacher_prob, reduction="sum") * temperature**2
        if hard_label_weight:
            ce_sum = ce_sum + F.cross_entropy(student_chunk, targets[start : start + chunk_size], reduction="sum")
        if compute_metrics:
            with torch.no_grad():
                for key, value in _agreement_metrics(student_chunk, teacher_chunk, metrics_top_k).items():
                    metric_sums[key] = metric_sums.get(key, 0) + value

    local_denominator = token_count.clamp_min(1)
    distill_mean = distill_sum / local_denominator
    ce_mean = ce_sum / local_denominator
    # `loss_denominator` is the supervised-token count of the whole gradient-accumulation
    # window, so each microbatch contributes its exact share of one global token mean instead
    # of an average of per-microbatch means (which over-weights short microbatches).
    denominator = (
        torch.as_tensor(loss_denominator, device=distill_sum.device).clamp_min(1)
        if loss_denominator is not None
        else local_denominator
    )
    combined = distill_weight * (distill_sum / denominator) + hard_label_weight * (ce_sum / denominator)
    metrics = {key: value.detach() for key, value in metric_sums.items()}
    return DistillLossOutput(combined, distill_mean.detach(), ce_mean.detach(), token_count, metrics)


def distill_hidden_states_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    temperature: float = 1.0,
    distill_weight: float = 1.0,
    hard_label_weight: float = 0.0,
    token_chunk_size: int = 128,
    compute_metrics: bool = False,
    metrics_top_k: tuple[int, ...] = (1, 5, 10),
    loss_denominator: torch.Tensor | int | None = None,
) -> DistillLossOutput:
    """Compute exact distillation without materializing sequence-length × vocabulary logits.

    Each supervised-token chunk projects through the frozen output head independently. The
    chunk body is activation-checkpointed, so its full-vocabulary softmax is recomputed during
    backward instead of retaining one such tensor for every token chunk. Diagnostics, when
    requested, are reduced to token-sums inside that same projection and therefore retain no
    full-vocabulary tensors between chunks.
    """

    shifted_labels = labels[..., 1:].reshape(-1)
    valid_indices = torch.nonzero(shifted_labels != IGNORE_INDEX, as_tuple=False).flatten()
    token_count = valid_indices.new_tensor(valid_indices.numel())
    if valid_indices.numel() == 0:
        zero = student_hidden.sum() * 0
        return DistillLossOutput(zero, zero.detach(), zero.detach(), token_count, {})

    student_flat = student_hidden[..., :-1, :].reshape(-1, student_hidden.shape[-1])
    teacher_flat = teacher_hidden[..., :-1, :].reshape(-1, teacher_hidden.shape[-1])
    chunk_size = token_chunk_size or valid_indices.numel()
    distill_sum = student_hidden.new_zeros((), dtype=torch.float32)
    ce_sum = student_hidden.new_zeros((), dtype=torch.float32)
    metric_sums: dict[str, torch.Tensor] = {}

    metric_keys = [
        "fwd_kl",
        "rev_kl",
        "teacher_entropy",
        "student_entropy",
        "top1_agree",
    ]
    for k in sorted({k for k in metrics_top_k if k > 1}):
        metric_keys.extend((f"s_in_t_top{k}", f"t_in_s_top{k}", f"jaccard_top{k}"))

    def chunk_loss(student_chunk, teacher_chunk, targets):
        student_logits = F.linear(student_chunk, output_weight).float()
        with torch.no_grad():
            teacher_logits = F.linear(teacher_chunk, output_weight).float()
            teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
        chunk_distill = F.kl_div(student_log_prob, teacher_prob, reduction="sum") * temperature**2
        chunk_ce = (
            F.cross_entropy(student_logits, targets, reduction="sum")
            if hard_label_weight
            else student_logits.sum() * 0
        )
        outputs = [chunk_distill, chunk_ce]
        if compute_metrics:
            with torch.no_grad():
                chunk_metrics = {}
                for metric_start in range(0, student_logits.shape[0], _STREAMED_METRICS_ROW_BLOCK):
                    block = _agreement_metrics(
                        student_logits[metric_start : metric_start + _STREAMED_METRICS_ROW_BLOCK],
                        teacher_logits[metric_start : metric_start + _STREAMED_METRICS_ROW_BLOCK],
                        metrics_top_k,
                    )
                    for key, value in block.items():
                        chunk_metrics[key] = chunk_metrics.get(key, 0) + value
            outputs.extend(chunk_metrics[key] for key in metric_keys)
        return tuple(outputs)

    for start in range(0, valid_indices.numel(), chunk_size):
        indices = valid_indices[start : start + chunk_size]
        student_chunk = student_flat.index_select(0, indices)
        teacher_chunk = teacher_flat.index_select(0, indices)
        targets = shifted_labels.index_select(0, indices)
        outputs = (
            checkpoint(
                chunk_loss,
                student_chunk,
                teacher_chunk,
                targets,
                use_reentrant=False,
            )
            if student_chunk.requires_grad
            else chunk_loss(student_chunk, teacher_chunk, targets)
        )
        chunk_distill, chunk_ce, *chunk_metrics = outputs
        distill_sum = distill_sum + chunk_distill
        ce_sum = ce_sum + chunk_ce
        for key, value in zip(metric_keys, chunk_metrics):
            metric_sums[key] = metric_sums.get(key, 0) + value

    local_denominator = token_count.clamp_min(1)
    distill_mean = distill_sum / local_denominator
    ce_mean = ce_sum / local_denominator
    # Same normalization as distill_loss: the gradient uses the accumulation window's token
    # count when the trainer provides it.
    denominator = (
        torch.as_tensor(loss_denominator, device=distill_sum.device).clamp_min(1)
        if loss_denominator is not None
        else local_denominator
    )
    combined = distill_weight * (distill_sum / denominator) + hard_label_weight * (ce_sum / denominator)
    metrics = {key: value.detach() for key, value in metric_sums.items()}
    return DistillLossOutput(combined, distill_mean.detach(), ce_mean.detach(), token_count, metrics)
