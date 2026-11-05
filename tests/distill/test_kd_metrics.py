import torch

from qatfactory.distill import kd_loss
from qatfactory.distill.kd_loss import _agreement_block, _agreement_metrics, distill_loss


def _one_hot_logits(rows: list[list[int]], vocab: int) -> torch.Tensor:
    """Logits whose per-row descending argsort equals the given index lists.

    Each row lists token ids in the intended top order; earlier ids get higher logits so
    ``topk`` returns exactly that prefix.
    """

    logits = torch.zeros(len(rows), vocab)
    for r, ids in enumerate(rows):
        for rank, tok in enumerate(ids):
            logits[r, tok] = float(len(ids) - rank)
    return logits


def test_jaccard_topk_matches_hand_computation():
    vocab = 20
    # Row 0: identical top-5 sets  -> J = 5/5 = 1.
    # Row 1: 2 shared of 5 (ids 0,1) -> intersection 2, union 8 -> J = 2/8 = 0.25.
    student = _one_hot_logits([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], vocab)
    teacher = _one_hot_logits([[0, 1, 2, 3, 4], [0, 1, 10, 11, 12]], vocab)

    metrics = _agreement_metrics(student, teacher, top_k=(1, 5))

    # Metric is a token-SUM of per-token Jaccard.
    assert torch.isclose(metrics["jaccard_top5"], torch.tensor(1.0 + 0.25))
    # Jaccard@1 is not emitted (identical to top1_agree).
    assert "jaccard_top1" not in metrics
    # Symmetric: swapping student/teacher yields the same sum.
    swapped = _agreement_metrics(teacher, student, top_k=(1, 5))
    assert torch.isclose(swapped["jaccard_top5"], metrics["jaccard_top5"])


def test_jaccard_flows_through_distill_loss_as_token_sum():
    vocab = 32
    seq = 6
    torch.manual_seed(0)
    student_logits = torch.randn(1, seq, vocab)
    teacher_logits = torch.randn(1, seq, vocab)
    labels = torch.full((1, seq), -100)
    labels[0, 2:] = torch.randint(0, vocab, (seq - 2,))  # supervise a few positions

    out = distill_loss(
        student_logits,
        teacher_logits,
        labels,
        compute_metrics=True,
        metrics_top_k=(1, 5, 10),
    )

    for k in (5, 10):
        key = f"jaccard_top{k}"
        assert key in out.metrics
        per_token_mean = out.metrics[key] / out.num_tokens
        assert 0.0 <= per_token_mean.item() <= 1.0


def test_row_block_accumulation_matches_single_block():
    """Sub-chunking the metric reductions is exact for token-SUMS (memory-only change)."""

    vocab = 200
    rows = kd_loss._METRICS_ROW_BLOCK * 2 + 37  # spans several row blocks
    torch.manual_seed(1)
    student = torch.randn(rows, vocab)
    teacher = torch.randn(rows, vocab)

    ks = [5, 10]
    single = _agreement_block(student, teacher, ks, max_k=10)
    chunked = _agreement_metrics(student, teacher, top_k=(1, 5, 10))

    assert single.keys() == chunked.keys()
    for key in single:
        assert torch.allclose(single[key], chunked[key], atol=1e-4), key
