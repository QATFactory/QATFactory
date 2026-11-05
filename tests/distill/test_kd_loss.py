import torch
import torch.nn.functional as F

from qatfactory.distill.kd_loss import IGNORE_INDEX, distill_hidden_states_loss, distill_loss


def test_hidden_state_distillation_matches_materialized_logits_gradient_and_metrics():
    torch.manual_seed(0)
    base_student = torch.randn(2, 5, 7)
    teacher = torch.randn(2, 5, 7)
    output_weight = torch.randn(11, 7)
    labels = torch.tensor(
        [
            [IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5],
            [IGNORE_INDEX, 2, 1, IGNORE_INDEX, 6],
        ]
    )

    materialized_student = base_student.clone().requires_grad_(True)
    materialized = distill_loss(
        F.linear(materialized_student, output_weight),
        F.linear(teacher, output_weight),
        labels,
        temperature=1.7,
        distill_weight=0.8,
        hard_label_weight=0.3,
        logits_chunk_size=2,
        compute_metrics=True,
        metrics_top_k=(1, 5, 10),
    )
    materialized.loss.backward()

    streamed_student = base_student.clone().requires_grad_(True)
    streamed = distill_hidden_states_loss(
        streamed_student,
        teacher,
        labels,
        output_weight,
        temperature=1.7,
        distill_weight=0.8,
        hard_label_weight=0.3,
        token_chunk_size=2,
        compute_metrics=True,
        metrics_top_k=(1, 5, 10),
    )
    streamed.loss.backward()

    torch.testing.assert_close(streamed.loss, materialized.loss)
    torch.testing.assert_close(streamed.distill_loss, materialized.distill_loss)
    torch.testing.assert_close(streamed.ce_loss, materialized.ce_loss)
    torch.testing.assert_close(streamed_student.grad, materialized_student.grad)
    assert streamed.num_tokens.item() == materialized.num_tokens.item()
    assert streamed.metrics.keys() == materialized.metrics.keys()
    for key in streamed.metrics:
        torch.testing.assert_close(streamed.metrics[key], materialized.metrics[key])
