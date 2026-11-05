"""QADTrainer mechanics: shared-base teacher switching and accumulation-window normalization."""

import pytest
import torch
from torch import nn

from qatfactory.distill.kd_loss import distill_loss
from qatfactory.quant.formats.lora import shared_base_teacher_enabled
from qatfactory.trainer.qad_trainer import QADTrainer


class _ModeTrackingModel(nn.Module):
    """Records the mode and teacher-context state seen by each forward."""

    def __init__(self, fail: bool = False):
        super().__init__()
        self.child = nn.Linear(2, 2)
        self.fail = fail
        self.observed = []

    def forward(self, **kwargs):
        self.observed.append((self.training, self.child.training, shared_base_teacher_enabled()))
        if self.fail:
            raise RuntimeError("teacher forward exploded")
        return "teacher-logits"


@pytest.mark.parametrize("fail", [False, True])
def test_shared_base_teacher_forward_restores_modes_and_context(fail):
    model = _ModeTrackingModel(fail=fail)
    model.train()
    model.child.eval()  # a deliberately mixed starting state

    if fail:
        with pytest.raises(RuntimeError, match="teacher forward exploded"):
            QADTrainer._shared_base_teacher_forward(model, {})
    else:
        assert QADTrainer._shared_base_teacher_forward(model, {}) == "teacher-logits"

    # The teacher pass runs in eval mode inside the shared-base context...
    assert model.observed == [(False, False, True)]
    # ...and every module's original mode is restored, even when the forward raised.
    assert model.training is True and model.child.training is False
    assert not shared_base_teacher_enabled()


def test_scale_gradient_changes_only_the_gradient():
    value = torch.tensor(3.0, requires_grad=True)
    scaled = QADTrainer._scale_gradient(value * 1.0, 4)

    assert scaled.item() == pytest.approx(3.0)
    scaled.backward()
    assert value.grad.item() == pytest.approx(4.0)


VOCAB = 7
SEQ_LEN = 9
# Supervised-token counts chosen to differ per row, which is what separates a true global
# token mean from an average of per-microbatch means.
SUPERVISED_PER_ROW = (8, 2, 6, 1, 7, 3)


def _uneven_batch(generator):
    rows = len(SUPERVISED_PER_ROW)
    teacher = torch.randn(rows, SEQ_LEN, VOCAB, generator=generator)
    student = torch.randn(rows, SEQ_LEN, VOCAB, generator=generator)
    labels = torch.randint(0, VOCAB, (rows, SEQ_LEN), generator=generator)
    for row, supervised in enumerate(SUPERVISED_PER_ROW):
        # The loss shifts by one, so a row of length L has at most L-1 supervised tokens.
        labels[row, : SEQ_LEN - supervised] = -100
    return student, teacher, labels


@pytest.mark.parametrize("microbatches", [(3, 3), (1, 5), (2, 1, 3), (1, 1, 1, 1, 1, 1)])
def test_accumulated_microbatches_match_the_full_batch_token_mean(microbatches):
    """A gradient-accumulation window must equal one big batch, not an average of means."""

    generator = torch.Generator().manual_seed(0)
    student_base, teacher, labels = _uneven_batch(generator)

    student = student_base.clone().requires_grad_(True)
    reference = distill_loss(student, teacher, labels).loss
    reference.backward()
    reference_grad = student.grad.clone()
    window_tokens = distill_loss(student_base, teacher, labels).num_tokens

    sharded = student_base.clone().requires_grad_(True)
    parts, cursor = [], 0
    for size in microbatches:
        rows = slice(cursor, cursor + size)
        parts.append(
            distill_loss(
                sharded[rows], teacher[rows], labels[rows], loss_denominator=window_tokens
            ).loss
        )
        cursor += size
    accumulated = torch.stack(parts).sum()
    accumulated.backward()

    torch.testing.assert_close(accumulated, reference, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(sharded.grad, reference_grad, atol=1e-6, rtol=1e-6)


def test_averaging_microbatch_means_is_what_the_denominator_fixes():
    """Guard the premise: without the window denominator, uneven microbatches disagree."""

    generator = torch.Generator().manual_seed(0)
    student, teacher, labels = _uneven_batch(generator)

    reference = distill_loss(student, teacher, labels).loss
    per_microbatch_means = torch.stack(
        [distill_loss(student[row : row + 1], teacher[row : row + 1], labels[row : row + 1]).loss
         for row in range(student.shape[0])]
    ).mean()

    assert not torch.isclose(per_microbatch_means, reference, atol=1e-4)


def test_translate_fused_expert_target_to_its_on_disk_module():
    ondisk = {
        "model.language_model.layers.3.mlp.experts.gate_up_proj",
        "model.language_model.layers.3.mlp.experts.down_proj",
        "model.language_model.layers.3.self_attn.q_proj.weight",
    }

    translate = QADTrainer._translate_to_ondisk
    assert (
        translate("model.layers.3.mlp.experts.7.w1", ondisk)
        == "model.language_model.layers.3.mlp.experts.7.w1"
    )
    assert (
        translate("model.layers.3.mlp.experts.7.w2", ondisk)
        == "model.language_model.layers.3.mlp.experts.7.w2"
    )
    assert (
        translate("model.layers.3.self_attn.q_proj", ondisk)
        == "model.language_model.layers.3.self_attn.q_proj"
    )
    with pytest.raises(RuntimeError, match="Could not"):
        translate("model.layers.9.mlp.experts.0.w1", ondisk)
    # The fused experts module itself (owner of the per-layer activation scales).
    assert translate("model.layers.3.mlp.experts", ondisk) == "model.language_model.layers.3.mlp.experts"
    with pytest.raises(RuntimeError, match="Could not"):
        translate("model.layers.9.mlp.experts", ondisk)


def test_input_scale_keys_resolve_to_their_converted_module():
    # Linears emit `<module>.input_scale`; fused experts emit `<module>.w13.input_scale` and
    # `<module>.w2.input_scale`. Both must map back onto the converted module that owns them.
    name_map = {
        "model.layers.0.self_attn.q_proj": "model.language_model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.experts": "model.language_model.layers.0.mlp.experts",
    }
    owner = QADTrainer._owning_converted_module
    assert owner("model.layers.0.self_attn.q_proj.input_scale", name_map) == "model.layers.0.self_attn.q_proj"
    assert owner("model.layers.0.mlp.experts.w13.input_scale", name_map) == "model.layers.0.mlp.experts"
    assert owner("model.layers.0.mlp.experts.w2.input_scale", name_map) == "model.layers.0.mlp.experts"
    with pytest.raises(KeyError, match="does not belong"):
        owner("model.layers.1.mlp.experts.w2.input_scale", name_map)


def test_read_ondisk_keys_includes_fused_expert_tensors(tmp_path):
    """A full-parameter MoE save stores experts as bare 3D tensors; the name map must see them."""
    from safetensors.torch import save_file

    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
            "model.layers.0.mlp.experts.gate_up_proj": torch.zeros(2, 4, 2),
            "model.layers.0.mlp.experts.down_proj": torch.zeros(2, 2, 2),
            "model.layers.0.mlp.gate.e_score_correction_bias": torch.zeros(2),
        },
        str(tmp_path / "model.safetensors"),
    )
    keys = QADTrainer._read_ondisk_weight_keys(str(tmp_path))
    assert keys == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    }
    target = QADTrainer._translate_to_ondisk("model.layers.0.mlp.experts.1.w3", keys)
    assert target == "model.layers.0.mlp.experts.1.w3"


def test_adapter_names_take_the_released_spelling_except_routed_experts():
    """An adapters-only LoRA save renames dense targets like save_pretrained would, so the
    exporter and merge_lora_bf16.py find them next to the released checkpoint."""
    ondisk = {
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        # Qwen3.5's released MTP head reuses the backbone's layer-qualified suffixes.
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
    }
    translate = QADTrainer._adapter_translator(ondisk)
    assert translate("model.layers.0.self_attn.q_proj") == "model.language_model.layers.0.self_attn.q_proj"
    assert translate("model.layers.0.linear_attn.in_proj_qkv") == "model.language_model.layers.0.linear_attn.in_proj_qkv"
    assert translate("model.layers.0.mlp.gate_proj") == "model.language_model.layers.0.mlp.gate_proj"
    # Routed experts stay in the in-memory fused layout.
    assert translate("model.layers.0.mlp.experts") == "model.layers.0.mlp.experts"
    assert translate("model.layers.0.mlp.experts.3.w1") == "model.layers.0.mlp.experts.3.w1"

    owners = {
        "model.layers.0.self_attn.q_proj": translate("model.layers.0.self_attn.q_proj"),
        "model.layers.0.mlp.experts": translate("model.layers.0.mlp.experts"),
    }
    rename = QADTrainer._rename_under_owner
    assert rename("model.layers.0.self_attn.q_proj.lora_A", owners) == "model.language_model.layers.0.self_attn.q_proj.lora_A"
    assert rename("model.layers.0.mlp.experts.gate_up_lora_B", owners) == "model.layers.0.mlp.experts.gate_up_lora_B"
    assert rename("model.norm.weight", owners) == "model.norm.weight"
