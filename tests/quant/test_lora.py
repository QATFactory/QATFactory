"""LoRA-QAD: adapter mechanics, freezing, shared-base teacher-mode, and export merge."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from qatfactory.quant import (
    apply_fake_quant,
    is_lora_quantized,
    merge_lora_weight,
    shared_base_teacher,
    shared_base_teacher_enabled,
)
from qatfactory.quant.formats.mxfp4 import MXFP4Linear, fake_quantize_mxfp4
from qatfactory.quant.formats.nvfp4 import NVFP4Linear, fake_quantize_nvfp4


class _Attn(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(dim, dim, bias=False))


class _MLP(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, nn.Linear(dim, dim, bias=False))


class _Config:
    def get_text_config(self):
        return self

    num_experts = None
    tie_word_embeddings = False


class _Model(nn.Module):
    """A stand-in whose module names hit the fused-family suffixes (``self_attn``/``mlp``)."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.self_attn = _Attn(dim)
        self.mlp = _MLP(dim)
        self.config = _Config()


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _lora_model(quant_format="nvfp4", rank=4, alpha=8, fused=True):
    model = _Model()
    converted = apply_fake_quant(
        model,
        TARGETS,
        quant_format=quant_format,
        exclude_prefixes=(),
        fused_runtime_scales=fused,
        lora_rank=rank,
        lora_alpha=alpha,
    )
    return model, converted


def test_init_delta_is_zero_and_effective_weight_is_base():
    model, converted = _lora_model()
    assert converted.lora is True
    qp = model.self_attn.q_proj
    assert is_lora_quantized(qp)
    assert torch.count_nonzero(qp.lora_B) == 0  # B starts at zero
    assert torch.count_nonzero(qp.lora_A) > 0  # A is Kaiming-initialised
    torch.testing.assert_close(qp.lora_delta(), torch.zeros_like(qp.weight))
    torch.testing.assert_close(qp._effective_weight(), qp.weight)


def test_only_adapters_are_trainable():
    model, converted = _lora_model()
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == converted.trainable_parameters
    assert all(name.endswith((".lora_A", ".lora_B")) for name in trainable)
    for module in model.modules():
        if isinstance(module, NVFP4Linear):
            assert module.weight.requires_grad is False


def test_student_starts_at_pure_ptq_and_grad_reaches_only_adapters():
    model, _ = _lora_model()
    qp = model.self_attn.q_proj
    x = torch.randn(3, 32)

    expected = F.linear(x, fake_quantize_nvfp4(qp.weight, tensor_amax=qp._amax_group.tensor_amax()), None)
    torch.testing.assert_close(qp(x), expected)

    qp(x).sum().backward()
    assert qp.lora_A.grad is not None and qp.lora_B.grad is not None
    assert qp.weight.grad is None


def test_shared_base_teacher_is_plain_full_precision():
    model, _ = _lora_model()
    qp = model.self_attn.q_proj
    with torch.no_grad():
        qp.lora_B.copy_(torch.randn_like(qp.lora_B))  # make the adapter non-trivial
    x = torch.randn(3, 32)

    with shared_base_teacher():
        torch.testing.assert_close(qp(x), F.linear(x, qp.weight, None))
    # Student now differs from the base because B != 0.
    assert not torch.allclose(qp(x), F.linear(x, qp.weight, None))


def test_shared_base_teacher_context_is_exception_safe():
    model, _ = _lora_model()
    with pytest.raises(RuntimeError, match="teacher blew up"):
        with shared_base_teacher():
            assert shared_base_teacher_enabled()
            raise RuntimeError("teacher blew up")
    assert not shared_base_teacher_enabled()
    qp = model.self_attn.q_proj
    x = torch.randn(3, 32)
    assert not torch.equal(qp(x), F.linear(x, qp.weight, None))


def test_fused_family_amax_tracks_the_merged_weight():
    model, _ = _lora_model()
    qp = model.self_attn.q_proj
    group = qp._amax_group
    group.tensor_amax_value = None  # force the non-FSDP recompute path
    base_amax = torch.stack([m.weight.detach().abs().amax() for m in group.members]).amax()

    with torch.no_grad():
        qp.lora_B[0, 0] = 1000.0  # blow up one merged-weight entry
    merged_amax = group.tensor_amax()
    assert merged_amax > base_amax
    assert merged_amax >= qp._effective_weight().detach().abs().amax() - 1e-3


def test_mxfp4_lora_forward_quantizes_merged_weight():
    model, _ = _lora_model(quant_format="mxfp4", fused=False)
    gp = model.mlp.gate_proj
    assert isinstance(gp, MXFP4Linear)
    with torch.no_grad():
        gp.lora_B.copy_(torch.randn_like(gp.lora_B) * 0.1)
    x = torch.randn(3, 32)
    expected = F.linear(x, fake_quantize_mxfp4(gp._effective_weight()), None)
    torch.testing.assert_close(gp(x), expected)


def test_q4_k_lora_is_rejected():
    model = _Model()
    with pytest.raises(ValueError, match="does not support LoRA"):
        apply_fake_quant(
            model,
            TARGETS,
            quant_format="q4_k",
            exclude_prefixes=(),
            fused_runtime_scales=False,
            lora_rank=4,
            lora_alpha=8,
        )


def test_merge_lora_weight_matches_manual_fold():
    torch.manual_seed(0)
    weight = torch.randn(16, 32, dtype=torch.bfloat16)
    lora_a = torch.randn(4, 32, dtype=torch.bfloat16)
    lora_b = torch.randn(16, 4, dtype=torch.bfloat16)
    scaling = 2.5
    merged = merge_lora_weight(weight, lora_a, lora_b, scaling)
    expected = weight + (scaling * (lora_b @ lora_a)).to(weight.dtype)
    assert merged.dtype == torch.bfloat16
    torch.testing.assert_close(merged, expected, atol=0, rtol=0)


def test_export_merge_is_bit_identical_to_the_training_forward():
    """The exporter must fold the adapter exactly where training folded it.

    Merging in a wider dtype than the training forward used would quantize a weight the
    student never saw, so the deployed lattice would drift from the trained one.
    """

    model, _ = _lora_model()
    qp = model.self_attn.q_proj
    with torch.no_grad():
        qp.lora_B.normal_(0, 0.1)

    exported = merge_lora_weight(qp.weight.detach(), qp.lora_A.detach(), qp.lora_B.detach(), qp.lora_scaling)
    torch.testing.assert_close(exported, qp._effective_weight().detach(), atol=0, rtol=0)


class _StubStore:
    def __init__(self, tensors):
        self._tensors = tensors

    def keys(self):
        return set(self._tensors)

    def get_tensor(self, key):
        return self._tensors[key]


def test_export_read_target_weight_folds_adapter():
    from qatfactory.quant.export.nvfp4_vllm import _read_target_weight

    weight = torch.randn(16, 32, dtype=torch.bfloat16)
    lora_a = torch.randn(4, 32, dtype=torch.bfloat16)
    lora_b = torch.randn(16, 4, dtype=torch.bfloat16)
    store = _StubStore(
        {"m.weight": weight, "m.lora_A": lora_a, "m.lora_B": lora_b}
    )
    # No LoRA -> raw weight; with scaling -> merged.
    torch.testing.assert_close(_read_target_weight(store, "m", None), weight)
    torch.testing.assert_close(
        _read_target_weight(store, "m", 2.0), merge_lora_weight(weight, lora_a, lora_b, 2.0)
    )
