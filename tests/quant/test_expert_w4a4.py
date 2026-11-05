"""W4A4 over fused routed experts: activation scales, their lifecycle, and the wider gap."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from qatfactory.quant import apply_fake_quant, shared_base_teacher
from qatfactory.quant.formats.nvfp4 import (
    NVFP4_SCALE_DENOMINATOR,
    commit_nvfp4_input_scales,
    finalize_nvfp4_input_scales,
    load_nvfp4_runtime_state_dict,
    nvfp4_runtime_state_dict,
)
from qatfactory.quant.formats.nvfp4_experts import NVFP4ExpertLoRA

transformers_minimax = pytest.importorskip("transformers.models.minimax_m2.modeling_minimax_m2")
minimax_config = pytest.importorskip("transformers.models.minimax_m2.configuration_minimax_m2")


def _model(seed: int = 0):
    torch.manual_seed(seed)
    config = minimax_config.MiniMaxM2Config(
        vocab_size=64, hidden_size=32, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        num_local_experts=4, num_experts_per_tok=2, rotary_dim=8, tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = transformers_minimax.MiniMaxM2ForCausalLM(config)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.mul_(4.0)
        for layer in model.model.layers:
            layer.mlp.experts.gate_up_proj.mul_(4.0)
            layer.mlp.experts.down_proj.mul_(4.0)
    return model.eval()


def _experts(model) -> NVFP4ExpertLoRA:
    return model.model.layers[0].mlp.experts


def _instrument(model, *, w4a4: bool):
    return apply_fake_quant(
        model, [], lora_rank=4, lora_alpha=8,
        fused_runtime_scales=True, quantize_activations=w4a4,
    )


def _forward(model, tokens: int = 16, seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, model.config.vocab_size, (2, tokens), generator=generator)
    return model(input_ids=input_ids, use_cache=False).logits


def test_activation_quant_reaches_the_experts():
    model = _model()
    _instrument(model, w4a4=True)
    experts = _experts(model)

    assert experts._quantize_activations is True
    assert experts._track_input_amax is True
    model.train()
    _forward(model)
    # Both matmul inputs get their own static scale, bootstrapped on the first batch.
    assert experts._gate_up_input_amax_static is not None
    assert experts._down_input_amax_static is not None
    assert experts._gate_up_input_amax_static > 0
    assert experts._down_input_amax_static > 0


def test_w4a4_widens_the_gap_against_the_shared_base():
    """Quantizing activations must move the student further than weight-only does."""

    gaps = {}
    for name, w4a4 in (("w4a16", False), ("w4a4", True)):
        model = _model()
        _instrument(model, w4a4=w4a4)
        model.train()
        _forward(model)  # bootstrap the static scales
        model.eval()
        with torch.no_grad():
            with shared_base_teacher():
                teacher = _forward(model).float()
            student = _forward(model).float()
        gaps[name] = F.kl_div(
            F.log_softmax(student, dim=-1), F.log_softmax(teacher, dim=-1),
            log_target=True, reduction="batchmean",
        ).item()

    assert gaps["w4a16"] > 0
    # The ratio depends on the torch build (about 1.9 on x86 CPU torch 2.15); an activation quantizer
    # that does nothing gives exactly 1.
    assert gaps["w4a4"] > 1.5 * gaps["w4a16"], gaps


def test_static_scale_is_frozen_within_a_step_and_republished_between_them():
    model = _model()
    _instrument(model, w4a4=True)
    experts = _experts(model)
    model.train()
    _forward(model)

    frozen = experts._gate_up_input_amax_static.clone()
    # A larger batch grows the running max but must not move the in-step static scale.
    with torch.no_grad():
        experts._gate_up_input_amax = experts._gate_up_input_amax * 10
    _forward(model)
    torch.testing.assert_close(experts._gate_up_input_amax_static, frozen, atol=0, rtol=0)

    commit_nvfp4_input_scales(model)
    assert experts._gate_up_input_amax_static > frozen


def test_finalized_scales_cover_both_expert_projections():
    model = _model()
    converted = _instrument(model, w4a4=True)
    model.train()
    _forward(model)

    scales = finalize_nvfp4_input_scales(model)
    prefix = converted.experts[0]
    assert set(scales) == {f"{prefix}.w13.input_scale", f"{prefix}.w2.input_scale"}
    experts = _experts(model)
    expected = experts._gate_up_input_amax / NVFP4_SCALE_DENOMINATOR
    torch.testing.assert_close(scales[f"{prefix}.w13.input_scale"], expected, atol=0, rtol=0)


def test_runtime_state_round_trips_the_expert_scales():
    model = _model()
    _instrument(model, w4a4=True)
    model.train()
    _forward(model)
    experts = _experts(model)
    saved = nvfp4_runtime_state_dict(model)

    restored = _model()
    _instrument(restored, w4a4=True)
    load_nvfp4_runtime_state_dict(restored, saved)

    for attribute in ("_gate_up_input_amax", "_down_input_amax"):
        torch.testing.assert_close(
            getattr(_experts(restored), attribute), getattr(experts, attribute), atol=0, rtol=0
        )
        # A resumed run must be able to quantize before its first commit.
        assert getattr(_experts(restored), f"{attribute}_static") is not None


def test_weight_only_run_leaves_activations_alone():
    model = _model()
    _instrument(model, w4a4=False)
    experts = _experts(model)
    model.train()
    _forward(model)

    assert experts._quantize_activations is False
    assert experts._gate_up_input_amax_static is None
    # The running max is still tracked so a weight-only run can be exported or resumed.
    assert experts._gate_up_input_amax is not None
