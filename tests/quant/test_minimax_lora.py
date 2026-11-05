"""LoRA-QAD over fused routed experts and block-FP8 frozen bases (MiniMax-M2 layout)."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from qatfactory.quant import apply_fake_quant, dequantize_fp8_blocks, shared_base_teacher
from qatfactory.quant.formats.nvfp4 import NVFP4Linear, fake_quantize_nvfp4
from qatfactory.quant.formats.nvfp4_experts import NVFP4ExpertLoRA

transformers_minimax = pytest.importorskip("transformers.models.minimax_m2.modeling_minimax_m2")
minimax_config = pytest.importorskip("transformers.models.minimax_m2.configuration_minimax_m2")

HIDDEN = 32
INTERMEDIATE = 16
NUM_EXPERTS = 4
TOP_K = 2


def _tiny_config(**overrides):
    settings = {
        "vocab_size": 64,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "rotary_dim": 8,
        "tie_word_embeddings": False,
    }
    settings.update(overrides)
    config = minimax_config.MiniMaxM2Config(**settings)
    config._attn_implementation = "eager"
    return config


def _tiny_experts():
    experts = transformers_minimax.MiniMaxM2Experts(_tiny_config())
    with torch.no_grad():
        experts.gate_up_proj.normal_(0, 0.1)
        experts.down_proj.normal_(0, 0.1)
    return experts


def _routing(num_tokens: int = 6):
    torch.manual_seed(0)
    hidden = torch.randn(num_tokens, HIDDEN)
    index = torch.stack(
        [torch.randperm(NUM_EXPERTS)[:TOP_K] for _ in range(num_tokens)]
    )
    weights = torch.full((num_tokens, TOP_K), 1.0 / TOP_K)
    return hidden, index, weights


def _reference_experts(experts, hidden, index, weights):
    """MiniMax's own eager expert math, written out independently of the wrapper."""

    output = torch.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for slot in range(index.shape[1]):
            expert = int(index[token, slot])
            gate, up = F.linear(hidden[token], experts.gate_up_proj[expert]).chunk(2, dim=-1)
            value = F.linear(experts.act_fn(gate) * up, experts.down_proj[expert])
            output[token] += value * weights[token, slot]
    return output


def test_native_minimax_uses_fused_three_dimensional_experts():
    experts = _tiny_experts()
    assert not isinstance(experts, nn.ModuleList)
    assert experts.gate_up_proj.shape == (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    assert experts.down_proj.shape == (NUM_EXPERTS, HIDDEN, INTERMEDIATE)


def test_teacher_expert_forward_matches_the_reference_math():
    experts = _tiny_experts()
    hidden, index, weights = _routing()
    expected = _reference_experts(experts, hidden, index, weights)

    wrapped = NVFP4ExpertLoRA(experts, rank=4, alpha=4)
    with torch.no_grad(), shared_base_teacher():
        actual = wrapped(hidden, index, weights)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_student_starts_at_pure_ptq_and_only_adapters_train():
    experts = _tiny_experts()
    hidden, index, weights = _routing()
    wrapped = NVFP4ExpertLoRA(experts, rank=4, alpha=4)

    # B is zero-initialized, so the step-0 student is exactly fake_quant(base).
    quantized = copy.deepcopy(experts)
    with torch.no_grad():
        for expert in range(NUM_EXPERTS):
            quantized.gate_up_proj[expert] = fake_quantize_nvfp4(experts.gate_up_proj[expert])
            quantized.down_proj[expert] = fake_quantize_nvfp4(experts.down_proj[expert])
    expected = _reference_experts(quantized, hidden, index, weights)
    torch.testing.assert_close(wrapped(hidden, index, weights), expected, atol=1e-5, rtol=1e-5)

    wrapped(hidden, index, weights).square().mean().backward()
    assert wrapped.gate_up_proj.grad is None and wrapped.down_proj.grad is None
    assert wrapped.gate_up_lora_B.grad.abs().sum() > 0
    assert wrapped.down_lora_B.grad.abs().sum() > 0


@pytest.mark.parametrize("hybrid_shared", [True, False])
def test_expert_adapter_layouts(hybrid_shared):
    wrapped = NVFP4ExpertLoRA(_tiny_experts(), rank=4, alpha=8, hybrid_shared=hybrid_shared)
    shared_a = (2, 4, HIDDEN)
    unshared_a = (NUM_EXPERTS, 2, 4, HIDDEN)
    shared_b = (HIDDEN, 4)
    unshared_b = (NUM_EXPERTS, HIDDEN, 4)

    assert wrapped.gate_up_lora_A.shape == (shared_a if hybrid_shared else unshared_a)
    assert wrapped.down_lora_B.shape == (shared_b if hybrid_shared else unshared_b)
    # The per-expert factors are always independent in both layouts.
    assert wrapped.gate_up_lora_B.shape == (NUM_EXPERTS, 2, INTERMEDIATE, 4)
    assert wrapped.down_lora_A.shape == (NUM_EXPERTS, 4, INTERMEDIATE)


def test_fully_unshared_layout_keeps_experts_independent():
    wrapped = NVFP4ExpertLoRA(_tiny_experts(), rank=4, alpha=4, hybrid_shared=False)
    with torch.no_grad():
        wrapped.gate_up_lora_B[0].normal_(0, 0.1)
        wrapped.down_lora_B[0].normal_(0, 0.1)

    # Expert 0 moved; every other expert must still sit exactly on its frozen base.
    assert not torch.equal(wrapped.effective_gate_up(0), wrapped.base_gate_up(0))
    for expert in range(1, NUM_EXPERTS):
        torch.testing.assert_close(
            wrapped.effective_gate_up(expert), wrapped.base_gate_up(expert), atol=0, rtol=0
        )
        torch.testing.assert_close(
            wrapped.effective_down(expert), wrapped.base_down(expert), atol=0, rtol=0
        )


def test_batched_routing_matches_the_naive_per_expert_lookup():
    """The single-transfer routing must select exactly what per-expert `where` would.

    The loop slices one `nonzero` result by host-side counts instead of asking the device
    per expert. That is only correct because `nonzero` returns rows in row-major order, so
    each expert's rows are contiguous and in the same (slot, token) order.
    """

    _, index, _ = _routing(num_tokens=12)
    expert_mask = F.one_hot(index, num_classes=NUM_EXPERTS).permute(2, 1, 0)

    assignments = expert_mask.nonzero()
    counts = torch.bincount(assignments[:, 0], minlength=NUM_EXPERTS).tolist()

    start = 0
    seen = 0
    for expert, count in enumerate(counts):
        expected_pos, expected_token = torch.where(expert_mask[expert])
        rows = assignments[start : start + count]
        start += count
        seen += count
        torch.testing.assert_close(rows[:, 1], expected_pos, atol=0, rtol=0)
        torch.testing.assert_close(rows[:, 2], expected_token, atol=0, rtol=0)
    assert seen == assignments.shape[0] == 12 * TOP_K


def test_expert_checkpointing_does_not_change_gradients():
    hidden, index, weights = _routing()
    source = _tiny_experts()

    grads = []
    for expert_checkpointing in (False, True):
        torch.manual_seed(7)
        wrapped = NVFP4ExpertLoRA(
            copy.deepcopy(source), rank=4, alpha=4, expert_checkpointing=expert_checkpointing
        )
        with torch.no_grad():
            wrapped.gate_up_lora_B.normal_(0, 0.1)
            wrapped.down_lora_B.normal_(0, 0.1)
        wrapped(hidden, index, weights).square().mean().backward()
        grads.append((wrapped.gate_up_lora_B.grad.clone(), wrapped.down_lora_B.grad.clone()))

    for plain, recomputed in zip(grads[0], grads[1], strict=True):
        torch.testing.assert_close(plain, recomputed, atol=0, rtol=0)


def _fp8_quantize_blocks(weight: torch.Tensor, block: int = 8):
    """Quantize to block-scaled FP8 the way a released checkpoint stores it."""

    rows, cols = weight.shape[-2:]
    blocked = weight.reshape(*weight.shape[:-2], rows // block, block, cols // block, block)
    amax = blocked.abs().amax(dim=(-3, -1), keepdim=True).clamp_min(1e-12)
    scale = amax / torch.finfo(torch.float8_e4m3fn).max
    codes = (blocked / scale).to(torch.float8_e4m3fn).reshape(weight.shape)
    scale_inv = scale.reshape(*weight.shape[:-2], rows // block, cols // block).float()
    return codes, scale_inv


def test_fp8_dequantization_recovers_the_block_scaled_weight():
    torch.manual_seed(3)
    weight = torch.randn(32, 32, dtype=torch.bfloat16)
    codes, scale_inv = _fp8_quantize_blocks(weight)

    recovered = dequantize_fp8_blocks(codes, scale_inv, torch.bfloat16)

    assert codes.dtype == torch.float8_e4m3fn
    # E4M3 keeps ~2 decimal digits; the point is that the block scales are applied to the
    # right elements, so a loose tolerance with an exact shape/dtype check is the signal.
    assert recovered.shape == weight.shape and recovered.dtype == torch.bfloat16
    torch.testing.assert_close(recovered.float(), weight.float(), atol=0.05, rtol=0.05)


def test_fp8_dequantization_rejects_a_mismatched_scale_grid():
    weight = torch.zeros(32, 30, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="not divisible by its scale grid"):
        dequantize_fp8_blocks(weight, torch.ones(4, 4))


class _FP8Linear(nn.Module):
    """Stand-in for the block-FP8 linear a pre-quantized checkpoint loads as."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        codes, scale_inv = _fp8_quantize_blocks(weight)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.weight = nn.Parameter(codes, requires_grad=False)
        self.weight_scale_inv = nn.Parameter(scale_inv, requires_grad=False)
        self.bias = None


def test_fp8_linear_keeps_codes_frozen_and_dequantizes_on_demand():
    torch.manual_seed(5)
    reference = torch.randn(32, 32, dtype=torch.bfloat16)
    converted = NVFP4Linear.from_fp8_linear(_FP8Linear(reference))
    converted.init_lora(4, 4.0)

    # The stored base stays FP8; only the transient view is high precision.
    assert converted.weight.dtype == torch.float8_e4m3fn
    assert converted.base_weight().dtype == torch.bfloat16
    assert converted.weight.requires_grad is False

    x = torch.randn(3, 32, dtype=torch.bfloat16)
    with torch.no_grad(), shared_base_teacher():
        teacher = converted(x)
    torch.testing.assert_close(teacher, F.linear(x, converted.base_weight()), atol=0, rtol=0)

    # Zero-init B keeps the student at the pure PTQ point of the dequantized base.
    student = converted(x)
    expected = F.linear(x, fake_quantize_nvfp4(converted.base_weight()))
    torch.testing.assert_close(student, expected, atol=0, rtol=0)


def test_fp8_fused_experts_wrap_without_a_high_precision_copy():
    experts = _tiny_experts()
    reference_gate_up = experts.gate_up_proj.detach().clone()
    gate_up_codes, gate_up_scale = _fp8_quantize_blocks(reference_gate_up)
    down_codes, down_scale = _fp8_quantize_blocks(experts.down_proj.detach())
    experts.gate_up_proj = nn.Parameter(gate_up_codes, requires_grad=False)
    experts.down_proj = nn.Parameter(down_codes, requires_grad=False)
    experts.register_buffer("gate_up_proj_scale_inv", gate_up_scale)
    experts.register_buffer("down_proj_scale_inv", down_scale)

    wrapped = NVFP4ExpertLoRA(experts, rank=4, alpha=4)

    assert wrapped.is_fp8_base
    assert wrapped.gate_up_proj.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        wrapped.base_gate_up(0).float(), reference_gate_up[0].float(), atol=0.05, rtol=0.05
    )

    hidden, index, weights = _routing()
    hidden = hidden.to(torch.bfloat16)
    with torch.no_grad(), shared_base_teacher():
        teacher = wrapped(hidden, index, weights)
    assert torch.isfinite(teacher).all()

    wrapped(hidden, index, weights).float().square().mean().backward()
    assert wrapped.gate_up_lora_B.grad.abs().sum() > 0


def test_persistent_router_buffer_is_promoted_for_fsdp2():
    """FSDP2 replicates buffers but its full-state load expects only sharded parameters.

    MiniMax keeps the router's correction bias as a persistent buffer, which is exactly the
    case that breaks CPU-RAM-efficient loading, so it must become a frozen parameter.
    """

    from qatfactory.models.llm import _promote_persistent_buffers_for_fsdp2

    model = transformers_minimax.MiniMaxM2ForCausalLM(_tiny_config())
    router = model.model.layers[0].mlp
    assert "e_score_correction_bias" in dict(router.named_buffers(recurse=False))
    before = model.state_dict()["model.layers.0.mlp.e_score_correction_bias"].clone()

    promoted = _promote_persistent_buffers_for_fsdp2(model)

    assert "model.layers.0.mlp.e_score_correction_bias" in promoted
    parameters = dict(model.named_parameters())
    assert "model.layers.0.mlp.e_score_correction_bias" in parameters
    assert parameters["model.layers.0.mlp.e_score_correction_bias"].requires_grad is False
    # The state dict keeps the same key and value, so checkpoints are unaffected.
    torch.testing.assert_close(
        model.state_dict()["model.layers.0.mlp.e_score_correction_bias"], before, atol=0, rtol=0
    )
    # Non-persistent buffers (rotary caches) stay buffers.
    assert not any("inv_freq" in name for name in promoted)


def test_experts_only_run_needs_no_dense_targets():
    """A MoE run may quantize only its routed experts and leave attention alone."""

    torch.manual_seed(13)
    config = _tiny_config()
    student = transformers_minimax.MiniMaxM2ForCausalLM(config).eval()

    converted = apply_fake_quant(student, [], lora_rank=2, lora_alpha=2)

    assert converted.linears == ()
    assert converted.experts == ("model.layers.0.mlp.experts",)
    assert len(converted.expert_targets) == NUM_EXPERTS * 3
    trainable = {name for name, parameter in student.named_parameters() if parameter.requires_grad}
    assert trainable == converted.trainable_parameters
    assert not any("self_attn" in name for name in trainable)
    # Attention still runs in full precision.
    assert isinstance(student.model.layers[0].self_attn.q_proj, nn.Linear)
    assert not isinstance(student.model.layers[0].self_attn.q_proj, NVFP4Linear)


def test_untargeted_fp8_linears_are_dequantized_not_left_to_the_vendor_kernel():
    """Anything we do not quantize must still stop calling the library's FP8 matmul.

    A broken FP8 kernel is invisible to the loss — the shared-base teacher runs through the
    same layers, so teacher and student degrade together — which makes this substitution a
    correctness requirement rather than a performance choice.
    """

    from qatfactory.quant.fake_quant import dequantize_remaining_fp8_linears

    torch.manual_seed(5)
    reference = torch.randn(32, 32, dtype=torch.bfloat16)
    parent = nn.Module()
    parent.q_proj = _FP8Linear(reference)
    model = nn.Module()
    model.self_attn = parent

    dequantized = dequantize_remaining_fp8_linears(model)

    assert dequantized == ["self_attn.q_proj"]
    replacement = model.self_attn.q_proj
    assert type(replacement) is nn.Linear
    assert replacement.weight.dtype == torch.bfloat16
    assert replacement.weight.requires_grad is False
    x = torch.randn(3, 32, dtype=torch.bfloat16)
    torch.testing.assert_close(replacement(x), F.linear(x, replacement.weight), atol=0, rtol=0)
    torch.testing.assert_close(replacement.weight.float(), reference.float(), atol=0.05, rtol=0.05)


def test_dense_only_model_still_requires_targets():
    """Only a MoE may omit them: a dense model with no targets would quantize nothing."""

    dense = nn.Module()
    dense.config = _tiny_config(num_local_experts=0)
    with pytest.raises(ValueError, match="non-empty target_modules"):
        apply_fake_quant(dense, [], lora_rank=2, lora_alpha=2)


def test_apply_fake_quant_instruments_a_whole_tiny_minimax_model():
    torch.manual_seed(11)
    config = _tiny_config()
    student = transformers_minimax.MiniMaxM2ForCausalLM(config).eval()
    teacher = copy.deepcopy(student).eval()

    converted = apply_fake_quant(
        student,
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        fused_runtime_scales=True,
        lora_rank=2,
        lora_alpha=2,
    )

    # One fused experts module plus 4 attention projections; the experts module expands to
    # three deployed matrices per expert.
    assert converted.experts == ("model.layers.0.mlp.experts",)
    assert len(converted.expert_targets) == NUM_EXPERTS * 3
    assert len(converted) == 4 + NUM_EXPERTS * 3

    trainable = {name for name, parameter in student.named_parameters() if parameter.requires_grad}
    assert trainable == converted.trainable_parameters
    # The router, its correction bias and the embeddings must never train.
    assert not any("mlp.gate" in name or "e_score" in name or "embed" in name for name in trainable)

    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    with torch.no_grad():
        expected = teacher(input_ids=input_ids, use_cache=False).logits
        with shared_base_teacher():
            actual = student(input_ids=input_ids, use_cache=False).logits
        quantized = student(input_ids=input_ids, use_cache=False).logits

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    assert not torch.equal(quantized, actual)
