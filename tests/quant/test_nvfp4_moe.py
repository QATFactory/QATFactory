import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from qatfactory.quant import (
    NVFP4Experts,
    NVFP4Linear,
    apply_fake_quant,
    fake_quantize_nvfp4,
    fake_quantize_nvfp4_activation,
    fake_quantize_nvfp4_experts,
    finalize_nvfp4_input_scales,
)
from qatfactory.quant.export.nvfp4_vllm import (
    SafetensorDirectory,
    _expected_targets,
    _split_expert_targets,
    export_nvfp4_vllm_checkpoint,
)


NUM_EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 4, 32, 16, 2


def _random_experts_weights(seed=0):
    torch.manual_seed(seed)
    gate_up = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    down = torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE)
    return gate_up, down


def test_expert_fake_quant_is_bit_identical_to_per_expert_2d():
    gate_up, _ = _random_experts_weights()
    amax = gate_up.detach().abs().amax(dim=(1, 2))

    batched = fake_quantize_nvfp4_experts(gate_up, amax)
    for expert in range(NUM_EXPERTS):
        reference = fake_quantize_nvfp4(gate_up[expert], tensor_amax=amax[expert])
        torch.testing.assert_close(batched[expert], reference, atol=0, rtol=0)


def test_expert_fake_quant_slice_matches_whole_slice_quantization():
    """gate rows quantized alone equal the same rows inside the whole gate+up slice."""

    gate_up, _ = _random_experts_weights(seed=1)
    slice_2d = gate_up[0]
    amax = slice_2d.abs().amax()

    whole = fake_quantize_nvfp4(slice_2d, tensor_amax=amax)
    gate_only = fake_quantize_nvfp4(slice_2d[:INTERMEDIATE], tensor_amax=amax)

    torch.testing.assert_close(whole[:INTERMEDIATE], gate_only, atol=0, rtol=0)


def test_expert_fake_quant_zero_expert_and_ste():
    gate_up, _ = _random_experts_weights(seed=2)
    with torch.no_grad():
        gate_up[1].zero_()
    gate_up.requires_grad_(True)
    amax = gate_up.detach().abs().amax(dim=(1, 2))

    quantized = fake_quantize_nvfp4_experts(gate_up, amax)
    torch.testing.assert_close(quantized[1], torch.zeros_like(quantized[1]))
    quantized.sum().backward()
    torch.testing.assert_close(gate_up.grad[1], torch.ones_like(gate_up.grad[1]))
    assert gate_up.grad.shape == gate_up.shape


def test_expert_fake_quant_rejects_bad_shapes():
    gate_up, _ = _random_experts_weights()
    with pytest.raises(ValueError, match="3D"):
        fake_quantize_nvfp4_experts(gate_up[0], gate_up.abs().amax(dim=(1, 2)))
    with pytest.raises(ValueError, match="shape"):
        fake_quantize_nvfp4_experts(gate_up, torch.ones(2))
    with pytest.raises(ValueError, match="divisible"):
        fake_quantize_nvfp4_experts(torch.randn(2, 4, 24), torch.ones(2))


class _PackedExperts(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        gate_up, down = _random_experts_weights(seed)
        self.gate_up_proj = nn.Parameter(gate_up)
        self.down_proj = nn.Parameter(down)
        self.act_fn = F.silu


def _routing(tokens, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(tokens, NUM_EXPERTS)
    weights, indices = torch.topk(torch.softmax(logits.float(), dim=-1), TOP_K, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return indices, weights.to(logits.dtype)


def test_nvfp4_experts_forward_matches_quantized_reference():
    experts = NVFP4Experts.from_experts(_PackedExperts())
    hidden = torch.randn(6, HIDDEN)
    indices, weights = _routing(6)

    output = experts(hidden, indices, weights)

    # Reference mirrors the module's own accumulation order (per expert, index_add) with
    # independently quantized 2D slices, so the comparison is exact.
    gate_up_amax = experts.gate_up_proj.detach().abs().amax(dim=(1, 2))
    down_amax = experts.down_proj.detach().abs().amax(dim=(1, 2))
    expected = torch.zeros_like(hidden)
    for expert in range(NUM_EXPERTS):
        token_index, k_position = torch.where(indices == expert)
        if token_index.numel() == 0:
            continue
        gate_up_q = fake_quantize_nvfp4(experts.gate_up_proj[expert], tensor_amax=gate_up_amax[expert])
        down_q = fake_quantize_nvfp4(experts.down_proj[expert], tensor_amax=down_amax[expert])
        gate, up = F.linear(hidden[token_index], gate_up_q).chunk(2, dim=-1)
        out = F.linear(F.silu(gate) * up, down_q) * weights[token_index, k_position, None]
        expected.index_add_(0, token_index, out)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_nvfp4_experts_ste_gradients_reach_latent_weights():
    experts = NVFP4Experts.from_experts(_PackedExperts())
    experts.gate_up_proj.requires_grad_(True)
    experts.down_proj.requires_grad_(True)
    hidden = torch.randn(6, HIDDEN, requires_grad=True)
    indices, weights = _routing(6)

    experts(hidden, indices, weights).sum().backward()

    assert experts.gate_up_proj.grad is not None and experts.gate_up_proj.grad.abs().sum() > 0
    assert experts.down_proj.grad is not None and experts.down_proj.grad.abs().sum() > 0
    assert hidden.grad is not None


def test_nvfp4_experts_tracks_input_running_max_in_training_only():
    experts = NVFP4Experts.from_experts(_PackedExperts())
    experts._track_input_amax = True
    experts.train()
    indices, weights = _routing(4)

    experts(torch.full((4, HIDDEN), 2.0), indices, weights)
    experts(torch.full((4, HIDDEN), 1.0), indices, weights)

    torch.testing.assert_close(experts._gate_up_input_amax, torch.tensor(2.0))
    assert experts._down_input_amax is not None and experts._down_input_amax > 0
    recorded = (experts._gate_up_input_amax.clone(), experts._down_input_amax.clone())
    experts.eval()
    experts(torch.full((4, HIDDEN), 9.0), indices, weights)
    torch.testing.assert_close(experts._gate_up_input_amax, recorded[0])
    torch.testing.assert_close(experts._down_input_amax, recorded[1])


class _Router(nn.Module):
    """transformers>=5.9-style router: returns (raw logits, top-k weights, indices)."""

    def __init__(self):
        super().__init__()
        self.top_k = TOP_K
        self.num_experts = NUM_EXPERTS
        self.norm_topk_prob = True
        self.weight = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN))

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.reshape(-1, HIDDEN), self.weight)
        probs = torch.softmax(logits.float(), dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights.to(logits.dtype), indices


class _SharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)


class _MoEBlock(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        self.gate = _Router()
        self.experts = _PackedExperts(seed)
        self.shared_expert = _SharedExpert()
        self.shared_expert_gate = nn.Linear(HIDDEN, 1, bias=False)

    def forward(self, hidden_states):
        _, weights, indices = self.gate(hidden_states)
        return self.experts(hidden_states, indices, weights)


class _AttnBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)


class _MoELayer(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        self.self_attn = _AttnBlock()
        self.mlp = _MoEBlock(seed)


class _MoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        text_config = SimpleNamespace(num_experts=NUM_EXPERTS, tie_word_embeddings=False)
        self.config = SimpleNamespace(
            model_type="qwen3_moe",
            tie_word_embeddings=False,
            get_text_config=lambda: text_config,
        )
        self.layers = nn.ModuleList([_MoELayer(seed) for seed in range(2)])
        self.lm_head = nn.Linear(HIDDEN, 64)


_MOE_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def test_apply_fake_quant_converts_experts_and_shared_expert():
    model = _MoEModel()

    converted = apply_fake_quant(model, _MOE_TARGETS, fused_runtime_scales=True)

    assert set(converted.experts) == {"layers.0.mlp.experts", "layers.1.mlp.experts"}
    assert "layers.0.mlp.shared_expert.gate_proj" in converted.linears
    assert isinstance(model.layers[0].mlp.experts, NVFP4Experts)
    assert isinstance(model.layers[0].mlp.shared_expert.gate_proj, NVFP4Linear)
    assert not isinstance(model.layers[0].mlp.shared_expert_gate, NVFP4Linear)
    assert not isinstance(model.layers[0].mlp.gate, NVFP4Linear)
    assert not model.layers[0].mlp.gate.weight.requires_grad
    assert model.layers[0].mlp.experts._track_input_amax

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == converted.trainable_parameters
    assert "layers.0.mlp.experts.gate_up_proj" in trainable

    families = {group.member_names for group in model._nvfp4_amax_groups}
    assert ("layers.0.mlp.shared_expert.gate_proj", "layers.0.mlp.shared_expert.up_proj") in families


def test_apply_fake_quant_shared_expert_opt_out():
    model = _MoEModel()

    converted = apply_fake_quant(
        model, _MOE_TARGETS, fused_runtime_scales=True, quantize_shared_expert=False
    )

    assert not any("shared_expert" in name for name in converted.linears)
    assert not isinstance(model.layers[0].mlp.shared_expert.gate_proj, NVFP4Linear)
    assert not model.layers[0].mlp.shared_expert.gate_proj.weight.requires_grad


def test_apply_fake_quant_experts_only_mode():
    """Experts-only NVFP4 (MiniMax scope): quantize packed experts, keep every
    attention/dense linear in BF16. No linear suffixes are passed."""

    model = _MoEModel()
    original_attn_weight = model.layers[0].self_attn.q_proj.weight.detach().clone()

    converted = apply_fake_quant(model, [], fused_runtime_scales=True)

    # Only the packed experts convert; there are no quantized linears at all.
    assert set(converted.experts) == {"layers.0.mlp.experts", "layers.1.mlp.experts"}
    assert converted.linears == ()
    assert isinstance(model.layers[0].mlp.experts, NVFP4Experts)

    # Every attention and dense (shared-expert) linear stays a plain BF16 nn.Linear.
    for layer in model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert type(getattr(layer.self_attn, name)) is nn.Linear
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert type(getattr(layer.mlp.shared_expert, name)) is nn.Linear
    torch.testing.assert_close(model.layers[0].self_attn.q_proj.weight, original_attn_weight)

    # Trainable parameters are exactly the expert projections; everything else frozen.
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == converted.trainable_parameters
    assert trainable == {
        f"layers.{index}.mlp.experts.{projection}"
        for index in range(2)
        for projection in ("gate_up_proj", "down_proj")
    }

    # fused_runtime_scales with zero linears: no fused families are built, but the
    # experts input-amax tracking must still engage for a fused-scale export.
    assert not getattr(model, "_nvfp4_amax_groups", [])
    assert all(layer.mlp.experts._track_input_amax for layer in model.layers)


def test_apply_fake_quant_experts_only_forward_is_weight_only():
    """The experts-only conversion yields plain W4A16 numerics: expert weights are
    NVFP4 fake-quantized while the activation stays BF16."""

    model = _MoEModel()
    gate_up = model.layers[0].mlp.experts.gate_up_proj.detach().clone()
    down = model.layers[0].mlp.experts.down_proj.detach().clone()

    apply_fake_quant(model, [], fused_runtime_scales=True)
    model.eval()

    hidden = torch.randn(6, HIDDEN)
    indices, weights = _routing(6)
    output = model.layers[0].mlp.experts(hidden, indices, weights)

    gate_up_amax = gate_up.abs().amax(dim=(1, 2))
    down_amax = down.abs().amax(dim=(1, 2))
    expected = torch.zeros_like(hidden)
    for expert in range(NUM_EXPERTS):
        token_index, k_position = torch.where(indices == expert)
        if token_index.numel() == 0:
            continue
        gate_up_q = fake_quantize_nvfp4(gate_up[expert], tensor_amax=gate_up_amax[expert])
        down_q = fake_quantize_nvfp4(down[expert], tensor_amax=down_amax[expert])
        gate, up = F.linear(hidden[token_index], gate_up_q).chunk(2, dim=-1)
        out = F.linear(F.silu(gate) * up, down_q) * weights[token_index, k_position, None]
        expected.index_add_(0, token_index, out)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_apply_fake_quant_empty_targets_requires_moe():
    """An empty target_modules set is only valid for an MoE model (experts-only mode)."""

    model = _MoEModel()
    model.config.get_text_config = lambda: SimpleNamespace(num_experts=None, tie_word_embeddings=False)
    with pytest.raises(ValueError, match="non-empty target_modules"):
        apply_fake_quant(model, [])


def test_finalize_input_scales_experts_only():
    """finalize_nvfp4_input_scales tolerates a model with zero fused linear families."""

    model = _MoEModel()
    apply_fake_quant(model, [], fused_runtime_scales=True)
    model.train()
    indices, weights = _routing(4)
    for layer in model.layers:
        layer.mlp.experts(torch.randn(4, HIDDEN), indices, weights)

    input_scales = finalize_nvfp4_input_scales(model)

    assert set(input_scales) == {
        f"layers.{index}.mlp.experts.{group}.input_scale" for index in range(2) for group in ("w13", "w2")
    }
    assert all(scale > 0 for scale in input_scales.values())


def test_activation_fake_quant_matches_2d_weight_path():
    torch.manual_seed(3)
    x = torch.randn(4, 6, HIDDEN)  # [B, S, hidden]
    amax = x.detach().abs().amax()

    q = fake_quantize_nvfp4_activation(x, amax)
    ref = fake_quantize_nvfp4(x.reshape(-1, HIDDEN), tensor_amax=amax).reshape(x.shape)

    torch.testing.assert_close(q, ref, atol=0, rtol=0)


def test_activation_fake_quant_ste_passes_gradient_upstream():
    x = torch.randn(4, HIDDEN, requires_grad=True)
    fake_quantize_nvfp4_activation(x, x.detach().abs().amax()).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_w4a4_linear_quantizes_input_and_flows_gradient():
    linear = NVFP4Linear.from_linear(nn.Linear(HIDDEN, 8, bias=False))
    linear._track_input_amax = True
    linear._quantize_activations = True
    linear.train()
    x = torch.randn(4, HIDDEN, requires_grad=True)

    out = linear(x)
    # W4A4 output differs from weight-only (activation is now quantized too).
    weight_only = torch.nn.functional.linear(x, fake_quantize_nvfp4(linear.weight))
    assert not torch.allclose(out, weight_only)
    assert linear._input_amax is not None  # running max bootstrapped/updated
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_w4a4_apply_fake_quant_sets_flag_and_requires_fused():
    model = _MoEModel()
    converted = apply_fake_quant(
        model, _MOE_TARGETS, fused_runtime_scales=True, quantize_activations=True
    )
    assert converted.experts
    flagged = [m for m in model.modules() if isinstance(m, (NVFP4Linear, NVFP4Experts))]
    assert flagged and all(m._quantize_activations for m in flagged)

    with pytest.raises(ValueError, match="fused_runtime_scales"):
        apply_fake_quant(_MoEModel(), _MOE_TARGETS, quantize_activations=True)


def test_w4a4_experts_forward_quantizes_activations():
    experts = NVFP4Experts.from_experts(_PackedExperts())
    experts._track_input_amax = True
    experts._quantize_activations = True
    experts.train()
    hidden = torch.randn(6, HIDDEN, requires_grad=True)
    indices, weights = _routing(6)

    out = experts(hidden, indices, weights)
    assert experts._gate_up_input_amax is not None and experts._down_input_amax is not None
    out.sum().backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_apply_fake_quant_full_parameter_experts_without_lora():
    model = _MoEModel()
    converted = apply_fake_quant(model, _MOE_TARGETS, fused_runtime_scales=True)
    assert not converted.lora
    assert all(isinstance(layer.mlp.experts, NVFP4Experts) for layer in model.layers)
    assert {f"layers.0.mlp.experts.{e}.{p}" for e in range(NUM_EXPERTS) for p in ("w1", "w3", "w2")} <= set(
        converted.expert_targets
    )


def test_apply_fake_quant_shared_expert_opt_out_requires_shared_experts():
    model = _MoEModel()
    for layer in model.layers:
        del layer.mlp.shared_expert
        del layer.mlp.shared_expert_gate
    with pytest.raises(ValueError, match="no shared-expert"):
        apply_fake_quant(model, _MOE_TARGETS, quantize_shared_expert=False)


def test_finalize_input_scales_includes_expert_entries():
    model = _MoEModel()
    apply_fake_quant(model, _MOE_TARGETS, fused_runtime_scales=True)
    model.train()
    for module in model.modules():
        if isinstance(module, NVFP4Linear) and module._track_input_amax:
            module._input_amax = torch.tensor(1.0)
    indices, weights = _routing(4)
    for layer in model.layers:
        layer.mlp.experts(torch.randn(4, HIDDEN), indices, weights)

    input_scales = finalize_nvfp4_input_scales(model)

    assert "layers.0.mlp.experts.w13.input_scale" in input_scales
    assert "layers.1.mlp.experts.w2.input_scale" in input_scales
    assert all(scale > 0 for scale in input_scales.values())


_EXPERTS = "model.layers.0.mlp.experts"
_EXPORT_EXPERTS, _EXPORT_DIM = 2, 16


def _expert_tensors() -> dict[str, torch.Tensor]:
    """Fused expert tensors as the trainer saves them (``save_original_format=False``)."""

    torch.manual_seed(11)
    gate_up = torch.randn(_EXPORT_EXPERTS, 2 * _EXPORT_DIM, _EXPORT_DIM, dtype=torch.bfloat16)
    down = torch.randn(_EXPORT_EXPERTS, _EXPORT_DIM, _EXPORT_DIM, dtype=torch.bfloat16)
    return {f"{_EXPERTS}.gate_up_proj": gate_up, f"{_EXPERTS}.down_proj": down}


def _write_moe_source(tmp_path, *, model_type="qwen3_moe", experts_only=False):
    source = tmp_path / "source"
    source.mkdir()
    experts_key = "num_local_experts" if model_type == "minimax_m2" else "num_experts"
    config = {"model_type": model_type, "num_hidden_layers": 1, experts_key: _EXPORT_EXPERTS, "dtype": "bfloat16"}
    quantizable = _expected_targets(config)
    linears, experts = _split_expert_targets(quantizable)
    assert experts == {_EXPERTS: _EXPORT_EXPERTS}
    trained = sorted(quantizable) if not experts_only else sorted(set(quantizable) - set(linears))
    config["qatfactory_qad"] = {
        "format_version": 1,
        "quantizer": "standard_max",
        "fused_runtime_scales": True,
        "input_scale_source": "training_running_max",
        "target_modules": trained,
        "global_step": 3,
    }
    (source / "config.json").write_text(json.dumps(config))

    tensors = {f"{module}.weight": torch.randn(_EXPORT_DIM, _EXPORT_DIM, dtype=torch.bfloat16) for module in linears}
    tensors["model.layers.0.mlp.gate.weight"] = torch.randn(_EXPORT_EXPERTS, _EXPORT_DIM, dtype=torch.bfloat16)
    tensors["lm_head.weight"] = torch.randn(4, _EXPORT_DIM, dtype=torch.bfloat16)
    if model_type == "minimax_m2":
        tensors["model.layers.0.mlp.e_score_correction_bias"] = torch.zeros(_EXPORT_EXPERTS, dtype=torch.bfloat16)
    tensors.update(_expert_tensors())
    save_file(tensors, source / "model.safetensors")

    trained_linears = [] if experts_only else linears
    scales = {f"{module}.input_scale": torch.tensor(0.25) for module in trained_linears}
    scales[f"{_EXPERTS}.w13.input_scale"] = torch.tensor(0.5)
    scales[f"{_EXPERTS}.w2.input_scale"] = torch.tensor(0.125)
    save_file(scales, source / "input_scales.safetensors")
    return source, linears


def test_native_vllm_export_moe_layout(tmp_path):
    source, linears = _write_moe_source(tmp_path)
    output = tmp_path / "output"
    summary = export_nvfp4_vllm_checkpoint(
        source=source, model_assets=source, output=output, device="cpu", shard_size=1_000_000
    )

    assert summary["quantized_linears"] == len(linears) == 4
    assert summary["quantized_expert_linears"] == _EXPORT_EXPERTS * 3
    store = SafetensorDirectory(output)
    for expert in range(_EXPORT_EXPERTS):
        prefix = f"{_EXPERTS}.{expert}"
        assert f"{prefix}.gate_proj.weight_packed" in store.keys()
        gate_scale = store.get_tensor(f"{prefix}.gate_proj.weight_global_scale")
        up_scale = store.get_tensor(f"{prefix}.up_proj.weight_global_scale")
        assert torch.equal(gate_scale, up_scale)
        torch.testing.assert_close(store.get_tensor(f"{prefix}.gate_proj.input_global_scale"), torch.tensor([2.0]))
        torch.testing.assert_close(store.get_tensor(f"{prefix}.up_proj.input_global_scale"), torch.tensor([2.0]))
        torch.testing.assert_close(store.get_tensor(f"{prefix}.down_proj.input_global_scale"), torch.tensor([8.0]))
    # The packed weights decode back to the source expert slices' NVFP4 grid.
    from qatfactory.quant.formats.nvfp4 import dequantize_nvfp4

    gate_up = _expert_tensors()[f"{_EXPERTS}.gate_up_proj"]
    decoded = dequantize_nvfp4(
        store.get_tensor(f"{_EXPERTS}.1.up_proj.weight_packed"),
        store.get_tensor(f"{_EXPERTS}.1.up_proj.weight_scale"),
        store.get_tensor(f"{_EXPERTS}.1.up_proj.weight_global_scale").reciprocal(),
    )
    assert decoded.shape == gate_up[1, _EXPORT_DIM:].shape
    # Router carried in BF16 and declared ignored; no raw expert tensors leak.
    assert "model.layers.0.mlp.gate.weight" in store.keys()
    quantization = json.loads((output / "config.json").read_text())["quantization_config"]
    assert "re:.*mlp\\.gate$" in quantization["ignore"]
    assert not any(key.endswith(("gate_up_proj", "experts.down_proj")) for key in store.keys())
    assert json.loads((output / "nvfp4_export.json").read_text())["experts_modules"] == [_EXPERTS]


def test_native_vllm_export_excludes_raw_expert_weights_from_assets(tmp_path):
    """A per-expert base snapshot must not leak raw expert weights beside the packed export."""

    source, _ = _write_moe_source(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    fused = _expert_tensors()
    asset_tensors = {"extra_tower.weight": torch.randn(2, 2, dtype=torch.bfloat16)}
    for expert in range(_EXPORT_EXPERTS):
        asset_tensors[f"{_EXPERTS}.{expert}.gate_proj.weight"] = fused[f"{_EXPERTS}.gate_up_proj"][expert, :_EXPORT_DIM].clone()
        asset_tensors[f"{_EXPERTS}.{expert}.down_proj.weight"] = fused[f"{_EXPERTS}.down_proj"][expert].clone()
    save_file(asset_tensors, assets / "model.safetensors")

    output = tmp_path / "output"
    summary = export_nvfp4_vllm_checkpoint(
        source=source, model_assets=assets, output=output, device="cpu", shard_size=1_000_000
    )

    assert summary["carried_asset_tensors"] == 1  # only the genuine extra asset
    store = SafetensorDirectory(output)
    assert "extra_tower.weight" in store.keys()
    assert not any(key.endswith(".gate_proj.weight") for key in store.keys())


def test_native_vllm_export_rejects_partial_experts(tmp_path):
    source, _ = _write_moe_source(tmp_path)
    config = json.loads((source / "config.json").read_text())
    config["qatfactory_qad"]["target_modules"].remove(f"{_EXPERTS}.1.w2")
    (source / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="only some experts"):
        export_nvfp4_vllm_checkpoint(source=source, model_assets=source, output=tmp_path / "out", device="cpu")


def test_native_vllm_export_experts_only_qwen3_moe(tmp_path):
    """Experts-only scope: attention stays BF16 and is listed in the ignore set."""

    source, linears = _write_moe_source(tmp_path, experts_only=True)
    output = tmp_path / "output"
    summary = export_nvfp4_vllm_checkpoint(
        source=source, model_assets=source, output=output, device="cpu", shard_size=1_000_000
    )

    assert summary["quantized_linears"] == 0
    assert summary["quantized_expert_linears"] == _EXPORT_EXPERTS * 3
    store = SafetensorDirectory(output)
    for module in linears:
        assert store.get_dtype_shape(f"{module}.weight")[0] == "BF16"
        assert f"{module}.weight_packed" not in store.keys()
    quantization = json.loads((output / "config.json").read_text())["quantization_config"]
    assert any("self_attn" in entry for entry in quantization["ignore"])
    assert "re:.*mlp\\.gate$" in quantization["ignore"]


def test_native_vllm_export_minimax_serving_namespace(tmp_path):
    """minimax_m2 exports in vLLM's released layout: block_sparse_moe + w1/w3/w2, FP32 bias."""

    source, _ = _write_moe_source(tmp_path, model_type="minimax_m2", experts_only=True)
    output = tmp_path / "output"
    export_nvfp4_vllm_checkpoint(source=source, model_assets=source, output=output, device="cpu", shard_size=1_000_000)

    store = SafetensorDirectory(output)
    serving = "model.layers.0.block_sparse_moe"
    for expert in range(_EXPORT_EXPERTS):
        for projection in ("w1", "w3", "w2"):
            assert f"{serving}.experts.{expert}.{projection}.weight_packed" in store.keys()
    assert f"{serving}.gate.weight" in store.keys()
    assert store.get_dtype_shape(f"{serving}.e_score_correction_bias")[0] == "F32"
    assert not any(".mlp." in key for key in store.keys())
    quantization = json.loads((output / "config.json").read_text())["quantization_config"]
    assert "re:.*block_sparse_moe\\.gate$" in quantization["ignore"]
    assert "re:.*e_score_correction_bias$" in quantization["ignore"]
    assert any("block_sparse_moe" in target for target in quantization["config_groups"]["group_0"]["targets"])


def test_expert_fake_quant_chunked_is_bit_identical_with_equal_gradients(monkeypatch):
    """Slabbed expert fake-quant (val gate v6 OOM fix) must not change a single bit.

    Forces 2-experts-per-slab on a 5-expert tensor (uneven final slab, a zero-amax
    expert crossing a slab boundary) and asserts outputs AND STE gradients are
    bit-identical to the whole-tensor path.
    """

    from qatfactory.quant.formats import nvfp4_experts as nvfp4_module

    torch.manual_seed(7)
    experts, rows, features = 5, 6, 32
    chunked_weight = torch.randn(experts, rows, features, dtype=torch.bfloat16, requires_grad=True)
    whole_weight = chunked_weight.detach().clone().requires_grad_(True)
    amax = chunked_weight.detach().abs().amax(dim=(1, 2)).float()
    amax[2] = 0.0  # zero-amax expert sits at a slab boundary

    monkeypatch.setattr(nvfp4_module, "_EXPERT_FAKE_QUANT_CHUNK_ELEMENTS", 2 * rows * features)
    chunked = nvfp4_module.fake_quantize_nvfp4_experts(chunked_weight, amax)
    whole = nvfp4_module._fake_quantize_nvfp4_experts_whole(whole_weight, amax)
    assert torch.equal(chunked, whole)

    upstream = torch.randn_like(chunked)
    chunked.backward(upstream)
    whole.backward(upstream)
    assert torch.equal(chunked_weight.grad, whole_weight.grad)


def test_expert_fake_quant_default_chunking_no_op_for_small_tensors():
    from qatfactory.quant.formats import nvfp4_experts as nvfp4_module

    weight = torch.randn(4, 8, 32, dtype=torch.bfloat16)
    amax = weight.abs().amax(dim=(1, 2)).float()
    out = nvfp4_module.fake_quantize_nvfp4_experts(weight, amax)
    ref = nvfp4_module._fake_quantize_nvfp4_experts_whole(weight, amax)
    assert torch.equal(out, ref)


def test_expert_fake_quant_keeps_no_lattice_graph_for_backward():
    """Only the masked STE may sit between the output and the latent in the autograd graph.

    A tracked lattice computation keeps every FP32 intermediate of the layer's experts alive
    until backward; on MiniMax-M2.7 that was tens of GB per layer on a recompute and ran the
    2-node QAD out of memory in its first backward.
    """

    from qatfactory.quant.formats import nvfp4_experts as nvfp4_module

    weight = torch.randn(3, 8, 32, dtype=torch.bfloat16, requires_grad=True)
    amax = weight.detach().abs().amax(dim=(1, 2)).float()
    out = nvfp4_module.fake_quantize_nvfp4_experts(weight, amax)

    nodes, stack, seen = [], [out.grad_fn], set()
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        nodes.append(type(node).__name__)
        stack.extend(child for child, _ in node.next_functions)
    assert sorted(nodes) == ["AccumulateGrad", "_MaskedStraightThroughBackward"]

    # The estimator itself is unchanged: in-range elements pass the gradient through.
    _, in_range = nvfp4_module._fake_quantize_nvfp4_experts_values(weight.detach(), amax)
    upstream = torch.randn_like(out)
    out.backward(upstream)
    assert torch.equal(weight.grad, upstream * in_range)


def test_nvfp4_experts_match_the_whole_tensor_formulation_bit_for_bit():
    """Fake-quantizing inside each expert's matmul (weights recomputed in backward) must give the
    outputs and gradients of fake-quantizing the whole expert tensor first, exactly."""

    experts = NVFP4Experts.from_experts(_PackedExperts(seed=3))
    experts.gate_up_proj.requires_grad_(True)
    experts.down_proj.requires_grad_(True)
    hidden = torch.randn(10, HIDDEN, requires_grad=True)
    indices, weights = _routing(10, seed=3)
    output = experts(hidden, indices, weights)
    upstream = torch.randn_like(output)
    output.backward(upstream)

    gate_up_latent = experts.gate_up_proj.detach().clone().requires_grad_(True)
    down_latent = experts.down_proj.detach().clone().requires_grad_(True)
    hidden_ref = hidden.detach().clone().requires_grad_(True)
    gate_up = fake_quantize_nvfp4_experts(gate_up_latent, gate_up_latent.detach().abs().amax(dim=(1, 2)).float())
    down = fake_quantize_nvfp4_experts(down_latent, down_latent.detach().abs().amax(dim=(1, 2)).float())
    gate_up, down = gate_up.unbind(0), down.unbind(0)
    expected = torch.zeros_like(hidden_ref)
    for expert in torch.unique(indices).tolist():
        token_index, k_position = torch.where(indices == expert)
        gate, up = F.linear(hidden_ref[token_index], gate_up[expert]).chunk(2, dim=-1)
        out = F.linear(F.silu(gate) * up, down[expert]) * weights[token_index, k_position, None]
        expected.index_add_(0, token_index, out.to(expected.dtype))
    expected.backward(upstream)

    assert torch.equal(output.detach(), expected.detach())
    assert torch.equal(hidden.grad, hidden_ref.grad)
    assert torch.equal(experts.gate_up_proj.grad, gate_up_latent.grad)
    assert torch.equal(experts.down_proj.grad, down_latent.grad)
