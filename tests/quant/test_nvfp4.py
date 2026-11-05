import json
import re
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from qatfactory.quant import (
    NVFP4Linear,
    apply_fake_quant,
    build_nvfp4_fused_family_names,
    commit_nvfp4_input_scales,
    dequantize_nvfp4,
    fake_quantize_nvfp4,
    fake_quantize_nvfp4_activation,
    finalize_nvfp4_input_scales,
    load_nvfp4_runtime_state_dict,
    nvfp4_runtime_state_dict,
    pack_e2m1_rows,
    quantize_nvfp4,
    refresh_nvfp4_weight_amax,
    unpack_e2m1_rows,
)
from qatfactory.quant.formats.nvfp4 import NVFP4_SCALE_DENOMINATOR, _round_to_e2m1
from qatfactory.quant.export.nvfp4_vllm import (
    SafetensorDirectory,
    _expected_targets,
    export_nvfp4_vllm_checkpoint,
    unquantized_ignore_entries,
)


def test_nvfp4_known_values_and_ste_gradient():
    weight = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2], requires_grad=True)
    quantized = fake_quantize_nvfp4(weight)

    torch.testing.assert_close(quantized, weight)
    quantized.sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))


def test_e2m1_round_to_nearest_even_is_sign_symmetric():
    values = torch.tensor([-5.0, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    expected = torch.tensor([-4.0, -4.0, -2.0, -2.0, -1.0, -1.0, -0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])

    torch.testing.assert_close(_round_to_e2m1(values), expected)


def test_nvfp4_preserves_e4m3_subnormal_block_scales():
    weight = torch.zeros(1, 32)
    weight[0, 0] = 2688.0
    weight[0, 16:] = 0.003

    quantized = fake_quantize_nvfp4(weight)

    torch.testing.assert_close(quantized[0, 16:], torch.full((16,), 0.0029296875))
    torch.testing.assert_close(quantized[0, 0], weight[0, 0])


def test_nvfp4_masks_gradient_for_values_saturated_by_scale_rounding():
    weight = torch.zeros(1, 32, requires_grad=True)
    with torch.no_grad():
        weight[0, 0] = 2688.0
        weight[0, 16] = 5.8

    quantized = fake_quantize_nvfp4(weight)
    quantized.sum().backward()

    torch.testing.assert_close(quantized[0, 16], torch.tensor(5.625))
    torch.testing.assert_close(weight.grad[0, 0], torch.tensor(1.0))
    torch.testing.assert_close(weight.grad[0, 16], torch.tensor(0.0))


def test_nvfp4_zero_weight():
    weight = torch.zeros(2, 17, requires_grad=True)
    quantized = fake_quantize_nvfp4(weight)

    torch.testing.assert_close(quantized, weight)
    quantized.sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))


def test_e2m1_packing_uses_even_column_low_nibble():
    indices = torch.tensor([[0x1, 0xA, 0x7, 0xF]], dtype=torch.uint8)

    packed = pack_e2m1_rows(indices)

    torch.testing.assert_close(packed, torch.tensor([[0xA1, 0xF7]], dtype=torch.uint8))
    torch.testing.assert_close(unpack_e2m1_rows(packed), indices)


def test_standard_nvfp4_packing_matches_fake_quantization():
    torch.manual_seed(7)
    weight = torch.randn(3, 32, dtype=torch.bfloat16)

    result = quantize_nvfp4(weight)
    dequantized = dequantize_nvfp4(result.packed_weight, result.weight_scale, result.weight_scale_2)

    assert result.packed_weight.dtype == torch.uint8
    assert result.weight_scale.dtype == torch.float8_e4m3fn
    assert result.weight_scale_2.dtype == torch.float32
    torch.testing.assert_close(result.reconstruction, fake_quantize_nvfp4(weight), atol=0, rtol=0)
    torch.testing.assert_close(dequantized.to(weight.dtype), result.reconstruction, atol=0, rtol=0)


def test_standard_nvfp4_shared_family_scale_matches_fake_quantization():
    q_weight = torch.randn(4, 32) * 0.2
    k_weight = torch.randn(2, 32) * 0.5
    v_weight = torch.randn(2, 32)
    shared_amax = torch.stack([weight.abs().amax() for weight in (q_weight, k_weight, v_weight)]).amax()

    results = [quantize_nvfp4(weight, tensor_amax=shared_amax) for weight in (q_weight, k_weight, v_weight)]

    assert all(torch.equal(result.weight_scale_2, results[0].weight_scale_2) for result in results[1:])
    for weight, result in zip((q_weight, k_weight, v_weight), results):
        torch.testing.assert_close(
            result.reconstruction,
            fake_quantize_nvfp4(weight, tensor_amax=shared_amax),
            atol=0,
            rtol=0,
        )


def test_nvfp4_linear_preserves_activation_and_parameter_names():
    linear = nn.Linear(17, 4, bias=True)
    converted = NVFP4Linear.from_linear(linear)
    activation = torch.randn(2, 17, dtype=linear.weight.dtype)

    output = converted(activation)

    assert output.shape == (2, 4)
    assert set(converted.state_dict()) == {"weight", "bias"}
    assert converted.weight is linear.weight


def test_nvfp4_linear_tracks_training_input_running_max_only():
    converted = NVFP4Linear.from_linear(nn.Linear(16, 4, bias=False))
    converted._track_input_amax = True
    converted.train()

    converted(torch.full((2, 16), 2.0))
    converted(torch.full((2, 16), 1.0))

    torch.testing.assert_close(converted._input_amax, torch.tensor(2.0))
    converted.eval()
    converted(torch.full((2, 16), 5.0))
    torch.testing.assert_close(converted._input_amax, torch.tensor(2.0))


class _MLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(16, 32, bias=False)
        self.up_proj = nn.Linear(16, 32, bias=False)
        self.down_proj = nn.Linear(32, 16, bias=False)


class _FullAttnBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 32, bias=False)
        self.k_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.o_proj = nn.Linear(16, 16, bias=False)


class _LinearAttnBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(16, 48, bias=False)
        self.in_proj_z = nn.Linear(16, 16, bias=False)
        self.in_proj_b = nn.Linear(16, 4, bias=False)
        self.in_proj_a = nn.Linear(16, 4, bias=False)
        self.out_proj = nn.Linear(16, 16, bias=False)
        self.conv1d = nn.Conv1d(48, 48, 4, groups=48, bias=False)


class _FullLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FullAttnBlock()
        self.mlp = _MLPBlock()


class _LinearLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = _LinearAttnBlock()
        self.mlp = _MLPBlock()


class _VisualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_proj = nn.Linear(16, 16)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 16)
        self.layers = nn.ModuleList([_LinearLayer(), _FullLayer()])
        self.norm = nn.LayerNorm(16)


class _InnerQwen35Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()
        self.visual = nn.ModuleList([_VisualBlock()])


class _MtpModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_proj = nn.Linear(16, 16)


class _Qwen35Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3_5",
            num_experts=None,
            tie_word_embeddings=False,
            get_text_config=lambda: SimpleNamespace(num_experts=None, tie_word_embeddings=False),
        )
        self.model = _InnerQwen35Model()
        self.mtp = _MtpModule()
        self.lm_head = nn.Linear(16, 32)


_QWEN35_QAD_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
]


def test_apply_fake_quant_selects_text_backbone_and_excludes_vision_mtp():
    model = _Qwen35Model()
    keys_before = set(model.state_dict())

    converted = apply_fake_quant(model, _QWEN35_QAD_TARGETS, fused_runtime_scales=True)

    assert set(model.state_dict()) == keys_before
    assert set(converted) == {
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.mlp.up_proj",
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.1.self_attn.q_proj",
        "model.language_model.layers.1.self_attn.k_proj",
        "model.language_model.layers.1.self_attn.v_proj",
        "model.language_model.layers.1.self_attn.o_proj",
        "model.language_model.layers.1.mlp.gate_proj",
        "model.language_model.layers.1.mlp.up_proj",
        "model.language_model.layers.1.mlp.down_proj",
    }
    assert not isinstance(model.model.visual[0].out_proj, NVFP4Linear)
    assert not isinstance(model.mtp.out_proj, NVFP4Linear)
    assert isinstance(model.model.language_model.layers[0].linear_attn.conv1d, nn.Conv1d)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {f"{name}.weight" for name in converted}


def test_upcast_trainable_to_fp32_gives_latent_weights_an_fp32_master():
    """A BF16 latent weight cannot take a 1e-6 step: the update is below half a BF16 ulp
    for any weight above ~2.5e-4, so training silently stays at RTN. The student loader
    must hand the trainer FP32 masters for exactly the trainable parameters."""

    from qatfactory.models.llm import upcast_trainable_to_fp32

    model = _Qwen35Model().to(torch.bfloat16)
    converted = apply_fake_quant(model, _QWEN35_QAD_TARGETS, fused_runtime_scales=True)
    originals = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    upcast = upcast_trainable_to_fp32(model)

    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    assert upcast == sum(p.numel() for p in trainable.values()) > 0
    assert set(trainable) == {f"{name}.weight" for name in converted}
    assert all(p.dtype == torch.float32 for p in trainable.values())
    assert all(p.dtype == torch.bfloat16 for p in frozen.values())
    # the FP32 master holds exactly the loaded values, not a re-initialisation
    assert all(torch.equal(trainable[n], originals[n].float()) for n in trainable)
    # ... and a 1e-6 step lands on every element, where the BF16 original mostly rounds it away
    name, weight = next(iter(trainable.items()))
    bf16_stepped = originals[name] + 1e-6
    assert (bf16_stepped == originals[name]).float().mean() > 0.9
    with torch.no_grad():
        weight.add_(1e-6)
    assert torch.all(weight != originals[name].float())
    # idempotent
    assert upcast_trainable_to_fp32(model) == 0


def test_apply_fake_quant_rejects_moe_model_without_fused_experts():
    """A config that declares experts but exposes no fused 3D expert modules is rejected
    instead of silently quantizing only the dense linears."""
    model = _Qwen35Model()
    model.config.get_text_config = lambda: SimpleNamespace(num_experts=8, tie_word_embeddings=False)
    with pytest.raises(ValueError, match="no fused routed-expert modules"):
        apply_fake_quant(model, ["q_proj"])


def test_fused_runtime_scales_allow_partial_qkv():
    # Gemma-4 `full_attention` global layers reuse the key as the value (k_eq_v, no
    # v_proj); the serving runtime fuses whichever of q/k/v exist, so a partial QKV
    # subset is grouped as its own family instead of being rejected.
    families = build_nvfp4_fused_family_names(["model.layers.0.self_attn.q_proj"])
    assert families == [("model.layers.0.self_attn.q_proj",)]


def test_fused_runtime_scales_reject_partial_family():
    # Every other fused family (gate/up MLP, delta-net in_proj) must still be complete.
    with pytest.raises(ValueError, match="Incomplete fused NVFP4 family"):
        build_nvfp4_fused_family_names(["model.layers.0.mlp.gate_proj"])


def test_qwen3_5_native_export_targets_all_qad_linears():
    config = {
        "model_type": "qwen3_5",
        "text_config": {"layer_types": ["linear_attention"] * 24 + ["full_attention"] * 8},
    }
    targets = _expected_targets(config)
    families = build_nvfp4_fused_family_names(sorted(targets))

    assert len(targets) == 248
    assert set().union(*map(set, families)) == targets


def test_fused_runtime_state_reduces_input_maxima():
    model = _Qwen35Model()
    apply_fake_quant(model, _QWEN35_QAD_TARGETS, fused_runtime_scales=True)
    refresh_nvfp4_weight_amax(model)
    expected_scales = {}
    for group_index, group in enumerate(model._nvfp4_amax_groups, start=1):
        for member_index, module in enumerate(group.members, start=1):
            module._input_amax = torch.tensor(float(group_index + member_index))
        family_max = max(module._input_amax for module in group.members)
        for name in group.member_names:
            expected_scales[f"{name}.input_scale"] = family_max / NVFP4_SCALE_DENOMINATOR

    input_scales = finalize_nvfp4_input_scales(model)

    assert input_scales.keys() == expected_scales.keys()
    for key, expected in expected_scales.items():
        torch.testing.assert_close(input_scales[key], expected)


def test_native_vllm_export_writes_compressed_tensors_layout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {"model_type": "qwen3", "num_hidden_layers": 1, "dtype": "bfloat16"}
    targets = _expected_targets(config)
    target_list = sorted(targets)
    config["qatfactory_qad"] = {
        "format_version": 1,
        "quantizer": "standard_max",
        "fused_runtime_scales": True,
        "input_scale_source": "training_running_max",
        "target_modules": target_list,
        "global_step": 2,
    }
    (source / "config.json").write_text(json.dumps(config))
    tensors = {f"{module}.weight": torch.randn(2, 16, dtype=torch.bfloat16) for module in targets}
    tensors["lm_head.weight"] = torch.randn(4, 16, dtype=torch.bfloat16)
    save_file(tensors, source / "model.safetensors")
    save_file({f"{module}.input_scale": torch.tensor(0.25) for module in targets}, source / "input_scales.safetensors")
    output = tmp_path / "output"

    summary = export_nvfp4_vllm_checkpoint(source=source, model_assets=source, output=output, device="cpu", shard_size=1_000)

    store = SafetensorDirectory(output)
    assert summary["quantized_linears"] == 7
    assert summary["nonquantized_tensors"] == 1
    assert "model.layers.0.self_attn.q_proj.weight_packed" in store.keys()
    assert "model.layers.0.self_attn.q_proj.weight" not in store.keys()
    assert store.get_dtype_shape("model.layers.0.self_attn.q_proj.weight_packed")[0] == "U8"
    assert store.get_dtype_shape("model.layers.0.self_attn.q_proj.weight_scale")[0] == "F8_E4M3"
    qkv_scales = [
        store.get_tensor(f"model.layers.0.self_attn.{name}.weight_global_scale") for name in ("q_proj", "k_proj", "v_proj")
    ]
    assert all(torch.equal(scale, qkv_scales[0]) for scale in qkv_scales[1:])
    assert all(f"{module}.input_global_scale" in store.keys() for module in target_list)
    exported_config = json.loads((output / "config.json").read_text())
    assert exported_config["quantization_config"]["format"] == "nvfp4-pack-quantized"
    assert exported_config["quantization_config"]["quant_method"] == "compressed-tensors"
    group = exported_config["quantization_config"]["config_groups"]["group_0"]
    assert group["input_activations"]["dynamic"] == "local"


def test_native_vllm_export_rejects_checkpoint_without_qad_metadata(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 1, "dtype": "bfloat16"}))

    with pytest.raises(ValueError, match="missing qatfactory_qad"):
        export_nvfp4_vllm_checkpoint(source=source, model_assets=source, output=tmp_path / "output", device="cpu")


def _write_partial_target_checkpoint(source, trained):
    """A qwen3 checkpoint whose run quantized only ``trained`` of the quantizable linears."""

    source.mkdir()
    config = {"model_type": "qwen3", "num_hidden_layers": 1, "dtype": "bfloat16"}
    quantizable = _expected_targets(config)
    config["qatfactory_qad"] = {
        "format_version": 1,
        "quantizer": "standard_max",
        "fused_runtime_scales": True,
        "input_scale_source": "training_running_max",
        "target_modules": sorted(trained),
        "global_step": 2,
    }
    (source / "config.json").write_text(json.dumps(config))
    tensors = {f"{module}.weight": torch.randn(2, 16, dtype=torch.bfloat16) for module in quantizable}
    tensors["lm_head.weight"] = torch.randn(4, 16, dtype=torch.bfloat16)
    save_file(tensors, source / "model.safetensors")
    save_file({f"{module}.input_scale": torch.tensor(0.25) for module in trained}, source / "input_scales.safetensors")
    return quantizable


def test_native_vllm_export_honors_a_partially_quantized_run(tmp_path):
    # An MLP-only recipe (Gemma-4's default) leaves attention in BF16: those linears must be
    # copied through and declared in the engine's `ignore` list, not demanded as packed.
    source = tmp_path / "source"
    quantizable = _write_partial_target_checkpoint(source, {m for m in _expected_targets({"model_type": "qwen3", "num_hidden_layers": 1}) if ".mlp." in m})
    output = tmp_path / "output"

    summary = export_nvfp4_vllm_checkpoint(source=source, model_assets=source, output=output, device="cpu", shard_size=1_000)

    store = SafetensorDirectory(output)
    assert summary["quantized_linears"] == 3
    assert "model.layers.0.mlp.gate_proj.weight_packed" in store.keys()
    assert "model.layers.0.self_attn.q_proj.weight" in store.keys()
    assert "model.layers.0.self_attn.q_proj.weight_packed" not in store.keys()
    exported_config = json.loads((output / "config.json").read_text())
    ignore = exported_config["quantization_config"]["ignore"]
    attention = sorted(m for m in quantizable if ".self_attn." in m)
    # The whole attention block is unquantized -> one parent regex, so vLLM's fused qkv_proj
    # sees the same answer for every shard (including shards the checkpoint lacks).
    assert "re:.*\\.layers\\.0\\.self_attn\\..*" in ignore
    assert not any(".mlp." in entry for entry in ignore)
    export_summary = json.loads((output / "nvfp4_export.json").read_text())
    assert export_summary["unquantized_modules"] == attention


def test_unquantized_ignore_entries_regex_for_whole_blocks_and_names_for_partial_ones():
    quantizable = {
        "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj",
    }
    whole_block = unquantized_ignore_entries(sorted(m for m in quantizable if ".self_attn." in m), quantizable)
    assert whole_block == ["re:.*\\.layers\\.0\\.self_attn\\..*"]
    assert re.match(whole_block[0][3:], "language_model.model.layers.0.self_attn.v_proj")
    assert not re.match(whole_block[0][3:], "language_model.model.layers.10.self_attn.v_proj")
    partial = unquantized_ignore_entries(["model.layers.0.mlp.down_proj"], quantizable)
    assert partial == ["model.layers.0.mlp.down_proj"]
    assert unquantized_ignore_entries([], quantizable) == []


def test_native_vllm_export_rejects_targets_outside_the_architecture(tmp_path):
    source = tmp_path / "source"
    _write_partial_target_checkpoint(source, {"model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj", "model.layers.7.mlp.down_proj"})
    with pytest.raises(ValueError, match="unknown target modules"):
        export_nvfp4_vllm_checkpoint(source=source, model_assets=source, output=tmp_path / "output", device="cpu")


# --- W4A4: static-global-scale activation quantization ---


def test_nvfp4_w4a4_bootstrap_commit_and_step_static_quantization():
    torch.manual_seed(4)
    module = NVFP4Linear.from_linear(nn.Linear(16, 4, bias=False))
    module._track_input_amax = True
    module._quantize_activations = True
    module.train()

    first = torch.full((2, 16), 2.0)
    module(first)
    torch.testing.assert_close(module._input_amax_static, torch.tensor(2.0))  # bootstrap
    torch.testing.assert_close(module._input_amax, torch.tensor(2.0))

    # A louder batch grows the running max but the quantizer keeps reading the static
    # scale committed for this step (recompute determinism).
    second = torch.full((2, 16), 4.0)
    output = module(second)
    torch.testing.assert_close(module._input_amax, torch.tensor(4.0))
    torch.testing.assert_close(module._input_amax_static, torch.tensor(2.0))
    expected = nn.functional.linear(
        fake_quantize_nvfp4_activation(second, torch.tensor(2.0)),
        fake_quantize_nvfp4(module.weight),
    )
    assert torch.equal(output, expected)

    # The step boundary publishes running -> static, without aliasing.
    commit_nvfp4_input_scales(module)
    torch.testing.assert_close(module._input_amax_static, torch.tensor(4.0))
    module._input_amax.mul_(10.0)
    torch.testing.assert_close(module._input_amax_static, torch.tensor(4.0))


def test_nvfp4_w4a4_activation_saturates_and_masks_beyond_static_scale():
    activation = torch.zeros(1, 16, requires_grad=True)
    with torch.no_grad():
        activation[0, 0] = 12.0  # beyond the static amax of 6.0
        activation[0, 1] = 3.0

    quantized = fake_quantize_nvfp4_activation(activation, torch.tensor(6.0))
    quantized.sum().backward()

    assert quantized[0, 0].item() == pytest.approx(6.0)  # saturated at the grid top
    assert activation.grad[0, 0].item() == 0.0  # masked (contrast MXFP4: never)
    assert activation.grad[0, 1].item() == 1.0


def test_nvfp4_runtime_state_roundtrip_restores_the_static_scale():
    saved_module = NVFP4Linear.from_linear(nn.Linear(16, 4, bias=False))
    saved_module._track_input_amax = True
    saved_module._input_amax = torch.tensor(3.5)
    state = nvfp4_runtime_state_dict(saved_module)

    restored_module = NVFP4Linear.from_linear(nn.Linear(16, 4, bias=False))
    restored_module._track_input_amax = True
    load_nvfp4_runtime_state_dict(restored_module, state)

    torch.testing.assert_close(restored_module._input_amax, torch.tensor(3.5))
    # W4A4 resume: the restored running max seeds the static scale (no aliasing).
    torch.testing.assert_close(restored_module._input_amax_static, torch.tensor(3.5))
    restored_module._input_amax.mul_(2.0)
    torch.testing.assert_close(restored_module._input_amax_static, torch.tensor(3.5))


def test_apply_fake_quant_nvfp4_w4a4_requires_fused_runtime_scales():
    with pytest.raises(ValueError, match="requires fused_runtime_scales"):
        apply_fake_quant(
            _Qwen35Model(),
            _QWEN35_QAD_TARGETS,
            quantize_activations=True,
            fused_runtime_scales=False,
        )

    model = _Qwen35Model()
    converted = apply_fake_quant(
        model,
        _QWEN35_QAD_TARGETS,
        fused_runtime_scales=True,
        quantize_activations=True,
    )
    flagged = [module for module in model.modules() if isinstance(module, NVFP4Linear)]
    assert len(flagged) == len(converted.linears)
    assert all(module._quantize_activations for module in flagged)


def test_nvfp4_w4a4_gradient_checkpoint_recompute_matches_direct_backward():
    # torch.utils.checkpoint replays the forward during backward. The replay observes a
    # LOUDER running max than the original forward did (the running max grew in the
    # original pass), so quantizing with the live running max would drift; the
    # step-static scale is what makes the recompute bit-identical.
    torch.manual_seed(6)
    module = NVFP4Linear.from_linear(nn.Linear(64, 8, bias=False))
    module._track_input_amax = True
    module._quantize_activations = True
    module.train()
    module(torch.full((2, 64), 0.5))  # bootstrap the static scale below the test batch's amax

    activation = torch.randn(4, 64)
    direct = module(activation).square().sum()
    direct.backward()
    direct_grad = module.weight.grad.clone()
    static_after_direct = module._input_amax_static.clone()

    module.weight.grad = None
    checkpointed = torch.utils.checkpoint.checkpoint(module, activation, use_reentrant=False)
    checkpointed.square().sum().backward()

    assert torch.equal(module.weight.grad, direct_grad)
    # The recompute must not have moved the static scale either.
    torch.testing.assert_close(module._input_amax_static, static_after_direct)


def _w4a4_two_rank_scale_sync_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        module = NVFP4Linear.from_linear(nn.Linear(16, 4, bias=False))
        module._track_input_amax = True
        module._quantize_activations = True
        module.train()

        # Rank-dependent first batch: rank 0 amax 2.0, rank 1 amax 8.0. The bootstrap
        # must all-reduce, so both ranks start from the global max (8.0).
        module(torch.full((2, 16), 2.0 if rank == 0 else 8.0))
        bootstrap_static = module._input_amax_static.clone()

        # A louder rank-dependent second batch grows rank 1's local running max to
        # 16.0; the step-boundary commit must reconcile it to 16.0 on BOTH ranks.
        module(torch.full((2, 16), 3.0 if rank == 0 else 16.0))
        commit_nvfp4_input_scales(module)

        torch.save(
            {
                "bootstrap_static": bootstrap_static,
                "committed_static": module._input_amax_static.clone(),
                "running": module._input_amax.clone(),
            },
            f"{result_dir}/rank{rank}.pt",
        )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="requires torch.distributed")
def test_nvfp4_w4a4_static_scales_are_rank_synchronized(tmp_path):
    # Every rank must quantize activations with the SAME static scale, derived from
    # the global (cross-rank) running max — both at bootstrap and at every per-step
    # commit — matching the single input_global_scale the export finalizes.
    world_size = 2
    torch.multiprocessing.spawn(
        _w4a4_two_rank_scale_sync_worker,
        args=(world_size, str(tmp_path / "dist_init"), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    for rank in range(world_size):
        result = torch.load(tmp_path / f"rank{rank}.pt")
        torch.testing.assert_close(result["bootstrap_static"], torch.tensor(8.0))
        torch.testing.assert_close(result["committed_static"], torch.tensor(16.0))
        torch.testing.assert_close(result["running"], torch.tensor(16.0))


def test_native_vllm_export_reads_lora_base_weights_from_model_assets(tmp_path):
    """An adapters-only LoRA run exports like the equivalent full checkpoint of merged weights."""
    from qatfactory.quant.formats.lora import merge_lora_weight

    config = {"model_type": "qwen3", "num_hidden_layers": 1, "dtype": "bfloat16"}
    targets = sorted(_expected_targets(config))
    metadata = {
        "format_version": 1,
        "quantizer": "standard_max",
        "fused_runtime_scales": True,
        "input_scale_source": "training_running_max",
        "target_modules": targets,
        "global_step": 2,
    }
    generator = torch.Generator().manual_seed(0)
    base = {f"{module}.weight": torch.randn(2, 16, generator=generator).to(torch.bfloat16) for module in targets}
    adapters = {}
    for module in targets:
        adapters[f"{module}.lora_A"] = torch.randn(4, 16, generator=generator).to(torch.bfloat16)
        adapters[f"{module}.lora_B"] = torch.randn(2, 4, generator=generator).to(torch.bfloat16)
    scales = {f"{module}.input_scale": torch.tensor(0.25) for module in targets}

    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "config.json").write_text(json.dumps(config))
    # Released checkpoints may keep some frozen tensors in FP32; they are carried as BF16.
    save_file({**base, "lm_head.weight": torch.randn(4, 16), "model.norm.weight": torch.ones(16)}, assets / "model.safetensors")

    lora_run = tmp_path / "lora_run"
    lora_run.mkdir()
    (lora_run / "config.json").write_text(json.dumps({**config, "qatfactory_qad": {**metadata, "lora": {"rank": 4, "alpha": 8, "scaling": 2.0}}}))
    save_file(adapters, lora_run / "lora_adapters.safetensors")
    save_file(scales, lora_run / "input_scales.safetensors")

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text(json.dumps({**config, "qatfactory_qad": metadata}))
    merged_weights = {
        f"{module}.weight": merge_lora_weight(base[f"{module}.weight"], adapters[f"{module}.lora_A"], adapters[f"{module}.lora_B"], 2.0)
        for module in targets
    }
    save_file(merged_weights, merged / "model.safetensors")
    save_file(scales, merged / "input_scales.safetensors")

    export_nvfp4_vllm_checkpoint(source=lora_run, model_assets=assets, output=tmp_path / "out_lora", device="cpu")
    export_nvfp4_vllm_checkpoint(source=merged, model_assets=assets, output=tmp_path / "out_merged", device="cpu")

    lora_store, merged_store = SafetensorDirectory(tmp_path / "out_lora"), SafetensorDirectory(tmp_path / "out_merged")
    assert lora_store.keys() == merged_store.keys()
    for key in sorted(lora_store.keys()):
        assert torch.equal(lora_store.get_tensor(key), merged_store.get_tensor(key)), key
    assert lora_store.get_tensor("model.norm.weight").dtype == torch.bfloat16
    assert not any(key.endswith((".lora_A", ".lora_B")) for key in lora_store.keys())
