import json
from types import SimpleNamespace

import pytest
import torch
from reference_mxfp4 import qdq_mxfp4_torch
from safetensors.torch import save_file
from torch import nn

from qatfactory.quant import (
    MXFP4Linear,
    apply_fake_quant,
    dequantize_mxfp4,
    fake_quantize_mxfp4,
    fake_quantize_mxfp4_activation,
    quantize_mxfp4,
)
from qatfactory.quant.export.mxfp4_vllm import export_mxfp4_vllm_checkpoint
from qatfactory.quant.export.nvfp4_vllm import SafetensorDirectory, _expected_targets
from qatfactory.quant.formats.mxfp4 import E8M0_BIAS, _e8m0_exponent_ceil, _e8m0_exponent_even


def test_mxfp4_known_values_and_ste_gradient():
    weight = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 4], requires_grad=True)
    quantized = fake_quantize_mxfp4(weight)

    torch.testing.assert_close(quantized, weight)
    quantized.sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("magnitude", [1e-3, 1.0, 100.0])
def test_mxfp4_bit_matches_vllm_reference_on_random_weights(dtype, magnitude):
    torch.manual_seed(7)
    weight = (torch.randn(8, 128) * magnitude).to(dtype)

    quantized = fake_quantize_mxfp4(weight)

    assert quantized.dtype == dtype
    assert torch.equal(quantized, qdq_mxfp4_torch(weight))


def test_mxfp4_scale_rule_amax_fraction_regions():
    # f = 1.5: scale 2^(e-2), 6*scale == amax, lossless full-range fit.
    exact = torch.zeros(1, 32)
    exact[0, 0], exact[0, 1] = 1.5, 0.5
    quantized = fake_quantize_mxfp4(exact)
    torch.testing.assert_close(quantized, exact)

    # f in (1.5, 1.75): same scale, block max clips to 1.5*2^e.
    clipped = torch.zeros(1, 32)
    clipped[0, 0] = 1.6875
    torch.testing.assert_close(fake_quantize_mxfp4(clipped)[0, 0], torch.tensor(1.5))

    # f in [1.75, 2): amax promotes to scale 2^(e-1); 1.75/0.5 = 3.5 ties to 4.0 -> 2.0.
    promoted = torch.zeros(1, 32)
    promoted[0, 0] = 1.75
    torch.testing.assert_close(fake_quantize_mxfp4(promoted)[0, 0], torch.tensor(2.0))

    for weight in (exact, clipped, promoted):
        bf16 = weight.to(torch.bfloat16)
        assert torch.equal(fake_quantize_mxfp4(bf16), qdq_mxfp4_torch(bf16))


def test_mxfp4_e2m1_tie_values_round_to_even():
    ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    expected_codes = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
    weight = torch.zeros(1, 32)
    weight[0, : len(ties)] = torch.tensor(ties)
    weight[0, len(ties) : 2 * len(ties)] = -torch.tensor(ties)
    weight[0, -1] = 5.0  # block amax 5.0 -> f = 1.25 -> scale = 2^0, ties divide exactly

    quantized = fake_quantize_mxfp4(weight)

    torch.testing.assert_close(quantized[0, : len(ties)], torch.tensor(expected_codes))
    torch.testing.assert_close(quantized[0, len(ties) : 2 * len(ties)], -torch.tensor(expected_codes))
    bf16 = weight.to(torch.bfloat16)
    assert torch.equal(fake_quantize_mxfp4(bf16), qdq_mxfp4_torch(bf16))


def test_mxfp4_zero_blocks_and_zero_weight():
    weight = torch.zeros(2, 64, requires_grad=True)
    quantized = fake_quantize_mxfp4(weight)

    torch.testing.assert_close(quantized, weight)
    quantized.sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))

    mixed = torch.zeros(1, 64, dtype=torch.bfloat16)
    mixed[0, 32:] = torch.linspace(-4.0, 4.0, 32, dtype=torch.bfloat16)
    assert torch.equal(fake_quantize_mxfp4(mixed), qdq_mxfp4_torch(mixed))

    result = quantize_mxfp4(torch.zeros(2, 64, dtype=torch.bfloat16))
    assert torch.equal(result.weight_scale, torch.zeros(2, 2, dtype=torch.uint8))
    assert torch.equal(result.packed_weight, torch.zeros(2, 32, dtype=torch.uint8))


def test_mxfp4_pads_in_features_not_divisible_by_32():
    torch.manual_seed(11)
    weight = torch.randn(3, 40, dtype=torch.bfloat16)
    padded = torch.zeros(3, 64, dtype=torch.bfloat16)
    padded[:, :40] = weight

    quantized = fake_quantize_mxfp4(weight)

    assert quantized.shape == weight.shape
    assert torch.equal(quantized, qdq_mxfp4_torch(padded)[:, :40])


def test_mxfp4_masks_gradient_for_values_clipped_by_even_scale():
    weight = torch.zeros(1, 32, requires_grad=True)
    with torch.no_grad():
        weight[0, 0] = 1.6875  # f in (1.5, 1.75): EVEN rule clips the block max
        weight[0, 1] = 0.9

    quantized = fake_quantize_mxfp4(weight)
    quantized.sum().backward()

    torch.testing.assert_close(quantized[0, 0], torch.tensor(1.5))
    torch.testing.assert_close(quantized[0, 1], torch.tensor(1.0))
    torch.testing.assert_close(weight.grad[0, 0], torch.tensor(0.0))
    torch.testing.assert_close(weight.grad[0, 1], torch.tensor(1.0))


def test_mxfp4_e8m0_exponent_known_values():
    amax = torch.tensor([[0.0], [1.5], [4.0], [6.0], [7.0]])
    exponent = _e8m0_exponent_even(amax)

    torch.testing.assert_close(exponent.squeeze(-1), torch.tensor([-127.0, -2.0, 0.0, 0.0, 1.0]))
    result = quantize_mxfp4(torch.tensor([[1.5] + [0.0] * 31], dtype=torch.bfloat16))
    assert result.weight_scale.item() == E8M0_BIAS - 2


def test_mxfp4_packing_matches_fake_quantization():
    torch.manual_seed(3)
    weight = torch.randn(5, 96, dtype=torch.bfloat16)

    result = quantize_mxfp4(weight)
    dequantized = dequantize_mxfp4(result.packed_weight, result.weight_scale)

    assert result.packed_weight.dtype == torch.uint8
    assert result.packed_weight.shape == (5, 48)
    assert result.weight_scale.dtype == torch.uint8
    assert result.weight_scale.shape == (5, 3)
    torch.testing.assert_close(result.reconstruction, fake_quantize_mxfp4(weight), atol=0, rtol=0)
    torch.testing.assert_close(dequantized.to(weight.dtype), result.reconstruction, atol=0, rtol=0)


def test_mxfp4_quantize_rejects_non_divisible_in_features():
    with pytest.raises(ValueError, match="divisible"):
        quantize_mxfp4(torch.randn(2, 40, dtype=torch.bfloat16))


def test_mxfp4_linear_preserves_activation_and_parameter_names():
    linear = nn.Linear(17, 4, bias=True)
    converted = MXFP4Linear.from_linear(linear)
    activation = torch.randn(2, 17, dtype=linear.weight.dtype)

    output = converted(activation)

    assert output.shape == (2, 4)
    assert set(converted.state_dict()) == {"weight", "bias"}
    assert converted.weight is linear.weight


class _MLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(32, 64, bias=False)
        self.up_proj = nn.Linear(32, 64, bias=False)
        self.down_proj = nn.Linear(64, 32, bias=False)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3",
            num_experts=None,
            tie_word_embeddings=False,
            get_text_config=lambda: SimpleNamespace(num_experts=None, tie_word_embeddings=False),
        )
        self.embed_tokens = nn.Embedding(16, 32)
        self.mlp = _MLPBlock()
        self.lm_head = nn.Linear(32, 16)


def test_apply_fake_quant_mxfp4_converts_and_freezes():
    model = _TinyModel()

    converted = apply_fake_quant(model, ["gate_proj", "up_proj", "down_proj"], quant_format="mxfp4")

    assert converted.all == ["mlp.down_proj", "mlp.gate_proj", "mlp.up_proj"]
    assert converted.experts == ()
    assert isinstance(model.mlp.gate_proj, MXFP4Linear)
    assert not isinstance(model.lm_head, MXFP4Linear)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {f"{name}.weight" for name in converted}


def test_apply_fake_quant_mxfp4_rejects_fused_runtime_scales():
    with pytest.raises(ValueError, match="fused_runtime_scales"):
        apply_fake_quant(_TinyModel(), ["gate_proj"], quant_format="mxfp4", fused_runtime_scales=True)


def _write_mxfp4_source(tmp_path, *, global_step=2, in_features=32, with_metadata=True, trained=None):
    source = tmp_path / "source"
    source.mkdir()
    config = {"model_type": "qwen3", "num_hidden_layers": 1, "dtype": "bfloat16"}
    targets = _expected_targets(config)
    target_list = sorted(targets)
    if with_metadata:
        config["qatfactory_qad"] = {
            "format_version": 1,
            "quant_format": "mxfp4",
            "quantizer": "mx_even",
            "fused_runtime_scales": False,
            "input_scale_source": None,
            "target_modules": sorted(trained) if trained is not None else target_list,
            "global_step": global_step,
        }
    (source / "config.json").write_text(json.dumps(config))
    tensors = {f"{module}.weight": torch.randn(2, in_features, dtype=torch.bfloat16) for module in targets}
    tensors["lm_head.weight"] = torch.randn(4, in_features, dtype=torch.bfloat16)
    save_file(tensors, source / "model.safetensors")
    return source


def test_mxfp4_export_writes_compressed_tensors_layout(tmp_path):
    source = _write_mxfp4_source(tmp_path)
    output = tmp_path / "output"

    summary = export_mxfp4_vllm_checkpoint(
        source=source, model_assets=source, output=output, device="cpu", shard_size=1_000
    )

    store = SafetensorDirectory(output)
    assert summary["quantized_linears"] == 7
    assert summary["nonquantized_tensors"] == 1
    assert store.get_dtype_shape("model.layers.0.self_attn.q_proj.weight_packed") == ("U8", (2, 16))
    assert store.get_dtype_shape("model.layers.0.self_attn.q_proj.weight_scale") == ("U8", (2, 1))
    assert "model.layers.0.self_attn.q_proj.weight" not in store.keys()
    assert not any(key.endswith(("weight_global_scale", "input_global_scale", "input_scale")) for key in store.keys())

    quantization = json.loads((output / "config.json").read_text())["quantization_config"]
    assert quantization["format"] == "mxfp4-pack-quantized"
    assert quantization["quant_method"] == "compressed-tensors"
    weights_args = quantization["config_groups"]["group_0"]["weights"]
    # The five conditions of vLLM's CompressedTensorsConfig._is_mxfp4 dispatch.
    assert weights_args["strategy"] == "group"
    assert weights_args["symmetric"] is True
    assert weights_args["group_size"] == 32
    assert weights_args["type"] == "float"
    assert weights_args["num_bits"] == 4
    input_args = quantization["config_groups"]["group_0"]["input_activations"]
    assert input_args["dynamic"] is True and input_args["group_size"] == 32

    packed = store.get_tensor("model.layers.0.self_attn.q_proj.weight_packed")
    scale = store.get_tensor("model.layers.0.self_attn.q_proj.weight_scale")
    original = SafetensorDirectory(source).get_tensor("model.layers.0.self_attn.q_proj.weight")
    expected = fake_quantize_mxfp4(original)
    torch.testing.assert_close(dequantize_mxfp4(packed, scale).to(torch.bfloat16), expected, atol=0, rtol=0)


def test_mxfp4_export_honors_a_partially_quantized_run(tmp_path):
    quantizable = _expected_targets({"model_type": "qwen3", "num_hidden_layers": 1})
    mlp = {m for m in quantizable if ".mlp." in m}
    source = _write_mxfp4_source(tmp_path, trained=mlp)
    output = tmp_path / "output"

    summary = export_mxfp4_vllm_checkpoint(source=source, model_assets=source, output=output, device="cpu", shard_size=1_000)

    store = SafetensorDirectory(output)
    assert summary["quantized_linears"] == 3
    assert "model.layers.0.mlp.down_proj.weight_packed" in store.keys()
    assert "model.layers.0.self_attn.q_proj.weight" in store.keys()
    ignore = json.loads((output / "config.json").read_text())["quantization_config"]["ignore"]
    assert "re:.*\\.layers\\.0\\.self_attn\\..*" in ignore
    assert not any(".mlp." in entry for entry in ignore)


def test_mxfp4_export_allows_step_zero_source(tmp_path):
    """Step-0 export is the QATFactory-side RTN baseline (quantize the untrained latents)."""

    source = _write_mxfp4_source(tmp_path, global_step=0)

    summary = export_mxfp4_vllm_checkpoint(
        source=source, model_assets=source, output=tmp_path / "output", device="cpu"
    )

    assert summary["quantized_linears"] == 7


def test_mxfp4_export_rejects_checkpoint_without_qad_metadata(tmp_path):
    source = _write_mxfp4_source(tmp_path, with_metadata=False)

    with pytest.raises(ValueError, match="missing qatfactory_qad"):
        export_mxfp4_vllm_checkpoint(source=source, model_assets=source, output=tmp_path / "output", device="cpu")


def test_mxfp4_export_rejects_wrong_quant_format_metadata(tmp_path):
    source = _write_mxfp4_source(tmp_path)
    config = json.loads((source / "config.json").read_text())
    config["qatfactory_qad"]["quant_format"] = "nvfp4"
    config["qatfactory_qad"]["quantizer"] = "standard_max"
    (source / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="not eligible for MXFP4 export"):
        export_mxfp4_vllm_checkpoint(source=source, model_assets=source, output=tmp_path / "output", device="cpu")


def test_mxfp4_export_rejects_non_divisible_in_features(tmp_path):
    source = _write_mxfp4_source(tmp_path, in_features=40)

    with pytest.raises(ValueError, match="divisible"):
        export_mxfp4_vllm_checkpoint(source=source, model_assets=source, output=tmp_path / "output", device="cpu")


# --- W4A4: ceil-rule activation quantization (serving = flashinfer mxfp4_quantize cute-dsl) ---


def test_mxfp4_activation_ceil_rule_known_values():
    # amax = f * 2^e. The ceil rule returns the smallest power of two >= amax/6; it
    # diverges from the EVEN weight rule exactly on the f-in-(1.5, 1.75) band (EVEN
    # clips there). At boundary amaxes ({6, 3, 1.5} * 2^k) the FP32 product
    # amax * fp32(1/6) = power * (1 +- 2^-25) is within half an ulp of the exact power
    # and rounds back to it (RNE), so there is no bump — matching mathematical ceil
    # regardless of which 1-ulp rcp(6) variant the hardware uses.
    cases = {
        0.85: -2,  # divergence band: EVEN gives -3 (clips 0.85 -> 0.75); ceil never clips
        0.9: -2,  # f = 1.8 >= 1.75: both rules agree
        1.4: -2,  # f = 1.4 < 1.5: both rules agree
        1.5: -2,  # boundary amax: product rounds to exactly 2^-2 -> no bump, tight fit
        6.0: 0,  # boundary amax: product rounds to exactly 2^0 -> no bump, tight fit
        100.0: 5,  # 100/6 = 16.67 -> 2^5 = 32
    }
    amax = torch.tensor(list(cases.keys()), dtype=torch.float32)
    exponent, zero_block = _e8m0_exponent_ceil(amax)
    assert torch.equal(exponent, torch.tensor(list(cases.values()), dtype=torch.float32))
    assert not zero_block.any()
    # The never-clips invariant on every case: 6 * scale >= amax — with equality (a
    # lossless full-range fit) exactly at the boundary amaxes.
    assert bool((6.0 * torch.exp2(exponent) >= amax).all())
    assert 6.0 * torch.exp2(exponent[3]).item() == 1.5
    assert 6.0 * torch.exp2(exponent[4]).item() == 6.0
    # Divergence band vs the weight rule.
    assert _e8m0_exponent_even(torch.tensor(0.85)).item() == -3
    assert _e8m0_exponent_ceil(torch.tensor(0.85))[0].item() == -2

    # Zero and kernel-tiny blocks (q <= 2^-127) map to scale byte 0 -> all-zero codes.
    tiny = torch.tensor([0.0, 1e-38], dtype=torch.float32)
    _, tiny_zero = _e8m0_exponent_ceil(tiny)
    assert tiny_zero.all()


def test_mxfp4_activation_never_clips_and_passes_all_gradients():
    torch.manual_seed(0)
    for magnitude in (1e-3, 1.0, 100.0):
        activation = (torch.randn(8, 128, dtype=torch.bfloat16) * magnitude).requires_grad_()
        quantized = fake_quantize_mxfp4_activation(activation)

        blocks = activation.detach().float().reshape(8, -1, 32)
        block_amax = blocks.abs().amax(dim=-1, keepdim=True)
        exponent, zero_block = _e8m0_exponent_ceil(block_amax)
        assert bool((6.0 * torch.exp2(exponent) >= block_amax)[~zero_block].all())
        # Every quantized value is representable without saturation.
        assert bool((quantized.float().reshape(8, -1, 32).abs() <= 6.0 * torch.exp2(exponent)).all())

        # Unlike the EVEN weight rule, the STE mask is all-true: gradients pass unmasked.
        quantized.sum().backward()
        assert torch.equal(activation.grad, torch.ones_like(activation))


def test_mxfp4_activation_is_deterministic_and_shape_dtype_preserving():
    torch.manual_seed(1)
    activation = torch.randn(2, 5, 64, dtype=torch.bfloat16)
    first = fake_quantize_mxfp4_activation(activation.clone())
    second = fake_quantize_mxfp4_activation(activation.clone())
    # Pure function of x: same input -> bit-identical output (gradient-checkpoint
    # recompute determinism holds by construction, no step-static scale needed).
    assert torch.equal(first, second)
    assert first.shape == activation.shape
    assert first.dtype == activation.dtype
    with pytest.raises(ValueError, match="divisible by 32"):
        fake_quantize_mxfp4_activation(torch.randn(4, 33))


def test_mxfp4_activation_divergence_band_vs_weight_rule():
    # One block whose amax mantissa lies in (1.5, 1.75): the weight (EVEN) path clips
    # its max element and masks the gradient; the activation (ceil) path does neither.
    values = torch.zeros(1, 32)
    values[0, 0] = 0.85
    values[0, 1] = 0.4

    weight = values.clone().requires_grad_()
    weight_quantized = fake_quantize_mxfp4(weight)
    weight_quantized.sum().backward()
    assert weight_quantized[0, 0].item() == pytest.approx(0.75)  # clipped to 6 * 0.125
    assert weight.grad[0, 0].item() == 0.0  # masked

    activation = values.clone().requires_grad_()
    activation_quantized = fake_quantize_mxfp4_activation(activation)
    activation_quantized.sum().backward()
    assert activation_quantized[0, 0].item() == pytest.approx(0.75)  # 3 * 0.25, no clipping
    assert activation.grad[0, 0].item() == 1.0  # unmasked
    # Same reconstructed value here by coincidence of the grids; the scales differ.
    assert _e8m0_exponent_even(torch.tensor(0.85)).item() == -3
    assert _e8m0_exponent_ceil(torch.tensor(0.85))[0].item() == -2


def test_mxfp4_linear_w4a4_forward_quantizes_the_activation():
    torch.manual_seed(2)
    linear = nn.Linear(64, 8, bias=True)
    converted = MXFP4Linear.from_linear(linear)
    assert converted._quantize_activations is False
    activation = torch.randn(3, 64)

    weight_only = converted(activation)
    converted._quantize_activations = True
    w4a4 = converted(activation)

    expected = nn.functional.linear(
        fake_quantize_mxfp4_activation(activation), fake_quantize_mxfp4(converted.weight), converted.bias
    )
    assert torch.equal(w4a4, expected)
    assert not torch.equal(w4a4, weight_only)


def test_apply_fake_quant_mxfp4_w4a4_needs_no_runtime_scales_and_sets_flags():
    model = _TinyModel()
    converted = apply_fake_quant(
        model,
        ["gate_proj", "up_proj", "down_proj"],
        quant_format="mxfp4",
        quantize_activations=True,
    )
    flagged = [module for module in model.modules() if isinstance(module, MXFP4Linear)]
    assert len(flagged) == len(converted.linears) == 3
    assert all(module._quantize_activations for module in flagged)


def test_apply_fake_quant_rejects_w4a4_for_formats_without_activation_quant():
    with pytest.raises(ValueError, match="does not support input-activation fake-quant"):
        apply_fake_quant(
            _TinyModel(),
            ["gate_proj", "up_proj", "down_proj"],
            quant_format="q4_k",
            quantize_activations=True,
        )


def test_mxfp4_w4a4_gradient_checkpoint_recompute_matches_direct_backward():
    # torch.utils.checkpoint replays the forward during backward; the W4A4 activation
    # quantizer must make that replay bit-identical (for MXFP4 it is a pure function of
    # x, so this holds by construction). Grads through a checkpointed segment must equal
    # the direct ones bit-for-bit.
    torch.manual_seed(5)
    module = MXFP4Linear.from_linear(nn.Linear(64, 8, bias=False))
    module._quantize_activations = True
    module.train()
    activation = torch.randn(4, 64)

    direct = module(activation).square().sum()
    direct.backward()
    direct_grad = module.weight.grad.clone()

    module.weight.grad = None
    checkpointed = torch.utils.checkpoint.checkpoint(module, activation, use_reentrant=False)
    checkpointed.square().sum().backward()

    assert torch.equal(module.weight.grad, direct_grad)


def test_mxfp4_export_reads_lora_base_weights_from_model_assets(tmp_path):
    """An adapters-only LoRA run exports like the equivalent full checkpoint of merged weights."""
    from qatfactory.quant.formats.lora import merge_lora_weight

    config = {"model_type": "qwen3", "num_hidden_layers": 1, "dtype": "bfloat16"}
    targets = sorted(_expected_targets(config))
    metadata = {
        "format_version": 1,
        "quant_format": "mxfp4",
        "quantizer": "mx_even",
        "fused_runtime_scales": False,
        "input_scale_source": None,
        "target_modules": targets,
        "global_step": 2,
    }
    generator = torch.Generator().manual_seed(0)
    base = {f"{module}.weight": torch.randn(2, 32, generator=generator).to(torch.bfloat16) for module in targets}
    adapters = {}
    for module in targets:
        adapters[f"{module}.lora_A"] = torch.randn(4, 32, generator=generator).to(torch.bfloat16)
        adapters[f"{module}.lora_B"] = torch.randn(2, 4, generator=generator).to(torch.bfloat16)

    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "config.json").write_text(json.dumps(config))
    save_file({**base, "lm_head.weight": torch.randn(4, 32).to(torch.bfloat16)}, assets / "model.safetensors")

    lora_run = tmp_path / "lora_run"
    lora_run.mkdir()
    (lora_run / "config.json").write_text(json.dumps({**config, "qatfactory_qad": {**metadata, "lora": {"rank": 4, "alpha": 8, "scaling": 2.0}}}))
    save_file(adapters, lora_run / "lora_adapters.safetensors")

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text(json.dumps({**config, "qatfactory_qad": metadata}))
    save_file(
        {
            f"{module}.weight": merge_lora_weight(base[f"{module}.weight"], adapters[f"{module}.lora_A"], adapters[f"{module}.lora_B"], 2.0)
            for module in targets
        },
        merged / "model.safetensors",
    )

    export_mxfp4_vllm_checkpoint(source=lora_run, model_assets=assets, output=tmp_path / "out_lora", device="cpu")
    export_mxfp4_vllm_checkpoint(source=merged, model_assets=assets, output=tmp_path / "out_merged", device="cpu")

    lora_store, merged_store = SafetensorDirectory(tmp_path / "out_lora"), SafetensorDirectory(tmp_path / "out_merged")
    assert lora_store.keys() == merged_store.keys()
    for key in sorted(lora_store.keys()):
        assert torch.equal(lora_store.get_tensor(key), merged_store.get_tensor(key)), key
    assert not any(key.endswith((".lora_A", ".lora_B")) for key in lora_store.keys())
