"""Startup-gate tests: architecture match, frozen-tensor match, and pinned activation scales."""

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from qatfactory.quant.fake_quant import apply_fake_quant
from qatfactory.quant.formats.nvfp4 import NVFP4_SCALE_DENOMINATOR, NVFP4Linear
from qatfactory.rl.model import (
    pin_activation_scales,
    read_export_input_amax,
    validate_rollout_export,
    verify_frozen_weights,
)


BASE_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "vocab_size": 128,
}


class _Config:
    num_experts = None
    tie_word_embeddings = False


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(32, 64, bias=False)
        self.up_proj = nn.Linear(32, 64, bias=False)


class _CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()
        self.config = _Config()


def _write_checkpoint(directory, tensors, config):
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _nvfp4_config(**overrides):
    config = dict(BASE_CONFIG, **overrides)
    config["quantization_config"] = {"format": "nvfp4-pack-quantized"}
    return config


def test_validate_rollout_export_rejects_an_unquantized_rollout_model(tmp_path):
    _write_checkpoint(tmp_path / "latents", {"model.norm.weight": torch.ones(4)}, BASE_CONFIG)
    _write_checkpoint(tmp_path / "rollout", {"model.norm.weight": torch.ones(4)}, BASE_CONFIG)

    with pytest.raises(ValueError, match="not an NVFP4 checkpoint"):
        validate_rollout_export(str(tmp_path / "latents"), str(tmp_path / "rollout"))


def test_validate_rollout_export_rejects_an_architecture_mismatch(tmp_path):
    _write_checkpoint(tmp_path / "latents", {"model.norm.weight": torch.ones(4)}, BASE_CONFIG)
    _write_checkpoint(
        tmp_path / "rollout", {"model.norm.weight": torch.ones(4)}, _nvfp4_config(num_hidden_layers=2)
    )

    with pytest.raises(ValueError, match="different architectures"):
        validate_rollout_export(str(tmp_path / "latents"), str(tmp_path / "rollout"))


def test_verify_frozen_weights_accepts_matching_checkpoints(tmp_path):
    frozen = {"model.norm.weight": torch.ones(4), "model.embed_tokens.weight": torch.arange(8.0)}
    _write_checkpoint(tmp_path / "latents", {**frozen, "model.mlp.gate_proj.weight": torch.zeros(4, 4)}, BASE_CONFIG)
    _write_checkpoint(
        tmp_path / "rollout",
        {**frozen, "model.mlp.gate_proj.weight_packed": torch.zeros(4, 2, dtype=torch.uint8)},
        _nvfp4_config(),
    )

    assert verify_frozen_weights(str(tmp_path / "latents"), str(tmp_path / "rollout")) == len(frozen)


def test_verify_frozen_weights_rejects_a_differing_frozen_tensor(tmp_path):
    _write_checkpoint(tmp_path / "latents", {"model.norm.weight": torch.ones(4)}, BASE_CONFIG)
    _write_checkpoint(tmp_path / "rollout", {"model.norm.weight": torch.zeros(4)}, _nvfp4_config())

    with pytest.raises(ValueError, match="differs between the training latents"):
        verify_frozen_weights(str(tmp_path / "latents"), str(tmp_path / "rollout"))


def test_read_export_input_amax_inverts_the_stored_global_scale(tmp_path):
    amax = torch.tensor(3.5)
    _write_checkpoint(
        tmp_path / "rollout",
        {"mlp.gate_proj.input_global_scale": (NVFP4_SCALE_DENOMINATOR / amax).reshape(1)},
        _nvfp4_config(),
    )

    recovered = read_export_input_amax(str(tmp_path / "rollout"))

    assert set(recovered) == {"mlp.gate_proj"}
    assert torch.allclose(recovered["mlp.gate_proj"], amax)


def test_pin_activation_scales_freezes_the_running_maximum():
    torch.manual_seed(0)
    model = _CausalLM()
    apply_fake_quant(model, ("gate_proj", "up_proj"), quant_format="nvfp4", fused_runtime_scales=True)
    export_amax = {"mlp.gate_proj": torch.tensor(2.0), "mlp.up_proj": torch.tensor(4.0)}

    pin_activation_scales(model, export_amax)

    gate = model.mlp.gate_proj
    assert torch.equal(gate._input_amax_static, torch.tensor(2.0))
    assert not gate._track_input_amax
    # A forward with far larger activations must not move the pinned scale: it is the
    # deployment's static scale, and the saved checkpoint has to keep describing it.
    gate(torch.full((2, 32), 50.0))
    assert torch.equal(gate._input_amax, torch.tensor(2.0))
    assert torch.equal(gate._input_amax_static, torch.tensor(2.0))


def test_pin_activation_scales_requires_a_scale_for_every_linear():
    model = _CausalLM()
    apply_fake_quant(model, ("gate_proj", "up_proj"), quant_format="nvfp4", fused_runtime_scales=True)

    with pytest.raises(ValueError, match="no activation scale"):
        pin_activation_scales(model, {"mlp.gate_proj": torch.tensor(2.0)})


def test_pinned_scale_drives_w4a4_activation_quantization():
    torch.manual_seed(0)
    model = _CausalLM()
    apply_fake_quant(
        model,
        ("gate_proj", "up_proj"),
        quant_format="nvfp4",
        fused_runtime_scales=True,
        quantize_activations=True,
    )
    pin_activation_scales(model, {"mlp.gate_proj": torch.tensor(2.0), "mlp.up_proj": torch.tensor(2.0)})

    gate = model.mlp.gate_proj
    assert isinstance(gate, NVFP4Linear)
    # Values beyond the pinned static scale saturate rather than widening the scale.
    gate(torch.full((2, 32), 100.0))
    assert torch.equal(gate._input_amax_static, torch.tensor(2.0))
