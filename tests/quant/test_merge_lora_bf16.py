"""The merged BF16 checkpoint must be the same model the training forward simulated.

A weight-only run is only meaningful if what gets served equals what was trained: the merge
has to fold the adapters and land on the same NVFP4 grid, per expert, that
``NVFP4ExpertLoRA`` used in its student path. These tests pin that equality rather than the
merge's internals, so a change to either side that breaks the correspondence fails here.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from qatfactory.quant.formats.nvfp4 import fake_quantize_nvfp4
from qatfactory.quant.formats.nvfp4_experts import NVFP4ExpertLoRA
from scripts.merge_lora_bf16 import merge_checkpoint

from test_minimax_lora import _fp8_quantize_blocks, _tiny_experts


def _wrapped_experts(hybrid_shared: bool, rank: int = 4, alpha: float = 8.0):
    experts = _tiny_experts()
    gate_up_codes, gate_up_scale = _fp8_quantize_blocks(experts.gate_up_proj.detach())
    down_codes, down_scale = _fp8_quantize_blocks(experts.down_proj.detach())
    experts.gate_up_proj = torch.nn.Parameter(gate_up_codes, requires_grad=False)
    experts.down_proj = torch.nn.Parameter(down_codes, requires_grad=False)
    experts.register_buffer("gate_up_proj_scale_inv", gate_up_scale)
    experts.register_buffer("down_proj_scale_inv", down_scale)
    wrapped = NVFP4ExpertLoRA(experts, rank=rank, alpha=alpha, hybrid_shared=hybrid_shared)
    # Zero-initialised `B` factors would make the merge trivially correct, so give both
    # factors real values the way a trained checkpoint would have them.
    generator = torch.Generator().manual_seed(0)
    for name in ("gate_up_lora_A", "gate_up_lora_B", "down_lora_A", "down_lora_B"):
        parameter = getattr(wrapped, name)
        with torch.no_grad():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return wrapped


def _write_base(directory: Path, wrapped: NVFP4ExpertLoRA) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        "model.layers.0.mlp.experts.gate_up_proj": wrapped.gate_up_proj.detach(),
        "model.layers.0.mlp.experts.gate_up_proj_scale_inv": wrapped.gate_up_proj_scale_inv,
        "model.layers.0.mlp.experts.down_proj": wrapped.down_proj.detach(),
        "model.layers.0.mlp.experts.down_proj_scale_inv": wrapped.down_proj_scale_inv,
    }
    save_file(tensors, directory / "model.safetensors", metadata={"format": "pt"})
    (directory / "config.json").write_text(
        json.dumps({"model_type": "minimax", "quantization_config": {"quant_method": "fp8"}}) + "\n"
    )


def _write_adapters(path: Path, wrapped: NVFP4ExpertLoRA) -> None:
    prefix = "model.layers.0.mlp.experts"
    save_file(
        {
            f"{prefix}.{leaf}": getattr(wrapped, leaf).detach()
            for leaf in ("gate_up_lora_A", "gate_up_lora_B", "down_lora_A", "down_lora_B")
        },
        path,
        metadata={"format": "pt"},
    )


def _merged_tensors(tmp_path: Path, wrapped: NVFP4ExpertLoRA, alpha: float):
    base = tmp_path / "base"
    adapters = tmp_path / "adapters.safetensors"
    output = tmp_path / "merged"
    _write_base(base, wrapped)
    _write_adapters(adapters, wrapped)
    merge_checkpoint(base, adapters, output, lora_alpha=alpha, device="cpu", chunk=3)

    from safetensors import safe_open

    shard = next(output.glob("model-*.safetensors"))
    with safe_open(shard, framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}, output


@pytest.mark.parametrize("hybrid_shared", [True, False])
def test_merged_weights_equal_the_training_student_grid(tmp_path, hybrid_shared):
    alpha = 8.0
    wrapped = _wrapped_experts(hybrid_shared, alpha=alpha)
    merged, _ = _merged_tensors(tmp_path, wrapped, alpha)

    for expert in range(wrapped.num_experts):
        expected_gate_up = fake_quantize_nvfp4(wrapped.effective_gate_up(expert)).to(torch.bfloat16)
        expected_down = fake_quantize_nvfp4(wrapped.effective_down(expert)).to(torch.bfloat16)
        torch.testing.assert_close(
            merged["model.layers.0.mlp.experts.gate_up_proj"][expert], expected_gate_up,
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            merged["model.layers.0.mlp.experts.down_proj"][expert], expected_down,
            rtol=0, atol=0,
        )


def test_merged_checkpoint_carries_no_fp8_codes_or_scales(tmp_path):
    wrapped = _wrapped_experts(hybrid_shared=True)
    merged, output = _merged_tensors(tmp_path, wrapped, alpha=8.0)

    assert all(not key.endswith("_scale_inv") for key in merged)
    assert all(value.dtype is torch.bfloat16 for value in merged.values())
    config = json.loads((output / "config.json").read_text())
    assert "quantization_config" not in config
    assert config["dtype"] == "bfloat16"


def test_merge_without_adapters_is_a_plain_dequantization(tmp_path):
    """Dropping the adapters must leave the base itself on the NVFP4 grid, not change it."""

    wrapped = _wrapped_experts(hybrid_shared=True)
    with torch.no_grad():
        wrapped.gate_up_lora_B.zero_()
        wrapped.down_lora_B.zero_()
    merged, _ = _merged_tensors(tmp_path, wrapped, alpha=8.0)

    for expert in range(wrapped.num_experts):
        torch.testing.assert_close(
            merged["model.layers.0.mlp.experts.gate_up_proj"][expert],
            fake_quantize_nvfp4(wrapped.base_gate_up(expert)).to(torch.bfloat16),
            rtol=0, atol=0,
        )


def test_released_layout_refuses_into_the_fused_tensor(tmp_path):
    """Per-expert output must be exactly what the loader would fuse back together."""

    from safetensors import safe_open

    from scripts.fuse_moe_checkpoint import fuse_checkpoint

    wrapped = _wrapped_experts(hybrid_shared=True)
    fused, _ = _merged_tensors(tmp_path / "fused", wrapped, alpha=8.0)

    base = tmp_path / "rel_base"
    released = tmp_path / "rel_out"
    _write_base(base, wrapped)
    _write_adapters(tmp_path / "rel_adapters.safetensors", wrapped)
    merge_checkpoint(
        base, tmp_path / "rel_adapters.safetensors", released,
        lora_alpha=8.0, device="cpu", chunk=3, released_layout=True,
    )

    shard = next(released.glob("model-*.safetensors"))
    with safe_open(shard, framework="pt") as handle:
        keys = list(handle.keys())
    assert all(".block_sparse_moe.experts." in key for key in keys)
    assert any(key.endswith(".w1.weight") for key in keys)

    refused = tmp_path / "refused"
    fuse_checkpoint(released, refused)
    with safe_open(next(refused.glob("model-*.safetensors")), framework="pt") as handle:
        rebuilt = {key: handle.get_tensor(key) for key in handle.keys()}
    for key, value in fused.items():
        torch.testing.assert_close(rebuilt[key], value, rtol=0, atol=0)


def test_rtn_baseline_matches_merging_an_untrained_adapter(tmp_path):
    """The no-adapter mode is the round-to-nearest control the trained run is read against."""

    from safetensors import safe_open

    wrapped = _wrapped_experts(hybrid_shared=True)
    with torch.no_grad():
        wrapped.gate_up_lora_B.zero_()
        wrapped.down_lora_B.zero_()
    trained_zero, _ = _merged_tensors(tmp_path / "zeroed", wrapped, alpha=8.0)

    base = tmp_path / "rtn_base"
    output = tmp_path / "rtn_out"
    _write_base(base, wrapped)
    merge_checkpoint(base, None, output, lora_alpha=0.0, device="cpu", chunk=3)
    shard = next(output.glob("model-*.safetensors"))
    with safe_open(shard, framework="pt") as handle:
        baseline = {key: handle.get_tensor(key) for key in handle.keys()}

    for key, value in trained_zero.items():
        torch.testing.assert_close(baseline[key], value, rtol=0, atol=0)


def test_released_layout_follows_the_model_family(tmp_path):
    """A Qwen3-MoE base is written back as `mlp.experts.{e}.gate_proj/up_proj/down_proj`."""

    from safetensors import safe_open

    from scripts.fuse_moe_checkpoint import fuse_checkpoint

    wrapped = _wrapped_experts(hybrid_shared=True)
    fused, _ = _merged_tensors(tmp_path / "fused", wrapped, alpha=8.0)

    base = tmp_path / "qwen_base"
    _write_base(base, wrapped)
    (base / "config.json").write_text(
        json.dumps({"model_type": "qwen3_moe", "quantization_config": {"quant_method": "fp8"}}) + "\n"
    )
    _write_adapters(tmp_path / "qwen_adapters.safetensors", wrapped)
    released = tmp_path / "qwen_out"
    merge_checkpoint(
        base, tmp_path / "qwen_adapters.safetensors", released,
        lora_alpha=8.0, device="cpu", chunk=3, released_layout=True,
    )

    with safe_open(next(released.glob("model-*.safetensors")), framework="pt") as handle:
        keys = list(handle.keys())
    assert all(".mlp.experts." in key for key in keys)
    leaves = {key.rsplit(".", 2)[-2] for key in keys}
    assert leaves == {"gate_proj", "up_proj", "down_proj"}

    refused = tmp_path / "qwen_refused"
    fuse_checkpoint(released, refused)
    with safe_open(next(refused.glob("model-*.safetensors")), framework="pt") as handle:
        rebuilt = {key: handle.get_tensor(key) for key in handle.keys()}
    for key, value in fused.items():
        torch.testing.assert_close(rebuilt[key], value, rtol=0, atol=0)


def test_adapters_without_a_fused_base_are_refused(tmp_path):
    """A released per-expert base would be copied through unmerged; refuse instead."""

    wrapped = _wrapped_experts(hybrid_shared=True)
    base = tmp_path / "released_base"
    base.mkdir()
    save_file(
        {"model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.zeros(4, 4, dtype=torch.bfloat16)},
        base / "model.safetensors",
        metadata={"format": "pt"},
    )
    (base / "config.json").write_text(json.dumps({"model_type": "minimax"}) + "\n")
    _write_adapters(tmp_path / "adapters.safetensors", wrapped)
    with pytest.raises(KeyError, match="fuse_moe_checkpoint"):
        merge_checkpoint(base, tmp_path / "adapters.safetensors", tmp_path / "out", lora_alpha=8.0, device="cpu")
