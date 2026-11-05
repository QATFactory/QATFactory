"""Offline expert fusion must reproduce what the loader would have built in memory."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.fuse_moe_checkpoint import fuse_checkpoint


EXPERTS = 4
HIDDEN = 8
INTERMEDIATE = 6
BLOCK = 2


def _write_source(directory, *, fp8: bool = False):
    """A miniature released-style checkpoint with per-expert w1/w2/w3 tensors."""

    torch.manual_seed(0)
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    tensors = {
        "model.embed_tokens.weight": torch.randn(5, HIDDEN, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16),
        "model.layers.0.block_sparse_moe.gate.weight": torch.randn(EXPERTS, HIDDEN, dtype=torch.bfloat16),
        "model.layers.0.block_sparse_moe.e_score_correction_bias": torch.randn(EXPERTS, dtype=torch.bfloat16),
    }
    expected = {}
    for layer in (0, 1):
        for expert in range(EXPERTS):
            base = f"model.layers.{layer}.block_sparse_moe.experts.{expert}"
            for projection, shape in (
                ("w1", (INTERMEDIATE, HIDDEN)),
                ("w3", (INTERMEDIATE, HIDDEN)),
                ("w2", (HIDDEN, INTERMEDIATE)),
            ):
                value = torch.randn(*shape, dtype=torch.bfloat16).to(dtype)
                tensors[f"{base}.{projection}.weight"] = value
                expected[(layer, expert, projection, "weight")] = value
                if fp8:
                    scale = torch.rand(shape[0] // BLOCK, shape[1] // BLOCK, dtype=torch.float32) + 0.5
                    tensors[f"{base}.{projection}.weight_scale_inv"] = scale
                    expected[(layer, expert, projection, "weight_scale_inv")] = scale
    tensors["model.layers.1.self_attn.q_proj.weight"] = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16)

    save_file(tensors, directory / "model.safetensors", metadata={"format": "pt"})
    (directory / "config.json").write_text(json.dumps({"model_type": "minimax_m2"}))
    return tensors, expected


def _load_output(directory):
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    merged = {}
    for shard in sorted(set(index["weight_map"].values())):
        merged.update(load_file(directory / shard))
    return merged, index


@pytest.mark.parametrize("fp8", [False, True])
def test_fused_tensors_match_the_loader_operations(tmp_path, fp8):
    source, output = tmp_path / "src", tmp_path / "out"
    source.mkdir()
    _, expected = _write_source(source, fp8=fp8)

    fuse_checkpoint(source, output, shard_size=4096)
    merged, _ = _load_output(output)

    for layer in (0, 1):
        prefix = f"model.layers.{layer}.mlp.experts"
        for suffix, leaf in (("weight", ""), ("weight_scale_inv", "_scale_inv")):
            if suffix == "weight_scale_inv" and not fp8:
                continue
            # MergeModulelist(dim=0) stacks experts, Concatenate(dim=1) puts w1 above w3.
            gate_up = merged[f"{prefix}.gate_up_proj{leaf}"]
            down = merged[f"{prefix}.down_proj{leaf}"]
            rows = gate_up.shape[1] // 2
            for expert in range(EXPERTS):
                torch.testing.assert_close(
                    gate_up[expert][:rows].float(),
                    expected[(layer, expert, "w1", suffix)].float(),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    gate_up[expert][rows:].float(),
                    expected[(layer, expert, "w3", suffix)].float(),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    down[expert].float(), expected[(layer, expert, "w2", suffix)].float(), atol=0, rtol=0
                )


def test_fused_shapes_and_dtypes_match_the_module_allocation(tmp_path):
    source, output = tmp_path / "src", tmp_path / "out"
    source.mkdir()
    _write_source(source, fp8=True)

    fuse_checkpoint(source, output, shard_size=4096)
    merged, _ = _load_output(output)
    prefix = "model.layers.0.mlp.experts"

    assert merged[f"{prefix}.gate_up_proj"].shape == (EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    assert merged[f"{prefix}.down_proj"].shape == (EXPERTS, HIDDEN, INTERMEDIATE)
    assert merged[f"{prefix}.gate_up_proj_scale_inv"].shape == (
        EXPERTS,
        2 * INTERMEDIATE // BLOCK,
        HIDDEN // BLOCK,
    )
    assert merged[f"{prefix}.down_proj_scale_inv"].shape == (EXPERTS, HIDDEN // BLOCK, INTERMEDIATE // BLOCK)
    assert merged[f"{prefix}.gate_up_proj"].dtype == torch.float8_e4m3fn
    assert merged[f"{prefix}.gate_up_proj_scale_inv"].dtype == torch.float32


def test_non_expert_tensors_are_copied_verbatim(tmp_path):
    source, output = tmp_path / "src", tmp_path / "out"
    source.mkdir()
    tensors, _ = _write_source(source, fp8=True)

    fuse_checkpoint(source, output, shard_size=4096)
    merged, index = _load_output(output)

    for key in (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.e_score_correction_bias",
    ):
        torch.testing.assert_close(merged[key], tensors[key], atol=0, rtol=0)
    # No per-expert leaf survives, and the index covers exactly what was written.
    assert not any(".experts.0." in key for key in merged)
    assert set(index["weight_map"]) == set(merged)
    assert (output / "config.json").is_file()


def test_sharding_splits_and_names_files_consistently(tmp_path):
    source, output = tmp_path / "src", tmp_path / "out"
    source.mkdir()
    _write_source(source, fp8=True)

    fuse_checkpoint(source, output, shard_size=512)
    _, index = _load_output(output)

    shards = sorted(set(index["weight_map"].values()))
    assert len(shards) > 1
    for name in shards:
        assert name.endswith(f"-of-{len(shards):05d}.safetensors")
        assert (output / name).is_file()


def test_missing_projection_is_rejected(tmp_path):
    source, output = tmp_path / "src", tmp_path / "out"
    source.mkdir()
    save_file(
        {"model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.zeros(2, 2)},
        source / "model.safetensors",
        metadata={"format": "pt"},
    )
    with pytest.raises(ValueError, match="missing its w3"):
        fuse_checkpoint(source, output)


def test_qwen_style_experts_fuse_under_their_own_names(tmp_path):
    """Qwen3-MoE spells the experts `mlp.experts.{e}.gate_proj/up_proj/down_proj`."""

    source, output = tmp_path / "src", tmp_path / "out"
    source.mkdir()
    torch.manual_seed(0)
    tensors = {"model.layers.0.mlp.gate.weight": torch.randn(EXPERTS, HIDDEN, dtype=torch.bfloat16)}
    expected = {}
    for expert in range(EXPERTS):
        for projection, shape in (
            ("gate_proj", (INTERMEDIATE, HIDDEN)),
            ("up_proj", (INTERMEDIATE, HIDDEN)),
            ("down_proj", (HIDDEN, INTERMEDIATE)),
        ):
            value = torch.randn(*shape, dtype=torch.bfloat16)
            tensors[f"model.layers.0.mlp.experts.{expert}.{projection}.weight"] = value
            expected[(expert, projection)] = value
    save_file(tensors, source / "model.safetensors", metadata={"format": "pt"})
    (source / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))

    fuse_checkpoint(source, output)
    merged, _ = _load_output(output)

    assert set(merged) == {
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    }
    gate_up, down = merged["model.layers.0.mlp.experts.gate_up_proj"], merged["model.layers.0.mlp.experts.down_proj"]
    assert gate_up.shape == (EXPERTS, 2 * INTERMEDIATE, HIDDEN) and down.shape == (EXPERTS, HIDDEN, INTERMEDIATE)
    for expert in range(EXPERTS):
        # gate rows above up rows, exactly as the qwen2_moe WeightConverter concatenates them.
        torch.testing.assert_close(gate_up[expert][:INTERMEDIATE], expected[(expert, "gate_proj")], rtol=0, atol=0)
        torch.testing.assert_close(gate_up[expert][INTERMEDIATE:], expected[(expert, "up_proj")], rtol=0, atol=0)
        torch.testing.assert_close(down[expert], expected[(expert, "down_proj")], rtol=0, atol=0)


def test_mixed_expert_spellings_are_rejected(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    save_file(
        {
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(2, 2),
            "model.layers.0.mlp.experts.0.w3.weight": torch.zeros(2, 2),
        },
        source / "model.safetensors",
        metadata={"format": "pt"},
    )
    with pytest.raises(ValueError, match="exactly one known MoE layout"):
        fuse_checkpoint(source, tmp_path / "out")
