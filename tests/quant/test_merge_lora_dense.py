"""The dense merge must reproduce the student weight the training forward simulated.

Same contract as the expert merge: folding the adapter and landing on the NVFP4 grid has to
equal `NVFP4Linear`'s own effective weight, bit for bit, or what gets served is not what was
trained. The baseline path is pinned too, since a divergence measure is only readable against
the same grid with no trained correction.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qatfactory.quant.formats.nvfp4 import NVFP4Linear, fake_quantize_nvfp4
from scripts.merge_lora_bf16 import merge_checkpoint

TARGET = "model.language_model.layers.0.self_attn.q_proj"
VISION = "model.vision_tower.layers.0.attn.q_proj"


def _linear_with_adapter(rank=4, alpha=8.0, in_features=32, out_features=16):
    generator = torch.Generator().manual_seed(0)
    # BF16 throughout: a checkpoint is loaded in the compute dtype, and quantizing an fp32
    # weight lands on different grid points than quantizing its bf16 rounding.
    base = torch.nn.Linear(in_features, out_features, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        base.weight.copy_(torch.randn(base.weight.shape, generator=generator).to(torch.bfloat16))
    converted = NVFP4Linear.from_linear(base)
    converted.init_lora(rank, alpha)
    with torch.no_grad():
        converted.lora_A.copy_(
            (torch.randn(converted.lora_A.shape, generator=generator) * 0.05).to(converted.lora_A.dtype)
        )
        converted.lora_B.copy_(
            (torch.randn(converted.lora_B.shape, generator=generator) * 0.05).to(converted.lora_B.dtype)
        )
    return converted


def _write_inputs(tmp_path: Path, module, *, with_adapters: bool):
    base_dir = tmp_path / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            f"{TARGET}.weight": module.weight.detach().clone(),
            f"{VISION}.weight": module.weight.detach().clone(),
            "model.language_model.norm.weight": torch.ones(4),
        },
        base_dir / "model.safetensors",
        metadata={"format": "pt"},
    )
    (base_dir / "config.json").write_text(json.dumps({"model_type": "muse_glimmer"}) + "\n")

    adapters = None
    if with_adapters:
        adapters = tmp_path / "adapters.safetensors"
        save_file(
            {f"{TARGET}.lora_A": module.lora_A.detach(), f"{TARGET}.lora_B": module.lora_B.detach()},
            adapters,
            metadata={"format": "pt"},
        )
    return base_dir, adapters


def _read(output: Path):
    with safe_open(next(output.glob("model-*.safetensors")), framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def test_dense_merge_equals_the_training_student_weight(tmp_path):
    module = _linear_with_adapter(alpha=8.0)
    base_dir, adapters = _write_inputs(tmp_path, module, with_adapters=True)

    merge_checkpoint(base_dir, adapters, tmp_path / "out", lora_alpha=8.0, device="cpu")
    merged = _read(tmp_path / "out")

    expected = fake_quantize_nvfp4(module._effective_weight()).to(torch.bfloat16)
    torch.testing.assert_close(merged[f"{TARGET}.weight"], expected, rtol=0, atol=0)


def test_a_module_without_an_adapter_is_left_alone(tmp_path):
    """The vision tower shares leaf names with the text tower and must pass through."""

    module = _linear_with_adapter()
    base_dir, adapters = _write_inputs(tmp_path, module, with_adapters=True)

    merge_checkpoint(base_dir, adapters, tmp_path / "out", lora_alpha=8.0, device="cpu")
    merged = _read(tmp_path / "out")

    torch.testing.assert_close(
        merged[f"{VISION}.weight"], module.weight.detach().to(torch.bfloat16), rtol=0, atol=0
    )


def test_the_dense_baseline_quantizes_only_the_named_leaves(tmp_path):
    module = _linear_with_adapter()
    base_dir, _ = _write_inputs(tmp_path, module, with_adapters=False)

    merge_checkpoint(
        base_dir, None, tmp_path / "out", lora_alpha=0.0, device="cpu",
        dense_target_leaves=frozenset({"q_proj"}),
        exclude_prefixes=("model.vision_tower",),
    )
    merged = _read(tmp_path / "out")

    torch.testing.assert_close(
        merged[f"{TARGET}.weight"],
        fake_quantize_nvfp4(module.weight.detach()).to(torch.bfloat16),
        rtol=0, atol=0,
    )
    # Excluded by prefix even though its leaf name matches.
    torch.testing.assert_close(
        merged[f"{VISION}.weight"], module.weight.detach().to(torch.bfloat16), rtol=0, atol=0
    )
