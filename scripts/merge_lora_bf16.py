"""Fold trained LoRA adapters into a frozen base and write a plain BF16 checkpoint.

A weight-only NVFP4 run never changes the base weights, so what it produces is a set of
adapters plus the promise that inference will quantize ``base + delta``. Writing that promise
out as BF16 numbers makes the trained model servable by any stack that reads a normal
Hugging Face checkpoint: the expert weights carry values that are already on the NVFP4 grid,
so a BF16 forward reproduces W4A16 serving numerics without needing a packed-NVFP4 kernel or
an exporter that understands fused 3D experts.

The block-FP8 base is dequantized on the way through, so the result is uniformly BF16 and
carries no ``quantization_config`` -- vLLM then serves it as an ordinary BF16 model.

    python scripts/merge_lora_bf16.py <fused_base_dir> <adapters.safetensors> <output_dir>
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qatfactory.quant.export.moe_layout import ExpertLayout, layout_for_model_type
from qatfactory.quant.formats.fp8_base import dequantize_fp8_blocks, is_fp8_tensor
from qatfactory.quant.formats.nvfp4 import fake_quantize_nvfp4
from scripts.fuse_moe_checkpoint import _ShardReader, _ShardWriter, _shard_map

ASSET_SUFFIXES = (".json", ".jinja", ".txt", ".model", ".py", ".md")
EXPERT_LEAVES = ("gate_up_proj", "down_proj")
SCALE_SUFFIX = "_scale_inv"


def _released_layout_for(base: Path) -> ExpertLayout:
    """Which per-expert spelling the serving stack expects for this model family."""

    model_type = json.loads((base / "config.json").read_text()).get("model_type", "")
    return layout_for_model_type(model_type)


def _split_released(tensor: torch.Tensor, leaf: str) -> tuple[torch.Tensor, ...]:
    if leaf != "gate_up_proj":
        return (tensor,)
    # The fused tensor stacks w1's rows above w3's, which is how the loader concatenates them.
    half = tensor.shape[0] // 2
    return (tensor[:half], tensor[half:])


def _adapter_layout(adapters: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {leaf: adapters[f"{prefix}.{leaf}"] for leaf in
            ("gate_up_lora_A", "gate_up_lora_B", "down_lora_A", "down_lora_B")}


def _is_dense_target(prefix: str, leaves: frozenset[str], exclude_prefixes: tuple[str, ...]) -> bool:
    """Which dense linears the baseline quantizes, mirroring the trainer's own selection."""

    if prefix.rpartition(".")[2] not in leaves:
        return False
    return not any(prefix == p or prefix.startswith(p + ".") for p in exclude_prefixes)


def _merge_dense_weight(
    base: torch.Tensor,
    scale_inv: torch.Tensor | None,
    lora_a: torch.Tensor | None,
    lora_b: torch.Tensor | None,
    scaling: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    """The dense counterpart of the expert merge: fold the adapter, land on the NVFP4 grid."""

    weight = base.to(device)
    if scale_inv is not None:
        weight = dequantize_fp8_blocks(weight, scale_inv.to(device), torch.bfloat16)
    else:
        weight = weight.to(torch.bfloat16)
    if lora_a is not None and lora_b is not None:
        # Same dtype flow as `LoRAMixin._effective_weight`, so the rounding onto the grid
        # matches the student forward the run trained.
        delta = torch.matmul(lora_b.to(device), lora_a.to(device)) * scaling
        weight = weight + delta.reshape(weight.shape).to(weight.dtype)
    return fake_quantize_nvfp4(weight).to(torch.bfloat16).cpu()


def _merge_expert_tensor(
    base: torch.Tensor,
    scale_inv: torch.Tensor | None,
    lora: tuple[torch.Tensor, torch.Tensor, str] | None,
    scaling: float,
    *,
    hybrid_shared: bool,
    device: torch.device,
    chunk: int,
) -> torch.Tensor:
    """Dequantize, add the LoRA delta if there is one, and put each expert on the NVFP4 grid."""

    experts = base.shape[0]
    out = torch.empty(base.shape, dtype=torch.bfloat16, device="cpu")
    for start in range(0, experts, chunk):
        stop = min(start + chunk, experts)
        block = base[start:stop].to(device)
        if scale_inv is not None:
            block = dequantize_fp8_blocks(block, scale_inv[start:stop].to(device), torch.bfloat16)
        else:
            block = block.to(torch.bfloat16)

        if lora is not None:
            lora_a, lora_b, shared_factor = lora
            # `A` is shared across experts for the gate/up factor and `B` for the down factor
            # in the hybrid layout; the other side always carries a leading expert axis.
            if shared_factor == "A":
                a = lora_a.to(device) if hybrid_shared else lora_a[start:stop].to(device)
                b = lora_b[start:stop].to(device)
            else:
                a = lora_a[start:stop].to(device)
                b = lora_b.to(device) if hybrid_shared else lora_b[start:stop].to(device)

            # The dtype flow has to be the training one: `NVFP4ExpertLoRA` forms the delta and
            # adds it in the parameter dtype, and doing the same arithmetic in fp32 would round
            # differently onto the NVFP4 grid for values sitting near a step boundary.
            delta = torch.matmul(b, a) * scaling
            block = block + delta.reshape(block.shape).to(block.dtype)

        # Each expert is quantized on its own, matching the per-expert tensor scale the
        # training forward used; a scale shared across the 3D stack would be a different model.
        for index in range(stop - start):
            out[start + index] = fake_quantize_nvfp4(block[index]).to(torch.bfloat16).cpu()
        del block
    return out


def merge_checkpoint(
    base: Path,
    adapter_path: Path | None,
    output: Path,
    *,
    lora_alpha: float,
    device: str = "cuda:0",
    chunk: int = 16,
    shard_size: int = 5_000_000_000,
    expert_targets: set[str] | None = None,
    released_layout: bool = False,
    dense_target_leaves: frozenset[str] = frozenset(),
    exclude_prefixes: tuple[str, ...] = (),
) -> dict[str, int]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty {output}.")
    output.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)

    adapters: dict[str, torch.Tensor] = {}
    if adapter_path is not None:
        with safe_open(adapter_path, framework="pt") as handle:
            for key in handle.keys():
                adapters[key] = handle.get_tensor(key)
        if not adapters:
            raise ValueError(f"{adapter_path} holds no tensors.")

    if adapters:
        sample = next(
            (value for key, value in adapters.items() if key.endswith("gate_up_lora_A")),
            None,
        )
        if sample is None:
            # A dense run carries plain `<module>.lora_A` factors and no expert layout.
            sample = next(value for key, value in adapters.items() if key.endswith(".lora_A"))
            hybrid_shared = True
        else:
            hybrid_shared = sample.ndim == 3
        rank = int(sample.shape[-2])
    else:
        # Round-to-nearest baseline: the same quantization grid with no trained correction,
        # which is the only thing a LoRA-QAD number can be read against.
        hybrid_shared = True
        rank = 0
    scaling = lora_alpha / rank if rank else 0.0

    shard_map = _shard_map(base)
    layout = _released_layout_for(base) if released_layout else None
    # Every adapter must land on a base tensor. Fused-expert adapters need the *fused* base
    # (`<experts>.gate_up_proj` / `.down_proj`, see fuse_moe_checkpoint.py); a released
    # per-expert base has no such key and would otherwise be copied through unmerged.
    expected_targets = {f"{key[: -len('.gate_up_lora_A')]}.{leaf}" for key in adapters if key.endswith(".gate_up_lora_A") for leaf in EXPERT_LEAVES}
    expected_targets |= {f"{key[: -len('.lora_A')]}.weight" for key in adapters if key.endswith(".lora_A")}
    missing = sorted(expected_targets - set(shard_map))
    if missing:
        raise KeyError(
            f"{len(missing)} adapter target(s) have no tensor in {base} (first: {missing[0]!r}). "
            "For routed experts the base must be in the fused layout: run "
            "scripts/fuse_moe_checkpoint.py on the released checkpoint first."
        )
    reader = _ShardReader(base, shard_map)
    writer = _ShardWriter(output, shard_size)
    stats = {"merged": 0, "dequantized": 0, "copied": 0}
    scale_keys = {key for key in shard_map if key.endswith(SCALE_SUFFIX)}
    try:
        for key in sorted(shard_map):
            if key in scale_keys:
                continue
            prefix, _, leaf = key.rpartition(".")
            scale_key = f"{key}{SCALE_SUFFIX}"
            scale_inv = reader.get(scale_key) if scale_key in scale_keys else None

            is_expert = leaf in EXPERT_LEAVES and (
                f"{prefix}.gate_up_lora_A" in adapters
                or (not adapters and (expert_targets is None or prefix in expert_targets))
            )
            if is_expert:
                lora = None
                if adapters:
                    factors = _adapter_layout(adapters, prefix)
                    lora = (
                        (factors["gate_up_lora_A"], factors["gate_up_lora_B"], "A")
                        if leaf == "gate_up_proj"
                        else (factors["down_lora_A"], factors["down_lora_B"], "B")
                    )
                merged = _merge_expert_tensor(
                    reader.get(key), scale_inv, lora, scaling,
                    hybrid_shared=hybrid_shared, device=torch_device, chunk=chunk,
                )
                if released_layout:
                    for expert in range(merged.shape[0]):
                        names = layout.released_keys(prefix, leaf, expert)
                        for name, part in zip(names, _split_released(merged[expert], leaf)):
                            writer.add(name, part)
                else:
                    writer.add(key, merged)
                stats["merged"] += 1
                continue

            if leaf == "weight" and (
                f"{prefix}.lora_A" in adapters
                or (not adapters and _is_dense_target(prefix, dense_target_leaves, exclude_prefixes))
            ):
                writer.add(
                    key,
                    _merge_dense_weight(
                        reader.get(key), scale_inv,
                        adapters.get(f"{prefix}.lora_A"), adapters.get(f"{prefix}.lora_B"),
                        scaling, device=torch_device,
                    ),
                )
                stats["merged"] += 1
                continue

            tensor = reader.get(key)
            if scale_inv is not None and is_fp8_tensor(tensor):
                tensor = dequantize_fp8_blocks(
                    tensor.to(torch_device), scale_inv.to(torch_device), torch.bfloat16
                ).cpu()
                stats["dequantized"] += 1
            else:
                if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
                    tensor = tensor.to(torch.bfloat16)
                stats["copied"] += 1
            writer.add(key, tensor)
        writer.finalize()
    finally:
        reader.close()

    for path in sorted(base.iterdir()):
        if path.is_file() and path.suffix in ASSET_SUFFIXES and path.name != "model.safetensors.index.json":
            shutil.copy2(path, output / path.name)

    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    # The output holds no FP8 codes, so leaving the block-FP8 declaration behind would make
    # every loader look for scales that are not there.
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    config.pop("torch_dtype", None)
    config["qatfactory_merged_lora"] = {
        "adapters": adapter_path.name if adapter_path is not None else None,
        "lora_rank": int(rank),
        "lora_alpha": float(lora_alpha) if rank else None,
        "hybrid_shared": bool(hybrid_shared) if rank else None,
        "weight_grid": "nvfp4",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return stats


def _recorded_lora_alpha(adapters: Path) -> float:
    """The alpha a LoRA-QAD run recorded in the config saved next to its adapters (0.0 if none)."""

    config = adapters.parent / "config.json"
    if not config.is_file():
        return 0.0
    lora = json.loads(config.read_text(encoding="utf-8")).get("qatfactory_qad", {}).get("lora") or {}
    return float(lora.get("alpha") or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument(
        "adapters",
        help="Trained adapter safetensors, or 'none' for the round-to-nearest baseline.",
    )
    parser.add_argument("output")
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=0.0,
        help="LoRA alpha of the run. Defaults to the qatfactory_qad.lora.alpha recorded in the "
             "config.json next to the adapters.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=5_000_000_000)
    parser.add_argument(
        "--released-layout",
        action="store_true",
        help="Write per-expert tensors in the released spelling of the model family "
             "(w1/w3/w2 or gate_proj/up_proj/down_proj), which is what serving stacks load.",
    )
    parser.add_argument(
        "--dense-target-leaves",
        default="",
        help="Comma-separated leaf names to quantize when building a dense baseline "
             "(ignored when adapters are supplied, which already name their targets).",
    )
    parser.add_argument(
        "--exclude-prefixes",
        default="",
        help="Comma-separated module prefixes the baseline must leave alone (e.g. the vision tower).",
    )
    args = parser.parse_args()

    adapters = None if args.adapters.lower() == "none" else Path(args.adapters)
    if adapters is not None and not args.lora_alpha:
        args.lora_alpha = _recorded_lora_alpha(adapters)
        if not args.lora_alpha:
            parser.error(f"--lora-alpha is required: {adapters.parent / 'config.json'} records no qatfactory_qad.lora.alpha.")

    stats = merge_checkpoint(
        Path(args.base), adapters, Path(args.output),
        lora_alpha=args.lora_alpha, device=args.device,
        chunk=args.chunk, shard_size=args.shard_size,
        released_layout=args.released_layout,
        dense_target_leaves=frozenset(x for x in args.dense_target_leaves.split(",") if x),
        exclude_prefixes=tuple(x for x in args.exclude_prefixes.split(",") if x),
    )
    print(
        f"merged {stats['merged']} expert tensors, dequantized {stats['dequantized']}, "
        f"copied {stats['copied']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
