"""Rewrite a per-expert MoE checkpoint into the fused 3D layout the model loads into.

Released MoE checkpoints store every routed expert separately — Mixtral-style
``...experts.{e}.w1/w2/w3.weight`` (MiniMax-M2 among them) or Qwen-style
``...experts.{e}.gate_proj/up_proj/down_proj.weight`` (Qwen3-MoE) — while the ``transformers``
module keeps two fused 3D parameters. The conversion normally happens at load time, and because concatenation
must allocate, *every* rank builds its own full-size copy in anonymous memory — 8 ranks of a
230B FP8 backbone need ~1.7TiB of host RAM and do not fit on one node.

Doing the fusion once, offline, turns loading back into a plain memory-mapped assignment: the
file pages are then shared by all ranks through the page cache, so a whole node costs roughly
one model copy instead of one per rank. That is what lets the run go back to eight GPUs on a
single host, where FSDP all-gathers stay on NVLink instead of crossing the network.

The rewrite is exact. It reproduces the loader's own ops (``MergeModulelist(dim=0)`` then
``Concatenate(dim=1)``) on both the FP8 codes and their per-block scale grids, and copies
every other tensor through untouched.

    python scripts/fuse_moe_checkpoint.py <source_dir> <output_dir>
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qatfactory.quant.export.moe_layout import EXPERT_KEY, detect_layout  # noqa: E402

ASSET_SUFFIXES = (".json", ".jinja", ".txt", ".model", ".py", ".md")
# Per-expert leaf suffix -> the fused parameter's trailing name. A block-FP8 checkpoint
# carries the codes and their scale grid under two different suffixes, and both fuse the
# same way; anything else would silently produce a key the model does not have.
FUSED_SUFFIX = {"weight": "", "weight_scale_inv": "_scale_inv"}


def _shard_map(source: Path) -> dict[str, str]:
    index = source / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text())["weight_map"]
    shard_map = {}
    for shard in sorted(source.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            shard_map.update(dict.fromkeys(handle.keys(), shard.name))
    if not shard_map:
        raise FileNotFoundError(f"No safetensors found under {source}.")
    return shard_map


class _ShardReader:
    """Reads tensors by key, keeping one shard file open at a time."""

    def __init__(self, source: Path, shard_map: dict[str, str]):
        self.source = source
        self.shard_map = shard_map
        self._name = None
        self._handle = None

    def get(self, key: str) -> torch.Tensor:
        name = self.shard_map[key]
        if name != self._name:
            if self._handle is not None:
                self._handle.__exit__(None, None, None)
            self._handle = safe_open(self.source / name, framework="pt")
            self._handle.__enter__()
            self._name = name
        return self._handle.get_tensor(key)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.__exit__(None, None, None)
            self._handle = None


class _ShardWriter:
    """Accumulates tensors and flushes them into ~``shard_size`` byte safetensors files."""

    def __init__(self, output: Path, shard_size: int):
        self.output = output
        self.shard_size = shard_size
        self.buffer: dict[str, torch.Tensor] = {}
        self.buffered_bytes = 0
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0
        self._shard_count = 0

    def add(self, key: str, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        self.buffer[key] = tensor
        self.buffered_bytes += tensor.numel() * tensor.element_size()
        if self.buffered_bytes >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        self._shard_count += 1
        name = f"model-{self._shard_count:05d}.safetensors"
        save_file(self.buffer, self.output / name, metadata={"format": "pt"})
        for key in self.buffer:
            self.weight_map[key] = name
        self.total_bytes += self.buffered_bytes
        self.buffer.clear()
        self.buffered_bytes = 0

    def finalize(self) -> None:
        self._flush()
        count = self._shard_count
        # safetensors' convention encodes the shard count in every filename, so the files are
        # renamed once the total is known.
        renamed = {}
        for index in range(1, count + 1):
            old = f"model-{index:05d}.safetensors"
            new = f"model-{index:05d}-of-{count:05d}.safetensors"
            (self.output / old).rename(self.output / new)
            renamed[old] = new
        weight_map = {key: renamed[name] for key, name in self.weight_map.items()}
        index_path = self.output / "model.safetensors.index.json"
        index_path.write_text(
            json.dumps({"metadata": {"total_size": self.total_bytes}, "weight_map": weight_map}, indent=2) + "\n"
        )


def fuse_checkpoint(source: Path, output: Path, *, shard_size: int = 5_000_000_000) -> dict[str, int]:
    """Write ``source`` to ``output`` with every routed-expert group fused into 3D tensors."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty {output}.")
    output.mkdir(parents=True, exist_ok=True)

    shard_map = _shard_map(source)
    # Which family spells the per-expert tensors (w1/w3/w2 vs gate_proj/up_proj/down_proj).
    layout = detect_layout(shard_map)
    # group[(prefix, suffix)][projection][expert] = key
    groups: dict[tuple[str, str], dict[str, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
    passthrough = []
    for key in shard_map:
        match = EXPERT_KEY.match(key)
        if match:
            groups[(match["prefix"], match["suffix"])][match["projection"]][int(match["expert"])] = key
        else:
            passthrough.append(key)

    reader = _ShardReader(source, shard_map)
    writer = _ShardWriter(output, shard_size)
    stats = {"fused_groups": 0, "copied": 0}
    try:
        for key in sorted(passthrough):
            writer.add(key, reader.get(key))
            stats["copied"] += 1

        for (prefix, suffix), projections in sorted(groups.items()):
            if suffix not in FUSED_SUFFIX:
                raise ValueError(f"Unsupported per-expert tensor suffix {suffix!r} under {prefix}.")
            target_prefix = layout.to_fused_prefix(prefix)
            for name, sources in (("gate_up_proj", layout.gate_up), ("down_proj", (layout.down,))):
                halves = []
                for projection in sources:
                    per_expert = projections.get(projection)
                    if not per_expert:
                        raise ValueError(f"{prefix} is missing its {projection} tensors for suffix {suffix!r}.")
                    expected = list(range(len(per_expert)))
                    if sorted(per_expert) != expected:
                        raise ValueError(f"{prefix}.{projection} expert indices are not contiguous.")
                    # MergeModulelist(dim=0): stack the per-expert 2D tensors on a new axis.
                    halves.append(torch.stack([reader.get(per_expert[e]) for e in expected], dim=0))
                # Concatenate(dim=1): gate (w1) rows above up (w3) rows within each expert.
                fused = halves[0] if len(halves) == 1 else torch.cat(halves, dim=1)
                writer.add(f"{target_prefix}.{name}{FUSED_SUFFIX[suffix]}", fused)
                stats["fused_groups"] += 1
        writer.finalize()
    finally:
        reader.close()

    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix in ASSET_SUFFIXES and path.name != "model.safetensors.index.json":
            shutil.copy2(path, output / path.name)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--shard-size", type=int, default=5_000_000_000)
    args = parser.parse_args()

    stats = fuse_checkpoint(Path(args.source), Path(args.output), shard_size=args.shard_size)
    print(f"fused {stats['fused_groups']} expert tensors, copied {stats['copied']} others -> {args.output}")


if __name__ == "__main__":
    main()
