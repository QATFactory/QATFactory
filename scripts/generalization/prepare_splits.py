#!/usr/bin/env python3
"""Build balanced 45K math/code splits from explicit multi-source strata (``--scale`` for a pilot)."""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


MATH_META = "meta-math/MetaMathQA"
MATH_ORCA = "HuggingFaceH4/orca-math-word-problems-200k"
CODE_EVOL = "theblackcat102/evol-codealpaca-v1"
ULTRAINTERACT = "openbmb/UltraInteract_sft"

TRAIN_COUNTS = {
    "math/meta": 11_250,
    "math/orca": 11_250,
    "math/ui-cot": 11_250,
    "math/ui-pot": 11_250,
    "code/evol": 22_500,
    "code/ui": 22_500,
}
EVAL_COUNTS = {
    "math/meta": 128,
    "math/orca": 128,
    "math/ui-cot": 128,
    "math/ui-pot": 128,
    "code/evol": 256,
    "code/ui": 256,
}
MONITOR_COUNTS = {
    "math/meta": 8,
    "math/orca": 8,
    "math/ui-cot": 8,
    "math/ui-pot": 8,
    "code/evol": 16,
    "code/ui": 16,
}


@dataclass(frozen=True, slots=True)
class Candidate:
    path_index: int
    offset: int
    length: int
    sample_id: str
    source: str
    stratum: str
    total_tokens: int
    score: str


def record_id(record: dict) -> str:
    return f"{record['source']}:{record['orig_row']}"


def score(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode()).hexdigest()


def first_user(record: dict) -> str:
    return next(
        (message.get("content", "") for message in record.get("messages", []) if message.get("role") == "user"),
        "",
    )


def load_ultrainteract_tasks(path: Path) -> dict[str, str]:
    table = pq.read_table(path, columns=["task", "instruction"])
    result = {}
    for task, instruction in zip(table["task"].to_pylist(), table["instruction"].to_pylist()):
        previous = result.setdefault(instruction, task)
        if previous != task:
            raise ValueError(f"UltraInteract instruction maps to multiple tasks: {instruction[:80]!r}.")
    return result


def classify(record: dict, ui_tasks: dict[str, str]) -> str | None:
    source = record.get("source")
    if source == MATH_META:
        return "math/meta"
    if source == MATH_ORCA:
        return "math/orca"
    if source == CODE_EVOL:
        return "code/evol"
    if source != ULTRAINTERACT:
        return None
    task = ui_tasks.get(first_user(record))
    if task == "Coding":
        return "code/ui"
    if task == "Math_CoT":
        return "math/ui-cot"
    if task == "Math_PoT":
        return "math/ui-pot"
    return None


def scan(paths: list[Path], ui_tasks: dict[str, str], seed: int, min_total_tokens: int):
    candidates = defaultdict(list)
    input_metadata = []
    seen = set()
    for path_index, path in enumerate(paths):
        digest = hashlib.sha256()
        rows = 0
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                digest.update(line)
                rows += 1
                record = json.loads(line)
                if record.get("finish_reason") != "stop":
                    continue
                total_tokens = (record.get("usage") or {}).get("total_tokens")
                if not isinstance(total_tokens, int) or total_tokens <= min_total_tokens:
                    continue
                stratum = classify(record, ui_tasks)
                if stratum is None:
                    continue
                sample_id = record_id(record)
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample id: {sample_id}.")
                seen.add(sample_id)
                candidates[stratum].append(
                    Candidate(
                        path_index=path_index,
                        offset=offset,
                        length=len(line),
                        sample_id=sample_id,
                        source=record["source"],
                        stratum=stratum,
                        total_tokens=total_tokens,
                        score=score(sample_id, seed),
                    )
                )
        input_metadata.append(
            {
                "path": str(path),
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    for rows in candidates.values():
        rows.sort(key=lambda item: (item.score, item.sample_id))
    return candidates, input_metadata


def write_split(path: Path, rows: list[Candidate], input_paths: list[Path]) -> dict:
    temporary = path.with_suffix(".jsonl.tmp")
    handles = [source.open("rb") for source in input_paths]
    digest = hashlib.sha256()
    sources = Counter()
    strata = Counter()
    try:
        with temporary.open("wb") as output:
            for candidate in sorted(rows, key=lambda item: (item.score, item.sample_id)):
                handle = handles[candidate.path_index]
                handle.seek(candidate.offset)
                line = handle.read(candidate.length)
                output.write(line if line.endswith(b"\n") else line + b"\n")
                digest.update(candidate.sample_id.encode())
                digest.update(b"\n")
                sources[candidate.source] += 1
                strata[candidate.stratum] += 1
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        for handle in handles:
            handle.close()
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "id_sha256": digest.hexdigest(),
        "source_counts": dict(sorted(sources.items())),
        "stratum_counts": dict(sorted(strata.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--ultrainteract-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-total-tokens", type=int, default=4096)
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Multiply every stratum count (train, eval, monitor) by this factor, for a smaller "
             "pilot of the study; 1.0 is the paper's 45K-per-domain split.",
    )
    args = parser.parse_args()
    if args.scale <= 0:
        raise ValueError(f"--scale must be positive, got {args.scale}.")

    def scaled(counts: dict[str, int]) -> dict[str, int]:
        return {stratum: max(1, round(count * args.scale)) for stratum, count in counts.items()}

    train_counts, eval_counts, monitor_counts = (
        scaled(TRAIN_COUNTS), scaled(EVAL_COUNTS), scaled(MONITOR_COUNTS))

    input_paths = [Path(path) for path in args.input]
    ui_path = Path(args.ultrainteract_parquet)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}.")

    candidates, inputs = scan(
        input_paths,
        load_ultrainteract_tasks(ui_path),
        args.seed,
        args.min_total_tokens,
    )
    selected_train = {"math": [], "code": []}
    selected_eval = {"math": [], "code": []}
    selected_monitor = {"math": [], "code": []}
    available_counts = {}
    reserve_counts = {}
    for stratum, train_count in train_counts.items():
        eval_count = eval_counts[stratum]
        rows = candidates[stratum]
        required = train_count + eval_count
        if len(rows) < required:
            raise ValueError(f"{stratum} has {len(rows)} eligible rows; {required} required.")
        eval_rows = rows[:eval_count]
        train_rows = rows[eval_count:required]
        domain = stratum.partition("/")[0]
        selected_eval[domain].extend(eval_rows)
        selected_train[domain].extend(train_rows)
        selected_monitor[domain].extend(eval_rows[: monitor_counts[stratum]])
        available_counts[stratum] = len(rows)
        reserve_counts[stratum] = len(rows) - required

    split_metadata = {}
    for domain in ("math", "code"):
        for split_name, rows in (
            ("train", selected_train[domain]),
            ("eval", selected_eval[domain]),
            ("eval_monitor", selected_monitor[domain]),
        ):
            name = f"{domain}_{split_name}"
            split_metadata[name] = write_split(output_dir / f"{name}.jsonl", rows, input_paths)

    ui_sha = hashlib.sha256(ui_path.read_bytes()).hexdigest()
    manifest = {
        "format_version": 1,
        "seed": args.seed,
        "scale": args.scale,
        "eligibility": {
            "finish_reason": "stop",
            "min_total_tokens_exclusive": args.min_total_tokens,
            "ultrainteract_tasks": {
                "code": ["Coding"],
                "math": ["Math_CoT", "Math_PoT"],
                "ignored": ["Logic"],
            },
        },
        "inputs": inputs,
        "ultrainteract_parquet": {
            "path": str(ui_path),
            "bytes": ui_path.stat().st_size,
            "sha256": ui_sha,
        },
        "available_counts": dict(sorted(available_counts.items())),
        "reserve_counts_before_4k_repair": dict(sorted(reserve_counts.items())),
        "splits": split_metadata,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
