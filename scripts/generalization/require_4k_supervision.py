#!/usr/bin/env python3
"""Replace expanded code rows whose assistant supervision starts after 4K.

Run after prepare_splits.py on the same inputs. It audits every selected code row, writes the
audit to code_4k_zero_supervision.json, and swaps each row with no supervised token inside 4K
for the next 4K-valid reserve row of its stratum, in the split's seeded order.
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer

from qatfactory.data import resolve_chat_template


CODE_EVOL = "theblackcat102/evol-codealpaca-v1"
ULTRAINTERACT = "openbmb/UltraInteract_sft"


def record_id(record: dict) -> str:
    return f"{record['source']}:{record['orig_row']}"


def score(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode()).hexdigest()


def first_user(record: dict) -> str:
    return next(
        (message.get("content", "") for message in record.get("messages", []) if message.get("role") == "user"),
        "",
    )


def load_ui_tasks(path: Path) -> dict[str, str]:
    table = pq.read_table(path, columns=["task", "instruction"])
    return dict(zip(table["instruction"].to_pylist(), table["task"].to_pylist()))


def code_stratum(record: dict, ui_tasks: dict[str, str]) -> str | None:
    if record.get("source") == CODE_EVOL:
        return "code/evol"
    if record.get("source") == ULTRAINTERACT and ui_tasks.get(first_user(record)) == "Coding":
        return "code/ui"
    return None


def read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [(json.loads(line), line) for line in handle]


def has_4k_supervision(record, tokenizer, template):
    encoded = tokenizer.apply_chat_template(
        record["messages"],
        chat_template=template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    mask = encoded["assistant_masks"][:4096]
    return any(mask), len(encoded["input_ids"]), sum(mask)


def audit_4k(rows, tokenizer, template) -> dict:
    """Rows with no supervised token inside the first 4,096: they train nothing in a 4K arm."""

    invalid = [index for index, (record, _line, _stratum) in enumerate(rows)
               if not has_4k_supervision(record, tokenizer, template)[0]]
    return {"invalid_count": len(invalid), "invalid_indices": invalid}


def write_rows(path: Path, rows) -> dict:
    temporary = path.with_suffix(".jsonl.tmp")
    digest = hashlib.sha256()
    sources = Counter()
    strata = Counter()
    with temporary.open("w", encoding="utf-8") as output:
        for record, line, stratum in rows:
            output.write(line if line.endswith("\n") else line + "\n")
            digest.update(record_id(record).encode())
            digest.update(b"\n")
            sources[record["source"]] += 1
            strata[stratum] += 1
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
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
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ultrainteract-parquet", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    seed = int(manifest["seed"])
    minimum = int(manifest["eligibility"]["min_total_tokens_exclusive"])
    ui_tasks = load_ui_tasks(Path(args.ultrainteract_parquet))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    template = resolve_chat_template(AutoConfig.from_pretrained(args.model_path).model_type)

    split_rows = {}
    selected_ids = set()
    for split in ("code_train", "code_eval"):
        rows = []
        for record, line in read_rows(data_dir / f"{split}.jsonl"):
            stratum = code_stratum(record, ui_tasks)
            if stratum is None:
                raise ValueError(f"Selected non-code row: {record_id(record)}.")
            rows.append([record, line, stratum])
            selected_ids.add(record_id(record))
        split_rows[split] = rows

    audit = {split: audit_4k(rows, tokenizer, template) for split, rows in split_rows.items()}
    (data_dir / "code_4k_zero_supervision.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    reserves = defaultdict(list)
    for path in map(Path, args.input):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                total_tokens = (record.get("usage") or {}).get("total_tokens")
                if record.get("finish_reason") != "stop" or not isinstance(total_tokens, int) or total_tokens <= minimum:
                    continue
                stratum = code_stratum(record, ui_tasks)
                if stratum is None:
                    continue
                sample_id = record_id(record)
                if sample_id not in selected_ids:
                    reserves[stratum].append((score(sample_id, seed), sample_id, record, line))
    for rows in reserves.values():
        rows.sort(key=lambda item: (item[0], item[1]))

    reserve_positions = defaultdict(int)
    replacements = []
    for split in ("code_train", "code_eval"):
        for row_index in audit[split]["invalid_indices"]:
            current = split_rows[split][row_index]
            stratum = current[2]
            candidates = reserves[stratum]
            while reserve_positions[stratum] < len(candidates):
                _score, sample_id, record, line = candidates[reserve_positions[stratum]]
                reserve_positions[stratum] += 1
                valid, length, supervised = has_4k_supervision(record, tokenizer, template)
                if not valid:
                    continue
                replaced_id = record_id(current[0])
                split_rows[split][row_index] = [record, line, stratum]
                selected_ids.add(sample_id)
                replacements.append(
                    {
                        "split": split,
                        "stratum": stratum,
                        "removed_id": replaced_id,
                        "replacement_id": sample_id,
                        "replacement_length": length,
                        "replacement_supervised_4k": supervised,
                    }
                )
                break
            else:
                raise RuntimeError(f"Ran out of 4K-valid reserve rows for {stratum}.")

    for split, rows in split_rows.items():
        manifest["splits"][split] = write_rows(data_dir / f"{split}.jsonl", rows)

    # Rebuild the monitor subset from the repaired eval split with prepare_splits' per-stratum counts.
    monitor_counts = manifest["splits"]["code_eval_monitor"]["stratum_counts"]
    monitor = []
    per_stratum = Counter()
    for row in split_rows["code_eval"]:
        if per_stratum[row[2]] < monitor_counts.get(row[2], 0):
            monitor.append(row)
            per_stratum[row[2]] += 1
    if len(monitor) != sum(monitor_counts.values()):
        raise RuntimeError(f"Expected {sum(monitor_counts.values())} monitor rows, got {len(monitor)}.")
    manifest["splits"]["code_eval_monitor"] = write_rows(data_dir / "code_eval_monitor.jsonl", monitor)
    manifest["canonical_4k_validation"] = {
        "model_path": args.model_path,
        "invalid_before_repair": {
            split: int(audit[split]["invalid_count"]) for split in ("code_train", "code_eval")
        },
        "replacement_count": len(replacements),
        "replacements": replacements,
        "unused_4k_validity_reserve_not_exhaustively_counted": {
            stratum: len(reserves[stratum]) - reserve_positions[stratum] for stratum in reserves
        },
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    print(
        json.dumps(
            {
                "replacement_count": len(replacements),
                "by_stratum": dict(Counter(item["stratum"] for item in replacements)),
                "monitor_strata": dict(per_stratum),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
