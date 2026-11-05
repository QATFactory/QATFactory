#!/usr/bin/env python3
"""Materialize the trainer's held-out split, token for token, for deployment-side scoring.

``train_llm_qat.py`` with ``data.train_eval_path`` tokenizes one JSONL through the model's QAD
chat template and holds out ``eval_size`` conversations chosen by
``numpy.random.default_rng(split_seed).permutation(n)[:eval_size]``. This script rebuilds that
exact set (same tokenized cache, same permutation) and writes it as a list of
``(input_ids, labels)`` pairs so it can be scored in a different environment — the vLLM venv
usually pins a different ``transformers`` than the training venv, and a different tokenizer
fingerprint would mean a different cache and a different split.

Run this in the TRAINING environment::

    python scripts/eval/dump_heldout.py configs/llm/qwen3_5_9b_nvfp4_qad.yaml \\
        --out eval/heldout-qwen35.pt
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from transformers import AutoConfig  # noqa: E402

from qatfactory.data import build_tokenized_dataset, load_tokenizer  # noqa: E402
from qatfactory.utils import apply_dotted_overrides, load_yaml_config, resolve_model_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Training YAML whose data section defines the corpus, template and split.")
    parser.add_argument("--out", required=True, help="Output .pt: list of (input_ids, labels) for the held-out set.")
    known, overrides = parser.parse_known_args()
    cfg = apply_dotted_overrides(load_yaml_config(known.config), overrides)
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    if not data_cfg.get("train_eval_path"):
        parser.error("config must use data.train_eval_path (the trainer's held-out split is derived from it).")

    # Configs name the QAD chat template relative to the repo root (the trainer runs there).
    template = data_cfg.get("chat_template_file")
    if template and not os.path.isfile(template) and os.path.isfile(os.path.join(REPO_ROOT, template)):
        template = os.path.join(REPO_ROOT, template)

    model_dir = resolve_model_dir(model_cfg["model_path"])
    tokenizer = load_tokenizer(model_dir)
    model_type = AutoConfig.from_pretrained(model_dir).model_type
    full = build_tokenized_dataset(
        data_cfg["train_eval_path"],
        tokenizer,
        int(data_cfg.get("max_seq_len", 8192)),
        model_type=model_type,
        cache_dir=data_cfg.get("cache_dir", "cache/tokenized"),
        num_proc=data_cfg.get("num_proc"),
        chat_template_file=template,
        truncate=bool(data_cfg.get("truncate", True)),
    )
    eval_size = int(data_cfg.get("eval_size", 500))
    permutation = np.random.default_rng(int(data_cfg.get("split_seed", 42))).permutation(len(full))
    held = full.select(permutation[:eval_size], keep_in_memory=True)
    seqs = [(list(map(int, row["input_ids"])), list(map(int, row["labels"]))) for row in held]
    os.makedirs(os.path.dirname(os.path.abspath(known.out)), exist_ok=True)
    torch.save(seqs, known.out)
    assistant = sum(sum(1 for label in labels if label != -100) for _, labels in seqs)
    print(f"wrote {len(seqs)} held-out conversations ({assistant} assistant tokens) -> {known.out}")


if __name__ == "__main__":
    main()
