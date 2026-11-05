#!/usr/bin/env python3
"""Tokenize the paper's distribution-fidelity corpora for deployment-side scoring.

The paper reports KL(BF16 || quantized) on three public text sets that are *not* the
distillation data: UltraChat-200k (``test_sft``, rendered through the model's own chat
template), the Pile (``NeelNanda/pile-10k``) and WikiText-2 (raw, test split, concatenated and
cut into non-overlapping windows). This script materializes those sequences with a given
model's tokenizer in the same ``(input_ids, labels)`` format as ``dump_heldout.py`` — every
position is scored (``labels == input_ids``) — so ``deploy_kl_vllm.py`` and
``q4k_kld_llamacpp.sh`` can consume them unchanged.

Sampling matches ``scripts/eval/quality/measure_kld.py``: ``--num-samples``
documents per set (WikiText-2 test yields fewer 8K windows than that), a fixed shuffle seed,
documents shorter than ``--min-tokens`` skipped, each truncated to ``--max-seq-len`` tokens.
Sequences are tokenizer-specific, so dump once per model family::

    python scripts/eval/dump_text_corpora.py Qwen/Qwen3.5-9B --out-dir eval/corpora --name qwen35
    # -> eval/corpora/qwen35-ultrachat.pt  qwen35-pile.pt  qwen35-wikitext.pt
"""

import argparse
import os
import random
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from qatfactory.data import load_tokenizer  # noqa: E402
from qatfactory.utils import resolve_model_dir  # noqa: E402


def _ids(encoded):
    # transformers returns a BatchEncoding (a UserDict, not a dict) from tokenize=True on some versions.
    return list(encoded) if isinstance(encoded, list) else list(encoded["input_ids"])


def _sample_docs(texts, tokenizer, num_samples, max_seq_len, min_tokens, seed):
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    seqs = []
    for index in order:
        text = texts[index]
        if not text or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < min_tokens:
            continue
        seqs.append(ids[:max_seq_len])
        if len(seqs) >= num_samples:
            break
    return seqs


def load_ultrachat(tokenizer, num_samples, max_seq_len, min_tokens, seed):
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    seqs = []
    for index in order:
        ids = _ids(tokenizer.apply_chat_template(dataset[index]["messages"], tokenize=True, add_generation_prompt=False))
        if len(ids) < min_tokens:
            continue
        seqs.append(ids[:max_seq_len])
        if len(seqs) >= num_samples:
            break
    return seqs


def load_pile(tokenizer, num_samples, max_seq_len, min_tokens, seed):
    from datasets import load_dataset

    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    return _sample_docs(dataset["text"], tokenizer, num_samples, max_seq_len, min_tokens, seed)


def load_wikitext(tokenizer, num_samples, max_seq_len, min_tokens, seed):
    """WikiText rows are single lines: concatenate the split and cut fixed-size windows."""
    from datasets import load_dataset

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer("".join(dataset["text"]), add_special_tokens=False)["input_ids"]
    seqs = []
    for start in range(0, len(ids), max_seq_len):
        chunk = ids[start : start + max_seq_len]
        if len(chunk) < min_tokens:
            break
        seqs.append(chunk)
        if len(seqs) >= num_samples:
            break
    return seqs


LOADERS = {"ultrachat": load_ultrachat, "pile": load_pile, "wikitext": load_wikitext}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Checkpoint dir or Hub id whose tokenizer (and chat template) to use.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", required=True, help="File prefix, e.g. the model family (qwen35, dsr1).")
    parser.add_argument("--datasets", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = load_tokenizer(resolve_model_dir(args.model))
    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.datasets:
        seqs = LOADERS[name](tokenizer, args.num_samples, args.max_seq_len, args.min_tokens, args.seed)
        pairs = [(list(map(int, ids)), list(map(int, ids))) for ids in seqs]
        out = os.path.join(args.out_dir, f"{args.name}-{name}.pt")
        torch.save(pairs, out)
        lengths = [len(ids) for ids in seqs]
        print(f"{name}: {len(seqs)} sequences, {sum(lengths)} tokens (min {min(lengths)}, max {max(lengths)}) -> {out}")


if __name__ == "__main__":
    main()
