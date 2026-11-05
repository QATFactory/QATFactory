#!/usr/bin/env python3
"""Deployment-side fidelity of a served checkpoint against the BF16 model.

Loads a checkpoint in vLLM (the packed NVFP4/MXFP4 export, an RTN export, or the BF16 model)
and scores fixed token sequences — the trainer's held-out conversations from
``scripts/eval/dump_heldout.py`` (assistant tokens only) or the paper's public text sets from
``scripts/eval/dump_text_corpora.py`` (every token) — reporting per scored token

- NLL / perplexity (exact, from the realised token's prompt logprob);
- ``kl_topk_mean``: KL(BF16 || model) over the union of both models' top-K tokens, both
  renormalised over that support;
- ``kl_ref_topk_mean``: KL(BF16 || model) over the BF16 model's top-K support with the BF16
  probabilities left unnormalised and tokens missing from the model's top-K floored at its
  smallest returned logprob — the convention of the trainer's ``eval_fwd_kl`` metric and of the
  paper's UltraChat / Pile / WikiText tables;
- top-1 agreement with the BF16 reference.

Run once for the BF16 model to write the reference, then once per quantized checkpoint::

    python scripts/eval/deploy_kl_vllm.py --model Qwen/Qwen3.5-9B --heldout eval/heldout-qwen35.pt \\
        --tag bf16 --ref-out eval/ref-qwen35.pt --out eval/results/bf16.json
    python scripts/eval/deploy_kl_vllm.py --model exports/<run>-vllm-nvfp4 --heldout eval/heldout-qwen35.pt \\
        --tag qad --ref eval/ref-qwen35.pt --out eval/results/qad.json

Several sets can be scored with one engine load by passing the same number of ``--heldout``,
``--out`` and ``--ref`` / ``--ref-out`` paths (matched position by position).

Run in the environment that has vLLM; the sequence files carry token ids, so no tokenizer or
``transformers`` version needs to match the training side. On Hopper vLLM serves FP4 weights
through the W4A16 Marlin path, so these numbers measure W4A16 deployment there.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch


def load_engine(model, sets, *, max_model_len, gpu_mem, topk, tp):
    from vllm import LLM

    longest = max(len(ids) for seqs in sets for ids, _ in seqs)
    # Scoring asks for one output token after the full prompt, so the engine must admit
    # prompt + 1; a held-out set truncated at the training max_seq_len sits exactly at the cap.
    return LLM(
        model=model,
        max_model_len=max(max_model_len, longest + 1),
        gpu_memory_utilization=gpu_mem,
        max_logprobs=topk,
        enable_prefix_caching=False,
        trust_remote_code=True,
        tensor_parallel_size=tp,
    )


def score(llm, seqs, *, topk):
    from vllm import SamplingParams

    scored = sum(sum(1 for label in labels if label != -100) for _, labels in seqs)
    print(f"[eval] {len(seqs)} sequences, {scored} scored tokens, longest {max(len(ids) for ids, _ in seqs)}", flush=True)
    params = SamplingParams(max_tokens=1, prompt_logprobs=topk, temperature=0)
    started = time.time()
    outputs = llm.generate([{"prompt_token_ids": ids} for ids, _ in seqs], params, use_tqdm=False)
    print(f"[eval] scored in {time.time() - started:.0f}s", flush=True)

    # Per sequence: list of (position, target token, {token: logprob}) over scored positions.
    records = []
    for (ids, labels), output in zip(seqs, outputs):
        prompt_logprobs = output.prompt_logprobs  # len == len(ids); [0] is None
        rows = []
        for i in range(1, len(ids)):
            if labels[i] == -100 or prompt_logprobs[i] is None:
                continue
            rows.append((i, ids[i], {int(t): float(v.logprob) for t, v in prompt_logprobs[i].items()}))
        records.append(rows)
    return records


def compare(reference, records):
    """KL(ref || model) in the two truncated conventions described in the module docstring, and top-1 agreement."""

    kls, kls_ref, agree, n = [], [], 0, 0
    for ref_rows, model_rows in zip(reference, records):
        model_by_pos = {i: d for i, _, d in model_rows}
        for i, _, ref_dist in ref_rows:
            model_dist = model_by_pos.get(i)
            if model_dist is None:
                continue
            union = list(set(ref_dist) | set(model_dist))
            ref_lp = np.array([ref_dist.get(t, -30.0) for t in union])
            model_lp = np.array([model_dist.get(t, -30.0) for t in union])
            ref_p = np.exp(ref_lp - np.logaddexp.reduce(ref_lp))
            model_lp = model_lp - np.logaddexp.reduce(model_lp)
            kls.append(float((ref_p * (np.log(ref_p + 1e-30) - model_lp)).sum()))

            floor = min(model_dist.values())
            kls_ref.append(sum(math.exp(lp) * (lp - model_dist.get(t, floor)) for t, lp in ref_dist.items()))

            agree += int(max(ref_dist, key=ref_dist.get) == max(model_dist, key=model_dist.get))
            n += 1
    return {
        "kl_topk_mean": float(np.mean(kls)),
        "kl_ref_topk_mean": float(np.mean(kls_ref)),
        "top1_agree_vs_bf16": agree / max(n, 1),
        "n_compared": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Checkpoint dir or Hub id to serve.")
    parser.add_argument("--heldout", required=True, nargs="+", help="Sequence file(s) from dump_heldout.py / dump_text_corpora.py.")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True, nargs="+", help="JSON result path per --heldout.")
    parser.add_argument("--ref", nargs="+", default=None, help="Reference logprobs per --heldout, written by a BF16 pass (--ref-out).")
    parser.add_argument("--ref-out", nargs="+", default=None, help="Write this run's logprobs as the reference, per --heldout.")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--tp", type=int, default=1, help="Tensor-parallel size.")
    args = parser.parse_args()
    for name in ("out", "ref", "ref_out"):
        paths = getattr(args, name)
        if paths is not None and len(paths) != len(args.heldout):
            parser.error(f"--{name.replace('_', '-')} needs one path per --heldout ({len(args.heldout)})")

    sets = [torch.load(path, weights_only=False) for path in args.heldout]
    llm = load_engine(args.model, sets, max_model_len=args.max_model_len, gpu_mem=args.gpu_mem, topk=args.topk, tp=args.tp)
    for index, (heldout, seqs, out) in enumerate(zip(args.heldout, sets, args.out)):
        records = score(llm, seqs, topk=args.topk)
        nll = np.array([-d[target] for rows in records for _, target, d in rows if target in d])
        result = {
            "tag": args.tag,
            "model": args.model,
            "heldout": heldout,
            "n_tokens": int(nll.size),
            "nll_mean": float(nll.mean()),
            "ppl": float(math.exp(nll.mean())),
            "topk": args.topk,
        }
        if args.ref_out:
            torch.save(records, args.ref_out[index])
            print(f"[eval] wrote reference logprobs -> {args.ref_out[index]}")
        if args.ref:
            result.update(compare(torch.load(args.ref[index], weights_only=False), records))
        print(json.dumps(result, indent=2))
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)


if __name__ == "__main__":  # vLLM spawns workers for some models; they must not re-run this file
    main()
