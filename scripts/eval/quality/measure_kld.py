#!/usr/bin/env python3
"""Approximate forward KL divergence KL(P_teacher || P_student) between a
reference model (e.g. the BF16 baseline) and one or more quantized models,
evaluated teacher-forced on real text, by querying ALREADY-RUNNING vLLM or
SGLang servers (one endpoint per model; engines can be mixed, e.g. a vLLM BF16
teacher against SGLang students).

This is the server/API-based counterpart to loading each model offline: because
it hits live endpoints, all models are scored concurrently across their GPUs and
nothing is (re)loaded here.

Method
------
1. For each dataset, build N token sequences with the SHARED tokenizer (all
   models must use the same tokenizer so token ids align), truncated to
   --max-seq-len tokens.
2. Send the SAME token-id prompt to every endpoint via POST /v1/completions with
   {max_tokens: 1, temperature: 0, prompt_logprobs: K}. vLLM returns, for every
   prompt position, the top-K predicted next-token log-probabilities
   (choices[0].prompt_logprobs). This is a top-K approximation of the true
   distribution (serving stacks cap logprobs), applied identically to every
   model so the relative comparison is fair.
   NOTE: K must be <= each server's --max-logprobs (vLLM default 20); a larger K
   is rejected with HTTP 400. Launch servers with --max-logprobs >= K for k>20.
   SGLang endpoints (``label=url=served=sglang``) are queried through SGLang's
   native POST /generate with {input_ids, max_new_tokens: 1, temperature: 0,
   return_logprob, logprob_start_len: 0, top_logprobs_num: K}; the per-position
   ``meta_info.input_top_logprobs`` it returns are the same quantity as vLLM's
   prompt_logprobs (position 0 has none), so both engines are scored identically.
3. Per position, compute an approximate forward KL restricted to the teacher's
   returned top-K support: for each teacher token use the student's returned
   log-prob if present, else fall back to the student's smallest returned top-K
   log-prob at that position (a conservative floor for tokens outside its top-K).
   Token-averaged over all scored positions. Teacher probabilities are NOT
   renormalized over the truncated support -- identical convention to
   qatfactory.distill.kd_loss's ``fwd_kl`` metric.

Memory is bounded: samples are processed in batches; each batch fetches all
(endpoint x sample) distributions in parallel, accumulates scalar KL sums, then
discards the distributions before the next batch.
"""
import argparse
import hashlib
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


# --------------------------------------------------------------------------- #
# Dataset loaders: each returns a list of token-id sequences (<= max_seq_len).
# --------------------------------------------------------------------------- #
def _sample_docs(texts, tokenizer, num_samples, max_seq_len, min_tokens, seed, add_special_tokens=False):
    rng = random.Random(seed)
    n = len(texts)
    order = list(range(n))
    rng.shuffle(order)
    seqs = []
    for i in order:
        text = texts[i]
        if not text or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
        if len(ids) < min_tokens:
            continue
        seqs.append(ids[:max_seq_len])
        if len(seqs) >= num_samples:
            break
    return seqs


def load_ultrachat(tokenizer, num_samples, max_seq_len, min_tokens, seed):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    seqs = []
    for i in order:
        messages = ds[i]["messages"]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        # Some transformers versions return a BatchEncoding (a UserDict, NOT a
        # dict subclass) when tokenize=True; unwrap to the raw id list.
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        if len(ids) < min_tokens:
            continue
        seqs.append(ids[:max_seq_len])
        if len(seqs) >= num_samples:
            break
    return seqs


def load_pile(tokenizer, num_samples, max_seq_len, min_tokens, seed):
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    return _sample_docs(ds["text"], tokenizer, num_samples, max_seq_len, min_tokens, seed)


def load_wikitext(tokenizer, num_samples, max_seq_len, min_tokens, seed):
    """Wikitext rows are short lines, so concatenate the corpus and chunk it into
    non-overlapping max_seq_len-token windows; take the first num_samples chunks."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    full = "".join(ds["text"])
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    seqs = []
    for start in range(0, len(ids), max_seq_len):
        chunk = ids[start : start + max_seq_len]
        if len(chunk) < min_tokens:
            break
        seqs.append(chunk)
        if len(seqs) >= num_samples:
            break
    return seqs


DATASET_LOADERS = {
    "ultrachat": load_ultrachat,
    "pile": load_pile,
    "wikitext": load_wikitext,
}


# --------------------------------------------------------------------------- #
# Endpoint querying
# --------------------------------------------------------------------------- #
API_KINDS = ("vllm", "sglang")


def _request(api, base_url, served_name, token_ids, top_k):
    """(url, json payload) for one teacher-forced prompt-logprobs request."""
    if api == "sglang":
        root = base_url.rstrip("/")
        if root.endswith("/v1"):  # accept the OpenAI-style base URL too
            root = root[: -len("/v1")]
        return root + "/generate", {
            "input_ids": token_ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": top_k,
        }
    return base_url.rstrip("/") + "/completions", {
        "model": served_name,
        "prompt": token_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": top_k,
        "logprobs": 0,
    }


def _parse_response(api, data):
    """Per-position list of {int token_id: logprob}; positions without a distribution -> None."""
    if api == "sglang":
        # meta_info.input_top_logprobs: one entry per input token, each a list of
        # [logprob, token_id, token_text] (token_text is None unless text was requested).
        return [{int(entry[1]): float(entry[0]) for entry in pos} if pos else None
                for pos in data["meta_info"]["input_top_logprobs"]]
    out = []
    for entry in data["choices"][0].get("prompt_logprobs") or []:
        if entry is None:
            out.append(None)
        else:
            out.append({int(tid): info["logprob"] for tid, info in entry.items()})
    return out


def fetch_prompt_logprobs(base_url, served_name, token_ids, top_k, timeout, retries=3, api="vllm"):
    """Return a per-position list of {int token_id: logprob} (position 0 -> None)."""
    url, payload = _request(api, base_url, served_name, token_ids, top_k)
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            return _parse_response(api, resp.json())
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch from {served_name} @ {base_url}: {last_err}")


def kl_accumulate(teacher_seq, student_seq):
    """Forward KL(teacher||student) summed over positions (teacher top-K support,
    student-floor fallback), plus the number of scored positions."""
    kl_sum = 0.0
    tokens = 0
    for t_pos, s_pos in zip(teacher_seq, student_seq):
        if not t_pos or not s_pos:
            continue
        s_floor = min(s_pos.values())
        pos_kl = 0.0
        for tid, t_logp in t_pos.items():
            t_p = math.exp(t_logp)
            s_logp = s_pos.get(tid, s_floor)
            pos_kl += t_p * (t_logp - s_logp)
        kl_sum += pos_kl
        tokens += 1
    return kl_sum, tokens


def _batches(n, size):
    for start in range(0, n, size):
        yield list(range(start, min(start + size, n)))


def parse_endpoint(spec):
    """'label=base_url=served_name[=api]' -> (label, base_url, served_name, api); api is vllm (default) or sglang."""
    parts = spec.split("=")
    if len(parts) == 3:
        parts.append("vllm")
    if len(parts) != 4 or parts[3] not in API_KINDS:
        raise argparse.ArgumentTypeError(
            f"endpoint must be label=base_url=served_name[=vllm|sglang], got: {spec}")
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True, type=parse_endpoint,
                    help="reference endpoint: label=base_url=served_name[=vllm|sglang]")
    ap.add_argument("--student", required=True, action="append", type=parse_endpoint,
                    help="student endpoint label=base_url=served_name[=vllm|sglang] (repeatable)")
    ap.add_argument("--tokenizer", required=True, help="shared tokenizer path/repo (must match all models)")
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_LOADERS),
                    choices=list(DATASET_LOADERS))
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--top-k", type=int, default=20,
                    help="prompt_logprobs K; must be <= every server's --max-logprobs (vLLM default 20)")
    ap.add_argument("--min-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="samples fetched in parallel per batch (each endpoint gets this many "
                         "concurrent long-sequence prompt_logprobs requests; keep it modest -- "
                         "a single max-seq-len request materializes a big logits tensor)")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--cache-dir", default="outputs/kld/cache")
    ap.add_argument("--output", default="outputs/kld/kld_results.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # Cached sequences are token ids, so they are only valid for the tokenizer that built them.
    tok_tag = hashlib.sha256(json.dumps(sorted(tok.get_vocab().items())).encode()).hexdigest()[:12]
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    endpoints = [args.teacher] + args.student  # (label, base_url, served, api)
    teacher_label = args.teacher[0]
    student_labels = [s[0] for s in args.student]

    # results[student_label][dataset] = {"kl_sum":..., "tokens":...}
    results = {sl: {ds: {"kl_sum": 0.0, "tokens": 0} for ds in args.datasets} for sl in student_labels}

    for ds_name in args.datasets:
        seq_cache = Path(args.cache_dir) / f"seqs_{ds_name}_n{args.num_samples}_L{args.max_seq_len}_s{args.seed}_tok{tok_tag}.json"
        if seq_cache.exists():
            sequences = json.loads(seq_cache.read_text())
            print(f"[kld] {ds_name}: loaded {len(sequences)} cached sequences", flush=True)
        else:
            print(f"[kld] {ds_name}: building up to {args.num_samples} sequences (max_seq_len={args.max_seq_len})...",
                  flush=True)
            sequences = DATASET_LOADERS[ds_name](tok, args.num_samples, args.max_seq_len, args.min_tokens, args.seed)
            seq_cache.write_text(json.dumps(sequences))
        if not sequences:
            print(f"[kld] {ds_name}: NO sequences found, skipping", flush=True)
            continue
        lens = [len(s) for s in sequences]
        print(f"[kld] {ds_name}: {len(sequences)} sequences (tokens min={min(lens)} max={max(lens)} "
              f"mean={sum(lens)//len(lens)})", flush=True)

        for batch in _batches(len(sequences), args.batch_size):
            # Fetch every (endpoint, sample) distribution for this batch in parallel.
            dists = {}  # (label, i) -> per-position list
            with ThreadPoolExecutor(max_workers=len(endpoints) * len(batch)) as ex:
                futs = {}
                for lbl, url, served, api in endpoints:
                    for i in batch:
                        f = ex.submit(fetch_prompt_logprobs, url, served, sequences[i], args.top_k, args.timeout,
                                      api=api)
                        futs[f] = (lbl, i)
                for f in as_completed(futs):
                    lbl, i = futs[f]
                    dists[(lbl, i)] = f.result()

            for i in batch:
                t_seq = dists[(teacher_label, i)]
                for sl in student_labels:
                    kl_sum, ntok = kl_accumulate(t_seq, dists[(sl, i)])
                    results[sl][ds_name]["kl_sum"] += kl_sum
                    results[sl][ds_name]["tokens"] += ntok
            done = min(batch[-1] + 1, len(sequences))
            print(f"[kld]   {ds_name}: {done}/{len(sequences)} samples scored", flush=True)

    # Aggregate: per-dataset KLD + across-dataset mean per student.
    summary = {}
    for sl in student_labels:
        per_ds = {}
        nats_list = []
        for ds_name in args.datasets:
            r = results[sl][ds_name]
            nats = r["kl_sum"] / r["tokens"] if r["tokens"] else float("nan")
            per_ds[ds_name] = {"kld_nats": nats, "kld_bits": nats / math.log(2), "num_tokens": r["tokens"]}
            if r["tokens"]:
                nats_list.append(nats)
        mean_nats = sum(nats_list) / len(nats_list) if nats_list else float("nan")
        summary[sl] = {
            "per_dataset": per_ds,
            "mean_nats": mean_nats,
            "mean_bits": mean_nats / math.log(2),
        }

    out = {
        "config": {
            "teacher": teacher_label,
            "engines": {lbl: api for lbl, _url, _served, api in endpoints},
            "tokenizer": args.tokenizer,
            "datasets": args.datasets,
            "num_samples": args.num_samples,
            "max_seq_len": args.max_seq_len,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "results": summary,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[kld] wrote {args.output}")

    # Pretty table.
    print(f"\n=== approximate forward KL(teacher={teacher_label} || student), top-{args.top_k}, nats ===")
    header = f"{'student':22s} " + " ".join(f"{d:>10s}" for d in args.datasets) + f" {'mean':>10s}"
    print(header)
    print("-" * len(header))
    for sl in student_labels:
        row = f"{sl:22s} "
        for d in args.datasets:
            v = summary[sl]["per_dataset"][d]["kld_nats"]
            row += f" {v:10.4f}"
        row += f" {summary[sl]['mean_nats']:10.4f}"
        print(row)


if __name__ == "__main__":
    main()
