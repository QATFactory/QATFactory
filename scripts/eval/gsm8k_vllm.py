#!/usr/bin/env python3
"""GSM8K accuracy of a served checkpoint (vLLM offline, greedy) — a task-level sanity check
that the held-out KL gains are not bought with accuracy.

Build the fixed question subset once (needs ``datasets``)::

    python scripts/eval/gsm8k_vllm.py --build-questions eval/gsm8k_test200.jsonl --n 200

Then score each deployment (thinking on, with a large budget, or off)::

    python scripts/eval/gsm8k_vllm.py --model exports/<run>-vllm-nvfp4 --tag qad \\
        --questions eval/gsm8k_test200.jsonl --out eval/gsm8k/qad-think.json --max-tokens 16384 --max-model-len 20480
    python scripts/eval/gsm8k_vllm.py --model exports/<run>-vllm-nvfp4 --tag qad --no-think \\
        --questions eval/gsm8k_test200.jsonl --out eval/gsm8k/qad-nothink.json --max-tokens 2048

The answer is the last number in the text after ``</think>`` (inside ``\\boxed{}`` when present).
At n=200 the standard error is ~2 points: differences smaller than that are noise. Greedy
decoding with thinking on tends to loop on some prompts; ``truncated`` counts those.
"""

import argparse
import json
import os
import random
import re
import time

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_number(text: str):
    body = text.split("</think>")[-1] if "</think>" in text else text
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", body)
    candidates = NUMBER.findall(boxed[-1]) if boxed else NUMBER.findall(body)
    if not candidates:
        return None
    try:
        return float(candidates[-1].replace(",", ""))
    except ValueError:
        return None


def build_questions(path: str, n: int, seed: int) -> None:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    indices = sorted(indices[:n])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for index in indices:
            row = dataset[index]
            file.write(json.dumps({"idx": index, "question": row["question"], "answer": row["answer"]}) + "\n")
    print(f"wrote {len(indices)} of {len(dataset)} test questions -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-questions", metavar="PATH", help="Write the fixed question subset and exit.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--tag")
    parser.add_argument("--questions", default="eval/gsm8k_test200.jsonl")
    parser.add_argument("--out")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--no-think", action="store_true", help="chat_template_kwargs enable_thinking=False")
    args = parser.parse_args()

    if args.build_questions:
        build_questions(args.build_questions, args.n, args.seed)
        return
    if not (args.model and args.tag and args.out):
        parser.error("--model, --tag and --out are required for scoring")

    rows = [json.loads(line) for line in open(args.questions, encoding="utf-8")][: args.n]
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        tensor_parallel_size=args.tp,
        enable_prefix_caching=False,
        trust_remote_code=True,
        seed=0,
    )
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0)
    suffix = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    prompts = [[{"role": "user", "content": row["question"] + suffix}] for row in rows]
    kwargs = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_think else {}
    started = time.time()
    outputs = llm.chat(prompts, params, use_tqdm=True, **kwargs)
    elapsed = time.time() - started

    records, correct, truncated = [], 0, 0
    for row, output in zip(rows, outputs):
        text = output.outputs[0].text
        gold = float(row["answer"].split("####")[-1].strip().replace(",", ""))
        pred = final_number(text)
        ok = pred is not None and abs(pred - gold) < 1e-6
        correct += ok
        truncated += output.outputs[0].finish_reason == "length"
        records.append(
            {
                "idx": row.get("idx"),
                "gold": gold,
                "pred": pred,
                "correct": ok,
                "finish_reason": output.outputs[0].finish_reason,
                "completion_tokens": len(output.outputs[0].token_ids),
                "text": text,
            }
        )
    result = {
        "tag": args.tag,
        "model": args.model,
        "n": len(rows),
        "accuracy": correct / len(rows),
        "correct": correct,
        "truncated": truncated,
        "thinking": not args.no_think,
        "max_tokens": args.max_tokens,
        "seconds": round(elapsed, 1),
        "mean_completion_tokens": sum(r["completion_tokens"] for r in records) / len(records),
        "records": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=1)
    print(
        f"[GSM8K] {args.tag}: acc={result['accuracy']:.3f} ({correct}/{len(rows)}) truncated={truncated} "
        f"mean_tokens={result['mean_completion_tokens']:.0f} time={elapsed:.0f}s -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
