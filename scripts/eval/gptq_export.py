"""GPTQ post-training baseline (the PTQ rows next to RTN and QAD in the paper's tables).

llm-compressor's GPTQModifier quantizes every Linear except the output head, vision tower, MTP head
and embeddings to NVFP4 or MXFP4, calibrated on conversations from the QAD training corpus, and
saves a compressed-tensors checkpoint that vLLM serves like the QAD exports (the NVFP4 scheme also
calibrates the static input global scales, so the result is W4A4-servable).

Settings match the paper's GPTQ baselines: llm-compressor
0.12.0, block size 128, dampening 0.01, static activation order, 512 calibration conversations
drawn with ``random.Random(42).sample(range(n_rows), 512)`` (sorted), rendered with the model's
stock chat template and truncated to 8,192 tokens.

    pip install -e . 'llmcompressor==0.12.0'
    python scripts/eval/gptq_export.py --model Qwen/Qwen3.5-9B \
        --corpus data/openperfectblend_100k_Qwen3.5-9B_think_train.jsonl \
        --scheme NVFP4 --output exports/GPTQ-qwen35-nvfp4
"""

import argparse
import json
import random

import datasets
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from qatfactory.data.text import load_tokenizer

IGNORE = ["lm_head", "re:.*visual.*", "re:.*mtp.*", "re:.*embed_tokens.*"]


def calibration_texts(corpus: str, tokenizer, num_samples: int, seed: int) -> list[str]:
    with open(corpus, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    picked = sorted(random.Random(seed).sample(range(len(lines)), num_samples))
    return [tokenizer.apply_chat_template(json.loads(lines[i])["messages"], tokenize=False) for i in picked]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hub repo id or local BF16 model directory.")
    parser.add_argument("--corpus", required=True, help="QAD training corpus (conversation JSONL).")
    parser.add_argument("--scheme", required=True, choices=["NVFP4", "MXFP4"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # load_tokenizer guards against a mis-declared tokenizer class (DeepSeek-R1-Distill loaded from a
    # local directory), which would corrupt the calibration text and the saved tokenizer.
    tokenizer = load_tokenizer(args.model)
    texts = calibration_texts(args.corpus, tokenizer, args.num_samples, args.seed)
    dataset = datasets.Dataset.from_dict({"text": texts}).map(
        lambda row: tokenizer(row["text"], max_length=args.max_seq_len, truncation=True, add_special_tokens=False),
        remove_columns=["text"],
    )

    # Multimodal checkpoints (e.g. Qwen3.5) keep their vision tower, unquantized, so the output
    # serves under the same architecture as the base model and the QAD exports.
    config = AutoConfig.from_pretrained(args.model)
    model_class = AutoModelForImageTextToText if hasattr(config, "vision_config") else AutoModelForCausalLM
    model = model_class.from_pretrained(args.model, dtype="auto")

    recipe = GPTQModifier(
        targets="Linear", scheme=args.scheme, ignore=IGNORE, block_size=128, dampening_frac=0.01, actorder="static"
    )
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.num_samples,
    )
    model.save_pretrained(args.output, save_compressed=True)
    tokenizer.save_pretrained(args.output)
    if model_class is AutoModelForImageTextToText:
        AutoProcessor.from_pretrained(args.model).save_pretrained(args.output)
    print(f"GPTQ {args.scheme} checkpoint: {args.output}")


if __name__ == "__main__":
    main()
