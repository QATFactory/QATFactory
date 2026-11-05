"""Build a tiny block-FP8 MiniMax-M2 checkpoint that mirrors the released layout.

Used to exercise the FP8 LoRA-QAD path (loading, FSDP2 sharding of FP8 parameters, the
fused-expert wrapper) without downloading a 230B checkpoint.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from transformers.models.minimax_m2 import MiniMaxM2Config, MiniMaxM2ForCausalLM


BLOCK = 128


def quantize_blocks(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    if rows % BLOCK or cols % BLOCK:
        raise ValueError(f"weight {tuple(weight.shape)} is not a multiple of the {BLOCK} block")
    blocked = weight.float().reshape(rows // BLOCK, BLOCK, cols // BLOCK, BLOCK)
    amax = blocked.abs().amax(dim=(-3, -1), keepdim=True).clamp_min(1e-12)
    scale = amax / torch.finfo(torch.float8_e4m3fn).max
    codes = (blocked / scale).to(torch.float8_e4m3fn).reshape(rows, cols)
    return codes, scale.reshape(rows // BLOCK, cols // BLOCK).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--examples", type=int, default=256)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config = MiniMaxM2Config(
        vocab_size=len(tokenizer),
        hidden_size=256,
        intermediate_size=128,
        num_hidden_layers=args.layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_local_experts=args.experts,
        num_experts_per_tok=2,
        rotary_dim=32,
        tie_word_embeddings=False,
        max_position_embeddings=2048,
    )
    model = MiniMaxM2ForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)

    # Re-encode every convertible linear as block-scaled FP8, exactly as the released
    # checkpoint stores it (codes plus a per-block `weight_scale_inv` grid).
    skip = ("gate.weight", "e_score_correction_bias", "lm_head.weight")
    for shard in sorted(output.glob("*.safetensors")):
        tensors = load_file(shard)
        converted = {}
        for key, value in tensors.items():
            quantizable = (
                key.endswith(".weight")
                and value.ndim == 2
                and not key.endswith(skip)
                and "norm" not in key
                and "embed_tokens" not in key
            )
            if quantizable and value.shape[0] % BLOCK == 0 and value.shape[1] % BLOCK == 0:
                codes, scale_inv = quantize_blocks(value)
                converted[key] = codes
                converted[key[: -len(".weight")] + ".weight_scale_inv"] = scale_inv
            else:
                converted[key] = value
        save_file(converted, shard, metadata={"format": "pt"})

    config_path = output / "config.json"
    payload = json.loads(config_path.read_text())
    payload["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "float8_e4m3fn",
        "quant_method": "fp8",
        "weight_block_size": [BLOCK, BLOCK],
        "modules_to_not_convert": ["gate", "e_score_correction_bias", "lm_head"],
    }
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    random.seed(0)
    words = "quantization adapter expert router distillation gradient tensor kernel".split()
    for name, count in (("train.jsonl", args.examples), ("eval.jsonl", 16)):
        with (output / name).open("w", encoding="utf-8") as handle:
            for _ in range(count):
                blob = lambda n: " ".join(random.choice(words) for _ in range(n))  # noqa: E731
                handle.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": blob(60)},
                                {"role": "assistant", "reasoning_content": blob(40), "content": blob(80)},
                            ]
                        }
                    )
                    + "\n"
                )
    print(f"wrote tiny FP8 MiniMax checkpoint to {output}")


if __name__ == "__main__":
    main()
