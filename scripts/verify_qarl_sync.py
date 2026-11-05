#!/usr/bin/env python3
"""End-to-end proof that the QARL weight sync reproduces the served NVFP4 checkpoint.

Given the BF16 QAD latents an export was built from, re-packing those latents and
publishing them into the running engine must leave every quantized parameter *bit
identical* to what vLLM loaded from disk. That makes the first sync a verifiable no-op and
exercises the whole path — FSDP-free latent read, fused-family amax tying, packing,
vLLM's own post-load relayout, and the in-place copy — against the real kernel rather than
a stand-in.

Run on one GPU, e.g. against the rollout checkpoint examples/run_qarl_qwen3_8b.sh builds::

    CUDA_VISIBLE_DEVICES=0 python scripts/verify_qarl_sync.py \\
        --latents  exports/RTN-Qwen3-8B-Base-nvfp4/latents \\
        --rollout  exports/RTN-Qwen3-8B-Base-nvfp4/vllm-nvfp4
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qatfactory.verify_qarl")

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", required=True, help="BF16 QAD checkpoint the export was built from.")
    parser.add_argument("--rollout", required=True, help="nvfp4-pack-quantized export directory.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention backend for the student load; sdpa works without the flash extra.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")

    from vllm import LLM, SamplingParams

    from qatfactory.quant.formats.nvfp4 import refresh_nvfp4_weight_amax
    from qatfactory.rl.model import load_qarl_student
    from qatfactory.rl.vllm_nvfp4_sync import NVFP4WeightPublisher, vllm_model_from_engine

    logger.info("Loading the fake-quantized student from %s.", args.latents)
    student, converted = load_qarl_student(
        args.latents, args.rollout, TARGET_MODULES, attn_implementation=args.attn_implementation
    )
    student = student.to("cuda").eval()
    # Outside the Trainer there is no FusedScaleRefreshCallback, so populate the cached
    # per-family maxima the publisher reads exactly as on_train_begin would.
    refresh_nvfp4_weight_amax(student)
    logger.info("Student ready: %d quantized linears.", len(converted))

    llm = LLM(
        model=args.rollout,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        distributed_executor_backend="external_launcher",
    )
    vllm_model = vllm_model_from_engine(llm)

    sampling = SamplingParams(max_tokens=24, temperature=0.0)
    before_text = llm.generate([args.prompt], sampling)[0].outputs[0].text
    baseline = {name: tensor.detach().clone() for name, tensor in vllm_model.state_dict().items()}

    publisher = NVFP4WeightPublisher(student, vllm_model)
    published = publisher.sync()
    logger.info("Published %d quantized layers.", published)

    after = vllm_model.state_dict()
    quantized_names = {
        f"{target.vllm_name}.{leaf}"
        for target in publisher.targets
        for leaf in ("weight", "weight_scale", "weight_global_scale", "alpha")
    }
    differing = [
        name
        for name, tensor in baseline.items()
        if name in after and not torch.equal(tensor, after[name])
    ]

    logger.info("Compared %d engine tensors (%d of them quantized).", len(baseline), len(quantized_names))
    if differing:
        logger.error("%d tensors changed on the first sync; expected none:", len(differing))
        for name in differing[:10]:
            delta = (baseline[name].float() - after[name].float()).abs().max().item()
            logger.error("  %s  max|delta|=%g", name, delta)
        return 1

    after_text = llm.generate([args.prompt], sampling)[0].outputs[0].text
    if before_text != after_text:
        logger.error("Greedy generation changed across a no-op sync:\n  before: %r\n  after:  %r", before_text, after_text)
        return 1

    logger.info("PASS: re-packing the latents reproduced the served checkpoint exactly.")
    logger.info("Greedy sample unchanged: %r", after_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
