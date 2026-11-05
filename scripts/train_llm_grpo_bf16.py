#!/usr/bin/env python3
"""BF16 GRPO baseline for the QARL comparison (the paper's Table 8, BF16 RL rows).

Same config file, data, reward functions and GRPOConfig as scripts/train_llm_qarl.py; the only
difference is the policy: a plain BF16 model, rolled out by TRL's stock colocated vLLM engine with
its BF16 weight sync, instead of NVFP4 fake-quantized latents rolled out on the packed export. The
config's `model.rollout_path` and `qad` sections are ignored.

Launch with torchrun, e.g.::

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
        scripts/train_llm_grpo_bf16.py configs/llm/qwen3_8b_nvfp4_qarl.yaml \\
        --model.model_path Qwen/Qwen3-8B-Base --train.output_dir outputs/Qwen3-8B-Base-bf16-grpo-deepmath

For the paper's "BF16 RL + NVFP4 RTN" row, quantize the result with scripts/eval/rtn_export.sh.
"""

import argparse
import logging
import os
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qatfactory.data import load_tokenizer  # noqa: E402
from qatfactory.rl.data import build_datasets, resolve_reward_funcs  # noqa: E402
from qatfactory.trainer.fsdp import resolve_fsdp_options  # noqa: E402
from qatfactory.utils import apply_dotted_overrides, load_yaml_config, resolve_model_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qatfactory.grpo_bf16")


def main() -> None:
    from trl import GRPOConfig, GRPOTrainer

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    known, overrides = parser.parse_known_args()

    config = apply_dotted_overrides(load_yaml_config(known.config), overrides)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = dict(config.get("train", {}))
    wandb_cfg = config.get("wandb", {})

    model_dir = resolve_model_dir(model_cfg["model_path"])
    tokenizer = load_tokenizer(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
    )

    train_dataset, eval_dataset = build_datasets(data_cfg)
    reward_funcs = resolve_reward_funcs(data_cfg.get("reward_funcs", ["accuracy_reward"]))
    resume_from_checkpoint = train_cfg.pop("resume_from_checkpoint", None)
    fsdp, fsdp_config = resolve_fsdp_options(train_cfg, policy)

    report_to = ["wandb"] if wandb_cfg.get("enable") else []
    if wandb_cfg.get("enable"):
        os.environ.setdefault("WANDB_PROJECT", wandb_cfg.get("project", "qatfactory-qarl"))

    training_args = GRPOConfig(
        fsdp=fsdp,
        fsdp_config=fsdp_config,
        report_to=report_to,
        run_name=wandb_cfg.get("name"),
        use_vllm=True,
        vllm_mode="colocate",
        **train_cfg,
    )
    trainer = GRPOTrainer(
        model=policy,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    logger.info("BF16 GRPO complete; final checkpoint at %s", training_args.output_dir)


if __name__ == "__main__":
    main()
