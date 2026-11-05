#!/usr/bin/env python3
"""Entry point for NVFP4 quantization-aware reinforcement learning (GRPO) of a causal LM.

The policy trains as NVFP4 fake-quantized latents while rolling out on the packed NVFP4
export in a colocated vLLM engine, so the sampled policy is the deployed one.

Launch with torchrun, e.g.::

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \\
        scripts/train_llm_qarl.py configs/llm/qwen3_8b_nvfp4_qarl.yaml \\
        --train.learning_rate 1e-6 --train.max_steps 500
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qatfactory.data import load_tokenizer  # noqa: E402
from qatfactory.rl.data import build_datasets, resolve_reward_funcs  # noqa: E402
from qatfactory.rl.grpo_trainer import QuantGRPOTrainer  # noqa: E402
from qatfactory.rl.model import load_qarl_student  # noqa: E402
from qatfactory.trainer.fsdp import resolve_fsdp_options  # noqa: E402
from qatfactory.utils import apply_dotted_overrides, load_yaml_config, resolve_model_dir  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qatfactory.qarl")


def main() -> None:
    from trl import GRPOConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    known, overrides = parser.parse_known_args()

    config = apply_dotted_overrides(load_yaml_config(known.config), overrides)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = dict(config.get("train", {}))
    qad_cfg = config.get("qad", {})
    wandb_cfg = config.get("wandb", {})

    # Latents may be a local dir or a bare Hub repo id; the rollout export is always a local
    # directory produced by scripts/export_nvfp4_vllm.py.
    model_dir = resolve_model_dir(model_cfg["model_path"])
    rollout_dir = model_cfg["rollout_path"]
    if not os.path.isdir(rollout_dir):
        raise FileNotFoundError(f"model.rollout_path {rollout_dir!r} is not a directory.")

    tokenizer = load_tokenizer(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    student, converted = load_qarl_student(
        model_dir,
        rollout_dir,
        qad_cfg["target_modules"],
        quantize_activations=qad_cfg.get("quantize_activations", False),
        track_activation_scales=qad_cfg.get("sync_activation_scales", False),
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
    )

    train_dataset, eval_dataset = build_datasets(data_cfg)
    reward_funcs = resolve_reward_funcs(data_cfg.get("reward_funcs", ["accuracy_reward"]))

    resume_from_checkpoint = train_cfg.pop("resume_from_checkpoint", None)

    fsdp, fsdp_config = resolve_fsdp_options(train_cfg, student)

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

    trainer = QuantGRPOTrainer(
        model=student,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        rollout_model_path=rollout_dir,
        converted=converted,
        original_config_path=os.path.join(model_dir, "config.json"),
        sync_activation_scales=qad_cfg.get("sync_activation_scales", False),
        activation_scale_warmup_steps=qad_cfg.get("activation_scale_warmup_steps", 8),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    logger.info("QARL complete; final checkpoint at %s", training_args.output_dir)


if __name__ == "__main__":
    main()
