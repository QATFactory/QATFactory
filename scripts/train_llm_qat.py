#!/usr/bin/env python3
"""Entry point for NVFP4 quantization-aware distillation of a Hugging Face causal LM.

Launch with torchrun, e.g.::

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \\
        scripts/train_llm_qat.py configs/llm/qwen3_5_9b_nvfp4_qad.yaml \\
        --train.learning_rate 1e-6 --train.max_steps 4000
"""

import argparse
import logging
import os
import sys
from datetime import timedelta

import numpy as np

# Triton writes compiled kernels as non-atomic multi-file cache entries, so ranks sharing one
# cache directory can read a half-written kernel. Give each local rank its own directory before
# any Triton-backed module (fla, the FP4 kernels) is imported.
if "LOCAL_RANK" in os.environ and "TRITON_CACHE_DIR" not in os.environ:
    _triton_root = os.environ.get("QATFACTORY_TRITON_CACHE_ROOT", "/tmp/qatfactory-triton")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(_triton_root, f"rank-{os.environ['LOCAL_RANK']}")
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

import torch
from transformers import AutoConfig, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qatfactory.data import QADCollator, build_tokenized_dataset, load_tokenizer  # noqa: E402
from qatfactory.models import load_student_and_teacher  # noqa: E402
from qatfactory.trainer import FusedScaleRefreshCallback, QADConfig, QADTrainer  # noqa: E402
from qatfactory.trainer.fsdp import resolve_fsdp_options  # noqa: E402
from qatfactory.utils import apply_dotted_overrides, load_yaml_config, resolve_model_dir  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qatfactory.train")


def build_datasets(data_cfg, tokenizer, max_seq_len, dataset_kwargs):
    """Assemble the train/eval datasets.

    Two mutually-exclusive sources:

    - ``train_path`` (+ optional ``eval_path``): pre-split files, tokenized independently.
    - ``train_eval_path``: a single JSONL tokenized once, then deterministically split into a
      disjoint train / held-out eval via ``train_test_split`` (``eval_size`` examples held out,
      ``split_seed`` for reproducibility).
    """

    train_eval_path = data_cfg.get("train_eval_path")
    if train_eval_path:
        if data_cfg.get("train_path") or data_cfg.get("eval_path"):
            raise ValueError(
                "data.train_eval_path is mutually exclusive with data.train_path/data.eval_path; "
                "set either the combined file or the pre-split pair, not both."
            )
        eval_size = int(data_cfg.get("eval_size", 500))
        split_seed = int(data_cfg.get("split_seed", 42))
        full = build_tokenized_dataset(train_eval_path, tokenizer, max_seq_len, **dataset_kwargs)
        if eval_size <= 0 or eval_size >= len(full):
            raise ValueError(f"data.eval_size={eval_size} must be in (0, {len(full)}) for {train_eval_path!r}.")
        # Same partition as ``Dataset.train_test_split(test_size, seed, shuffle=True)`` —
        # ``default_rng(seed).permutation(n)``, eval = first ``eval_size`` — but selected with
        # ``keep_in_memory=True`` so no indices cache-*.arrow is written next to the tokenized
        # cache. Every rank performs this split; with ``train_test_split`` they all race to
        # write the same indices file, which on a shared filesystem (Weka/NFS) fails on some
        # ranks with a spurious FileNotFoundError. The indices are a few hundred KB.
        permutation = np.random.default_rng(split_seed).permutation(len(full))
        eval_dataset = full.select(permutation[:eval_size], keep_in_memory=True)
        train_dataset = full.select(permutation[eval_size:], keep_in_memory=True)
        logger.info(
            "Split %s into %d train / %d eval (held-out, seed=%d).",
            train_eval_path, len(train_dataset), len(eval_dataset), split_seed,
        )
        return train_dataset, eval_dataset

    train_dataset = build_tokenized_dataset(data_cfg["train_path"], tokenizer, max_seq_len, **dataset_kwargs)
    eval_dataset = (
        build_tokenized_dataset(data_cfg["eval_path"], tokenizer, max_seq_len, **dataset_kwargs)
        if data_cfg.get("eval_path")
        else None
    )
    return train_dataset, eval_dataset


def _build_training_args(train_cfg: dict, wandb_cfg: dict, *, model) -> TrainingArguments:
    """Build ``TrainingArguments`` (which is also what exports the FSDP env vars).

    ``model=None`` means the model has not been loaded yet, so the FSDP2 wrap classes must be
    named in the config instead of inferred from the instantiated modules.
    """

    fsdp, fsdp_config = resolve_fsdp_options(train_cfg, model)

    if wandb_cfg.get("enable"):
        os.environ.setdefault("WANDB_PROJECT", wandb_cfg.get("project", "qatfactory-qat"))

    return TrainingArguments(
        fsdp=fsdp,
        fsdp_config=fsdp_config,
        report_to=["wandb"] if wandb_cfg.get("enable") else [],
        run_name=wandb_cfg.get("name"),
        remove_unused_columns=False,
        label_names=["labels"],
        **train_cfg,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    known, overrides = parser.parse_known_args()

    config = apply_dotted_overrides(load_yaml_config(known.config), overrides)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = dict(config.get("train", {}))
    qad_cfg = config.get("qad", {})
    wandb_cfg = config.get("wandb", {})

    # Create the process group with the recipe's ddp_timeout before anything else does. Without
    # cpu_ram_efficient_loading, dataset caching (PartialState().main_process_first()) would
    # create it first with the 10-minute NCCL default, and TrainingArguments(ddp_timeout=...)
    # finds it initialized and silently keeps that default: a checkpoint save of a large model
    # then trips the NCCL watchdog.
    from accelerate import PartialState

    PartialState(timeout=timedelta(seconds=int(train_cfg.get("ddp_timeout", 1800))))

    # A local directory is used as-is; a bare Hub repo id (e.g. Qwen/Qwen3-8B) is
    # materialized to a local snapshot so downstream config/asset reads have an on-disk dir.
    model_dir = resolve_model_dir(model_cfg["model_path"])

    tokenizer = load_tokenizer(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        # fla's FusedRMSNormGated binds torch.cuda.current_device() at construction, before
        # accelerate sets the per-rank device; pin it from torchrun's LOCAL_RANK so
        # linear-attention modules land on their own rank's GPU.
        if "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        # The cuDNN SDPA backend failed mid-run on B200/CUDA 13 on a variable-length batch
        # ("mha_graph.execute(...).is_good()"); the native flash / mem-efficient kernels are
        # the tested paths.
        torch.backends.cuda.enable_cudnn_sdp(False)

    quant_format = qad_cfg.get("quant_format", "nvfp4")
    lora_cfg = qad_cfg.get("lora") or {}
    lora_rank = int(lora_cfg.get("rank", 0)) if lora_cfg.get("enable") else 0
    lora_alpha = float(lora_cfg.get("alpha", lora_rank * 2))
    lora_moe_hybrid_shared = bool(lora_cfg.get("moe_hybrid_shared", True))
    moe_cfg = qad_cfg.get("moe") or {}
    # YAML parses a bare `off` as False (and `null` as None); both mean disabled.
    replay_mode = moe_cfg.get("routing_replay", "off") or "off"
    replay_mode = None if replay_mode == "off" else replay_mode
    low_bit_optim = train_cfg.pop("low_bit_optim", None) or None

    # `from_pretrained` decides whether a non-main rank may skip materializing the weights by
    # reading the FSDP env vars that TrainingArguments exports. For a backbone too large for
    # one full host copy per rank the arguments therefore have to exist *before* the load,
    # which in turn means the FSDP wrap classes cannot be derived from the loaded model.
    fsdp_cfg = train_cfg.get("fsdp_config") or {}
    # transformers does not forward fsdp_config.state_dict_type to accelerate's FSDP2 plugin,
    # which then defaults to SHARDED_STATE_DICT (per-rank dist_cp shards, and a refused
    # save_only_model). Export it the way accelerate reads it.
    if train_cfg.get("fsdp") and fsdp_cfg.get("state_dict_type"):
        os.environ.setdefault("FSDP_STATE_DICT_TYPE", str(fsdp_cfg["state_dict_type"]).upper())
    # Likewise fsdp_config.activation_checkpointing (FSDP-level activation checkpointing, which
    # the 32K recipes need in place of HF gradient checkpointing).
    if train_cfg.get("fsdp") and "activation_checkpointing" in fsdp_cfg:
        os.environ["FSDP_ACTIVATION_CHECKPOINTING"] = str(bool(fsdp_cfg["activation_checkpointing"])).lower()
    ram_efficient = bool(fsdp_cfg.get("cpu_ram_efficient_loading"))
    training_args = _build_training_args(train_cfg, wandb_cfg, model=None) if ram_efficient else None

    student, teacher, converted = load_student_and_teacher(
        model_dir,
        qad_cfg.get("target_modules") or [],
        quant_format=quant_format,
        teacher_model_path=qad_cfg.get("teacher_model_path"),
        fused_runtime_scales=qad_cfg.get("fused_runtime_scales", True),
        quantize_activations=qad_cfg.get("quantize_activations", False),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_moe_hybrid_shared=lora_moe_hybrid_shared,
        expert_checkpointing=qad_cfg.get("expert_checkpointing", True),
        quantize_shared_expert=moe_cfg.get("quantize_shared_expert", True),
        # A separate teacher is loaded only once the student is sharded, so two full host
        # copies of the backbone never coexist on a rank.
        lazy_teacher=True,
        master_fp32=qad_cfg.get("master_fp32", True),
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
    )

    max_seq_len = data_cfg.get("max_seq_len", 8192)
    model_type = AutoConfig.from_pretrained(model_dir).model_type
    dataset_kwargs = dict(
        model_type=model_type,
        cache_dir=data_cfg.get("cache_dir", "cache/tokenized"),
        num_proc=data_cfg.get("num_proc"),
        chat_template_file=data_cfg.get("chat_template_file"),
        truncate=data_cfg.get("truncate", True),
    )
    train_dataset, eval_dataset = build_datasets(data_cfg, tokenizer, max_seq_len, dataset_kwargs)
    collator = QADCollator(tokenizer.pad_token_id)

    if training_args is None:
        training_args = _build_training_args(train_cfg, wandb_cfg, model=student)
    resume_from_checkpoint = training_args.resume_from_checkpoint

    qad_config = QADConfig(
        temperature=qad_cfg.get("temperature", 1.0),
        distill_weight=qad_cfg.get("distill_weight", 1.0),
        hard_label_weight=qad_cfg.get("hard_label_weight", 0.0),
        fused_runtime_scales=qad_cfg.get("fused_runtime_scales", True),
        logits_chunk_size=qad_cfg.get("logits_chunk_size", 2048),
        quant_format=quant_format,
        target_modules=converted.all,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_moe_hybrid_shared=lora_moe_hybrid_shared,
        replay_decay_start=moe_cfg.get("replay_decay_start", 0.5),
        replay_decay_end=moe_cfg.get("replay_decay_end", 0.8),
        replay_in_eval=moe_cfg.get("replay_in_eval", False),
        streamed_logits=bool(qad_cfg.get("streamed_logits", False)),
        kernel_warmup_seq_len=int(qad_cfg.get("kernel_warmup_seq_len", 0)),
        metrics_top_k=tuple(qad_cfg.get("metrics_top_k", (1, 5, 10))),
        metrics_logging_only=qad_cfg.get("metrics_logging_only", True),
        teacher_cpu_offload=qad_cfg.get("teacher_cpu_offload", False),
    )

    callbacks = [FusedScaleRefreshCallback(quant_format=quant_format)] if qad_config.fused_runtime_scales else []
    trainer = QADTrainer(
        model=student,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        teacher=teacher,
        qad_config=qad_config,
        converted=converted,
        original_config_path=os.path.join(model_dir, "config.json"),
        routing_replay_mode=replay_mode,
        low_bit_optimizer=low_bit_optim,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if trainer.replay is not None:
        trainer.replay.remove()
    trainer.save_model(training_args.output_dir)
    logger.info("Training complete; final checkpoint at %s", training_args.output_dir)


if __name__ == "__main__":
    main()
