#!/usr/bin/env python3
"""Evaluate one QAD checkpoint on the math/code × 4K/32K held-out matrix."""

import argparse
import json
import os
import sys
from pathlib import Path

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if "LOCAL_RANK" in os.environ:
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
triton_root = os.environ.get("QATFACTORY_TRITON_CACHE_ROOT", "/tmp/qatfactory-triton")
os.environ["TRITON_CACHE_DIR"] = os.path.join(triton_root, f"rank-{local_rank}")
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from transformers import AutoConfig, AutoTokenizer, TrainingArguments


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qatfactory.data import QADCollator, build_tokenized_dataset  # noqa: E402
from qatfactory.models import load_student_and_teacher  # noqa: E402
from qatfactory.trainer import QADConfig, QADTrainer  # noqa: E402
from qatfactory.utils import load_yaml_config, resolve_model_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-proc", type=int, default=16)
    args = parser.parse_args()

    checkpoint = resolve_model_dir(args.checkpoint)
    teacher_dir = resolve_model_dir(args.teacher)
    config = load_yaml_config(args.config)
    qad_cfg = config["qad"]

    tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_type = AutoConfig.from_pretrained(checkpoint).model_type

    student, teacher, converted = load_student_and_teacher(
        checkpoint,
        qad_cfg["target_modules"],
        quant_format=qad_cfg.get("quant_format", "nvfp4"),
        teacher_model_path=teacher_dir,
        fused_runtime_scales=qad_cfg.get("fused_runtime_scales", True),
        quantize_activations=qad_cfg.get("quantize_activations", False),
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
    )

    data_dir = Path(args.data_dir)
    eval_datasets = {}
    for domain in ("math", "code"):
        path = data_dir / f"{domain}_eval.jsonl"
        for length_name, max_seq_len in (("4k", 4096), ("32k", 32768)):
            eval_datasets[f"{domain}_{length_name}"] = build_tokenized_dataset(
                str(path),
                tokenizer,
                max_seq_len,
                model_type=model_type,
                cache_dir=args.cache_dir,
                num_proc=args.num_proc,
                truncate=True,
            )

    training_args = TrainingArguments(
        output_dir=os.path.join("/tmp", f"qad-transfer-{Path(args.checkpoint).name}"),
        per_device_eval_batch_size=1,
        dataloader_num_workers=2,
        bf16=True,
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        prediction_loss_only=True,
    )
    qad_config = QADConfig(
        temperature=qad_cfg.get("temperature", 1.0),
        distill_weight=qad_cfg.get("distill_weight", 1.0),
        hard_label_weight=qad_cfg.get("hard_label_weight", 0.0),
        fused_runtime_scales=qad_cfg.get("fused_runtime_scales", True),
        logits_chunk_size=qad_cfg.get("logits_chunk_size", 256),
        streamed_logits=qad_cfg.get("streamed_logits", True),
        kernel_warmup_seq_len=0,
        quant_format=qad_cfg.get("quant_format", "nvfp4"),
        target_modules=converted.all,
        metrics_top_k=tuple(qad_cfg.get("metrics_top_k", (1, 5, 10))),
        metrics_logging_only=False,
    )
    trainer = QADTrainer(
        model=student,
        args=training_args,
        data_collator=QADCollator(tokenizer.pad_token_id),
        eval_dataset=eval_datasets,
        teacher=teacher,
        qad_config=qad_config,
        converted=converted,
        original_config_path=os.path.join(teacher_dir, "config.json"),
    )
    metrics = trainer.evaluate()

    if trainer.args.should_save:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
