#!/usr/bin/env python3
"""End-to-end MoE QAD smoke: tiny random Qwen3-MoE, FSDP2, routing replay, save, export.

Exercises the full loop the unit tests cannot: teacher/student dual forward under
gradient checkpointing + recompute-safe replay, fused-scale finalize on save, the
packed expert layout the trainer saves, param-FQN bookkeeping, and the
vLLM NVFP4 export from the trainer-saved checkpoint.

Run (2 GPUs):
    torchrun --nproc_per_node=2 tests/e2e/smoke_moe_qad.py /tmp/qatf_smoke
"""

import json
import os
import shutil
import sys

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, TrainingArguments
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qatfactory.data import QADCollator  # noqa: E402
from qatfactory.distill.kd_loss import IGNORE_INDEX  # noqa: E402
from qatfactory.quant import apply_fake_quant  # noqa: E402
from qatfactory.quant.export import export_nvfp4_vllm_checkpoint  # noqa: E402
from qatfactory.trainer import FusedScaleRefreshCallback, QADConfig, QADTrainer  # noqa: E402


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    work_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qatf_smoke"
    base_dir = os.path.join(work_dir, "base")
    output_dir = os.path.join(work_dir, "outputs")
    export_dir = os.path.join(work_dir, "export")
    rank = int(os.environ.get("RANK", "0"))

    if rank == 0:
        shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir)
        torch.manual_seed(0)
        config = Qwen3MoeConfig(
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_experts=8,
            num_experts_per_tok=2,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=128,
            max_position_embeddings=256,
            norm_topk_prob=True,
            tie_word_embeddings=False,
        )
        Qwen3MoeForCausalLM(config).save_pretrained(base_dir)
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.is_initialized():
        dist.barrier()

    def load():
        return AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16, attn_implementation="sdpa")

    student = load()
    converted = apply_fake_quant(student, TARGETS, fused_runtime_scales=True, quantize_activations=True)
    assert converted.experts, "tiny Qwen3-MoE must expose packed experts"

    torch.manual_seed(1234)
    dataset = []
    for _ in range(16):
        tokens = torch.randint(0, 128, (48,))
        labels = tokens.clone()
        labels[:16] = IGNORE_INDEX
        dataset.append({"input_ids": tokens.tolist(), "labels": labels.tolist()})

    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=4,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=42,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp="full_shard auto_wrap" if dist.is_initialized() else "",
        fsdp_config=(
            {"fsdp_version": 2, "version": 2, "transformer_layer_cls_to_wrap": ["Qwen3MoeDecoderLayer"]}
            if dist.is_initialized()
            else {}
        ),
    )
    qad_config = QADConfig(
        fused_runtime_scales=True,
        logits_chunk_size=64,
        target_modules=converted.all,
        replay_decay_start=0.5,
        replay_decay_end=0.75,
    )
    trainer = QADTrainer(
        model=student,
        args=args,
        data_collator=QADCollator(pad_token_id=0),
        train_dataset=dataset,
        teacher=load,  # lazy loader: materialized only after the student is sharded
        qad_config=qad_config,
        converted=converted,
        routing_replay_mode="mask",
        low_bit_optimizer="adamw8bit_sr",  # production regime: BF16 + stochastic rounding
        original_config_path=os.path.join(base_dir, "config.json"),
        callbacks=[FusedScaleRefreshCallback()],
    )
    trainer.train()
    logged = {key for entry in trainer.state.log_history for key in entry}
    assert "distill_loss" in logged, f"distill_loss missing from logs: {sorted(logged)}"
    assert "routing_flip_rate" in logged, f"routing metrics missing from logs: {sorted(logged)}"
    trainer.replay.remove()
    trainer.save_model(output_dir)

    if rank == 0:
        saved = json.load(open(os.path.join(output_dir, "config.json")))
        metadata = saved["qatfactory_qad"]
        expert_targets = [name for name in metadata["target_modules"] if ".experts." in name]
        assert len(expert_targets) == 2 * 8 * 3, metadata  # per-expert w1/w3/w2 for 2 layers
        summary = export_nvfp4_vllm_checkpoint(
            source=output_dir,
            model_assets=base_dir,
            output=export_dir,
            device="cpu",
        )
        assert summary["quantized_expert_linears"] == 2 * 8 * 3, summary
        assert summary["carried_asset_tensors"] == 0, summary  # no raw expert-weight leak
        print("SMOKE OK:", json.dumps(summary, indent=2, sort_keys=True))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
