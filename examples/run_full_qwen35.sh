#!/bin/bash
# Train Qwen3.5-9B NVFP4 QAD (8x FSDP2) then export the final checkpoint to vLLM NVFP4.
# Run from anywhere; paths are resolved relative to the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=online
export PYTORCH_ALLOC_CONF=expandable_segments:True

NAME=Qwen3.5-9B-nvfp4-qad-openperfectblend-100k-1epoch-lr1e-6   # must match train.output_dir in the config
OUT=outputs/$NAME
EXPORT=exports/${NAME}-vllm-nvfp4
# The base model auto-downloads from the Hub on first launch; model-assets accepts the
# same repo id for the export step. Set REF to an existing vLLM NVFP4 checkpoint of the
# same architecture for an optional schema check.
BASE=Qwen/Qwen3.5-9B
REF=${REF:-}

echo "=== [$(date)] START TRAINING ==="
torchrun --nproc_per_node=8 --master_port=29560 scripts/train_llm_qat.py \
  configs/llm/qwen3_5_9b_nvfp4_qad.yaml \
  --data.num_proc 32 \
  --wandb.project qatfactory-qat

echo "=== [$(date)] TRAINING DONE; START EXPORT ==="
CUDA_VISIBLE_DEVICES=0 python scripts/export_nvfp4_vllm.py \
  --source "$OUT" \
  --model-assets "$BASE" \
  --output "$EXPORT" \
  --device cuda:0 \
  ${REF:+--reference "$REF"}

echo "=== [$(date)] ALL DONE ==="
