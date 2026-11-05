#!/bin/bash
# Train Qwen3-8B Q4_K QAD (8x FSDP2) then export the final checkpoint to a GGUF whose target
# linears are losslessly Q4_K. Run from anywhere; paths resolve relative to the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=online
export PYTORCH_ALLOC_CONF=expandable_segments:True

NAME=Qwen3-8B-q4k-qad-openperfectblend-100k-1epoch-lr1e-6   # must match train.output_dir in the config
OUT=outputs/$NAME
EXPORT=exports/${NAME}.gguf
# Path to a llama.cpp checkout that provides convert_hf_to_gguf.py (builds the GGUF container).
CONVERT=${LLAMA_CPP:?set LLAMA_CPP to a llama.cpp checkout}/convert_hf_to_gguf.py

echo "=== [$(date)] START TRAINING ==="
torchrun --nproc_per_node=8 --master_port=29560 scripts/train_llm_qat.py \
  configs/llm/qwen3_8b_q4k_qad.yaml \
  --data.num_proc 32 \
  --wandb.project qatfactory-qat

echo "=== [$(date)] TRAINING DONE; START EXPORT ==="
python scripts/export_q4k_gguf.py \
  --source "$OUT" \
  --output "$EXPORT" \
  --convert-script "$CONVERT"

echo "=== [$(date)] ALL DONE; GGUF at $EXPORT ==="
