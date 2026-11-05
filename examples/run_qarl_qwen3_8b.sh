#!/bin/bash
# Quantization-aware RL for Qwen3-8B-Base NVFP4 (the paper's Tables 8 and 9): GRPO on verifiable
# math, rolled out on the packed NVFP4 export, then re-export the RL-tuned policy for vLLM.
# Run from anywhere; paths are resolved relative to the repository root. The rollout export's
# activation scales are calibrated on a small batch of the Qwen3-8B corpus (docs/data.md).
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_ALLOC_CONF=expandable_segments:True
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

NAME=Qwen3-8B-Base-nvfp4-qarl-deepmath
BASE=Qwen/Qwen3-8B-Base
RTN=exports/RTN-Qwen3-8B-Base-nvfp4     # rollout engine checkpoint: RTN NVFP4 export of $BASE
OUT=outputs/$NAME
EXPORT=exports/${NAME}-vllm-nvfp4

echo "=== [$(date)] BUILD THE ROLLOUT CHECKPOINT ==="
# Writes $RTN/latents (the packed BF16 weights plus calibrated activation scales) and the
# compressed-tensors export vLLM serves, $RTN/vllm-nvfp4. Skipped if both exist.
bash scripts/eval/rtn_export.sh configs/llm/qwen3_8b_nvfp4_qad.yaml "$CUDA_VISIBLE_DEVICES" "$BASE" nvfp4 "$RTN" \
  --model.model_path "$BASE" --data.max_seq_len 512

echo "=== [$(date)] VERIFY WEIGHT SYNC AGAINST THE ROLLOUT ENGINE ==="
# Re-packing the latents the export was built from must reproduce it bit for bit.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} python scripts/verify_qarl_sync.py \
  --latents "$RTN/latents" \
  --rollout "$RTN/vllm-nvfp4"

echo "=== [$(date)] START QARL ==="
# The policy starts from $BASE, whose weights the RTN latents hold unchanged.
torchrun --nproc_per_node="$NGPU" --master_port=29570 scripts/train_llm_qarl.py \
  configs/llm/qwen3_8b_nvfp4_qarl.yaml \
  --model.model_path "$BASE" \
  --model.rollout_path "$RTN/vllm-nvfp4" \
  --train.output_dir "$OUT"

echo "=== [$(date)] QARL DONE; START EXPORT ==="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} python scripts/export_nvfp4_vllm.py \
  --source "$OUT" \
  --model-assets "$BASE" \
  --output "$EXPORT" \
  --device cuda:0

echo "=== [$(date)] ALL DONE ==="
