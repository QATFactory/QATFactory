#!/bin/bash
# RTN (round-to-nearest) baseline export for a recipe: run the training config for ONE optimizer
# step at learning_rate 0 with plain SGD (weights unchanged; for NVFP4 the activation scales
# are calibrated on that one batch), let the trainer write its final checkpoint, then export it
# with the same exporter as the QAD runs. The result is what serving the untrained student
# looks like — the PTQ baseline every QAD number in the README is read against.
#
#   scripts/eval/rtn_export.sh <config.yaml> <CUDA_VISIBLE_DEVICES> <model-assets> <nvfp4|mxfp4> <out-dir> [extra dotted overrides...]
#
# e.g. scripts/eval/rtn_export.sh configs/llm/qwen3_5_9b_nvfp4_qad.yaml 0,1,2,3 Qwen/Qwen3.5-9B nvfp4 exports/RTN-qwen35-nvfp4 \
#        --data.max_seq_len 512
# (the calibration step only needs a small batch; a 512-token smoke subset is enough).
set -euo pipefail
CFG=$1; GPUS=$2; ASSETS=$3; FMT=$4; OUT=$5; shift 5
REPO=$(cd "$(dirname "$0")/../.." && pwd)
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
cd "$REPO"
mkdir -p "$OUT"
if [ ! -f "$OUT/latents/model.safetensors" ]; then
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node="$NPROC" scripts/train_llm_qat.py "$CFG" \
    --train.output_dir "$OUT/latents" --train.max_steps 1 --train.learning_rate 0 --train.warmup_steps 0 --train.optim sgd \
    --train.eval_strategy no --train.save_strategy no --train.logging_steps 1 --wandb.enable false "$@"
fi
if [ ! -f "$OUT/vllm-$FMT/config.json" ]; then
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} python "scripts/export_${FMT}_vllm.py" --source "$OUT/latents" --model-assets "$ASSETS" --output "$OUT/vllm-$FMT"
fi
echo "RTN export: $OUT/vllm-$FMT"
