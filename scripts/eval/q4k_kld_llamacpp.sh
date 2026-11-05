#!/bin/bash
# Deployment-side comparison for Q4_K with llama.cpp's own tooling, on the trainer's held-out
# conversations (decoded text; all tokens):
#   BF16 GGUF (convert_hf_to_gguf.py --outtype bf16) -> llama-perplexity --save-all-logits  (reference)
#   llama-quantize BF16 -> Q4_K                       (PTQ baseline: llama.cpp's MSE-optimal Q4_K)
#   the QAD GGUF from scripts/export_q4k_gguf.py
#   llama-perplexity --kl-divergence  -> exact full-vocabulary KL, top-1 agreement, PPL vs BF16
#
#   scripts/eval/q4k_kld_llamacpp.sh <hub-id-or-model-dir> <heldout.pt> <qad.gguf> <work-dir>
#
# Needs LLAMA_CPP (checkout with a CUDA build under build/bin) and the training environment on
# PATH for the held-out decode (qatfactory + transformers); the QAD GGUF must have been built
# from the same model. Reference artifacts in <work-dir> are reused across runs.
set -euo pipefail
MODEL=$1; HELDOUT=$2; QAD_GGUF=$3; WORK=$4
LLAMA_CPP=${LLAMA_CPP:?set LLAMA_CPP to a llama.cpp checkout with build/bin}
BIN=$LLAMA_CPP/build/bin
CTX=${CTX:-8192}; BATCH=${BATCH:-2048}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$WORK"
TAG=$(echo "$MODEL" | tr '/' '_')
LOG=$WORK/log-$TAG-$(basename "$QAD_GGUF" .gguf).txt
echo "=== $(date -u +%FT%TZ) Q4_K KLD: $MODEL vs $QAD_GGUF ===" | tee -a "$LOG"

TXT=$WORK/heldout-$TAG.txt
if [ ! -f "$TXT" ]; then
  python - "$MODEL" "$HELDOUT" "$TXT" <<PY
import sys, torch
sys.path.insert(0, "$REPO")
from qatfactory.data import load_tokenizer
from qatfactory.utils import resolve_model_dir
tokenizer = load_tokenizer(resolve_model_dir(sys.argv[1]))
seqs = torch.load(sys.argv[2], weights_only=False)
with open(sys.argv[3], "w", encoding="utf-8") as handle:
    for ids, _ in seqs:
        handle.write(tokenizer.decode(ids, skip_special_tokens=False) + "\n\n")
print("wrote", sys.argv[3], "conversations:", len(seqs))
PY
fi

BF16=$WORK/$TAG-bf16.gguf
if [ ! -f "$BF16" ]; then
  SNAP=$(python -c "import sys; sys.path.insert(0, '$REPO'); from qatfactory.utils import resolve_model_dir; print(resolve_model_dir('$MODEL'))")
  python "$LLAMA_CPP/convert_hf_to_gguf.py" "$SNAP" --outtype bf16 --outfile "$BF16" >> "$LOG" 2>&1
  echo "bf16 gguf: $(du -sh "$BF16" | cut -f1)" | tee -a "$LOG"
fi

BASE=$WORK/$TAG-bf16.kld
if [ ! -f "$BASE" ]; then
  "$BIN/llama-perplexity" -m "$BF16" -f "$TXT" -c "$CTX" -b "$BATCH" -ngl 99 --save-all-logits "$BASE" 2>&1 | tail -3 | tee -a "$LOG"
  [ -f "$BASE" ] || { echo "reference logits were not written" | tee -a "$LOG"; exit 1; }
fi

PTQ=$WORK/$TAG-ptq-q4_k.gguf
if [ ! -f "$PTQ" ]; then
  "$BIN/llama-quantize" "$BF16" "$PTQ" Q4_K >> "$LOG" 2>&1
  echo "ptq gguf: $(du -sh "$PTQ" | cut -f1)" | tee -a "$LOG"
fi

for pair in "PTQ-Q4_K:$PTQ" "QAD:$QAD_GGUF"; do
  name=${pair%%:*}; gguf=${pair#*:}
  echo "--- $name ($gguf) ---" | tee -a "$LOG"
  "$BIN/llama-perplexity" -m "$gguf" -f "$TXT" -c "$CTX" -b "$BATCH" -ngl 99 --kl-divergence-base "$BASE" --kl-divergence 2>&1 \
    | grep -E "^Mean|^Maximum|^KLD|^Same top|^Final|Median|99.9|PPL|Cor" | tee -a "$LOG"
done
echo "=== $(date -u +%FT%TZ) done ===" | tee -a "$LOG"
