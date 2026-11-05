#!/bin/bash
# Compute approximate forward KL(teacher || student) between one reference model
# and every other model in an endpoints config file, by querying the live vLLM
# servers (see measure_kld.py). The teacher's model_path doubles as the shared
# tokenizer, so all endpoints must use the same tokenizer. A 6th column `sglang`
# (or ENGINE=sglang) queries that endpoint through SGLang's native /generate.
#
# Usage: run_kld.sh [endpoints_file]
#   endpoints_file : pipe-separated config (default: $ENDPOINTS_FILE, else
#                    endpoints.txt beside this script).
#
# Env knobs:
#   TEACHER_LABEL  label (col 1) of the reference model  (default: original-bf16)
#   TOP_K          prompt_logprobs K; must be <= each server's --max-logprobs
#                  (vLLM default 20)                     (default: 20)
#   NUM_SAMPLES    samples per dataset                    (default: 100)
#   MAX_SEQ_LEN    truncate each sample to this many tokens (default: 8192)
#   DATASETS       space-separated subset of {ultrachat pile wikitext}
#                                                         (default: all three)
#   BATCH_SIZE     samples fetched in parallel per batch  (default: 4)
#   HOST           host the servers listen on             (default: localhost)
#   OUTPUT         results JSON path            (default: outputs/kld/kld_results.json)
#   ENGINE         engine for lines without a 6th column  (default: vllm)
#
# Requires evalscope's env or any env with transformers + datasets + requests
# (e.g. `conda activate qatfactory`). Run from anywhere; paths resolve to repo root.
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_CONFIG="$(cd "$(dirname "$0")" && pwd)/endpoints.txt"
CONFIG="${1:-${ENDPOINTS_FILE:-$DEFAULT_CONFIG}}"
[[ -f "$CONFIG" ]] || { echo "endpoints config not found: $CONFIG" >&2; exit 2; }

TEACHER_LABEL="${TEACHER_LABEL:-original-bf16}"
TOP_K="${TOP_K:-20}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
DATASETS="${DATASETS:-ultrachat pile wikitext}"
BATCH_SIZE="${BATCH_SIZE:-4}"
HOST="${HOST:-localhost}"
OUTPUT="${OUTPUT:-outputs/kld/kld_results.json}"
ENGINE="${ENGINE:-vllm}"

teacher_arg=""
tokenizer=""
student_args=()
while IFS='|' read -r label gpu port served model_path engine _rest; do
  label="$(echo "${label:-}" | xargs)"
  [[ -z "$label" || "${label:0:1}" == "#" ]] && continue
  port="$(echo "${port:-}" | xargs)"
  served="$(echo "${served:-}" | xargs)"
  model_path="$(echo "${model_path:-}" | xargs)"
  engine="$(echo "${engine:-}" | xargs)"; engine="${engine:-$ENGINE}"
  [[ -z "$port" || -z "$served" ]] && { echo "skipping malformed line '$label'" >&2; continue; }
  url="http://${HOST}:${port}/v1"
  if [[ "$label" == "$TEACHER_LABEL" ]]; then
    teacher_arg="${label}=${url}=${served}=${engine}"
    tokenizer="$model_path"
  else
    student_args+=(--student "${label}=${url}=${served}=${engine}")
  fi
done < "$CONFIG"

[[ -n "$teacher_arg" ]] || { echo "teacher label '$TEACHER_LABEL' not found in $CONFIG" >&2; exit 2; }
[[ ${#student_args[@]} -gt 0 ]] || { echo "no student endpoints found in $CONFIG" >&2; exit 2; }

echo "=== KLD: teacher=${TEACHER_LABEL}, ${#student_args[@]} students (=$(( ${#student_args[@]} / 2 ))), "
echo "    top_k=${TOP_K}, num_samples=${NUM_SAMPLES}, max_seq_len=${MAX_SEQ_LEN}, datasets='${DATASETS}' ==="

python3 "${SCRIPT_DIR}/measure_kld.py" \
  --teacher "$teacher_arg" \
  "${student_args[@]}" \
  --tokenizer "$tokenizer" \
  --datasets $DATASETS \
  --num-samples "$NUM_SAMPLES" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --top-k "$TOP_K" \
  --batch-size "$BATCH_SIZE" \
  --output "$OUTPUT"
