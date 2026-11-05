#!/bin/bash
# Launch one OpenAI-compatible server (vLLM, or SGLang per line) per model listed
# in an endpoints config file, pinning each to a single GPU. Env-agnostic: it uses
# whatever `vllm` / `python3 -m sglang` is on PATH, so activate your serving env
# beforehand. Nothing here hardcodes an env, model, or port -- everything comes
# from the config file.
#
# Config file format (see endpoints.example.txt): one model per line,
#   label|gpu|port|served_model_name|model_path[|engine]
# engine is vllm or sglang (default: $ENGINE, else vllm); blank lines and '#'
# comments ignored; field whitespace trimmed.
#
# Usage:
#   serve_endpoints.sh <endpoints_file> [max_model_len]
#     endpoints_file : path to the pipe-separated config (required)
#     max_model_len  : vLLM --max-model-len (default 65536; env MAX_MODEL_LEN)
#
# Env knobs:
#   MAX_MODEL_LEN     override --max-model-len (arg 2 takes precedence)
#   GPU_MEM_UTIL      --gpu-memory-utilization (default 0.90)
#   MAX_LOGPROBS      if set, adds --max-logprobs (vLLM default is 20). Needed
#                     for KLD with top-k > 20 (measure_kld.py): set it >= top_k.
#   READY_TIMEOUT     seconds to wait per server for /v1/models (default 900)
#   EXTRA_VLLM_ARGS   extra flags appended verbatim to every `vllm serve`
#   ENGINE            default engine for lines without a 6th column (vllm)
#   SGLANG_MEM_FRACTION  SGLang --mem-fraction-static (default 0.85; unlike
#                     vLLM's utilization it excludes activations, so keep it
#                     below GPU_MEM_UTIL)
#   EXTRA_SGLANG_ARGS extra flags appended verbatim to every SGLang launch
#
# Notes:
# - SGLang lines run `python3 -m sglang.launch_server` with the same served
#   name, port and context length; a compressed-tensors NVFP4 export with
#   input-activation quantization is served W4A4 there too (docs/sglang.md).
# - `--no-enable-flashinfer-autotune` is always passed to vLLM: FlashInfer autotuning
#   adds minutes of startup kernel search for no serving benefit here, so it is
#   disabled for faster launch.
# - Servers are started with nohup + disown so they SURVIVE this script exiting;
#   this script only launches them and then waits until each answers /v1/models.
# - Logs: outputs/serve_logs/<label>.log (created/truncated per launch).
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root

CONFIG="${1:?usage: serve_endpoints.sh <endpoints_file> [max_model_len]}"
MAX_MODEL_LEN="${2:-${MAX_MODEL_LEN:-65536}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
ENGINE="${ENGINE:-vllm}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"
MAX_LOGPROBS="${MAX_LOGPROBS:-}"
[[ -n "$MAX_LOGPROBS" ]] && EXTRA_VLLM_ARGS="--max-logprobs ${MAX_LOGPROBS} ${EXTRA_VLLM_ARGS}"

[[ -f "$CONFIG" ]] || { echo "config file not found: $CONFIG" >&2; exit 2; }

LOGDIR="outputs/serve_logs"
mkdir -p "$LOGDIR"

labels=(); ports=()
while IFS='|' read -r label gpu port served model_path engine _rest; do
  label="$(echo "${label:-}" | xargs)"
  [[ -z "$label" || "${label:0:1}" == "#" ]] && continue
  gpu="$(echo "${gpu:-}" | xargs)"
  port="$(echo "${port:-}" | xargs)"
  served="$(echo "${served:-}" | xargs)"
  model_path="$(echo "${model_path:-}" | xargs)"
  engine="$(echo "${engine:-}" | xargs)"; engine="${engine:-$ENGINE}"
  if [[ -z "$gpu" || -z "$port" || -z "$served" || -z "$model_path" ]]; then
    echo "skipping malformed line for label '$label' (need gpu|port|served|model_path)" >&2
    continue
  fi
  echo "=== serving ${label} (${engine}): GPU ${gpu}, port ${port}, model ${model_path} ==="
  case "$engine" in
    vllm)
      command -v vllm >/dev/null 2>&1 || { echo "no 'vllm' on PATH; activate a serving env first" >&2; exit 2; }
      CUDA_VISIBLE_DEVICES="$gpu" nohup vllm serve "$model_path" \
        --served-model-name "$served" \
        --port "$port" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --no-enable-flashinfer-autotune \
        $EXTRA_VLLM_ARGS \
        > "${LOGDIR}/${label}.log" 2>&1 &
      ;;
    sglang)
      python3 -c "import sglang" 2>/dev/null || { echo "python3 cannot import sglang; activate an SGLang env first" >&2; exit 2; }
      CUDA_VISIBLE_DEVICES="$gpu" nohup python3 -m sglang.launch_server \
        --model-path "$model_path" \
        --served-model-name "$served" \
        --host 0.0.0.0 \
        --port "$port" \
        --context-length "$MAX_MODEL_LEN" \
        --mem-fraction-static "$SGLANG_MEM_FRACTION" \
        $EXTRA_SGLANG_ARGS \
        > "${LOGDIR}/${label}.log" 2>&1 &
      ;;
    *) echo "unknown engine '$engine' for label '$label' (vllm|sglang)" >&2; exit 2 ;;
  esac
  disown
  labels+=("$label"); ports+=("$port")
done < "$CONFIG"

n=${#labels[@]}
[[ "$n" -gt 0 ]] || { echo "no valid endpoints in $CONFIG" >&2; exit 2; }
echo "launched ${n} servers (max_model_len=${MAX_MODEL_LEN}, gpu_mem_util=${GPU_MEM_UTIL}); waiting up to ${READY_TIMEOUT}s each..."

status=0
for i in "${!labels[@]}"; do
  label="${labels[$i]}"; port="${ports[$i]}"
  deadline=$(( $(date +%s) + READY_TIMEOUT ))
  ok=0
  while [[ $(date +%s) -lt $deadline ]]; do
    if curl -s -m 5 "http://localhost:${port}/v1/models" 2>/dev/null | grep -q '"id"'; then
      ok=1; break
    fi
    # Fail fast if the server process already died (log will show the traceback).
    if ! pgrep -f "vllm serve .* --port ${port}( |$)" >/dev/null 2>&1 \
       && ! pgrep -f "port ${port}" >/dev/null 2>&1; then
      : # process check is best-effort; keep polling until deadline regardless
    fi
    sleep 5
  done
  if [[ "$ok" == 1 ]]; then
    echo "=== ${label} READY on :${port} ==="
  else
    status=1
    echo "=== ${label} NOT READY on :${port} after ${READY_TIMEOUT}s; see ${LOGDIR}/${label}.log ==="
  fi
done

echo "=== serve_endpoints done (status ${status}); servers keep running in background ==="
exit "$status"
