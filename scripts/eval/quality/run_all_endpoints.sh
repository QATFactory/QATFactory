#!/bin/bash
# Launch run_endpoint_eval.sh in parallel against every endpoint listed in an
# endpoints config file (the SAME file serve_endpoints.sh uses). Nothing here is
# hardcoded to a particular model set -- point it at a different config to
# evaluate a different batch of models.
#
# Usage: run_all_endpoints.sh [groups] [endpoints_file]
#   groups         : optional space-separated subset of {a,b,c,d,e}
#                    (default: "a b c d e")
#   endpoints_file : optional path to the pipe-separated config
#                    (default: $ENDPOINTS_FILE, else endpoints.txt next to
#                    this script). Format: label|gpu|port|served_model_name|model_path
#
# Requires evalscope on PATH (activate the env that has it, e.g.
# `conda activate qatfactory`). Run from anywhere; paths resolve to repo root.
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root

SCRIPT_DIR="scripts/eval/quality"
DEFAULT_CONFIG="$(cd "$(dirname "$0")" && pwd)/endpoints.txt"

TASK_GROUPS="${1:-a b c d e}"
CONFIG="${2:-${ENDPOINTS_FILE:-$DEFAULT_CONFIG}}"

[[ -f "$CONFIG" ]] || { echo "endpoints config not found: $CONFIG" >&2; exit 2; }

LOGDIR="outputs/evalscope/launch_logs"
mkdir -p "$LOGDIR"

pids=()
labels=()
while IFS='|' read -r label gpu port served model_path _rest; do
  label="$(echo "${label:-}" | xargs)"
  [[ -z "$label" || "${label:0:1}" == "#" ]] && continue
  port="$(echo "${port:-}" | xargs)"
  served="$(echo "${served:-}" | xargs)"
  if [[ -z "$port" || -z "$served" ]]; then
    echo "skipping malformed line for label '$label' (need port and served_model_name)" >&2
    continue
  fi
  echo "=== launching eval for ${label} (port ${port}, model ${served}, groups: ${TASK_GROUPS}) ==="
  bash "${SCRIPT_DIR}/run_endpoint_eval.sh" "$port" "$served" "$label" "$TASK_GROUPS" \
    > "${LOGDIR}/${label}.log" 2>&1 &
  pids+=($!)
  labels+=("$label")
done < "$CONFIG"

echo "launched ${#pids[@]} endpoint evaluations; waiting..."
status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "=== ${labels[$i]} OK ==="
  else
    status=1
    echo "=== ${labels[$i]} FAILED; see ${LOGDIR}/${labels[$i]}.log ==="
  fi
done
echo "=== all endpoint evaluations done (status ${status}) ==="
exit "$status"
