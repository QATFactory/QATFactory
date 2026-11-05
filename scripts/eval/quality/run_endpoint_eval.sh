#!/bin/bash
# Runs the standard QATFactory benchmark suite against ONE OpenAI-compatible
# vLLM endpoint via evalscope's Native backend, split into FIVE evalscope
# invocations:
#
#   Group A (single-subset, no sandbox/judge dependency): gpqa_diamond,
#     winogrande, hellaswag. --limit applies directly per subset.
#
#   Group B (multi-subset, 14 subsets each): mmlu_pro, mmmlu.
#     --limit is set to ~limit/14 per subset so the total stays <=limit/task.
#
#   Group E (aime25, repeated 5x per problem): its own invocation with
#     `--repeats 5`. aime25's aggregation is a plain "mean" over every scored
#     sample, and `--repeats N` duplicates each of the ~30 problems N times
#     *after* `--limit` is applied (see evalscope's RemoteDataLoader.load()),
#     so the reported score is automatically the average accuracy across all
#     5 attempts per problem -- no extra aggregation needed on our side.
#     Kept in its own invocation so `--repeats` doesn't also 5x the other
#     Group A/B tasks (they'd want their own, separate `--repeats` semantics).
#
#   Group C (bigcodebench, the full 1140-task set; --limit 300) and Group D
#     (live_code_bench, pinned to subset release_v6) each get their own
#     evalscope invocation: evalscope's SandboxService keeps one Docker sandbox
#     pool per process and ignores the per-benchmark image, so two sandboxed
#     benchmarks in one invocation would share containers. BigCodeBench runs in
#     docker/bigcodebench-keepalive/Dockerfile (passed as evalscope's
#     docker_build_context): the stock bigcodebench-evaluate image has a one-shot
#     ENTRYPOINT that exits right away, so every pooled exec failed with
#     "409 Conflict: container is not running". Grading needs evalscope >= 1.14.0
#     or patch_evalscope_bigcodebench.py. pool_size equals the eval batch size for
#     both groups, so requests never queue on the pool.
#
# Usage: run_endpoint_eval.sh <port> <served_model_name> <label> [groups]
#   port:              vLLM server port on localhost
#   served_model_name: model id as returned by GET /v1/models
#   label:             short slug used for the output directory name
#   groups:            optional space-separated subset of {a,b,c,d,e} to run
#                       (default: all five)
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root

PORT="$1"
MODEL="$2"
LABEL="$3"
TASK_GROUPS="${4:-a b c d e}"

API_URL="http://localhost:${PORT}/v1"
WORK_DIR="outputs/evalscope/${LABEL}"
mkdir -p "$WORK_DIR"

GEN_CONFIG='{"temperature":1.0,"top_p":0.95,"top_k":20,"max_tokens":32768,"timeout":1800}'
CONCURRENCY=512

LIMIT=300
GROUP_A_DATASETS="gpqa_diamond winogrande hellaswag"

GROUP_B_DATASETS="mmlu_pro mmmlu"
GROUP_B_LIMIT=21   # 21 * 14 subsets = 294 <= 300 per task

AIME_REPEATS=5

# Sandbox groups: pool_size == eval-batch-size (no oversubscription).
BCB_CONCURRENCY=32
LCB_CONCURRENCY=64
BCB_DOCKER_BUILD_CONTEXT="$(cd "$(dirname "$0")/docker/bigcodebench-keepalive" && pwd)"

pids=()
names=()

run_group() {
  local name="$1"; shift
  echo "=== [${LABEL}] ${name} starting ==="
  "$@" > "${WORK_DIR}/${name}.log" 2>&1 &
  pids+=($!)
  names+=("$name")
}

for g in $TASK_GROUPS; do
  case "$g" in
    a)
      run_group group_a evalscope eval \
        --model "$MODEL" --api-url "$API_URL" --api-key EMPTY --eval-type openai_api \
        --datasets $GROUP_A_DATASETS \
        --limit "$LIMIT" --generation-config "$GEN_CONFIG" --judge-strategy rule \
        --eval-batch-size "$CONCURRENCY" --work-dir "${WORK_DIR}/group_a" --no-timestamp
      ;;
    b)
      run_group group_b evalscope eval \
        --model "$MODEL" --api-url "$API_URL" --api-key EMPTY --eval-type openai_api \
        --datasets $GROUP_B_DATASETS --limit "$GROUP_B_LIMIT" \
        --generation-config "$GEN_CONFIG" --judge-strategy rule \
        --eval-batch-size "$CONCURRENCY" --work-dir "${WORK_DIR}/group_b" --no-timestamp
      ;;
    c)
      # Full BigCodeBench (1140 tasks), --limit 300. REQUIRES the grading patch
      # (patch_evalscope_bigcodebench.py): unpatched evalscope <= 1.9.1 never
      # runs unittest, so every sample false-passes (~100%). See SKILL.md.
      # bigcodebench/bigcodebench-evaluate:latest sets BOTH an ENTRYPOINT
      # (["python3","-m","bigcodebench.evaluate"], a one-shot CLI) and a CMD
      # (a cleanup script). ms_enclave's DockerSandboxConfig can only override
      # CMD, never ENTRYPOINT -- so even `default_config.command` overrides
      # become ARGUMENTS appended after the fixed entrypoint, e.g. our
      # "tail -f /dev/null" attempt actually ran
      # `python3 -m bigcodebench.evaluate tail -f /dev/null`, which the CLI
      # rejected as bad args and exited (code 2). Either way (default CMD or
      # a command override), the container exits almost immediately, and
      # every subsequent pooled `exec` 409s with "container ... is not
      # running" (looked like ~95% random failures, NOT a concurrency race).
      # python:3.11-slim (live_code_bench) is unaffected only by accident: it
      # has no ENTRYPOINT, so its default CMD (bare `python3` REPL) runs
      # directly and blocks forever on stdin (ms_enclave sets stdin_open+tty).
      # Fix: evalscope's own `docker_build_context` extra_param builds a
      # local image and swaps ONLY BenchmarkMeta.sandbox_config['image']
      # (memory_limit/tools_config preserved) -- point it at
      # docker/bigcodebench-keepalive/Dockerfile, which clears the entrypoint
      # so a real keep-alive CMD can take effect.
      run_group group_c evalscope eval \
        --model "$MODEL" --api-url "$API_URL" --api-key EMPTY --eval-type openai_api \
        --datasets bigcodebench --limit "$LIMIT" \
        --dataset-args "{\"bigcodebench\": {\"extra_params\": {\"docker_build_context\": \"${BCB_DOCKER_BUILD_CONTEXT}\"}}}" \
        --generation-config "$GEN_CONFIG" --judge-strategy rule \
        --sandbox "{\"enabled\": true, \"engine\": \"docker\", \"pool_size\": ${BCB_CONCURRENCY}, \"default_config\": {\"command\": \"tail -f /dev/null\"}}" \
        --eval-batch-size "$BCB_CONCURRENCY" --work-dir "${WORK_DIR}/group_c" --no-timestamp
      ;;
    d)
      run_group group_d evalscope eval \
        --model "$MODEL" --api-url "$API_URL" --api-key EMPTY --eval-type openai_api \
        --datasets live_code_bench --dataset-args '{"live_code_bench": {"subset_list": ["release_v6"]}}' \
        --limit "$LIMIT" --generation-config "$GEN_CONFIG" --judge-strategy rule \
        --sandbox "{\"enabled\": true, \"engine\": \"docker\", \"pool_size\": ${LCB_CONCURRENCY}}" \
        --eval-batch-size "$LCB_CONCURRENCY" --work-dir "${WORK_DIR}/group_d" --no-timestamp
      ;;
    e)
      run_group group_e evalscope eval \
        --model "$MODEL" --api-url "$API_URL" --api-key EMPTY --eval-type openai_api \
        --datasets aime25 --limit "$LIMIT" --repeats "$AIME_REPEATS" \
        --generation-config "$GEN_CONFIG" --judge-strategy rule \
        --eval-batch-size "$CONCURRENCY" --work-dir "${WORK_DIR}/group_e" --no-timestamp
      ;;
  esac
done

status=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { status=1; echo "[${LABEL}] ${names[$i]} FAILED"; }
done
echo "=== [${LABEL}] done (status ${status}) ==="
exit "$status"
