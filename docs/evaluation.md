# Evaluation: task accuracy and KL divergence

Two complementary quality measurements against already-running vLLM
OpenAI-compatible endpoints, both driven by one shared endpoints config file:

1. **Task accuracy** — `evalscope eval` (Native backend, `openai_api`
   eval-type); see "Usage".
2. **Distributional fidelity** — approximate forward KL-divergence to a
   reference model over top-k logprobs; see "KL-divergence vs. a reference
   model".

Typical end-to-end flow: launch the servers (see "Launching the servers"),
then run task accuracy and/or KLD against them.

The task-accuracy suite splits into FIVE `evalscope eval` invocations per
endpoint (groups a-e; `run_all_endpoints.sh` runs every endpoint in parallel,
and within each endpoint the five groups run as parallel processes — see
`scripts/eval/quality/run_endpoint_eval.sh` header for the exact dataset list per group):

- **Group a**: gpqa_diamond, winogrande, hellaswag (`--limit 300`, no sandbox/judge).
- **Group b**: mmlu_pro, mmmlu (multi-subset, `--limit 21`/subset -> ~294 total).
- **Group c**: bigcodebench (full 1140-task set, `--limit 300`; Docker sandbox,
  custom keep-alive image + grading patch). Replaces the deprecated
  `bigcodebench_hard`.
- **Group d**: live_code_bench, subset `release_v6` (Docker sandbox).
- **Group e**: aime25, `--repeats 5` (average accuracy across 5 attempts/problem).

These scripts are **agnostic of the Python env, model, and endpoint names** — all
of those come from a plain-text endpoints config file (see "Endpoints config"
below), so the same scripts evaluate any batch of served models. The scripts
never hardcode an env name; whatever `vllm`/`evalscope` is first on `PATH` is used.

## Prerequisites

The scripts assume the following are available (nothing is auto-installed):

- **Serving env** — any Python env with **vLLM** on `PATH` (`pip install vllm`;
  validated on vLLM 0.26.0). Activate it for `serve_endpoints.sh`
  (e.g. `conda activate <serve_env>`). GPUs: one per model in the config.
  - **If `vllm serve` dies instantly with a `torchcodec` / `libtorchcodec_core*.so`
    traceback, export `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` before
    launching.** For a multimodal arch (e.g. Qwen3.5 VL), `vllm serve` imports the
    multimodal stack → `torchcodec` → an FFmpeg shared library, which can dlopen the
    env's `libopenvino.so` and fail against the *system* `/lib/x86_64-linux-gnu/
    libstdc++.so.6` (`version 'CXXABI_1.3.15' not found`) when the conda env ships a
    newer `libstdc++`. The failure is at CLI import time, before args are parsed, so
    every server in the config dies identically and `serve_endpoints.sh` just reports
    NOT READY. vLLM guards this import with `except ImportError`, but torchcodec
    raises `RuntimeError`/`OSError`, so the guard does not catch it. Verify with
    `LD_LIBRARY_PATH=$CONDA_PREFIX/lib python -c "import torchcodec"`.
- **Eval env** — any Python env with **evalscope[sandbox]** on `PATH`
  (`pip install 'evalscope[sandbox]'`; validated on evalscope 1.9.1). Activate it
  for `run_endpoint_eval.sh` / `run_all_endpoints.sh` / `run_kld.sh`
  (e.g. `conda activate <eval_env>`). This env also needs **transformers**,
  **datasets**, and **requests** for KLD (evalscope pulls the first two in; add
  `requests` if missing).
- **Docker** — required for the coding benchmarks (BigCodeBench and LiveCodeBench
  run generated code inside Docker sandboxes). The invoking user must be able to
  run `docker`. Pre-pull the base images ONCE to avoid concurrent-pull
  contention across endpoints:

  ```bash
  docker pull bigcodebench/bigcodebench-evaluate:latest   # BigCodeBench (group c)
  docker pull python:3.11-slim                            # LiveCodeBench (group d)
  ```

- **BigCodeBench grading patch** — BigCodeBench requires patching the installed
  evalscope once (unpatched evalscope <= 1.9.1 false-passes every sample; see the
  grading-bug note under "Key facts"):

  ```bash
  conda activate <eval_env>
  python scripts/eval/quality/patch_evalscope_bigcodebench.py
  ```

- **Writable HF cache (KLD only)** — if `$HF_HOME` points at a shared/read-only
  cache, set it somewhere writable (`export HF_HOME=cache/hf`) before `run_kld.sh`
  or the ultrachat/wikitext download fails with `PermissionError`.
- **GPU memory** — task-accuracy benchmarks use `--gpu-memory-utilization 0.90`
  (default); long-sequence KLD may need a `GPU_MEM_UTIL=0.6` relaunch (see KLD
  section).

## Endpoints config

One pipe-separated file is the single source of truth for a run and is read by
BOTH the serving and eval scripts. Format (blank lines / `#` comments ignored,
field whitespace trimmed):

```
# label|gpu|port|served_model_name|model_path
my-quant-model|0|8001|My-Model-quant|checkpoints/My-Model-quant
original-bf16|1|8002|My-Model|/path/to/My-Model-bf16
```

- `label` → output dir slug (`outputs/evalscope/<label>/`).
- `gpu` → CUDA device the server binds to (one GPU per model).
- `port` → localhost port of the vLLM OpenAI server.
- `served_model_name` → id from `GET /v1/models`, passed as `--model` to evalscope.
- `model_path` → HF checkpoint dir (local path or Hub repo id).

`scripts/eval/quality/endpoints.example.txt` is a copyable template; copy it to
`scripts/eval/quality/endpoints.txt` (gitignored) for the current run's models. Point scripts at a different file via arg or
`$ENDPOINTS_FILE` to evaluate other models.

## Launching the servers

Serve every model in the config, one GPU each, env-agnostic:

```bash
conda activate <serve_env>          # any env with vLLM, e.g. `vllm`
bash scripts/eval/quality/serve_endpoints.sh \
  scripts/eval/quality/endpoints.txt 65536
```

- A sixth column `sglang` in the endpoints file (or `ENGINE=sglang`) serves that model with SGLang instead of
  vLLM; the eval and KL scripts work against either engine. See [sglang.md](sglang.md).
- Second arg (or `$MAX_MODEL_LEN`) is `--max-model-len` (default 65536). Keep it
  comfortably above the eval `max_tokens` (32768) plus the largest prompt: at
  65536 a 32K-token response leaves 32K for the prompt.
- **`--no-enable-flashinfer-autotune` is always passed** — FlashInfer autotuning
  adds minutes of startup kernel search with no serving benefit here, so it is
  disabled for faster launch.
- Also sets `--gpu-memory-utilization` (`$GPU_MEM_UTIL`, default 0.90); append
  arbitrary extra flags via `$EXTRA_VLLM_ARGS`.
- `$MAX_LOGPROBS` (unset by default) adds `--max-logprobs`; needed only for KLD
  with `top_k > 20` (vLLM's default cap). See the KLD section.
- Servers are `nohup`/`disown`ed so they survive the launcher exiting; the script
  then blocks until each answers `/v1/models` (up to `$READY_TIMEOUT`, default
  900s). Per-server logs: `outputs/serve_logs/<label>.log`.

> **Use the 0.90 default for task-accuracy benchmarks** — the high-concurrency
> eval workload (up to 256 in flight, long generations) needs a large KV cache to
> be effective, so do NOT lower it for benchmarking. KLD is different: its long
> `prompt_logprobs` requests need spare VRAM for activations and can OOM at 0.90.
> If a KLD run OOMs (`EngineDeadError` → connection refused), relaunch the servers
> with `GPU_MEM_UTIL=0.6` just for KLD (see the KLD section), then restore 0.90
> for further benchmarking.

Check liveness / relaunch a single one manually:

```bash
for p in $(awk -F'|' '$1!~/^#/ && NF>=5 {gsub(/ /,"",$3); print $3}' \
    scripts/eval/quality/endpoints.txt); do
  curl -s "http://localhost:$p/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])"
done
```

## Key facts

- **CRITICAL NVFP4 W4A4 serving bug (vLLM 0.25.0): a forward pass of EXACTLY 24
  tokens returns garbage. Drop 24 from `cudagraph_capture_sizes` before
  benchmarking any W4A4 checkpoint, or a chunk of every score is noise.**
  Symptom: the model emits one nonsense token (observed: `'apun'`) then
  end-of-turn, `finish_reason=stop`, `completion_tokens=2`; the first-token
  distribution is flat garbage (top-1 at logprob ~-4.2) while BF16 puts ~all mass
  on the right token. It looks like a destroyed checkpoint, and it is NOT — the
  same weights served W4A16 reproduce BF16's first-token logprobs to two decimals.
  Measured on Qwen3.5-9B NVFP4 exports:

  | serving config | token counts diverging from BF16 |
  |---|---|
  | W4A4, default | 17, 18, 19, 20, 21, 22, 23, **24** |
  | W4A4, `--enforce-eager` | **24** only |
  | W4A4, capture size 24 removed | none |
  | W4A16 | none |

  So there is ONE bad size (24); CUDA graphs pad every 17–24-token batch up into
  the captured 24-token graph and spread it across the band. `--enforce-eager`
  narrows it to 24 but does not fix it (and costs throughput). The fix is to make
  those batches pad to 32 instead, by passing the default capture list minus 24:

  ```bash
  CC=$(python3 -c "
  import json
  d=[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,
     176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,
     400,416,432,448,464,480,496,512]
  print(json.dumps({'cudagraph_capture_sizes':[x for x in d if x!=24]},separators=(',',':')))")
  EXTRA_VLLM_ARGS="--compilation-config $CC" bash .../serve_endpoints.sh <config> 65536
  ```

  (Keep the JSON space-free: `serve_endpoints.sh` expands `$EXTRA_VLLM_ARGS`
  unquoted, so any space splits the argument.) The trigger is **tokens in the
  forward pass, not prompt length** — a decode step with 17–24 concurrent
  sequences is also a 17–24-token forward, so this corrupts output mid-run as
  concurrency drains at the end of a benchmark, not just on short prompts. Not
  root-caused below the kernel: why 24 specifically is unknown (it is not simply
  "not a multiple of 16" — 40 is not either and is fine), and only vLLM 0.25.0 +
  Qwen3.5 NVFP4 was tested, so re-check on other versions/formats rather than
  assuming. Cheap pre-flight before trusting any W4A4 numbers: sweep prompt token
  counts ~10–40 against the BF16 reference via `/v1/completions` with
  `max_tokens:1, temperature:0, logprobs:1` and compare top-1 tokens; a benign
  near-tie flip (both models plausible, close logprobs) is normal, a flat garbage
  distribution is this bug.
- **CRITICAL BigCodeBench grading bug (inflates scores to ~100%): apply the
  patch before trusting any BigCodeBench number.** In evalscope <= 1.9.1,
  `BigCodeBenchAdapter.match_score` builds the sandbox check program as
  `solution + '\n' + problem['test']`. The dataset's `test` field only DEFINES a
  `unittest.TestCase` subclass — nothing runs it — so the script exits 0 for any
  syntactically-valid completion and EVERY sample grades `success`/`acc True`
  (observed ~142/148 "pass" with empty execution output; this is why early
  scores looked implausibly high, e.g. 0.9+). This affects **both `bigcodebench`
  and the deprecated `bigcodebench_hard`** — `BigCodeBenchHardAdapter` subclasses
  `BigCodeBenchAdapter` with `pass`, so they share the one buggy `match_score`.
  It is a grading bug in the Python adapter, NOT a sandbox/Docker issue (the
  keepalive image below is a separate, still-needed fix). Fixed upstream in
  evalscope 1.14.0 ([modelscope/evalscope#1859](https://github.com/modelscope/evalscope/pull/1859)), which needs no patch; for older versions, append an explicit
  `unittest.main(argv=['__bcb_check__'], verbosity=1)` runner so failing tests
  raise a non-zero `SystemExit` → sandbox status `error` → `acc False`. Apply it
  to the installed evalscope with the idempotent patcher in `scripts/eval/quality/`, then
  (re-)run group c:

  ```bash
  conda activate <eval_env>
  python scripts/eval/quality/patch_evalscope_bigcodebench.py
  bash scripts/eval/quality/run_all_endpoints.sh "c"
  ```

  The patcher edits the active env's site-packages (found via `import evalscope`),
  writes a one-time `.bak`, and no-ops if already patched (detects the
  `__bcb_check__` sentinel). Any BigCodeBench results collected before patching
  are invalid and should be discarded/re-run.
- **`--limit N` = N samples per subset**, applied uniformly to every dataset
  in a single `evalscope eval` invocation. Multi-subset tasks (MMLU-Pro: 14
  subjects, MMMLU: 14 languages) need a separate, smaller `--limit` (e.g. 21,
  giving 14×21=294) to keep the per-task total under a budget — they cannot
  share a command with single-subset tasks that want a larger `--limit`.
- **`--repeats N` duplicates each dataset item N times** (after `--limit` is
  applied). For a benchmark whose aggregation is a plain `mean` (like
  aime25), this makes the reported score automatically the average accuracy
  across all N attempts per problem — no extra aggregation needed. Keep a
  repeated dataset in its own invocation so `--repeats` doesn't also multiply
  cost for unrelated datasets sharing the command.
- **CRITICAL sandbox bug (the actual root cause of near-100% `409 Conflict:
  container ... is not running` failures, ~95% of samples, looking like
  random flakiness): `bigcodebench/bigcodebench-evaluate:latest` sets BOTH an
  ENTRYPOINT (`["python3","-m","bigcodebench.evaluate"]`, a one-shot CLI) and
  a CMD (a cleanup script that exits in ~1s).** ms_enclave's
  `DockerSandboxConfig` can only override CMD, never ENTRYPOINT — so even a
  `--sandbox` `default_config.command` override becomes ARGUMENTS appended
  after the fixed entrypoint (e.g. `python3 -m bigcodebench.evaluate tail -f
  /dev/null`), which the CLI rejects as bad args and exits (code 2). Either
  way the container dies almost immediately after the pool warms up, and
  every subsequent pooled `exec` 409s. `live_code_bench`'s default image
  (`python:3.11-slim`) is unaffected *only by accident*: it has no
  ENTRYPOINT, so its default CMD (bare `python3` REPL) runs directly and
  blocks forever on stdin (ms_enclave sets `stdin_open=True, tty=True`).
  **Fix**: evalscope's `bigcodebench` extra_param `docker_build_context`
  builds a local image and swaps ONLY `BenchmarkMeta.sandbox_config['image']`
  (`memory_limit`/`tools_config` preserved) — point it at
  `docker/bigcodebench-keepalive/Dockerfile` (in `scripts/eval/quality/`), which does
  `FROM bigcodebench/bigcodebench-evaluate:latest` + `ENTRYPOINT []` to clear
  the bad entrypoint, so a `--sandbox default_config.command: "tail -f
  /dev/null"` override can actually keep the container alive. (Applies to the
  full `bigcodebench` set used now and the deprecated `bigcodebench_hard` alike —
  same image and `_EXTRA_PARAMS`.) Verified: without this fix, BigCodeBench
  scored near-zero (only samples lucky enough to hit a still-alive container
  passed); real pass rates are far
  higher.
- Separately (real but not what caused the above): `evalscope.api.sandbox.
  SandboxService` caches one Docker sandbox manager+pool keyed by `(engine,
  manager_config)` **per process**, and that key does NOT include the
  per-benchmark `sandbox_config`/image. If two sandboxed benchmarks with
  different required images ran as `--datasets` in the *same* invocation,
  whichever runs first would "win" the pool for both. Give each sandboxed
  benchmark its own `evalscope eval` invocation (groups c and d) regardless,
  and set the sandbox `pool_size` EQUAL to `--eval-batch-size` (no
  oversubscription/queueing against the pool).
- `top_k` is a supported `--generation-config` field for vLLM/openai_api
  endpoints (not standard OpenAI API, but evalscope forwards it).

## Usage

Evaluate one endpoint (all 5 groups, or a subset):

```bash
conda activate <eval_env>           # any env with evalscope, e.g. `qatfactory`
bash scripts/eval/quality/run_endpoint_eval.sh \
  <port> <served_model_name> <label> ["a b c d e"]
```

Evaluate every endpoint in the config file in parallel (defaults to
`scripts/eval/quality/endpoints.txt`; override with a 2nd arg or `$ENDPOINTS_FILE`).
Optionally restrict to specific groups (e.g. re-running just one
previously-broken task):

```bash
conda activate <eval_env>
# all groups, default config:
bash scripts/eval/quality/run_all_endpoints.sh
# subset of groups + a custom config:
bash scripts/eval/quality/run_all_endpoints.sh \
  "c d" path/to/other_endpoints.txt
```

Results land under `outputs/evalscope/<label>/group_{a,b,c,d,e}/` (evalscope's
standard report/review/prediction JSON layout); launch logs go to
`outputs/evalscope/<label>/group_*.log`.

## KL-divergence vs. a reference model

`scripts/eval/quality/measure_kld.py` (+ the `scripts/eval/quality/run_kld.sh` wrapper) measures how far each
model's output distribution diverges from a reference model (e.g. the BF16
baseline) on real text, by querying the SAME live vLLM endpoints. It computes an
approximate forward `KL(P_teacher || P_student)` in nats, token-averaged, over
the teacher's returned top-K support (for teacher tokens outside a student's
top-K, the student's smallest returned logprob at that position is used as a
floor; teacher probs are NOT renormalized). This matches the convention of
`qatfactory.distill.kd_loss`'s `fwd_kl` metric (an equivalent offline,
model-loaded implementation of the same measure also exists in the repo).

How it works: the shared tokenizer (the teacher's `model_path`) tokenizes each
sample once, and the SAME token-id prompt is sent to every endpoint via
`POST /v1/completions` with `{max_tokens:1, temperature:0, prompt_logprobs:K}`;
vLLM returns per-position top-K next-token logprobs (`choices[0].prompt_logprobs`).
All endpoints are queried concurrently and samples are processed in batches so
memory stays bounded.

Datasets (default all three; `--datasets`/`$DATASETS` to subset), up to
`--num-samples` (default 100) each, truncated to `--max-seq-len` (default 8192)
tokens:

- `HuggingFaceH4/ultrachat_200k` (`test_sft`) — chat-templated (no gen prompt).
- `NeelNanda/pile-10k` (`train`) — raw text.
- `Salesforce/wikitext` (`wikitext-2-raw-v1`, `test`) — corpus concatenated and
  chunked into non-overlapping `max_seq_len`-token windows (rows are short
  lines). NOTE: this split is small, so it yields FEWER than `--num-samples`
  windows (e.g. ~37 at 8K, ~10 at 32K); that is expected, not a bug.

**⚠️ Server memory headroom (OOM if ignored).** A single `prompt_logprobs`
request materializes a full-vocab logits tensor for EVERY prompt position, so its
peak activation grows linearly with `--max-seq-len`. For Qwen3.5 (~250K vocab,
fp32 logits) that is roughly **~1 GiB per 1K tokens**: ~7 GiB at 8K, ~30 GiB at
32K — per concurrent request, on top of weights + KV cache. Consequences:

- The 0.90 serving default (kept high for high-concurrency benchmarks) leaves too
  little room for these activations and the engine can OOM and die mid-run
  (`EngineDeadError` → subsequent requests get connection refused). If that
  happens, relaunch the servers with `GPU_MEM_UTIL=0.6` for the KLD run — the KV
  cache is oversized for this 1-token-generation workload anyway — then restore
  0.90 for further benchmarking. (Short-sequence KLD, e.g. ≤8K, often fits at
  0.90; long-sequence KLD, e.g. 32K, generally needs the 0.6 relaunch.)
- Size `--batch-size` (samples fetched in parallel, = concurrent requests PER
  endpoint) to the sequence length: **8K → 4 (default) is safe; 32K → use
  `BATCH_SIZE=1`.** Rule of thumb: keep `batch_size × (max_seq_len/1K) GiB`
  well under the per-GPU free VRAM at your `GPU_MEM_UTIL`.
- Sanity-check `nvidia-smi` mid-run the first time at a new seq length; peak
  should stay comfortably under total VRAM.

**⚠️ `top_k` must be `<= each server's --max-logprobs` (vLLM default 20).** A
larger K is rejected with HTTP 400. `--top-k` therefore defaults to 20 so it
works against default-launched servers; for higher K, (re)launch the servers
with `MAX_LOGPROBS >= K`, e.g.:

```bash
conda activate <serve_env>
MAX_LOGPROBS=100 bash scripts/eval/quality/serve_endpoints.sh \
  scripts/eval/quality/endpoints.txt 65536
```

Run it (reads the same `endpoints.txt`; `original-bf16` is the teacher by
default, every other row is a student):

```bash
conda activate <eval_env>            # needs transformers + datasets + requests
# If $HF_HOME points at a shared/read-only cache, redirect it somewhere
# writable first, or the ultrachat/wikitext download fails with PermissionError:
export HF_HOME=cache/hf
bash scripts/eval/quality/run_kld.sh
# knobs: TEACHER_LABEL, TOP_K, NUM_SAMPLES, MAX_SEQ_LEN, DATASETS, BATCH_SIZE, HOST, OUTPUT

# 32K-context variant (note BATCH_SIZE=1 for memory, separate OUTPUT to keep 8K):
MAX_SEQ_LEN=32768 BATCH_SIZE=1 OUTPUT=outputs/kld/kld_results_32k.json \
  bash scripts/eval/quality/run_kld.sh
```

Results (per-student × per-dataset KLD in nats+bits, plus a cross-dataset mean)
are written to `$OUTPUT` (default `outputs/kld/kld_results.json`) and printed as
a table; tokenized sample sets are cached under `outputs/kld/cache/` (keyed by
dataset/num_samples/max_seq_len/seed, so changing those rebuilds them).

## PTQ baselines (RTN and GPTQ)

The tables compare every QAD checkpoint with two post-training baselines of the same format,
served and evaluated exactly like the QAD exports:

- **RTN** (round-to-nearest): `scripts/eval/rtn_export.sh <recipe> <gpus> <model> <nvfp4|mxfp4> <out-dir>`
  runs the recipe for one optimizer step at learning rate 0 (weights unchanged; NVFP4 activation
  scales are calibrated on that batch) and exports it with the QAD exporter.
- **GPTQ**: `scripts/eval/gptq_export.py` (llm-compressor 0.12.0, `pip install 'llmcompressor==0.12.0'`)
  quantizes every Linear except the output head, vision tower, MTP head and embeddings with
  `GPTQModifier` (block 128, dampening 0.01, static activation order), calibrated on 512
  conversations of the recipe's training corpus drawn with `random.Random(42)`, rendered with the
  stock chat template and truncated to 8,192 tokens. One GPU is enough for an 8B-9B model:

  ```bash
  python scripts/eval/gptq_export.py --model Qwen/Qwen3.5-9B \
      --corpus data/openperfectblend_100k_Qwen3.5-9B_think_train.jsonl \
      --scheme NVFP4 --output exports/GPTQ-qwen35-nvfp4
  ```

  The NVFP4 scheme also calibrates static input global scales, so the checkpoint is W4A4-servable
  like the QAD exports.

**DeepSeek-R1-Distill BF16 reference.** Its `tokenizer_config.json` declares `LlamaTokenizerFast`
over a byte-level BPE vocab. Loaded from a local directory, transformers 5.x honors that class and
the server emits raw byte tokens (`Ġ`, `Ċ`), which zeroes the code benchmarks and breaks answer
extraction. Serve the BF16 model by Hub id (`vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B`),
which is unaffected, or from a copy whose `tokenizer_class` is `PreTrainedTokenizerFast`. QAD, RTN
and GPTQ exports already carry the corrected class.

## Adjusting the task suite or budget

Edit the `GROUP_*` variables directly in `scripts/eval/quality/run_endpoint_eval.sh`:

- Move a dataset between groups based on its subset count (`evalscope
  benchmark-info <name>` shows `Subsets`) and whether it needs a sandbox or
  LLM judge (`grep -rl "llm_judge_default = True" .../evalscope/benchmarks/`
  and check each adapter's `sandbox_config`).
- Single-subset tasks share one `--limit`; recompute `GROUP_B_LIMIT` as
  `floor(target_total / num_subsets)` for any multi-subset task.
- `GEN_CONFIG` holds `temperature`/`top_p`/`top_k`/`max_tokens`/`timeout` —
  keep `timeout` generous (long-context + large `max_tokens` requests can take
  many minutes).
- Never put two sandboxed benchmarks with different required images in the
  same `--datasets` list (see the sandbox bug above).

## Tasks not in the default suite

`trivia_qa`, `logi_qa`, `needle_haystack`, and `openai_mrcr` were dropped from
the default suite. If re-adding `needle_haystack` or `openai_mrcr`:

- **evalscope has no native RULER adapter** (Native backend); real RULER
  requires the heavier OpenCompass backend. `openai_mrcr` (context filtered to
  a long bucket) and `needle_haystack` (context lengths pushed long) were used
  as long-context proxies instead.
- **`needle_haystack` needs `--judge-strategy llm`, NOT `rule`.** Its adapter
  sets `llm_judge_default = True`; `rule` falls back to a strict
  post-normalization `exact_match`, which scores 0 for any reasoning-model
  response that embeds the correct needle phrase inside a longer `<think>`
  trace instead of emitting it standalone. Point `--judge-model-args` at one
  of your own already-running local endpoints to avoid needing an external
  API key (default judge needs `MODELSCOPE_API_KEY`); use a fixed judge model
  across all endpoints being compared, with a generous judge `max_tokens`.
- **Context length budget**: `needle_haystack`'s `context_lengths_max` (and
  `openai_mrcr`'s `max_context_size`) must leave room for `max_tokens` under
  the server's `--max-model-len`. A 400 `BadRequestError` about total tokens
  exceeding the model's max length means the prompt + `max_tokens` budget
  doesn't fit — lower the context filter, not `max_tokens`.
- **`openai_mrcr` needs to run WITHOUT `--limit`, in its own invocation.**
  evalscope's `RemoteDataLoader` applies `--limit` to the RAW record pool
  (`dataset.select(range(limit))`) *before* the adapter's own per-record
  `min/max_context_size` filtering runs, and the raw ~2400-record MRCR
  dataset is not shuffled by default. Its records happen to be ordered such
  that the first 300 are all outside a `(65536, 100000]`-token window, so
  `--limit 300` combined with that filter silently yields **zero** samples
  ("No samples found in any subset... Skipping" — looks like a dataset
  problem, not a limit-ordering interaction). Running it with no `--limit` at
  all lets the filter do the narrowing deterministically and correctly
  (verified via direct dataset inspection: ~187/2400 records fall in that
  window).
