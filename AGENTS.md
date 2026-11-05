# AGENTS.md

Guidance for agents and contributors working in QATFactory.

## What this repo is

Quantization-aware training/distillation/RL for low-bit deployment. Today it implements
**NVFP4 and MXFP4 quantization-aware distillation (QAD)** for dense, hybrid and MoE Hugging
Face causal LMs (validated on dense Qwen3-8B and DeepSeek-R1-Distill-Llama-8B, hybrid
Qwen3.5-9B, Gemma-4-12B, and the MoEs Qwen3-30B-A3B and MiniMax-M2.7) — weight-only
(W4A16) by
default, or **W4A4** with `qad.quantize_activations: true` (input activations
fake-quantized with the serving-side rule: static-global-scale NVFP4 / fully dynamic
ceil-rule MXFP4) — and exports checkpoints that load and run in **vLLM**
(`compressed-tensors nvfp4-pack-quantized` / `mxfp4-pack-quantized`), plus **Q4_K** with
GGUF export for llama.cpp. It also implements **NVFP4 quantization-aware RL (QARL)**: TRL
GRPO whose rollouts come from a colocated vLLM engine serving the model's own packed NVFP4
export, so the sampling policy is the deployed policy. It is Hugging Face-first and does not
fork `transformers`.

## Environment

- The package is packaging-agnostic: `pip install -e .`. **Do not couple the repo to a
  specific environment manager** (no uv/conda assumptions in code, configs, or scripts).
- Use any Python env that provides the dependencies in `pyproject.toml` (torch,
  transformers, accelerate, safetensors, datasets, flash-attn). `accelerate` is required by
  the HF `Trainer` path; `flash-attn` is the default `attn_implementation` and typically
  needs a prebuilt wheel or a source build with `--no-build-isolation` against the installed
  torch/CUDA.
- Multi-GPU runs use `torchrun`; set `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` explicitly.
  Export `PYTORCH_ALLOC_CONF=expandable_segments:True` for large-sequence training to reduce
  fragmentation OOM.
- If `$HOME` lives on a network filesystem (Weka/NFS), point `TRITON_CACHE_DIR` at local disk
  (e.g. `/tmp/triton-cache`): concurrent ranks compiling the same Triton kernel race on the
  shared `~/.triton/cache` and fail with a spurious `FileNotFoundError: ...cubin`.
- Hybrid Qwen3.5 on **Hopper** needs `pip install tilelang` next to `flash-linear-attention`:
  fla refuses the gated delta-net backward under Triton >= 3.4 on H100 without it. Without
  flash-attn, launch with `--model.attn_implementation sdpa`.
- Smoke-test footprint: dense 8B fits 2 GPUs; hybrid Qwen3.5-9B full-parameter QAD needs
  4 GPUs even at `max_seq_len 512` (2 GPUs OOM at 80 GB).
- **FSDP2 sharding granularity:** `train.fsdp` must include `auto_wrap` (the default, via
  `trainer.fsdp.resolve_fsdp_options`) so each decoder layer is its own FSDP2 unit.
  `transformers` forwards `transformer_layer_cls_to_wrap` only under `auto_wrap`; a bare
  `full_shard` shards the root module only, so every rank all-gathers the entire model each
  forward/backward (+1 BF16 model copy per rank: OOM for a 30B MoE, ~2.5x slower on 9B).

## Repository layout

```
qatfactory/quant/       NVFP4 fake-quant (STE), instrumentation, vLLM export
qatfactory/distill/     distillation loss (temperature-KL + optional hard CE)
qatfactory/models/      HF loader (student/teacher construction)
qatfactory/data/        chat-template tokenization + cache; templates/*_qad.jinja
qatfactory/trainer/     HF Trainer subclass + fused-scale callback + shared checkpoint save
qatfactory/rl/          QARL: GRPO trainer + NVFP4 weight sync into a live vLLM engine
qatfactory/moe/         MoE teacher routing replay
scripts/                Python entrypoints (train_llm_qat.py, train_llm_qarl.py, export_nvfp4_vllm.py,
                        export_mxfp4_vllm.py, export_q4k_gguf.py, fuse_moe_checkpoint.py,
                        merge_lora_bf16.py, generate_distill_dataset.py) + diagnostics
examples/               shell launchers (*.sh), e.g. run_full_qwen3_8b.sh, run_qarl_qwen3_8b.sh
configs/llm/            training configs (YAML)
outputs/                training run checkpoints (gitignored)
exports/                vLLM NVFP4 serving checkpoints (gitignored)
cache/                  tokenized dataset cache (gitignored)
docs/                   user guides
.cursor/skills/         project skills: qatfactory-qad, qatfactory-qarl, qatfactory-nvfp4-export,
                        qatfactory-q4k-export
```

## Running

Unit tests (fast, no GPU needed for the quant core):

```bash
python -m pytest tests/quant tests/data -q
```

Bounded smoke before any full run (small model footprint, a few steps, 2 GPUs). `model_path`
may be a local dir or a Hub repo id (e.g. `Qwen/Qwen3-8B`), auto-downloaded on first launch:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=2 scripts/train_llm_qat.py configs/llm/qwen3_8b_nvfp4_qad.yaml \
  --data.max_seq_len 512 --train.max_steps 3 --train.eval_strategy no \
  --train.output_dir /tmp/smoke --wandb.enable false
```

For code changes, run a bounded smoke that completes training, eval and save before any full
run, and keep the unit tests passing.

Full train → export (`run_full_qwen3_8b.sh` for dense Qwen3-8B, `run_full_qwen35.sh` for
hybrid Qwen3.5-9B):

```bash
bash examples/run_full_qwen3_8b.sh   # or the torchrun + export commands in README.md
```

Any config value can be overridden with dotted CLI keys, e.g.
`--train.learning_rate 1e-6 --train.max_steps 4000 --wandb.project <project>`.

## LoRA-QAD

Set `qad.lora.enable: true` and only the injected adapters train; every base weight stays
frozen. Two properties make this different from ordinary QLoRA, and both matter downstream:

- **The merged weight is what gets quantized**, `Q(W + BA)`, not `Q(W) + BA`. Training
  therefore sees exactly the values that deployment-time merging produces, and the exported
  checkpoint carries no adapter keys.
- **The frozen base is the teacher.** `shared_base_teacher()` runs the same modules with
  the adapters and quantizers bypassed, so no second model copy is held. This is what puts a
  230B MoE on one node.

Routed experts are adapted through the fused 3D expert tensors, in one of two layouts:

| `qad.lora.moe_hybrid_shared` | gate/up `A`, down `B` | gate/up `B`, down `A` |
|---|---|---|
| `true` (shared) | one factor shared by all experts | per-expert |
| `false` (unshared) | per-expert | per-expert |

A block-FP8 base (MiniMax-M2.7) is kept in its released precision on disk and dequantized
one weight at a time inside the forward; promoting the whole backbone would double parameter
memory for no numerical gain. Such a checkpoint must be fused offline first
(`scripts/fuse_moe_checkpoint.py`), otherwise every rank materializes its own copy of the
per-expert-to-3D concatenation and a node runs out of host RAM.

Training writes only the adapters (`lora_adapters.safetensors`, next to a `config.json` that
records the run's LoRA alpha and scaling). `export_nvfp4_vllm.py` / `export_mxfp4_vllm.py`
take the run directory directly and read the frozen base from `--model-assets`. To get a BF16
checkpoint instead, fold the adapters back in:

```bash
python scripts/merge_lora_bf16.py <base_dir> <run_dir>/lora_adapters.safetensors <out_dir> \
  --device cuda:0            # add --released-layout for MoE checkpoints
```

The result is BF16 whose target weights already sit on the format's grid, so a BF16 runtime
reproduces W4A16 serving numerics. Passing `none` instead of an adapter path produces the
round-to-nearest baseline through the identical code path — a divergence number against the
reference is not interpretable without it.

Two failure modes worth knowing before debugging a flat loss curve. Selecting targets by
leaf name reaches into a vision tower, whose attention layers reuse `q_proj`/`k_proj`/
`v_proj`; the exclude-prefix list, not the target list, is what keeps the image encoder
frozen. And a quantization gap that is already small (top-1 agreement above ~0.95) leaves
little to recover, so a flat curve may be correct.

## Conventions and invariants

- **Freezing:** only quantized-linear latent weights train; every non-quantized module
  (embeddings, `lm_head`, norms, biases, conv, gates) is frozen. `apply_fake_quant` enforces
  this and raises if any non-quantized parameter is trainable. The teacher is fully frozen.
- **Tied embeddings:** models that tie `lm_head` to the input embedding (e.g. Gemma-4) are
  supported for weight-only/W4A4 QAD — the shared tensor is never trained or quantized.
  `models.llm._untie_frozen_lm_head` clones `lm_head` into its own frozen parameter at load so
  FSDP2 can shard two distinct storages; `QADTrainer.save_model` drops that redundant clone
  when the deployment config is tied, so the checkpoint keeps the tied layout.
- **Exportable runs:** set `qad.fused_runtime_scales: true`. This shares max-based tensor
  scales across the runtime-fused families (QKV, gate/up, Qwen3.5 delta-net inputs) and
  records activation running maxima, written to `input_scales.safetensors`. Only these runs
  are eligible for the vLLM exporter.
- **Bit-exactness:** the fake quantizer must match the vLLM kernel value-for-value. NVFP4:
  E2M1 values, group-16 along in-features, E4M3 block scales, FP32 tensor scale,
  round-to-nearest-even. MXFP4: E2M1 values, group-32 along in-features, EVEN-rule E8M0
  block scales (biased uint8), no global scale; oracle is the vendored
  `tests/quant/reference_mxfp4.py` (vLLM `tests/quantization/reference_mxfp4.py`). MXFP4
  W4A4 activations use a DIFFERENT scale rule than the weights — CEIL
  (`__nv_cvt_float_to_e8m0(amax * rcp(6), SATFINITE, RoundPosInf)`, never clips),
  matching flashinfer `mxfp4_quantize` (cute-dsl), the quantizer vLLM's native W4A4
  path calls at serve time. Do not change the rounding/scale math in
  `qatfactory/quant/formats/nvfp4.py` or `qatfactory/quant/formats/mxfp4.py` without a
  matching bit-exactness test.
- **Data:** conversation JSONL tokenized via the model's chat template (assistant-only loss
 mask; `reasoning_content` rendered natively). Tokenization is parallel and cached under
 `cache/tokenized/<key>/`; only the main process builds the cache. Keep train/eval splits
 disjoint. Always load the tokenizer through `data.load_tokenizer`, never bare
 `AutoTokenizer.from_pretrained`: a `tokenizer_config.json` may declare a `tokenizer_class`
 that mismatches its `tokenizer.json` backend (DeepSeek-R1-Distill names `LlamaTokenizerFast`
 over a byte-level BPE vocab), and transformers honors it for local directories — silently
 dropping every space. `load_tokenizer` probes an encode/decode round-trip and falls back to
 the generic fast backend.
- **Checkpoints:** training writes BF16 latent weights + full model config + QAD metadata to
  `outputs/<run>`; the packed low-bit artifact is a separate export under `exports/`. QAD and
  QARL emit the identical layout (`trainer/checkpoint.py`), so either is a valid export input.
  Never commit anything under `outputs/`, `exports/`, `cache/`, or `wandb/`.
- **Secrets:** never commit tokens or API keys (HF, W&B, GitHub, cloud). Before pushing, run
  `gitleaks git --log-opts="origin/main..HEAD" --redact` (or `gitleaks dir .` for the working
  tree); `.gitleaks.toml` extends the default rules and allowlists known false positives
  (HF revision pins, lm-eval `task_hashes`). A real finding must be revoked, not just deleted.
- **QARL weight sync:** rollouts run on the packed NVFP4 export, which has no BF16 parameters
  to overwrite, so TRL's stock sync is replaced by `qatfactory/rl/vllm_nvfp4_sync.py`. Two
  rules keep it honest: the tensor scale comes from the training model's `NVFP4AmaxGroup`
  (never recomputed, so rollout and training share one lattice), and the deployed layout comes
  from running vLLM's own `process_weights_after_loading` on a scratch layer (never
  reimplemented). Updates are in-place `copy_` calls because vLLM captures CUDA graphs over
  those buffers. Activation scales are pinned to the export and never rewritten. Verify with
  `scripts/verify_qarl_sync.py`: re-packing an export's own latents must reproduce it bit for
  bit, making the first sync a no-op.

## Test fixtures

`tests/fixtures/make_tiny_fp8_minimax.py` builds a tiny block-FP8 MiniMax-M2 checkpoint in the
released layout, for distributed smoke tests of the FP8 LoRA-QAD path that a unit test cannot
cover.

## Skills

Prefer the project skills for the core workflows, under `.cursor/skills/`: `qatfactory-qad`
(distillation), `qatfactory-qarl` (quantization-aware RL), `qatfactory-nvfp4-export` (vLLM
NVFP4 and MXFP4 export) and `qatfactory-q4k-export` (llama.cpp GGUF export). They describe how
to use the tools; the rules for changing the code are in this file.

## Scope

Supported: dense, hybrid and MoE LLM QAD on NVIDIA GPUs (pure FSDP2 data sharding, BF16 mixed
precision) for the registered formats `nvfp4`, `mxfp4`, `q4_k` — weight-only (W4A16)
always; W4A4 (`qad.quantize_activations`) for `nvfp4` (requires `fused_runtime_scales`)
and `mxfp4` (stateless). Validated backbones: dense Qwen3, hybrid Qwen3.5, and Gemma-4
(`gemma4_unified`, a multimodal-wrapped text backbone quantized text-only; tied embeddings
handled, k_eq_v global layers fuse q/k, SDPA required for its head_dim-512 layers). Dense
Llama-3 (`llama`, e.g. DeepSeek-R1-Distill-Llama-8B) is wired end-to-end — config, NVFP4
export, and a generation-marked DeepSeek-R1-Distill template opted into via
`data.chat_template_file`. Tied input embeddings are supported (weight-only/W4A4).

Two adaptation modes: full-parameter QAD (default; every quantized linear's latent weight
trains) and **LoRA-QAD** (base weights frozen and shared with the
teacher, rank-r adapters on every quantized linear train, quantization applied to the merged
`W + (alpha/r) B A`; enabled with `qad.lora: {enable: true, rank, alpha}`, see `quant/formats/lora.py` and
`configs/llm/qwen3_5_9b_nvfp4_lora_qad.yaml`).

**MoE LoRA-QAD** is supported (`qad.lora.enable`). Adapters attach to the quantized
linears and the merged weight is what gets fake-quantized, so training sees the values the
deployed merge produces. The frozen base doubles as the teacher (`shared_base_teacher()`),
so there is no second model copy. Routed experts are adapted through their fused 3D tensors
in either of two layouts, chosen by `qad.lora.moe_hybrid_shared`: shared (one factor of each
pair common to all experts) or fully unshared. A block-FP8 base is kept in its released
precision and dequantized one weight at a time inside the forward. Validated on MiniMax-M2.7
(230B MoE, FP8 base, experts only) and Muse-Glimmer-30B (dense, multimodal, text tower only).

**Full-parameter MoE QAD** is supported for `qwen3_moe` (Qwen3-30B-A3B: attention and routed
experts, W4A4) and `minimax_m2` (MiniMax-M2.7: routed experts only, multi-node). Routed experts
train through their fused 3D tensors; `qad.moe.routing_replay` (`mask` or `full`) optionally
replays the teacher's expert choices early in training and anneals to free routing. The MoE
recipes keep BF16 latents with `train.low_bit_optim: adamw8bit_sr` (torchao), save the
optimizer per rank for resumable multi-job runs, and write checkpoints shard by shard.

Not supported: other formats (q4_0), MXFP4 export of routed experts, and packed export of
LoRA-adapted routed experts (merge them to BF16 instead) — reject unsupported configurations
explicitly rather than falling back to ordinary fine-tuning.

QARL is narrower: dense Qwen3 `nvfp4` only, TRL GRPO, vLLM colocate mode with
tensor-parallel size 1. Server mode is rejected because the sync mutates packed tensors
inside the engine process. LoRA-QARL, MoE, and MXFP4/Q4_K rollout engines are not wired up.
It needs `trl`, `vllm` and `math-verify` beyond the base dependencies (installed via the
`rl` extra).
