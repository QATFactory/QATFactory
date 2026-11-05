---
name: qatfactory-qad
description: >-
  Runs QATFactory quantization-aware distillation (QAD) for Hugging Face causal LMs — dense
  (Qwen3, Llama-3 / DeepSeek-R1-Distill), hybrid (Qwen3.5) and MoE (Qwen3-30B-A3B,
  MiniMax-M2.7) — in NVFP4 or MXFP4 (vLLM) and Q4_K (llama.cpp/GGUF), full-parameter or
  LoRA, weight-only (W4A16) or with activation quantization (W4A4), with a frozen teacher +
  fake-quantized student, assistant-only loss masking, cached parallel tokenization, FSDP2,
  and W&B. Use when the user asks for QAD, quantization-aware distillation, NVFP4/MXFP4/Q4_K
  recovery training, or a teacher/student quantized fine-tune in this repo.
---

# QATFactory Quantization-Aware Distillation

Self-distillation recovery training: a frozen full-precision teacher and a student whose
target linears are fake-quantized on every forward (masked STE). Only the quantized latent
weights (or, in LoRA mode, the adapters) train; everything else is frozen. The format is
selected by `qad.quant_format`:

- **`nvfp4`** (default) — E2M1 group-16 lattice with E4M3 block scales and an FP32 tensor
  scale, for vLLM `nvfp4-pack-quantized`; uses fused runtime scales. Export with
  `/qatfactory-nvfp4-export`.
- **`mxfp4`** — E2M1 group-32 lattice with E8M0 block scales and no global scale, for vLLM
  `mxfp4-pack-quantized`. Export with `/qatfactory-nvfp4-export` (MXFP4 section).
- **`q4_k`** / **`q4_k_grid_search`** — llama.cpp two-level GGUF `block_q4_K` (super-blocks of
  256, 8×32 affine sub-blocks); no fused/shared scales. Export with `/qatfactory-q4k-export`.

| Component | Location |
|-----------|----------|
| Trainer (HF `Trainer` subclass) | `qatfactory/trainer/qad_trainer.py` |
| Checkpoint save / sharded resume state | `qatfactory/trainer/checkpoint.py` |
| Fused-scale refresh callback (NVFP4 only) | `qatfactory/trainer/callbacks.py` |
| Quant-format registry (`nvfp4` / `mxfp4` / `q4_k` / `q4_k_grid_search`) | `qatfactory/quant/formats/__init__.py` |
| Fake quantizers (STE) | `qatfactory/quant/formats/nvfp4.py`, `mxfp4.py`, `q4_k.py` |
| LoRA-QAD (merged-weight quantization) | `qatfactory/quant/formats/lora.py` |
| Module instrumentation + freeze | `qatfactory/quant/fake_quant.py` (`apply_fake_quant`) |
| MoE teacher routing replay | `qatfactory/moe/routing_replay.py` |
| Distillation loss (KL + optional CE) | `qatfactory/distill/kd_loss.py` |
| Data (chat-template mask + cache) | `qatfactory/data/text.py`, `qatfactory/data/templates/*_qad.jinja` |
| Model loader (student/teacher) | `qatfactory/models/llm.py` |
| Model-dir resolver (local path or HF repo id) | `qatfactory/utils/hub.py` (`resolve_model_dir`) |
| Distillation data generator (teacher's own answers) | `scripts/generate_distill_dataset.py` |
| MoE checkpoint preparation | `scripts/fuse_moe_checkpoint.py`, `scripts/dequant_fp8_minimax.py` |
| Entry point | `scripts/train_llm_qat.py` |
| Launchers (train→export) | `examples/run_full_qwen3_8b.sh`, `examples/run_full_qwen35.sh` (NVFP4); `examples/run_full_qwen3_8b_q4k.sh`, `examples/run_full_qwen35_q4k.sh` (Q4_K) |
| Recipes | `configs/llm/*.yaml` — each file's header states its scope, memory needs and launch command |

## Supported scope

| Family | `model_type` | Recipes in `configs/llm/` |
|---|---|---|
| Dense Qwen3 | `qwen3` | `qwen3_8b_nvfp4_qad`, `qwen3_8b_q4k_qad` |
| Dense Llama-3 (DeepSeek-R1-Distill-Llama-8B) | `llama` | `deepseek_r1_distill_llama_8b_{nvfp4,mxfp4}_qad[_w4a4]` |
| Hybrid Qwen3.5 (gated delta-net + attention) | `qwen3_5` | `qwen3_5_9b_{nvfp4,mxfp4}_qad[_w4a4]`, `qwen3_5_9b_nvfp4_lora_qad`, `qwen3_5_9b_q4k_qad`, `qwen3_5_9b_nvfp4_qad_generalization_*` |
| MoE Qwen3-30B-A3B | `qwen3_moe` | `qwen3_30b_a3b_nvfp4_qad` (attention + routed experts, W4A4), `qwen3_30b_a3b_nvfp4_lora_qad` |
| MoE MiniMax-M2.7 (~229B) | `minimax_m2` | `minimax_m2_7_nvfp4_qad` (routed experts only, W4A16) |
| Gemma-4 (`gemma4_unified`) | `gemma4_unified` | none (code path only; no released distillation corpus) |

- **What is quantized.** Attention and MLP projections (plus the Qwen3.5 delta-net linear
  projections) for dense/hybrid models; for MoE, the fused 3D routed-expert tensors are
  converted automatically and `target_modules` selects any additional linears (MiniMax trains
  experts only: `target_modules: []`). The vision tower, MTP head, delta-net conv, router,
  embeddings, norms, biases and `lm_head` stay unquantized and frozen; `apply_fake_quant`
  raises if any of them is trainable.
- **Weight-only vs W4A4.** Weight-only (W4A16) training is the default.
  `qad.quantize_activations: true` also fake-quantizes linear inputs with the serving rule —
  NVFP4 with static global scales (requires `fused_runtime_scales: true`), MXFP4 with fully
  dynamic CEIL-rule scales. Q4_K is weight-only.
- **LoRA-QAD.** `qad.lora: {enable: true, rank, alpha}` trains rank-r adapters on every
  quantized linear and quantizes the merged weight `Q(W + BA)`; the frozen base doubles as the
  teacher, so there is no second model copy. Routed experts are adapted through their fused
  3D tensors (`qad.lora.moe_hybrid_shared` picks the shared or per-expert layout). See the
  LoRA-QAD section of `AGENTS.md` for saving and export.
- **MoE routing replay.** `qad.moe.routing_replay: mask` (or `full`) makes the student use
  the teacher's top-k expert choices early in training, then anneals to free routing between
  `replay_decay_start` and `replay_decay_end` (fractions of training).
- **Gemma-4 specifics** (code path, no recipe): use `attn_implementation: sdpa` (its
  `full_attention` global layers use head_dim 512, which flash-attn 2.x rejects); the global
  layers reuse K as V (`attention_k_eq_v`), so they have only q/k and fuse as q+k; tied
  embeddings are untied into a frozen clone at load and re-tied at save; use the
  `gemma4_unified_qad.jinja` template.
- **Not supported:** MXFP4 export of routed experts (the MXFP4 exporter packs dense and
  hybrid models only), packed NVFP4 export of LoRA-adapted routed experts (merge with
  `scripts/merge_lora_bf16.py` instead), and other formats such as `q4_0`. Reject unsupported configurations explicitly; never fall back to
  ordinary fine-tuning.

## Quantization formats (`qad.quant_format`)

| Aspect | `nvfp4` | `mxfp4` | `q4_k` / `q4_k_grid_search` |
|--------|---------|---------|------|
| Lattice | E2M1, group-16, E4M3 block scale + FP32 tensor scale | E2M1, group-32, E8M0 block scale, no global scale | two-level `block_q4_K`: 256-weight super-blocks, 8×32 affine sub-blocks, fp16 `d`/`dmin` |
| `fused_runtime_scales` | **`true`** (required for export) | not used | **`false`** (raises if `true`) |
| W4A4 training | yes (static activation scales) | yes (dynamic CEIL-rule scales) | no |
| Final checkpoint extra | `input_scales.safetensors` | none | `q4k_weight.safetensors` (deploy-layout blocks packed once at save) |
| `in_features` constraint | multiple of 16 | multiple of 32 | multiple of 256 |
| Export target | vLLM `nvfp4-pack-quantized` | vLLM `mxfp4-pack-quantized` | llama.cpp GGUF |

Q4_K specifics:

- No shared/fused scales — each tensor is encoded independently, so there is no callback and
  no `input_scales.safetensors`.
- The exact Q4_K blocks are packed once, on the training device, into `q4k_weight.safetensors`;
  export splices them into a GGUF unchanged (lossless by construction). Never re-quantize at
  export — llama.cpp's Q4_K encoder is not bit-reproducible.
- Qwen3.5 linear-attn targets are GGUF-reordered on export; `configure_q4k_reorders` attaches
  the V-head permutations (`out_proj` uses column-reorder-aware fake-quant). This is automatic
  in the model loader for `q4_k` and `q4_k_grid_search`.
- `q4_k` = naive asymmetric min-max (default, cheapest); `q4_k_grid_search` = llama.cpp
  `make_qkx2_quants` weighted-LS search. Both emit identical `block_q4_K` bytes. A fused Triton
  kernel accelerates both on CUDA (`QATFACTORY_Q4K_DISABLE_TRITON=1` forces the PyTorch
  reference).

## Environment

`pip install -e .` into any Python environment that provides the dependencies in
`pyproject.toml` (torch, transformers, accelerate, safetensors, datasets, flash-attn). MoE
recipes also need `torchao` (8-bit AdamW with BF16 stochastic rounding, `low_bit_optim:
adamw8bit_sr`). `model_path` may be a local directory or a Hugging Face Hub repo id (e.g.
`Qwen/Qwen3-8B`), which `resolve_model_dir` downloads to the default HF cache on first launch.
See `AGENTS.md` (Environment) for Hopper/flash-linear-attention and Triton-cache notes.

## Workflow

1. Read the recipe, including its header comments (scope, memory, launch).
2. Validate the model checkpoint, train/eval data (disjoint), GPU availability, and W&B auth.
   Refuse a nonempty output directory unless resuming; set `CUDA_VISIBLE_DEVICES` /
   `--nproc_per_node` explicitly.
3. Data is raw conversation JSONL (`{"messages": [...]}`) with optional per-message
   `reasoning_content`; the chat template renders thinking and supervises the assistant turn.
   The released recipes distill the model's own answers to Open Perfect Blend prompts, built
   with `scripts/generate_distill_dataset.py`. Tokenization is parallel and cached under
   `cache/tokenized/<key>/` (only rank 0 builds).
4. Run a bounded smoke (small `--data.max_seq_len`, `--train.max_steps`, 2 GPUs for dense
   8B, 4 for Qwen3.5-9B) before a full run.
5. Launch, then monitor to the W&B URL, first steps, and the first eval; watch loss,
   grad-norm, OOM, NaN/Inf, tracebacks. Stop on NaN/Inf, OOM, or traceback.
6. Outputs go under `outputs/`; the final checkpoint holds BF16 latent weights (or only
   `lora_adapters.safetensors` in LoRA mode) and the full config with `qatfactory_qad`
   metadata. NVFP4 also writes `input_scales.safetensors`; Q4_K writes
   `q4k_weight.safetensors`. Then export with `/qatfactory-nvfp4-export` or
   `/qatfactory-q4k-export`.

### Large MoE runs

- **Memory.** FP32 masters plus Adam moments for 30B+ latents do not fit; the MoE recipes keep
  latents in BF16 and use `train.low_bit_optim: adamw8bit_sr`.
- **Multi-node (MiniMax-M2.7).** The static state needs 16 GPUs (two 8×B200 nodes, or four
  4-GPU GB200/GB300 nodes). Launch `torchrun --nnodes=<N> --nproc_per_node=<G>
  --rdzv_backend=c10d --rdzv_endpoint=<host>:<port>` on every node with the model and the
  tokenized cache staged at the same path. `cpu_ram_efficient_loading: true` makes only the
  main rank read the checkpoint.
- **Checkpoint preparation.** MiniMax publishes FP8 weights; dequantize to BF16 with
  `scripts/dequant_fp8_minimax.py`, then fuse the per-expert tensors into the 3D layout the
  model loads with `scripts/fuse_moe_checkpoint.py`, which turns loading into a memory-mapped
  assignment instead of a per-rank concatenation of every expert.
- **Resumable runs.** Long runs are usually split into time-limited jobs. Keep
  `train.save_only_model: false` so checkpoints carry the optimizer state (saved per rank for
  `low_bit_optim`), size `save_steps` so at least one checkpoint lands inside every job, and
  relaunch with `--train.resume_from_checkpoint <outputs/run/checkpoint-N>`. A checkpoint
  directory cut off by a job time limit has no `trainer_state.json`; resume from the previous
  one. Large MoE checkpoints are written shard by shard, so a save fits in one node's host
  memory.

## Command

Dense Qwen3-8B NVFP4 (base auto-downloads from the Hub on first launch):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=online PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=8 \
  scripts/train_llm_qat.py configs/llm/qwen3_8b_nvfp4_qad.yaml \
  --data.num_proc 32 --wandb.project qatfactory-qat
```

Swap the recipe for other model/format combinations (see the table above). Override any
config value with dotted CLI keys, e.g. `--train.learning_rate 1e-6 --train.max_steps 4000
--train.output_dir outputs/<run>`. Or run a whole train→export pipeline:
`bash examples/run_full_qwen3_8b.sh` / `run_full_qwen35.sh` (NVFP4),
`run_full_qwen3_8b_q4k.sh` / `run_full_qwen35_q4k.sh` (Q4_K).

## Recommended configuration

```yaml
data: { max_seq_len: 8192, cache_dir: cache/tokenized, num_proc: 16, truncate: true }
train:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1        # global batch 8 on 8 GPUs
  learning_rate: 1.0e-6                 # small recovery LR; raise only after stable loss
  warmup_steps: 0.01                    # float < 1 = fraction of total steps
  lr_scheduler_type: cosine
  max_grad_norm: 1.0
  bf16: true
  gradient_checkpointing: true
  fsdp: full_shard auto_wrap            # auto_wrap makes every decoder layer its own FSDP2 unit
  fsdp_config: { fsdp_version: 2 }
qad:
  quant_format: nvfp4                    # or mxfp4 / q4_k / q4_k_grid_search
  quantize_activations: false            # true = W4A4 training (nvfp4 / mxfp4)
  temperature: 1.0
  distill_weight: 1.0
  hard_label_weight: 0.0
  fused_runtime_scales: true             # nvfp4 only; must be false for q4_k
  logits_chunk_size: 2048
```

For Q4_K, set `quant_format: q4_k` (naive min-max) or `q4_k_grid_search` (weighted-LS search)
and `fused_runtime_scales: false`, and add the linear-attn leaves (`in_proj_qkv/z/a/b`,
`out_proj`) to `target_modules` for Qwen3.5.

## Verification

```bash
python -m pytest tests/quant tests/data -q
```

Sanity signals with all QAD targets enabled:

- **Dense Qwen3-8B** (36 layers): **252 quantized linears** (7 per layer: q/k/v/o +
  gate/up/down); `config.json` has `model_type: qwen3` and `qatfactory_qad` metadata with the
  matching `quant_format`. NVFP4 writes 252 keys in `input_scales.safetensors`; Q4_K writes 252
  keys in `q4k_weight.safetensors`.
- **Hybrid Qwen3.5-9B**: 248 quantized linears / ~77% of params trainable.
- **MoE**: the trainer logs the number of saved activation scales at every save — 288 for
  Qwen3-30B-A3B (attention + experts) and 124 for MiniMax-M2.7 (experts only).
- Q4_K only: `qatfactory_qad.quantizer` is `q4_k_minmax` (`quant_format: q4_k`) or
  `q4_k_grid_search`, no `input_scales.safetensors`, and every target `in_features` is a
  multiple of 256.

For any model: `eval_loss` below the initial value and decreasing; healthy grad-norm; no OOM
at the recipe's sequence length.
