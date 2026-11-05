---
name: qatfactory-nvfp4-export
description: >-
  Exports a QATFactory QAD (or QARL) Hugging Face checkpoint directly to the
  compressed-tensors nvfp4-pack-quantized or mxfp4-pack-quantized layout consumed by vLLM,
  without NVIDIA ModelOpt — dense, hybrid and MoE models, full-parameter or LoRA runs. Use when
  the user asks to pack, export, validate, or convert a QAD checkpoint to native NVFP4 or MXFP4
  for vLLM in this repo.
---

# QATFactory Native NVFP4 Export

Packs trained BF16 latent weights + recorded training input scales into vLLM's
`nvfp4-pack-quantized` compressed-tensors layout. Uses QATFactory's own max-based quantizer
(`qatfactory/quant/formats/nvfp4.py`); it does not import ModelOpt, llm-compressor, or the
compressed-tensors Python APIs.

| Component | Location |
|-----------|----------|
| Exporter | `qatfactory/quant/export/nvfp4_vllm.py` |
| CLI | `scripts/export_nvfp4_vllm.py` |
| MXFP4 exporter / CLI | `qatfactory/quant/export/mxfp4_vllm.py` / `scripts/export_mxfp4_vllm.py` |
| LoRA merge to BF16 (alternative to packing) | `scripts/merge_lora_bf16.py` |

## When to use

NVFP4: only for runs trained with `qad.fused_runtime_scales: true`. vLLM fuses QKV, gate/up, and
Qwen3.5 delta-net input projections under shared tensor scales, so independent-scale training
does not represent the deployed lattice. The exporter rejects checkpoints missing the
`qatfactory_qad` metadata or complete training-time input scales. MXFP4 runs have no such
requirement (see the MXFP4 section).

Supported `model_type`s: `qwen3`, `llama` (DeepSeek-R1-Distill-Llama), `qwen3_5`,
`gemma4_unified` (Gemma-4), and the MoE types `qwen3_moe` (Qwen3-30B-A3B) and `minimax_m2`
(MiniMax-M2.7). For MoE the fused 3D routed-expert tensors are packed per expert under the
serving names vLLM loads (`mlp.experts.<e>.{gate,up,down}_proj` for `qwen3_moe`;
`block_sparse_moe.experts.<e>.w1/w3/w2` for `minimax_m2`, whose router keys move under
`block_sparse_moe` too); experts-only runs keep attention in BF16. For Gemma-4 the
exporter targets the text backbone (`sliding_attention` → q/k/v/o, `full_attention` → q/k/o
for the k_eq_v global layers, plus gate/up/down), carries the frozen vision/audio embedders in
BF16, ignores them (`embed_vision`/`embed_audio`/`vision_embedder`), and — because Gemma-4
ties `lm_head` to the input embedding — keeps `tie_word_embeddings: True` and drops the
redundant `lm_head.weight`.

## Inputs

- Final QAD Hugging Face checkpoint (the training `outputs/<run>` dir): BF16 target weights,
  full config with `qatfactory_qad` metadata, and `input_scales.safetensors`. A QARL output has
  the same layout. A LoRA-QAD run saves only `lora_adapters.safetensors`; pass the run
  directory as `--source` and the exporter reads the frozen base weights from
  `--model-assets` and packs the merged weights.
- Original model assets — a local directory or a Hugging Face Hub repo id, e.g. `Qwen/Qwen3-8B`
  (tokenizer, plus any frozen weights such as a vision tower carried into the serving checkpoint).
- Fresh output directory (the exporter refuses to overwrite).
- Optional known-good vLLM NVFP4 checkpoint for schema comparison (`--reference`).

## Command

Dense Qwen3-8B (`--model-assets` accepts a local dir or a Hub repo id):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_nvfp4_vllm.py \
  --source outputs/<run> \
  --model-assets Qwen/Qwen3-8B \
  --output exports/<run>-vllm-nvfp4 \
  --device cuda:0 \
  --reference <existing vLLM NVFP4 checkpoint of the same arch>   # optional schema check
```

Hybrid Qwen3.5-9B: `--model-assets Qwen/Qwen3.5-9B` (a Hub id or a local snapshot dir).
Qwen3-30B-A3B: `--model-assets Qwen/Qwen3-30B-A3B`. MiniMax-M2.7: point `--model-assets` at the
BF16 copy the run trained from (MiniMax publishes FP8 only). The MoE exports stream tensor by
tensor; MiniMax-M2.7 takes about 70 minutes on one GPU and needs ~130 GB of output space.

LoRA-adapted routed experts cannot be packed: merge them with
`python scripts/merge_lora_bf16.py <base> <run>/lora_adapters.safetensors <out> --released-layout`
and serve the BF16 result (its target weights already sit on the NVFP4 grid).

Exported tensor names (compressed-tensors `nvfp4-pack-quantized`):

- `weight_packed` — packed E2M1, `torch.uint8`
- `weight_scale` — per-group E4M3, `torch.float8_e4m3fn`
- `weight_global_scale` — reciprocal tensor scale, FP32 shape `(1,)`
- `input_global_scale` — reciprocal training running-max scale, FP32 shape `(1,)`

The frozen vision tower is carried in BF16; the MTP head is dropped (vLLM does not use it);
embeddings, norms, and `lm_head` stay BF16; KV-cache quantization is disabled.

## Verification

```bash
python -m pytest tests/quant tests/data -q
```

With every QAD target enabled, expect:

- **Dense Qwen3-8B** (36 layers): **252 quantized linears**; 0 carried tensors (the trained
  checkpoint already holds every weight); **1155 total tensors**; **~6.4 GB**. QKV and gate/up
  families share both global scales (`o_proj`/`down_proj` are singletons). Passes `--reference`
  against a ModelOpt-produced Qwen3-8B NVFP4 vLLM checkpoint.
- **Hybrid Qwen3.5-9B**: 248 quantized linears; 333 carried vision tensors; **1504 total
  tensors**; **~8.87 GB** in two default-size shards. QKV, gate/up, and delta-net input families
  share both global scales.
- **MiniMax-M2.7** (experts-only recipe): **47,616 quantized expert linears** (62 layers ×
  256 experts × 3), 0 other quantized linears, 623 BF16 tensors (attention, router, norms,
  embeddings), ~134 GB in 27 shards; `config.json` declares `model_type: minimax_m2` and a
  `block_sparse_moe` expert target pattern.
- **Gemma-4-12B** (`gemma4_unified`, 48 layers): **328 quantized linears** (40 sliding layers
  × 7 + 8 global k_eq_v layers × 6); `config.json` keeps `tie_word_embeddings: True` with **no
  `lm_head`** tensor and `embed_tokens` present; `ignore` lists `embed_vision`/`embed_audio`/
  `vision_embedder`. On the global layers q/k share one scale (no `v_proj`); gate/up share;
  o_proj/down_proj are singletons. Validated end-to-end in vLLM.

For any model:

- packed dequantization is bit-equal to the exporter's NVFP4 reconstruction (asserted)
- `config.json` declares `quant_method: compressed-tensors` and `format: nvfp4-pack-quantized`
- no `weight_scale_2`, `input_scale`, `k_scale`, or `v_scale` tensors

## Serving precision

The NVFP4 export records the training input scales and declares `input_activations` in its
`quantization_config`, so vLLM on Blackwell serves it **W4A4** (FP4 weights and activations).
GPUs without FP4 tensor cores (e.g. H100) fall back to vLLM's **weight-only** Marlin FP4
kernels. The two modes differ: measured with vLLM 0.30 on GB300, KL to the reference model
served W4A4 is roughly twice the weight-only KL (Qwen3.5-9B QAD 0.093 vs 0.050; MiniMax-M2.7
RTN 0.119 vs 0.062), so report which mode a number was measured with. To serve weight-only on Blackwell,
use a copy whose `config.json` sets every config group's `input_activations` to `null` (for
`qwen3_moe` also drop the non-expert `*.input_global_scale` tensors, which vLLM's weight-only
attention path rejects).

H100 can build and validate the checkpoint; native W4A4 inference requires Blackwell.

## MXFP4

Runs trained with `qad.quant_format: mxfp4` export with `scripts/export_mxfp4_vllm.py`
(same `--source` / `--model-assets` / `--output` / `--reference` arguments). It re-quantizes the
BF16 latents with the training EVEN-rule encoder into `mxfp4-pack-quantized` tensors
(`weight_packed` uint8 `[N, K/2]`, `weight_scale` biased-exponent uint8 `[N, K/32]`). MXFP4 has
no global or activation scales, so there are no fused families and no `input_global_scale`;
whether vLLM serves W4A4 or W4A16 is decided by the engine and hardware at load time. The MXFP4
exporter handles dense and hybrid models; it does not pack routed experts.
