---
name: qatfactory-q4k-export
description: >-
  Exports a QATFactory Q4_K QAD Hugging Face checkpoint to a llama.cpp GGUF whose target
  linears are losslessly Q4_K, by building the container with convert_hf_to_gguf.py and
  splicing in the Q4_K blocks packed at training-checkpoint save time. Use when the user asks
  to export, pack, convert, or verify a Q4_K QAD checkpoint to GGUF for llama.cpp in this repo.
---

# QATFactory Q4_K GGUF Export

Builds the GGUF container / metadata / tokenizer / architecture reorders with the trusted
`convert_hf_to_gguf.py` (`--outtype bf16`), then **splices** the pre-computed `block_q4_K`
blocks (`q4k_weight.safetensors`, packed once on the training device) into the target tensors.
Because ggml's `dequantize_row_q4_K` is bit-identical to `qatfactory.quant.formats.q4_k.
dequantize_q4_k`, the deployed GGUF dequantizes value-for-value to the trained student — lossless
by construction. **Never re-run llama.cpp's Q4_K encoder** for the targets: its grid search is
not bit-reproducible across float backends.

| Component | Location |
|-----------|----------|
| Exporter (convert + splice + verify) | `qatfactory/quant/export/q4_k_gguf.py` |
| CLI | `scripts/export_q4k_gguf.py` |
| Q4_K core (decode/reorders) | `qatfactory/quant/formats/q4_k.py` |

## When to use

Only for runs trained with a Q4_K format — `qad.quant_format: q4_k` (naive min-max) or
`q4_k_grid_search` — and `fused_runtime_scales: false`. The exporter rejects checkpoints whose
`qatfactory_qad` metadata is missing, is not a Q4_K run (`quantizer` not in
`{q4_k_minmax, q4_k_grid_search, q4_k_ref}`), or that lack `q4k_weight.safetensors`. Both Q4_K
variants emit identical `block_q4_K` bytes, so the export path is the same. For NVFP4/vLLM use
`/qatfactory-nvfp4-export` instead.

## Inputs

- Final Q4_K QAD Hugging Face checkpoint (the training `outputs/<run>` dir): BF16 target
  weights, full config with `qatfactory_qad` metadata (`quant_format: q4_k` or
  `q4_k_grid_search`), and `q4k_weight.safetensors` (deploy-layout Q4_K blocks, keyed by
  on-disk target module name).
- A llama.cpp checkout that provides `convert_hf_to_gguf.py` (builds the GGUF container and
  applies architecture weight reorders; not vendored). The example launchers read it from
  `$LLAMA_CPP`.
- The `gguf` package (ggml/llama.cpp, MIT): `pip install -e .[q4k]` (or `pip install gguf`).
- A fresh output `.gguf` path (the exporter refuses to overwrite) on a **large partition**.

## Disk / scratch requirements

The exporter first writes a full **BF16 scaffold GGUF** (the whole model unquantized) into a
temp dir *next to `--output`*, then splices the Q4_K blocks into the final file. So peak disk is
roughly **model_bf16 + final_q4k ≈ ~2.5× the final GGUF** (e.g. Qwen3.5-9B: ~18 GB scaffold +
~8 GB output). Point `--output` at a big volume (e.g. `/data`), **not `/tmp`** — a small/full
`/tmp` fails with `OSError: Not enough free space`. The scaffold is deleted on success (keep it
with `--keep-scaffold`).

## Command

```bash
python scripts/export_q4k_gguf.py \
  --source outputs/<run> \
  --output exports/<run>.gguf \
  --convert-script "$LLAMA_CPP"/convert_hf_to_gguf.py
```

Options: `--python-exe` (interpreter to run the converter; defaults to the current one — must
have `gguf`, `torch`, `transformers`, `safetensors`), `--keep-scaffold` (retain the
intermediate BF16 GGUF), `--embed-output-dtype {q8_0,bf16}` (storage of the frozen
`token_embd`/`output` tensors; default `q8_0`). Or run the full train→export pipeline:
`bash examples/run_full_qwen3_8b_q4k.sh` (dense) / `examples/run_full_qwen35_q4k.sh` (hybrid).

## What it does

1. Validate `qatfactory_qad` metadata and load `q4k_weight.safetensors`.
2. Map each on-disk target module to its GGUF tensor name (dense `attn_*`/`ffn_*`; Qwen3.5
   linear-attn `attn_qkv`/`attn_gate`/`ssm_alpha`/`ssm_beta`/`ssm_out`).
3. Run `convert_hf_to_gguf.py --outtype bf16` to build the scaffold GGUF (deploy layout).
4. Splice: overwrite each target tensor's data with its Q4_K blocks (`raw_dtype=Q4_K`); copy
   all KV metadata and non-target tensors unchanged (norms F32). The frozen `token_embd`/`output`
   vocab tensors default to a **Q8_0** re-encode from the BF16 scaffold (deterministic RTN;
   `--embed-output-dtype bf16` keeps them BF16). These are not QAD-trained, so re-encoding them
   carries no bit-exactness impact on the trained student.
5. Verify: assert each spliced tensor dequantizes bit-exactly to our decoder, and is *close* to
   the scaffold's (deploy-layout) BF16 tensor (catches any name/reorder mismatch).

## Verification

```bash
python -m pytest tests/quant tests/data -q
```

Expected on a Q4_K checkpoint:

- Only the QAD target linears become `Q4_K`; `token_embd`/`output` are `Q8_0` by default
  (or BF16 with `--embed-output-dtype bf16`) and norms F32.
- The exporter prints a JSON summary with `quantized_linears` (dense Qwen3: 7 per layer; Qwen3.5:
  248), `lossless_verified: true`, and `verified_tensors`; the logs also carry a
  `Q4_K lossless verify PASSED: N/N ...` line and the max scaffold-closeness ratio.

## Serving & coherence check (llama.cpp CUDA)

Prefer **`llama-server` + `curl`**, not `llama-cli`: on some builds `llama-cli` ignores
`-no-cnv`/`-n`, drops into an interactive loop, and floods stdout. Load with all layers offloaded:

```bash
llama-server -m exports/<run>.gguf -ngl 99 --host 127.0.0.1 --port 8082 -c 4096 &
curl -s http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"What is 17 * 23?"}],
  "temperature":0, "max_tokens":2048 }'
```

These are **reasoning (thinking) models**: budget a **high `max_tokens` (2k–4k)** or the reply
truncates mid-`<think>` and `content` comes back empty (the reasoning is in `reasoning_content`).
For a quick, concise check, disable thinking:
`"chat_template_kwargs": {"enable_thinking": false}`.

Notes:

- **Lossless guarantee** holds only when the blocks in `q4k_weight.safetensors` are carried
  through unchanged — do not requantize the targets. The closeness check bounds a wrong
  name/reorder mapping, not a re-encode drift.
- **Qwen3.5 reorders** are already baked into the packed blocks at save time (row perms for
  `attn_qkv` V-rows / `attn_gate` / `ssm_alpha`/`ssm_beta`; column reorder for `ssm_out`), so
  the splice is a straight overwrite. If the scaffold tensor names or reorders diverge from a
  newer llama.cpp, the closeness check fails loudly rather than shipping a broken checkpoint.
