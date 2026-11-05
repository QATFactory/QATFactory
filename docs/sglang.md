# Serving QATFactory checkpoints with SGLang

QATFactory's NVFP4 exports are compressed-tensors checkpoints, and SGLang loads them unchanged, the same way vLLM
does. This page covers which exports SGLang serves, how to point the evaluation scripts at an SGLang server, and how
SGLang compares with vLLM on numerics and benchmarks.

All measurements were taken on NVIDIA GB300 (aarch64) with vLLM 0.30.0, SGLang 0.5.23, a SGLang main nightly
(`797af717`, for the MXFP4 rows) and evalscope 1.9.1.

## Support matrix

| Export | vLLM 0.30 (Blackwell) | SGLang 0.5.23 | SGLang main (`797af717`) |
|---|---|---|---|
| Dense / hybrid NVFP4 (released config, with input activations) | W4A4 | W4A4 | not measured |
| Dense / hybrid NVFP4, `input_activations` removed from `quantization_config` | W4A16 | W4A16 | not measured |
| Dense / hybrid MXFP4 | W4A4 | does not load ("No compressed-tensors compatible scheme") | weight-only via FP4 Marlin |
| MoE NVFP4, `qwen3_moe` (Qwen3-30B-A3B) | W4A4 | W4A4 (`CompressedTensorsW4A4Nvfp4MoE`) | not measured |
| MoE NVFP4, `minimax_m2` (MiniMax-M2.7) | W4A4 | W4A4 experts (`CompressedTensorsW4A4Nvfp4MoE`) | not measured |
| QARL rollouts (live NVFP4 weight sync) | yes | no | no |

## Usage

Launch a server directly:

```bash
python3 -m sglang.launch_server --model-path qatfactory/Qwen3.5-9B-NVFP4-QAD \
  --served-model-name Qwen3.5-9B-NVFP4-QAD --port 8001 --context-length 65536 --dp-size 4
```

The evaluation scripts in `scripts/eval/quality/` accept SGLang endpoints. An endpoints file takes an optional
sixth column, `engine` (`vllm` by default, or `sglang`), and one file can mix both engines:

```
# label|gpu|port|served_model_name|model_path[|engine]
original-bf16|0|8002|Qwen3.5-9B|Qwen/Qwen3.5-9B|vllm
qad-nvfp4-sglang|1|8001|Qwen3.5-9B-NVFP4-QAD|qatfactory/Qwen3.5-9B-NVFP4-QAD|sglang
```

- `serve_endpoints.sh` launches `python3 -m sglang.launch_server` for `sglang` lines with the same served name, port
  and context length. `SGLANG_MEM_FRACTION` (default 0.85) and `EXTRA_SGLANG_ARGS` play the role of `GPU_MEM_UTIL`
  and `EXTRA_VLLM_ARGS`; `ENGINE` sets the default for lines without the column.
- `run_endpoint_eval.sh` needs no change: it talks to the OpenAI-compatible API, which SGLang serves on `/v1`.
- `run_kld.sh` / `measure_kld.py` query `sglang` endpoints through SGLang's native `POST /generate`
  (`return_logprob`, `logprob_start_len: 0`, `top_logprobs_num: K`). The per-position `meta_info.input_top_logprobs`
  are the same quantity as vLLM's `prompt_logprobs`, so KL is computed identically for both engines. On the command
  line an endpoint is `label=url=served_name[=vllm|sglang]`. SGLang has no `--max-logprobs` cap.

## Validation

### Logit-level numerics

The same 32 UltraChat sequences of 2,048 tokens were scored teacher-forced by QATFactory's own fake-quant forward of
the trained latents ("sim"), by vLLM and by SGLang, and every pair was compared with the paper's truncated KL
(KL(ref || m) on the reference's top-20 tokens). The floor is the same simulation with eager instead of SDPA
attention: how far two correct implementations drift from kernel and accumulation order alone.

| model (positions) | BF16 floor | W4A16: sim–vLLM / sim–SGLang / vLLM–SGLang | W4A4 floor | W4A4: sim–vLLM / sim–SGLang / vLLM–SGLang |
|---|---|---|---|---|
| Qwen3-8B-Base NVFP4 RTN (14,329) | 0.0016–0.0017 | 0.0017 / 0.0016 / 0.0015 | 0.0273 | 0.0274 / 0.0273 / 0.0272 |
| Qwen3.5-9B NVFP4 QAD (18,423) | 0.0015–0.0018 | 0.0022 / 0.0016 / 0.0014 | 0.0508 | 0.0500 / 0.0537 / 0.0504 |

Every engine pair sits at the floor of its precision. The W4A4 floor is larger because 4-bit activation rounding
amplifies tiny upstream differences, in every engine. KL(BF16 || quantized) for the Qwen3.5-9B QAD checkpoint is
0.088 on SGLang and 0.093 on vLLM served W4A4, and 0.053 and 0.050 served W4A16.

### QAD vs RTN on each engine

Same-engine KL(BF16 || quantized), 32 sequences of 2,048 tokens per corpus, Qwen3.5-9B NVFP4 served W4A4:

| corpus | vLLM RTN | vLLM QAD | SGLang RTN | SGLang QAD |
|---|---|---|---|---|
| UltraChat | 0.1042 | 0.0930 | 0.1019 | 0.0876 |
| Pile | 0.0486 | 0.0383 | 0.0477 | 0.0401 |
| WikiText | 0.0757 | 0.0621 | 0.0753 | 0.0622 |
| mean | 0.0762 | 0.0645 (−15%) | 0.0750 | 0.0633 (−16%) |

QAD's gain over round-to-nearest carries over to SGLang.

### Benchmarks, three runs per engine

`qatfactory/Qwen3.5-9B-NVFP4-QAD` served W4A4 on one 4-GPU node (data parallel 4), same export and evalscope
harness for both engines, sampling temperature 1.0, top_p 0.95, top_k 20, up to 32,768 new tokens. Each run samples
every benchmark once (AIME25: 5 samples per problem); BigCodeBench (BCB) is graded separately and is not part of the
7-benchmark average.

| run | GPQA-D | MMLU-Pro | MMMLU | WinoGrande | HellaSwag | AIME25 | LCB | 7-avg | BCB |
|---|---|---|---|---|---|---|---|---|---|
| SGLang, run 1 | 77.3 | 71.4 | 87.4 | 88.0 | 69.3 | 45.3 | 78.3 | 73.86 | 31.7 |
| SGLang, run 2 | 77.8 | 72.5 | 86.1 | 89.0 | 73.3 | 46.7 | 77.7 | 74.73 | 36.0 |
| SGLang, run 3 | 76.3 | 73.1 | 85.0 | 88.7 | 69.3 | 45.3 | 80.0 | 73.96 | 32.0 |
| **SGLang mean** | 77.1 | 72.3 | 86.2 | 88.6 | 70.6 | 45.8 | 78.7 | **74.18** | **33.2** |
| vLLM, run 1 | 76.3 | 73.1 | 84.0 | 90.3 | 71.7 | 45.3 | 80.0 | 74.39 | 35.3 |
| vLLM, run 2 | 78.3 | 73.8 | 86.4 | 88.7 | 69.0 | 46.0 | 77.3 | 74.21 | 32.7 |
| vLLM, run 3 | 74.8 | 73.8 | 84.7 | 90.3 | 67.7 | 46.7 | 79.3 | 73.90 | 34.3 |
| **vLLM mean** | 76.5 | 73.6 | 85.0 | 89.8 | 69.5 | 46.0 | 78.9 | **74.17** | **34.1** |
| paper (vLLM, one run) | 77.8 | 73.5 | 82.7 | 89.3 | 72.0 | 43.3 | 83.0 | 74.5 | 29.7 |

The engine means differ by 0.01 points; single runs of one engine span up to 0.9 points (SGLang 73.86–74.73, vLLM
73.90–74.39), and BCB varies by about 4 points between runs. A single run of this suite carries roughly ±1 point of
sampling noise on the 7-benchmark average (GPQA-D has 198 questions, the other multiple-choice sets and LCB about 300,
AIME25 30 × 5), so compare engines or checkpoints on the mean of several runs.

### MoE: Qwen3-30B-A3B and MiniMax-M2.7

A Qwen3-30B-A3B NVFP4 QAD checkpoint (step 768 of the released recipe's 2,000), exported with
`scripts/export_nvfp4_vllm.py`, served on one GPU per engine. SGLang 0.5.23 loads it with
`CompressedTensorsW4A4Nvfp4MoE` (W4A4 experts, 18 s, 16.9 GB); vLLM 0.30 uses the FlashInfer TRTLLM NVFP4 MoE
backend, also W4A4.

KL(BF16 || quantized), one run of `run_kld.sh` with a vLLM BF16 teacher and the same export served by both engines
(32 sequences of 2,048 tokens per corpus):

| served | UltraChat | Pile | WikiText | mean |
|---|---|---|---|---|
| vLLM | 0.0578 | 0.0448 | 0.0617 | 0.0547 |
| SGLang | 0.0559 | 0.0420 | 0.0585 | 0.0521 |

Benchmarks, one run per engine (same harness and sampling as above, 40,960-token context):

| engine | GPQA-D | AIME25 | BCB | LCB | 4-avg |
|---|---|---|---|---|---|
| vLLM | 63.6 | 70.7 | 44.7 | 92.7 | 67.9 |
| SGLang | 61.1 | 71.3 | 44.7 | 92.7 | 67.5 |

The engines agree within single-run noise (GPQA-D has 198 questions, about ±3.4 points), and SGLang's KL is slightly
lower, as on the dense Qwen3.5-9B.

MiniMax-M2.7, a QAD checkpoint (step 270 of 2,000) of the released recipe, which quantizes the routed experts only.
SGLang 0.5.23 loads the experts with `CompressedTensorsW4A4Nvfp4MoE` and keeps the other layers in BF16
("Falling back to UnquantizedLinearMethod", as exported); one replica fits on one GB300 (125 GB, loaded in 128 s).
KL against the FP8 original served by vLLM, same export on both engines:

| served | UltraChat | Pile | WikiText | mean |
|---|---|---|---|---|
| vLLM (1 GPU) | 0.0634 | 0.0749 | 0.1492 | 0.0958 |
| SGLang (1 GPU) | 0.0553 | 0.0665 | 0.1394 | 0.0871 |

| engine | GPQA-D | AIME25 | BCB | LCB | 4-avg |
|---|---|---|---|---|---|
| vLLM (TP 2 × DP 2) | 76.3 | 83.3 | 26.0 | 87.0 | 68.2 |
| SGLang (DP 4) | 77.8 | 83.3 | 21.7 | 86.0 | 67.2 |

The averages differ by 1 point, almost all of it BigCodeBench, which varies by about 4 points between runs of the
same checkpoint (see the repeats above).

### MXFP4 on SGLang main

SGLang 0.5.23 has no compressed-tensors scheme for dense MXFP4. SGLang main loads both the W4A4 and the W4A16
config of the Qwen3.5-9B MXFP4 QAD checkpoint and serves them weight-only through FP4 Marlin ("MXFP4 w4a4 has no
native kernel; serving this checkpoint weight-only via FP4 Marlin instead"). Its logits match the weight-only
simulation at the kernel floor (KL 0.00107 vs a floor of 0.00105), and KL(BF16 || served) is 0.074, against 0.192 on
vLLM, which serves the checkpoint W4A4 on Blackwell. A checkpoint trained weight-only is therefore served the way it
was trained only on SGLang main.

## Notes

- **Served precision matters more than the engine.** Serving an NVFP4 checkpoint W4A4 instead of weight-only raises
  its KL to the reference model 1.7–1.9× (Qwen3.5-9B QAD on SGLang and vLLM; MiniMax-M2.7 on vLLM), far more than
  the engines differ from each other. Report which mode a number was measured with.
- **QARL** still needs vLLM: rollouts sync the policy's packed NVFP4 weights into a live vLLM engine
  (`qatfactory/rl/vllm_nvfp4_sync.py`), which has no SGLang counterpart yet.
