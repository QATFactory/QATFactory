# Sequence-length and cross-domain generalization (paper Finding 4)

The study trains Qwen3.5-9B NVFP4 QAD on two 45K-conversation corpora, one math and one
code, each at two truncation lengths (4K and 32K tokens), with the same number of supervised
training tokens per domain. It then evaluates every checkpoint on AIME25, BigCodeBench and
LiveCodeBench-v6, and on held-out KL in both domains and at both lengths.

## 1. Answered corpus

Build the answered corpus with the generator and
`configs/data/openperfectblend_generalization_qwen3_5_9b.yaml`: Open Perfect Blend prompts from its
four math and code sources, answered by Qwen3.5-9B in thinking mode with a 32K budget, so that 32K
truncation matters. The recipe pins the teacher and prompt-set revisions and answers every row
(no prompt dedup): UltraInteract has only ~81K distinct instructions over ~272K rows, and one
answer per distinct instruction cannot fill the code/ui stratum. About 440K generations; split
them across nodes with `--output.num_workers`/`--output.worker_index` (see the recipe header).

The split builder needs each row's `source`, `orig_row`, `finish_reason` and
`usage.total_tokens`, which the generator writes. Pass every answered file you want the split to
draw from as an `--input`.

## 2. Splits

```bash
python scripts/generalization/prepare_splits.py \
    --input data/<answered corpus>.jsonl \
    --ultrainteract-parquet <openbmb/UltraInteract_sft parquet> \
    --output-dir data/generalization
python scripts/generalization/require_4k_supervision.py \
    --input data/<answered corpus>.jsonl \
    --data-dir data/generalization \
    --model-path Qwen/Qwen3.5-9B \
    --ultrainteract-parquet <openbmb/UltraInteract_sft parquet>
```

`prepare_splits.py` keeps conversations longer than 4,096 tokens, so that 32K truncation
changes what is supervised. It then draws balanced strata, ranked by a seeded hash of
`source:orig_row`:
- math: MetaMathQA, Orca-Math, UltraInteract chain-of-thought and program-of-thought, 11,250 each;
- code: Evol-CodeAlpaca and UltraInteract, 22,500 each.

It writes `{math,code}_{train,eval,eval_monitor}.jsonl` and a `manifest.json` with every
input's fingerprint. `--scale 0.03125` (for example) shrinks every stratum count proportionally for
a pilot run on a smaller corpus; the default 1.0 is the paper's split.

`require_4k_supervision.py` then replaces the code rows whose assistant turn starts after the
4K cutoff. Those rows would have no supervised token in the 4K arm. Replacements come from the
next eligible rows in the same stratum's order. Both steps are deterministic. Run them in this
order to reproduce the paper's splits.

## 3. Training

```bash
torchrun --nproc_per_node=8 scripts/train_llm_qat.py configs/llm/qwen3_5_9b_nvfp4_qad_generalization_math_4k.yaml
torchrun --nproc_per_node=8 scripts/train_llm_qat.py configs/llm/qwen3_5_9b_nvfp4_qad_generalization_math_32k.yaml
# likewise code_4k, code_32k
```

All four recipes use `qad.streamed_logits`, which computes the distillation loss from hidden
states in token chunks. 32K tokens × Qwen3.5's 248K vocabulary would not fit as
materialized logits. They also use FSDP activation checkpointing. The 4K arms train for
5,264 steps at global batch 16. Each 32K arm stops where its supervised-token count matches
its domain's 4K arm: about 336.7M tokens for math and 307.9M for code, 1,817–2,164 steps in
the paper.

## 4. Evaluation

- Held-out KL in both domains and at both lengths:
  ```bash
  python scripts/generalization/eval_transfer.py --checkpoint <run dir> --teacher Qwen/Qwen3.5-9B \
      --config <its recipe> --data-dir data/generalization --cache-dir cache/tokenized --output <metrics.json>
  ```
- AIME25, BigCodeBench and LiveCodeBench-v6 on the exported checkpoints: see
  [evaluation.md](evaluation.md). Use a 32K generation budget for this study.
