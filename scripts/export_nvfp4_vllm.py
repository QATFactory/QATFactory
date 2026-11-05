#!/usr/bin/env python3
"""Pack a fused-scale QAD Hugging Face checkpoint into vLLM compressed-tensors NVFP4."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qatfactory.quant.export import export_nvfp4_vllm_checkpoint  # noqa: E402
from qatfactory.utils import resolve_model_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a fused-scale QATFactory QAD checkpoint to vLLM compressed-tensors NVFP4."
    )
    parser.add_argument("--source", required=True, help="Final QAD Hugging Face checkpoint directory.")
    parser.add_argument("--model-assets", required=True, help="Original model/tokenizer asset directory.")
    parser.add_argument("--output", required=True, help="Fresh output directory.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-size", type=int, default=5_000_000_000)
    parser.add_argument("--reference", default=None, help="Optional vLLM NVFP4 checkpoint for schema validation.")
    args = parser.parse_args()

    summary = export_nvfp4_vllm_checkpoint(
        source=args.source,
        model_assets=resolve_model_dir(args.model_assets),
        output=args.output,
        device=args.device,
        shard_size=args.shard_size,
        reference=args.reference,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
