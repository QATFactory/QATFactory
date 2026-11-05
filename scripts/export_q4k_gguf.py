#!/usr/bin/env python3
"""Export a Q4_K QAD Hugging Face checkpoint to a GGUF with lossless Q4_K target weights.

The exporter builds the GGUF container with the trusted ``convert_hf_to_gguf.py`` (BF16
out-type) and splices in the Q4_K blocks that were packed at training-checkpoint save time,
so the deployed GGUF dequantizes value-for-value to the trained student.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qatfactory.quant.export import export_q4k_gguf_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Q4_K QATFactory QAD checkpoint to GGUF.")
    parser.add_argument("--source", required=True, help="Final Q4_K QAD Hugging Face checkpoint directory.")
    parser.add_argument("--output", required=True, help="Fresh output GGUF path.")
    parser.add_argument(
        "--convert-script",
        required=True,
        help="Path to llama.cpp convert_hf_to_gguf.py used to build the GGUF container.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python interpreter to run convert_hf_to_gguf.py (defaults to the current one).",
    )
    parser.add_argument("--keep-scaffold", action="store_true", help="Keep the intermediate BF16 GGUF.")
    parser.add_argument(
        "--embed-output-dtype",
        choices=("q8_0", "bf16"),
        default="q8_0",
        help="Storage dtype for the frozen token_embd/output tensors (default: q8_0).",
    )
    args = parser.parse_args()

    summary = export_q4k_gguf_checkpoint(
        source=args.source,
        output=args.output,
        convert_script=args.convert_script,
        python_exe=args.python_exe,
        keep_scaffold=args.keep_scaffold,
        embed_output_dtype=args.embed_output_dtype,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
