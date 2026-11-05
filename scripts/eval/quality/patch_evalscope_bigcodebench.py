#!/usr/bin/env python3
"""Idempotently patch the installed evalscope BigCodeBench adapter so it ACTUALLY
runs the unittest suite when grading.

Bug (evalscope <= 1.9.1): BigCodeBenchAdapter.match_score builds the sandbox
check program as `solution + '\\n' + problem['test']`. The dataset's `test`
field only DEFINES a `unittest.TestCase` subclass; nothing invokes it, so the
script exits 0 for any syntactically-valid completion and EVERY sample grades as
'success' -> acc True (observed ~142/148 "pass" on BigCodeBench-Hard with empty
execution output). Scores are therefore meaningless (inflated toward ~100%).

Fix (upstream in evalscope 1.14.0, modelscope/evalscope#1859; older versions need this patch): append an explicit
`unittest.main(argv=['__bcb_check__'], verbosity=1)` runner so failing tests
raise a non-zero SystemExit, which the sandbox maps to status 'error' -> acc
False.

This script edits the file in the CURRENTLY-ACTIVE environment's site-packages
(found via `import evalscope`), so activate the eval env first. It is safe to run
repeatedly: it detects the sentinel `__bcb_check__` and no-ops if already patched.
A one-time backup is written next to the file as `<name>.bak`.
"""
import os
import shutil
import sys

OLD = (
    "        # Construct full test program: solution + unittest test class\n"
    "        check_program = solution + '\\n' + problem['test']\n"
)

NEW = (
    "        # Construct full test program: solution + unittest test class + an\n"
    "        # explicit runner. The dataset's `test` field only DEFINES the\n"
    "        # TestCases class; without invoking unittest the script exits 0 and\n"
    "        # every syntactically-valid sample grades as passed (observed: 142/148\n"
    '        # "pass" with empty execution output). unittest.main raises SystemExit\n'
    "        # non-zero on any test failure, which the sandbox maps to status\n"
    "        # 'error' -> acc False.\n"
    "        check_program = (\n"
    "            solution + '\\n' + problem['test'] + '\\n'\n"
    "            + 'import unittest\\n'\n"
    "            + \"unittest.main(argv=['__bcb_check__'], verbosity=1)\\n\"\n"
    "        )\n"
)

SENTINEL = "__bcb_check__"


def main() -> int:
    try:
        import evalscope
    except ImportError:
        print("ERROR: evalscope is not importable; activate the eval env first.", file=sys.stderr)
        return 2

    path = os.path.join(
        os.path.dirname(evalscope.__file__),
        "benchmarks", "bigcodebench", "bigcodebench_adapter.py",
    )
    if not os.path.isfile(path):
        print(f"ERROR: adapter not found at {path}", file=sys.stderr)
        return 2

    with open(path, "r") as f:
        src = f.read()

    if SENTINEL in src:
        print(f"[patch] already patched (sentinel present): {path}")
        return 0

    if OLD not in src:
        print(
            "ERROR: expected buggy block not found; the installed evalscope may be a\n"
            f"       different version than this patch targets. Inspect manually:\n       {path}",
            file=sys.stderr,
        )
        return 3

    backup = path + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"[patch] backup written: {backup}")

    with open(path, "w") as f:
        f.write(src.replace(OLD, NEW, 1))
    print(f"[patch] patched: {path}")
    print("[patch] BigCodeBench now runs unittest.main(); re-run group c to get valid scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
