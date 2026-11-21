#!/usr/bin/env python3
# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Thin CLI for the F2.1 generation harness.

    python scripts/run_eval.py --model B0 --split dev --backend hf \\
        --model-id meta-llama/Llama-3.2-3B-Instruct --out evaluation/results/b0_dev.jsonl

Test splits need ``DOCMIND_FT_ALLOW_TEST=1``, a pre-declared model id and an
unused (model, split) ledger line (gate I7).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import docmind_ft.vendor  # noqa: F401, E402  # installs the docmind_service alias

from docmind_ft.config import Settings  # noqa: E402
from docmind_ft.generate import (  # noqa: E402
    HFBackend,
    OllamaBackend,
    ProviderBackend,
    SPLITS,
    TEST_SPLITS,
    TestGuardError,
    check_test_guard,
    run_generation,
)


def _backend(args):
    if args.backend == "hf":
        return HFBackend(
            args.model_id or args.model,
            revision=args.revision,
            adapter_path=args.adapter,
            merged_bf16=args.merged_bf16,
        )
    if args.backend == "ollama":
        return OllamaBackend(args.model_id or args.model, host=args.host)
    return ProviderBackend()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="run id, e.g. B0, FT-main")
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--backend", required=True, choices=("hf", "ollama", "provider"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model-id", default=None, help="HF repo/path or Ollama tag")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--merged-bf16", action="store_true")
    parser.add_argument("--host", default=None, help="Ollama host URL")
    args = parser.parse_args(argv)

    if args.split in TEST_SPLITS and not Settings.from_env().docmind_ft_allow_test:
        print(
            f"refusing split {args.split!r}: DOCMIND_FT_ALLOW_TEST=1 is required",
            file=sys.stderr,
        )
        return 2
    try:
        check_test_guard(args.model, args.split)
    except TestGuardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = run_generation(
        split=args.split,
        model=args.model,
        backend=_backend(args),
        out_path=args.out,
    )
    print(
        json.dumps(
            {
                "split": summary.split,
                "model": summary.model,
                "questions": summary.questions,
                "written": summary.written,
                "skipped": summary.skipped,
                "out": str(summary.output_path),
                "sha256": summary.sha256,
                "ledger_written": summary.ledger_written,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
