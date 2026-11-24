#!/usr/bin/env python3
"""I2/I3 gate verification scripts.

Uses the shared ``build_example`` from ``docmind_ft.train`` so the I3 gate
measures the real training pipeline tokenization, not a parallel fiction.

I2: token-length statistics of data/train.jsonl; over-length examples dropped
    and counted; at least 0.99 of the file must remain valid.

I3 on 20 random real examples (seeded, seed 20261207):
    all prompt tokens masked (label -100);
    at least 95% of completion tokens carry a label;
    BOS token appears exactly once at the start of each sequence;
    the chat template ends the prompt exactly where the completion begins.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import the shared example-to-tokens logic from the trainer.
# This is the one shared code path: I2 and I3 both use the exact same
# tokenization that TRL's SFTTrainer will use during training.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_JSONL = DATA_DIR / "train.jsonl"
SEED = 20261207


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i2", action="store_true", help="Run I2 token-length statistics check")
    parser.add_argument("--i3", action="store_true", help="Run I3 mask test check")
    parser.add_argument(
        "--max-length",
        type=int,
        default=3072,
        help="Maximum sequence length for I2 check (default 3072)",
    )
    parser.add_argument(
        "--min-completion-frac",
        type=float,
        default=0.95,
        help="Minimum completion fraction for I3 (default 0.95)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed for I3 (default 20261207)"
    )
    args = parser.parse_args()

    if not args.i2 and not args.i3:
        parser.print_help()
        return 2

    # Load tokenizer (same one the trainer would use)
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        local_files_only=True,
    )

    exit_code = 0

    if args.i2:
        print("=== I2 Token-Length Statistics ===")
        from docmind_ft.train import run_i2_check

        stats = run_i2_check(tokenizer, TRAIN_JSONL, max_length=args.max_length)
        print(f"  Rows total:   {stats['rows_total']}")
        print(f"  Rows valid:   {stats['rows_valid']}")
        print(f"  Rows dropped: {stats['rows_dropped']}")
        print(f"  Fraction valid: {stats['frac_valid']}")
        if stats["exit_code"] == 0:
            print("  I2: PASS (>= 0.99 fraction valid)")
        else:
            print("  I2: FAIL (< 0.99 fraction valid)")
            exit_code = 1

    if args.i3:
        print(f"\n=== I3 Mask Test (seed={args.seed}, n=20) ===")
        from docmind_ft.train import run_i3_mask_check

        i3_result = run_i3_mask_check(
            tokenizer,
            TRAIN_JSONL,
            n_examples=20,
            seed=args.seed,
            max_length=args.max_length,
            min_completion_frac=args.min_completion_frac,
        )
        print(f"  Tested examples: {i3_result['n_tested']}")
        print(f"  Passed: {i3_result['passed']}")
        print(f"  Failed: {i3_result['failed']}")
        print(f"  I3: {'PASS' if i3_result['ok'] else 'FAIL'}")

        if not i3_result["ok"]:
            exit_code = 1

        # Print per-example details (first 5)
        for r in i3_result["per_example"][:5]:
            print(
                f"  Example {r['index']}: ok={r['ok']} "
                f"frac={r['completion_trained_frac']} bos_first={r['bos_first']}"
                f" bos_once={r['bos_once']}"
                f" len={r['input_ids_len']}"
            )
        if len(i3_result["per_example"]) > 5:
            print(f"  ... and {len(i3_result['per_example']) - 5} more examples")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
