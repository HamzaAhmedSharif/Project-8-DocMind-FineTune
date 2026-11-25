"""Build the F3.2 probe sets from data/train.jsonl by token length.

Selection uses build_example (the real trainer tokenization path):
  probe_median6: 6 rows closest to the median length (ties by question_id)
  probe_short6:  the 6 shortest rows
  probe_long6:   the 6 longest rows that still pass I2 (<= max_length)

Writes data/probe_<set>6.jsonl, reports/F3-probe-ids.json, then verifies I2
on each written file (all rows length_ok).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    import transformers

    from docmind_ft.train import build_example, load_config, run_i2_check

    cfg = load_config(REPO_ROOT / "configs" / "train_main.yaml")
    max_length = cfg["train"]["max_length"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg["base_model"], revision=cfg["revision"], local_files_only=True
    )

    data_path = REPO_ROOT / "data" / "train.jsonl"
    rows = [json.loads(line) for line in data_path.open(encoding="utf-8")]
    scored = []
    for row in rows:
        input_ids, _, length_ok = build_example(
            tokenizer, row["prompt"], row["completion"], max_length
        )
        scored.append((len(input_ids), length_ok, row))

    valid = [item for item in scored if item[1]]
    lengths = sorted(item[0] for item in valid)
    median = statistics.median(lengths)
    print(f"rows={len(rows)} valid={len(valid)} dropped={len(rows) - len(valid)}")
    print(f"median={median:.0f} band={0.95 * median:.0f}-{1.05 * median:.0f}")

    def key_qid(item):
        return item[2]["meta"]["question_id"]

    selections = {
        "median": sorted(valid, key=lambda item: (abs(item[0] - median), key_qid(item)))[:6],
        "short": sorted(valid, key=lambda item: (item[0], key_qid(item)))[:6],
        "long": sorted(valid, key=lambda item: (-item[0], key_qid(item)))[:6],
    }

    report = {
        "rows_total": len(rows),
        "rows_valid": len(valid),
        "median_length": median,
        "band_5pct": [round(0.95 * median), round(1.05 * median)],
        "sets": {},
    }
    for name, chosen in selections.items():
        out_path = REPO_ROOT / "data" / f"probe_{name}6.jsonl"
        with out_path.open("w", encoding="utf-8", newline="\n") as handle:
            for _, _, row in chosen:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["sets"][name] = [
            {
                "question_id": row["meta"]["question_id"],
                "paper_id": row["meta"]["paper_id"],
                "length": length,
                "case": row["meta"].get("case"),
                "answer_type": row["meta"].get("answer_type"),
            }
            for length, _, row in chosen
        ]
        print(f"{name}: lengths={[length for length, _, _ in chosen]} -> {out_path.name}")

    ids_path = REPO_ROOT / "reports" / "F3-probe-ids.json"
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    with ids_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"ids written to {ids_path}")

    failed = False
    for name in selections:
        result = run_i2_check(tokenizer, REPO_ROOT / "data" / f"probe_{name}6.jsonl", max_length)
        print(f"I2 {name}: {result}")
        failed = failed or result["exit_code"] != 0 or result["rows_valid"] != 6
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
