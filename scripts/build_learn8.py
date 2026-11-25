"""Build the F3.3 learning-test set: 8 short examples chosen by id hash.

Pool: shortest quartile (length <= 2,243 tokens, build_example). Within the
pool, pick 2 abstention targets (meta.case == "U") and 6 answerable ones, each
group sorted by sha256(question_id) so the choice is deterministic.

Writes data/learn8.jsonl and evaluation/splits/learn8_ids.json (Class B
deviation: new file in the frozen splits dir; no frozen file is edited).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SHORT_QUARTILE_CUTOFF = 2243


def main() -> int:
    import transformers

    from docmind_ft.train import build_example, load_config, run_i2_check

    cfg = load_config(REPO_ROOT / "configs" / "train_main.yaml")
    max_length = cfg["train"]["max_length"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg["base_model"], revision=cfg["revision"], local_files_only=True
    )

    rows = [
        json.loads(line) for line in (REPO_ROOT / "data" / "train.jsonl").open(encoding="utf-8")
    ]
    scored = []
    for row in rows:
        input_ids, _, length_ok = build_example(
            tokenizer, row["prompt"], row["completion"], max_length
        )
        if length_ok and len(input_ids) <= SHORT_QUARTILE_CUTOFF:
            digest = hashlib.sha256(row["meta"]["question_id"].encode("utf-8")).hexdigest()
            scored.append((digest, len(input_ids), row))

    abstention = sorted(item for item in scored if item[2]["meta"]["case"] == "U")
    answerable = sorted(item for item in scored if item[2]["meta"]["case"] != "U")
    print(f"short-quartile pool={len(scored)} abstention={len(abstention)}")
    if len(abstention) < 2:
        print("FATAL: fewer than 2 abstention targets in the short quartile")
        return 1
    if len(answerable) < 6:
        print("FATAL: fewer than 6 answerable targets in the short quartile")
        return 1

    chosen = abstention[:2] + answerable[:6]
    chosen.sort(key=lambda item: item[0])

    out_path = REPO_ROOT / "data" / "learn8.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for _, _, row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    ids_path = REPO_ROOT / "evaluation" / "splits" / "learn8_ids.json"
    ids = [
        {
            "question_id": row["meta"]["question_id"],
            "paper_id": row["meta"]["paper_id"],
            "length": length,
            "case": row["meta"]["case"],
            "answer_type": row["meta"]["answer_type"],
            "id_sha256": digest,
        }
        for digest, length, row in chosen
    ]
    with ids_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(ids, handle, indent=2, ensure_ascii=False)
    for item in ids:
        print(f"  {item['case']} len={item['length']} qid={item['question_id'][:12]}")
    print(f"wrote {out_path.name} ({len(chosen)} rows) and {ids_path}")

    result = run_i2_check(tokenizer, out_path, max_length)
    print(f"I2 learn8: {result}")
    return 0 if result["exit_code"] == 0 and result["rows_valid"] == 8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
