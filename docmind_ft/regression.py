#!/usr/bin/env python3
# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""F5.5: ARC-Easy regression (HY5, second part).

Zero-shot multiple choice: for each question, compare the log-likelihood of
each answer option as a continuation of the question and pick the highest.
Seeded sample of 200 test questions (seed 20261207). Runs for the base model
(B0) and the final adapter (FT-main). Reports the accuracy difference with a
simple bootstrap interval over questions.

Dataset: allenai/ai2_arc, config ARC-Easy. If the dataset cannot be
downloaded (script-based loader, network), try the parquet revision, then a
mirror for that command only.

Writes evaluation/results/regression.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import docmind_ft.vendor  # noqa: F401, E402  # installs the docmind_service alias

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
ADAPTER = REPO_ROOT / "runs" / "ft-main"
OUT_PATH = REPO_ROOT / "evaluation" / "results" / "regression.json"
SEED = 20261207
N_SAMPLE = 200
BOOTSTRAP_RESAMPLES = 10000


def load_arc_easy():
    """Load ARC-Easy test split; try hub, then parquet revision, then mirror."""
    from datasets import load_dataset

    last_error: Exception | None = None
    attempts = [
        ("hub", lambda: load_dataset("allenai/ai2_arc", "ARC-Easy")),
        (
            "parquet",
            lambda: load_dataset("allenai/ai2_arc", "ARC-Easy", revision="refs/convert/parquet"),
        ),
        (
            "mirror-hf-mirror",
            lambda: load_dataset(
                "allenai/ai2_arc",
                "ARC-Easy",
                download_config=None,
            ),
        ),
    ]
    for name, fn in attempts:
        try:
            ds = fn()
            print(f"ARC-Easy loaded via {name}", flush=True)
            return ds
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"ARC-Easy load via {name} failed: {exc}", flush=True)
    raise RuntimeError(f"could not load ARC-Easy: {last_error}") from last_error


def option_logprob(model, tokenizer, device, prompt: str, option: str) -> float:
    """Sum log-prob of the option tokens given the prompt (teacher forcing)."""
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    option_ids = tokenizer(option, add_special_tokens=False, return_tensors="pt").input_ids.to(
        device
    )
    input_ids = torch.cat([prompt_ids, option_ids], dim=1)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0]
    # Log-prob of each option token given all previous tokens.
    logprobs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    # Option token at position j is predicted by logits at position len(prompt)+j-1.
    start = prompt_ids.shape[1]
    for j in range(option_ids.shape[1]):
        pos = start + j - 1
        tok = int(option_ids[0, j])
        total += float(logprobs[pos, tok])
    return total


def run_model(adapter: Path | None) -> tuple[float, list[dict]]:
    """Run ARC-Easy for one model; returns (accuracy, per-question records)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    ds = load_arc_easy()
    test = ds["test"]
    rng = random.Random(SEED)
    indices = rng.sample(range(len(test)), min(N_SAMPLE, len(test)))
    print(f"sampled {len(indices)} ARC-Easy test questions (seed {SEED})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=bnb,
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    device = next(model.parameters()).device

    records: list[dict] = []
    correct = 0
    for n, idx in enumerate(indices):
        item = test[idx]
        question = str(item["question"])
        choices = item["choices"]
        labels = list(choices["label"])
        texts = list(choices["text"])
        answer_key = str(item["answerKey"])
        if answer_key not in labels:
            continue
        gold_idx = labels.index(answer_key)
        prompt = f"Question: {question}\nAnswer:"
        scored = []
        for _label, text in zip(labels, texts, strict=False):
            lp = option_logprob(model, tokenizer, device, prompt, f" {text}")
            scored.append(lp)
        pred_idx = max(range(len(scored)), key=lambda i: scored[i])
        ok = pred_idx == gold_idx
        correct += int(ok)
        records.append(
            {
                "id": str(item.get("id", idx)),
                "gold": labels[gold_idx],
                "pred": labels[pred_idx],
                "correct": bool(ok),
            }
        )
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(indices)} done, running acc={correct / (n + 1):.3f}", flush=True)
    accuracy = correct / len(records) if records else 0.0
    del model
    torch.cuda.empty_cache()
    return accuracy, records


def bootstrap_diff(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Simple bootstrap interval over questions for mean(a)-mean(b)."""
    import random as _random

    rng = _random.Random(SEED)
    n = len(a_correct)
    deltas = [float(a) - float(b) for a, b in zip(a_correct, b_correct, strict=False)]
    point = sum(deltas) / n if n else 0.0
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        picked = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(picked) / n)
    means.sort()
    return {
        "point": point,
        "ci_low": means[int(0.025 * len(means))],
        "ci_high": means[int(0.975 * len(means))],
        "n_questions": n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["B0", "FT-main"])
    args = parser.parse_args()

    per_model: dict[str, dict] = {}
    for name in args.models:
        adapter = None if name == "B0" else ADAPTER
        print(f"=== ARC-Easy: {name} (adapter={adapter}) ===", flush=True)
        acc, records = run_model(adapter)
        per_model[name] = {"accuracy": acc, "n": len(records), "records": records}
        print(f"{name}: accuracy={acc:.4f} (n={len(records)})", flush=True)

    diff = None
    if "B0" in per_model and "FT-main" in per_model:
        a = [r["correct"] for r in per_model["FT-main"]["records"]]
        b = [r["correct"] for r in per_model["B0"]["records"]]
        diff = bootstrap_diff(a, b)
        diff["drop"] = per_model["B0"]["accuracy"] - per_model["FT-main"]["accuracy"]
        print(
            f"FT-main - B0 accuracy diff: {diff['point']:.4f} "
            f"CI=[{diff['ci_low']:.4f}, {diff['ci_high']:.4f}] "
            f"(drop={diff['drop']:.4f})",
            flush=True,
        )

    payload = {
        "dataset": "allenai/ai2_arc, config ARC-Easy, test split",
        "n_sample": N_SAMPLE,
        "seed": SEED,
        "method": "zero-shot multiple choice by option log-likelihood as continuation",
        "per_model": {k: {"accuracy": v["accuracy"], "n": v["n"]} for k, v in per_model.items()},
        "difference_ft_main_minus_b0": diff,
        "records": {k: v["records"] for k, v in per_model.items()},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
