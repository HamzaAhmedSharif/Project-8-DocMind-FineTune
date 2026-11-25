"""F3.3 learning-test evaluator: fresh process, greedy, exact match.

Loads the base model + LoRA adapter, generates one greedy answer per row of
data/learn8.jsonl, and compares whitespace-normalized text to the training
target. Exit 0 when at least 6 of 8 match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_adapter(path: Path) -> Path:
    """Latest checkpoint-* under path when present, else path itself."""
    checkpoints = sorted(path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return checkpoints[-1] if checkpoints else path


def normalize(text: str) -> str:
    return " ".join(text.split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=REPO_ROOT / "runs" / "ft-learn")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "learn8.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from docmind_ft.train import load_config

    cfg = load_config(REPO_ROOT / "configs" / "train_main.yaml")
    adapter_dir = resolve_adapter(args.adapter)
    print(f"adapter: {adapter_dir}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg["base_model"], revision=cfg["revision"], local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_by_name[cfg["quant"]["bnb_4bit_compute_dtype"]]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["quant"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quant"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["quant"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        revision=cfg["revision"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    rows = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    passed = 0
    for index, row in enumerate(rows, start=1):
        target = row["completion"][0]["content"]
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        ok = normalize(generated) == normalize(target)
        passed += int(ok)
        print(f"[{index}/{len(rows)}] {'PASS' if ok else 'FAIL'} target={target!r}")
        if not ok:
            print(f"          generated={generated!r}")
    print(f"match: {passed}/{len(rows)} (pass needs >= 6/8)")
    return 0 if passed >= 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
