#!/usr/bin/env python3
"""F0.3 step 4: ungated QLoRA smoke test + I3 loss-mask prototype.

Loads a tiny public model (default Qwen/Qwen3-0.6B) in 4-bit NF4, attaches
LoRA (r=16, all linear), trains 5 steps with TRL on 20 synthetic
prompt-completion examples, saves the adapter, reloads it in a fresh model
instance, calls model.eval(), and generates one answer. Also runs the
loss-mask test (gate I3 prototype) on the trainer's processed dataset:
every prompt token must have label -100 and at least 95% of completion
tokens must carry a label. This test detects a TRL option rename by
checking the labels the trainer actually produced.

Run with HF_HOME=B:\\p8\\hf-cache and, if Hugging Face is unreachable,
HF_ENDPOINT=https://hf-mirror.com (script RK3, mirror for that command only).
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from docmind_ft.masking import check_completion_mask  # noqa: E402

SEED = 20261207

QUESTIONS = [
    ("What method do the authors use for retrieval?", "dense retrieval"),
    ("How many papers are in the evaluation set?", "twenty-four papers"),
    ("What metric is reported for answer quality?", "token-level F1"),
    ("Which model is used as the teacher?", "a local 8B model"),
]


def build_examples(n: int = 20) -> list[dict]:
    examples = []
    for i in range(n):
        q, a = QUESTIONS[i % len(QUESTIONS)]
        context = (
            f"<<<c1 page=1>>> The paper studies {a} in detail. "
            f"The evaluation uses {a} throughout. <<<end c1>>>"
        )
        prompt = (
            "Answer the question using only the evidence below. "
            "Cite the supporting chunk like [c1].\n"
            f"{context}\nQuestion: {q}"
        )
        completion = f"The paper uses {a}. [c1]"
        examples.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "completion": [{"role": "assistant", "content": completion}],
            }
        )
    return examples


def resolve_template(tokenizer) -> str:
    if tokenizer.chat_template:
        return "tokenizer"
    # Minimal Qwen-style template fallback (recorded if used).
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}"
        "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    return "fallback-qwen-style"


def sft_kwargs(sft_config_cls, output_dir: str) -> dict:
    wanted = {
        "output_dir": output_dir,
        "max_length": 512,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_steps": 5,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 1,
        "save_strategy": "no",
        "bf16": True,
        "gradient_checkpointing": True,
        "completion_only_loss": True,
        "packing": False,
        "report_to": [],
        "seed": SEED,
        "dataset_num_proc": 1,
    }
    params = inspect.signature(sft_config_cls).parameters
    kept = {k: v for k, v in wanted.items() if k in params}
    if "max_length" not in kept and "max_seq_length" in params:
        kept["max_seq_length"] = 512
    dropped = sorted(set(wanted) - set(kept))
    if dropped:
        print(f"SFTConfig: dropped unsupported options: {dropped}")
    return kept


def mask_test_on_dataset(trainer, tokenizer, examples: list[dict]) -> bool:
    """I3 prototype on the trainer's processed dataset (or one dataloader batch)."""
    dataset = trainer.train_dataset
    rows = None
    if hasattr(dataset, "column_names") and "input_ids" in dataset.column_names:
        rows = [(r["input_ids"], r["labels"]) for r in dataset]
        print(f"mask test: using processed dataset ({len(rows)} rows)")
    else:
        loader = trainer.get_train_dataloader()
        batch = next(iter(loader))
        input_ids = batch["input_ids"].tolist()
        labels = batch["labels"].tolist()
        attn = batch["attention_mask"].tolist()
        rows = [
            (
                ids[: sum(m)],
                labs[: sum(m)],
            )
            for ids, labs, m in zip(input_ids, labels, attn, strict=True)
        ]
        print(f"mask test: using one dataloader batch ({len(rows)} rows)")
    all_ok = True
    for i, (ids, labs) in enumerate(rows):
        # transformers 5.x: apply_chat_template(tokenize=True) returns a
        # BatchEncoding (len == number of keys), not a list of ids.
        prompt_ids = tokenizer.apply_chat_template(
            examples[i % len(examples)]["prompt"],
            tokenize=True,
            add_generation_prompt=True,
        )["input_ids"]
        result = check_completion_mask(prompt_ids, ids, labs)
        print(f"  example {i}: ok={result['ok']} {result}")
        all_ok = all_ok and result["ok"]
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default=r"B:\p8\runs\f0-smoke")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    t0 = time.time()
    print(f"smoke model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    template_source = resolve_template(tokenizer)
    print(f"chat template: {template_source}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    print(f"model loaded on {model.device if hasattr(model, 'device') else 'auto'}")

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    from trl import SFTConfig, SFTTrainer

    examples = build_examples(20)
    dataset = Dataset.from_list(examples)
    sft_args = sft_kwargs(SFTConfig, args.output_dir)
    sft_args["max_steps"] = args.steps
    trainer_kwargs: dict = {
        "model": model,
        "args": SFTConfig(**sft_args),
        "train_dataset": dataset,
    }
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "peft_config" in trainer_params:
        trainer_kwargs["peft_config"] = lora_cfg
        trainer = SFTTrainer(**trainer_kwargs)
    else:
        trainer_kwargs["model"] = get_peft_model(model, lora_cfg)
        trainer = SFTTrainer(**trainer_kwargs)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable}")

    mask_ok = mask_test_on_dataset(trainer, tokenizer, examples)
    print(f"I3 mask prototype: {'PASS' if mask_ok else 'FAIL'}")

    trainer.train()
    losses = [entry.get("loss") for entry in trainer.state.log_history]
    print(f"train losses: {[round(v, 4) for v in losses if v is not None]}")
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak VRAM: {peak_vram:.2f} GB")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"adapter saved to {args.output_dir}")

    del trainer
    del model
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    reloaded = PeftModel.from_pretrained(base, args.output_dir)
    reloaded.eval()
    test_prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Answer using only the evidence below. Cite like [c1].\n"
                "<<<c1 page=1>>> The paper uses dense retrieval. <<<end c1>>>\n"
                "Question: What method do the authors use for retrieval?",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(test_prompt, return_tensors="pt").to(reloaded.device)
    with torch.no_grad():
        out = reloaded.generate(**inputs, max_new_tokens=60, do_sample=False)
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"reload generation: {answer!r}")

    elapsed = time.time() - t0
    print(
        f"SMOKE_OK model={args.model} steps={args.steps} "
        f"mask={'PASS' if mask_ok else 'FAIL'} peak_vram_gb={peak_vram:.2f} "
        f"elapsed_s={elapsed:.0f}"
    )
    return 0 if mask_ok else 1


if __name__ == "__main__":
    sys.exit(main())
