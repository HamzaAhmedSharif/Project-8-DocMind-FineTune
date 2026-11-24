"""TRL SFT Trainer (config-driven, YAML).

Real trainer for F3.1 — loads base 4-bit NF4 + double quant, attaches LoRA,
builds TRL SFTConfig/SFTTrainer, pre-tokenizes data/train.jsonl through build_example
(input_ids/labels/attention_mask columns), and trains.  ``--check-only`` runs
CPU-only verification with no model loading.

Data format (TRL conversational): each row in data/train.jsonl has
  ``prompt`` = [{"role": "user", "content": "<text>"}],
  ``completion`` = [{"role": "assistant", "content": "<text>"}].

CLI:
    python -m docmind_ft.train --config configs/train_main.yaml
    python -m docmind_ft.train --config configs/train_main.yaml --max-steps 10
    python -m docmind_ft.train --config configs/train_main.yaml --resume-from runs/checkpoint-5
    python -m docmind_ft.train --config configs/train_main.yaml --check-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import peft
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

# Repo root and paths
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
METADATA_DIR = REPO_ROOT / "reports"
LOG_DIR = REPO_ROOT / "reports" / "logs"
RUNS_DIR = REPO_ROOT / "runs"
SEED = 20261207


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUNS_DIR / "ft-main",
        help="Output directory for checkpoints and adapter",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Override max training steps from config"
    )
    parser.add_argument(
        "--resume-from", type=Path, default=None, help="Resume from checkpoint directory"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="CPU-only structural check; no model loading or training",
    )
    return parser.parse_args(argv)


def load_config(config_path: Path) -> dict:
    import yaml

    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Fix the typo from Round 2
    cfg.setdefault("base_model", "meta-llama/Llama-3.2-3B-Instruct")
    cfg.setdefault("revision", "0cb88a4f764b7a12671c53f0838cd831a0843b95")
    cfg.setdefault("seed", SEED)
    return cfg


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes. Never hash an object."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_library_versions() -> dict:
    v: dict = {}
    try:
        import transformers

        v["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import torch

        v["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import peft

        v["peft"] = peft.__version__
    except ImportError:
        pass
    try:
        import datasets

        v["datasets"] = datasets.__version__
    except ImportError:
        pass
    try:
        import trl

        v["trl"] = trl.__version__
    except ImportError:
        v["trl"] = "unknown"
    return v


def get_git_commit() -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def hash_tokenizer_files(snapshot_dir: Path) -> str:
    """SHA-256 over the tokenizer files in a local HF snapshot dir.

    Hashes each file's name + bytes so a renamed file also changes the digest.
    Raises FileNotFoundError when no tokenizer file is present (never returns
    a placeholder hash).
    """
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "vocab.txt",
        "spiece.model",
        "added_tokens.json",
    )
    digest = hashlib.sha256()
    found = 0
    for name in names:
        path = snapshot_dir / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
            found += 1
    if found == 0:
        raise FileNotFoundError(f"no tokenizer files in {snapshot_dir}")
    return digest.hexdigest()


def build_example(tokenizer, prompt_msgs, completion_msgs, max_length: int):
    """Tokenize one conversational prompt/completion pair for training.

    The training format uses the chat template with add_generation_prompt=True
    (assistant header is part of the prompt).  Labels are -100 for every prompt
    position and the real token id for every completion position.

    Crucially, this function guarantees alignment BY CONSTRUCTION:
      input_ids  = prompt_ids + completion_ids
    So input_ids and labels always have the same length and the boundary is exact.

    Returns (input_ids, labels, length_ok) where length_ok is computed on the
    UNTRUNCATED length (over-length rows are dropped, never truncated).
    """
    # Prompt: apply chat template with assistant header
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    # Tokenize the prompt text (chat template already includes BOS;
    # add_special_tokens=False avoids duplicate BOS)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    # Completion: tokenize just the content, then append EOS
    comp_content = completion_msgs[0]["content"]
    ids_1 = tokenizer(comp_content, add_special_tokens=False)["input_ids"]
    ids_2 = [tokenizer.eos_token_id]
    completion_ids = ids_1 + ids_2

    # Concatenate: prompt then completion
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    # Length check BEFORE any truncation — over-length examples are dropped
    length_ok = len(input_ids) <= max_length
    return input_ids, labels, length_ok


def run_i2_check(tokenizer, train_path: Path, max_length: int) -> dict:
    """I2: token-length statistics on the real training file.

    Uses build_example for tokenization.  Returns dict with rows_total,
    rows_valid, rows_dropped, frac_valid, exit_code.  Exits non-zero if
    frac_valid < 0.99.
    """
    import datasets

    dataset = datasets.load_dataset("json", data_files=str(train_path), split="train")

    total = len(dataset)
    valid = 0
    dropped = 0

    for row in dataset:
        prompt_msgs = row.get("prompt", [])
        completion_msgs = row.get("completion", [])
        if not prompt_msgs or not completion_msgs:
            dropped += 1
            continue
        _, _, length_ok = build_example(tokenizer, prompt_msgs, completion_msgs, max_length)
        if length_ok:
            valid += 1
        else:
            dropped += 1

    frac_valid = valid / total if total > 0 else 1.0
    exit_code = 0 if frac_valid >= 0.99 else 1
    return {
        "rows_total": total,
        "rows_valid": valid,
        "rows_dropped": dropped,
        "frac_valid": round(frac_valid, 4),
        "exit_code": exit_code,
    }


def run_i3_mask_check(
    tokenizer,
    train_path: Path,
    n_examples: int = 20,
    seed: int = SEED,
    max_length: int = 3072,
    min_completion_frac: float = 0.95,
) -> dict:
    """I3: mask test on n_examples random real examples from train.jsonl.

    seed 20261207 fixed.  Uses build_example for tokenization so the mask
    gate measures the exact same tokenization that training uses.

    Returns dict with ok, n_tested, passed, failed, per_example list.
    Exits non-zero on any failure.
    """
    import random

    random.seed(seed)

    import datasets

    dataset = datasets.load_dataset("json", data_files=str(train_path), split="train")
    total = len(dataset)
    if total == 0:
        return {"ok": False, "error": "empty dataset"}

    sample_size = min(n_examples, total)
    indices = random.sample(range(total), sample_size)

    from docmind_ft.masking import check_completion_mask

    results = []
    all_ok = True

    for idx in indices:
        row = dataset[idx]
        prompt_msgs = row.get("prompt", [])
        completion_msgs = row.get("completion", [])
        if not prompt_msgs or not completion_msgs:
            results.append({"index": idx, "ok": False, "reason": "missing prompt or completion"})
            all_ok = False
            continue

        input_ids, labels, length_ok = build_example(
            tokenizer, prompt_msgs, completion_msgs, max_length
        )

        # Build chat text for prompt_ids extraction (same as build_example does)
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = tokenizer(prompt_text, add_special_tokens=False)
        prompt_ids = prompt_inputs["input_ids"]

        # Run the mask check with the correctly-aligned labels
        mask_result = check_completion_mask(
            prompt_ids=prompt_ids,
            input_ids=input_ids,
            labels=labels,
            min_completion_frac=min_completion_frac,
        )

        # BOS check: BOS token must appear exactly once at position 0
        bos_id = tokenizer.bos_token_id
        bos_first = bos_id is not None and input_ids[0] == bos_id
        bos_count = input_ids.count(bos_id) if bos_id else 0
        bos_once = bos_first and (bos_count == 1)

        example_ok = mask_result["ok"] and bos_once
        all_ok = all_ok and example_ok

        results.append(
            {
                "index": idx,
                "ok": example_ok,
                "mask_result": mask_result,
                "bos_first": bos_first,
                "bos_once": bos_once,
                "input_ids_len": len(input_ids),
                "completion_trained_frac": mask_result.get("completion_trained_frac", "N/A"),
            }
        )

    overall_ok = all_ok and len(results) == n_examples
    passed = sum(1 for r in results if r["ok"])
    failed = sum(1 for r in results if not r["ok"])
    exit_code = 0 if overall_ok else 1
    return {
        "ok": overall_ok,
        "n_tested": len(results),
        "passed": passed,
        "failed": failed,
        "per_example": results,
        "exit_code": exit_code,
    }


def write_model_metadata(
    metadata_path: Path,
    base_id: str,
    revision: str | None,
    tokenizer_hash: str,
    lora_config: dict,
    data_sha256: str,
    config_sha256: str,
    lib_versions: dict,
    git_commit: str,
    seed: int,
) -> None:
    """Write model_metadata.json to the checkpoint directory."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid silent overwrite by appending a counter if file already exists
    if metadata_path.is_file():
        base = metadata_path.stem
        ext = metadata_path.suffix
        counter = 1
        while (metadata_path.parent / f"{base}_{counter}{ext}").exists():
            counter += 1
        metadata_path = metadata_path.parent / f"{base}_{counter}{ext}"

    meta = {
        "base_id": base_id,
        "revision": revision,
        "tokenizer_hash": tokenizer_hash,
        "lora_config": lora_config,
        "data_file_sha256": data_sha256,
        "config_sha256": config_sha256,
        "library_versions": lib_versions,
        "git_commit": git_commit,
        "seed": seed,
    }
    with metadata_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    print(f"  model_metadata.json written to {metadata_path}")


class LossHeartbeatCallback(TrainerCallback):
    """Log train/dev loss to CSV, touch a heartbeat file, track peak VRAM,
    and write model_metadata.json into every checkpoint directory."""

    def __init__(
        self,
        output_dir: Path,
        log_steps: int = 10,
        metadata: dict | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.log_steps = log_steps
        self.metadata = metadata
        self.loss_csv = self.output_dir / "train_loss.csv"
        self.heartbeat = self.output_dir / "heartbeat.log"
        self.peak_vram_gb = 0.0
        self.epochs_completed = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_handle = self.loss_csv.open("w", encoding="utf-8", newline="\n")
        self._csv_handle.write("step,train_loss,eval_loss\n")
        self._csv_handle.flush()

    def on_log(self, args, state, control, **kwargs):
        """Called at every logging step."""
        if torch.cuda.is_available():
            self.peak_vram_gb = max(
                self.peak_vram_gb, round(torch.cuda.max_memory_allocated() / 1e9, 2)
            )
        if state.global_step > 0 and state.global_step % self.log_steps == 0:
            last = state.log_history[-1] if state.log_history else {}
            train_loss = last.get("loss", "")
            eval_loss = last.get("eval_loss", "")
            if train_loss != "" or eval_loss != "":
                self._csv_handle.write(f"{state.global_step},{train_loss},{eval_loss}\n")
                self._csv_handle.flush()
        self.heartbeat.write_text(f"step {state.global_step}\n", encoding="utf-8")

    def on_save(self, args, state, control, **kwargs):
        """Write model_metadata.json into the checkpoint just saved."""
        if self.metadata is None:
            return None
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.is_dir():
            return None
        metadata_path = checkpoint_dir / "model_metadata.json"
        with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.metadata, handle, indent=2, ensure_ascii=False, default=str)
        return None

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epochs_completed += 1

    def on_train_end(self, args, state, control, **kwargs):
        self._csv_handle.close()


def warmup_steps_for(n_rows, eff_batch, epochs, warmup_ratio, max_steps=None):
    """Translate a warmup ratio into warmup steps.

    TRL 1.12's SFTConfig has no warmup_ratio field (only warmup_steps), so
    the recipe's ratio is converted from the real total optimizer steps.
    """
    total = ((n_rows + eff_batch - 1) // eff_batch) * epochs
    if max_steps is not None:
        total = min(total, max_steps)
    return int(warmup_ratio * total)


def main() -> int:
    args = parse_args()

    if not args.config.is_file():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    model_id = cfg["base_model"]
    revision = cfg.get("revision")
    seed = cfg["seed"]
    max_length = cfg["train"]["max_length"]

    # Compute hashes early for metadata
    config_sha256 = sha256_file(args.config)
    lib_versions = get_library_versions()
    git_commit = get_git_commit()

    # Setup output directories
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- CHECK-ONLY MODE: CPU-only structural verification ---
    if args.check_only:
        print(f"Check-only mode with config: {args.config}")
        print(f"  base_model: {model_id}")
        print(f"  revision: {revision}")
        print(f"  seed: {seed}")
        print(f"  max_length: {max_length}")

        # Validate required config keys (errors, not warnings)
        required_top = {"base_model", "revision", "seed", "quant", "lora", "train"}
        missing_top = required_top - set(cfg.keys())
        if missing_top:
            missing_msg = f"  FATAL: config missing required top-level keys: {missing_top}"
            print(missing_msg, file=sys.stderr)
            return 1

        required_lora = {"r", "alpha", "dropout", "bias", "task_type", "target_modules"}
        missing_lora = required_lora - set(cfg.get("lora", {}).keys())
        if missing_lora:
            print(f"  FATAL: config missing lora keys: {missing_lora}", file=sys.stderr)
            return 1

        required_train = {
            "epochs",
            "learning_rate",
            "lr_scheduler",
            "warmup_ratio",
            "weight_decay",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "max_length",
            "gradient_checkpointing",
            "bf16",
            "packing",
            "completion_only_loss",
            "logging_steps",
        }
        missing_train = required_train - set(cfg.get("train", {}).keys())
        if missing_train:
            print(f"  FATAL: config missing train keys: {missing_train}", file=sys.stderr)
            return 1

        # Verify lora target_modules include required projections
        required_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
        provided_targets = set(cfg.get("lora", {}).get("target_modules", []))
        missing_targets = required_targets - provided_targets
        if missing_targets:
            tgt_msg = f"  FATAL: lora target_modules missing required: {missing_targets}"
            print(tgt_msg, file=sys.stderr)
            return 1

        # --- I2 token-length check on real train.jsonl ---
        print("\n--- I2: Token-Length Statistics (real data) ---")
        try:
            import transformers

            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id, revision=revision, local_files_only=True
            )
        except Exception:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)

        i2_result = run_i2_check(tokenizer, REPO_ROOT / "data" / "train.jsonl", max_length)
        print(f"  rows total:    {i2_result['rows_total']}")
        print(f"  rows valid:    {i2_result['rows_valid']}")
        print(f"  rows dropped:  {i2_result['rows_dropped']}")
        print(f"  frac valid:    {i2_result['frac_valid']}")
        if i2_result["exit_code"] == 0:
            print("  I2: PASS (>= 0.99 fraction valid)")
        else:
            print("  I2: FAIL (< 0.99 fraction valid)")

        # --- I3 mask test on 20 real examples ---
        print("\n--- I3: Mask Test (20 real examples, seed=20261207) ---")
        i3_result = run_i3_mask_check(
            tokenizer,
            REPO_ROOT / "data" / "train.jsonl",
            n_examples=20,
            seed=seed,
            max_length=max_length,
            min_completion_frac=0.95,
        )
        print(f"  tested:    {i3_result['n_tested']}")
        print(f"  passed:    {i3_result['passed']}")
        print(f"  failed:    {i3_result['failed']}")
        if i3_result["ok"]:
            print("  I3: PASS (all 20 examples OK)")
        else:
            print("  I3: FAIL")

        # Print per-example summary for failures
        for r in i3_result.get("per_example", [])[:5]:
            msg = f"  Example {r['index']}: ok={r['ok']} "
            msg += f"frac={r.get('completion_trained_frac', '?')} "
            msg += f"bos_first={r.get('bos_first', False)} "
            msg += f"bos_once={r.get('bos_once', True)}"
            print(msg)
        if len(i3_result.get("per_example", [])) > 5:
            print(f"  ... and {len(i3_result['per_example']) - 5} more examples")

        print("\nCheck-only mode complete.")
        return 0 if (i2_result["exit_code"] == 0 and i3_result["exit_code"] == 0) else 1

    # --- REAL TRAINING MODE (GPU) ---
    print(f"Training config: {model_id} revision={revision}")
    max_steps_label = args.max_steps if args.max_steps is not None else "from config (epochs)"
    print(f"  max_steps={max_steps_label}")
    print(f"  epochs={cfg['train']['epochs']}")
    print(f"  completion_only_loss={cfg['train']['completion_only_loss']}")

    # Load tokenizer
    try:
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
    except Exception:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenizer hash from the local HF snapshot, computed once before training
    # so the per-checkpoint metadata callback can use it.
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)
    )
    tokenizer_hash = hash_tokenizer_files(snapshot_dir)

    # Build LoRA config dict from cfg
    lora_cfg = {
        "r": cfg["lora"]["r"],
        "lora_alpha": cfg["lora"]["alpha"],
        "lora_dropout": cfg["lora"]["dropout"],
        "bias": cfg["lora"]["bias"],
        "task_type": "CAUSAL_LM",
        "target_modules": cfg["lora"]["target_modules"],
    }

    # Pre-tokenize the dataset through build_example
    import datasets

    dataset = datasets.load_dataset(
        "json", data_files=str(REPO_ROOT / "data" / "train.jsonl"), split="train"
    )
    original_len = len(dataset)

    def _tokenize_fn(example):
        prompt_msgs = example.get("prompt", [])
        completion_msgs = example.get("completion", [])
        if not prompt_msgs or not completion_msgs:
            example["input_ids"] = []
            example["labels"] = []
            example["attention_mask"] = []
            example["length_ok"] = False
            return example
        input_ids, labels, length_ok = build_example(
            tokenizer, prompt_msgs, completion_msgs, max_length
        )
        example["input_ids"] = input_ids
        example["labels"] = labels
        # attention_mask: 1 for real tokens, 0 for padding (no padding yet)
        example["attention_mask"] = [1] * len(input_ids)
        example["length_ok"] = length_ok
        return example

    dataset = dataset.map(_tokenize_fn, batched=False)
    kept = sum(1 for ex in dataset if ex.get("length_ok"))
    dropped = original_len - kept
    frac = kept / original_len if original_len else 1.0
    frac_str = f"{kept}/{original_len} kept, {dropped} dropped ({frac:.1%})"
    print(f"I2 length filter (pre-tokenized): {frac_str}")
    if frac < 0.99:
        print(f"  WARNING: I2 bar not met ({frac:.1%} < 99%)", file=sys.stderr)

    # Filter to length_ok rows and drop unwanted columns
    dataset = dataset.filter(lambda ex: ex.get("length_ok", False))
    dataset = dataset.remove_columns(["prompt", "completion", "meta", "length_ok"])
    print(f"Dataset prepared with input_ids column; {kept} rows pass I2 filter.")

    # Dev-loss eval set: data/dev_loss.jsonl when it exists (built by
    # scripts/build_dev_loss.py), pre-tokenized through the same build_example.
    dev_loss_path = REPO_ROOT / "data" / "dev_loss.jsonl"
    eval_dataset = None
    if dev_loss_path.is_file():
        eval_dataset = datasets.load_dataset("json", data_files=str(dev_loss_path), split="train")
        eval_dataset = eval_dataset.map(_tokenize_fn, batched=False)
        eval_dataset = eval_dataset.filter(lambda ex: ex.get("length_ok", False))
        eval_dataset = eval_dataset.remove_columns(["prompt", "completion", "meta", "length_ok"])
        print(f"Dev-loss eval set: {len(eval_dataset)} rows from {dev_loss_path}")
    else:
        print(
            f"WARNING: no dev-loss set at {dev_loss_path} — "
            "eval disabled (run scripts/build_dev_loss.py)",
            file=sys.stderr,
        )

    # 4-bit NF4 + double quant, bf16 compute — the QLoRA recipe (§3.7).
    # The compute dtype comes from cfg["quant"] as a string ("bfloat16").
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

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
        model_id,
        revision=revision,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    # Prepare model for kbit training (standard QLoRA recipe)
    model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Attach LoRA
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_cfg["target_modules"],
        ),
    )
    model.print_trainable_parameters()

    # SFTConfig from the YAML config.  The dataset is already pre-tokenized
    # (input_ids/labels/attention_mask), so preparation must be skipped via
    # dataset_kwargs — in TRL 1.12 `skip_prepare_dataset` is ONLY supported
    # there (trl/trainer/sft_config.py), not as a top-level SFTConfig field.
    # completion_only_loss is deliberately NOT passed: with skip_prepare_dataset
    # the trainer uses our labels as-is, and they already encode the mask
    # (-100 on prompt tokens, verified by the I3 gate).
    eval_strategy = "steps" if eval_dataset is not None else "no"
    warmup_steps = warmup_steps_for(
        len(dataset),
        cfg["train"]["per_device_train_batch_size"] * cfg["train"]["gradient_accumulation_steps"],
        cfg["train"]["epochs"],
        cfg["train"]["warmup_ratio"],
        max_steps=args.max_steps,
    )
    print(f"warmup: {warmup_steps} steps (ratio {cfg['train']['warmup_ratio']})")
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_length=max_length,
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["train"]["epochs"],
        learning_rate=cfg["train"]["learning_rate"],
        lr_scheduler_type=cfg["train"]["lr_scheduler"],
        warmup_steps=warmup_steps,
        weight_decay=cfg["train"]["weight_decay"],
        bf16=cfg["train"]["bf16"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        optim=cfg["train"]["optim"],
        logging_steps=cfg["train"]["logging_steps"],
        eval_strategy=eval_strategy,
        eval_steps=cfg["train"].get("eval_steps", 100),
        save_steps=cfg["train"].get("save_steps", 100),
        save_total_limit=cfg["train"].get("save_total_limit", 3),
        load_best_model_at_end=cfg["train"].get("load_best_model_at_end", False),
        report_to=[],
        seed=seed,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Override max-steps if provided
    if args.max_steps is not None:
        sft_config.max_steps = args.max_steps

    # Metadata payload: shared by the per-checkpoint callback and the final write.
    from dataclasses import asdict

    lora_config_dict = asdict(model.peft_config["default"])
    data_hash = sha256_file(REPO_ROOT / "data" / "train.jsonl")
    metadata = {
        "base_id": model_id,
        "revision": revision,
        "tokenizer_hash": tokenizer_hash,
        "lora_config": lora_config_dict,
        "data_file_sha256": data_hash,
        "config_sha256": config_sha256,
        "library_versions": lib_versions,
        "git_commit": git_commit,
        "seed": seed,
    }

    # Loss/heartbeat/metadata callback
    callback = LossHeartbeatCallback(
        args.output_dir, log_steps=cfg["train"]["logging_steps"], metadata=metadata
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[callback],
    )

    # --- Train ---
    checkpoint_path = args.resume_from if args.resume_from else None
    trainer.train(resume_from_checkpoint=checkpoint_path)
    trainer.save_model()  # save the final adapter
    tokenizer.save_pretrained(args.output_dir)

    # Write model_metadata.json into the run directory (the per-checkpoint
    # copies were written by the callback's on_save during training).
    write_model_metadata(
        metadata_path=args.output_dir / "model_metadata.json",
        base_id=metadata["base_id"],
        revision=metadata["revision"],
        tokenizer_hash=metadata["tokenizer_hash"],
        lora_config=metadata["lora_config"],
        data_sha256=metadata["data_file_sha256"],
        config_sha256=metadata["config_sha256"],
        lib_versions=metadata["library_versions"],
        git_commit=metadata["git_commit"],
        seed=metadata["seed"],
    )

    print(f"\nTraining complete. Checkpoint saved to {args.output_dir}")
    print(f"  tokenizer_hash: {tokenizer_hash[:16]}...")
    print(f"  data_file_sha256: {data_hash[:16]}...")
    print(f"  config_sha256: {config_sha256[:16]}...")
    print(f"  peak VRAM: {callback.peak_vram_gb} GB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
