# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Generation harness (F2.1).

One interface, three backends: ``hf`` (Transformers, 4-bit NF4 base with an
optional LoRA adapter, or a merged bf16 model), ``ollama`` (the
OpenAI-compatible endpoint) and ``provider`` (the vendored hosted adapter).

Every run reads a frozen ``evaluation/contexts/contexts_<set>.jsonl`` file,
renders the vendored p7-v3 prompt (or a supplied variant template) with a
fresh seeded canary per question, generates greedily (``do_sample=False``,
200 new tokens, batch size 1), and appends one JSON line per question.
Finished question ids are skipped, so an interrupted run resumes without
re-sampling.  Test splits are guarded by gate
I7 (``DOCMIND_FT_ALLOW_TEST=1`` + a pre-declared model + one ledger line per
model/split pair).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

import docmind_ft.vendor  # noqa: F401  (installs the docmind_service alias)

from docmind_ft.contexts import (  # noqa: E402
    CONTEXTS_DIR,
    HOLD_PAGES_DIR,
    PAPER16_PAGES,
    _load_holdout_questions,
    _load_paper16_questions,
    load_work_questions,
)
from docmind_ft.frozen import (  # noqa: E402
    FrozenError,
    frozen_b1_prompt_sha256,
    verify_context_hash,
)
from docmind_ft.jsonl import read_jsonl as _read_jsonl  # noqa: E402
from docmind_ft.prompts import build_prompt_variant  # noqa: E402
from docmind_ft.vendor.chunking import chunk_pages  # noqa: E402
from docmind_service.domain import RetrievedChunk  # noqa: E402
from docmind_service.guardrails import build_prompt_v3  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = Path(r"B:\p8\data\work")
CHUNKS_PATH = WORK_DIR / "chunks.jsonl"
LEDGER_PATH = REPO_ROOT / "reports" / "test-eval-ledger.jsonl"
CLAIMS_DIR = REPO_ROOT / "evaluation" / "results" / ".claims"

SPLITS = ("dev", "qtest", "hold", "paper16")
TEST_SPLITS = frozenset({"qtest", "hold", "paper16"})
DECLARED_MODELS = (
    "B0",
    "B1",
    "FT-main",
    "FT-nodistill",
    "FT-r64",
    "FT-data25",
    "R",
)
SPLIT_CONTEXTS = {
    "dev": CONTEXTS_DIR / "contexts_qdev.jsonl",
    "qtest": CONTEXTS_DIR / "contexts_qtest.jsonl",
    "hold": CONTEXTS_DIR / "contexts_hold.jsonl",
    "paper16": CONTEXTS_DIR / "contexts_paper16.jsonl",
}
MAX_NEW_TOKENS = 200
SEED = 20261207
UTC = timezone.utc  # noqa: UP017  # Python 3.10 has no datetime.UTC


class TestGuardError(RuntimeError):
    """Gate I7 refused a test-split run."""


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int | None
    generated_tokens: int | None


class Backend(Protocol):
    name: str
    model_id: str
    revision: str | None
    adapter_sha256: str | None

    def generate(self, prompt: str) -> Generation: ...


@dataclass(frozen=True)
class RunSummary:
    split: str
    model: str
    questions: int
    written: int
    skipped: int
    output_path: Path
    sha256: str
    ledger_written: bool


def adapter_hash(path: str | Path) -> str:
    """SHA-256 over an adapter's file names and bytes (sorted, deterministic)."""
    target = Path(path)
    files = (
        sorted(item for item in target.rglob("*") if item.is_file())
        if target.is_dir()
        else [target]
    )
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def check_test_guard(
    model: str,
    split: str,
    *,
    ledger_path: Path | None = None,
    allow_test: bool | None = None,
) -> None:
    """Gate I7: env flag, pre-declared model id and an unused ledger line."""
    if split not in TEST_SPLITS:
        return
    allowed = _env_bool("DOCMIND_FT_ALLOW_TEST") if allow_test is None else allow_test
    if not allowed:
        raise TestGuardError("refusing a test-split run: DOCMIND_FT_ALLOW_TEST=1 is required")
    if model not in DECLARED_MODELS:
        raise TestGuardError(f"refusing a test-split run: model {model!r} is not pre-declared")
    if split == "qtest" and model == "R":
        raise TestGuardError("refusing a test-split run: R is not pre-declared for 'qtest'")
    # R is the only split-restricted model (script 3.10: R on HOLD and
    # PAPER16 only); B0, B1 and the FT-* models run on all test splits.
    for entry in _read_jsonl(ledger_path or LEDGER_PATH, recover_tail=False):
        if entry.get("model") == model and entry.get("split") == split:
            raise TestGuardError(f"refusing a second test-split run for {model!r} on {split!r}")


def _default_question_texts(split: str) -> dict[str, str]:
    if split in ("dev", "qtest"):
        return {qid: entry["question"] for qid, entry in load_work_questions().items()}
    if split == "hold":
        return {qid: question for qid, _, question in _load_holdout_questions()}
    if split == "paper16":
        return {qid: question for qid, _, question in _load_paper16_questions()}
    raise ValueError(f"unknown split: {split!r}")


class ChunkTextResolver:
    """Chunk text by id: QASPER cache, HOLD pages, or the PAPER16 pages."""

    def __init__(
        self,
        *,
        chunks_path: Path = CHUNKS_PATH,
        hold_dir: Path = HOLD_PAGES_DIR,
        paper16_pages: Path = PAPER16_PAGES,
    ) -> None:
        self.chunks_path = chunks_path
        self.hold_dir = hold_dir
        self.paper16_pages = paper16_pages
        self._qasper: dict[str, str] | None = None
        self._page_chunks: dict[Path, dict[int, str]] = {}

    def _qasper_text(self, chunk_id: str) -> str:
        if self._qasper is None:
            by_id: dict[str, str] = {}
            with self.chunks_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    by_id[str(row["chunk_id"])] = str(row["text"])
            self._qasper = by_id
        try:
            return self._qasper[chunk_id]
        except KeyError as exc:
            raise ValueError(f"context chunk {chunk_id} is not in {self.chunks_path}") from exc

    def _page_text(self, pages_path: Path, chunk_id: str) -> str:
        _, _, ordinal_text = chunk_id.rpartition(":")
        try:
            ordinal = int(ordinal_text)
        except ValueError as exc:
            raise ValueError(f"bad chunk id: {chunk_id!r}") from exc
        if pages_path not in self._page_chunks:
            data = json.loads(pages_path.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not all(isinstance(page, str) for page in data):
                raise ValueError(f"{pages_path}: expected a JSON list of page strings")
            self._page_chunks[pages_path] = {
                chunk.ordinal: chunk.text for chunk in chunk_pages(data)
            }
        try:
            return self._page_chunks[pages_path][ordinal]
        except KeyError as exc:
            raise ValueError(f"chunk {chunk_id} is not in {pages_path}") from exc

    def text(self, chunk_id: str, paper_id: str, split: str) -> str:
        if split in ("dev", "qtest"):
            return self._qasper_text(chunk_id)
        if split == "hold":
            if Path(paper_id).name != paper_id:
                raise ValueError(f"unsafe paper id: {paper_id!r}")
            return self._page_text(self.hold_dir / f"{paper_id}.pages.json", chunk_id)
        if split == "paper16":
            return self._page_text(self.paper16_pages, chunk_id)
        raise ValueError(f"unknown split: {split!r}")


def _build_chunks(
    context: Mapping,
    paper_id: str,
    resolver: ChunkTextResolver,
    split: str,
) -> list[RetrievedChunk]:
    chunks = []
    for item in context["chunks"]:
        chunk_id = str(item["chunk_id"])
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=uuid5(NAMESPACE_URL, paper_id),
                page=int(item["page"]),
                text=resolver.text(chunk_id, paper_id, split),
                score=float(item["score"]),
            )
        )
    return chunks


def _append_ledger(path: Path, model: str, split: str, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "model": model,
        "split": split,
        "time": datetime.now(UTC).isoformat(),
        "sha256": digest,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _claim_test_run(model: str, split: str, claims_dir: Path | None = None) -> Path | None:
    """Atomically claim a test-split run before generation starts."""
    if split not in TEST_SPLITS:
        return None
    directory = claims_dir or CLAIMS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model}__{split}.claim"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise TestGuardError(
            f"refusing a test-split run: a claim already exists for {model!r} on {split!r} ({path})"
        ) from exc
    payload = {
        "model": model,
        "split": split,
        "pid": os.getpid(),
        "time": datetime.now(UTC).isoformat(),
    }
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _release_claim(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


def _resume_ids(path: Path, *, model: str, split: str, backend: Backend) -> set[str]:
    """Read finished ids, refusing rows from another run or duplicates."""
    expected = {
        "model": model,
        "split": split,
        "model_id": backend.model_id,
        "backend": backend.name,
        "revision": backend.revision,
        "adapter_sha256": backend.adapter_sha256,
    }
    finished: set[str] = set()
    for line_number, row in enumerate(_read_jsonl(path), start=1):
        question_id = row.get("question_id")
        if question_id is None:
            continue
        question_id = str(question_id)
        if question_id in finished:
            raise ValueError(f"{path}:{line_number}: duplicate question_id {question_id!r}")
        finished.add(question_id)
        for key, value in expected.items():
            if key in row and row[key] != value:
                raise ValueError(
                    f"{path}:{line_number}: {key} {row[key]!r} does not match this run ({value!r})"
                )
    return finished


def run_generation(
    *,
    split: str,
    model: str,
    backend: Backend,
    out_path: Path,
    contexts_path: Path | None = None,
    question_texts: Mapping[str, str] | None = None,
    chunk_texts: Mapping[str, str] | None = None,
    prompt_file: str | Path | None = None,
    ledger_path: Path | None = None,
    claims_path: Path | None = None,
    seed: int = SEED,
    max_new_tokens: int = MAX_NEW_TOKENS,
    allow_test: bool | None = None,
) -> RunSummary:
    """Generate, append and resume one model/split run."""
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split!r}")
    check_test_guard(model, split, ledger_path=ledger_path, allow_test=allow_test)
    if prompt_file is not None and not Path(prompt_file).is_file():
        raise FileNotFoundError(f"prompt file not found: {prompt_file}")

    context_path = Path(contexts_path) if contexts_path is not None else SPLIT_CONTEXTS[split]
    if contexts_path is None or context_path.resolve().parent == CONTEXTS_DIR.resolve():
        verify_context_hash(context_path)
    contexts = _read_jsonl(context_path, recover_tail=False)
    questions = (
        dict(question_texts) if question_texts is not None else _default_question_texts(split)
    )
    missing = [
        str(context["question_id"])
        for context in contexts
        if str(context["question_id"]) not in questions
    ]
    if missing:
        raise ValueError(f"no question text for {len(missing)} ids (e.g. {missing[:5]})")
    resolver = _MappingResolver(chunk_texts) if chunk_texts is not None else ChunkTextResolver()
    prompt_path = Path(prompt_file) if prompt_file is not None else None
    prompt_template_sha256 = (
        hashlib.sha256(prompt_path.read_bytes()).hexdigest() if prompt_path is not None else None
    )
    if model == "B1":
        expected = frozen_b1_prompt_sha256()
        if expected is None:
            raise FrozenError("evaluation/gates.json has no meta.b1_prompt_sha256")
        if prompt_template_sha256 is None:
            raise FrozenError("model B1 requires the frozen prompt file")
        if prompt_template_sha256 != expected:
            raise FrozenError(
                f"frozen B1 prompt mismatch: expected {expected}, got {prompt_template_sha256}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path = _claim_test_run(model, split, claims_path)
    try:
        # Recheck under the claim: another process may have finished and
        # written the ledger between the first guard check and this claim.
        check_test_guard(model, split, ledger_path=ledger_path, allow_test=allow_test)
        finished = _resume_ids(out_path, model=model, split=split, backend=backend)
        context_ids = {str(context["question_id"]) for context in contexts}
        unexpected = finished - context_ids
        if unexpected:
            raise ValueError(
                f"{out_path}: {len(unexpected)} finished ids are not in split {split!r} "
                f"(e.g. {sorted(unexpected)[:5]})"
            )
        rng = random.Random(seed)
        written = 0
        skipped = 0
        with out_path.open("a", encoding="utf-8", newline="\n") as handle:
            for context in contexts:
                question_id = str(context["question_id"])
                canary = f"P8-{rng.getrandbits(48):012x}"
                if question_id in finished:
                    skipped += 1
                    continue
                question = questions[question_id]
                paper_id = str(context["paper_id"])
                chunks = _build_chunks(context, paper_id, resolver, split)
                if prompt_file is None:
                    prompt, _ = build_prompt_v3(question, chunks, canary)
                    # No template file: hash the exact prompt sent for this row.
                    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                else:
                    prompt = build_prompt_variant(prompt_file, question, chunks, canary)
                    prompt_sha256 = prompt_template_sha256
                started = time.perf_counter()
                generated = backend.generate(prompt)
                seconds = time.perf_counter() - started
                row = {
                    "question_id": question_id,
                    "split": split,
                    "model": model,
                    "model_id": backend.model_id,
                    "revision": backend.revision,
                    "adapter_sha256": backend.adapter_sha256,
                    "backend": backend.name,
                    "prompt_sha256": prompt_sha256,
                    "canary": canary,
                    "seed": seed,
                    "max_new_tokens": getattr(backend, "max_new_tokens", max_new_tokens),
                    "prompt_tokens": generated.prompt_tokens,
                    "generated_tokens": generated.generated_tokens,
                    "seconds": round(seconds, 6),
                    "output": generated.text,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                finished.add(question_id)
                written += 1

        if split in TEST_SPLITS and written + skipped != len(contexts):
            raise ValueError(
                f"{out_path}: wrote {written} and skipped {skipped}, "
                f"expected {len(contexts)} questions for split {split!r}"
            )
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        ledger_written = split in TEST_SPLITS
        if ledger_written:
            _append_ledger(ledger_path or LEDGER_PATH, model, split, digest)
        return RunSummary(
            split=split,
            model=model,
            questions=len(contexts),
            written=written,
            skipped=skipped,
            output_path=out_path,
            sha256=digest,
            ledger_written=ledger_written,
        )
    finally:
        _release_claim(claim_path)


class _MappingResolver:
    """Test/small-run resolver backed by a plain chunk-id -> text mapping."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.mapping = mapping

    def text(self, chunk_id: str, paper_id: str, split: str) -> str:
        try:
            return self.mapping[chunk_id]
        except KeyError as exc:
            raise ValueError(f"no chunk text for {chunk_id!r}") from exc


def _resolved_commit(model) -> str | None:
    """The commit hash Transformers records when it resolves a hub revision."""
    value = getattr(getattr(model, "config", None), "_commit_hash", None)
    return str(value) if value else None


class HFBackend:
    """Transformers backend: 4-bit NF4 base, optional LoRA, or merged bf16."""

    name = "hf"

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        adapter_path: str | Path | None = None,
        merged_bf16: bool = False,
        load_in_4bit: bool = True,
        max_new_tokens: int = MAX_NEW_TOKENS,
        token: str | None = None,
    ) -> None:
        if adapter_path and merged_bf16:
            raise ValueError("a merged bf16 model cannot also take an adapter")
        self.model_id = model_id
        self.revision = revision
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.adapter_sha256 = adapter_hash(self.adapter_path) if self.adapter_path else None
        self.merged_bf16 = merged_bf16
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.token = token
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, token=self.token
        )
        kwargs = {
            "revision": self.revision,
            "device_map": "auto",
            "dtype": torch.bfloat16,
            "token": self.token,
        }
        if not self.merged_bf16 and self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        resolved = _resolved_commit(model)
        if resolved:
            self.revision = resolved
        if self.adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(self.adapter_path))
        model.eval()
        self._model = model
        self._device = next(model.parameters()).device

    def generate(self, prompt: str) -> Generation:
        self._ensure_loaded()
        messages = [{"role": "user", "content": prompt}]
        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(rendered, return_tensors="pt").to(self._device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][prompt_tokens:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return Generation(text, prompt_tokens, int(generated_ids.shape[0]))


class OllamaBackend:
    """Ollama's OpenAI-compatible endpoint (used for the F6 service runs)."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        timeout: float = 300.0,
        max_new_tokens: int = MAX_NEW_TOKENS,
        seed: int = SEED,
    ) -> None:
        self.model_id = model
        self.revision = None
        self.adapter_sha256 = None
        base = host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
        self.url = base.rstrip("/") + "/v1/chat/completions"
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        self.seed = seed

    def generate(self, prompt: str) -> Generation:
        import requests

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": self.max_new_tokens,
            "seed": self.seed,
            "stream": False,
        }
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )
        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Ollama returned an unexpected response") from exc
        usage = data.get("usage") or {}
        return Generation(
            str(text).strip(),
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )


class ProviderBackend:
    """The vendored hosted adapter (reference model R)."""

    name = "provider"

    def __init__(self, settings=None) -> None:
        from docmind_service.config import Settings
        from docmind_service.llm import HostedLLM

        self.settings = settings or Settings()
        self._llm = HostedLLM(self.settings)
        self.model_id = str(self.settings.llm_model)
        self.revision = None
        self.adapter_sha256 = None
        # The vendored adapter's P7 completion limit is intentional for R.
        self.max_new_tokens = self.settings.llm_max_completion_tokens

    def generate(self, prompt: str) -> Generation:
        result = self._llm.complete(prompt)
        return Generation(result.text, result.prompt_tokens, result.completion_tokens)
