# Project 8 — DocMind FineTune: agent instructions

QLoRA fine-tune of Llama 3.2 3B Instruct for citation-grounded answering
over retrieved evidence, with calibrated abstention. One agent builds and
verifies (script section 0); every milestone ends with verification
protocol V in a fresh process. Never grade your own homework: recompute
gates from raw files, never from a report.

## Reply style — token discipline
- Answer in the shortest form that stays complete. No greetings, preambles,
  recaps of the request, or step-by-step narration.
- Never open with filler ("Happy to help", "Great question", "Let me explain").
- Prefer short declarative sentences; drop filler words where meaning stays clear.
- If a tool call is needed, run it first and report only the result.
- State what changed and what remains once; do not repeat context across replies.

## Rules that never bend
- Zero API cost. No paid service, card, or paid API without Hamza's approval.
- Pre-register, then measure. Gates, budgets, stop rules are committed
  before any measured run (`evaluation/gates.json`, frozen at F2.7).
- Verification before completion. An attempted action is not a completed one.
- Never fabricate, never cherry-pick. Original runs stay in every table.
- Append-only decisions. `DECISIONS.md` entries are never rewritten.
- Budget is law. Provider requests: at most 150 total, 60 per UTC day.
- Credentials never in chat, logs, or the repo. Secrets live in `.env`
  (git-ignored); check keys by name only.
- Checkpoints and adapters are never deleted by a script.
- Projects 1–7 are frozen. Read and vendor from them; never modify them.
- Never `git push` from SSH (Windows keyring unreachable). Hamza pushes.

## Skills (global OpenCode skills, auto-loaded when the task matches)
- `finetuning-experiments` — experiment records, checkpoint protection,
  base-vs-fine-tuned on the same eval set.
- `rag-evaluation` — gates/budget pre-registration, retrieval scored
  separately from generation.
- `ship-check` — pre-commit/close-out checks (ruff check + format,
  pytest, secrets scan, demo verification).
- `ml-pipeline` — six-stage ML template, leakage and metric-identity audit.
- Ponytail (`/ponytail`, `/ponytail-audit`, `/ponytail ultra`) — minimal-code
  review after implementation, at every milestone.

## Environment
- Laptop only (Windows 11). Conda env `docmind-ft` at
  `B:\Applications\MiniConda\envs\docmind-ft`, Python 3.10.21 (clone of
  `llm_finetune`; see DECISIONS.md).
- Long jobs (>10 min) run through `scripts\run_detached.ps1` — never
  hold an SSH session open. Never run two GPU jobs at once.
- Large files live on `B:\p8\` (hf-cache, runs, data, models). The repo
  holds code, configs, small results, and documents only.
- Line endings LF, UTF-8 without BOM. Write files from Python with
  `encoding="utf-8", newline="\n"`.
