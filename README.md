# Project 8 — DocMind FineTune

Status: F2.7 — gates frozen; B0/B1 baselines and reference outputs complete.

A QLoRA fine-tune of a small open model (Llama 3.2 3B Instruct) for
citation-grounded, query-focused answering over retrieved evidence, with
calibrated abstention — the generator step of the DocMind RAG pipeline
(Project 7, DocMind Service). Evaluated against a best-effort prompting
baseline of the same model and a 120B provider reference, on QASPER plus a
held-out set of post-cutoff papers. Zero API cost; training runs locally on
an 8 GB laptop GPU.

Sealed outputs are complete; hypothesis metrics remain sealed until F5.
Numbers will appear here only after they are measured under the
pre-registered protocol in `evaluation/gates.json`.

## Layout

- `docmind_ft/` — package (config, masking, detached-runner helpers, `vendor/`)
- `scripts/` — runner, checks, vendoring, smoke tests
- `evaluation/` — gates, splits, held-out set, results
- `configs/` — empty placeholder for training configurations
- `docs/` — data notes and learning log
- `DECISIONS.md` — append-only decision record

## Run it

Requires the `docmind-ft` conda environment on the laptop. See
`docs/DATA.md` for the data layout and build commands; the milestone reports
under `reports/` (git-ignored) record the exact commands for each run.
