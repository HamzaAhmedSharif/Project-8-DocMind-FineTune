# Project 8 — DocMind FineTune

Status: F0 in progress.

A QLoRA fine-tune of a small open model (Llama 3.2 3B Instruct) for
citation-grounded, query-focused answering over retrieved evidence, with
calibrated abstention — the generator step of the DocMind RAG pipeline
(Project 7, DocMind Service). Evaluated against a best-effort prompting
baseline of the same model and a 120B provider reference, on QASPER plus a
held-out set of post-cutoff papers. Zero API cost; training runs locally on
an 8 GB laptop GPU.

No results yet. Numbers will appear here only after they are measured under
the pre-registered protocol (see `evaluation/gates.json` once frozen at F2.7).

## Layout

- `docmind_ft/` — package (config, masking, detached-runner helpers, `vendor/`)
- `scripts/` — runner, checks, vendoring, smoke tests
- `evaluation/` — gates, splits, held-out set, results
- `configs/` — training configurations
- `docs/` — architecture, data, training, evaluation, model card, facts
- `DECISIONS.md` — append-only decision record

## Run it

Requires the `docmind-ft` conda environment on the laptop (see
`docs/OPERATIONS.md` from F7; until then, the milestone reports under
`reports/` — git-ignored — record the exact commands).
