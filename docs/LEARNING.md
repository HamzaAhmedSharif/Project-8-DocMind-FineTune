# Project 8 — Learning log

One entry per milestone: the concept, what Hamza ran, what it proved.
Doubles as interview prep.

Hands-on rule (Hamza, 2026-11-14): Sol/OpenCode automate long repetitive
runs, scaffolding, bulk ops and verification; Hamza does the fundamentals
himself with Sol's exact steps, so he can debug/fix without an agent:
GitHub Actions (read the CI file, trigger + read runs, diagnose failures),
smoke tests (what they are, run + read output), CLI + local runs (conda
env, pytest/ruff, nvidia-smi, run a train/eval command), debugging (read
tracebacks/logs, reproduce errors).

## F0 — hands-on items (walkthrough after the F0 verdict)

1. Smoke test: what a smoke test is, run `scripts\smoke_qlora.py`, read the
   output (losses, peak VRAM, mask result, SMOKE_OK line).
   From the repo dir, in cmd:
     set HF_HOME=B:\p8\hf-cache
     set HF_ENDPOINT=https://hf-mirror.com
     B:\Applications\MiniConda\envs\docmind-ft\python.exe scripts\smoke_qlora.py
   (HF_HOME keeps the model cache on B:; HF_ENDPOINT points at the
   hf-mirror because the laptop cannot reach HuggingFace directly.)
2. CI file: read `.github/workflows/ci.yml`, trigger a run on the pushed
   repo, read the run output.
3. Local run: activate the `docmind-ft` env, run pytest + ruff,
   `nvidia-smi`, one train/eval command.
