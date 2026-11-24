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

## F3.1 — QLoRA trainer over a pre-tokenized masked dataset (2026-11-25)

- Concept: the trainer runs over a pre-tokenized masked dataset via
  `dataset_kwargs={"skip_prepare_dataset": True}`, with
  `completion_only_loss` left to auto-detect. TRL 1.12's `SFTConfig`
  has no `warmup_ratio` field, so the warmup ratio is translated to
  `warmup_steps` before the config is built.
- What ran: dev-loss build — 374 sampled / 201 kept / gate PASSED
  (I2 valid share 1.0, all six overlaps 0). 5-step smoke train
  (`--max-steps 5`), 04:33–06:17 (~1h44m), checkpoint-5 saved with
  adapter + metadata. No train loss was logged: `logging_steps` (10)
  exceeds `max_steps` (5), so no train-log event fired and
  `train_loss.csv` holds only its header. The only loss on record is
  eval_loss 0.7897 at step 5 (eval entropy 0.8748, mean token
  accuracy 0.7902). Peak VRAM was not recorded in the artifacts;
  live `nvidia-smi` during the run showed ~7.9 GB used at 100% GPU.
- What it proved: mask gates I2/I3 pass and the GPU training path
  works end to end.
- Environment notes: `HF_HOME` is now a persistent user variable
  (`B:\p8\hf-cache`, set with `setx`); the HF login token lives at
  `B:\p8\hf-cache\token`. The smoke took ~1h44m, not 3-5 min, because
  5 optimizer steps = 80 micro-steps at ~50 s each (16 accumulation
  steps per optimizer step).
