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
- **HO-1 (F3.2 pace, verified)**: median ~9.7 s, short ~6.2 s, long ~36 s per micro-step at GA 1. Confirmed by the three probe variants: median train_runtime ~58 s / 6 steps = 9.7 s/step; short ~37 s / 6 steps = 6.2 s/step; long sequences exceed 8 GB and die with OOM before completing all steps. At GA 16, each optimizer step is ~15 min. **Lesson**: the pacing is length-dependent; short sequences train proportionally faster, long sequences hit the 8 GB VRAM ceiling.

- **HO-2 (F3.4 resume)**: the 12-step run completed (94.85s, peak VRAM 5.98 GB) but the external kill-resume (HO-2) is still outstanding. Hamza must perform the kill by hand at heartbeat step 6, then resume with --resume-from checkpoint-4 --max-steps 12. Pass criteria: resumed log starts at step 5, final global_step 12, LR at step 5 after resume equals pre-kill. Results will be appended to this log after the run completes.

- **HO-2 (F3.4 resume, VERIFIED PASS 2026-11-26)**: the resume from checkpoint-4 works after the torch 2.6.0 upgrade. The first 12-step run finished in 105 s before the manual kill landed, so the kill itself never happened -- but the resume (the part that matters) was verified: resumed log starts at step 5, final global_step 12, LR at step 5 after resume = 0.00015 = pre-kill. The torch upgrade (2.5.1 -> 2.6.0) was needed because transformers refuses torch.load below 2.6 (CVE-2025-32434); upgrading torch also forced torchvision/torchaudio to match (the old torchvision broke the peft import). The learn test re-run on torch 2.6.0 passes all three criteria (ratio 0.0184, 8/8 exact match, no NaN), so the upgrade did not change the learning. Note: a resumed run saves its adapter in F32 (97 MB) instead of BF16 (48 MB) -- same weights, just a different dtype.

## F0/F1 hands-on walkthrough -- completed by Hamza (2026-11-26)

- F0 item 1 (smoke test): ran `scripts\smoke_qlora.py` himself.
  `SMOKE_OK model=Qwen/Qwen3-0.6B steps=5 mask=PASS peak_vram_gb=1.65
  elapsed_s=47`; mask test 20/20, train losses 1.7556 -> 0.0053,
  adapter saved to `B:\p8\runs\f0-smoke`; reload generation produced
  a citation-grounded answer. Lesson: a folder path typed as a command
  fails; `cd` into the folder first, then run the command.
- F0 item 2 (CI file): read `.github/workflows/ci.yml` (4 jobs: lint,
  unit, gates, security) and checked the Actions tab. The two latest
  runs were red; diagnosis found two real defects, both fixed hands-on:
  (1) `tests/test_train.py` imported torch, which CI does not install
  -- fixed with `pytest.importorskip("torch")` (commit 6323ec7);
  (2) `tests/test_fetch_heldout.py` read a hardcoded Windows path to
  the Project-7 repo on the laptop -- fixed by building a 24-paper
  manifest with the file's own `_paper()`/`_write_manifest()` helpers
  in `tmp_path` (commit bda2cd6). CI run #8 is green (17 s, all jobs
  pass). Lesson: a test must be portable -- it may only read files
  inside the repo or files it builds itself; a test that reads an
  absolute path on one machine can never pass on CI.
- F0 item 3 (local run): pytest 289 passed (fresh basetemp each run;
  reusing a stale basetemp dir produces mass teardown PermissionErrors,
  and running pytest from the wrong directory sweeps up unrelated
  files); ruff check + format clean; `nvidia-smi` shows the GPU free;
  `scripts\eval_learn8.py` 8/8 exact match, including both abstention
  targets. Lesson: `cd` to the repo first, then run; pytest is CPU-only
  so 0% GPU during a test run is expected.
- F1 item 1 (inspect QASPER): ran `reports\logs\inspect_qasper.py`.
  papers 888, questions 2593. First paper 1909.00694 (77 paragraphs,
  9 questions); first question "What is the seed lexicon?" with two
  annotator answers (one free-form, one extractive span); one
  unanswerable question found (paper 1908.06606) -- the abstention
  case the training data is built around.
- What it proved: the full local verification loop (smoke, tests, lint,
  eval, CI) runs green end to end, and the two CI failures found were
  real portability defects, not flakes.

## F5 — sealed test evaluation (2026-11-27)

Concept: the sealed test is evaluated exactly once, with pre-registered
thresholds, and a refuted hypothesis is reported as refuted.
What I ran: `scripts/final_eval_compute.py` (paper-cluster bootstrap,
seed 20261207) on the 11 per-(model,split) metrics; `scripts/run_redteam.py`
(15 cases, 2 repeats, B0 + FT-main); `docmind_ft/regression.py` (ARC-Easy,
seeded 200); `scripts/error_analysis.py` (30 Q-TEST + 20 HOLD failures).
What it proved: format/citation are near-perfect (HY1 confirmed) but the
answer-F1 gain (+0.027) is below the bar (HY2 refuted), abstention improved
a lot yet over-refuses (HY3 refuted), and keyfact coverage did not improve
(HY4 refuted). No regression and no jailbreak (HY5 confirmed).
Lesson: the 120B reference does not follow the citation contract, so it is
not a quality ceiling on HOLD — a reference that ignores the format cannot
set the bar for coverage. And a model can learn "say I don't know" too well:
abstain_acc 0.636 with over_refusal 0.143.
