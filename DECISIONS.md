# Project 8 — DocMind FineTune: decisions

Append-only record. Entries are never rewritten; a superseding entry is
added with its date. Format per script 15.3. The eight defining decisions
of script 3.1 open the F0 section, with their evidence.

## F0 - Repository, environment, runner, vendored Project 7 pieces (2026-11-14)

### Decision: Task — citation-grounded, query-focused answering with calibrated abstention
- Context: the roadmap named "query rewriting or domain summarization inside the RAG pipeline"; this is the second, read as query-focused summarization of retrieved scientific text — the generator step of DocMind. Project 7 measured where its answer quality was lost (generator-side grounding and refusal behaviour, plus retrieval rank for three facts) and that its capacity is provider-bound; public data with human evidence annotations exists (QASPER); the metrics are rule-based.
- Chosen: train the generator step: citation-grounded, query-focused answering over retrieved evidence, with calibrated abstention. Evidence: script 3.1 decision 1; P7 retrieval 23–24/26 facts (P7.3 close-out, 2026-11-13/14).
- Rejected: query rewriting (retrieval already reaches ~90% on well-formed single-turn questions, so a rewriter has almost no headroom; honest testing needs conversational data and a multi-turn feature DocMind does not have — stays Tier 3); free summarization (no objective ground truth, no pre-registered metric that is not an LLM judge).
- Tradeoff accepted: the task is narrower than the roadmap's wording; query rewriting is deferred to Tier 3.
- Would revisit if: the evaluation design is shown flawed after the freeze (script 14(c)).

### Decision: Base model — meta-llama/Llama-3.2-3B-Instruct, with a ladder
- Context: the roadmap's primary model; Llama access (H3) is unconfirmed at F0.
- Chosen: `meta-llama/Llama-3.2-3B-Instruct`. Ladder if Llama access, license acceptance or the smoke test fails: `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0), then `microsoft/Phi-4-mini-instruct` (MIT). Evidence: script 3.1 decision 2, 3.15. Llama's license requires the published adapter's name to start with "Llama", a "Built with Llama" notice and the license text.
- Rejected: starting on the ladder without testing Llama access (the roadmap's primary is Llama).
- Tradeoff accepted: F0 proceeds PENDING-H3; the base-model smoke (F0.3 step 5) is deferred until H3 resolves.
- Would revisit if: H3 confirms no Llama access (move to the ladder rung by rung).

### Decision: Data — QASPER (CC BY 4.0) with a local teacher
- Context: human-annotated evidence and answers make correctness verifiable without an LLM judge.
- Chosen: QASPER (CC BY 4.0). Teacher outputs come from the local `qwen3:8b` (Apache-2.0), filtered against the human answers; the Groq provider is never used for training data. Evidence: script 3.1 decision 3.
- Rejected: Groq as teacher (daily token limit makes it infeasible: ~80 answers/day, budget shared with the live demo); an LLM judge (self-grading, avoided across Hamza's projects).
- Tradeoff accepted: a local 8B teacher is weaker than a provider model; the F1.3 pilot decides distilled vs Option A targets.
- Would revisit if: the F1.3 pilot fails both the pass-rate and rubric gates (falls back to Option A targets, same data).

### Decision: Three test sets with different jobs
- Context: no single set covers statistical power, unseen-data honesty, and continuity with Projects 6–7.
- Chosen: QASPER test subset (statistical power, in-distribution); the held-out set authored for this project on post-cutoff papers (the only set a model cannot have seen; also the Project 9 anchor); the 16 Transformer-paper questions (continuity; famous-paper confound). Headline claims rest on the held-out set and the QASPER subset together, never on the Transformer questions alone. Evidence: script 3.1 decision 4.
- Rejected: a single test set (each job needs its own set); headline claims on the Transformer questions alone (famous-paper confound).
- Tradeoff accepted: three sets to build and guard under the one-shot rule (gate I7).
- Would revisit if: never after the F2.7 freeze (hypotheses and thresholds cannot change post-freeze).

### Decision: Comparator is a best-effort prompting baseline (B1), not a straw man
- Context: a fine-tuning result that beats only a naive prompt proves little.
- Chosen: B1 gets up to three prompt iterations tuned on the development set. Evidence: script 3.1 decision 5.
- Rejected: a naive-prompt baseline (proves little); unlimited B1 iteration (overfits the dev set).
- Tradeoff accepted: B1 tuning costs provider budget (capped in the 150-request budget).
- Would revisit if: never after the freeze.

### Decision: Everything runs locally at zero cost (RTX 4060, 8 GB)
- Context: the laptop GPU is an RTX 4060 Laptop, 8,188 MiB total, 7,956 MiB free at F0.1; zero API cost is a project rule.
- Chosen: native Windows compute path (conda env `docmind-ft`, Python 3.12). Ladder if the local path fails: WSL2, then Kaggle (needs phone verification, Class C), then Colab free. Evidence: script 3.1 decision 6; F0.1 inventory 2026-11-14 (GPU RTX 4060 Laptop, driver 610.88, CUDA UMD 13.3, 8,188 MiB; B: 168.5 GB free).
- Rejected: cloud-first (cost; the laptop card fits QLoRA 3B at 4-bit).
- Tradeoff accepted: 8 GB VRAM constrains batch size and sequence length (fallbacks in script 3.13, in order).
- Would revisit if: the native path fails after three fix attempts per step (script F0.3 step 6 → WSL2).

### Decision: Integration without changing Project 7
- Context: Project 7 is frozen and closed.
- Chosen: merge, quantize, import into Ollama, and point the unchanged service at Ollama's OpenAI-compatible endpoint. Evidence: script 3.1 decision 7.
- Rejected: modifying the service (P7 frozen); a new serving stack (unnecessary surface).
- Tradeoff accepted: the adapter must serve through Ollama's constraints (GGUF quantization).
- Would revisit if: Ollama cannot serve the merged model (F6 decision).

### Decision: Pre-registration and one-shot testing enforced by files and code
- Context: one agent builds and verifies (script 0), so the discipline must live in files and code, not in a second agent.
- Chosen: hypotheses, thresholds, metrics, splits and statistical procedures committed before any model is trained (3.11, F2.7 gate freeze); test sets touched once per model (gate I7, enforced by `generate.py` refusing without `DOCMIND_FT_ALLOW_TEST=1` and an unused ledger line). Evidence: script 3.1 decision 8, 1.4, 3.11.
- Rejected: relying on operator discipline alone.
- Tradeoff accepted: any post-freeze change to a hypothesis or threshold is forbidden; only integrity-gate procedures can change, and only for a demonstrated test defect.
- Would revisit if: never (the freeze is the point).

### Decision: Project 7 pin — b0d1f0fbc7eae71ebd3aa4029bde90f5364b1296
- Context: F0.2 pins the last pushed commit of P7 main whose CI run is green. The script's a4911c4 was superseded by the P7.3 close-out.
- Chosen: `b0d1f0fbc7eae71ebd3aa4029bde90f5364b1296` (main in sync with origin/main; CI green: workflow `ci` success + `release-smoke` success, 2026-11-14T03:10:15Z / 03:13:42Z). Evidence: GitHub API `GET /repos/HamzaAhmedSharif/Project-7-DocMind-Service/actions/runs?branch=main`, read 2026-11-14. The P7 checkout's only uncommitted items are untracked files (`.opencode/`, `AGENTS.md`, `evaluation/results/retrieval-local.json`, `retrieval-p73-local.json`, `graphify-out/`) — ignored per script.
- Rejected: a4911c4 (not the last green head).
- Tradeoff accepted: none.
- Would revisit if: never (the pin is immutable; P7 is closed).

### Decision: Sol executes F0 directly (manager clarification, 1.3)
- Context: script section 0 designates one agent to build and verify; the operating manual section 6 assigns environment/access work to Sol. F0 is environment, access and scaffolding, not bounded feature implementation.
- Chosen: Sol executes F0 directly. OpenCode + Ponytail apply from F1 implementation steps. Evidence: script section 0; project operating manual section 6.
- Rejected: delegating F0 to OpenCode (no token saving on env/access work; the split rule below reserves OpenCode for bounded mechanical code-writing).
- Tradeoff accepted: none.
- Would revisit if: n/a (F0-scoped).

### Decision: OpenCode/Sol split rule for P8 (Hamza, 2026-11-14)
- Context: Hamza's own words, 2026-11-14 ~19:56: use OpenCode only for tasks where it is better AND saves Sol tokens without making the project more difficult; Sol decides the split per step.
- Chosen: OpenCode — bounded mechanical code-writing with a clear spec and verifiable output (e.g. F1 data code, F2 harness/metrics), one step at a time, Sol verifies before the next step. Sol — environment/GPU ops, measurement, verification protocol V, decisions, debugging, and anything OpenCode fails at structurally (M7 fall-back rule: no retry burn). F0 itself stays Sol-direct. Evidence: Hamza 2026-11-14 ~19:56, relayed by the parent agent.
- Rejected: defaulting all implementation to OpenCode (token cost without benefit on ops/measurement work); defaulting everything to Sol (wastes Sol tokens on mechanical code-writing OpenCode does well).
- Tradeoff accepted: Sol judges the split per step; a wrong call costs a step, not a milestone (verify-before-next-step).
- Would revisit if: the split proves slower or costlier on a milestone (record the outcome, adjust the rule).

### Decision: Detached runner uses the windowless launcher pattern (1.3 deviation)
- Context: plain Task Scheduler powershell.exe actions surface visible Windows Terminal windows on this laptop despite `-WindowStyle Hidden` (five accumulated windows, 2026-11-01).
- Chosen: the scheduled task runs wscript.exe on a generated .vbs launcher; the launcher starts the job via WMI `Win32_Process.Create` with a `Win32_ProcessStartup` object set to SW_HIDE. Evidence: AGENTS.md lesson 2026-11-01; F0.4 test 2026-11-14: 0 visible PowerShell windows (3 powershell processes, all MainWindowHandle 0).
- Rejected: plain scheduled-task powershell action (visible windows); `-WindowStyle Hidden` (proven to fail on this laptop).
- Tradeoff accepted: the task's lifetime decouples from the job's (wscript exits immediately) — see the no-trigger Note below.
- Would revisit if: the laptop's terminal delegation changes (re-test with one dummy job).

### Decision: Tsinghua PyPI mirror for installs (Class B)
- Context: PyPI runs ~6 MB/min from the laptop; the Tsinghua mirror finished a large install in ~35 s reusing the uv cache (2026-11-09).
- Chosen: `--index-url https://pypi.tuna.tsinghua.edu.cn/simple/` for pip installs in F0.3. Evidence: AGENTS.md lesson 2026-11-09; F0.3 full-stack install completed 2026-11-14 in ~12 min.
- Rejected: pypi.org direct (too slow for a multi-GB stack).
- Tradeoff accepted: the mirror may lag PyPI; its torch listing carries only the CPU wheel (see the torch Note).
- Would revisit if: the mirror is stale or unreachable for a needed package.

### Decision: Vendor list extended to 10 files (1.3)
- Context: the 7 files in script 3.12 cannot be imported standalone.
- Chosen: vendor 10 files — the script's 7 plus `docmind_service/domain.py`, `config.py`, `metrics.py`. Evidence: import analysis at the pin, 2026-11-14 (GitHub raw at b0d1f0f): `query.py` imports domain/guardrails/metrics; `guardrails.py` imports domain; `llm.py` imports openai + config/domain/metrics; `ingest.py` imports pypdf + chunking; `metrics.py` imports prometheus_client + `docmind_service.guardrails`; `domain.py` imports numpy; `config.py` imports pydantic/pydantic-settings. `chunking.py`, `scoring.py`, `ledger.py` are stdlib-only.
- Rejected: vendoring only the 7 (imports fail); editing vendored files to remove imports (breaks the byte-exact rule).
- Tradeoff accepted: three more files to keep byte-exact; `pydantic`, `pydantic-settings`, `prometheus_client` added to the env (transitive deps of the vendored modules).
- Would revisit if: never (provenance is fixed at the pin).

### Decision: vendor/__init__.py installs a sys.modules alias "docmind_service"
- Context: vendored files import each other as `docmind_service.<name>`; byte-exact copies cannot be edited.
- Chosen: `docmind_ft/vendor/__init__.py` installs a `sys.modules` alias `"docmind_service"` → the vendor package. Import rule (documented in the file): service modules as `docmind_service.<name>` after importing `docmind_ft.vendor` once; `scoring`/`ledger` as `docmind_ft.vendor.<name>`; never import the same file under both names (distinct module objects would break dataclass identity). Evidence: import analysis 2026-11-14; F0.6 pytest (`tests/test_vendor.py` passes).
- Rejected: editing vendored imports (breaks byte-exactness); copying the package as a top-level `docmind_service` (name collision with a P7 checkout on sys.path).
- Tradeoff accepted: import order is load-bearing in test files (ruff I001 noqa on `tests/test_vendor.py`, documented in the file).
- Would revisit if: n/a.

### Decision (1.3 deviation): docmind-ft created by cloning llm_finetune, not a fresh install
- Context: F0.3 built a fresh Python 3.12 env and installed the full stack, including a CUDA torch wheel downloaded from the Aliyun mirror with a curl resume loop (download completed 21:16, sha256 verified; CUDA verified: torch 2.14.1+cu130, cuda_available True, bf16 True, matmul on cuda:0; bitsandbytes SUCCESS). The ungated smoke test was running on that env when Hamza redirected (2026-11-14 ~21:36): the laptop's existing `llm_finetune` env already holds the full fine-tuning stack, verified live (Python 3.10.21, torch 2.5.1+cu121, transformers 5.16.1, trl 1.12.0, peft 0.20.0, bitsandbytes 0.50.2, accelerate 1.14.0, datasets 5.0.1, huggingface_hub 1.29.0, numpy 2.2.6, psutil 7.2.2, PyYAML 6.0.3, tqdm 4.70.0; env dir 5.03 GB; B: 162.8 GB free).
- Chosen (Hamza's direction): stop the fresh-env work; remove the fresh docmind-ft env; `conda create --clone llm_finetune -p B:\Applications\MiniConda\envs\docmind-ft`; install only the missing small packages (fastembed, pypdf, openai, pytest, ruff). The clone keeps the docmind-ft name, so all script commands work unchanged, and keeps isolation (the original llm_finetune env is untouched). Rationale: the full stack is already present and proven (Hamza's earlier fine-tuning env); cloning skips the multi-GB torch download that had stalled the fresh path.
- Tradeoff accepted: Python 3.10.21 vs the script's 3.12/3.11, and older library versions (torch 2.5.1+cu121, transformers 5.16.1, trl 1.12.0). The F0.3 smoke test is the acceptance check: if it passes on the clone, 3.10 is acceptable for P8; if a 3.10-specific failure appears, fall back to the script's fresh 3.12 install path (whose torch download + install are already proven - see the torch entry below and reports/F0.md).
- Rejected: continuing the fresh 3.12 env (already complete and verified, but Hamza directed the clone; the fresh path remains the documented fallback).
- Would revisit if: the smoke test fails on a 3.10-specific issue, or a later milestone needs a feature the older stack lacks.

### Decision: torch 2.14.1+cu130 (not 2.15.0), wheel via Aliyun mirror + curl resume
- Status: this entry documents the FRESH env's torch (proven working 2026-11-14). The fresh env was replaced by the llm_finetune clone per the deviation above; this entry stands as the proven fallback path (the wheel file remains at `B:\p8\wheels\`).
- Context: PyPI's Windows torch wheels are CPU-only (2.11.0/2.14.1/2.15.0 wheels are all ~115–127 MB per Content-Length; the Tsinghua mirror serves the same). No PyTorch wheel index carries a 2.15.0 CUDA build. Large downloads from `download.pytorch.org` and the Aliyun mirror are reset partway by the laptop's network (curl: 96.8 MB at 2.67 MB/s, then "Connection was reset"; pip stalled with 0 bytes written for 25 min on the PyTorch CDN).
- Chosen: `torch==2.14.1+cu130`. The wheel (1,990,608,063 bytes, sha256 `0a09031e93632d14ef49553acd6211ece3ae9cdc0969a369b8c24e3cc4d15be9` from the PyTorch index) is downloaded from the Aliyun PyTorch wheel mirror (`https://mirrors.aliyun.com/pytorch-wheels/cu130/`) with a `curl -C -` resume loop (detached job, size + sha256 verified, exit 0 only on a match); the nvidia dependencies come from the Tsinghua PyPI index at `pip install` time. Evidence: `pip index versions torch` 2026-11-14 — cu128 tops at 2.11.0+cu128, cu129 at 2.9.0+cu129, cu130 at 2.14.1+cu130, no cu131 index; F0.3 fix attempts 1–5 (1: Tsinghua exact pin → +cpu; 2: cu128 index → no 2.15.0 build; 3: cu130 pip install → download stalled, CDN reset verified by curl; 4: Aliyun find-links pip → stalled the same way; 5: curl resume loop → working, ~1.7 MB/s).
- Rejected: torch 2.15.0+cpu (no CUDA — QLoRA impossible); torch 2.11.0+cu128 (older than 2.14.1); the conda pytorch channel (Tsinghua mirror tops at 2.5.1, too old for transformers 5.23 / trl 1.18); the pypi.org CUDA wheel (does not exist — PyPI wheels are CPU-only); VM download + transfer (the laptop cannot reach the VM's Tailscale IP directly; the SSH proxy is the only path).
- Tradeoff accepted: one minor version behind the PyPI latest; transformers 5.23.0 / trl 1.18.0 compatibility is verified by the F0 smoke test, not assumed; the wheel download is a resume loop, not a single pip install (documented in `reports/F0.md`).
- Would revisit if: a torch 2.15 CUDA build appears on an accessible index AND a measured incompatibility forces it.

### Decision: Smoke model — Qwen/Qwen3-0.6B (ungated); Llama smoke deferred PENDING-H3
- Context: H3 (Llama access) is unconfirmed; F0.3 step 4 uses the ungated smoke; step 5 (base-model smoke) is skipped on that basis — the script allows F0 to finish with the ungated smoke.
- Chosen: `scripts/smoke_qlora.py` defaults to `Qwen/Qwen3-0.6B`, 5 steps, 4-bit QLoRA, with an I3 loss-mask prototype on the processed dataset (dataloader-batch fallback). The Llama access test in F0.5 runs only if a token exists; no Llama download is attempted without access. Evidence: smoke-f0d (clone env, 2026-11-15 00:15-00:18, exit 0):
  `SMOKE_OK model=Qwen/Qwen3-0.6B steps=5 mask=PASS peak_vram_gb=1.65
  elapsed_s=157`. Train losses [1.7224, 0.6086, 0.0587, 0.0128, 0.0052]
  over 5 steps (train_runtime 16.1 s); trainable parameters 10,092,544;
  peak VRAM 1.65 GB; adapter saved to `B:\p8\runs\f0-smoke`; reload
  generation `'<think>\n\n</think>\n\nThe paper uses dense retrieval. [c1]'`.
  I3 mask prototype PASS: all 20 examples ok, prompt fully masked
  (66/66, 68/68, 69/69, 73/73 per question), completion 100% trained.
  The smoke-f0c run (2026-11-14 22:38-22:42) trained identically but
  reported mask=FAIL: the check passed a BatchEncoding to
  `check_completion_mask` because transformers 5.x
  `apply_chat_template(tokenize=True)` returns a BatchEncoding, not a
  list of ids (len 2 = number of keys). TRL's masking was correct all
  along (labels: 67x -100 then the completion ids, confirmed by the
  diag-mask3 token-level dump); the fix extracts `["input_ids"]` in
  `mask_test_on_dataset`. Python 3.10.21 is therefore ACCEPTED for P8
  (the clone deviation's acceptance check passes).
- Rejected: attempting a Llama download without access (script forbids it); skipping the smoke (the smoke decides the compute path).
- Tradeoff accepted: the Llama-specific smoke (step 5) is deferred until H3 resolves.
- Would revisit if: H3 confirms Llama access (run step 5 then).

### Note: P7 main moved after the pin — the pin stands
- Symptom: P7's main moved after F0 pinned b0d1f0f (owner-authorized polish pass: commits c33c9f1, a2edae8 lockfile regen, a8d4db8 Dependabot; v1.0.0 RELEASED 2026-11-14, tag → 1ed8f06, head 1ed8f06).
- Cause: Hamza authorized a presentation-polish pass in the P7 chat after F0 started; it is docs/presentation, not a re-measurement.
- Fix: none — the pin stands. b0d1f0f is a valid green head (CI green at pin time) and vendoring is hash-pinned to it (`check_vendor.py` verifies every file against the pin's blobs). Do not chase the moving head and do not re-pin after vendoring. Evidence: parent-agent heads-up 2026-11-14 ~20:47; vendor check PASSED at b0d1f0f.
- Lesson: pin once, at the F0.2 rule's green head; later upstream movement is not a reason to re-pin.
- Symptom: the script's F6.4 section predates the P7.3 close-out.
- Cause: the script was written before P7.3's A4/B4 runs (2026-11-13/14).
- Fix: at F6.4, compare against A4 23/26, B4 24/26, ceiling 26/26 (P7.3 close-out), not the script's older numbers. Evidence: P7 close-out docs at b0d1f0f (`docs/EVALUATION.md`).
- Lesson: cross-check script numbers against the P7 close-out before using them.

### Note: vendored ledger.py default LEDGER_PATH resolves to docmind_ft/reports/ in P8
- Symptom: `ledger.py`'s default path is relative to its package root; in P8 that is `docmind_ft/reports/` (vs the repo root in P7).
- Cause: the vendored module's `__file__`-relative default.
- Fix: P8 callers pass `path=` explicitly. Evidence: F0.6 ledger test uses an explicit tmp path.
- Lesson: check vendored defaults against the new layout.

### Note: vendored ledger.py needs datetime.UTC — shimmed for Python 3.10
- Symptom: `check.ps1` failed at pytest on the clone env (2026-11-15):
  `ImportError: cannot import name 'UTC' from 'datetime'` —
  vendored `ledger.py` does `from datetime import UTC, datetime`, and
  `datetime.UTC` exists only from Python 3.11. The clone env is 3.10.21.
- Cause: P7 targets 3.11+; the P8 clone env is 3.10. A repo-wide grep
  found no other 3.11+ markers (`Self`, `ExceptionGroup`, `tomllib`,
  `StrEnum`, `TaskGroup`); `UTC` in ledger.py is the only one.
- Fix: `docmind_ft/vendor/__init__.py` (our file, not vendored) sets
  `datetime.UTC = datetime.timezone.utc` when the attribute is missing
  — the identical object `UTC` aliases in 3.11+. The package `__init__`
  always runs before any vendored submodule import, so the shim is in
  place before ledger.py's `from datetime import UTC` executes. The
  vendored files stay byte-exact.
- Lesson: a Python version step down needs a stdlib-compat grep over
  vendored code, not just the package versions.

### Note: is_abstention(NO_EVIDENCE_ANSWER) is False
- Symptom: the vendored abstention rule does not match the service's own abstention sentence ("could not find" matches no pattern).
- Cause: the rule's patterns do not cover that phrasing. Evidence: verified by running the vendored scorer on the VM, 2026-11-14.
- Fix: none — P8's abstain_acc checks the exact sentence OR the rule (script 3.8), so the design is already robust. A test records this behaviour, not a fix.
- Lesson: verify vendored helpers against the service's own outputs, not just their docstrings.

### Note: PROMPT_VERSION is p7-v3 at the pin
- Symptom: vendored `guardrails.py` sets `PROMPT_VERSION = "p7-v3"`; the file's own comment documents the p7-v3 vs p7-v2 difference (a COMPLETENESS instruction added below the unchanged RULES hardening text — "the only prompt change").
- Cause: n/a (record only).
- Fix: none. Evidence: `git log -S p7-v3` in P7 → commit `806954b` ("P7.2: quote-anchored citation repair + prompt p7-v3 completeness instruction"); `git show 806954b -- docmind_service/guardrails.py` confirms: PROMPT_VERSION p7-v2→p7-v3, RULES hardening text and `sanitize_context` unchanged, COMPLETENESS block added ("Answer every part of the question..."), `build_prompt_v2` renamed to `build_prompt_v3` with a compatibility alias. Verified 2026-11-14.
- Lesson: n/a.

### Note: Runner double-start (F0.4) — time trigger removed, mutex added
- Symptom: the first f0-test run fired twice (heartbeat "start" at 20:16:41 and 20:18:40); two concurrent dummy jobs; the log mixed both children's output.
- Cause: wscript.exe exits as soon as the WMI Create call returns, so the scheduler saw the task as Ready again within a second and the +2 min trigger fired a second wrapper instance. `MultipleInstances IgnoreNew` cannot dedupe a task that never appears Running.
- Fix: the task is registered with NO trigger; `run_detached.ps1` starts it with `Start-ScheduledTask` and waits up to 60 s for the wrapper's heartbeat "start" line (throws and removes the task on failure). The wrapper also claims a named mutex (`p8-job-<Name>`) and exits silently if another instance holds it. Evidence: f0-test2 re-test 2026-11-14 — a single "start" line, no second instance.
- Lesson: with a fire-and-forget launcher, the scheduler's instance policy is blind — dedupe in the wrapper, not the task.

### Note: Runner ExitCode empty (F0.4) — .NET Process.Start
- Symptom: f0-test wrote `.failed` with `exit_code=` (empty) for a job that exited 0.
- Cause: `Start-Process -PassThru` `.ExitCode` reads empty in this host (TOOLS.md, 2026-10-20).
- Fix: the wrapper launches the job with raw .NET `[Diagnostics.Process]::Start` (`cmd.exe /c`, whose redirection writes stdout/stderr straight to the log files — no pipe buffer to fill) and reads `.ExitCode` from the .NET Process object. Evidence: f0-test3 re-test 2026-11-14 — `.done` with `exit_code=0`; f0-fail wrote `.failed` with `exit_code=1`.
- Lesson: apply the TOOLS.md quirks list when writing new launch code — do not re-learn it.

### Note: Unregister-ScheduledTask failed from the wrapper (F0.4) — elevation split, fixed with COM + SDDL
- Symptom: `p8-f0-test` survived both wrapper instances (still Ready after both ended at 20:21:40); f0-test2/f0-fail heartbeats logged `unregister: cmdlet failed: Access is denied.; schtasks exit=1`.
- Cause: the SSH session runs ELEVATED (high integrity, Administrators enabled); a task created by the elevated token gets a DACL of user:(R) + Administrators:(F) (icacls on `C:\Windows\System32\Tasks\p8-f0-test2`, 2026-11-14). The wrapper runs as the same user (same SID `S-1-5-21-527483915-428460999-916734862-1001`, owner `HAMZAAHMEDSHARI\HAMZA AHMED SHARIF`) but with a FILTERED token (UAC, InteractiveToken logon), so it has only read access to the task and deletion is denied. Evidence: whoami / Export-ScheduledTask / icacls comparison, 2026-11-14.
- Fix: registration now goes through the Task Scheduler COM API with an explicit SDDL granting the user full control (`D:(A;;GA;;;SID)(A;;GA;;;SY)(A;;GA;;;BA)`); the finally block still tries the cmdlet, falls back to `schtasks.exe /Delete /F`, and logs the outcome. COM property names differ from the cmdlet's (`DisallowStartIfOnBatteries`/`StopIfGoingOnBatteries`, set false). Evidence: f0-test3 re-test 2026-11-14 — heartbeat `unregister: ok`, task gone (schtasks query + task file both absent).
- Lesson: never swallow a cleanup error silently — log it; and an elevated creator plus a filtered-token worker is an access-denied trap.

### Note: PowerShell 5.1 .ps1 files must be ASCII-only (F0.4)
- Symptom: run_detached.ps1 v5 (stale-cleanup error message) failed to parse on the laptop: "Missing closing '}'", "Unexpected token ')'" at the throw line, and the wrapper here-string misparsed as code - while PSParser.Tokenize reported SYNTAX OK (it only lexes; brace matching is a parse check).
- Cause: the throw string contained an em-dash (U+2014). The file is BOM-free UTF-8; PowerShell 5.1 reads BOM-less files as the system ANSI code page (CP1252), where the em-dash's bytes E2 80 94 decode as `a"-` and the `"` (U+201D) terminates the double-quoted string early, corrupting the rest of the parse. Em-dashes in comments were harmless (v4 had them and worked); the one inside a double-quoted string broke it.
- Fix: replaced all em-dashes with hyphens (grep -P '[^\x00-\x7F]' now 0); re-transferred; the smoke job started cleanly.
- Lesson: keep .ps1 files ASCII-only. (Python files are fine with UTF-8 - Python 3 reads source as UTF-8 by default.)

### Note: Killing a detached job must match the whole process tree (F0.4)
- Symptom: after killing the dl-torch2 download job by its job name, the replacement job's file shrank (701 MB → 660 MB); two curl processes were found writing the same wheel file concurrently.
- Cause: the kill matched the command-line pattern 'dl-torch2', but the job script is `dl-torch.ps1` (shared across job names) — the job powershell survived, kept its loop running, and started a new curl alongside the replacement job's curl. The wheel file was corrupted by concurrent writers (and a truncation) and had to be deleted.
- Fix: kill by the broad pattern 'dl-torch' (wrapper + cmd + job powershell) plus any curl.exe, verify zero remain, and only then restart. A future improvement is a stop script that kills the wrapper's process tree (or a job object). Evidence: kill-dl-all.ps1 found and killed 2 curls + 2 powershells + 1 cmd; REMAINING DL PROCESSES: 0.
- Lesson: a detached job is a process TREE (wrapper → cmd → job → children); killing the wrapper alone orphans the rest.

### Note: Model downloads use HF_HOME=B:\p8\hf-cache and the hf-mirror (script RK3)
- Symptom: the laptop cannot reach HuggingFace directly.
- Cause: network restriction (script RK3).
- Fix: `HF_HOME=B:\p8\hf-cache` + `HF_ENDPOINT=https://hf-mirror.com` for that command only.
- Lesson: n/a (script rule).

## F1 - Evaluation sets and training data (2026-11-15)

### Decision: Retrieval is scoped to each question's own paper (F1.2)
- Context: script 3.5 step 3 says "retrieve top 10 per question by cosine similarity" without stating the candidate pool. QASPER questions are paper-specific and every gold evidence paragraph lives in the question's paper; context assembly (3.5 step 5) assumes a per-paper ranked list ("fill with the next-ranked non-gold chunks").
- Chosen: each question is ranked only against its own paper's chunks (docmind_ft/build_data.py main: rows_by_paper; retrieval.retrieve over one paper's pool). Evidence: F1.2 dry run 2026-11-15 — 46,411 chunks over 5.75M words; Q-DEV/Q-TEST splits valid.
- Rejected: corpus-wide retrieval across all 1,585 papers (gold chunks could be crowded out of the top 10 by other papers' chunks, breaking the gold mapping and the abstention design; the retrieved context would mostly be same-topic noise from other papers).
- Tradeoff accepted: the retrieval task is easier than corpus-wide (within-paper ranking); the gold_in_top10 rate measures within-paper retrieval quality, not corpus retrieval. The fine-tuned generator is still evaluated on identical frozen contexts as the baselines, so the comparison stays fair.
- Would revisit if: never after the F2.7 freeze (contexts are frozen in F1.6).

### Note: Chunk count 46,411, not the script's ~100,000 (F1.2)
- The script (F1.2) estimated "about 100,000 chunks". The dry run counted 46,411 over 5.75M words: at a 125-word step that is the correct arithmetic, not a data problem. The embedding run is correspondingly shorter than the 10-25 minute estimate.
