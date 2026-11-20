# Project 8 — Data

## QASPER v0.3

- Source: Allen Institute for AI, https://allenai.org/data/qasper
- License: CC BY 4.0. Attribution: Dasigi, P., Lo, K., Beltagy, I.,
  Cohan, A., Smith, N. A., & Gardner, M. (2021). A Dataset of
  Information-Seeking Questions and Answers Anchored in Research Papers.
  See `NOTICE`.
- Local location: `B:\p8\data\qasper\`.

| Split | Papers | Questions |
|---|---|---|
| train | 888 | 2593 |
| validation (dev) | 281 | 1005 |
| test | 416 | 1451 |

Archives and checksums:

- `https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz`
  — SHA-256 `a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a`;
  contains `qasper-train-v0.3.json` and `qasper-dev-v0.3.json`.
- `https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz`
  — SHA-256 `72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72`;
  contains `qasper-test-v0.3.json`.

Fetch: `python scripts/fetch_qasper.py`. It retries each download up to 3
times, verifies the SHA-256 before unpacking, skips an archive already on
disk with a matching checksum, and exits non-zero on any checksum or count
mismatch.

Loader: `docmind_ft.qasper.load_qasper(path)` normalizes one split into
papers with title, abstract, paragraphs (section name plus half-open word
offsets over the document's whitespace-split stream), and questions with
their annotator answers. Test-split questions keep whatever answer list
the file carries; the released test file includes annotator answers, so
Q-TEST stratification uses them.

## F1.2 build (2026-11-15)

- Chunking: vendored P7 `chunk_pages` (150 words, 25 overlap) over
  pseudo-pages (page = 1 + floor(word_offset / 500)). 46,411 chunks from
  1,585 papers; 1,584 papers yield at least one chunk (one paper yields
  none). Mean 29.3 chunks/paper, median 28, min 1, max 209.
- Splits (seed 20261207, at most 3 questions per paper, stratified by
  answer type): Q-DEV 200 questions / 144 papers
  (SHA-256 `82a21e035582d999eb510f37dc6b49ebfbe4de90dd9e5a29ca531b854a259e8c`),
  Q-TEST 400 questions / 271 papers
  (SHA-256 `4d236746a55c55b05f3e8b5318b1a674a6fa3295747541fa5e65f99e8d2d43e3`).
  Mixed-label questions (annotators disagree on answerability): 13 in
  Q-DEV, 51 in Q-TEST; kept for F1, excluded from abstention counts.
- Retrieval is scoped to each question's own paper (see DECISIONS.md):
  top-10 comes from that paper's chunks only. Embeddings: fastembed
  `sentence-transformers/all-MiniLM-L6-v2` (ONNX, 384-dim), exact cosine,
  chunk-id tie-break. Embeddings are cached fingerprint-keyed under
  `B:\p8\data\work\`; the full chunk embedding took 20,216 s
  single-threaded, and the re-run hit the cache (build in 225 s).
- Gold coverage (script section 3.5 step 4): 4,303 of 5,049 questions
  have gold chunks (the rest are unanswerable or gold-less). A gold chunk
  appears in the top-10 for 86.5% overall (3,721 / 4,303): train 82.7%
  (1,738 / 2,101), dev 88.9% (793 / 892), test 90.8% (1,190 / 1,310).
  On the frozen eval samples: Q-DEV 86.4% (153 / 177), Q-TEST 90.6%
  (328 / 362). By answer type: extractive 87.1%, free-form 85.5%,
  yes/no 84.9%. Note: Q-DEV yes/no is 65.0% (13 / 20) on a small
  sample — a watch item, not a gate failure.
- Train composition for F1.4: 2,593 questions — 2,101 answerable with
  gold, 271 unanswerable, 221 answerable without gold (excluded from
  training: they cannot pass the V2 gold-citation check).
- Artifacts under `B:\p8\data\work\`: `chunks.jsonl` (46,411 rows),
  `embeddings.npy` (46,411 x 384), `questions.jsonl` (5,049 rows,
  including `gold_scores`: gold chunk id to real cosine score).

## F1.4 train files (2026-11-20)

Three frozen train files under `data/`, produced from the 2,593 Q-TRAIN
questions (2,101 answerable with gold, 271 unanswerable; 221 answerable
without gold excluded from training).

| File | Rows | Abstention share | I2 valid share |
|---|---|---:|---:|
| `data/train.jsonl` | 1,444 | 0.1891 | 0.9903 |
| `data/train_nodistill.jsonl` | 1,444 | 0.1891 | 1.0000 |
| `data/train_data25.jsonl` | 362 | 0.1796 | 0.9945 |

Each row stores `prompt`, `completion` and `meta` (`question_id`, `paper_id`,
`case`, `answer_type`, `branch`, `canary`). Full per-file I2 error details in
`reports/F1-train-stats.json`.

## F1.5 HOLD set (frozen 2026-11-20)

- 96 items (84 answerable, 12 unanswerable), 4 per paper across 24 post-cutoff
  papers, IDs `h01`-`h96`.
- Location: `evaluation/holdout/holdout_questions.jsonl`.
- Freeze hash (SHA-256): `43b7637e2841b81eef50e8745b7275f2abeadaec5af975c3423e631a5be6a8b3`
  (recorded in `evaluation/gates.json`).
- Retrievability: 68 of the 84 answerable items retrievable in the top-10
  (informational only; unretrievable items are kept).
- Verifier: `scripts/verify_holdout.py` passes 96/96 (exit 0).

## F1.6 PAPER16 and frozen contexts (2026-11-20)

### PAPER16
- 16 questions (`d01`-`d12`, `a01`-`a04`) vendored from Project 7 into
  `evaluation/paper16/questions.py`; web questions (`w01`-`w04`) excluded.
- Source paper: "Attention Is All You Need" (arXiv 1706.03762v7).
- Local copy: `B:\p8\data\paper16\attention.pages.json` (15 pages).
- PDF SHA-256: `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697`
  (verified against the P7 corpus manifest).

### Frozen retrieval contexts
Four files under `evaluation/contexts/` (chunk texts not stored; re-derived from
paper text at harness run time). Row format:
`{"question_id", "paper_id", "chunks": [{"chunk_id", "score", "page"}, ...]}`,
top-10 per question, own paper only, scores rounded to 6 decimals.

| File | Rows | SHA-256 |
|---|---|---|
| `contexts_qdev.jsonl` | 200 | `1e6a9acecb34900e46d62db27a5e10bb9bc9daf1eb45a14a44b0e7ffe9718ead` |
| `contexts_qtest.jsonl` | 400 | `867228f1d345327cb0d77cfdaafc955b0f9a24bd748895f144d1ea3b61948bac` |
| `contexts_hold.jsonl` | 96 | `4e51514b0b1dabace57b04a3b114eaf6b9abeca95b616f777624bd12377ce044` |
| `contexts_paper16.jsonl` | 16 | `d5ac1fe4a34af4f2478bdcea690d686a7607e7846d77f6c5e10574312be505d1` |

Hashes recorded in `reports/hashes.json`; build is byte-for-byte reproducible
(run twice, identical output).

## F1.7 leakage gate I1 (2026-11-20)

Gate I1 proves the training data cannot leak the evaluation sets. All three
checks pass (`scripts/check_leakage.py`, exit 0; report in
`reports/F1-leakage.md`):

1. **Paper-id overlap**: zero overlap among Q-TRAIN (888 papers), Q-DEV
   (144 papers) and Q-TEST (271 papers).
2. **Title overlap**: zero normalized-title overlap between the 1,585 QASPER
   papers and the 24 HOLD manifest titles or the Transformer paper title
   ("Attention Is All You Need").
3. **13-gram matches**: zero exact 13-gram matches between any train row
   (5,204,916 instances; 1,586,838 distinct) and any HOLD/PAPER16 question,
   gold answer or key fact (195 distinct eval 13-grams).
