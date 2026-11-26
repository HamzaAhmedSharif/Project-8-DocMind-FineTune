# Evaluation (F5)

Fine-tuned model FT-main (QLoRA, Llama 3.2 3B, 2560-token cutoff, 1,318 rows,
180 steps) evaluated once on the sealed sets Q-TEST (400), HOLD (96),
PAPER16 (16). Baselines B0 (base model, B0 prompt) and B1 (base model, frozen
v3 prompt); reference R = Groq `openai/gpt-oss-120b` (HOLD/PAPER16 only).
Metrics are section 3.8; intervals are 95% paper-cluster bootstrap
(resample papers, seed 20261207, 10,000 resamples). Full tables in
`evaluation/results/test_metrics.json`; hypotheses in
`evaluation/results/hypotheses.json`.

## (a) Per-type scores (Q-TEST answer_f1)

| type | FT-main | B1 | B0 |
|---|---|---|---|
| extractive | 0.359 | 0.335 | 0.228 |
| free_form | 0.218 | 0.222 | 0.155 |
| yes_no | 0.342 | 0.280 | 0.247 |
| overall | 0.338 | 0.311 | 0.222 |

FT-main improves extractive (+0.024) and yes/no (+0.062) over B1, but is
slightly worse on free-form (-0.004). The overall gain (+0.027) is below the
pre-registered HY2 bar of +0.05 and its bootstrap interval
[-0.004, 0.058] includes zero.

## (b) Failure taxonomy (FT-main, seeded sample, seed 20261207)

Q-TEST: 281 of 400 questions count as failures under the strict definition
(answer_f1 < 0.5, or an unanswerable question not abstained on, or a format
error). Counts in the sampled 30: over_refusal 47, other 21,
citation_error 66, wrong_answer 118, retrieval_miss 28, format_error 1
(counts are over all 281 failures, not just the sample).

HOLD: 28 of 96 fail (a key fact missed, a format error, or an over-refusal).
Counts: citation_error 11, wrong_answer 3, over_refusal 5, retrieval_miss 7,
other 2.

### Q-TEST examples (ids + quotes of at most 25 words)

- retrieval_miss — `4f57ac24f3f4689a2f885715cd84b7d867fe3f12` (f1 1.0):
  "Yes, they compare their system's output against a small set of
  automatically generated emails provided by the authors of BIBREF6 [c6]."
- retrieval_miss — `2ba2ff6c21a16bd295b07af1ef635b3b4c5bd17e` (f1 0.19):
  "The baselines used include a single-channel beamforming followed by
  speaker diarization and speech recognition, and a single-channel
  beamforming followed by speaker diarization and speech recognition"
- retrieval_miss — `67672648e7ebcbef18921006e2c8787966f8cdf2` (f1 0.29):
  "The competing objectives for their unsupervised method include maximizing
  content fidelity and minimizing style deviation [c10]."
- wrong_answer — `4c7ec282697f4f6646eb1c19f46bbaf8670b0de6` (f1 0.42):
  "The weak supervision signal used in the Baidu Baike corpus is the title
  of each sample as a pseudo label [c1]."
- wrong_answer — `8c46a26f9b0b41c656b5b55142d491600663defa` (f1 0.43):
  "The dataset used includes the GlobalPhone corpus of speech read from
  news articles BIBREF20, the Buckeye corpus BIBREF38, and the NCHLT
  corpus BIBREF39 [c4]."
- wrong_answer — `08c0d4db14773cbed8a63e69381a2265e85f8765` (f1 0.10):
  "Professional profiles are crawled from LinkedIn, as the study mentions
  that they used LinkedIn BIBREF30 for collecting professional profile
  descriptions [c1]."
- over_refusal — `3614c1f1435b7c1fd1f7f0041219eebf5bcff473` (f1 0.0):
  "I could not find relevant information in the indexed documents."
- over_refusal — `952fe4fbf4e0bcfcf44fab2dbd3ed85dd961eff3` (f1 0.0):
  "I could not find relevant information in the indexed documents."
- over_refusal — `6c8dc31a199b155e73c84173816c1e252137a0af` (f1 0.0):
  "I could not find relevant information in the indexed documents."
- citation_error — `8ffae517bc0efa453b7e316d41bd9f1b6679b158` (f1 0.22):
  "Applications in Overton are presented as a set of tasks and data
  payloads, where the tasks capture the model's objectives and the
  payloads represent the data sources [c6]."
- citation_error — `b21f61c0f95fefdb1bdb90d51cbba4655cd59896` (f1 0.0):
  "Yes, the nonsymbolic representation can represent two same strings
  differently depending on the context."
- citation_error — `d3cfbe497a30b750a8de3ea7f2cecf4753a4e1f9` (f1 0.33):
  "The word embeddings utilized for the EmoContext task are fasttext and
  ELMo [c6]."
- format_error — `4c40fa01f626def0b69d1cb7bf9181b574ff6382` (f1 0.24):
  "The datasets used in this work include data-wiki-att, data-wiki-agg,
  data-fox-news, data-bully, data-gazzetta, data-twitter-wh,
  data-twitter-f, data-yahoo-fin-dj, data-yahoo-fin-dj, data-yahoo-fin-dj,"
- other — `fcb0ac1934e2fd9f58f4b459e6853999a27844f9` (f1 0.0):
  "I could not find relevant information in the indexed documents."
- other — `740cc392c0c8bfadfe6b3a60c0be635c03e17f2a` (f1 0.0):
  "I could not find relevant information in the indexed documents."
- other — `a7e03d24549961b38e15b5386d9df267900ef4c8` (f1 0.0):
  "I could not find relevant information in the indexed documents."

### HOLD examples (ids + quotes of at most 25 words)

- citation_error — `h94` (keyfact cov 0.0):
  "The number of independent ERA5 reanalysis realizations used is 10 [c1]."
- citation_error — `h07` (keyfact cov 0.0):
  "Hierarchical sentence representations provide less benefit once
  turn-level context is available because the contextual signal within a
  short window is already exhausted"
- citation_error — `h75` (keyfact cov 0.0):
  "Hindsight return relabeling destabilizes importance-sampling
  reinforcement learning for Decision Transformers because it introduces a
  mismatch in the conditioning variable g"
- wrong_answer — see `evaluation/results/error_analysis.json` for the
  sampled wrong_answer HOLD items (3 items).
- over_refusal — `h74` (keyfact cov 0.0):
  "I could not find relevant information in the indexed documents."
- over_refusal — `h88` (keyfact cov 0.0):
  "I could not find relevant information in the indexed documents."
- over_refusal — `h73` (keyfact cov 0.0):
  "I could not find relevant information in the indexed documents."
- retrieval_miss — `h43` (keyfact cov 0.0):
  "The CoT method outperforms the embedding benchmark for e-commerce query
  categorization because it can detect problems within a hierarchical
  taxonomy"
- other — `h78` (keyfact cov 0.0):
  "I could not find relevant information in the indexed documents."
- other — `h60` (keyfact cov 0.0):
  "I could not find relevant information in the indexed documents."

Note: the "other" Q-TEST cases are unanswerable questions where the model
abstained but the abstention was not counted as correct by the strict
definition (the gold answerable flag disagrees with the context). The single
Q-TEST format_error is a runaway list that exceeded the 120-word cap.

## (c) Five largest B1-beats-FT items (Q-TEST answer_f1 gap)

| question_id | B1 f1 | FT-main f1 | gap |
|---|---|---|---|
| dfd07a8e2de80c3a8d075a0f400fb13a1f1d4c60 | 1.0 | 0.0 | 1.0 |
| d788076c0d19781ff3f6525bd9c05b0ef0ecd0f1 | 1.0 | 0.0 | 1.0 |
| c4a3f270e942803dab9b40e5e871a2e8886ce444 | 1.0 | 0.0 | 1.0 |
| bc7081aaa207de2362e0bea7bc8108d338aee36f | 1.0 | 0.0 | 1.0 |
| ae2142ee9e093ce485025168f4bcb3da4602739d | 1.0 | 0.0 | 1.0 |

All five are questions where B1 answered exactly right and FT-main scored
zero — in each case FT-main abstained or gave an unrelated answer. These are
the over-refusal cost of the fine-tune in its sharpest form.

## (d) Answer-length statistics and what they imply for F1

| model | Q-TEST mean words | HOLD mean words |
|---|---|---|
| FT-main | 24.28 | 26.23 |
| B1 | 31.22 | 26.90 |
| B0 | 95.47 | — |

FT-main answers are shorter than B1's on Q-TEST (24 vs 31 words) and far
shorter than B0's (95 words). Shorter answers help format_valid (the 120-word
cap is rarely approached) and reduce the chance of an unsupported claim, but
they also risk dropping a second required fact on multi-part questions — the
free-form type, where FT-main (0.218) is marginally below B1 (0.222), is the
clearest symptom. The extractive and yes/no gains show the model learned to
copy the right span and to commit to a polarity; the free-form plateau
suggests the length pressure trades against completeness.

## (e) Distillation and data-size effects — EXPLORATORY (unavailable)

The pre-registered exploratory comparisons FT-main vs FT-nodistill
(distillation effect) and FT-main vs FT-data25 (data-size effect) could not
be computed: FT-nodistill and FT-data25 were never trained (no such adapters
exist in this project; only FT-main was trained in F4). These comparisons are
recorded as unavailable, not estimated. No claim is made about the
distillation or data-size effect.

## (f) Gap to the reference R on HOLD

| model | keyfact_cov (HOLD) | groundable | format_valid | citation_support |
|---|---|---|---|---|
| FT-main | 0.671 | 0.727 (56 of 77) | 1.000 | 0.675 |
| B1 | 0.706 | 0.779 (60 of 77) | 0.917 | 0.536 |
| R (120B) | 0.576 | 0.636 (49 of 77) | 0.115 | 0.085 |

On HOLD keyfact coverage the 120B reference (0.576) is BELOW both B1 (0.706)
and FT-main (0.671). The reference does not follow the citation/format
contract (format_valid 0.115, citation_support 0.085), so its keyfact
coverage is not directly comparable as a quality ceiling: it answers freely
without grounding. The gap-closure figure is therefore not meaningful here
(FT-main closed 27% of the B1→R gap only because R sits below B1). The honest
reading: on this held-out set the fine-tuned 3B model covers more key facts
than the 120B reference while following the format, and B1 covers the most
key facts of the three.

On PAPER16, FT-main and B1 tie at keyfact_cov 0.769 (groundable 0.864);
R is 0.731 with format_valid 0.125.

## (g) QASPER contamination risk and the small held-out set

Q-TEST is drawn from QASPER test (400 questions). Llama 3.2's pre-training
data is not public, so contamination cannot be ruled out: a famous paper or
a widely quoted passage could be answered from model memory rather than from
the retrieved context. Two mitigations apply. First, the HOLD set (96
questions on post-cutoff papers) is the contamination-free check: FT-main
covers 0.671 of its key facts vs B1's 0.706, i.e. the fine-tune does not beat
the prompt baseline where memory cannot help, which is consistent with the
Q-TEST gain being partly a format/citation effect rather than new knowledge.
Second, the citation_support metric (0.645 on Q-TEST for FT-main vs 0.443
for B1) measures grounding in the provided context, not memory.

The held-out sets are small (HOLD 96, PAPER16 16). PAPER16's 16 questions
come from a single famous paper, so its numbers are indicative, not precise:
a single question swings PAPER16 keyfact_cov by ~0.04. HOLD's 96 questions
across 24 papers give a coarser but more trustworthy read. All intervals in
test_metrics.json are paper-cluster bootstrap intervals, which account for
the within-paper correlation.

## (h) Key-fact tables

### HOLD (96 questions, 85 key facts, groundable ceiling 77)

| model | keyfact_cov | keyfact_cov_groundable | memory_bleed | format_valid |
|---|---|---|---|---|
| FT-main | 0.671 (57/85) | 0.727 (56 of 77 reachable) | 1 | 1.000 |
| B1 | 0.706 (60/85) | 0.779 (60 of 77 reachable) | 0 | 0.917 |
| R | 0.576 (49/85) | 0.636 (49 of 77 reachable) | 0 | 0.115 |

### PAPER16 (16 questions, 26 key facts, groundable ceiling 22)

| model | keyfact_cov | keyfact_cov_groundable | memory_bleed | format_valid |
|---|---|---|---|---|
| FT-main | 0.769 (20/26) | 0.864 (19 of 22 reachable) | 1 | 1.000 |
| B1 | 0.769 (20/26) | 0.864 (19 of 22 reachable) | 1 | 0.938 |
| R | 0.731 (19/26) | 0.864 (19 of 22 reachable) | 0 | 0.125 |

`memory_bleed` counts answers that state a key fact not present in the
provided context (FT-main: 1 on HOLD, 1 on PAPER16; B1: 0 on HOLD, 1 on
PAPER16). The single FT-main HOLD bleed is a fact the model knew from
pre-training and stated without a supporting chunk.
