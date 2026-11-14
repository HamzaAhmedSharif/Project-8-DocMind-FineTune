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
| test | 416 | not pinned |

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
their annotator answers. Test-split questions keep an empty answer list.
