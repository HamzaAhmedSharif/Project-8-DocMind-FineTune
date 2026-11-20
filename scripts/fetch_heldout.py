#!/usr/bin/env python3
"""Fetch and extract the 24 post-cutoff HOLD-set papers (F1.5).

The PDFs come from the Project 7 held-out manifest.  Every download is
verified against the manifest SHA-256; a mismatching file is deleted and
retried once, then the export.arxiv.org URL is tried as a fallback.  A PDF
already on disk with a matching hash is not downloaded again.  Extracted
pages are written as a JSON list of page strings under
``B:\\p8\\data\\holdout\\``; the PDFs and page files are never committed.
A paper that still cannot be fetched or extracted is reported as BLOCKED,
written to ``evaluation/holdout/extract_hashes.json`` with its reason, and
the script exits 1 without substituting another paper.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MANIFEST_PATH = Path(
    r"A:\HDD\Work\VS Code - Python\GitHub\Project-7-DocMind-Service"
    r"\evaluation\heldout\manifest.json"
)
DATA_DIR = Path(r"B:\p8\data\holdout")
REPORT_PATH = REPO_ROOT / "evaluation" / "holdout" / "extract_hashes.json"
FALLBACK_BASE = "https://export.arxiv.org/pdf"
ATTEMPTS = 3
BACKOFF_SECONDS = 5
PAGE_MARGIN = 10


class FetchError(Exception):
    """A paper could not be fetched and verified."""


class DownloadError(FetchError):
    """A URL failed after its network retries."""


class ChecksumError(FetchError):
    """A downloaded file did not match the manifest SHA-256."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("papers"), list):
        raise ValueError(f"{path}: expected a JSON object with a papers list")
    papers = data["papers"]
    if "count" in data and data["count"] != len(papers):
        raise ValueError(f"{path}: count {data['count']} != {len(papers)} papers")
    required = ("arxiv_id", "title", "primary_category", "pdf_url", "sha256", "pages")
    seen: set[str] = set()
    for index, paper in enumerate(papers):
        if not isinstance(paper, dict):
            raise ValueError(f"{path}: papers[{index}] is not an object")
        missing = [key for key in required if key not in paper]
        if missing:
            raise ValueError(f"{path}: papers[{index}] missing {', '.join(missing)}")
        if not isinstance(paper["arxiv_id"], str) or not paper["arxiv_id"]:
            raise ValueError(f"{path}: papers[{index}].arxiv_id must be a non-empty string")
        if paper["arxiv_id"] in seen:
            raise ValueError(f"{path}: duplicate arxiv_id {paper['arxiv_id']!r}")
        seen.add(paper["arxiv_id"])
        for key in ("title", "primary_category", "pdf_url"):
            if not isinstance(paper[key], str) or not paper[key]:
                raise ValueError(f"{path}: papers[{index}].{key} must be a non-empty string")
        if not isinstance(paper["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", paper["sha256"]
        ):
            raise ValueError(f"{path}: papers[{index}].sha256 must be lowercase hex")
        if (
            not isinstance(paper["pages"], int)
            or isinstance(paper["pages"], bool)
            or paper["pages"] < 1
        ):
            raise ValueError(f"{path}: papers[{index}].pages must be a positive integer")
    return papers


def _download_once(url: str, part: Path, get) -> None:
    with get(url, stream=True, timeout=(15, 300)) as response:
        response.raise_for_status()
        with part.open("wb") as handle:
            for block in response.iter_content(1 << 20):
                if block:
                    handle.write(block)


def _download_url(
    url: str,
    expected_sha: str,
    dest: Path,
    *,
    get,
    sleep,
    attempts: int = ATTEMPTS,
    backoff: int = BACKOFF_SECONDS,
) -> None:
    part = dest.with_name(dest.name + ".part")
    network_failures = 0
    mismatch_retries = 0
    while True:
        part.unlink(missing_ok=True)
        try:
            print(f"fetch   {dest.stem} {url} (network attempt {network_failures + 1}/{attempts})")
            _download_once(url, part, get)
        except (requests.RequestException, OSError) as exc:
            network_failures += 1
            part.unlink(missing_ok=True)
            if network_failures >= attempts:
                raise DownloadError(f"{url}: {exc}") from exc
            print(f"retry   {dest.stem}: {exc}")
            sleep(backoff * network_failures)
            continue
        actual = file_sha256(part)
        if actual == expected_sha:
            part.replace(dest)
            print(f"ok      {dest.stem} sha256 {actual}")
            return
        part.unlink(missing_ok=True)
        if mismatch_retries < 1:
            mismatch_retries += 1
            print(
                f"retry   {dest.stem}: checksum mismatch "
                f"(expected {expected_sha}, got {actual}); retrying once"
            )
            continue
        raise ChecksumError(f"{url}: expected {expected_sha}, got {actual}")


def fetch_pdf(
    paper: dict,
    dest_dir: Path = DATA_DIR,
    *,
    get=requests.get,
    sleep=time.sleep,
) -> tuple[Path, str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{paper['arxiv_id']}.pdf"
    expected = paper["sha256"]
    if dest.exists():
        actual = file_sha256(dest)
        if actual == expected:
            print(f"cached  {paper['arxiv_id']}")
            return dest, "cached"
        print(f"discard {paper['arxiv_id']}: cached checksum mismatch")
        dest.unlink(missing_ok=True)

    urls = [paper["pdf_url"]]
    fallback = f"{FALLBACK_BASE}/{paper['arxiv_id']}"
    if fallback not in urls:
        urls.append(fallback)
    errors: list[str] = []
    for url in urls:
        try:
            _download_url(url, expected, dest, get=get, sleep=sleep)
            return dest, url
        except FetchError as exc:
            print(f"error   {paper['arxiv_id']}: {exc}")
            errors.append(str(exc))
    raise FetchError("; ".join(errors))


def _record(paper: dict, pages: list[str], pages_bytes: bytes) -> dict:
    return {
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "primary_category": paper["primary_category"],
        "pdf_sha256": paper["sha256"],
        "text_sha256": sha256_bytes(pages_bytes),
        "pages_manifest": paper["pages"],
        "pages_extracted": len(pages),
        "extractable_chars": sum(len(page) for page in pages),
        "status": "ok",
    }


def _blocked_record(paper: dict, reason: str, pdf_sha256: str | None = None) -> dict:
    return {
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "primary_category": paper["primary_category"],
        "pdf_sha256": pdf_sha256,
        "text_sha256": None,
        "pages_manifest": paper["pages"],
        "pages_extracted": None,
        "extractable_chars": None,
        "status": "BLOCKED",
        "reason": reason,
    }


def _write_report(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _default_extract(data: bytes, *, max_pages: int) -> list[str]:
    import docmind_ft.vendor  # noqa: F401, I001  (installs the docmind_service alias)
    from docmind_service.ingest import extract_pages

    return extract_pages(data, max_pages=max_pages)


def run(
    manifest_path: Path = MANIFEST_PATH,
    dest_dir: Path = DATA_DIR,
    report_path: Path = REPORT_PATH,
    *,
    get=requests.get,
    sleep=time.sleep,
    extract=None,
) -> int:
    papers = load_manifest(manifest_path)
    if not papers:
        raise ValueError(f"{manifest_path}: no papers")
    if extract is None:
        extract = _default_extract
    dest_dir.mkdir(parents=True, exist_ok=True)
    max_pages = max(paper["pages"] for paper in papers) + PAGE_MARGIN
    records: list[dict] = []

    for paper in papers:
        paper_id = paper["arxiv_id"]
        try:
            pdf_path, source = fetch_pdf(paper, dest_dir, get=get, sleep=sleep)
        except FetchError as exc:
            reason = f"fetch failed: {exc}"
            records.append(_blocked_record(paper, reason))
            _write_report(report_path, records)
            print(f"BLOCKED {paper_id}: {reason}")
            return 1

        try:
            pages = extract(pdf_path.read_bytes(), max_pages=max_pages)
        except Exception as exc:  # noqa: BLE001  (any extractor failure blocks this paper)
            reason = f"extraction failed: {exc}"
            records.append(_blocked_record(paper, reason, pdf_sha256=paper["sha256"]))
            _write_report(report_path, records)
            print(f"BLOCKED {paper_id}: {reason}")
            return 1
        if not isinstance(pages, list) or not all(isinstance(page, str) for page in pages):
            reason = "extraction failed: extractor did not return a list of strings"
            records.append(_blocked_record(paper, reason, pdf_sha256=paper["sha256"]))
            _write_report(report_path, records)
            print(f"BLOCKED {paper_id}: {reason}")
            return 1

        pages_bytes = (json.dumps(pages, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        (dest_dir / f"{paper_id}.pages.json").write_bytes(pages_bytes)
        record = _record(paper, pages, pages_bytes)
        records.append(record)
        _write_report(report_path, records)

        source_note = "cached" if source == "cached" else "downloaded"
        print(
            f"ok      {paper_id} pages {len(pages)}/{paper['pages']} "
            f"chars {record['extractable_chars']} ({source_note})"
        )
        if len(pages) != paper["pages"]:
            print(
                f"warn    {paper_id}: extracted {len(pages)} pages, manifest says {paper['pages']}"
            )

    _write_report(report_path, records)
    print(f"papers ok {len(records)}/{len(papers)}")
    return 0


def main() -> int:
    try:
        return run()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
