"""Tests for the HOLD-set fetch script (F1.5); no network is used."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from scripts.fetch_heldout import (
    FetchError,
    fetch_pdf,
    load_manifest,
    run,
    sha256_bytes,
)

PDF = b"%PDF-1.4\nfake holdout pdf\n"


class FakeResponse:
    def __init__(self, data: bytes):
        self.data = data

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self.data), chunk_size):
            yield self.data[start : start + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


class FakeGet:
    def __init__(self, handler):
        self.handler = handler
        self.calls: list[str] = []

    def __call__(self, url: str, **kwargs):
        self.calls.append(url)
        result = self.handler(url, len(self.calls))
        if isinstance(result, BaseException):
            raise result
        return FakeResponse(result)


def _paper(arxiv_id: str = "2601.00001", *, pages: int = 2) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "primary_category": "cs.CL",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "sha256": sha256_bytes(PDF),
        "pages": pages,
    }


def _write_manifest(path: Path, papers: list[dict]) -> Path:
    path.write_text(
        json.dumps({"count": len(papers), "papers": papers}),
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_manifest_has_24_papers_with_required_fields(tmp_path: Path) -> None:
    papers = [_paper(f"2601.{i:05d}") for i in range(24)]
    path = _write_manifest(tmp_path / "manifest.json", papers)

    loaded = load_manifest(path)

    assert len(loaded) == 24
    required = {"arxiv_id", "title", "primary_category", "pdf_url", "sha256", "pages"}
    assert all(required <= set(paper) for paper in loaded)
    assert len({paper["arxiv_id"] for paper in loaded}) == 24
    assert all(len(paper["sha256"]) == 64 for paper in loaded)


def test_manifest_rejects_missing_field(tmp_path: Path) -> None:
    paper = _paper()
    del paper["sha256"]
    path = _write_manifest(tmp_path / "manifest.json", [paper])

    with pytest.raises(ValueError, match="missing sha256"):
        load_manifest(path)


def test_sha256_good_and_mismatched_bytes() -> None:
    good = "6bcb93b0fa5c83237bf84b213879823c66d44d67bb123f3ac1218417bf18ed30"
    assert sha256_bytes(PDF) != good
    assert sha256_bytes(PDF) == sha256_bytes(PDF)
    assert sha256_bytes(PDF) != sha256_bytes(PDF + b"changed")


def test_resume_skips_a_pdf_with_matching_sha(tmp_path: Path) -> None:
    paper = _paper()
    (tmp_path / f"{paper['arxiv_id']}.pdf").write_bytes(PDF)

    def should_not_be_called(url: str, call_number: int):
        raise AssertionError("network must not be used for a cached PDF")

    get = FakeGet(should_not_be_called)
    path, source = fetch_pdf(paper, tmp_path, get=get, sleep=lambda _: None)

    assert path.read_bytes() == PDF
    assert source == "cached"
    assert get.calls == []


def test_download_retries_once_after_a_checksum_mismatch(tmp_path: Path) -> None:
    paper = _paper()
    get = FakeGet(lambda url, call_number: b"wrong" if call_number == 1 else PDF)

    path, source = fetch_pdf(paper, tmp_path, get=get, sleep=lambda _: None)

    assert path.read_bytes() == PDF
    assert source == paper["pdf_url"]
    assert len(get.calls) == 2
    assert not (tmp_path / f"{paper['arxiv_id']}.pdf.part").exists()


def test_network_retries_then_falls_back_to_export(tmp_path: Path) -> None:
    paper = _paper()
    get = FakeGet(lambda url, call_number: requests.ConnectionError("offline"))

    with pytest.raises(FetchError, match="offline"):
        fetch_pdf(paper, tmp_path, get=get, sleep=lambda _: None)

    assert len(get.calls) == 6
    assert get.calls[:3] == [paper["pdf_url"]] * 3
    assert get.calls[3:] == [f"https://export.arxiv.org/pdf/{paper['arxiv_id']}"] * 3
    assert not (tmp_path / f"{paper['arxiv_id']}.pdf.part").exists()


def test_run_writes_pages_and_report(tmp_path: Path) -> None:
    paper = _paper()
    manifest = _write_manifest(tmp_path / "manifest.json", [paper])
    get = FakeGet(lambda url, call_number: PDF)
    seen: dict[str, object] = {}

    def extract(data: bytes, *, max_pages: int) -> list[str]:
        seen["data"] = data
        seen["max_pages"] = max_pages
        return ["page one text", "page two text"]

    code = run(
        manifest,
        tmp_path / "holdout",
        tmp_path / "extract_hashes.json",
        get=get,
        sleep=lambda _: None,
        extract=extract,
    )

    assert code == 0
    assert seen == {"data": PDF, "max_pages": paper["pages"] + 10}
    pages = json.loads(
        (tmp_path / "holdout" / f"{paper['arxiv_id']}.pages.json").read_text(encoding="utf-8")
    )
    assert pages == ["page one text", "page two text"]
    report = json.loads((tmp_path / "extract_hashes.json").read_text(encoding="utf-8"))
    assert len(report) == 1
    assert report[0]["status"] == "ok"
    assert report[0]["pdf_sha256"] == paper["sha256"]
    assert report[0]["pages_manifest"] == 2
    assert report[0]["pages_extracted"] == 2
    assert report[0]["extractable_chars"] == len("page one text") + len("page two text")
    assert (tmp_path / "holdout" / f"{paper['arxiv_id']}.pdf").read_bytes() == PDF


def test_blocked_paper_is_written_and_exits_one(tmp_path: Path) -> None:
    paper = _paper()
    manifest = _write_manifest(tmp_path / "manifest.json", [paper])
    get = FakeGet(lambda url, call_number: requests.ConnectionError("offline"))
    extract_called = False

    def extract(data: bytes, *, max_pages: int) -> list[str]:
        nonlocal extract_called
        extract_called = True
        return []

    code = run(
        manifest,
        tmp_path / "holdout",
        tmp_path / "extract_hashes.json",
        get=get,
        sleep=lambda _: None,
        extract=extract,
    )

    assert code == 1
    assert extract_called is False
    report = json.loads((tmp_path / "extract_hashes.json").read_text(encoding="utf-8"))
    assert len(report) == 1
    assert report[0]["status"] == "BLOCKED"
    assert report[0]["pdf_sha256"] is None
    assert report[0]["text_sha256"] is None
    assert "fetch failed" in report[0]["reason"]


def test_extraction_failure_is_blocked_and_keeps_the_verified_pdf(tmp_path: Path) -> None:
    paper = _paper()
    manifest = _write_manifest(tmp_path / "manifest.json", [paper])
    get = FakeGet(lambda url, call_number: PDF)

    def extract(data: bytes, *, max_pages: int) -> list[str]:
        raise RuntimeError("bad pdf")

    code = run(
        manifest,
        tmp_path / "holdout",
        tmp_path / "extract_hashes.json",
        get=get,
        sleep=lambda _: None,
        extract=extract,
    )

    assert code == 1
    report = json.loads((tmp_path / "extract_hashes.json").read_text(encoding="utf-8"))
    assert report[0]["status"] == "BLOCKED"
    assert report[0]["pdf_sha256"] == paper["sha256"]
    assert "extraction failed: bad pdf" in report[0]["reason"]
    assert (tmp_path / "holdout" / f"{paper['arxiv_id']}.pdf").exists()
