#!/usr/bin/env python3
# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Run leakage gate I1 (F1.7).

Prints each check's result with counts and writes ``reports/F1-leakage.md``.
Exits 0 only if all three checks pass; on any failure exits 1 and lists every
violation.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import docmind_ft.vendor  # noqa: F401, E402  # installs the docmind_service alias

from docmind_ft.leakage import run_all_checks  # noqa: E402

REPORTS_DIR = REPO_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "F1-leakage.md"


def _write_report(results: list[dict]) -> None:
    lines: list[str] = [
        "# Leakage gate I1 (F1.7)",
        "",
        "Proves the training data cannot leak the evaluation sets.  Three "
        "independent checks; all must pass.",
        "",
    ]
    all_pass = True
    for result in results:
        passed = result["passed"]
        all_pass = all_pass and passed
        lines.append(f"## {result['name']}: {'PASS' if passed else 'FAIL'}")
        lines.append("")
        lines.append("| Count | Value |")
        lines.append("|-------|-------|")
        for key, value in result["counts"].items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        if result["violations"]:
            lines.append("Violations:")
            lines.append("")
            for violation in result["violations"]:
                lines.append(f"- {violation}")
        else:
            lines.append("No violations.")
        lines.append("")
    lines.append(f"## Overall: {'PASS' if all_pass else 'FAIL'}")
    lines.append("")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    results = run_all_checks()
    _write_report(results)
    all_pass = True
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        all_pass = all_pass and result["passed"]
        print(f"[{status}] {result['name']}: {result['counts']}")
        for violation in result["violations"]:
            print(f"  violation: {violation}")
    if not all_pass:
        print("I1 gate FAILED", file=sys.stderr)
        return 1
    print("I1 gate PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
