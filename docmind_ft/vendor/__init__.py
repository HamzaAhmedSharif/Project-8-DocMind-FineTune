"""Vendored Project 7 modules (read-only copies).

Copied from the pinned Project 7 commit by scripts/vendor_p7.py; provenance
(commit, per-file SHA-256, date) in PROVENANCE.json. Never edit these files
here — they are verbatim copies, checked by scripts/check_vendor.py.

Import rule (1.3 decision, see DECISIONS.md): the vendored service modules
import each other as ``from docmind_service... import ...``. Rather than
editing the byte-exact copies, this package registers itself under the
name ``docmind_service`` in ``sys.modules``, so those imports resolve to
the vendored files. Import the service modules as ``docmind_service.<name>``
(after importing this package once); import the standalone evaluation
modules as ``docmind_ft.vendor.<name>``. Never import the same file under
both names — the two module objects would be distinct.
"""

import datetime as _datetime
import sys as _sys

# Python 3.10 compatibility: vendored ledger.py does
# ``from datetime import UTC, datetime``, but ``datetime.UTC`` exists only
# from 3.11. The clone env is 3.10.21, so backport the name before any
# vendored module is imported (this __init__ always runs first).
# ``datetime.timezone.utc`` is the identical object ``UTC`` aliases in 3.11+.
if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc

_sys.modules.setdefault("docmind_service", _sys.modules[__name__])
