"""Shared fixture-loading helpers for DSP tests.

Fixtures under ``tests/fixtures/iq/`` are not committed (see the README in
that directory) — a fresh clone has none, and every test that wants one
must skip cleanly rather than fail, the same convention the hardware
integration tests use for "no device attached".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "iq"


@pytest.fixture
def load_iq_fixture():
    """A callable fixture: ``load_iq_fixture("wbfm-99.9")`` -> ``(raw, metadata)``.

    A fixture factory rather than a plain importable function, since this
    project's test directories are not Python packages (no
    ``__init__.py``) — pytest's own fixture injection is the supported way
    to share helpers across test files here, not relative imports.

    Skips the test (via :func:`pytest.skip`) if either the ``.iq`` or the
    ``.json`` sidecar is missing, per ``tests/fixtures/iq/README.md`` —
    expected on a fresh clone, not a failure.
    """

    def _load(name: str) -> tuple[bytes, dict]:
        iq_path = FIXTURES_DIR / f"{name}.iq"
        sidecar_path = FIXTURES_DIR / f"{name}.json"
        if not iq_path.exists() or not sidecar_path.exists():
            pytest.skip(f"IQ fixture {name!r} not present locally; see tests/fixtures/iq/README.md")
        raw = iq_path.read_bytes()
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return raw, metadata

    return _load
