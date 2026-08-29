"""Shared fixtures for the UI tests.

Everything shared between the UI test modules lives here rather than in
one of them, because **test modules in this tree cannot import each
other.** There is no ``__init__.py`` anywhere under ``tests/`` (see
:mod:`tests.conftest` and the Session 20 note about pytest deriving
module names from basenames), so ``tests`` is not an importable package.
A ``from tests.unit.ui.test_theme import ...`` appears to work under
``python -m pytest``, which puts the working directory on ``sys.path``,
and fails under ``uv run pytest``, which does not. conftest is the
mechanism pytest provides for exactly this, and it has no such problem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qsorbit.ui.theme import DEFAULT_THEMES_DIR, Colormap, Theme, load_theme


@pytest.fixture(scope="session")
def deep_space() -> Theme:
    """The shipped default theme, loaded from its real file.

    Deliberately the real shipped file rather than a synthetic fixture.
    A synthetic ramp that models the wrong thing flatters everything
    checked against it -- the Session 20 lesson -- and the rendering
    assertions here (a carrier reads brighter than its noise floor) are
    only worth anything against a ramp somebody will actually look at.
    """
    return load_theme(DEFAULT_THEMES_DIR / "deep-space.toml")


@pytest.fixture(scope="session")
def colormap(deep_space: Theme) -> Colormap:
    """The default theme's waterfall ramp."""
    return deep_space.waterfall


#: A valid theme file with nothing optional in it, as TOML text.
#: Module-level rather than a fixture because several tests build
#: variants of it with ``str.replace`` before writing them out.
MINIMAL_THEME = """
name = "Test"
[palette]
bg = "#000000"
panel = "#111111"
panel_alt = "#222222"
inset = "#010101"
edge = "#333333"
text = "#eeeeee"
dim = "#888888"
accent = "#00aaff"
ok = "#00ff00"
warn = "#ffaa00"
alarm = "#ff0000"
[waterfall]
stops = [[0.0, "#000000"], [1.0, "#ffffff"]]
"""


@pytest.fixture
def minimal_theme() -> str:
    """The body of a valid, minimal theme file."""
    return MINIMAL_THEME


@pytest.fixture
def write_theme(tmp_path):
    """Write a theme file into ``tmp_path`` and return its path."""

    def _write(name: str = "custom.toml", body: str = MINIMAL_THEME) -> Path:
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        return path

    return _write
