"""Tests for the Custom tab's config file format and its loader.

No Qt anywhere in this file, on purpose -- :mod:`qsorbit.ui.custom_tab`
is importable without PySide6, the same split
:mod:`qsorbit.ui.theme`/:mod:`qsorbit.ui.theme_manager` settled at PR1,
so the format can be tested on a machine with no display at all. Tests
that build an actual :class:`~qsorbit.ui.tabs.CustomTab` widget belong
in ``test_shell_window.py`` and ``test_tab_layout.py``, where the rest
of the Qt-backed shell tests already live.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qsorbit.ui.custom_tab import (
    CONFIG_FILENAME,
    KNOWN_WIDGETS,
    CustomTabConfig,
    CustomTabConfigError,
    custom_tab_config_path,
    load_custom_tab_config,
)

#: A minimal valid config, as TOML text. Module-level rather than a
#: fixture because several tests below build variants of it with
#: ``str.replace``, mirroring ``test_theme.py``'s own ``MINIMAL_THEME``.
MINIMAL_CONFIG = """
layout = "grid"
columns = 2
widgets = ["waterfall", "rotor_readout"]
"""


@pytest.fixture
def write_custom_tab(tmp_path: Path):
    """Write a ``custom_tab.toml`` body into ``tmp_path`` and return its path."""

    def _write(name: str = CONFIG_FILENAME, body: str = MINIMAL_CONFIG) -> Path:
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        return path

    return _write


# ----------------------------------------------------------------------
# Where the file lives
# ----------------------------------------------------------------------


def test_the_config_path_is_beside_the_themes_directory():
    """Same resolution as ``app_themes_dir`` -- see the module docstring.

    Both are ``user_config_dir()`` plus a fixed name, not tied to
    wherever ``--config`` actually pointed station config at.
    """
    from qsorbit.core.station import user_config_dir
    from qsorbit.ui.theme import app_themes_dir

    assert custom_tab_config_path().parent == app_themes_dir().parent == user_config_dir()
    assert custom_tab_config_path().name == "custom_tab.toml"


# ----------------------------------------------------------------------
# A valid file
# ----------------------------------------------------------------------


def test_a_valid_config_loads(write_custom_tab):
    config = load_custom_tab_config(write_custom_tab())
    assert config == CustomTabConfig(columns=2, widgets=("waterfall", "rotor_readout"))


def test_widget_names_can_repeat(write_custom_tab):
    """Repeats are the whole point of a Custom tab, not a mistake to reject.

    Making the *feed* name unique is :meth:`~qsorbit.ui.feed_hub.FeedHub.spectrum`'s
    job, at claim time -- this loader only has to agree that the widget
    list itself may repeat a name.
    """
    path = write_custom_tab(
        body='layout = "grid"\ncolumns = 2\nwidgets = ["waterfall", "waterfall", "waterfall"]\n'
    )
    config = load_custom_tab_config(path)
    assert config.widgets == ("waterfall", "waterfall", "waterfall")


def test_the_shipped_example_file_loads():
    """The example in the repo root must stay valid, or it is a bad example.

    Sibling of ``test_theme.py``'s shipped-set check: a constant or a
    sample file that nothing verifies against the real schema is a
    comment that happens to look like data.
    """
    example = Path(__file__).parents[3] / "custom_tab.example.toml"
    assert example.is_file(), f"custom_tab.example.toml not found at {example}"
    config = load_custom_tab_config(example)
    assert config.widgets
    assert all(name in KNOWN_WIDGETS for name in config.widgets)


# ----------------------------------------------------------------------
# Missing or unreadable
# ----------------------------------------------------------------------


def test_missing_file_names_the_path(tmp_path):
    missing = tmp_path / "custom_tab.toml"
    with pytest.raises(CustomTabConfigError, match="No custom tab config at"):
        load_custom_tab_config(missing)


def test_invalid_toml_is_reported_as_such(write_custom_tab):
    path = write_custom_tab(body="this is not [ valid toml")
    with pytest.raises(CustomTabConfigError, match="not valid TOML"):
        load_custom_tab_config(path)


# ----------------------------------------------------------------------
# Schema errors -- each one names the file and the key at fault
# ----------------------------------------------------------------------


def test_an_unknown_top_level_key_is_an_error(write_custom_tab):
    path = write_custom_tab(body=MINIMAL_CONFIG + "rows = 3\n")
    with pytest.raises(CustomTabConfigError, match=r"Unknown key.*\[top level\]"):
        load_custom_tab_config(path)


def test_a_missing_key_is_an_error(write_custom_tab):
    path = write_custom_tab(body='layout = "grid"\ncolumns = 2\n')
    with pytest.raises(CustomTabConfigError, match="Missing required key 'widgets'"):
        load_custom_tab_config(path)


def test_layout_must_be_grid(write_custom_tab):
    path = write_custom_tab(body=MINIMAL_CONFIG.replace('layout = "grid"', 'layout = "rows"'))
    with pytest.raises(CustomTabConfigError, match="must be one of 'grid'"):
        load_custom_tab_config(path)


@pytest.mark.parametrize(
    "bad",
    [
        "columns = 0",
        "columns = -1",
        "columns = 1.5",
        'columns = "2"',
        # bool is a subclass of int in Python, and `true` is not a
        # column count -- the same guard station.py's _as_float carries
        # for the identical reason.
        "columns = true",
    ],
)
def test_columns_must_be_a_positive_integer(write_custom_tab, bad):
    path = write_custom_tab(body=MINIMAL_CONFIG.replace("columns = 2", bad))
    with pytest.raises(CustomTabConfigError, match="must be a positive integer"):
        load_custom_tab_config(path)


def test_widgets_must_be_a_non_empty_array(write_custom_tab):
    path = write_custom_tab(body='layout = "grid"\ncolumns = 2\nwidgets = []\n')
    with pytest.raises(CustomTabConfigError, match="non-empty array"):
        load_custom_tab_config(path)


def test_a_non_string_widget_entry_is_an_error(write_custom_tab):
    path = write_custom_tab(body='layout = "grid"\ncolumns = 2\nwidgets = [3]\n')
    with pytest.raises(CustomTabConfigError, match=r"widgets\[0\].*must be a string"):
        load_custom_tab_config(path)


def test_an_unknown_widget_name_is_an_error(write_custom_tab):
    path = write_custom_tab(body='layout = "grid"\ncolumns = 2\nwidgets = ["next_pass"]\n')
    with pytest.raises(CustomTabConfigError, match="does not know how to draw"):
        load_custom_tab_config(path)


@pytest.mark.parametrize("name", sorted(KNOWN_WIDGETS))
def test_every_known_widget_name_loads_on_its_own(write_custom_tab, name):
    path = write_custom_tab(body=f'layout = "grid"\ncolumns = 1\nwidgets = ["{name}"]\n')
    config = load_custom_tab_config(path)
    assert config.widgets == (name,)
