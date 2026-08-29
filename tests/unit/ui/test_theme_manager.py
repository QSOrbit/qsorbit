"""Tests for applying a theme to a live QApplication.

These are the first tests in the project that construct Qt objects. Every
previous Qt chunk was written blind because PySide6 was not installable
in the session sandbox; it is now, and it runs under the ``offscreen``
platform plugin. What that buys is real: construction, wiring, signal
delivery, stylesheet and palette application, and teardown all execute
here. What it does not buy is anything visual -- offscreen has no window
manager, no real DPI and not the operator's font stack -- so "it looks
right" remains a bench check on a real monitor and is not claimed by
anything in this file.
"""

from __future__ import annotations

import pytest

# A submodule, not the package: `import PySide6` succeeds on a machine
# with no Qt system libraries, and only `PySide6.QtWidgets` fails --
# with an ImportError for libEGL.so.1 rather than anything mentioning
# Qt. Guarding the package alone let CI die at collection.
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtGui import QPalette  # noqa: E402

from qsorbit.ui.theme import (  # noqa: E402
    DEFAULT_THEME_NAME,
    DEFAULT_THEMES_DIR,
    SHIPPED_THEME_SLUGS,
    ThemeError,
    discover_themes,
    load_theme,
)
from qsorbit.ui.theme_manager import (  # noqa: E402
    ThemeManager,
    build_qpalette,
    register_bundled_fonts,
)


@pytest.fixture
def manager(qapp):
    return ThemeManager(discover_themes((DEFAULT_THEMES_DIR,)))


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_discover_finds_the_shipped_themes(qapp):
    assert ThemeManager.discover((DEFAULT_THEMES_DIR,)).slugs == SHIPPED_THEME_SLUGS


def test_it_starts_on_the_default_theme(manager):
    assert manager.current.slug == DEFAULT_THEME_NAME


def test_an_unknown_starting_slug_falls_back_to_the_default(qapp):
    themes = discover_themes((DEFAULT_THEMES_DIR,))
    assert ThemeManager(themes, "no-such-theme").current.slug == DEFAULT_THEME_NAME


def test_a_manager_with_no_themes_refuses_to_exist(qapp):
    with pytest.raises(ValueError, match="at least one theme"):
        ThemeManager({})


def test_discover_on_an_empty_directory_is_a_broken_install(qapp, tmp_path):
    with pytest.raises(ThemeError, match="install is incomplete"):
        ThemeManager.discover((tmp_path,))


def test_asking_for_an_unknown_theme_names_the_ones_that_exist(manager):
    with pytest.raises(KeyError, match="deep-space"):
        manager.theme("hologram")


# ----------------------------------------------------------------------
# Applying
# ----------------------------------------------------------------------


@pytest.mark.parametrize("slug", SHIPPED_THEME_SLUGS)
def test_applying_a_theme_puts_its_colours_on_the_application(manager, qapp, slug):
    theme = manager.apply(slug)
    assert theme.slug == slug
    assert theme.palette.bg in qapp.styleSheet()
    assert qapp.palette().color(QPalette.ColorRole.Window).name() == theme.palette.bg


def test_switching_theme_replaces_the_previous_stylesheet(manager, qapp):
    """Not appends to it -- an accumulating stylesheet would keep the
    old theme's rules live and produce a screen wearing both."""
    manager.apply("deep-space")
    dark_bg = manager.current.palette.bg
    manager.apply("daylight")
    sheet = qapp.styleSheet()
    assert manager.current.palette.bg in sheet
    assert dark_bg not in sheet


def test_applying_emits_the_change_with_the_new_theme(manager):
    """The signal is how custom-painted widgets learn to repaint."""
    seen = []
    manager.changed.connect(seen.append)
    manager.apply("mars")
    manager.apply("luna")
    assert [theme.slug for theme in seen] == ["mars", "luna"]


def test_applying_with_no_slug_reapplies_the_current_theme(manager, qapp):
    manager.apply("earth")
    seen = []
    manager.changed.connect(seen.append)
    assert manager.apply().slug == "earth"
    assert [theme.slug for theme in seen] == ["earth"]


def test_a_failed_apply_leaves_the_current_theme_alone(manager, qapp):
    manager.apply("night-ops")
    with pytest.raises(KeyError):
        manager.apply("hologram")
    assert manager.current.slug == "night-ops"
    assert manager.current.palette.bg in qapp.styleSheet()


def test_the_disabled_group_is_dimmer_than_the_enabled_one(manager):
    """Branch B's meters grey out until Chunk E, so 'disabled' carries
    information rather than decoration and has to actually read as it."""
    theme = manager.theme("deep-space")
    palette = build_qpalette(theme)
    enabled = palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.WindowText)
    disabled = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText)
    assert enabled.name() != disabled.name()
    assert disabled.name() == theme.palette.dim


# ----------------------------------------------------------------------
# A theme that did not ship with the app
# ----------------------------------------------------------------------


def test_a_theme_file_from_anywhere_can_be_loaded_and_applied(
    manager, qapp, tmp_path, minimal_theme
):
    """Chunk C's done-when, end to end on a live application."""
    path = tmp_path / "my-shack.toml"
    path.write_text(
        minimal_theme.replace('accent = "#00aaff"', 'accent = "#abcdef"'), encoding="utf-8"
    )

    theme = manager.load_and_add(path)
    assert "my-shack" in manager.slugs

    manager.apply("my-shack")
    assert qapp.palette().color(QPalette.ColorRole.Highlight).name() == "#abcdef"
    assert "#abcdef" in qapp.styleSheet()
    assert theme.source == path


def test_adding_a_theme_does_not_disturb_the_current_one(manager, tmp_path, minimal_theme):
    manager.apply("luna")
    path = tmp_path / "extra.toml"
    path.write_text(minimal_theme, encoding="utf-8")
    manager.load_and_add(path)
    assert manager.current.slug == "luna"


# ----------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------


def test_the_bundled_fonts_register_under_the_names_the_themes_ask_for(qapp):
    """LCARS names Antonio and WOPR names Share Tech Mono. A typo in
    either theme file would silently fall back to the platform font and
    the chrome would look wrong rather than fail, so the join between
    the file and the font is worth asserting."""
    families = register_bundled_fonts()
    assert "Antonio" in families
    assert "Share Tech Mono" in families
    assert load_theme(DEFAULT_THEMES_DIR / "lcars.toml").chrome.font in families
    assert load_theme(DEFAULT_THEMES_DIR / "wopr.toml").chrome.font in families


def test_registering_twice_is_harmless(qapp):
    assert register_bundled_fonts() == register_bundled_fonts()


def test_switching_away_from_a_theme_restores_the_default_typeface(manager, qapp):
    """A theme switch has to be symmetric.

    This failed twice when it was first written, and both failures were
    the same shape. ``apply`` first only touched the application font
    when the incoming theme *named* a family, so LCARS -> Deep Space
    left Antonio on screen under Deep Space's colours. Storing a "base
    font" to restore then failed differently: a manager constructed
    after some other theme had been applied captured *that* theme's
    typeface as its baseline. Typography moved into the stylesheet
    instead, which is replaced wholesale on every apply and so cannot
    accumulate. Caught by this test rather than by looking at it, which
    is the first time in this project a Qt behaviour could be caught
    that way at all.
    """
    manager.apply("deep-space")
    assert "Antonio" not in qapp.styleSheet()

    manager.apply("lcars")
    assert "Antonio" in qapp.styleSheet()

    manager.apply("deep-space")
    assert "Antonio" not in qapp.styleSheet()


# ----------------------------------------------------------------------
# The bench affordance: Ctrl+T cycles themes in a live window
# ----------------------------------------------------------------------


def test_the_window_cycles_forward_and_wraps(manager, qapp):
    """A bench shortcut, but the wrap is the part that would bite.

    Off-by-one on the last theme would either raise or stick, and both
    would be discovered at the bench in the dark rather than here.
    """
    from qsorbit.ui.instrument_window import InstrumentWindow

    window = InstrumentWindow(themes=manager)
    manager.apply(manager.slugs[0])

    seen = [manager.current.slug]
    for _ in range(len(manager.slugs)):
        window._cycle_theme(1)
        seen.append(manager.current.slug)

    assert seen[:-1] == list(manager.slugs)
    assert seen[-1] == manager.slugs[0], "cycling past the last theme must wrap to the first"
    window.close()


def test_the_window_cycles_backward(manager, qapp):
    from qsorbit.ui.instrument_window import InstrumentWindow

    window = InstrumentWindow(themes=manager)
    manager.apply(manager.slugs[0])
    window._cycle_theme(-1)
    assert manager.current.slug == manager.slugs[-1]
    window.close()


def test_cycling_actually_restyles_the_application(manager, qapp):
    from qsorbit.ui.instrument_window import InstrumentWindow

    window = InstrumentWindow(themes=manager)
    manager.apply("deep-space")
    before = qapp.styleSheet()
    window._cycle_theme(1)
    assert qapp.styleSheet() != before
    assert manager.current.palette.bg in qapp.styleSheet()
    window.close()


def test_a_window_without_a_theme_manager_still_works(qapp):
    """The manager is optional, so the existing callers that pass none
    must not gain a shortcut that would dereference it."""
    from qsorbit.ui.instrument_window import InstrumentWindow

    window = InstrumentWindow()
    window._cycle_theme(1)  # must not raise
    window.close()
