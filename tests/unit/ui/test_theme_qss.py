"""Tests for stylesheet generation.

The load-bearing one is
:func:`test_the_stylesheet_contains_no_colour_that_is_not_a_theme_token`.
Phase 3's standing rule is that no widget hardcodes a colour, ever, and
this module is the single place in the app whose whole job is emitting
colours -- so it is the one place where that rule can be checked
mechanically instead of by review.
"""

from __future__ import annotations

import re

import pytest

from qsorbit.ui.theme import DEFAULT_THEMES_DIR, SHIPPED_THEME_SLUGS, load_theme
from qsorbit.ui.theme_qss import (
    DEFAULT_MONO_STACK,
    build_stylesheet,
    chrome_structure,
    mono_families,
    palette_roles,
)

_HEX_IN_CSS = re.compile(r"#[0-9a-fA-F]{3,8}")


def theme(slug):
    return load_theme(DEFAULT_THEMES_DIR / f"{slug}.toml")


@pytest.mark.parametrize("slug", SHIPPED_THEME_SLUGS)
def test_the_stylesheet_contains_no_colour_that_is_not_a_theme_token(slug):
    """The standing rule, enforced by arithmetic rather than by review."""
    subject = theme(slug)
    tokens = {
        getattr(subject.palette, name).lower() for name in subject.palette.__dataclass_fields__
    }
    found = {match.lower() for match in _HEX_IN_CSS.findall(build_stylesheet(subject))}
    assert found <= tokens, f"{slug} stylesheet has non-token colours: {sorted(found - tokens)}"


@pytest.mark.parametrize("slug", SHIPPED_THEME_SLUGS)
def test_every_theme_produces_a_stylesheet_naming_its_own_tokens(slug):
    subject = theme(slug)
    sheet = build_stylesheet(subject)
    # The four tokens that must reach the screen for the app to be legible.
    for token in ("bg", "text", "accent", "panel"):
        assert getattr(subject.palette, token) in sheet


@pytest.mark.parametrize("slug", SHIPPED_THEME_SLUGS)
def test_the_stylesheet_styles_the_widgets_the_shell_will_actually_use(slug):
    sheet = build_stylesheet(theme(slug))
    for selector in ("QTabBar::tab", "QWidget#Card", "QComboBox", "QPushButton", "QToolTip"):
        assert selector in sheet


@pytest.mark.parametrize("slug", SHIPPED_THEME_SLUGS)
def test_palette_roles_cover_what_custom_painted_widgets_read(slug):
    """``self.palette().windowText().color()`` has to resolve to a theme colour."""
    roles = palette_roles(theme(slug))
    for role in ("WindowText", "Window", "Base", "Highlight", "Text"):
        assert role in roles
        assert roles[role].startswith("#")


def test_a_theme_names_its_own_selected_tab_in_its_accent():
    subject = theme("deep-space")
    sheet = build_stylesheet(subject)
    selected = sheet.split("QTabBar::tab:selected")[1].split("}")[0]
    assert subject.palette.accent in selected


# ----------------------------------------------------------------------
# Chrome
# ----------------------------------------------------------------------


def test_default_chrome_asks_the_shell_for_nothing_extra():
    structure = chrome_structure(theme("deep-space"))
    assert not structure.accent_bars
    assert not structure.scanlines
    assert not structure.glow


def test_lcars_asks_for_accent_bars():
    structure = chrome_structure(theme("lcars"))
    assert structure.accent_bars
    assert structure.uppercase_headings
    assert not structure.scanlines


def test_crt_asks_for_scanlines_and_glow():
    structure = chrome_structure(theme("wopr"))
    assert structure.scanlines
    assert structure.glow
    assert not structure.accent_bars


def test_lcars_tabs_are_pills_rather_than_underlines():
    """LCARS gets its structure from filled shapes, not from line borders."""
    sheet = build_stylesheet(theme("lcars"))
    tab_rule = sheet.split("QTabBar::tab {")[1].split("}")[0]
    assert "border-radius: 13px" in tab_rule
    assert "border: none" in tab_rule
    assert "border-bottom: 2px solid" not in tab_rule


def test_default_tabs_are_underlines_rather_than_pills():
    sheet = build_stylesheet(theme("deep-space"))
    tab_rule = sheet.split("QTabBar::tab {")[1].split("}")[0]
    assert "border-bottom: 2px solid transparent" in tab_rule


def test_crt_squares_its_corners():
    sheet = build_stylesheet(theme("wopr"))
    card_rule = sheet.split("QWidget#Card {")[1].split("}")[0]
    assert "border-radius: 0px" in card_rule


# ----------------------------------------------------------------------
# Typography
# ----------------------------------------------------------------------


def test_a_theme_without_a_mono_family_gets_the_default_stack():
    assert mono_families(theme("deep-space")) == DEFAULT_MONO_STACK


def test_a_theme_with_a_mono_family_gets_it_first_and_keeps_the_fallbacks():
    families = mono_families(theme("wopr"))
    assert families[0] == "Share Tech Mono"
    assert families[1:] == DEFAULT_MONO_STACK


def test_the_ui_font_reaches_the_stylesheet_when_a_theme_names_one():
    assert "Antonio" in build_stylesheet(theme("lcars"))
