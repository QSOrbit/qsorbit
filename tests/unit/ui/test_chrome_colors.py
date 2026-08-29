"""Tests for the derived colours the structural chrome paints with.

These live in :mod:`qsorbit.ui.theme_manager` because that is the module
whose job is turning tokens into colours, and it is the one place in
``ui/`` allowed to name one. What is checked here is that everything it
derives still comes *from* the theme -- the standing Phase 3 rule read
one step further than "no literals in widget code", because a colour
computed from a token is only honest if the computation follows the
token when the theme changes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from qsorbit.ui.theme import DEFAULT_THEMES_DIR, SHIPPED_THEME_SLUGS, load_theme  # noqa: E402
from qsorbit.ui.theme_manager import (  # noqa: E402
    ACCENT_BAR_HUE_OFFSETS,
    accent_bar_color,
    glow_color,
    scanline_color,
    theme_color,
)


@pytest.fixture(scope="module")
def lcars():
    return load_theme(DEFAULT_THEMES_DIR / "lcars.toml")


def themes():
    return [load_theme(DEFAULT_THEMES_DIR / f"{slug}.toml") for slug in sorted(SHIPPED_THEME_SLUGS)]


# ----------------------------------------------------------------------
# Accent bars
# ----------------------------------------------------------------------


def test_the_first_bar_is_the_theme_s_own_accent(lcars):
    assert accent_bar_color(lcars, 0).name() == theme_color(lcars, "accent").name()


def test_the_bars_cycle_and_are_distinct(lcars):
    colors = [accent_bar_color(lcars, i) for i in range(len(ACCENT_BAR_HUE_OFFSETS))]
    assert len({c.name() for c in colors}) == len(colors)
    # The cycle wraps rather than running out.
    assert accent_bar_color(lcars, len(colors)).name() == colors[0].name()


def test_the_bar_sequence_reproduces_the_mockup_s_hues(lcars):
    """The order matters and is easy to get backwards.

    The mockup's ``nth-child(3n+2)`` rule puts the periwinkle on the
    *second* card and the mauve on the third. Swapping the two offsets
    passes every other test in this file and is obvious the moment two
    cards sit next to each other -- which is exactly how it was caught,
    by rendering the Radio tab rather than by reading the constant.
    """
    hues = [accent_bar_color(lcars, i).hue() for i in range(3)]
    assert hues[0] == pytest.approx(theme_color(lcars, "accent").hue(), abs=1)
    assert hues[1] == pytest.approx(240, abs=6)  # periwinkle, the mockup's #9c9cff
    assert hues[2] == pytest.approx(300, abs=6)  # mauve, the mockup's #cc99cc


@pytest.mark.parametrize("theme", themes(), ids=lambda t: t.slug)
def test_every_theme_s_bars_stay_in_its_own_key(theme):
    """A user theme picking ``style = "lcars"`` must not get Star Trek.

    Saturation and value are taken from that theme's accent, so the
    bars are recognisably the same palette rather than three hues
    imported from somewhere else.
    """
    accent = theme_color(theme, "accent")
    for index in range(len(ACCENT_BAR_HUE_OFFSETS)):
        bar = accent_bar_color(theme, index)
        assert bar.saturation() == accent.saturation()
        assert bar.value() == accent.value()


def test_a_greyscale_accent_degrades_to_itself_rather_than_inventing_hues(lcars):
    """A monochrome theme has no hue to rotate, and should stay monochrome.

    Rotating the hue of a grey produces an arbitrary colour, which for a
    palette that deliberately had none is worse than three identical
    bars.
    """
    from dataclasses import replace

    grey = replace(lcars, palette=replace(lcars.palette, accent="#888888"))
    colors = {accent_bar_color(grey, i).name() for i in range(3)}
    assert colors == {theme_color(grey, "accent").name()}


# ----------------------------------------------------------------------
# CRT chrome
# ----------------------------------------------------------------------


@pytest.mark.parametrize("theme", themes(), ids=lambda t: t.slug)
def test_a_scanline_is_the_theme_s_background_at_partial_alpha(theme):
    """It darkens what is under it rather than tinting it.

    A scanline is an absence of phosphor, not a colour of its own -- so
    it is drawn from ``bg``, which also means a light theme picking the
    CRT chrome scans lighter rather than laying black bars over white.
    """
    line = scanline_color(theme)
    assert (line.red(), line.green(), line.blue()) == theme.palette.rgb("bg")
    assert 0 < line.alpha() < 255


@pytest.mark.parametrize("theme", themes(), ids=lambda t: t.slug)
def test_the_glow_is_the_theme_s_accent_at_partial_alpha(theme):
    glow = glow_color(theme)
    assert (glow.red(), glow.green(), glow.blue()) == theme.palette.rgb("accent")
    assert 0 < glow.alpha() < 255


def test_deriving_a_colour_does_not_mutate_the_theme(lcars):
    """``QColor.setAlpha`` mutates in place, and ``theme_color`` returns a fresh one.

    If it ever returned a shared instance, asking for a scanline would
    quietly leave the accent semi-transparent everywhere else -- a
    one-way override, which is the failure shape this project has
    already met once in the theme system.
    """
    before = theme_color(lcars, "accent").alpha()
    glow_color(lcars)
    scanline_color(lcars)
    assert theme_color(lcars, "accent").alpha() == before == 255
