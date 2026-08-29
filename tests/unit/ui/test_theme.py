"""Tests for the theme file format, its loader, and the shipped themes."""

from __future__ import annotations

import numpy as np
import pytest

from qsorbit.ui.theme import (
    CHROME_STYLES,
    DEFAULT_THEME_NAME,
    DEFAULT_THEMES_DIR,
    PALETTE_TOKENS,
    SHIPPED_THEME_SLUGS,
    THEME_FORMAT,
    Colormap,
    ThemeError,
    discover_themes,
    load_theme,
    theme_search_paths,
)

# ----------------------------------------------------------------------
# The shipped set
# ----------------------------------------------------------------------


def test_the_shipped_list_matches_what_is_actually_on_disk():
    """The named constant and the directory must not drift apart.

    ``SHIPPED_THEME_SLUGS`` is written out by hand so that quietly
    losing a theme from the package is a failure rather than a smaller
    parametrised loop. That only works if something checks the two
    agree -- otherwise the constant becomes a comment that happens to
    be executable.
    """
    on_disk = tuple(sorted(path.stem for path in DEFAULT_THEMES_DIR.glob("*.toml")))
    assert on_disk == SHIPPED_THEME_SLUGS


def test_every_shipped_theme_loads():
    found = discover_themes((DEFAULT_THEMES_DIR,))
    assert tuple(sorted(found)) == SHIPPED_THEME_SLUGS


def test_the_default_theme_is_one_of_them():
    assert DEFAULT_THEME_NAME in discover_themes((DEFAULT_THEMES_DIR,))


@pytest.mark.parametrize("slug", SHIPPED_THEME_SLUGS)
def test_shipped_theme_has_every_palette_token(slug):
    theme = load_theme(DEFAULT_THEMES_DIR / f"{slug}.toml")
    for token in PALETTE_TOKENS:
        assert theme.palette.rgb(token)


@pytest.mark.parametrize("slug", SHIPPED_THEME_SLUGS)
def test_shipped_theme_brightness_is_monotonic_across_its_ramp(slug):
    """'Brighter means stronger' has to hold everywhere, or the panel lies.

    This is Session 19's fixed-scale argument carried into theme data.
    It used to be asserted against the single module-level ramp; now it
    is asserted against all eight, because a user-visible theme that
    dips in the middle of its ramp would draw a band that looks like a
    feature and is not. Either direction is legal -- Daylight runs
    bright-to-dark so a carrier is the darkest thing on a white panel --
    but it must not turn around.
    """
    table = load_theme(DEFAULT_THEMES_DIR / f"{slug}.toml").waterfall.table
    luminance = 0.2126 * table[:, 0] + 0.7152 * table[:, 1] + 0.0722 * table[:, 2]
    steps = np.diff(luminance.astype(np.float64))
    rises = (steps >= -0.5).all()
    falls = (steps <= 0.5).all()
    assert rises or falls, f"{slug}'s ramp turns around"
    assert abs(luminance[-1] - luminance[0]) > 100.0, f"{slug}'s ramp has too little range"


@pytest.mark.parametrize("slug", SHIPPED_THEME_SLUGS)
def test_shipped_theme_chrome_style_is_implemented(slug):
    assert load_theme(DEFAULT_THEMES_DIR / f"{slug}.toml").chrome.style in CHROME_STYLES


def test_the_two_chrome_themes_are_the_ones_that_exercise_chrome():
    """LCARS and WOPR exist to prove [chrome] is load-bearing."""
    styles = {
        slug: load_theme(DEFAULT_THEMES_DIR / f"{slug}.toml").chrome.style
        for slug in SHIPPED_THEME_SLUGS
    }
    assert styles["lcars"] == "lcars"
    assert styles["wopr"] == "crt"
    assert {slug for slug, style in styles.items() if style != "default"} == {"lcars", "wopr"}


def test_night_ops_is_red_all_the_way_down():
    """The theme that makes semantic token names necessary.

    Its 'ok' token is a red, because every pixel has to preserve dark
    adaptation. A palette keyed on colour names could not express this
    without lying.
    """
    palette = load_theme(DEFAULT_THEMES_DIR / "night-ops.toml").palette
    for token in ("text", "accent", "ok", "warn", "alarm"):
        red, green, blue = palette.rgb(token)
        assert red > green and red > blue, f"night-ops {token} is not a red"


# ----------------------------------------------------------------------
# Loading a theme that did not ship with the app
# ----------------------------------------------------------------------


def test_a_theme_file_from_anywhere_loads(tmp_path, write_theme):
    """Chunk C's done-when: a theme not shipped with the app works."""
    theme = load_theme(write_theme("my-shack.toml"))
    assert theme.name == "Test"
    assert theme.slug == "my-shack"
    assert theme.source == tmp_path / "my-shack.toml"
    assert theme.palette.accent == "#00aaff"
    assert theme.waterfall.table.shape == (256, 3)


def test_a_user_theme_shadows_a_shipped_one_with_the_same_slug(
    tmp_path, minimal_theme, write_theme
):
    body = minimal_theme.replace('name = "Test"', 'name = "My Deep Space"')
    write_theme("deep-space.toml", body)
    found = discover_themes((DEFAULT_THEMES_DIR, tmp_path))
    assert found["deep-space"].name == "My Deep Space"
    assert found["deep-space"].source == tmp_path / "deep-space.toml"
    # And it shadows rather than replaces the set.
    assert tuple(sorted(found)) == SHIPPED_THEME_SLUGS


def test_discovery_skips_a_directory_that_does_not_exist(tmp_path):
    found = discover_themes((DEFAULT_THEMES_DIR, tmp_path / "nope"))
    assert tuple(sorted(found)) == SHIPPED_THEME_SLUGS


def test_discovery_skips_one_bad_file_rather_than_failing(tmp_path, write_theme):
    write_theme("good.toml")
    (tmp_path / "bad.toml").write_text('name = "Bad"\n', encoding="utf-8")
    found = discover_themes((tmp_path,))
    assert tuple(found) == ("good",)


def test_search_paths_put_the_user_directory_last():
    paths = theme_search_paths()
    assert paths[0] == DEFAULT_THEMES_DIR
    assert paths[-1].name == "themes"
    assert paths[-1] != DEFAULT_THEMES_DIR


# ----------------------------------------------------------------------
# Validation: missing, unknown, and malformed
# ----------------------------------------------------------------------


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(ThemeError, match="No theme file at"):
        load_theme(tmp_path / "absent.toml")


def test_invalid_toml_is_reported_as_such(tmp_path):
    path = tmp_path / "broken.toml"
    path.write_text("name = \n", encoding="utf-8")
    with pytest.raises(ThemeError, match="not valid TOML"):
        load_theme(path)


@pytest.mark.parametrize("token", PALETTE_TOKENS)
def test_a_missing_palette_token_is_an_error(tmp_path, token, minimal_theme, write_theme):
    body = "\n".join(
        line for line in minimal_theme.splitlines() if not line.startswith(f"{token} =")
    )
    with pytest.raises(ThemeError, match=f"Missing required key '{token}'"):
        load_theme(write_theme(body=body))


def test_an_unknown_palette_token_is_an_error(tmp_path, minimal_theme, write_theme):
    """A misspelling must not silently fall back to a default.

    ``acent`` quietly defaulting would give a theme that is *almost*
    right, which is far harder to spot by eye than one that refuses to
    load and names the key.
    """
    body = minimal_theme.replace('accent = "#00aaff"', 'accent = "#00aaff"\nacent = "#00aaff"')
    with pytest.raises(ThemeError, match="Unknown key in \\[palette\\].*acent"):
        load_theme(write_theme(body=body))


def test_an_unknown_top_level_key_is_an_error(tmp_path, minimal_theme, write_theme):
    body = minimal_theme + "\n[typography]\nsize = 12\n"
    with pytest.raises(ThemeError, match="Unknown key in \\[top level\\].*typography"):
        load_theme(write_theme(body=body))


@pytest.mark.parametrize("bad", ['"3fd0c9"', '"#3fd0c"', '"#gggggg"', "12", '"red"'])
def test_a_malformed_colour_is_an_error(tmp_path, bad, minimal_theme, write_theme):
    body = minimal_theme.replace('accent = "#00aaff"', f"accent = {bad}")
    with pytest.raises(ThemeError, match="must be a colour like"):
        load_theme(write_theme(body=body))


def test_a_missing_name_is_an_error(tmp_path, minimal_theme, write_theme):
    body = minimal_theme.replace('name = "Test"\n', "")
    with pytest.raises(ThemeError, match="Missing required key 'name'"):
        load_theme(write_theme(body=body))


def test_an_empty_name_is_an_error(tmp_path, minimal_theme, write_theme):
    body = minimal_theme.replace('name = "Test"', 'name = "   "')
    with pytest.raises(ThemeError, match="non-empty string"):
        load_theme(write_theme(body=body))


# ----------------------------------------------------------------------
# The colormap
# ----------------------------------------------------------------------


def test_stops_must_be_sorted_and_ascending(tmp_path, minimal_theme, write_theme):
    body = minimal_theme.replace(
        'stops = [[0.0, "#000000"], [1.0, "#ffffff"]]',
        'stops = [[0.0, "#000000"], [0.7, "#777777"], [0.3, "#333333"], [1.0, "#ffffff"]]',
    )
    with pytest.raises(ThemeError, match="sorted and strictly ascending"):
        load_theme(write_theme(body=body))


def test_stops_must_span_the_whole_range(tmp_path, minimal_theme, write_theme):
    body = minimal_theme.replace(
        'stops = [[0.0, "#000000"], [1.0, "#ffffff"]]',
        'stops = [[0.1, "#000000"], [0.9, "#ffffff"]]',
    )
    with pytest.raises(ThemeError, match="span the whole range"):
        load_theme(write_theme(body=body))


def test_a_position_outside_zero_to_one_is_an_error(tmp_path, minimal_theme, write_theme):
    body = minimal_theme.replace(
        'stops = [[0.0, "#000000"], [1.0, "#ffffff"]]',
        'stops = [[0.0, "#000000"], [1.5, "#ffffff"]]',
    )
    with pytest.raises(ThemeError, match="between 0 and 1"):
        load_theme(write_theme(body=body))


def test_a_single_stop_is_an_error(tmp_path, minimal_theme, write_theme):
    body = minimal_theme.replace(
        'stops = [[0.0, "#000000"], [1.0, "#ffffff"]]', 'stops = [[0.0, "#000000"]]'
    )
    with pytest.raises(ThemeError, match="at least two"):
        load_theme(write_theme(body=body))


def test_colormap_interpolates_between_its_stops():
    ramp = Colormap.from_stops(((0.0, "#000000"), (1.0, "#ffffff")))
    assert tuple(ramp.table[0]) == (0, 0, 0)
    assert tuple(ramp.table[255]) == (255, 255, 255)
    assert 120 <= int(ramp.table[128][0]) <= 136


def test_colormap_floor_colour_is_its_darkest_end():
    ramp = Colormap.from_stops(((0.0, "#0a0e14"), (1.0, "#ffffff")))
    assert ramp.floor_color == (10, 14, 20)


def test_a_colormap_can_sit_in_a_list_without_raising():
    """The Session 19 lesson, asserted rather than remembered.

    A dataclass carrying a numpy array with a generated ``__eq__``
    raises "the truth value of an array with more than one element is
    ambiguous" from innocuous places like ``x in some_list``. Colormap
    sets ``eq=False`` to avoid it, and this is the test that fails if
    someone removes that.
    """
    one = Colormap.from_stops(((0.0, "#000000"), (1.0, "#ffffff")))
    two = Colormap.from_stops(((0.0, "#111111"), (1.0, "#eeeeee")))
    assert one in [one, two]
    assert one not in [two]


# ----------------------------------------------------------------------
# Chrome
# ----------------------------------------------------------------------


def test_chrome_defaults_when_the_section_is_absent(tmp_path, write_theme):
    chrome = load_theme(write_theme()).chrome
    assert chrome.style == "default"
    assert chrome.font is None
    assert chrome.mono is None


def test_an_unimplemented_chrome_style_degrades_rather_than_failing(
    tmp_path, minimal_theme, write_theme
):
    """A shared theme from a newer QSOrbit still has usable colours.

    The asymmetry with the palette is the point: a palette typo is
    fatal because that token set is closed, and an unknown chrome style
    is not because that set grows. Refusing eleven good colours and a
    working colormap over a border treatment would be disproportionate,
    and it is exactly the version-skew case that makes a theme
    unshareable.
    """
    body = minimal_theme + '\n[chrome]\nstyle = "hologram"\n'
    theme = load_theme(write_theme(body=body))
    assert theme.chrome.style == "default"
    assert theme.chrome.declared_style == "hologram"
    assert theme.palette.accent == "#00aaff"


def test_an_unimplemented_chrome_style_is_reported_in_notes(tmp_path, minimal_theme, write_theme):
    """Degraded, but not silently -- otherwise someone spends an hour
    wondering why a downloaded theme looks nothing like its screenshot."""
    body = minimal_theme + '\n[chrome]\nstyle = "hologram"\n'
    theme = load_theme(write_theme(body=body))
    assert len(theme.notes) == 1
    assert "hologram" in theme.notes[0]
    assert "colours are unaffected" in theme.notes[0]


def test_a_theme_this_build_understands_has_no_notes(tmp_path, write_theme):
    assert load_theme(write_theme()).notes == ()


def test_a_non_string_chrome_style_is_still_an_error(tmp_path, minimal_theme, write_theme):
    body = minimal_theme + "\n[chrome]\nstyle = 7\n"
    with pytest.raises(ThemeError, match="must be a string naming a chrome style"):
        load_theme(write_theme(body=body))


# ----------------------------------------------------------------------
# Format and attribution -- the two things sharing needs to exist up front
# ----------------------------------------------------------------------


def test_format_defaults_to_the_current_one_when_absent(tmp_path, write_theme):
    assert load_theme(write_theme()).format == THEME_FORMAT


def test_a_theme_may_declare_the_format_it_was_written_for(tmp_path, minimal_theme, write_theme):
    body = f"format = {THEME_FORMAT}\n" + minimal_theme
    assert load_theme(write_theme(body=body)).format == THEME_FORMAT


def test_a_theme_from_a_newer_qsorbit_says_so(tmp_path, minimal_theme, write_theme):
    """The whole reason the key exists: tell "newer file" apart from "typo".

    Without it the first added token turns every shared theme into an
    unknown-key error that reads like a misspelling, and the author has
    no way to know their file is simply too new.
    """
    body = f"format = {THEME_FORMAT + 1}\n" + minimal_theme
    with pytest.raises(ThemeError, match="made for a newer QSOrbit"):
        load_theme(write_theme(body=body))


@pytest.mark.parametrize("bad", ["0", "-1", '"1"', "1.5", "true"])
def test_a_nonsense_format_is_an_error(tmp_path, bad, minimal_theme, write_theme):
    body = f"format = {bad}\n" + minimal_theme
    with pytest.raises(ThemeError, match="positive whole number"):
        load_theme(write_theme(body=body))


def test_attribution_is_optional(tmp_path, write_theme):
    theme = load_theme(write_theme())
    assert theme.author is None
    assert theme.description is None
    assert theme.url is None


def test_attribution_is_carried_when_present(tmp_path, minimal_theme, write_theme):
    body = (
        'author = "VE3XYZ"\n'
        'description = "Harbour at night"\n'
        'url = "https://example.invalid/themes"\n' + minimal_theme
    )
    theme = load_theme(write_theme(body=body))
    assert theme.author == "VE3XYZ"
    assert theme.description == "Harbour at night"
    assert theme.url == "https://example.invalid/themes"


@pytest.mark.parametrize("key", ["author", "description", "url"])
def test_an_empty_attribution_field_is_an_error(tmp_path, key, minimal_theme, write_theme):
    body = f'{key} = "  "\n' + minimal_theme
    with pytest.raises(ThemeError, match="non-empty string"):
        load_theme(write_theme(body=body))


@pytest.mark.parametrize("slug", SHIPPED_THEME_SLUGS)
def test_every_shipped_theme_is_attributed(slug):
    """The shipped files are the templates people copy, so they model
    what a publishable theme looks like rather than the bare minimum."""
    theme = load_theme(DEFAULT_THEMES_DIR / f"{slug}.toml")
    assert theme.author
    assert theme.description
    assert theme.format == THEME_FORMAT


def test_chrome_carries_font_families(tmp_path, minimal_theme, write_theme):
    body = (
        minimal_theme
        + '\n[chrome]\nstyle = "crt"\nfont = "Share Tech Mono"\nmono = "Share Tech Mono"\n'
    )
    chrome = load_theme(write_theme(body=body)).chrome
    assert chrome.style == "crt"
    assert chrome.font == "Share Tech Mono"
    assert chrome.mono == "Share Tech Mono"


def test_an_unknown_chrome_key_is_an_error(tmp_path, minimal_theme, write_theme):
    body = minimal_theme + "\n[chrome]\nscanlines = true\n"
    with pytest.raises(ThemeError, match="Unknown key in \\[chrome\\].*scanlines"):
        load_theme(write_theme(body=body))
