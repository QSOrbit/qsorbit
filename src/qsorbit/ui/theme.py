"""Themes as files: palette tokens, a waterfall colormap, and optional chrome.

No Qt import anywhere in here, for the same reason
:mod:`qsorbit.ui.waterfall_render` has none: every decision worth
arguing about lives in plain data and plain functions that can be read
and tested without PySide6 installed. :mod:`qsorbit.ui.theme_qss` turns
a theme into a stylesheet, also without Qt; only
:mod:`qsorbit.ui.theme_manager` needs the toolkit, and it is the thin
remainder that applies the result and tells widgets to repaint.

**A theme is one small TOML file, and shipped themes and user themes are
the same mechanism.** The eight themes QSOrbit ships live beside this
module in ``themes/``; a user drops their own into
``<app dir>/themes/`` -- the same directory family as ``config.toml``,
resolved by :func:`~qsorbit.core.station.user_config_dir` -- and it loads
with no privilege the shipped ones do not also have. A user file wins a
name collision, which is what makes "edit a shipped theme" work without
editing anything inside the installed package.

**Validation is strict where the set is closed and forgiving where it
grows**, and that split is what makes a theme safe to share. A
misspelled ``acent`` is an error, mirroring :mod:`qsorbit.core.station`
and :mod:`qsorbit.core.profiles.catalog`: the palette's token set is
fixed, so an unrecognised key there is almost certainly a typo, and a
theme that is *almost* right is far worse to diagnose by eye than one
that refuses to load and names the key.

A theme is not a station config, though. A config file is yours, never
leaves your machine, and only ever meets the version of QSOrbit you are
running. **A theme is meant to be shared**, so it travels between
versions in both directions, and strict-on-everything would turn each
future addition into a compatibility break: add a twelfth token and
every theme in circulation stops loading. So the file carries an
optional ``format`` (see :data:`THEME_FORMAT`), the extensible
``[chrome]`` set degrades rather than raising, and the attribution keys
exist from the first release rather than being added once sharing is
real -- because under strict validation, adding a field later is exactly
the break this is here to avoid.

Two things about the token vocabulary are worth more than their line
count.

**The palette names what a token is for, not what colour it is.** The
reference mockup carried both spellings and they disagreed: its live CSS
used ``green``/``amber``/``red`` while its documented file format used
``ok``/``warn``/``alarm``. The semantic names win, and the deciding
evidence is in the theme data itself -- Night Ops sets its "ok" colour
to ``#d93a30``, which is a red, because the whole theme is red to
preserve dark adaptation. A token named ``green`` holding a red is a lie
that every future widget author has to read past. ``ok`` stays true in
all eight themes.

**The waterfall colormap is part of the theme, not part of the
renderer.** It used to be a module-level table in
:mod:`qsorbit.ui.waterfall_render`, built once at import from a private
ramp -- which meant no amount of restyling could change it without a
restart, and two panels could never differ. It is theme data now, and
:class:`Colormap` is what carries it.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

#: The themes shipped with QSOrbit, beside this module.
DEFAULT_THEMES_DIR: Final[Path] = Path(__file__).parent / "themes"

#: The theme applied when nothing else is asked for.
DEFAULT_THEME_NAME: Final[str] = "deep-space"

#: The theme file format this build reads. A theme may declare its own
#: ``format``; absent means 1. **This exists because themes are meant to
#: be shared**, and a shared file travels between versions of QSOrbit in
#: both directions. Without it, the first time a token is added, every
#: theme in circulation either breaks or forces validation to be
#: loosened permanently. With it, a file from a newer build is told
#: apart from a file with a typo in it, and gets a message someone can
#: act on.
#:
#: Bump this only when the format changes in a way an older build cannot
#: read. Adding a new *optional* key does not need a bump.
THEME_FORMAT: Final[int] = 1

#: The themes this build ships, by slug. Named here rather than globbed
#: from the directory so that quietly losing one from the package is a
#: failure rather than a smaller loop -- :func:`discover_themes` is what
#: reads the directory, and a test asserts the two agree exactly.
#:
#: In the package rather than in a test module because it is real
#: product information (the ``--theme`` help text and, later, the theme
#: picker both want it) and because test modules cannot import each
#: other: there is no ``__init__.py`` under ``tests/``, so ``tests`` is
#: not a package, and a cross-test import only appears to work under
#: ``python -m pytest``, which puts the working directory on
#: ``sys.path``. Under ``uv run pytest`` it does not.
SHIPPED_THEME_SLUGS: Final[tuple[str, ...]] = (
    "daylight",
    "deep-space",
    "earth",
    "lcars",
    "luna",
    "mars",
    "night-ops",
    "wopr",
)

#: Chrome styles the shell knows how to draw. A theme picks one and
#: recolours it; it cannot invent a new one, because chrome is code
#: (widget structure and custom painting) rather than data.
CHROME_STYLES: Final[frozenset[str]] = frozenset({"default", "lcars", "crt"})

#: Palette token names, in the order a theme file conventionally lists
#: them. Every one is required: a theme with a hole in it would fall
#: back to some other theme's colour for that token, which is precisely
#: the "almost right" failure strict validation exists to prevent.
PALETTE_TOKENS: Final[tuple[str, ...]] = (
    "bg",
    "panel",
    "panel_alt",
    "inset",
    "edge",
    "text",
    "dim",
    "accent",
    "ok",
    "warn",
    "alarm",
)

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


class ThemeError(Exception):
    """Raised when a theme file is missing, malformed, or incomplete.

    The message always names the file and the key at fault, for the same
    reason :class:`~qsorbit.core.station.ConfigError`'s docstring gives:
    it is read by someone editing a text file by hand, quite possibly
    outdoors, quite possibly in the dark.
    """


def parse_hex_color(
    value: Any, key: str, section: str, path: Path | None = None
) -> tuple[int, int, int]:
    """Parse ``"#rrggbb"`` into an ``(r, g, b)`` triple of ``0..255``.

    Raises:
        ThemeError: If the value is not a six-digit hex colour string.
    """
    where = f" of {path}" if path is not None else ""
    if not isinstance(value, str) or not _HEX_COLOR.match(value):
        raise ThemeError(
            f"'{key}' in [{section}]{where} must be a colour like \"#3fd0c9\", got {value!r}."
        )
    return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))


@dataclass(frozen=True)
class Palette:
    """The eleven colour tokens every widget styles itself from.

    Stored as the original ``"#rrggbb"`` strings rather than parsed
    triples because that is the form both consumers want: Qt stylesheets
    take them verbatim, and :meth:`rgb` is there for the custom-painted
    widgets that need numbers.

    Args:
        bg: The window's own background, behind every panel.
        panel: A card or panel surface sitting on ``bg``.
        panel_alt: A raised surface inside a panel -- combo boxes,
            buttons, spin boxes. Distinct from ``panel`` so an
            interactive control reads as interactive.
        inset: A recessed well: the waterfall canvas, a meter trough, a
            code block. Usually the darkest token in a dark theme and
            the lightest in a light one.
        edge: Borders and separators.
        text: Primary foreground text.
        dim: Secondary text, labels, axis ticks -- readable but receding.
        accent: The one colour that means "this, here": the selected
            tab, the tracked-frequency marker, the logo.
        ok: A good state. Green in most themes; not in all of them.
        warn: A state worth noticing but not acting on.
        alarm: A fault.
    """

    bg: str
    panel: str
    panel_alt: str
    inset: str
    edge: str
    text: str
    dim: str
    accent: str
    ok: str
    warn: str
    alarm: str

    def rgb(self, token: str) -> tuple[int, int, int]:
        """The ``(r, g, b)`` triple for one token name.

        Raises:
            KeyError: If ``token`` is not a palette token.
        """
        if token not in PALETTE_TOKENS:
            raise KeyError(f"{token!r} is not a palette token. Valid: {', '.join(PALETTE_TOKENS)}.")
        return parse_hex_color(getattr(self, token), token, "palette")


@dataclass(frozen=True, eq=False)
class Colormap:
    """A waterfall colour ramp: control points, and the table they build.

    ``eq=False`` is deliberate and is the Session 19 lesson written down
    in code. A generated ``__eq__`` on a dataclass carrying a numpy array
    compares that array with ``==``, gets an array back rather than a
    bool, and raises "the truth value of an array with more than one
    element is ambiguous" -- from innocuous places like ``x in
    some_list``. Identity comparison is both correct and sufficient here:
    a colormap is loaded once per theme and passed around by reference.

    Args:
        stops: Control points as ``(position, "#rrggbb")``, position in
            ``[0, 1]``, sorted ascending, first at 0.0 and last at 1.0.
        table: The 256x3 ``uint8`` lookup table interpolated from them.

    The table is built once, at load, rather than per row -- a waterfall
    renders one row per frame at display rate, and anything on that path
    is measured code (Session 16).
    """

    stops: tuple[tuple[float, str], ...]
    table: np.ndarray

    @classmethod
    def from_stops(cls, stops: tuple[tuple[float, str], ...]) -> Colormap:
        """Interpolate control points into a 256-entry table."""
        positions = np.array([position for position, _ in stops], dtype=np.float64)
        colors = np.array(
            [parse_hex_color(color, "stops", "waterfall") for _, color in stops],
            dtype=np.float64,
        )
        table = np.empty((256, 3), dtype=np.uint8)
        x = np.linspace(0.0, 1.0, 256)
        for channel in range(3):
            table[:, channel] = np.interp(x, positions, colors[:, channel]).astype(np.uint8)
        return cls(stops=stops, table=table)

    @property
    def floor_color(self) -> tuple[int, int, int]:
        """The colour of an empty row -- the table's darkest end."""
        return (int(self.table[0][0]), int(self.table[0][1]), int(self.table[0][2]))


@dataclass(frozen=True)
class Chrome:
    """Shape, typography and effects -- the part of a theme that is not colour.

    Chrome styles are a small enumerated set the shell implements, and a
    theme picks one rather than describing one. That asymmetry with the
    palette is on purpose: a colour is data, but LCARS's accent bars are
    child widgets and WOPR's scanlines are a painted overlay, and neither
    can be expressed as a value in a TOML file.

    Args:
        style: The style that will actually be drawn -- always one of
            :data:`CHROME_STYLES`.
        declared_style: What the file asked for, which may name a style
            this build does not implement. **An unknown chrome style is
            not fatal, and that asymmetry with the palette is
            deliberate.** The palette's token set is closed, so an
            unrecognised key there is overwhelmingly likely to be a
            typo and stays an error. Chrome is an extensible set the
            shell grows over time, so a theme from a newer QSOrbit
            naming a style this build lacks still has eleven perfectly
            good colours and a working colormap -- refusing all of that
            over a border treatment would be disproportionate. It
            renders in default chrome, and the mismatch is reported
            through :attr:`Theme.notes` rather than swallowed.
        font: Preferred UI font family, or ``None`` for the platform
            default. A family QSOrbit ships (``Antonio``,
            ``Share Tech Mono``) is registered before the theme applies;
            any other name falls back silently if the system lacks it,
            because a missing font should cost fidelity rather than
            refuse to start a receiver.
        mono: Preferred monospace family for readouts and frequencies,
            or ``None`` for the platform default.
    """

    style: str = "default"
    font: str | None = None
    mono: str | None = None
    declared_style: str = "default"


@dataclass(frozen=True, eq=False)
class Theme:
    """A loaded theme: what it is called, and everything it styles.

    ``eq=False`` for the same reason :class:`Colormap` has it -- a Theme
    holds one, transitively carrying a numpy array.

    Args:
        format: The file format version it declared. Absent means 1.
        name: Display name, as shown in the theme picker.
        author: Who made it, or ``None``. Optional, and present from
            the first release on purpose: themes are meant to be
            shared, and under strict validation a field added later
            would make every older build reject every newer theme.
        description: One line about it, or ``None``.
        url: Where it came from, or ``None``.
        notes: Human-readable remarks raised while loading -- currently
            only an unimplemented chrome style. Empty for a theme this
            build understands completely. Reported once by whoever
            applies the theme rather than raised, so a shared file that
            is merely *newer* still works.
        slug: Filename stem, and the stable identifier persisted in
            config. ``"Night Ops"`` is the name; ``"night-ops"`` is the
            slug.
        palette: The colour tokens.
        waterfall: The colormap.
        chrome: Shape and typography.
        source: The file it was loaded from, or ``None`` if built in
            memory. Shown in the picker so a user editing a theme can
            tell which copy is winning.
    """

    name: str
    slug: str
    palette: Palette
    waterfall: Colormap
    chrome: Chrome = Chrome()
    source: Path | None = None
    format: int = THEME_FORMAT
    author: str | None = None
    description: str | None = None
    url: str | None = None
    notes: tuple[str, ...] = ()


def app_themes_dir() -> Path:
    """Where a user's own theme files live, beside their ``config.toml``.

    Deliberately the same directory family as station config rather than
    somewhere inside the installed package: a user must never have to
    write into site-packages to add a theme, and a theme they added must
    survive an upgrade.
    """
    from qsorbit.core.station import user_config_dir

    return user_config_dir() / "themes"


def theme_search_paths() -> tuple[Path, ...]:
    """Directories searched for themes, lowest precedence first.

    Shipped themes first, the user's own second, so a user file with the
    same stem shadows a shipped one. That is what makes "start from
    Deep Space and change two colours" work: copy it out, edit it, keep
    the name.
    """
    return (DEFAULT_THEMES_DIR, app_themes_dir())


def load_theme(path: str | Path) -> Theme:
    """Load and validate one theme file.

    Args:
        path: The ``.toml`` file to read. Its stem becomes the slug.

    Raises:
        ThemeError: If the file is missing, is not valid TOML, or has a
            missing, unknown, or malformed key.
    """
    resolved = Path(path)
    try:
        with resolved.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as error:
        raise ThemeError(f"No theme file at {resolved}.") from error
    except tomllib.TOMLDecodeError as error:
        raise ThemeError(f"{resolved} is not valid TOML: {error}") from error
    except OSError as error:  # pragma: no cover - permissions, a directory, a bad mount
        raise ThemeError(f"Could not read {resolved}: {error}") from error

    _reject_unknown_keys(
        data,
        {"format", "name", "author", "description", "url", "palette", "waterfall", "chrome"},
        "top level",
        resolved,
    )

    declared_format = _load_format(data, resolved)

    name = _require(data, "name", "top level", resolved)
    if not isinstance(name, str) or not name.strip():
        raise ThemeError(f"'name' in [top level] of {resolved} must be a non-empty string.")

    chrome = _load_chrome(data.get("chrome", {}), resolved)
    notes: list[str] = []
    if chrome.declared_style != chrome.style:
        notes.append(
            f"{name.strip()} asks for chrome style {chrome.declared_style!r}, which this "
            f"build does not implement; drawing it in {chrome.style!r} chrome instead. "
            f"Its colours are unaffected."
        )

    return Theme(
        name=name.strip(),
        slug=resolved.stem,
        palette=_load_palette(_require(data, "palette", "top level", resolved), resolved),
        waterfall=_load_colormap(_require(data, "waterfall", "top level", resolved), resolved),
        chrome=chrome,
        source=resolved,
        format=declared_format,
        author=_optional_text(data, "author", resolved),
        description=_optional_text(data, "description", resolved),
        url=_optional_text(data, "url", resolved),
        notes=tuple(notes),
    )


def discover_themes(
    search_paths: tuple[Path, ...] | None = None,
) -> dict[str, Theme]:
    """Load every theme found on the search path, keyed by slug.

    Later directories shadow earlier ones on a slug collision, per
    :func:`theme_search_paths`. A directory that does not exist is
    skipped rather than raising -- a user with no ``themes/`` directory
    is the normal case, not an error.

    A single unreadable file does not sink the rest: it is skipped and
    its slug is absent from the result. The catalogue is a directory
    scan, and one bad hand-edited file should cost that theme, not the
    application. Callers wanting the specific error should use
    :func:`load_theme` on that path directly.
    """
    paths = theme_search_paths() if search_paths is None else search_paths
    found: dict[str, Theme] = {}
    for directory in paths:
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.glob("*.toml")):
            try:
                theme = load_theme(candidate)
            except ThemeError:
                continue
            found[theme.slug] = theme
    return found


# ---------------------------------------------------------------------------
# Section loaders
# ---------------------------------------------------------------------------


def _load_format(data: dict[str, Any], path: Path) -> int:
    """Read and check the declared file format.

    A file from a *newer* QSOrbit gets a message naming that as the
    cause, rather than an unknown-key error somewhere further down that
    reads like a typo. Telling those two apart is the whole reason this
    key exists.
    """
    raw = data.get("format", THEME_FORMAT)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ThemeError(
            f"'format' in [top level] of {path} must be a positive whole number, got {raw!r}."
        )
    if raw > THEME_FORMAT:
        raise ThemeError(
            f"{path} declares theme format {raw}, but this build of QSOrbit reads "
            f"format {THEME_FORMAT}. The theme was made for a newer QSOrbit - update, "
            f"or ask its author for a format-{THEME_FORMAT} version."
        )
    return raw


def _optional_text(data: dict[str, Any], key: str, path: Path) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ThemeError(
            f"'{key}' in [top level] of {path} must be a non-empty string, got {value!r}."
        )
    return value.strip()


def _load_palette(table: Any, path: Path) -> Palette:
    if not isinstance(table, dict):
        raise ThemeError(f"[palette] in {path} must be a table, got {table!r}.")
    _reject_unknown_keys(table, set(PALETTE_TOKENS), "palette", path)
    values: dict[str, str] = {}
    for token in PALETTE_TOKENS:
        raw = _require(table, token, "palette", path)
        parse_hex_color(raw, token, "palette", path)  # validate, keep the string
        values[token] = raw
    return Palette(**values)


def _load_colormap(table: Any, path: Path) -> Colormap:
    if not isinstance(table, dict):
        raise ThemeError(f"[waterfall] in {path} must be a table, got {table!r}.")
    _reject_unknown_keys(table, {"stops"}, "waterfall", path)
    raw_stops = _require(table, "stops", "waterfall", path)
    if not isinstance(raw_stops, list) or len(raw_stops) < 2:
        raise ThemeError(
            f"'stops' in [waterfall] of {path} must be a list of at least two "
            f"[position, colour] pairs, got {raw_stops!r}."
        )

    stops: list[tuple[float, str]] = []
    previous = -1.0
    for index, entry in enumerate(raw_stops):
        if not isinstance(entry, list) or len(entry) != 2:
            raise ThemeError(
                f"Stop {index} in [waterfall] of {path} must be a "
                f"[position, colour] pair, got {entry!r}."
            )
        position, color = entry
        if isinstance(position, bool) or not isinstance(position, (int, float)):
            raise ThemeError(
                f"Stop {index}'s position in [waterfall] of {path} must be a "
                f"number, got {position!r}."
            )
        position = float(position)
        if not 0.0 <= position <= 1.0:
            raise ThemeError(
                f"Stop {index}'s position in [waterfall] of {path} must be "
                f"between 0 and 1, got {position}."
            )
        if position <= previous:
            raise ThemeError(
                f"Stops in [waterfall] of {path} must be sorted and strictly "
                f"ascending; stop {index} is at {position} after {previous}."
            )
        parse_hex_color(color, f"stops[{index}]", "waterfall", path)
        stops.append((position, color))
        previous = position

    if stops[0][0] != 0.0 or stops[-1][0] != 1.0:
        raise ThemeError(
            f"Stops in [waterfall] of {path} must span the whole range: the "
            f"first must be at 0.0 and the last at 1.0, got {stops[0][0]} to "
            f"{stops[-1][0]}."
        )
    return Colormap.from_stops(tuple(stops))


def _load_chrome(table: Any, path: Path) -> Chrome:
    if not isinstance(table, dict):
        raise ThemeError(f"[chrome] in {path} must be a table, got {table!r}.")
    _reject_unknown_keys(table, {"style", "font", "mono"}, "chrome", path)
    declared = table.get("style", "default")
    if not isinstance(declared, str) or not declared.strip():
        raise ThemeError(
            f"'style' in [chrome] of {path} must be a string naming a chrome "
            f"style, got {declared!r}."
        )
    declared = declared.strip()
    # Unknown style: fall back rather than refuse. See Chrome's docstring
    # for why this is not inconsistent with the palette being strict.
    style = declared if declared in CHROME_STYLES else "default"
    font = table.get("font")
    mono = table.get("mono")
    for key, value in (("font", font), ("mono", mono)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ThemeError(
                f"'{key}' in [chrome] of {path} must be a non-empty string, got {value!r}."
            )
    return Chrome(
        style=style,
        declared_style=declared,
        font=font.strip() if isinstance(font, str) else None,
        mono=mono.strip() if isinstance(mono, str) else None,
    )


# ---------------------------------------------------------------------------
# Table and key helpers -- deliberately parallel to qsorbit.core.profiles.catalog
# ---------------------------------------------------------------------------


def _reject_unknown_keys(
    table: dict[str, Any], allowed: set[str], section: str, path: Path
) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ThemeError(
            f"Unknown key{'s' if len(unknown) > 1 else ''} in [{section}] of {path}: "
            f"{', '.join(unknown)}. Valid keys: {', '.join(sorted(allowed))}."
        )


def _require(table: dict[str, Any], key: str, section: str, path: Path) -> Any:
    if key not in table:
        raise ThemeError(f"Missing required key '{key}' in [{section}] of {path}.")
    return table[key]
