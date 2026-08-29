"""Applying a theme to a running application, and announcing the change.

The Qt half of the theme system, and the only part of it that imports
PySide6 -- :mod:`qsorbit.ui.theme` and :mod:`qsorbit.ui.theme_qss` stay
toolkit-free so the token model and the stylesheet are testable without
it. Same split, and the same reasoning, as
:mod:`qsorbit.ui.waterfall_render` against
:mod:`qsorbit.ui.waterfall_widget`.

**Styling goes out over two channels and a signal.** The stylesheet and
the QPalette both come from the theme and are both set on the
application, because Qt honours them in different places: standard
widgets take the stylesheet, and custom-painted ones read
``self.palette()``. Neither alone covers the screen. The signal is for
what neither can express -- the waterfall's colormap, and anything else
a widget draws pixel by pixel.

**A signal per theme change is fine, and this is not a contradiction of
Chunk F.** Session 19 ruled out cross-thread Qt signals for spectrum
frames because a live stream can produce ~1,000 of them a second and Qt
degrades silently under that load. A theme change is a human picking
from a dropdown: single-figure events per session, on the GUI thread,
with no producer behind them. The rate regime is the entire argument, so
it does not carry over.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from qsorbit.ui.theme import (
    DEFAULT_THEME_NAME,
    Theme,
    ThemeError,
    discover_themes,
    load_theme,
    parse_hex_color,
)
from qsorbit.ui.theme_qss import build_stylesheet, mono_families, palette_roles

#: Font files shipped with QSOrbit, registered once so the two chrome
#: themes that name them render as designed on a machine that has never
#: seen them. Both are SIL Open Font License; the licence text ships
#: beside them.
FONTS_DIR: Path = Path(__file__).parent / "fonts"


def register_bundled_fonts() -> tuple[str, ...]:
    """Register every shipped font with Qt, returning the family names.

    Safe to call more than once -- Qt de-duplicates by content, and the
    families are simply reported again. Returns an empty tuple if the
    fonts directory is missing, which is a degraded install rather than
    a fault: the themes that want those families fall back to the
    platform default, and a receiver that refused to start over a
    missing typeface would be a worse outcome than an ugly one.
    """
    if not FONTS_DIR.is_dir():  # pragma: no cover - only on a broken install
        return ()
    families: list[str] = []
    for path in sorted(FONTS_DIR.glob("*.ttf")):
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id != -1:
            families.extend(QFontDatabase.applicationFontFamilies(font_id))
    return tuple(dict.fromkeys(families))


def theme_color(theme: Theme, token: str) -> QColor:
    """One palette token, as a Qt colour, for a custom-painted widget.

    Exists so a widget never has to write ``QColor(...)`` itself.
    Everything a widget draws by hand -- the DC marker, axis ticks --
    goes through here, which keeps ``QColor(`` a bannable pattern
    everywhere outside the theme system and makes the *correct* way to
    get a colour the one with a name.

    Raises:
        KeyError: If ``token`` is not a palette token, so a typo fails
            loudly rather than painting something arbitrary.
    """
    return QColor(*theme.palette.rgb(token))


#: Hue rotations, in degrees, applied to a theme's accent to produce the
#: LCARS accent-bar sequence. Derived rather than declared, which was
#: Phil's call at PR2 kickoff: the alternative was a new optional
#: ``[chrome]`` key listing bar colours, and adding theme-format surface
#: one PR after v1 shipped is a cost that lands on every theme author
#: forever, to buy pixel fidelity on one shipped theme.
#:
#: **Where the numbers come from.** LCARS is the reference, so the
#: offsets are the ones that reproduce it: the mockup's bars are
#: ``#ff9966`` (hue 20), ``#cc99cc`` (hue 300) and ``#9c9cff`` (hue 240),
#: which is the accent plus 220 degrees and plus 280 degrees --
#: in that order, because the mockup's ``nth-child(3n+2)`` rule puts the
#: periwinkle on the *second* card and the mauve on the third. Getting
#: that pair the wrong way round is invisible in a unit test and obvious
#: the moment two cards sit next to each other, which is how it was
#: caught. Saturation
#: and value come from the accent rather than from the mockup, so the
#: sequence stays in that theme's own key -- a user theme picking
#: ``style = "lcars"`` gets three bars drawn from *its* accent, not from
#: a Star Trek palette bolted onto it. The visible cost, recorded here
#: rather than left to be discovered: QSOrbit's LCARS bars are more
#: saturated than the mockup's pastels, and the mockup gets updated to
#: match, per the standing rule that bench reality wins.
ACCENT_BAR_HUE_OFFSETS: Final[tuple[int, ...]] = (0, 220, 280)

#: Opacity of a single CRT scanline, out of 255. The mockup draws 20%
#: black on a 3-pixel period; this is that number, kept where the widget
#: that paints it can read it rather than inside the painter.
SCANLINE_ALPHA: Final = 51

#: Opacity of the CRT text halo. The mockup's glow is 55% of the accent.
GLOW_ALPHA: Final = 140


def accent_bar_color(theme: Theme, index: int) -> QColor:
    """The colour of the ``index``-th LCARS-style accent bar.

    Cycles through :data:`ACCENT_BAR_HUE_OFFSETS`, so consecutive cards
    down a column do not all wear the same stripe -- which is the thing
    that makes LCARS read as LCARS rather than as a themed left border.

    An accent with no meaningful hue -- a pure grey, which a monochrome
    user theme could easily have -- rotates to nothing, so every bar
    comes out as the accent itself. That is the right degradation:
    three identical bars in the theme's own colour, rather than three
    arbitrary hues invented for a palette that deliberately had none.
    """
    color = theme_color(theme, "accent")
    hue = color.hue()
    if hue < 0:
        return color
    offset = ACCENT_BAR_HUE_OFFSETS[index % len(ACCENT_BAR_HUE_OFFSETS)]
    if offset == 0:
        # Returned unrotated rather than round-tripped through HSV. The
        # colour would come back the same to the eye either way, but not
        # in the same *spec*: QColor compares by spec as well as value,
        # so an HSV-built copy of an RGB accent is unequal to it. Cheap
        # to avoid, and it keeps "the first bar is the accent" true as
        # an identity rather than only as a rendering.
        return color
    rotated = QColor()
    rotated.setHsv((hue + offset) % 360, color.saturation(), color.value())
    return rotated


def scanline_color(theme: Theme) -> QColor:
    """The colour of one CRT scanline, for the ``crt`` chrome's overlay.

    The background at partial alpha, so the overlay darkens whatever is
    under it rather than tinting it -- a scanline is an absence of
    phosphor, not a colour of its own. Taken from ``bg`` rather than
    from black so that a light theme which ever picks ``style = "crt"``
    scans *lighter*, which is what a gap between bright lines actually
    looks like.
    """
    color = theme_color(theme, "bg")
    color.setAlpha(SCANLINE_ALPHA)
    return color


def glow_color(theme: Theme) -> QColor:
    """The halo colour for the ``crt`` chrome's glowing headline text."""
    color = theme_color(theme, "accent")
    color.setAlpha(GLOW_ALPHA)
    return color


def build_qpalette(theme: Theme) -> QPalette:
    """Build the QPalette custom-painted widgets will read.

    Roles come from :func:`~qsorbit.ui.theme_qss.palette_roles`, which
    names them as strings so that module needs no Qt import. A role Qt
    does not have is skipped rather than raising: the map is a
    best-effort bridge onto a toolkit enum that has changed across Qt
    versions, and a missing role costs one shade, not the theme.
    """
    palette = QPalette()
    for role_name, color in palette_roles(theme).items():
        role = getattr(QPalette.ColorRole, role_name, None)
        if role is None:  # pragma: no cover - depends on the Qt version
            continue
        palette.setColor(role, QColor(*parse_hex_color(color, role_name, "palette")))
    # Disabled text has to be dimmer than enabled text or a greyed-out
    # control is indistinguishable from a live one -- which matters here
    # because branch B's meters are greyed out until Chunk E's second
    # SDR exists, and "disabled" is load-bearing information rather than
    # decoration.
    dim = QColor(*parse_hex_color(theme.palette.dim, "dim", "palette"))
    for role_name in ("WindowText", "Text", "ButtonText"):
        role = getattr(QPalette.ColorRole, role_name, None)
        if role is not None:
            palette.setColor(QPalette.ColorGroup.Disabled, role, dim)
    return palette


class ThemeManager(QObject):
    """Holds the available themes, applies one, and says when it changed.

    Handed to a widget the way a
    :class:`~qsorbit.ui.zoom_controller.ZoomController` is: as shared
    state the widget subscribes to itself. A widget that draws its own
    pixels connects to :attr:`changed` and re-reads what it needs;
    everything else is covered by the stylesheet and palette and needs
    no wiring at all.

    Args:
        themes: Available themes by slug. Must be non-empty.
        current: Slug to start on. Falls back to
            :data:`~qsorbit.ui.theme.DEFAULT_THEME_NAME`, then to
            whichever slug sorts first, so a manager built from a
            directory that happens not to contain the default still
            starts on something.

    Raises:
        ValueError: If ``themes`` is empty. A manager with no theme
            could not answer :attr:`current`, and every widget asking it
            for a colour would have to handle ``None`` -- so it fails
            here instead, at construction, where the message can say so.
    """

    #: Emitted after a new theme has been applied, carrying the
    #: :class:`~qsorbit.ui.theme.Theme`. ``object`` rather than a typed
    #: signal because Theme is a plain dataclass, not a QObject.
    changed = Signal(object)

    def __init__(
        self,
        themes: dict[str, Theme],
        current: str = DEFAULT_THEME_NAME,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not themes:
            raise ValueError("ThemeManager needs at least one theme.")
        self._themes = dict(themes)
        if current not in self._themes:
            current = (
                DEFAULT_THEME_NAME
                if DEFAULT_THEME_NAME in self._themes
                else sorted(self._themes)[0]
            )
        self._current = current
        self._fonts_registered = False
        self._noted: set[str] = set()

    @classmethod
    def discover(
        cls,
        search_paths: tuple[Path, ...] | None = None,
        current: str = DEFAULT_THEME_NAME,
        parent: QObject | None = None,
    ) -> ThemeManager:
        """Build a manager from every theme file on the search path.

        Raises:
            ThemeError: If no theme file was found anywhere, which means
                a broken install rather than an empty user directory --
                the eight shipped themes sit inside the package.
        """
        found = discover_themes(search_paths)
        if not found:
            raise ThemeError(
                "No theme files found. QSOrbit ships eight themes inside the "
                "package, so finding none means the install is incomplete."
            )
        return cls(found, current, parent)

    @property
    def current(self) -> Theme:
        """The theme in force."""
        return self._themes[self._current]

    @property
    def slugs(self) -> tuple[str, ...]:
        """Every available slug, sorted."""
        return tuple(sorted(self._themes))

    def theme(self, slug: str) -> Theme:
        """One theme by slug.

        Raises:
            KeyError: If no theme has that slug.
        """
        if slug not in self._themes:
            raise KeyError(f"No theme {slug!r}. Available: {', '.join(self.slugs)}.")
        return self._themes[slug]

    def add(self, theme: Theme) -> None:
        """Register a theme loaded from somewhere off the search path.

        The escape hatch behind "a theme file not shipped with the app
        loads and works": a caller with a path can
        :func:`~qsorbit.ui.theme.load_theme` it and hand it over without
        having to put it in a directory first.
        """
        self._themes[theme.slug] = theme

    def load_and_add(self, path: str | Path) -> Theme:
        """Load a theme file from anywhere and register it.

        Raises:
            ThemeError: If the file is missing or invalid.
        """
        theme = load_theme(path)
        self.add(theme)
        return theme

    def apply(self, slug: str | None = None) -> Theme:
        """Apply a theme to the running application and announce it.

        Args:
            slug: Which theme, or ``None`` to re-apply the current one
                (useful once, at startup, before any widget exists).

        Returns:
            The theme now in force.

        Raises:
            KeyError: If no theme has that slug.
            RuntimeError: If there is no QApplication to style.
        """
        if slug is not None:
            self.theme(slug)  # raises before any state changes
            self._current = slug
        theme = self.current

        app = QApplication.instance()
        if app is None:
            raise RuntimeError("A QApplication must exist before a theme can be applied.")

        if not self._fonts_registered:
            register_bundled_fonts()
            self._fonts_registered = True

        # Palette before stylesheet: the stylesheet is the more specific
        # of the two and should win where they overlap.
        #
        # Typography rides in the stylesheet rather than going out
        # through ``app.setFont()``, and that is the whole reason a
        # theme switch is symmetric. ``setFont`` is a one-way override
        # with no record of what it replaced, so restoring it means
        # storing the previous font somewhere -- and a manager built
        # after some other theme was already applied would capture that
        # theme's typeface as its "default" and never be able to get
        # back. The stylesheet is replaced wholesale on every apply, so
        # leaving LCARS takes Antonio with it and nothing has to
        # remember anything.
        # A theme's load-time notes are reported once, the first time it
        # is applied. They exist for shared themes: a file from a newer
        # QSOrbit naming a chrome style this build lacks still works, and
        # the operator should be told why it looks plainer than its
        # screenshot rather than left to wonder for an hour.
        if theme.slug not in self._noted:
            self._noted.add(theme.slug)
            for note in theme.notes:
                print(note, file=sys.stderr)

        app.setPalette(build_qpalette(theme))
        app.setStyleSheet(build_stylesheet(theme))

        self.changed.emit(theme)
        return theme

    def mono_font_families(self) -> tuple[str, ...]:
        """The current theme's monospace stack, for widgets that set one."""
        return mono_families(self.current)
