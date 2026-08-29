"""Cards: the panel frame every tab lays its widgets out in.

A card is chrome, not an instrument. It has no feed, reads no data and
knows nothing about what it contains beyond "one widget" -- which is the
other half of the standing rule. *Widgets receive feeds and know nothing
about their container* only works if there is a container that knows
nothing about its contents either; otherwise the ignorance is one-sided
and every new panel needs a new container to hold it.

**Nothing here sets a stylesheet.** It cannot: the standing Phase 3 rule
is enforced by a test that bans ``setStyleSheet(`` outright in every
non-theme module of this package, and that ban is load-bearing rather
than fussy -- an inline stylesheet is a colour that survives a theme
switch. Everything a card looks like comes from the application-wide
stylesheet PR1 already generates, reached through an object name
(``Card``, ``Inset``) or a ``role`` property (``heading``, ``dim``,
``ghost``, ``value``, ``readout``, ``ok``, ``warn``, ``alarm``). The
selectors for all of those were written in :mod:`qsorbit.ui.theme_qss`
one PR before anything existed to wear them.

What a card *does* own is the structural chrome QSS cannot express:
:class:`AccentBar`, the coloured stripe LCARS uses instead of a line
border. :func:`~qsorbit.ui.theme_qss.chrome_structure` says whether the
active theme wants one, and the card builds or hides it accordingly --
the split PR1 set up, where the stylesheet decides how things look and
the shell decides what things exist.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from qsorbit.ui.theme import Theme
from qsorbit.ui.theme_manager import ThemeManager, accent_bar_color
from qsorbit.ui.theme_qss import chrome_structure

#: Width of an LCARS accent bar, in pixels. The mockup's is 14.
ACCENT_BAR_WIDTH: int = 14

#: Corner radius of an accent bar. Slightly wider than half the bar so
#: the ends read as capsules rather than as rounded rectangles, which is
#: what LCARS actually does.
ACCENT_BAR_RADIUS: int = 9


class AccentBar(QWidget):
    """The coloured stripe down the leading edge of a card, LCARS-style.

    Painted rather than styled because QSS has no way to draw it: the
    mockup uses a CSS pseudo-element, and Qt's stylesheet language has
    no equivalent. That is exactly the case
    :func:`~qsorbit.ui.theme_qss.chrome_structure` exists to report.

    Args:
        themes: The theme manager, subscribed to directly. **Not passed
            a colour**, because a bar told its colour once would keep it
            through the next theme switch -- the asymmetric-apply
            failure PR1 met with fonts, in a different costume. A widget
            that subscribes itself cannot be forgotten.
        index: Which bar this is down the column, so consecutive cards
            do not all wear the same stripe. See
            :func:`~qsorbit.ui.theme_manager.accent_bar_color`.
    """

    def __init__(self, *, themes: ThemeManager, index: int = 0, parent: QWidget | None = None):
        super().__init__(parent)
        self._themes = themes
        self._index = index
        self.setFixedWidth(ACCENT_BAR_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        themes.changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, _theme: Theme) -> None:
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt's spelling
        """Fill the bar with this card's colour from the active theme."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent_bar_color(self._themes.current, self._index))
        painter.drawRoundedRect(self.rect(), ACCENT_BAR_RADIUS, ACCENT_BAR_RADIUS)
        painter.end()


class Card(QWidget):
    """One titled panel: a heading, and whatever widget was handed in.

    Args:
        title: The heading. Omit for a card with no title -- a spectrum
            panel that already labels itself, say.
        content: The widget to show. The card takes ownership the way
            any Qt parent does, and does not otherwise touch it: it is
            never asked for a feed, a size, or a name.
        themes: The theme manager, subscribed to for the structural
            chrome. Same shape as every other widget in this package.
        index: Position down a column, used only for the accent-bar
            colour cycle.
        stretch: Whether this card should take the slack when its column
            is resized. Exactly one card per column usually wants this
            -- the waterfall -- and every fixed-height row above it
            would only gain padding.

    The heading label is given a **Maximum** horizontal size policy, and
    that one line is a bench finding rather than a preference. LCARS
    draws its headings as filled pills; the mockup gets them pill-shaped
    with ``display: inline-block``, which QSS has no equivalent for, so
    under the default Preferred policy the pill stretched the full width
    of the card and read as a coloured banner (Session 27, recorded as
    an observation and carried into this PR). ``Maximum`` gives the label
    at most its own width and lets it shrink below that, so a long
    heading in a narrow card still elides rather than forcing the card
    wider -- which plain ``Fixed`` would have done.
    """

    def __init__(
        self,
        title: str | None,
        content: QWidget,
        *,
        themes: ThemeManager,
        index: int = 0,
        stretch: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        # Without this the ``QWidget#Card`` rule does nothing at all.
        # Qt honours a stylesheet background and border on a *custom
        # QWidget subclass* only when the widget either reimplements
        # paintEvent through QStyle or carries this attribute -- and the
        # failure is silent: the selector matches, the rule parses, and
        # nothing is drawn. It cost a render to find, and the render is
        # what found it: sampling a card's interior returned the
        # background token where the panel token should have been.
        #
        # A misleading first check made it worse. Tried in isolation the
        # subclass appeared to style itself correctly -- because the test
        # widget was top-level, and a top-level widget fills its
        # background anyway. The bug only exists for a *child*, which is
        # what every card in the shell actually is.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._themes = themes
        self.stretch = stretch

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._bar = AccentBar(themes=themes, index=index, parent=self)
        outer.addWidget(self._bar)

        body = QWidget(self)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(12, 10, 12, 12)
        body_layout.setSpacing(8)
        self._title = title
        self._heading: QLabel | None = None
        if title is not None:
            self._heading = QLabel(title, body)
            self._heading.setProperty("role", "heading")
            self._heading.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            body_layout.addWidget(self._heading)
        body_layout.addWidget(content, 1 if stretch else 0)
        outer.addWidget(body, 1)

        themes.changed.connect(self._on_theme_changed)
        self._apply_chrome(themes.current)

    def _on_theme_changed(self, theme: Theme) -> None:
        self._apply_chrome(theme)

    def _apply_chrome(self, theme: Theme) -> None:
        """Build or hide the structural chrome this theme asks for.

        ``setVisible`` rather than constructing the bar lazily: a theme
        switch is a single-figure event per session on the GUI thread
        (PR1's own note), and a bar that exists but is hidden cannot get
        out of step with the theme the way one rebuilt on every switch
        could.

        **The heading's case is set here, in code, and that is not
        redundant with the stylesheet.** PR1 emits ``text-transform:
        uppercase`` for the LCARS and CRT chromes, and *Qt Style Sheets
        do not implement ``text-transform``* -- it is a CSS property Qt
        silently ignores, not one it applies. That is precisely why
        :class:`~qsorbit.ui.theme_qss.ChromeStructure` reports
        ``uppercase_headings`` at all: its docstring says everything it
        carries is something QSS cannot do. The rule this belongs to is
        an old one here -- *a theme switch has to be symmetric* -- so
        the original title is kept and restored rather than being
        upper-cased in place, which would be a one-way door out of Deep
        Space and into LCARS forever.
        """
        structure = chrome_structure(theme)
        self._bar.setVisible(structure.accent_bars)
        if self._heading is not None and self._title is not None:
            self._heading.setText(
                self._title.upper() if structure.uppercase_headings else self._title
            )


class Placeholder(QLabel):
    """What a tab shows where an instrument would be if the hardware were there.

    Its own class rather than a bare :class:`QLabel` because the wording
    is the point. This project's standing rule is that **"off" and
    "broken" must never look the same** -- written down after a
    perfectly healthy headless run reported "453 block(s) dropped" and
    read as catastrophic data loss. A Radio tab with no SDR attached and
    a Radio tab whose SDR died both show no spectrum; only one of them
    is a fault, and an empty panel says neither.

    Args:
        message: What is absent and why, in a sentence. Written for
            somebody at a bench at night, so it says what to do about it
            where there is anything to do.
        compact: Left-aligned and only as tall as its text, for a small
            card in a side column. The default centres the text in
            whatever space it is given, which reads as deliberate in a
            large empty panel and as lost in a narrow one.
    """

    def __init__(
        self, message: str, *, compact: bool = False, parent: QWidget | None = None
    ) -> None:
        super().__init__(message, parent)
        self.setProperty("role", "ghost")
        self.setWordWrap(True)
        if compact:
            self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        else:
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
