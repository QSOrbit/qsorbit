"""The application shell: a top bar, five tabs, and the structural chrome.

This is the window Phase 3 has been building towards, and the thing
:class:`~qsorbit.ui.instrument_window.InstrumentWindow` said it was not.
That file's docstring promised the panels would move into a real shell
unchanged when one arrived. **They did**: not one widget in
:mod:`qsorbit.ui.tabs` was modified to live here, because each already
received its feed and knew nothing about its container. The convention
adopted in Session 19 and honoured from Chunk F onward paid for itself
in this file by costing nothing.

**Three things the stylesheet cannot do, which is why they are here.**
:func:`~qsorbit.ui.theme_qss.chrome_structure` reports what a theme
wants and the shell builds it: LCARS accent bars (owned by
:class:`~qsorbit.ui.cards.Card`), CRT scanlines
(:class:`ScanlineOverlay`), and the glow on headline text. Upper-casing
belongs to the same list -- Qt Style Sheets do not implement
``text-transform``, whatever CSS does.

**Stopping is by walk, not by list.** ``closeEvent`` finds every
descendant with a ``stop()`` and calls it, rather than holding the
hand-written tuple of panels :class:`InstrumentWindow` had. That tuple
worked because exactly six things existed and one function built all of
them. PR3's Custom tab builds its widgets from a config file, so nobody
here will know what exists -- and a panel whose timer was never stopped
keeps polling a dead session, which is how a "clean" shutdown becomes a
traceback nobody can place.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QKeySequence, QPainter, QPaintEvent, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from qsorbit.ui.feed_hub import FeedHub
from qsorbit.ui.tabs import DecodeTab, PlanTab, RadioTab, RotorTab
from qsorbit.ui.theme import Theme
from qsorbit.ui.theme_manager import ThemeManager, glow_color, scanline_color
from qsorbit.ui.theme_qss import chrome_structure

#: Vertical period of the CRT scanline pattern, in pixels: one dark line
#: every three, matching the mockup's repeating gradient.
SCANLINE_PERIOD: Final = 3

#: Blur radius of the CRT text halo.
GLOW_BLUR: Final = 14

#: How often the top bar's clocks tick. One second; they show seconds.
CLOCK_INTERVAL_MS: Final = 1_000

#: Tab titles, in display order.
TAB_TITLES: Final[tuple[str, ...]] = ("Radio", "Rotor", "Plan", "Decode", "Custom")


class ScanlineOverlay(QWidget):
    """Horizontal scanlines painted over the whole window, for CRT chrome.

    Transparent to the mouse, so it is decoration that cannot be clicked
    on by accident -- without
    :attr:`~Qt.WidgetAttribute.WA_TransparentForMouseEvents` this would
    swallow every gesture in the window, including the waterfall's own
    wheel-zoom, and the fault would look like a dead widget rather than
    like an overlay.

    Sized by the window's resize event rather than by a layout: it is
    not *in* the layout, deliberately. A scanline pattern that took part
    in layout would push the tabs down by its own height.
    """

    def __init__(self, themes: ThemeManager, parent: QWidget) -> None:
        super().__init__(parent)
        self._themes = themes
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        themes.changed.connect(lambda _theme: self.update())

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt's spelling
        """Draw one dark line every :data:`SCANLINE_PERIOD` pixels."""
        painter = QPainter(self)
        painter.setPen(scanline_color(self._themes.current))
        for y in range(0, self.height(), SCANLINE_PERIOD):
            painter.drawLine(0, y, self.width(), y)
        painter.end()


class TopBar(QWidget):
    """The strip above the tabs: what is being tracked, and the theme picker.

    **The theme picker is the real control**; PR1's Ctrl+T was a bench
    affordance explicitly waiting for this bar to exist. Both survive --
    the shortcut is still the fastest way to find a widget that quietly
    fails to repaint, which is what it was added for.

    **What this bar does not show.** The mockup carries "AOS -00:04:12"
    and "max el 62 degrees" beside the target name. Those come from
    Chunk B's pass prediction, which exists, but wiring live pass data
    into the shell is Chunk D's job and putting fixed numbers there
    meanwhile would be the exact dishonesty this project's readouts are
    built to avoid: a plausible value that is not a measurement. The
    bar shows what it can measure now -- the target's name, and the
    time in both zones -- and gains the rest when there is something
    true to put in it.

    Args:
        themes: Populates the picker and applies a chosen theme.
        target_name: What is being tracked, or ``None``.
    """

    def __init__(
        self,
        *,
        themes: ThemeManager,
        target_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._themes = themes
        self.setObjectName("Card")
        # See Card.__init__ -- a child QWidget subclass needs this before
        # a stylesheet background reaches it.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(14)

        self.logo = QLabel("QSOrbit", self)
        self.logo.setProperty("role", "accent")
        layout.addWidget(self.logo)

        if target_name is not None:
            caption = QLabel("target", self)
            caption.setProperty("role", "dim")
            target = QLabel(target_name, self)
            target.setProperty("role", "value")
            layout.addWidget(caption)
            layout.addWidget(target)

        layout.addStretch(1)

        self.picker = QComboBox(self)
        self.picker.setToolTip("Theme  (Ctrl+T cycles, Ctrl+Shift+T steps back)")
        for slug in themes.slugs:
            self.picker.addItem(themes.theme(slug).name, slug)
        self._sync_picker(themes.current)
        # Connected after the initial sync so populating the box does
        # not re-apply the theme that is already in force.
        self.picker.currentIndexChanged.connect(self._on_picked)
        themes.changed.connect(self._sync_picker)
        layout.addWidget(self.picker)

        self.local_clock = QLabel("", self)
        self.local_clock.setProperty("role", "dim")
        self.utc_clock = QLabel("", self)
        self.utc_clock.setProperty("role", "dim")
        layout.addWidget(self.local_clock)
        layout.addWidget(self.utc_clock)

        self._timer = QTimer(self)
        self._timer.setInterval(CLOCK_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._tick()

    def stop(self) -> None:
        """Stop the clocks. Found by the shell's stop-everything walk."""
        self._timer.stop()

    def _tick(self) -> None:
        """Show local and UTC side by side.

        Both, always, and never only one. Chunk B's readout established
        this and Session 22 paid for it: an hour of time-zone error is
        a setup for a satellite that has already gone, and Windows
        reports its zone as "Eastern Standard Time" in August as well as
        January, so the offset is not something to work out by eye.
        """
        now = datetime.now().astimezone()
        self.local_clock.setText(now.strftime("%H:%M:%S %Z"))
        self.utc_clock.setText(datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"))

    def _on_picked(self, index: int) -> None:
        slug = self.picker.itemData(index)
        if slug is not None and slug != self._themes.current.slug:
            self._themes.apply(slug)

    def _sync_picker(self, theme: Theme) -> None:
        """Follow the manager, so Ctrl+T moves the dropdown too.

        Without this the two controls disagree the moment either is
        used, and a picker showing the wrong theme is worse than no
        picker: it is a readout that lies about a state you can see.
        """
        index = self.picker.findData(theme.slug)
        if index >= 0 and index != self.picker.currentIndex():
            blocked = self.picker.blockSignals(True)
            self.picker.setCurrentIndex(index)
            self.picker.blockSignals(blocked)


class ShellWindow(QMainWindow):
    """The tabbed shell. Radio, Rotor, Plan, Decode -- and Custom in PR3.

    Args:
        hub: Where every tab's feeds come from.
        themes: The theme manager. Applied by the caller *before* this
            window is built, so no widget is ever constructed unthemed
            and repainted a frame later -- which shows up as a flash of
            default grey chrome on a slow start.
        nominal_hz: The tracked transmitter's rest frequency, for the
            Radio tab's Doppler line.
        title: Window title.
    """

    def __init__(
        self,
        hub: FeedHub,
        *,
        themes: ThemeManager,
        nominal_hz: float | None = None,
        title: str = "QSOrbit",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._themes = themes
        self.setWindowTitle(title)

        rotor = hub.rotor
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.top_bar = TopBar(
            themes=themes,
            target_name=rotor.target_name if rotor is not None else None,
            parent=central,
        )
        layout.addWidget(self.top_bar)

        self.tabs = QTabWidget(central)
        self.tabs.addTab(RadioTab(hub, themes=themes, nominal_hz=nominal_hz), TAB_TITLES[0])
        self.tabs.addTab(RotorTab(hub, themes=themes), TAB_TITLES[1])
        self.tabs.addTab(PlanTab(themes=themes), TAB_TITLES[2])
        self.tabs.addTab(DecodeTab(themes=themes), TAB_TITLES[3])
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        # Parented to the window itself rather than to `central`, so it
        # covers the top bar too -- a CRT effect that stopped at the tab
        # strip would look like a rendering fault rather than a style.
        self._scanlines = ScanlineOverlay(themes, self)
        self._scanlines.hide()

        QShortcut(QKeySequence("Ctrl+T"), self, activated=lambda: self.cycle_theme(1))
        QShortcut(QKeySequence("Ctrl+Shift+T"), self, activated=lambda: self.cycle_theme(-1))

        themes.changed.connect(self._apply_chrome)
        self._apply_chrome(themes.current)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def cycle_theme(self, step: int) -> Theme:
        """Apply the next (or previous) theme, wrapping around.

        Kept from PR1's instrument window on purpose. The picker in the
        top bar is the control an operator uses; this is the one that
        finds bugs, because stepping through eight themes in four
        seconds is how a widget that fails to repaint gets caught. Under
        a relaunch you would never see it.
        """
        slugs = self._themes.slugs
        index = slugs.index(self._themes.current.slug)
        return self._themes.apply(slugs[(index + step) % len(slugs)])

    def _apply_chrome(self, theme: Theme) -> None:
        """Build the structural pieces this theme asks for.

        Every branch here restores as well as applies. A chrome effect
        that could only be turned on would make the first CRT theme of a
        session permanent, which is the asymmetric-apply failure PR1 met
        with fonts and fixed by deleting state rather than adding more.
        """
        structure = chrome_structure(theme)

        self._scanlines.setVisible(structure.scanlines)
        if structure.scanlines:
            self._scanlines.raise_()
            self._scanlines.setGeometry(self.rect())

        for index, title in enumerate(TAB_TITLES[: self.tabs.count()]):
            self.tabs.setTabText(index, title.upper() if structure.uppercase_headings else title)

        glow = glow_color(theme) if structure.glow else None
        for label in (self.top_bar.logo, *self._readout_labels()):
            _set_glow(label, glow)

    def _readout_labels(self) -> list[QLabel]:
        """Every headline number in the window, for the CRT glow.

        Selected by the ``role`` property rather than by type or by a
        list of known widgets, so a panel added by PR3's config-driven
        Custom tab glows too without this file being told it exists.
        """
        return [label for label in self.findChildren(QLabel) if label.property("role") == "readout"]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def resizeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt's spelling
        """Keep the scanline overlay covering the whole window."""
        super().resizeEvent(event)
        self._scanlines.setGeometry(self.rect())

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt's spelling
        """Stop every panel polling. Does not stop the hardware.

        Whoever constructed the session, the stream and the rotor owns
        their lifetime -- the policy every layer below here already
        follows, and the reason a window can be closed mid-pass without
        abandoning the antenna mid-slew.
        """
        for target in self.findChildren(QObject):
            stop = getattr(target, "stop", None)
            if callable(stop):
                stop()
        super().closeEvent(event)


def _set_glow(widget: QWidget, color: object | None) -> None:
    """Apply or remove a text halo.

    ``setGraphicsEffect(None)`` is what removes it, and passing ``None``
    also deletes the previous effect, so nothing accumulates across
    seven theme switches.
    """
    if color is None:
        widget.setGraphicsEffect(None)
        return
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(GLOW_BLUR)
    effect.setOffset(0, 0)
    effect.setColor(color)
    widget.setGraphicsEffect(effect)
