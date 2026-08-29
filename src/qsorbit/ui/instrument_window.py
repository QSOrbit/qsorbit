"""A plain window that hosts Phase 2's debugging instruments.

Deliberately not the UI shell. The Phase 2 brief is explicit that tabs,
docks, tear-off windows and a dark theme belong to a later phase, and
that this phase's windows are lab instruments the real shell will
eventually absorb or replace. This is that window: a vertical stack of
whichever panels the caller passes in, and nothing else.

What it does carry forward is the convention adopted in Session 19 —
**every UI element is a widget that receives its feed and knows nothing
about what contains it, and no container gets built beyond what the
current chunk needs.** This window is the smallest container that meets
Chunk F's done-when (a waterfall on screen alongside the readout). When
the real shell arrives, the panels move into it unchanged and this file
goes away.

Each panel owns its own feed and its own timer, which is why the two can
be shown together, separately, or not at all: the readout drives a
:class:`~qsorbit.core.pointing.TrackingLoop` at 1 Hz on the GUI thread,
and the waterfall drains a background
:class:`~qsorbit.core.dsp.spectrum_stream.SpectrumStream` at display
rate. Neither knows the other exists, so a bench session with only a
rotor, or only an SDR, works exactly as well as one with both.
"""

from __future__ import annotations

import sys

from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from qsorbit.ui.quieting_widget import QuietingWidget
from qsorbit.ui.readout_widget import ReadoutWidget
from qsorbit.ui.spectrum_line_widget import SpectrumLineWidget
from qsorbit.ui.theme_manager import ThemeManager
from qsorbit.ui.waterfall_widget import WaterfallWidget
from qsorbit.ui.zoom_controller import ZoomController
from qsorbit.ui.zoom_controls_widget import ZoomControlsWidget

#: Shown when a window is opened with no panels at all.
_EMPTY_NOTE = (
    "No instruments attached. Pass a readout, a waterfall, a spectrum line, "
    "a quieting panel, or any combination."
)


class InstrumentWindow(QMainWindow):
    """Hosts a readout, a spectrum line, a waterfall, a quieting panel, or any combination.

    Args:
        readout: The rotor/sky readout panel, or ``None``.
        spectrum_line: The "wire" line-trace spectrum panel, or ``None``.
        waterfall: The spectrum waterfall panel, or ``None``.
        quieting: The live squelch-quieting panel, or ``None``.
        zoom_controls: The span/lock numeric controls shared by
            ``spectrum_line`` and ``waterfall``, or ``None``.
        themes: The theme manager, or ``None``. When given, **Ctrl+T
            cycles through every available theme** and Ctrl+Shift+T
            steps back. This is a bench affordance rather than the real
            control: the theme picker belongs in the shell's top bar,
            which is the next PR. It exists now because "switching
            theme at runtime restyles every widget, waterfall colormap
            included, without a restart" is the chunk's marquee
            done-when clause, and a shortcut is what turns that from a
            unit-test claim into something an operator watches happen
            on a real monitor. It is also the fastest way to find a
            widget that quietly fails to repaint — under a relaunch you
            would never see it.
        zoom_controller: The pan/zoom/lock state itself, or ``None`` —
            not laid out (it draws nothing of its own), held only so
            :meth:`closeEvent` can stop its tracked-frequency polling
            alongside every other panel's.
        title: Window title. Defaults to naming whatever is being
            tracked when a readout is present, matching Chunk B's
            "QSOrbit - tracking SUN", and to a plain product name
            otherwise — a waterfall alone is not tracking anything, and
            a title claiming it was would be the sort of small dishonesty
            this project's readouts go out of their way to avoid.

    Closing the window stops every panel polling. It does not stop the
    rotor, the SDR, or the streaming worker behind them: whoever
    constructed those owns their lifetime, the same policy every layer
    below here already follows.
    """

    def __init__(
        self,
        *,
        readout: ReadoutWidget | None = None,
        spectrum_line: SpectrumLineWidget | None = None,
        waterfall: WaterfallWidget | None = None,
        quieting: QuietingWidget | None = None,
        zoom_controls: ZoomControlsWidget | None = None,
        zoom_controller: ZoomController | None = None,
        themes: ThemeManager | None = None,
        title: str | None = None,
    ) -> None:
        super().__init__()
        self._readout = readout
        self._spectrum_line = spectrum_line
        self._waterfall = waterfall
        self._quieting = quieting
        self._zoom_controls = zoom_controls
        self._zoom_controller = zoom_controller
        self._themes = themes

        if title is None:
            title = (
                f"QSOrbit - tracking {readout.target_name}"
                if readout is not None
                else "QSOrbit - instruments"
            )
        self.setWindowTitle(title)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        if readout is not None:
            layout.addWidget(readout)
        if quieting is not None:
            # Placed with the readout rather than the spectrum panels: it
            # is a fixed-height row of text and a bar, the same shape as
            # the readout's own rows, and Session 22 asked for it beside
            # the squelch's effect on what's audible, not beside the
            # spectrum.
            layout.addWidget(quieting)
        if zoom_controls is not None:
            # A fixed-height row of its own, placed above the spectrum
            # panels it controls rather than beside the readout/quieting
            # rows above — it is about the spectrum, not about tracking.
            layout.addWidget(zoom_controls)
        if spectrum_line is not None:
            # Above the waterfall, matching the SDR#-style layout Phil
            # asked for — a fixed-height row like the readout/quieting
            # ones, so only the waterfall below gets the resize stretch.
            layout.addWidget(spectrum_line)
        if waterfall is not None:
            # Stretch factor 1: the waterfall takes the slack when the
            # window is resized. Every other row above it is a fixed
            # handful of pixels and gaining height would only pad them.
            layout.addWidget(waterfall, 1)
        if readout is None and spectrum_line is None and waterfall is None and quieting is None:
            layout.addWidget(QLabel(_EMPTY_NOTE))
        self.setCentralWidget(central)

        if themes is not None:
            # Parented to the window, so both shortcuts die with it
            # rather than outliving the thing they act on.
            QShortcut(QKeySequence("Ctrl+T"), self, activated=lambda: self._cycle_theme(1))
            QShortcut(QKeySequence("Ctrl+Shift+T"), self, activated=lambda: self._cycle_theme(-1))

    def _cycle_theme(self, step: int) -> None:
        """Apply the next (or previous) theme, wrapping around.

        Prints the theme's name, because a theme whose palette is close
        to its neighbour's is hard to tell apart at a glance and "which
        one am I looking at" is the first question at the bench.
        """
        if self._themes is None:  # pragma: no cover - guarded at construction
            return
        slugs = self._themes.slugs
        index = slugs.index(self._themes.current.slug)
        theme = self._themes.apply(slugs[(index + step) % len(slugs)])
        print(f"theme: {theme.name} ({theme.slug})", file=sys.stderr)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt's spelling
        """Stop every panel polling. Does not stop the hardware."""
        panels = (
            self._readout,
            self._spectrum_line,
            self._waterfall,
            self._quieting,
            self._zoom_controls,
            self._zoom_controller,
        )
        for panel in panels:
            if panel is not None:
                panel.stop()
        super().closeEvent(event)
