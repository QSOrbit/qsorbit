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

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from qsorbit.ui.readout_widget import ReadoutWidget
from qsorbit.ui.waterfall_widget import WaterfallWidget

#: Shown when a window is opened with no panels at all.
_EMPTY_NOTE = "No instruments attached. Pass a readout, a waterfall, or both."


class InstrumentWindow(QMainWindow):
    """Hosts a readout, a waterfall, either, or neither.

    Args:
        readout: The rotor/sky readout panel, or ``None``.
        waterfall: The spectrum waterfall panel, or ``None``.
        title: Window title. Defaults to naming whatever is being
            tracked when a readout is present, matching Chunk B's
            "QSOrbit - tracking SUN", and to a plain product name
            otherwise — a waterfall alone is not tracking anything, and
            a title claiming it was would be the sort of small dishonesty
            this project's readouts go out of their way to avoid.

    Closing the window stops both panels polling. It does not stop the
    rotor, the SDR, or the streaming worker behind them: whoever
    constructed those owns their lifetime, the same policy every layer
    below here already follows.
    """

    def __init__(
        self,
        *,
        readout: ReadoutWidget | None = None,
        waterfall: WaterfallWidget | None = None,
        title: str | None = None,
    ) -> None:
        super().__init__()
        self._readout = readout
        self._waterfall = waterfall

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
        if waterfall is not None:
            # Stretch factor 1: the waterfall takes the slack when the
            # window is resized. The readout is a fixed handful of text
            # rows and gaining height would only pad it.
            layout.addWidget(waterfall, 1)
        if readout is None and waterfall is None:
            layout.addWidget(QLabel(_EMPTY_NOTE))
        self.setCentralWidget(central)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt's spelling
        """Stop both panels polling. Does not stop the hardware."""
        for panel in (self._readout, self._waterfall):
            if panel is not None:
                panel.stop()
        super().closeEvent(event)
