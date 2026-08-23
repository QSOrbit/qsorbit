"""The waterfall panel — the thin Qt remainder over a spectrum source.

Everything worth arguing about lives in
:mod:`qsorbit.ui.waterfall_render`, which imports no Qt. This module owns
a timer, a ring of rendered rows, and a ``paintEvent``. That is the same
division of labour :mod:`qsorbit.ui.readout_formatting` and
:class:`~qsorbit.ui.readout_widget.ReadoutWidget` already use.

**This widget pulls; nothing pushes to it.** A ``QTimer`` drains
:meth:`~qsorbit.core.dsp.spectrum_stream.SpectrumStream.latest` at
display rate. That is the settled answer to Chunk F's design question and
the reasoning is recorded in full in ``spectrum_stream.py``'s module
docstring; the short version is that a live stream can produce roughly a
thousand spectrum rows a second, a signal-per-frame would post a thousand
events a second onto the GUI event queue, and Qt would accept every one
of them while the UI quietly degraded. Pulling makes back-pressure
structural.

**The widget knows nothing about what contains it.** It takes its feed as
a constructor argument and never reaches for a device, a window, or a
parent. That is the convention adopted in Session 19: a tab, a custom
tab, a dock, a tear-off and a plain debug window then differ only in who
the parent is.
"""

from __future__ import annotations

from collections import deque
from typing import Final, Protocol

import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

from qsorbit.core.dsp.spectrum import SpectrumConfig, frequency_axis_hz
from qsorbit.core.dsp.spectrum_stream import SpectrumFrame
from qsorbit.ui.waterfall_render import (
    WaterfallScale,
    blank_row,
    frequency_ticks,
    render_row,
    tick_position,
)

#: How often the widget drains its source, in milliseconds.
#:
#: **Keep this at or below the source's frame interval.** Polling slower
#: than frames arrive is exactly what
#: :attr:`~qsorbit.core.dsp.spectrum_stream.SpectrumStreamStats.frames_dropped`
#: counts, so the symptom is visible rather than mysterious — but it is
#: still a dropped frame. 50 ms pairs with the streaming layer's default
#: of 20 frames per second.
DEFAULT_POLL_INTERVAL_MS: Final = 50

#: Rows of history kept. At 20 frames per second this is about 30
#: seconds — long enough to watch a Doppler slope develop across a pass,
#: short enough that the oldest rows are still relevant to what is on the
#: radio now.
DEFAULT_HISTORY_ROWS: Final = 600

#: Pixels each row is rendered at, independent of the widget's size.
#:
#: Fixed on purpose. Rendering at the *widget's* width would leave every
#: row already in history the wrong length the moment someone dragged the
#: window edge, and the choice then is between re-rendering all of
#: history from frames no longer kept, or discarding the history on every
#: resize. A fixed render width and a scaled blit sidesteps both: rows
#: stay stackable forever and Qt handles the fit.
DEFAULT_RENDER_WIDTH: Final = 1024

#: Height reserved at the bottom for the frequency axis, in pixels.
AXIS_HEIGHT_PX: Final = 22

#: Roughly how many pixels each frequency label needs to itself. Used to
#: scale the tick count with the widget, so a narrow window thins its
#: labels out instead of overprinting them into a smear.
_PIXELS_PER_LABEL: Final = 90


class FrameSource(Protocol):
    """Anything that can hand over spectrum frames on demand.

    Declared structurally rather than importing
    :class:`~qsorbit.core.dsp.spectrum_stream.SpectrumStream` as a type,
    so a test double or a replayed capture satisfies it without
    subclassing anything — the same reasoning behind
    :class:`~qsorbit.core.tracker.Target`.
    """

    @property
    def config(self) -> SpectrumConfig:
        """The framing behind those frames, for labelling the axis."""
        ...

    def latest(self) -> list[SpectrumFrame]:
        """Return every frame buffered since the last call, oldest first."""
        ...


class WaterfallWidget(QWidget):
    """A scrolling spectrogram of whatever a frame source is producing.

    Newest row at the top, scrolling downward. A tick that raises stops
    the timer and paints the error in place of the waterfall, rather than
    leaving a frozen picture on screen with a dead radio underneath it —
    the same policy
    :class:`~qsorbit.ui.readout_widget.ReadoutWidget` applies to a failing
    loop tick, and for the same reason: a display that silently stops
    updating is indistinguishable from a quiet band.

    Args:
        source: Where frames come from. The widget neither builds it nor
            owns the device behind it; whoever constructed the source is
            responsible for starting and stopping it.
        scale: The dB window mapped onto the colour ramp. Defaults to
            :class:`~qsorbit.ui.waterfall_render.WaterfallScale`'s own
            measured defaults, which suit FM broadcast at bench gain and
            will want lowering for a satellite downlink.
        history_rows: Rows of history to keep.
        render_width: Pixels per row. See
            :data:`DEFAULT_RENDER_WIDTH` for why this is not the widget's
            width.
        poll_interval_ms: How often to drain the source.
    """

    def __init__(
        self,
        source: FrameSource,
        *,
        scale: WaterfallScale | None = None,
        history_rows: int = DEFAULT_HISTORY_ROWS,
        render_width: int = DEFAULT_RENDER_WIDTH,
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if history_rows <= 0:
            raise ValueError(f"history_rows must be positive, got {history_rows!r}.")
        if render_width <= 0:
            raise ValueError(f"render_width must be positive, got {render_width!r}.")

        self._source = source
        self._scale = scale if scale is not None else WaterfallScale()
        self._render_width = render_width
        # Pre-filled to full depth rather than grown from empty. A
        # growing history gets stretched to fill the panel, so the time
        # scale keeps moving until the buffer is full -- about 30 seconds
        # at the default rate -- and a Doppler slope would appear to
        # change angle while nothing about the signal had. Same objection
        # this widget's dB scale already answers, on the other axis.
        self._rows: deque[np.ndarray] = deque(
            (blank_row(render_width) for _ in range(history_rows)), maxlen=history_rows
        )
        self._frames_seen = 0
        self._error: str | None = None

        # Taken from the source's own config, once, so the labels cannot
        # drift from the frames. fftshift order puts the lowest frequency
        # first, which is also left-to-right on screen.
        axis = frequency_axis_hz(source.config)
        self._start_hz = float(axis[0])
        self._stop_hz = float(axis[-1])

        self.setMinimumSize(320, 120 + AXIS_HEIGHT_PX)

        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()

    @property
    def rows_held(self) -> int:
        """How many rows of *real* data are on screen.

        Counts frames received, not buffer occupancy: the buffer is
        pre-filled to full depth, so its length is a constant and would
        tell a caller nothing.
        """
        return min(self._frames_seen, self._rows.maxlen or 0)

    def stop(self) -> None:
        """Stop draining the source. Does not stop the source itself."""
        self._timer.stop()

    def _on_timer(self) -> None:
        try:
            frames = self._source.latest()
        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            self._timer.stop()
            self._error = f"stopped: {exc}"
            self.update()
            return

        if not frames:
            # Nothing new. Normal whenever the widget polls faster than
            # frames arrive, and not a reason to repaint.
            return

        for frame in frames:
            self._rows.appendleft(render_row(frame.power_db, self._render_width, self._scale))
        self._frames_seen += len(frames)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's spelling
        """Blit the history, then label the frequency axis beneath it."""
        painter = QPainter(self)
        if self._error is not None:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._error)
            return
        full = self.rect()
        image_rect = QRect(full.x(), full.y(), full.width(), max(1, full.height() - AXIS_HEIGHT_PX))
        buffer = np.ascontiguousarray(np.stack(tuple(self._rows)))
        height = buffer.shape[0]
        # .copy() forces QImage to own its pixels. Without it the image
        # borrows the buffer, which is a local about to go out of scope --
        # a dangling pointer that renders as garbage or crashes, and does
        # so intermittently, which is the worst way to find out.
        image = QImage(
            buffer.tobytes(),
            self._render_width,
            height,
            self._render_width * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        painter.drawImage(image_rect, image)
        if self._frames_seen == 0:
            # The dark field is already correct -- it says "nothing
            # received" -- but on startup that is indistinguishable from
            # a genuinely quiet band, so say which one it is.
            painter.drawText(
                image_rect, Qt.AlignmentFlag.AlignCenter, "waiting for spectrum frames..."
            )
        self._paint_axis(painter, image_rect)

    def _paint_axis(self, painter: QPainter, image_rect: QRect) -> None:
        """Draw frequency ticks and labels below the waterfall.

        Without this the panel can only say *something is there*, never
        *our signal is at our frequency* — which is the distinction
        Session 14's bring-up was built around, and the one that makes a
        Doppler slope readable as a rate rather than a smear.
        """
        painter.setPen(self.palette().windowText().color())
        top = image_rect.bottom() + 1
        max_ticks = max(2, min(9, image_rect.width() // _PIXELS_PER_LABEL))

        for frequency_hz, label in frequency_ticks(self._start_hz, self._stop_hz, max_ticks):
            x = int(
                round(
                    tick_position(frequency_hz, self._start_hz, self._stop_hz, image_rect.width())
                )
            )
            painter.drawLine(x, top, x, top + 4)
            painter.drawText(
                QRect(x - 45, top + 5, 90, AXIS_HEIGHT_PX - 5),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                label,
            )

        painter.drawText(
            QRect(image_rect.right() - 40, top + 5, 38, AXIS_HEIGHT_PX - 5),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            "MHz",
        )
