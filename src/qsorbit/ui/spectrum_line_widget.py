"""The "wire" spectrum panel — a classic line trace of amplitude vs frequency.

Phil's second Chunk I request, alongside the waterfall's own pan/zoom:
a plain 2-D spectrum plot of the same frequency, visually similar to
SDR#'s own line-above-waterfall layout. This widget is that line — it
shares :class:`~qsorbit.ui.zoom_controller.ZoomController` and the same
frequency axis with :class:`~qsorbit.ui.waterfall_widget.WaterfallWidget`
so a zoom/pan/lock gesture on either panel moves both, and a reader can
drop a finger straight down from a peak here to find it on the waterfall
beneath it.

Simpler than the waterfall in one real way: there is no history axis
here, so there is nothing analogous to that widget's raw/rendered buffer
split. Only the *newest* frame is ever drawn, re-sliced to the current
zoom on every repaint — cheap, because unlike a few hundred waterfall
rows this is exactly one frame's worth of arithmetic, whether that
repaint was triggered by a new frame or by the zoom changing.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QPainter, QPainterPath
from PySide6.QtWidgets import QWidget

from qsorbit.core.dsp.spectrum import frequency_axis_hz
from qsorbit.ui.spectrum_axis_paint import paint_frequency_axis
from qsorbit.ui.spectrum_zoom import dc_spike_in_view, visible_slice
from qsorbit.ui.theme import Theme
from qsorbit.ui.theme_manager import ThemeManager, theme_color
from qsorbit.ui.waterfall_render import WaterfallScale, bins_to_pixels, tick_position
from qsorbit.ui.waterfall_widget import FrameSource
from qsorbit.ui.zoom_controller import ZoomController

#: How often the widget checks for a new frame, in milliseconds. Same
#: default as :data:`~qsorbit.ui.waterfall_widget.DEFAULT_POLL_INTERVAL_MS`
#: — both panels drain the same underlying stream, and polling this one
#: slower would make it visibly lag the waterfall showing the same data.
DEFAULT_POLL_INTERVAL_MS: Final = 50

#: Pixels the trace is computed at, independent of the widget's size —
#: same reasoning as :data:`~qsorbit.ui.waterfall_widget.DEFAULT_RENDER_WIDTH`,
#: though here it only has to survive to the next repaint rather than
#: stay stackable in a history buffer.
DEFAULT_RENDER_WIDTH: Final = 1024

#: Height reserved at the bottom for the frequency axis, in pixels.
AXIS_HEIGHT_PX: Final = 22


class SpectrumLineWidget(QWidget):
    """A live line trace of the newest spectrum frame's power vs frequency.

    Args:
        source: Where frames come from — the same
            :class:`~qsorbit.ui.waterfall_widget.FrameSource` protocol
            the waterfall takes, typically the very same object.
        zoom: The shared pan/zoom/lock state. Pass the
            :class:`~qsorbit.ui.waterfall_widget.WaterfallWidget`'s own
            controller so the two panels move together; a private
            full-band controller is built when omitted, matching that
            widget's own default.
        scale: The dB window the trace is plotted against. Pass the same
            :class:`~qsorbit.ui.waterfall_render.WaterfallScale` given to
            the waterfall so the two panels agree on what "loud" means;
            defaults independently otherwise, which is the same numbers
            either way unless a caller customized one and not the other.
        render_width: Pixels the trace is downsampled to before painting
            — see :data:`DEFAULT_RENDER_WIDTH`.
        poll_interval_ms: How often to check for a new frame.
    """

    def __init__(
        self,
        source: FrameSource,
        *,
        themes: ThemeManager,
        zoom: ZoomController | None = None,
        scale: WaterfallScale | None = None,
        render_width: int = DEFAULT_RENDER_WIDTH,
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._themes = themes
        themes.changed.connect(self._on_theme_changed)
        if render_width <= 0:
            raise ValueError(f"render_width must be positive, got {render_width!r}.")

        self._source = source
        self._scale = scale if scale is not None else WaterfallScale()
        self._render_width = render_width

        self._axis_hz = frequency_axis_hz(source.config)
        band_start_hz = float(self._axis_hz[0])
        band_stop_hz = float(self._axis_hz[-1])
        self._dc_hz = source.config.center_freq_hz

        self._zoom_controller = (
            zoom if zoom is not None else ZoomController(band_start_hz, band_stop_hz)
        )
        self._zoom_controller.changed.connect(self._on_zoom_changed)

        #: The newest frame's power, or ``None`` before the first one
        #: arrives — distinct from an all-floor frame, the same "off vs
        #: not yet" distinction this project's other live readouts (e.g.
        #: :mod:`qsorbit.ui.quieting_formatting`) already insist on.
        self._latest_db: np.ndarray | None = None
        self._error: str | None = None

        self._start_hz = band_start_hz
        self._stop_hz = band_stop_hz
        self._refresh_visible_edges()

        self.setMinimumSize(320, 80 + AXIS_HEIGHT_PX)

        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()

    @property
    def zoom_controller(self) -> ZoomController:
        """The pan/zoom/lock state this widget reads and draws under."""
        return self._zoom_controller

    def stop(self) -> None:
        """Stop draining the source. Does not stop the source, or the
        shared :attr:`zoom_controller`, which is not this widget's to own."""
        self._timer.stop()

    def _refresh_visible_edges(self) -> None:
        _, start_hz, stop_hz = visible_slice(
            self._axis_hz, self._axis_hz, self._zoom_controller.zoom
        )
        self._start_hz = start_hz
        self._stop_hz = stop_hz

    def _on_zoom_changed(self) -> None:
        self._refresh_visible_edges()
        self.update()

    def _on_timer(self) -> None:
        try:
            frames = self._source.latest()
        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            self._timer.stop()
            self._error = f"stopped: {exc}"
            self.update()
            return

        if not frames:
            return

        # Only the newest frame matters for a live line trace — unlike
        # the waterfall there is no history axis here to keep the rest
        # of a batch for.
        self._latest_db = frames[-1].power_db
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's spelling
        painter = QPainter(self)
        if self._error is not None:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._error)
            return

        full = self.rect()
        plot_rect = QRect(full.x(), full.y(), full.width(), max(1, full.height() - AXIS_HEIGHT_PX))

        if self._latest_db is None:
            painter.drawText(
                plot_rect, Qt.AlignmentFlag.AlignCenter, "waiting for spectrum frames..."
            )
        else:
            sliced, _start_hz, _stop_hz = visible_slice(
                self._latest_db, self._axis_hz, self._zoom_controller.zoom
            )
            pixels_db = bins_to_pixels(sliced, max(1, plot_rect.width()))
            self._paint_trace(painter, plot_rect, pixels_db)

        marker_hz = dc_spike_in_view(self._dc_hz, self._start_hz, self._stop_hz)
        if marker_hz is not None:
            self._paint_dc_marker(painter, plot_rect, marker_hz)

        paint_frequency_axis(
            painter,
            plot_rect,
            self._start_hz,
            self._stop_hz,
            AXIS_HEIGHT_PX,
            self.palette().windowText().color(),
        )

    def _paint_trace(self, painter: QPainter, plot_rect: QRect, pixels_db: np.ndarray) -> None:
        """Draw ``pixels_db`` as a single polyline, floor at the bottom, ceiling at the top."""
        floor_db, ceiling_db = self._scale.floor_db, self._scale.ceiling_db
        clipped = np.clip(pixels_db, floor_db, ceiling_db)
        fractions = (clipped - floor_db) / self._scale.span_db
        ys = plot_rect.bottom() - fractions * plot_rect.height()

        path = QPainterPath()
        path.moveTo(float(plot_rect.left()), float(ys[0]))
        for offset, y in enumerate(ys[1:], start=1):
            path.lineTo(float(plot_rect.left() + offset), float(y))

        painter.setPen(self.palette().windowText().color())
        painter.drawPath(path)

    def _on_theme_changed(self, theme: Theme) -> None:
        self.update()

    def _paint_dc_marker(self, painter: QPainter, plot_rect: QRect, marker_hz: float) -> None:
        """See :meth:`~qsorbit.ui.waterfall_widget.WaterfallWidget._paint_dc_marker` —
        the same "visual marker only" decision, drawn identically here so
        the marker lines up between the two panels.

            The DC marker is drawn in the theme's ``warn`` colour,
            which is the semantically right token: the spike is a
            receiver artifact to be discounted, not a signal. It used to
            be ``Qt.GlobalColor.yellow`` in both spectrum panels -- a
            hardcoded colour that survived the first theme pass because
            the check for literals matched ``Qt.yellow`` and the code
            said ``Qt.GlobalColor.yellow``. It showed up as a bright
            yellow line under Night Ops, the one theme where a bright
            yellow line costs the operator their dark adaptation, which
            is exactly the failure the no-hardcoded-colour rule exists
            to prevent.
        """
        x = int(round(tick_position(marker_hz, self._start_hz, self._stop_hz, plot_rect.width())))
        painter.setPen(theme_color(self._themes.current, "warn"))
        painter.drawLine(x, plot_rect.top(), x, plot_rect.bottom())
        painter.drawText(
            QRect(x + 3, plot_rect.top() + 2, 24, 14), Qt.AlignmentFlag.AlignLeft, "DC"
        )
