"""The waterfall panel — the thin Qt remainder over a spectrum source.

Everything worth arguing about lives in
:mod:`qsorbit.ui.waterfall_render` and :mod:`qsorbit.ui.spectrum_zoom`,
neither of which imports Qt. This module owns a timer, a ring of
rendered rows, and a ``paintEvent``. That is the same division of labour
:mod:`qsorbit.ui.readout_formatting` and
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

**Chunk I adds pan/zoom/lock, and with it a second buffer.** Before this,
each arriving frame was rendered once, at ingest, straight into the
history that stayed on screen forever — cheap, because the visible
window never changed. A user-adjustable, lockable zoom breaks that: the
window a row was rendered against can now change after the row is
already on screen (most visibly while :attr:`~qsorbit.ui.zoom_controller.ZoomController.locked`
is following a Doppler-shifting downlink, which recenters roughly once a
tracking tick). Re-rendering *only the newest row* under a changed zoom
would leave every older row on screen still cropped to whatever window
was active when *it* arrived — a visible seam at the row where the zoom
last changed, present almost continuously while locked. So this widget
now keeps the raw ``power_db`` history as well as the rendered one:
:attr:`~qsorbit.ui.zoom_controller.ZoomController.changed` triggers a
full re-render of every row currently held, from the raw frames, under
the new window. That is the same cost this module's own history-rows
docstring already argued against paying on every frame — but it is only
paid when the zoom actually changes, not per frame, so the steady-state
cost while nothing is being zoomed or panned is exactly what it was
before this feature existed.
"""

from __future__ import annotations

from collections import deque
from typing import Final, Protocol

import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QWheelEvent
from PySide6.QtWidgets import QWidget

from qsorbit.core.dsp.spectrum import SpectrumConfig, frequency_axis_hz
from qsorbit.core.dsp.spectrum_stream import SpectrumFrame
from qsorbit.ui.spectrum_axis_paint import paint_frequency_axis
from qsorbit.ui.spectrum_zoom import ZoomSpan, dc_spike_in_view, visible_slice
from qsorbit.ui.theme import Colormap, Theme
from qsorbit.ui.theme_manager import ThemeManager, theme_color
from qsorbit.ui.waterfall_render import WaterfallScale, blank_row, render_row, tick_position
from qsorbit.ui.zoom_controller import ZoomController

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

#: A raw frame value read as silence, matching how
#: :func:`~qsorbit.ui.waterfall_render.render_row` already treats
#: anything at or below :attr:`~qsorbit.ui.waterfall_render.WaterfallScale.floor_db`
#: — used only to pre-fill history before any real frame has arrived, so
#: an early zoom change (rebuilding from history that is still all
#: placeholders) paints exactly the same dark field
#: :func:`~qsorbit.ui.waterfall_render.blank_row` already does.
_SILENT_RAW_DB: Final = -1e6

#: Multiplier applied to the visible span per mouse-wheel notch (120 units
#: of ``QWheelEvent.angleDelta()``). Below 1 so scrolling forward/up — the
#: conventional "zoom in" direction most spectrum tools and maps use —
#: narrows the span; scrolling the other way is this value's reciprocal.
WHEEL_ZOOM_FACTOR_PER_NOTCH: Final = 0.85


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
        themes: The active theme, as shared state the widget
            subscribes to itself -- the same shape as ``zoom`` below,
            and for the same reason. Required rather than defaulted:
            a default would be a colour chosen inside a widget, which
            is the one thing Phase 3's standing rule forbids.

            **Subscribing rather than being told is what makes the
            Custom tab work.** That tab builds its widgets from a list
            in a config file, so a second waterfall can exist that no
            code anywhere was written to construct -- and if restyling
            depended on its container remembering to connect a signal,
            the failure would be a panel stuck in the previous theme's
            colours beside correctly restyled ones, with nothing
            raising. Same shape as the frame-stealing bug Chunk A
            fixed: a display that is wrong while looking alive. A
            widget that subscribes itself cannot be forgotten.
        scale: The dB window mapped onto the colour ramp. Defaults to
            :class:`~qsorbit.ui.waterfall_render.WaterfallScale`'s own
            measured defaults, which suit FM broadcast at bench gain and
            will want lowering for a satellite downlink.
        history_rows: Rows of history to keep.
        render_width: Pixels per row. See
            :data:`DEFAULT_RENDER_WIDTH` for why this is not the widget's
            width.
        poll_interval_ms: How often to drain the source.
        zoom: The shared pan/zoom/lock state — see
            :class:`~qsorbit.ui.zoom_controller.ZoomController`. Pass the
            same controller given to a
            :class:`~qsorbit.ui.spectrum_line_widget.SpectrumLineWidget`
            so a gesture on either panel moves both. When omitted, a
            private controller is built spanning the full captured band
            — the widget always has *some* zoom state (there is no
            separate unzoomed code path to keep in sync with the zoomed
            one), it simply is not shared with anything unless a caller
            passes one in.
    """

    def __init__(
        self,
        source: FrameSource,
        *,
        themes: ThemeManager,
        scale: WaterfallScale | None = None,
        history_rows: int = DEFAULT_HISTORY_ROWS,
        render_width: int = DEFAULT_RENDER_WIDTH,
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        zoom: ZoomController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if history_rows <= 0:
            raise ValueError(f"history_rows must be positive, got {history_rows!r}.")
        if render_width <= 0:
            raise ValueError(f"render_width must be positive, got {render_width!r}.")

        self._source = source
        self._scale = scale if scale is not None else WaterfallScale()
        self._themes = themes
        self._colormap = themes.current.waterfall
        themes.changed.connect(self._on_theme_changed)
        self._render_width = render_width

        # Taken from the source's own config, once, so the labels cannot
        # drift from the frames. fftshift order puts the lowest frequency
        # first, which is also left-to-right on screen.
        self._axis_hz = frequency_axis_hz(source.config)
        band_start_hz = float(self._axis_hz[0])
        band_stop_hz = float(self._axis_hz[-1])
        # The tuner's own zero-IF centre spike — see dc_spike_in_view's
        # own docstring for why this is always exactly the config's own
        # center_freq_hz.
        self._dc_hz = source.config.center_freq_hz

        self._zoom_controller = (
            zoom if zoom is not None else ZoomController(band_start_hz, band_stop_hz)
        )
        self._zoom_controller.changed.connect(self._on_zoom_changed)

        # Two decks in lockstep, newest at index 0. _raw_frames is the
        # source of truth; _rendered_rows is a cache of it under the
        # *current* zoom, rebuilt in full only when the zoom changes —
        # see this module's own docstring for why both exist. Pre-filled
        # to full depth rather than grown from empty, for the same
        # "the time scale must not appear to change as the buffer fills"
        # reason the original single-deck version already documented.
        self._raw_frames: deque[np.ndarray] = deque(
            (
                np.full(self._axis_hz.shape, _SILENT_RAW_DB, dtype=np.float32)
                for _ in range(history_rows)
            ),
            maxlen=history_rows,
        )
        self._rendered_rows: deque[np.ndarray] = deque(
            (blank_row(render_width, self._colormap) for _ in range(history_rows)),
            maxlen=history_rows,
        )
        self._frames_seen = 0
        self._error: str | None = None

        # The *visible* edges, which move as the zoom changes — distinct
        # from band_start_hz/band_stop_hz above, which are fixed for the
        # source's whole life. Set from the zoom controller's own current
        # window rather than assumed, so a caller-supplied controller
        # that is not at the full band from the start (unusual, but nothing
        # stops it) still starts the axis labelled correctly.
        self._start_hz = band_start_hz
        self._stop_hz = band_stop_hz
        self._refresh_visible_edges()

        # Drag-to-pan state. None whenever no drag is in progress.
        self._drag_start_x: float | None = None
        self._drag_start_zoom: ZoomSpan | None = None

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
        return min(self._frames_seen, self._rendered_rows.maxlen or 0)

    @property
    def zoom_controller(self) -> ZoomController:
        """The pan/zoom/lock state this widget reads and draws under."""
        return self._zoom_controller

    def stop(self) -> None:
        """Stop draining the source. Does not stop the source, or the
        shared :attr:`zoom_controller` — which may still be in use by a
        :class:`~qsorbit.ui.spectrum_line_widget.SpectrumLineWidget`
        sharing it, and is not this widget's to own."""
        self._timer.stop()

    def _refresh_visible_edges(self) -> None:
        """Recompute :attr:`_start_hz`/:attr:`_stop_hz` from the current zoom.

        The *actual* bin-edge frequencies, exactly as
        :func:`~qsorbit.ui.spectrum_zoom.visible_slice` returns them —
        not the requested window — which is what keeps the axis labels
        honest about which bins are really on screen. Depends only on
        the axis and the zoom, never on frame data, so the axis array
        can stand in as its own dummy "power" argument here.
        """
        _, start_hz, stop_hz = visible_slice(
            self._axis_hz, self._axis_hz, self._zoom_controller.zoom
        )
        self._start_hz = start_hz
        self._stop_hz = stop_hz

    def _render_visible_row(self, power_db: np.ndarray) -> np.ndarray:
        sliced, _start_hz, _stop_hz = visible_slice(
            power_db, self._axis_hz, self._zoom_controller.zoom
        )
        return render_row(sliced, self._render_width, self._scale, self._colormap)

    def _on_theme_changed(self, theme: Theme) -> None:
        self.set_colormap(theme.waterfall)

    def set_colormap(self, colormap: Colormap) -> None:
        """Restyle to a new theme's ramp, keeping every row of history.

        The history is re-rendered from ``_raw_frames`` rather than
        discarded, which is the whole reason that deck is the source of
        truth and ``_rendered_rows`` is only a cache of it. A theme
        switch mid-pass must not cost the operator the pass they are
        watching — and re-colouring what is already on screen is also
        the only way to *see* that the switch worked, rather than
        watching a black panel slowly refill in new colours.
        """
        self._colormap = colormap
        self._rebuild_rendered_rows()

    def _rebuild_rendered_rows(self) -> None:
        self._rendered_rows = deque(
            (self._render_visible_row(raw) for raw in self._raw_frames),
            maxlen=self._rendered_rows.maxlen,
        )
        self.update()

    def _on_zoom_changed(self) -> None:
        self._refresh_visible_edges()
        self._rebuild_rendered_rows()

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
            self._raw_frames.appendleft(frame.power_db)
            self._rendered_rows.appendleft(self._render_visible_row(frame.power_db))
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
        buffer = np.ascontiguousarray(np.stack(tuple(self._rendered_rows)))
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

        marker_hz = dc_spike_in_view(self._dc_hz, self._start_hz, self._stop_hz)
        if marker_hz is not None:
            self._paint_dc_marker(painter, image_rect, marker_hz)

        paint_frequency_axis(
            painter,
            image_rect,
            self._start_hz,
            self._stop_hz,
            AXIS_HEIGHT_PX,
            self.palette().windowText().color(),
        )

    def _paint_dc_marker(self, painter: QPainter, image_rect: QRect, marker_hz: float) -> None:
        """Mark the tuner's own zero-IF spike, per the "visual marker only"
        decision — see :func:`~qsorbit.ui.spectrum_zoom.dc_spike_in_view`'s
        own docstring for why this project does not remove the spike from
        the data itself.

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
        x = int(round(tick_position(marker_hz, self._start_hz, self._stop_hz, image_rect.width())))
        painter.setPen(theme_color(self._themes.current, "warn"))
        painter.drawLine(x, image_rect.top(), x, image_rect.bottom())
        painter.drawText(
            QRect(x + 3, image_rect.top() + 2, 24, 14), Qt.AlignmentFlag.AlignLeft, "DC"
        )

    # ------------------------------------------------------------------
    # Mouse: scroll-to-zoom, drag-to-pan
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt's spelling
        """Zoom in or out, anchored on the frequency under the cursor."""
        notches = event.angleDelta().y() / 120.0
        if notches == 0.0 or self.width() <= 0:
            event.ignore()
            return
        factor = WHEEL_ZOOM_FACTOR_PER_NOTCH**notches
        anchor_hz = self._hz_at_x(event.position().x())
        self._zoom_controller.zoom_by(factor, anchor_hz=anchor_hz)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt's spelling
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self._drag_start_x = event.position().x()
        self._drag_start_zoom = self._zoom_controller.zoom
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt's spelling
        if self._drag_start_x is None or self._drag_start_zoom is None or self.width() <= 0:
            super().mouseMoveEvent(event)
            return
        dx = event.position().x() - self._drag_start_x
        # Content follows the cursor, the usual "grab and drag" feel: the
        # frequency under the pointer when the drag started stays under
        # the pointer as it moves, which is why the center moves opposite
        # the drag direction.
        new_center_hz = (
            self._drag_start_zoom.center_hz - (dx / self.width()) * self._drag_start_zoom.span_hz
        )
        self._zoom_controller.pan_to(new_center_hz)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt's spelling
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        self._drag_start_x = None
        self._drag_start_zoom = None
        event.accept()

    def _hz_at_x(self, x: float) -> float:
        """The frequency under pixel column ``x`` of the plot, at the current zoom."""
        return self._start_hz + (x / self.width()) * (self._stop_hz - self._start_hz)
