"""Shared zoom/pan/lock state for the waterfall and the spectrum-line panel.

Chunk I's waterfall-zoom item asked for two panels - the existing
waterfall and a new SDR#-style line-trace spectrum - that both zoom and
pan over the *same* visible window of the *same* frequency axis, so a
mouse gesture on either one moves both. This module is the one piece of
state they share: one :class:`ZoomController` per receive session's
spectrum, handed to both widgets at construction, exactly the way both
already share one :class:`~qsorbit.core.receive.ReceiveSession`.

**Every public method here is a thin call into one of the pure
``next_zoom_for_*`` functions in** :mod:`qsorbit.ui.spectrum_zoom`
**plus a change-detecting** ``Signal`` **emit.** No zoom/pan/lock
*decision* is made in this class - not "should a drag move the center
while locked," not "does the wheel's anchor apply," not "is this the
same window as before." Those are all questions
:mod:`qsorbit.ui.spectrum_zoom` answers as plain, Qt-free arithmetic,
importable and testable with no ``PySide6`` in sight - the same
render/logic split :mod:`qsorbit.ui.waterfall_render` established for
the waterfall itself. This class exists only because that arithmetic
has to live *somewhere* Qt can connect a signal to, and because two
independent widgets polling a session for a shared answer would drift
the moment one of them was updated and the other forgotten.
"""

from __future__ import annotations

from typing import Final, Protocol

from PySide6.QtCore import QObject, QTimer, Signal

from qsorbit.ui.spectrum_zoom import (
    ZoomSpan,
    next_zoom_for_follow,
    next_zoom_for_pan,
    next_zoom_for_scroll,
    next_zoom_for_span,
    zoom_spanning,
)

#: How often the controller polls a ``tracked_frequency_source`` for a
#: fresh frequency to :meth:`~ZoomController.follow`, in milliseconds.
#: Matched to :data:`~qsorbit.ui.quieting_widget.DEFAULT_POLL_INTERVAL_MS`
#: rather than the spectrum panels' faster 50 ms poll - a tracked
#: frequency only moves as fast as the Doppler correction itself
#: recomputes it (once per tracking tick, on the order of a second), so
#: polling at spectrum-frame rate would just be 19 out of 20 reads
#: finding nothing new.
DEFAULT_FOLLOW_POLL_INTERVAL_MS: Final = 200


class TrackedFrequencySource(Protocol):
    """Anything that can report the frequency currently being tracked.

    Declared structurally, the same reasoning as
    :class:`~qsorbit.ui.waterfall_widget.FrameSource` and
    :class:`~qsorbit.ui.quieting_widget.QuietingSource` - a test double
    satisfies it without subclassing anything.
    """

    @property
    def live_tracked_frequency_hz(self) -> float | None:
        """The tracked downlink's true RF frequency right now, or ``None``
        before a tracking sample has ever arrived - see
        :attr:`~qsorbit.core.receive.ReceiveSession.live_tracked_frequency_hz`."""
        ...


class ZoomController(QObject):
    """The shared visible-window state a waterfall and a line-spectrum panel poll.

    Args:
        band_start_hz: The captured band's low edge - see
            :func:`~qsorbit.ui.spectrum_zoom.clamp_zoom`.
        band_stop_hz: The captured band's high edge.
        tracked_frequency_source: Where a live tracked frequency comes
            from, e.g. a :class:`~qsorbit.core.receive.ReceiveSession`.
            When given, the controller polls it itself
            (:data:`DEFAULT_FOLLOW_POLL_INTERVAL_MS`) and calls
            :meth:`follow` on every reading - so turning
            :attr:`locked` on snaps into place on its own, with no
            widget needing to know tracking exists at all. Omit for a
            controller that only ever moves by mouse or spinbox - a
            captured file replayed with no live tracking, say - in which
            case :meth:`follow` is simply never called and
            :attr:`locked` stays cosmetic.
        follow_poll_interval_ms: How often to poll
            ``tracked_frequency_source``. Ignored when that source is
            omitted.
        parent: Standard Qt ownership; usually left ``None`` and the
            controller kept alive by whoever constructs it, the same as
            :class:`~qsorbit.ui.waterfall_widget.WaterfallWidget` takes
            no parent of its own frame source.

    Starts showing the whole captured band, unlocked - a session that
    never touches the new controls looks exactly as it did before this
    feature existed, matching
    :func:`~qsorbit.ui.spectrum_zoom.full_band_zoom`'s own reasoning.
    """

    #: Emitted after :attr:`zoom` or :attr:`locked` actually changes.
    #: Never emitted for a call that turned out to be a no-op - a pan
    #: while locked, a scroll or follow that clamped back to the window
    #: it already had - so a connected widget can repaint unconditionally
    #: on this signal without needing its own before/after comparison.
    changed = Signal()

    def __init__(
        self,
        band_start_hz: float,
        band_stop_hz: float,
        *,
        tracked_frequency_source: TrackedFrequencySource | None = None,
        follow_poll_interval_ms: int = DEFAULT_FOLLOW_POLL_INTERVAL_MS,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._band_start_hz = band_start_hz
        self._band_stop_hz = band_stop_hz
        self._zoom = zoom_spanning(band_start_hz, band_stop_hz)
        self._locked = False

        self._tracked_source = tracked_frequency_source
        self._follow_timer: QTimer | None = None
        if tracked_frequency_source is not None:
            self._follow_timer = QTimer(self)
            self._follow_timer.setInterval(follow_poll_interval_ms)
            self._follow_timer.timeout.connect(self._poll_tracked_frequency)
            self._follow_timer.start()

    @property
    def zoom(self) -> ZoomSpan:
        """The window currently visible."""
        return self._zoom

    @property
    def locked(self) -> bool:
        """Whether the window is following a tracked frequency."""
        return self._locked

    @property
    def band_start_hz(self) -> float:
        """The captured band's low edge, fixed for the controller's life."""
        return self._band_start_hz

    @property
    def band_stop_hz(self) -> float:
        """The captured band's high edge, fixed for the controller's life."""
        return self._band_stop_hz

    def set_locked(self, locked: bool) -> None:
        """Turn following a tracked frequency on or off.

        Takes effect starting with the *next* :meth:`follow` call, not
        immediately - this method only flips the flag. Callers should
        keep calling :meth:`follow` on every fresh tracked frequency
        regardless of lock state (see that method's own docstring), so
        turning the lock on mid-session snaps into place on the very
        next update rather than needing a frequency threaded through
        here as well.
        """
        if locked == self._locked:
            return
        self._locked = locked
        self.changed.emit()

    def set_span_hz(self, span_hz: float) -> None:
        """Change how wide the visible window is, e.g. from a spinbox. Center held fixed."""
        self._adopt(
            next_zoom_for_span(self._zoom, self._band_start_hz, self._band_stop_hz, span_hz)
        )

    def pan_to(self, center_hz: float) -> None:
        """Move the visible window to center on ``center_hz``, e.g. from a drag.

        A no-op while :attr:`locked` - see
        :func:`~qsorbit.ui.spectrum_zoom.next_zoom_for_pan`.
        """
        self._adopt(
            next_zoom_for_pan(
                self._zoom, self._locked, self._band_start_hz, self._band_stop_hz, center_hz
            )
        )

    def zoom_by(self, factor: float, anchor_hz: float | None = None) -> None:
        """Scale the visible span by ``factor``, e.g. from a mouse wheel notch.

        Args:
            factor: Below 1 zooms in, above 1 zooms out - see
                :func:`~qsorbit.ui.spectrum_zoom.rescale_around`.
            anchor_hz: The frequency to hold in place, e.g. under the
                cursor. Ignored while :attr:`locked` - see
                :func:`~qsorbit.ui.spectrum_zoom.next_zoom_for_scroll`.
        """
        self._adopt(
            next_zoom_for_scroll(
                self._zoom, self._locked, self._band_start_hz, self._band_stop_hz, factor, anchor_hz
            )
        )

    def follow(self, tracked_hz: float) -> None:
        """Re-center on ``tracked_hz`` if :attr:`locked`, a no-op otherwise.

        Meant to be called every time a fresh tracked frequency becomes
        available - e.g. every poll of
        :attr:`~qsorbit.core.receive.ReceiveSession.live_tracked_frequency_hz`
        - whether or not the lock happens to be on right now. Calling it
        unconditionally, rather than only while locked, is what lets
        :meth:`set_locked` stay a plain flag flip: the very next call
        here after the lock is turned on carries the snap into place.
        """
        self._adopt(
            next_zoom_for_follow(
                self._zoom, self._locked, self._band_start_hz, self._band_stop_hz, tracked_hz
            )
        )

    def _adopt(self, zoom: ZoomSpan | None) -> None:
        """Take ``zoom`` as the new visible window and emit, unless it's a no-op.

        ``None`` (a gesture the current lock state makes meaningless) and
        an unchanged window (already clamped to what a gesture asked for)
        are both silently ignored - see :attr:`changed`'s own docstring
        for why that matters to callers.
        """
        if zoom is None or zoom == self._zoom:
            return
        self._zoom = zoom
        self.changed.emit()

    def _poll_tracked_frequency(self) -> None:
        """Timer callback: read the source once, hand it to :meth:`follow`.

        A no-op read (``None`` - no tracking sample yet) is simply
        skipped rather than treated as a reason to stop polling; the
        next tick tries again, the same tolerance
        :attr:`~qsorbit.core.receive.ReceiveSession.live_tracked_frequency_hz`
        itself documents for "not yet" versus "never will be."
        """
        assert self._tracked_source is not None  # only ever connected when set
        tracked_hz = self._tracked_source.live_tracked_frequency_hz
        if tracked_hz is not None:
            self.follow(tracked_hz)

    def stop(self) -> None:
        """Stop polling ``tracked_frequency_source``, if one was given.

        Does not touch :attr:`zoom` or :attr:`locked` - a stopped
        controller still answers every read exactly as it last stood,
        it simply will not move on its own any more. A controller built
        without a tracked-frequency source has nothing to stop; calling
        this is then a harmless no-op, so every caller holding a
        ``ZoomController | None`` can call it uniformly, the same
        pattern :class:`~qsorbit.ui.instrument_window.InstrumentWindow`'s
        own ``closeEvent`` already uses across every panel it holds.
        """
        if self._follow_timer is not None:
            self._follow_timer.stop()
