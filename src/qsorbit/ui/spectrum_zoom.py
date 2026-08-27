"""Pan/zoom over a spectrum's frequency axis. No Qt import anywhere in here.

Same split as :mod:`qsorbit.ui.waterfall_render`: every decision about
*which frequencies are on screen* lives in plain functions and one frozen
value object, testable without PySide6 installed. The waterfall and the
new spectrum-line panel both slice down to the same visible window from
here, which is what lets a mouse gesture on either one move both -
:class:`ZoomController` (in :mod:`qsorbit.ui.zoom_controller`) is the one
piece of *shared* state; this module is the arithmetic underneath it.

**Why the slice is computed from the frequency axis, not from a bin
count.** A caller thinks in Hz - "center on the tracked downlink, show
30 kHz of it" - and the axis (:func:`~qsorbit.core.dsp.spectrum.frequency_axis_hz`)
is what turns that into which bins of a given frame to keep. Recomputing
the axis is deliberately avoided here: it is a property of the
:class:`~qsorbit.core.dsp.spectrum.SpectrumConfig`, identical for every
frame, so a caller passes it in once rather than this module rebuilding
it from the config on every frame - the same reasoning
:func:`~qsorbit.core.dsp.spectrum.frequency_axis_hz` itself already gives.

**Why the window is always clamped back inside the captured band.**
There is nothing to show past the edges of what the SDR actually
captured - zooming or panning past them would either show stale pixels
repeated forever or an empty gap, both of which read as data rather than
as "nothing there." :func:`clamp_zoom` is the one place that rule is
enforced, so every caller that changes the zoom (mouse wheel, drag, a
lock following the tracked frequency, a typed span) goes through it
rather than each reimplementing the edge case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

#: Floor on how narrow a zoom can get. Not chosen to protect the maths -
#: a one-bin window would slice and render fine - but to protect the
#: user from a runaway scroll gesture zooming to a span so narrow it
#: shows one repeated pixel and looks like the display broke. A typical
#: amateur FM channel is a few kHz wide; 1 kHz still leaves room to zoom
#: in past the whole channel if that is ever wanted.
MIN_ZOOM_SPAN_HZ: Final = 1_000.0


@dataclass(frozen=True)
class ZoomSpan:
    """One visible window on a spectrum's frequency axis.

    Args:
        center_hz: The middle of the visible window.
        span_hz: How wide the visible window is, in Hz.

    Raises:
        ValueError: If either value is not finite, or ``span_hz`` is not
            positive.
    """

    center_hz: float
    span_hz: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.center_hz):
            raise ValueError(f"center_hz must be finite, got {self.center_hz!r}.")
        if not math.isfinite(self.span_hz) or self.span_hz <= 0.0:
            raise ValueError(f"span_hz must be positive and finite, got {self.span_hz!r}.")

    @property
    def start_hz(self) -> float:
        """The window's low edge."""
        return self.center_hz - self.span_hz / 2.0

    @property
    def stop_hz(self) -> float:
        """The window's high edge."""
        return self.center_hz + self.span_hz / 2.0


def zoom_spanning(start_hz: float, stop_hz: float) -> ZoomSpan:
    """The window that exactly covers ``[start_hz, stop_hz]``.

    The shared arithmetic behind :func:`full_band_zoom` (which reads its
    edges off a frame's axis) and
    :class:`~qsorbit.ui.zoom_controller.ZoomController`'s own starting
    view (which is only ever given the two edges, not a whole axis array,
    since it holds no frames itself).
    """
    return ZoomSpan(center_hz=(start_hz + stop_hz) / 2.0, span_hz=stop_hz - start_hz)


def full_band_zoom(axis_hz: np.ndarray) -> ZoomSpan:
    """The zoom that shows the whole captured band - today's default view.

    Args:
        axis_hz: The full frequency axis, e.g. from
            :func:`~qsorbit.core.dsp.spectrum.frequency_axis_hz`, lowest
            frequency first.

    Every panel starts here: nothing is zoomed or panned until a user
    does something, so a session that never touches the controls looks
    exactly like it did before this feature existed.
    """
    return zoom_spanning(float(axis_hz[0]), float(axis_hz[-1]))


def clamp_zoom(
    zoom: ZoomSpan,
    band_start_hz: float,
    band_stop_hz: float,
    *,
    min_span_hz: float = MIN_ZOOM_SPAN_HZ,
) -> ZoomSpan:
    """Keep ``zoom`` inside the captured band and above the minimum span.

    Args:
        zoom: The requested window.
        band_start_hz: The captured band's low edge.
        band_stop_hz: The captured band's high edge.
        min_span_hz: Floor on ``span_hz`` - see :data:`MIN_ZOOM_SPAN_HZ`.

    Returns:
        A window with ``span_hz`` clamped to ``[min_span_hz, band width]``
        and ``center_hz`` clamped so the window never runs past either
        edge of the band. Span is clamped first, so a caller asking for
        a wider span than the band has ever gets the honest "as wide as
        there is," not a window that then gets recentred to fit a span
        it was never granted.

    Raises:
        ValueError: If ``band_stop_hz`` is not above ``band_start_hz``.
    """
    band_width_hz = band_stop_hz - band_start_hz
    if band_width_hz <= 0.0:
        raise ValueError(
            f"band_stop_hz ({band_stop_hz!r}) must be above band_start_hz ({band_start_hz!r})."
        )

    span_hz = min(max(zoom.span_hz, min_span_hz), band_width_hz)
    half = span_hz / 2.0
    center_hz = min(max(zoom.center_hz, band_start_hz + half), band_stop_hz - half)
    return ZoomSpan(center_hz=center_hz, span_hz=span_hz)


def visible_slice(
    power_db: np.ndarray, axis_hz: np.ndarray, zoom: ZoomSpan
) -> tuple[np.ndarray, float, float]:
    """Slice one frame down to what ``zoom`` shows.

    Args:
        power_db: One frame's power values, ordered to match ``axis_hz``.
        axis_hz: The full frequency axis - see
            :func:`~qsorbit.core.dsp.spectrum.frequency_axis_hz`.
        zoom: The window to show. Callers should pass this through
            :func:`clamp_zoom` first - this function clamps its own
            index range defensively, but a zoom entirely outside the
            band would otherwise slice to nothing.

    Returns:
        ``(sliced_power_db, start_hz, stop_hz)`` - the sliced values, and
        the *actual* frequencies of the first and last bin kept, which
        are what tick labels should be built from rather than the
        requested ``zoom.start_hz``/``zoom.stop_hz``: the axis is a fixed
        set of bins, and the true edges are whichever bin ends up first
        and last, not the exact Hz a mouse gesture asked for.

    Raises:
        ValueError: If ``power_db`` and ``axis_hz`` are not the same
            length, or either is empty.
    """
    if power_db.shape != axis_hz.shape:
        raise ValueError(
            f"power_db and axis_hz must be the same shape, got {power_db.shape!r} "
            f"and {axis_hz.shape!r}."
        )
    if axis_hz.shape[0] == 0:
        raise ValueError("axis_hz must not be empty.")

    lo = int(np.searchsorted(axis_hz, zoom.start_hz, side="left"))
    hi = int(np.searchsorted(axis_hz, zoom.stop_hz, side="right"))
    lo = min(max(lo, 0), axis_hz.shape[0] - 1)
    hi = min(max(hi, lo + 1), axis_hz.shape[0])
    return power_db[lo:hi], float(axis_hz[lo]), float(axis_hz[hi - 1])


def dc_spike_in_view(tuner_center_hz: float, start_hz: float, stop_hz: float) -> float | None:
    """Where the tuner's own DC spike falls, if it is inside ``[start_hz, stop_hz]``.

    Args:
        tuner_center_hz: The RF centre frequency the tuner was actually
            configured to - :attr:`~qsorbit.core.dsp.spectrum.SpectrumConfig.center_freq_hz`
            as built from :attr:`~qsorbit.core.sdr.device.AppliedSettings.center_hz`.
            The spike sits exactly here: it is a property of the tuner's
            own zero-IF architecture, not of the signal, and every
            :class:`~qsorbit.core.dsp.spectrum.SpectrumConfig` this
            project builds already carries it as ``center_freq_hz`` - see
            :func:`~qsorbit.core.dsp.spectrum.frequency_axis_hz`, whose
            baseband axis is symmetric around zero before this offset is
            added, putting the spike at the exact middle bin of the full
            band.
        start_hz: The visible window's low edge.
        stop_hz: The visible window's high edge.

    Returns:
        ``tuner_center_hz`` if it falls within the visible window, else
        ``None``. **This project deliberately does not remove the spike
        from the data** - it tunes with an offset from every downlink it
        listens to specifically so the spike lands away from the signal
        (see ``--offset`` in ``qsorbit receive --help``), so a caller
        zoomed in on the tracked frequency will normally get ``None``
        here. It only shows up in the wide, unzoomed view, which is
        exactly where marking it - rather than silently leaving an
        unexplained bright column that looks like a carrier - matters.
    """
    if start_hz <= tuner_center_hz <= stop_hz:
        return tuner_center_hz
    return None


def rescale_around(zoom: ZoomSpan, factor: float, anchor_hz: float | None = None) -> ZoomSpan:
    """The arithmetic behind a scroll-to-zoom gesture.

    Args:
        zoom: The window before the gesture.
        factor: Multiplier on ``span_hz`` - below 1 zooms in, above 1
            zooms out. Must be finite and positive; the caller (a mouse
            wheel handler) is responsible for turning "notches scrolled"
            into this ratio.
        anchor_hz: The frequency to hold fixed in place while the window
            scales around it - e.g. whatever frequency is under the
            mouse cursor. Defaults to ``zoom.center_hz``, which is a
            symmetric zoom that leaves the center exactly where it was.

    Returns:
        The rescaled window. Not yet clamped to any captured band or
        minimum span - pass the result through :func:`clamp_zoom`.

    Raises:
        ValueError: If ``factor`` is not finite and positive.
    """
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"factor must be positive and finite, got {factor!r}.")
    anchor = zoom.center_hz if anchor_hz is None else anchor_hz
    start_hz = anchor - (anchor - zoom.start_hz) * factor
    stop_hz = anchor + (zoom.stop_hz - anchor) * factor
    return zoom_spanning(start_hz, stop_hz)


# ----------------------------------------------------------------------
# Orchestration for ZoomController (qsorbit.ui.zoom_controller)
#
# Each function below answers one question a user gesture asks -- "what
# should the window become if the span spinbox changes / the user drags
# / the user scrolls / a fresh tracked frequency arrives" -- as plain,
# Qt-free arithmetic. ZoomController's own job is reduced to calling
# one of these, comparing the result to what it already had, and
# emitting its Signal if anything changed; no zoom/pan/lock *decision*
# is made in the Qt layer, matching this project's established
# render/logic split. A function returning ``None`` means "no-op, don't
# emit" -- the caller was asking for something the current lock state
# makes meaningless, not something that failed.
# ----------------------------------------------------------------------


def next_zoom_for_span(
    zoom: ZoomSpan, band_start_hz: float, band_stop_hz: float, span_hz: float
) -> ZoomSpan:
    """What the window becomes when the span control changes, center held fixed."""
    return clamp_zoom(
        ZoomSpan(center_hz=zoom.center_hz, span_hz=span_hz), band_start_hz, band_stop_hz
    )


def next_zoom_for_pan(
    zoom: ZoomSpan,
    locked: bool,
    band_start_hz: float,
    band_stop_hz: float,
    center_hz: float,
) -> ZoomSpan | None:
    """What the window becomes when a drag pans to ``center_hz``.

    ``None`` while locked: the lock is what owns the center, so a manual
    pan would only be overwritten by the next tracked-frequency update -
    applying it anyway would flash the window to a spot it immediately
    snaps back from, which reads as a display glitch rather than an
    ignored gesture.
    """
    if locked:
        return None
    return clamp_zoom(
        ZoomSpan(center_hz=center_hz, span_hz=zoom.span_hz), band_start_hz, band_stop_hz
    )


def next_zoom_for_scroll(
    zoom: ZoomSpan,
    locked: bool,
    band_start_hz: float,
    band_stop_hz: float,
    factor: float,
    anchor_hz: float | None = None,
) -> ZoomSpan:
    """What the window becomes when the mouse wheel scrolls.

    While locked, ``anchor_hz`` is ignored in favour of the window's own
    current center - the lock owns where the window is centered, so only
    the span is the user's to adjust with the wheel; honouring a cursor
    anchor here would fight the next tracked-frequency update for
    control of the center, the same reasoning :func:`next_zoom_for_pan`
    applies to a drag.
    """
    effective_anchor = None if locked else anchor_hz
    return clamp_zoom(rescale_around(zoom, factor, effective_anchor), band_start_hz, band_stop_hz)


def next_zoom_for_follow(
    zoom: ZoomSpan,
    locked: bool,
    band_start_hz: float,
    band_stop_hz: float,
    tracked_hz: float,
) -> ZoomSpan | None:
    """What the window becomes when a fresh tracked frequency arrives.

    ``None`` while not locked: an unlocked view is the user's own to
    pan and zoom, and a stream of tracked-frequency updates should not
    keep dragging it back toward the downlink whenever one is heard.
    """
    if not locked:
        return None
    return clamp_zoom(
        ZoomSpan(center_hz=tracked_hz, span_hz=zoom.span_hz), band_start_hz, band_stop_hz
    )
