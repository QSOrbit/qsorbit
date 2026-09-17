"""Pass prediction: when is a target above the horizon, and where.

A coarse SGP4 sweep across a search window finds where a target's
elevation crosses the visibility threshold; a bisection refines that
crossing to sub-second AOS/LOS times, and a bounded scalar minimizer
finds time of closest approach (TCA) the same way. All of it reuses
:meth:`Target.topocentric_state
<qsorbit.core.tracker.target.Target.topocentric_state>` -- the same
per-instant SGP4 computation Doppler correction already trusts, called
many times rather than a new bulk-propagation path, since nothing about
this chunk is performance-gated. See :func:`predict_passes`.

A pass can be filtered by a flat minimum elevation, or by a real
:class:`~qsorbit.core.horizon.HorizonMask` -- station data landing in
the same chunk as this module. Applying it *inside* the search, rather
than filtering a list of unfiltered passes afterward, matters because
the mask's threshold moves with azimuth as a pass crosses the sky: a
pass that never once climbs above what the mask allows at its own
current bearing has to be found by comparing against a moving target,
not a constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from scipy.optimize import minimize_scalar

from qsorbit.core.geometry import AzEl
from qsorbit.core.horizon import HorizonMask
from qsorbit.core.tracker._shared import require_timezone_aware
from qsorbit.core.tracker.observer import ObserverLocation
from qsorbit.core.tracker.sun import is_illuminated, sun_elevation_deg
from qsorbit.core.tracker.target import Target

#: Coarse sweep step, in seconds. Fine enough that a real LEO pass (the
#: shortest visibility windows this project cares about run several
#: minutes) cannot rise and set again between two samples; coarse
#: enough to keep a multi-hour search from being thousands of SGP4
#: calls for nothing.
DEFAULT_STEP_S = 30.0

#: How many bisection halvings refine an AOS/LOS crossing. 18 halvings
#: of the default 30 s coarse step is under a millisecond of residual
#: uncertainty -- far finer than anything downstream needs, and cheap,
#: since each halving only evaluates one more point.
_REFINE_ITERATIONS = 18

#: How high the Sun must be, in degrees, before an observer's sky
#: counts as "dark" for naked-eye visibility. -6 degrees is civil
#: twilight -- past sunset by eye, and on the brighter end of what
#: satellite-visibility guides commonly use. Not yet calibrated against
#: any of this station's own visual observations.
DEFAULT_TWILIGHT_SUN_ELEVATION_DEG = -6.0


@dataclass(frozen=True)
class PassEvent:
    """A target's sky position at one instant during a pass.

    ``PassEvent`` is a value object: immutable and comparable by value.

    Args:
        time: The UTC instant.
        sky_position: Where the target was in the sky at ``time``.
    """

    time: datetime
    sky_position: AzEl


@dataclass(frozen=True)
class VisibleWindow:
    """When during a pass a satellite can actually be seen by eye.

    Naked-eye visibility needs two things at once: the satellite lit by
    the Sun, and the observer's own sky dark. Neither holds for a whole
    pass in general -- a satellite can rise already sunlit and fly into
    Earth's shadow partway across, which is what the station watched
    SL-14 R/B do at 21:34 on the evening this feature was first
    proposed. A single flag at closest approach cannot say that, which
    is why this carries an interval instead.

    ``VisibleWindow`` is a value object: immutable and comparable by
    value.

    Args:
        begins: When the satellite becomes visible -- either the moment
            it leaves Earth's shadow, or the pass's own AOS if it was
            already lit when it rose.
        ends: When it stops being visible -- shadow entry, or LOS.
    """

    begins: PassEvent
    ends: PassEvent

    @property
    def duration_s(self) -> float:
        """How long the satellite is visible for, in seconds."""
        return (self.ends.time - self.begins.time).total_seconds()


@dataclass(frozen=True)
class Pass:
    """One continuous period a target is above the visibility threshold.

    ``Pass`` is a value object: immutable and comparable by value.

    Args:
        aos: Acquisition of signal -- where and when the target first
            crossed above the threshold.
        los: Loss of signal -- where and when it dropped back below.
        tca: Time of closest approach -- where and when elevation was
            highest during the pass.
        max_elevation_deg: ``tca.sky_position.elevation``, pulled to
            the top level since it's what most callers actually sort
            and filter on.
        az_track: The target's sky position sampled across the pass,
            ``aos`` to ``los`` inclusive -- for anything that needs the
            shape of the pass rather than just its three named points.
            The parked flip-mode pass planner is the obvious future
            consumer.
        illuminated: Whether the target was in sunlight, with the
            observer's own sky dark enough to see it, at ``tca`` --
            or ``None`` if illumination wasn't requested for this
            search. This is one flag for the whole pass, evaluated at
            the moment elevation (and so, typically, brightness)
            peaks; a target can enter or leave Earth's shadow mid-pass,
            and a caller that needs the fade-in/fade-out shape of that
            should sample :func:`~qsorbit.core.tracker.sun.is_illuminated`
            itself across ``az_track``'s own times.
            visible_window: When during the pass the satellite is actually
            visible, or ``None`` if it never is -- also ``None`` when
            illumination wasn't requested. This is the honest form of
            :attr:`illuminated`, which only answers for the single
            instant of closest approach; a pass can be visible for
            ninety seconds of its six minutes and ``illuminated`` would
            report whichever happened to be true at TCA. Prefer this.
    """

    aos: PassEvent
    los: PassEvent
    tca: PassEvent
    max_elevation_deg: float
    az_track: tuple[PassEvent, ...]
    illuminated: bool | None = None
    visible_window: VisibleWindow | None = None


def _effective_min_elevation_deg(
    sky_position: AzEl,
    *,
    min_elevation_deg: float,
    horizon_mask: HorizonMask | None,
) -> float:
    """The visibility threshold at ``sky_position``'s azimuth right now."""
    if horizon_mask is not None:
        return horizon_mask.min_elevation_at(sky_position.azimuth)
    return min_elevation_deg


def _margin_at(
    target: Target,
    observer: ObserverLocation,
    time: datetime,
    *,
    min_elevation_deg: float,
    horizon_mask: HorizonMask | None,
) -> float:
    """Elevation above (positive) or below (negative) the threshold at ``time``."""
    sky_position = target.topocentric_state(observer, time).sky_position
    threshold = _effective_min_elevation_deg(
        sky_position, min_elevation_deg=min_elevation_deg, horizon_mask=horizon_mask
    )
    return sky_position.elevation - threshold


def _refine_crossing(
    target: Target,
    observer: ObserverLocation,
    negative_time: datetime,
    positive_time: datetime,
    *,
    min_elevation_deg: float,
    horizon_mask: HorizonMask | None,
) -> datetime:
    """Bisect between a time below threshold and one at or above it.

    Args:
        negative_time: A time where the margin (elevation minus
            threshold) is known negative.
        positive_time: A time where the margin is known
            non-negative. Whichever of the two comes chronologically
            first does not matter to the bisection -- only that they
            land on opposite sides of the threshold.

    Returns:
        The instant partway between them where the margin crosses
        zero, to within about
        ``|positive_time - negative_time| / 2**_REFINE_ITERATIONS``.
    """
    for _ in range(_REFINE_ITERATIONS):
        midpoint = negative_time + (positive_time - negative_time) / 2
        margin = _margin_at(
            target,
            observer,
            midpoint,
            min_elevation_deg=min_elevation_deg,
            horizon_mask=horizon_mask,
        )
        if margin < 0.0:
            negative_time = midpoint
        else:
            positive_time = midpoint
    return negative_time + (positive_time - negative_time) / 2


def _find_tca(
    target: Target, observer: ObserverLocation, aos_time: datetime, los_time: datetime
) -> PassEvent:
    """Find the instant of highest elevation between ``aos_time`` and ``los_time``."""
    duration_s = (los_time - aos_time).total_seconds()

    def negative_elevation(offset_s: float) -> float:
        time = aos_time + timedelta(seconds=offset_s)
        return -target.topocentric_state(observer, time).sky_position.elevation

    result = minimize_scalar(negative_elevation, bounds=(0.0, duration_s), method="bounded")
    tca_time = aos_time + timedelta(seconds=result.x)
    sky_position = target.topocentric_state(observer, tca_time).sky_position
    return PassEvent(time=tca_time, sky_position=sky_position)


def _build_az_track(
    target: Target,
    observer: ObserverLocation,
    aos_time: datetime,
    los_time: datetime,
    *,
    track_step_s: float,
) -> tuple[PassEvent, ...]:
    """Sample ``target``'s sky position from ``aos_time`` to ``los_time`` inclusive."""
    track: list[PassEvent] = []
    step = timedelta(seconds=track_step_s)
    sample_time = aos_time
    while sample_time < los_time:
        sky_position = target.topocentric_state(observer, sample_time).sky_position
        track.append(PassEvent(time=sample_time, sky_position=sky_position))
        sample_time += step
    los_sky_position = target.topocentric_state(observer, los_time).sky_position
    track.append(PassEvent(time=los_time, sky_position=los_sky_position))
    return tuple(track)


def _is_visually_illuminated(
    target: Target,
    observer: ObserverLocation,
    event: PassEvent,
    *,
    twilight_sun_elevation_deg: float,
) -> bool:
    """Whether ``event`` is both sunlit and dark-sky-visible.

    Requires ``target.state_at()`` -- a satellite's own ECI position,
    for the shadow test in
    :func:`~qsorbit.core.tracker.sun.is_illuminated`. A generic
    :class:`~qsorbit.core.tracker.target.Target` (a star, the Moon)
    doesn't have this method, and Earth's-shadow illumination isn't
    the relevant question for one anyway -- only a concrete
    :class:`~qsorbit.core.tracker.satellite.Satellite` does.

    Raises:
        TypeError: If ``target`` has no ``state_at`` method.
    """
    state_at = getattr(target, "state_at", None)
    if state_at is None:
        raise TypeError(
            "include_illumination=True needs target.state_at() -- a satellite's own "
            "ECI position, for the Earth-shadow test -- but "
            f"{type(target).__name__!r} has no such method. Illumination geometry "
            "is only meaningful for a concrete Satellite, not a generic Target."
        )
    satellite_gcrs_km = state_at(event.time).position_km
    return is_illuminated(satellite_gcrs_km, event.time) and (
        sun_elevation_deg(observer, event.time) <= twilight_sun_elevation_deg
    )


#: How finely the visible-window search samples a pass before refining,
#: in seconds. A satellite crosses Earth's shadow boundary sharply, so
#: this only has to be fine enough not to step over a short window
#: entirely; the boundary itself is then found by bisection. Fifteen
#: seconds puts roughly forty samples across a typical LEO pass.
VISIBILITY_STEP_S = 15.0

#: How many bisection halvings refine each edge of a visible window.
#: Twelve halvings of a 15 s step is under 4 ms -- far finer than the
#: minute-resolution anything displays it at, and cheap.
VISIBILITY_REFINEMENTS = 12


def _refine_visibility_edge(
    is_visible,
    dark_time: datetime,
    lit_time: datetime,
) -> datetime:
    """Bisect between a known-dark and a known-lit instant.

    Unlike :func:`_refine_crossing`, which brackets a sign change in a
    continuous margin, this brackets a change in a *boolean*. There is
    no value to interpolate, so plain bisection is all that is
    available -- and all that is needed, since the underlying shadow
    boundary is geometrically sharp.
    """
    for _ in range(VISIBILITY_REFINEMENTS):
        midpoint = dark_time + (lit_time - dark_time) / 2
        if is_visible(midpoint):
            lit_time = midpoint
        else:
            dark_time = midpoint
    return lit_time


def _visible_window(
    target: Target,
    observer: ObserverLocation,
    aos_time: datetime,
    los_time: datetime,
    *,
    twilight_sun_elevation_deg: float,
) -> VisibleWindow | None:
    """Find when during a pass the satellite is visible to the naked eye.

    Samples the pass, finds the contiguous stretches where the target is
    both sunlit and the observer's sky is dark, and refines the edges of
    the longest one by bisection.

    **The longest stretch, not the first.** A pass can in principle
    produce more than one -- a grazing shadow crossing, or a long pass
    straddling the twilight threshold -- and reporting the first would
    hand the operator a ten-second sliver while ignoring the four
    minutes after it. Rare, but silently wrong when it happens.

    Returns:
        The window, or ``None`` if the satellite is never visible during
        this pass.

    Raises:
        TypeError: If ``target`` has no ``state_at`` method, via
            :func:`_is_visually_illuminated`.
    """

    def is_visible(when: datetime) -> bool:
        return _is_visually_illuminated(
            target,
            observer,
            PassEvent(
                time=when, sky_position=target.topocentric_state(observer, when).sky_position
            ),
            twilight_sun_elevation_deg=twilight_sun_elevation_deg,
        )

    duration_s = (los_time - aos_time).total_seconds()
    steps = max(2, int(duration_s // VISIBILITY_STEP_S) + 1)
    samples = [aos_time + timedelta(seconds=duration_s * n / steps) for n in range(steps + 1)]
    flags = [is_visible(when) for when in samples]

    # Contiguous runs of True, as (first_index, last_index) inclusive.
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, lit in enumerate(flags):
        if lit and start is None:
            start = index
        elif not lit and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))

    if not runs:
        return None

    first_index, last_index = max(runs, key=lambda run: run[1] - run[0])

    # An edge that sits on the pass's own boundary is not a shadow
    # crossing -- the satellite was already lit when it rose, or still
    # lit when it set -- so there is nothing to bisect toward.
    begins_at = (
        samples[first_index]
        if first_index == 0
        else _refine_visibility_edge(is_visible, samples[first_index - 1], samples[first_index])
    )
    ends_at = (
        samples[last_index]
        if last_index == len(samples) - 1
        else _refine_visibility_edge(is_visible, samples[last_index + 1], samples[last_index])
    )

    return VisibleWindow(
        begins=PassEvent(
            time=begins_at,
            sky_position=target.topocentric_state(observer, begins_at).sky_position,
        ),
        ends=PassEvent(
            time=ends_at,
            sky_position=target.topocentric_state(observer, ends_at).sky_position,
        ),
    )


def _build_pass(
    target: Target,
    observer: ObserverLocation,
    aos_time: datetime,
    los_time: datetime,
    *,
    track_step_s: float,
    include_illumination: bool,
    twilight_sun_elevation_deg: float,
) -> Pass:
    aos_sky = target.topocentric_state(observer, aos_time).sky_position
    los_sky = target.topocentric_state(observer, los_time).sky_position
    az_track = _build_az_track(target, observer, aos_time, los_time, track_step_s=track_step_s)
    tca_event = _find_tca(target, observer, aos_time, los_time)

    illuminated: bool | None = None
    window: VisibleWindow | None = None
    if include_illumination:
        illuminated = _is_visually_illuminated(
            target, observer, tca_event, twilight_sun_elevation_deg=twilight_sun_elevation_deg
        )
        window = _visible_window(
            target,
            observer,
            aos_time,
            los_time,
            twilight_sun_elevation_deg=twilight_sun_elevation_deg,
        )

    return Pass(
        aos=PassEvent(time=aos_time, sky_position=aos_sky),
        los=PassEvent(time=los_time, sky_position=los_sky),
        tca=tca_event,
        max_elevation_deg=tca_event.sky_position.elevation,
        az_track=az_track,
        illuminated=illuminated,
        visible_window=window,
    )


def predict_passes(
    target: Target,
    observer: ObserverLocation,
    start: datetime,
    end: datetime,
    *,
    min_elevation_deg: float = 0.0,
    horizon_mask: HorizonMask | None = None,
    step_s: float = DEFAULT_STEP_S,
    track_step_s: float = 60.0,
    include_illumination: bool = False,
    twilight_sun_elevation_deg: float = DEFAULT_TWILIGHT_SUN_ELEVATION_DEG,
) -> list[Pass]:
    """Find every complete pass of ``target`` between ``start`` and ``end``.

    Args:
        target: What to predict passes for -- a
            :class:`~qsorbit.core.tracker.satellite.Satellite` or
            anything else satisfying
            :class:`~qsorbit.core.tracker.target.Target`.
        observer: Where the antenna is.
        start: Search window start, timezone-aware.
        end: Search window end, timezone-aware. Must be after
            ``start``.
        min_elevation_deg: The flat visibility threshold, used when
            ``horizon_mask`` is ``None``. Defaults to ``0.0`` -- the
            geometric horizon, ignoring anything the station's own
            site blocks.
        horizon_mask: When given, replaces ``min_elevation_deg``
            entirely with the mask's own azimuth-dependent threshold --
            see :class:`~qsorbit.core.horizon.HorizonMask`.
        step_s: Coarse sweep step, in seconds. Must be positive.
        track_step_s: Spacing of ``Pass.az_track`` samples, in
            seconds. Must be positive.
        include_illumination: When ``True``, computes
            :attr:`Pass.illuminated` at each pass's TCA -- and
            requires ``target`` to provide ``state_at()`` (see
            :class:`~qsorbit.core.tracker.satellite.Satellite`).
            Left off by default, since it costs an extra Sun-position
            evaluation per pass that most callers -- anything not
            asking "will I be able to *see* this" -- don't need.
        twilight_sun_elevation_deg: How high the Sun may be at the
            observer, in degrees, before the sky counts as too bright
            for naked-eye visibility. Only consulted when
            ``include_illumination`` is ``True``.

    Returns:
        Every pass that both begins and ends inside ``[start, end]``,
        in chronological order. A pass already underway at ``start``,
        or one that has not yet ended by ``end``, is not returned --
        its AOS or LOS falls outside the window this function was
        asked to search, so reporting a half-known pass would be
        reporting something this function didn't actually check.

    Raises:
        ValueError: If ``start`` or ``end`` is naive, ``end`` is not
            after ``start``, or ``step_s``/``track_step_s`` isn't
            positive.
        TypeError: If ``include_illumination`` is ``True`` and
            ``target`` has no ``state_at`` method.
    """
    require_timezone_aware(start)
    require_timezone_aware(end)
    if end <= start:
        raise ValueError(f"end ({end.isoformat()}) must be after start ({start.isoformat()}).")
    if step_s <= 0.0:
        raise ValueError(f"step_s must be positive, got {step_s}.")
    if track_step_s <= 0.0:
        raise ValueError(f"track_step_s must be positive, got {track_step_s}.")

    passes: list[Pass] = []
    step = timedelta(seconds=step_s)

    previous_time = start
    previous_margin = _margin_at(
        target,
        observer,
        previous_time,
        min_elevation_deg=min_elevation_deg,
        horizon_mask=horizon_mask,
    )
    aos_candidate: datetime | None = None

    sample_time = start
    while sample_time < end:
        sample_time = min(sample_time + step, end)
        margin = _margin_at(
            target,
            observer,
            sample_time,
            min_elevation_deg=min_elevation_deg,
            horizon_mask=horizon_mask,
        )

        rising_crossing = margin >= 0.0 and previous_margin < 0.0
        falling_crossing = margin < 0.0 and previous_margin >= 0.0

        if aos_candidate is None and rising_crossing:
            aos_candidate = _refine_crossing(
                target,
                observer,
                previous_time,
                sample_time,
                min_elevation_deg=min_elevation_deg,
                horizon_mask=horizon_mask,
            )
        elif aos_candidate is not None and falling_crossing:
            los_time = _refine_crossing(
                target,
                observer,
                sample_time,
                previous_time,
                min_elevation_deg=min_elevation_deg,
                horizon_mask=horizon_mask,
            )
            passes.append(
                _build_pass(
                    target,
                    observer,
                    aos_candidate,
                    los_time,
                    track_step_s=track_step_s,
                    include_illumination=include_illumination,
                    twilight_sun_elevation_deg=twilight_sun_elevation_deg,
                )
            )
            aos_candidate = None

        previous_time = sample_time
        previous_margin = margin

    return passes
