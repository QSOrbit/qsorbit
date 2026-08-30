"""Ground tracks: a satellite's sub-satellite point sampled across a time window.

The map's own use for :func:`ground_track`: a series of
:class:`~qsorbit.core.tracker.state.Subpoint`, close enough together to
draw as a polyline, showing the arc the satellite has swept and will
sweep across Earth's surface around a given instant. Unlike
:func:`~qsorbit.core.tracker.pass_prediction.predict_passes`, this asks
nothing about a station's horizon or elevation threshold -- a ground
track is a fact about the orbit and the planet's rotation underneath
it, the same "true regardless of any one station" register
:mod:`qsorbit.core.orbit_geometry` occupies for the picker's
latitude-visibility filter.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from qsorbit.core.tracker._shared import require_timezone_aware
from qsorbit.core.tracker.satellite import Satellite
from qsorbit.core.tracker.state import Subpoint

#: How far either side of the requested instant a ground track extends
#: by default, in minutes. Matches the roadmap's own "Map view" line
#: and the shell mockup's own ground-track note -- long enough to show
#: a meaningful arc of a LEO orbit (roughly a 90-100 minute period)
#: without the map redrawing a nearly-complete loop on every refresh.
DEFAULT_SPAN_MINUTES = 90.0

#: The interval between sampled subpoints, in minutes. 37 points across
#: the default +/-90 minute span draws a visibly smooth polyline at any
#: on-screen map size this widget will ever draw, without propagating
#: far more instants than a static map actually needs.
DEFAULT_STEP_MINUTES = 5.0


def ground_track(
    satellite: Satellite,
    center: datetime,
    *,
    span_minutes: float = DEFAULT_SPAN_MINUTES,
    step_minutes: float = DEFAULT_STEP_MINUTES,
) -> tuple[Subpoint, ...]:
    """This satellite's sub-satellite points from ``span_minutes`` before ``center`` to after.

    Args:
        satellite: The satellite to propagate.
        center: The instant the track is centered on -- typically now.
            Timezone-aware (see :meth:`Satellite.state_at
            <qsorbit.core.tracker.satellite.Satellite.state_at>`).
        span_minutes: How far either side of ``center`` the track
            extends. Defaults to :data:`DEFAULT_SPAN_MINUTES`.
        step_minutes: The interval between sampled points. Defaults to
            :data:`DEFAULT_STEP_MINUTES`.

    Returns:
        Subpoints in chronological order, from ``center - span_minutes``
        to ``center + span_minutes`` inclusive of both ends regardless
        of how evenly ``step_minutes`` divides ``span_minutes``.

    Raises:
        ValueError: If ``center`` is naive, or ``span_minutes`` or
            ``step_minutes`` is not positive.
        PropagationError: If SGP4 cannot compute a valid position at any
            sampled instant -- typically because the window reaches too
            far from this TLE's epoch to still describe a physically
            sensible orbit (see :attr:`Satellite.epoch
            <qsorbit.core.tracker.satellite.Satellite.epoch>`).
    """
    require_timezone_aware(center)
    if span_minutes <= 0.0:
        raise ValueError(f"span_minutes must be positive, got {span_minutes}.")
    if step_minutes <= 0.0:
        raise ValueError(f"step_minutes must be positive, got {step_minutes}.")

    offsets = []
    offset = -span_minutes
    # Accumulated rather than a floating-point range(), so the track
    # lands exactly on +span_minutes at the far end regardless of how
    # step_minutes happens to divide it -- a caller reading "+/-90 min"
    # expects the track to actually reach both ends.
    while offset < span_minutes:
        offsets.append(offset)
        offset += step_minutes
    offsets.append(span_minutes)

    return tuple(satellite.subpoint_at(center + timedelta(minutes=offset)) for offset in offsets)
