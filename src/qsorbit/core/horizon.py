"""The horizon mask -- what this station's own site actually blocks.

A ground station's antenna doesn't have a clear view of the whole sky:
trees, buildings, terrain block some directions more than others. A
horizon mask describes that as a piecewise azimuth -> minimum-usable-
elevation profile, so pass prediction can filter out passes that never
clear the treeline instead of promising a satellite the antenna cannot
actually see.

This lives in its own module, not in :mod:`qsorbit.core.station` or
:mod:`qsorbit.core.tracker.pass_prediction`, for the same reason
:mod:`qsorbit.core.doppler` was pulled out of the tracker package in
Chunk G: station config depends on the tracker package (never the
other way round), but pass prediction needs to *apply* a horizon mask
that station config *stores* -- so the type has to live somewhere
neither of those two modules is, to avoid one importing the other
backwards. See :mod:`qsorbit.core.geometry` for the same argument made
about ``AzEl``.

Parked since Session 8 of the project for want of a consumer -- pass
prediction, landing alongside this in the same chunk, is that consumer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HorizonPoint:
    """One measured or estimated point on the horizon mask.

    ``HorizonPoint`` is a value object: immutable and comparable by
    value.

    Args:
        azimuth_deg: Compass bearing, ``0.0 <= azimuth_deg < 360.0``.
        min_elevation_deg: The lowest elevation usable at this azimuth,
            ``0.0 <= min_elevation_deg <= 90.0``. An open, unobstructed
            direction is written down as ``0.0`` rather than omitted --
            the same honest-identity-value reasoning
            :class:`~qsorbit.core.station.AlignmentSettings` uses for
            its own 0.0 default. See :class:`HorizonMask` for why a
            real obstruction needs points bracketing it back down to
            0.0 on both sides.

    Raises:
        ValueError: If either value is out of range.
    """

    azimuth_deg: float
    min_elevation_deg: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.azimuth_deg < 360.0:
            raise ValueError(f"azimuth_deg must be in [0.0, 360.0), got {self.azimuth_deg}.")
        if not 0.0 <= self.min_elevation_deg <= 90.0:
            raise ValueError(
                f"min_elevation_deg must be in [0.0, 90.0], got {self.min_elevation_deg}."
            )


@dataclass(frozen=True)
class HorizonMask:
    """A piecewise-linear azimuth -> minimum-elevation profile.

    ``HorizonMask`` is a value object: immutable and comparable by
    value. Between two points it linearly interpolates the minimum
    usable elevation; past the last point it wraps back around to the
    first, since azimuth is circular. An empty mask (no points at all)
    means no obstruction anywhere -- the honest default for a station
    that hasn't measured its horizon yet.

    Because interpolation is linear between whatever points exist, **an
    obstruction needs to be bracketed by 0.0 points on both sides**, or
    the mask draws a false slope across the empty sky between it and
    the next real point. Two real obstructions from this station
    (Session 22), entered as seven points::

        HorizonMask(points=(
            HorizonPoint(105.0, 0.0),
            HorizonPoint(111.0, 18.0),
            HorizonPoint(117.0, 0.0),
            HorizonPoint(188.0, 0.0),
            HorizonPoint(193.0, 20.0),
            HorizonPoint(199.0, 23.0),
            HorizonPoint(204.0, 0.0),
        ))

    Everywhere from 204 degrees around through north back to 105
    degrees stays flat at 0.0, because both ends of that long wrapped
    gap are themselves 0.0 points -- no separate "elsewhere" case is
    needed in the code, only in how the points are entered.

    Args:
        points: Points sorted strictly ascending by ``azimuth_deg``,
            with no two points sharing an azimuth. Defaults to empty
            (no mask).

    Raises:
        ValueError: If ``points`` isn't strictly sorted by
            ``azimuth_deg``, or two points share an azimuth. Rejecting
            an unsorted list rather than silently sorting it keeps a
            config file readable in the compass order a human wrote it
            in -- the same reasoning ``rotor.capabilities`` uses for
            rejecting a malformed value instead of guessing at it.
    """

    points: tuple[HorizonPoint, ...] = ()

    def __post_init__(self) -> None:
        for previous, current in zip(self.points, self.points[1:], strict=False):
            if current.azimuth_deg <= previous.azimuth_deg:
                raise ValueError(
                    "HorizonMask points must be strictly sorted by azimuth_deg with "
                    f"no duplicates; {previous.azimuth_deg} is followed by "
                    f"{current.azimuth_deg}."
                )

    def min_elevation_at(self, azimuth_deg: float) -> float:
        """The minimum usable elevation at ``azimuth_deg``, in degrees.

        Args:
            azimuth_deg: A compass bearing. Wrapped modulo 360
                internally, so a value outside ``[0.0, 360.0)`` (small
                floating-point overshoot from an upstream computation,
                say) is handled rather than rejected -- the same
                tolerance :meth:`Satellite.topocentric_state
                <qsorbit.core.tracker.satellite.Satellite.topocentric_state>`
                already gives its own azimuth output.

        Returns:
            ``0.0`` if this mask has no points. The single point's
            value, everywhere, if it has exactly one. Otherwise the
            piecewise-linear interpolation described in the class
            docstring, wrapping past the last point back to the first.
        """
        if not self.points:
            return 0.0
        if len(self.points) == 1:
            return self.points[0].min_elevation_deg

        count = len(self.points)
        for i in range(count):
            lower = self.points[i]
            upper = self.points[(i + 1) % count]
            span_deg = (upper.azimuth_deg - lower.azimuth_deg) % 360.0
            if span_deg == 0.0:
                span_deg = 360.0
            offset_deg = (azimuth_deg - lower.azimuth_deg) % 360.0
            if offset_deg <= span_deg:
                fraction = offset_deg / span_deg
                return lower.min_elevation_deg + fraction * (
                    upper.min_elevation_deg - lower.min_elevation_deg
                )
        raise AssertionError(  # pragma: no cover - every azimuth falls in some span
            f"azimuth_deg {azimuth_deg} matched no span in a {count}-point HorizonMask; "
            "this indicates a bug in min_elevation_at, not a bad input."
        )
