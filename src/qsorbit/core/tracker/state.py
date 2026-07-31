"""Satellite state model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from qsorbit.core.geometry import AzEl


@dataclass(frozen=True)
class EciState:
    """A satellite's position and velocity in an Earth-centered inertial frame.

    ``EciState`` is a value object: immutable and comparable by value.

    The frame is geocentric and inertial — specifically GCRS, the same
    frame :meth:`Satellite.state_at
    <qsorbit.core.tracker.satellite.Satellite.state_at>` computes it in
    — not Earth-fixed, so these coordinates don't rotate with the Earth.
    Converting to an observer's local az/el is the job of
    :meth:`Satellite.topocentric_state
    <qsorbit.core.tracker.satellite.Satellite.topocentric_state>`.

    Args:
        time: The UTC instant this state describes.
        position_km: ``(x, y, z)`` position in kilometers.
        velocity_km_s: ``(vx, vy, vz)`` velocity in kilometers per second.
    """

    time: datetime
    position_km: tuple[float, float, float]
    velocity_km_s: tuple[float, float, float]


@dataclass(frozen=True)
class TopocentricState:
    """A satellite's position relative to a ground observer.

    ``TopocentricState`` is a value object: immutable and comparable by
    value.

    Args:
        sky_position: Where the satellite appears in the sky — an
            :class:`~qsorbit.core.geometry.AzEl`, which is astronomical
            truth rather than a rotor command. It deliberately is *not*
            a :class:`qsorbit.core.rotor.Position`: pointing real
            hardware at this direction may require an alignment offset,
            a flip past zenith, or an azimuth unwrap, none of which
            belong in a statement about where the satellite is. See
            :mod:`qsorbit.core.pointing` for that conversion.
        range_km: Straight-line distance from observer to satellite, in
            kilometers.
        range_rate_km_s: Rate of change of range, in kilometers per
            second. Positive means the satellite is receding (range
            increasing); negative means it's approaching.
    """

    sky_position: AzEl
    range_km: float
    range_rate_km_s: float
