"""Satellite state model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class EciState:
    """A satellite's position and velocity in an Earth-centered inertial frame.

    ``EciState`` is a value object: immutable and comparable by value.

    The frame is geocentric and inertial — specifically GCRS, the same
    frame :meth:`Satellite.state_at
    <qsorbit.core.tracker.satellite.Satellite.state_at>` computes it in
    — not Earth-fixed, so these coordinates don't rotate with the Earth.
    Converting to an observer's local az/el is the job of the next
    chunk.

    Args:
        time: The UTC instant this state describes.
        position_km: ``(x, y, z)`` position in kilometers.
        velocity_km_s: ``(vx, vy, vz)`` velocity in kilometers per second.
    """

    time: datetime
    position_km: tuple[float, float, float]
    velocity_km_s: tuple[float, float, float]
