"""Rotor position model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    """An antenna pointing direction in degrees.

    A ``Position`` is a value object: immutable, validated on construction,
    and comparable by value. Two positions with the same angles are equal.

    Validation is strict — out-of-range values raise ``ValueError`` rather
    than being wrapped or clamped. A caller producing an azimuth of 725°
    almost certainly has a math bug upstream, and silently normalizing
    would hide it.

    Args:
        azimuth: Compass bearing in degrees, ``0.0 <= azimuth < 360.0``.
            North is 0, east is 90.
        elevation: Angle above the horizon in degrees,
            ``-90.0 <= elevation <= 90.0``. Straight up is 90; slightly
            negative values are allowed because rotors can physically
            point below the horizon.

    Raises:
        ValueError: If either angle is outside its valid range.
    """

    azimuth: float
    elevation: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.azimuth < 360.0:
            raise ValueError(
                f"Azimuth must be in [0.0, 360.0), got {self.azimuth}. "
                "Normalize (e.g. azimuth % 360.0) before constructing a Position."
            )
        if not -90.0 <= self.elevation <= 90.0:
            raise ValueError(f"Elevation must be in [-90.0, 90.0], got {self.elevation}.")
