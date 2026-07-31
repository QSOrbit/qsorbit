"""Shared geometric vocabulary for the whole application.

This module holds types that describe *directions in the sky* and
nothing else. It deliberately depends on no other QSOrbit module, so
any part of the app — tracker, rotor, calibration, a future camera
module — can speak this vocabulary without dragging in anything it
doesn't need.

The distinction that matters here: :class:`AzEl` is astronomical truth,
"where the object actually is as seen from the ground." It is *not* a
rotor command. Turning an :class:`AzEl` into something a rotor can be
told to do is the job of :mod:`qsorbit.core.pointing`, because that
conversion depends on the hardware (alignment error, travel limits,
whether the rotor can flip past zenith) and on the shape of the whole
pass. Keeping the two as separate types means that conversion can't be
skipped by accident.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AzEl:
    """A direction in the sky as seen from a ground observer, in degrees.

    ``AzEl`` is a value object: immutable, validated on construction, and
    comparable by value. Two directions with the same angles are equal.

    This type carries no hardware meaning. It says where something is,
    not where a rotor should point — see
    :func:`qsorbit.core.pointing.sky_to_rotor` for that conversion and
    :class:`qsorbit.core.rotor.Position` for the command-side type.

    Validation is strict: out-of-range values raise ``ValueError`` rather
    than being wrapped or clamped, because a caller producing an azimuth
    of 725° almost certainly has a math bug upstream and silently
    normalizing would hide it.

    Args:
        azimuth: Compass bearing in degrees, ``0.0 <= azimuth < 360.0``.
            North is 0, east is 90.
        elevation: Angle above the horizon in degrees,
            ``-90.0 <= elevation <= 90.0``. Straight up is 90. Negative
            values are meaningful and expected — a satellite that hasn't
            risen yet has a real, computable position below the horizon.
            The upper bound of 90 is definitional rather than a policy
            choice: you cannot be more than straight up.

    Raises:
        ValueError: If either angle is outside its valid range.
    """

    azimuth: float
    elevation: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.azimuth < 360.0:
            raise ValueError(
                f"Azimuth must be in [0.0, 360.0), got {self.azimuth}. "
                "Normalize (e.g. azimuth % 360.0) before constructing an AzEl."
            )
        if not -90.0 <= self.elevation <= 90.0:
            raise ValueError(f"Elevation must be in [-90.0, 90.0], got {self.elevation}.")
