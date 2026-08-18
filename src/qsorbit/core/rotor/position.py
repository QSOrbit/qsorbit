"""Rotor position model."""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Widest axis angle accepted by :class:`Position`, in degrees, either
#: direction. Three full turns — far past any real rotator's travel, but
#: comfortably outside anything a working axis could legitimately report.
#:
#: This is a **corruption filter, not a travel limit.** Its only job is to
#: reject values that can't have come from a functioning rotor, so a
#: garbled serial frame or a runaway calculation surfaces as an error
#: rather than a movement command. What a *particular* rotor may safely
#: be commanded to is a property of that hardware, and belongs to the
#: pointing layer — see :mod:`qsorbit.core.pointing`.
MAX_AXIS_DEGREES = 1080.0


@dataclass(frozen=True)
class Position:
    """A rotor axis position in degrees.

    A ``Position`` is a value object: immutable, validated on construction,
    and comparable by value. Two positions with the same angles are equal.

    This is a **mechanical axis reading**, measured from wherever the rotor
    homed its zero — not a compass bearing. Two consequences follow, both
    confirmed against real hardware:

    * **Negative values are normal.** A rotor homed against an end-stop
      commonly settles slightly past its zero and reports, say,
      ``AZ-1.5 EL2.0``. That is an honest reading, not an error.
    * **Values beyond 360° are meaningful** on a multi-turn azimuth axis,
      where 380° and 20° are different physical places reached by
      different amounts of travel.

    For "where something is in the sky" — a genuine compass bearing,
    strictly ``[0, 360)`` with elevation capped at 90° — use
    :class:`qsorbit.core.geometry.AzEl` instead. Converting between the
    two is the job of :mod:`qsorbit.core.pointing`.

    Validation here rejects only values that cannot have come from a
    working rotor: non-finite numbers, and magnitudes beyond
    :data:`MAX_AXIS_DEGREES`. It deliberately does **not** enforce travel
    limits. Those depend on the specific hardware — how far the azimuth
    axis can turn before the cable binds, whether elevation can pass
    vertical — and are enforced in the pointing layer, which knows what
    rotor it is talking to. Note that the SatNOGS controller firmware
    enforces no position limits whatsoever, so QSOrbit is the only
    backstop there is.

    Args:
        azimuth: Azimuth axis angle in degrees, relative to the rotor's
            homed zero.
        elevation: Elevation axis angle in degrees, relative to the
            rotor's homed zero.

    Raises:
        ValueError: If either angle is not finite, or its magnitude
            exceeds :data:`MAX_AXIS_DEGREES`.
    """

    azimuth: float
    elevation: float

    def __post_init__(self) -> None:
        for name, value in (("Azimuth", self.azimuth), ("Elevation", self.elevation)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}.")
            if abs(value) > MAX_AXIS_DEGREES:
                raise ValueError(
                    f"{name} must be within +/-{MAX_AXIS_DEGREES} degrees, got {value}. "
                    "This is a corruption check, not a travel limit — a value this "
                    "large means a garbled response or a calculation error."
                )
