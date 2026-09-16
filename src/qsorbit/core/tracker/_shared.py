"""Internal helpers shared across the tracker module.

Not part of the public API — see :mod:`qsorbit.core.tracker` for that.
"""

from __future__ import annotations

from datetime import datetime

from skyfield.api import load
from skyfield.framelib import true_equator_and_equinox_of_date

#: A single skyfield timescale shared by every satellite and observer
#: computation in the process. Built with ``builtin=True`` so it uses
#: skyfield's bundled leap-second/delta-T tables instead of downloading
#: fresh ones — no network access required, and identical results on
#: every machine and in CI, at the cost of losing sub-second delta-T
#: precision for dates far from when skyfield's bundled tables were
#: generated. An acceptable trade for this project.
ts = load.timescale(builtin=True)


def require_timezone_aware(time: datetime) -> None:
    """Raise ``ValueError`` if ``time`` has no ``tzinfo``.

    Every tracker computation that asks skyfield for a position at a
    particular instant needs to know which UTC instant is meant —
    naive datetimes are ambiguous, and guessing would be exactly the
    kind of bug that's invisible until it isn't.
    """
    if time.tzinfo is None:
        raise ValueError(
            "time must be timezone-aware (e.g. datetime.now(UTC)). Naive "
            "datetimes are ambiguous about which UTC instant is meant."
        )


def of_date_to_gcrs(vector_km: tuple[float, float, float], t) -> tuple[float, float, float]:
    """Rotate an equator-of-date vector into GCRS.

    Both closed-form series this project carries -- the Astronomical
    Almanac's Sun (:mod:`qsorbit.core.tracker.sun`) and Meeus's Moon
    (:mod:`qsorbit.core.tracker.moon`) -- produce coordinates referred to
    the mean equinox *of date*, which moves. Everything else in QSOrbit
    works in GCRS, whose axes are fixed at J2000, and the two drift apart
    by precession at about 50.3 arcseconds a year. Every such series has
    to be rotated before it is handed to anything else, which is why this
    lives here rather than in either module.

    Skyfield's frame object supplies the rotation that takes GCRS *to*
    the equator and equinox of date, so the transpose is what brings a
    vector the other way. Done by hand rather than with numpy, to keep
    the arithmetic readable.

    The frame used includes nutation while these series are referred to
    the *mean* equinox, so the rotation is wrong by the nutation of the
    day -- at most about 17 arcseconds, comfortably inside the roughly
    36-arcsecond accuracy of the series themselves, and skyfield exposes
    this frame publicly where it does not expose a mean-only one.

    Args:
        vector_km: ``(x, y, z)`` referred to the mean equator of date.
        t: The skyfield ``Time`` the vector was computed for.

    Returns:
        The same vector with GCRS axes.
    """
    rotation = true_equator_and_equinox_of_date.rotation_at(t)
    return (
        float(sum(rotation[axis][0] * vector_km[axis] for axis in range(3))),
        float(sum(rotation[axis][1] * vector_km[axis] for axis in range(3))),
        float(sum(rotation[axis][2] * vector_km[axis] for axis in range(3))),
    )
