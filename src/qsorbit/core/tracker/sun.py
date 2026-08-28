"""Closed-form solar position, and the illumination geometry it feeds.

QSOrbit deliberately carries no ephemeris file. Every other body it
tracks is a satellite from a TLE -- SGP4 alone, no JPL data -- and the
station's own bench scripts (outside this repo) that needed the Sun's
position for a shadow test or a ``--moon``/``--star`` target pulled in
the 17 MB de421 ephemeris to get it. That is a real dependency and
distribution-size decision this project has not made, so this module
does not make it either: it computes the Sun's direction from a
closed-form low-precision solar-position formula (the Astronomical
Almanac's, also given in Vallado's *Fundamentals of Astrodynamics and
Applications*), good to about 0.01 degrees for the years this project
will run in. That is a hundred times better than the illumination test
below needs, and costs nothing to ship or fetch.

Two things this module is for, and one it deliberately is not:

* :func:`sun_gcrs_km` -- the Sun's position, so a satellite's own
  illumination (is it in Earth's shadow right now?) can be tested
  against it. See :func:`is_illuminated`.
* :func:`sun_elevation_deg` -- how high the Sun is as seen from a given
  observer, so "is the *sky* dark enough for a satellite to be
  visible" can be tested too. Naked-eye visibility needs both: the
  satellite lit, and the observer's sky dark. See
  :mod:`qsorbit.core.tracker.pass_prediction`.
* It is **not** a general-purpose Sun target for the rotor to point
  at. A ``SunTarget`` satisfying
  :class:`~qsorbit.core.tracker.target.Target` (for sun-shadow
  alignment calibration, the way the out-of-repo ``rotor-track.py``
  already does with a real ephemeris) is future work -- it would want
  full topocentric az/el with proper parallax, which this module's
  simplifications deliberately skip. See :func:`sun_elevation_deg`'s
  own docstring for exactly what is skipped and why it is fine here
  but would not be fine for pointing hardware at the sky.
"""

from __future__ import annotations

import math
from datetime import datetime

from qsorbit.core.tracker._shared import require_timezone_aware, ts
from qsorbit.core.tracker.observer import ObserverLocation

#: Mean equatorial Earth radius, in kilometers (the WGS84 semi-major
#: axis). The shadow test below models Earth's shadow as a plain
#: cylinder of this radius rather than the true oblate spheroid or the
#: Sun's actual angular size -- a simplification standard for this
#: kind of visibility check, and well inside the precision this
#: module's Sun position already has.
EARTH_RADIUS_KM = 6378.137

#: One astronomical unit, in kilometers. Turns the solar-position
#: formula's output (given in AU) into the kilometers everything else
#: in this project already works in.
AU_KM = 149_597_870.7


def sun_gcrs_km(time: datetime) -> tuple[float, float, float]:
    """The Sun's geocentric position at ``time``, in GCRS-frame kilometers.

    Uses the Astronomical Almanac's low-precision solar coordinates
    (good to about 0.01 degrees from roughly 1950 to 2050), evaluated
    at the Julian centuries of TT since J2000.0. Matches the frame
    :meth:`Satellite.state_at
    <qsorbit.core.tracker.satellite.Satellite.state_at>` returns
    satellite positions in, so the two are directly comparable -- see
    :func:`is_illuminated`.

    Args:
        time: The instant to compute, as a timezone-aware datetime.

    Returns:
        ``(x, y, z)`` in kilometers, Earth-centered.

    Raises:
        ValueError: If ``time`` is naive (has no ``tzinfo``).
    """
    require_timezone_aware(time)
    t = ts.from_datetime(time)
    centuries = (t.tt - 2451545.0) / 36525.0

    mean_longitude_deg = (280.460 + 36000.771 * centuries) % 360.0
    mean_anomaly_deg = (357.5291092 + 35999.05034 * centuries) % 360.0
    mean_anomaly_rad = math.radians(mean_anomaly_deg)

    ecliptic_longitude_deg = (
        mean_longitude_deg
        + 1.914666471 * math.sin(mean_anomaly_rad)
        + 0.019994643 * math.sin(2.0 * mean_anomaly_rad)
    ) % 360.0
    ecliptic_longitude_rad = math.radians(ecliptic_longitude_deg)

    obliquity_rad = math.radians(23.439291 - 0.0130042 * centuries)

    distance_au = (
        1.000140612
        - 0.016708617 * math.cos(mean_anomaly_rad)
        - 0.000139589 * math.cos(2.0 * mean_anomaly_rad)
    )
    distance_km = distance_au * AU_KM

    x = distance_km * math.cos(ecliptic_longitude_rad)
    y = distance_km * math.cos(obliquity_rad) * math.sin(ecliptic_longitude_rad)
    z = distance_km * math.sin(obliquity_rad) * math.sin(ecliptic_longitude_rad)
    return (x, y, z)


def is_illuminated(satellite_gcrs_km: tuple[float, float, float], time: datetime) -> bool:
    """Whether a satellite is in sunlight, or inside Earth's shadow.

    A cylindrical-shadow test: Earth's shadow is modeled as an
    infinite cylinder of radius :data:`EARTH_RADIUS_KM`, aligned with
    the Earth-Sun line -- no umbra/penumbra taper, which the Sun's
    ~1 AU distance makes negligible at satellite altitudes. Validated
    in the project's own bench notes (Session 22) against two real
    rocket-body passes fading in and out of visibility exactly where
    this test predicts.

    Args:
        satellite_gcrs_km: The satellite's ``(x, y, z)`` position in
            GCRS-frame kilometers -- the same frame and units as
            :attr:`~qsorbit.core.tracker.state.EciState.position_km`
            and :func:`sun_gcrs_km`.
        time: The instant to compute the Sun's position for.

    Returns:
        ``True`` if the satellite is sunlit, ``False`` if it is inside
        Earth's shadow.

    Raises:
        ValueError: If ``time`` is naive (has no ``tzinfo``).
    """
    sun_km = sun_gcrs_km(time)
    sun_distance = math.sqrt(sum(c * c for c in sun_km))
    sun_unit = tuple(c / sun_distance for c in sun_km)

    along_sun_axis = sum(s * u for s, u in zip(satellite_gcrs_km, sun_unit, strict=True))
    satellite_distance = math.sqrt(sum(c * c for c in satellite_gcrs_km))
    off_axis_squared = satellite_distance * satellite_distance - along_sun_axis * along_sun_axis
    # Floating-point noise can push this fractionally negative when a
    # satellite sits almost exactly on the Sun-Earth axis; clamp
    # rather than let sqrt raise on a value that is mathematically
    # zero.
    off_axis_km = math.sqrt(max(off_axis_squared, 0.0))

    in_earths_shadow = along_sun_axis < 0.0 and off_axis_km < EARTH_RADIUS_KM
    return not in_earths_shadow


def sun_elevation_deg(observer: ObserverLocation, time: datetime) -> float:
    """How high the Sun is above ``observer``'s horizon, in degrees.

    Used to test whether the *observer's* sky is dark enough for a
    satellite to be visible -- illumination geometry needs both this
    and :func:`is_illuminated`, because a sunlit satellite over a
    daylit observer is invisible for a completely different reason
    than one sitting in Earth's shadow.

    **Deliberately not full topocentric az/el.** This treats the
    observer's zenith direction as the normalized geocentric position
    vector (the direction from Earth's center through the observer),
    rather than the true WGS84 geodetic zenith, and it does not
    correct for the Sun's parallax (up to about 8.8 arcseconds at
    1 AU). Both are exact for a sphere and slightly wrong for an
    oblate Earth or a nearby body -- by at most a few arcminutes here,
    irrelevant for a twilight threshold several degrees wide (see
    :mod:`qsorbit.core.tracker.pass_prediction`). **This would not be
    an acceptable shortcut for pointing hardware at the sky** -- see
    the module docstring.

    Args:
        observer: The ground observer's location.
        time: The instant to compute for.

    Returns:
        The Sun's elevation in degrees. Negative means below the
        horizon.

    Raises:
        ValueError: If ``time`` is naive (has no ``tzinfo``).
    """
    require_timezone_aware(time)
    t = ts.from_datetime(time)
    sun_km = sun_gcrs_km(time)
    observer_km = observer.skyfield_position.at(t).position.km
    observer_distance = math.sqrt(sum(c * c for c in observer_km))
    zenith_unit = tuple(c / observer_distance for c in observer_km)

    to_sun = tuple(s - o for s, o in zip(sun_km, observer_km, strict=True))
    to_sun_distance = math.sqrt(sum(c * c for c in to_sun))
    to_sun_unit = tuple(c / to_sun_distance for c in to_sun)

    # dot(zenith, sun direction) is cos(zenith angle), which is the
    # same number as sin(elevation) since elevation = 90 - zenith
    # angle -- so this feeds straight into asin() rather than needing
    # a separate 90-minus step.
    sin_elevation = sum(z * s for z, s in zip(zenith_unit, to_sun_unit, strict=True))
    # Clamp for the same floating-point reason as is_illuminated: the
    # dot product of two unit vectors is mathematically in [-1, 1], but
    # can land a hair outside it, and asin() raises on that.
    sin_elevation = max(-1.0, min(1.0, sin_elevation))
    return math.degrees(math.asin(sin_elevation))
