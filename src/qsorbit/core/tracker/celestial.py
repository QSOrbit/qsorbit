"""Celestial targets: pointing the rotor at things that are not satellites.

QSOrbit's pointing path never needed a satellite specifically -- it needed
something that could say where it is in the sky for an observer at a time,
which is what :class:`~qsorbit.core.tracker.target.Target` has described
since Phase 1. This module is the first thing other than
:class:`~qsorbit.core.tracker.satellite.Satellite` to satisfy it.

**No ephemeris file, deliberately.** The obvious way to point at the Sun or a
star is to load JPL's ``de421.bsp`` and ask skyfield, which is what this
station's own bench scripts do. This project does not, for the reason
:mod:`qsorbit.core.tracker.sun` already gave when it computed the Sun's
position from a closed-form series instead: the file is 16 MB against a 1.7 MB
package, it stops covering dates in 2050, and everything QSOrbit points at is
otherwise derived rather than tabulated. An analytic series is smaller, does
not expire, and is far more accurate than any rotator can use. What the
ephemeris is still good for is *checking* this module, which is exactly what
``tools/generate_celestial_oracle.py`` uses it for -- so the correctness is
bought without the dependency.

**What skyfield does for us, and why almost nothing here is arithmetic.**
The hard part of "where do I point" is not the body's position, it is the
observer's local frame: sidereal rotation, the geodetic vertical (which is
*not* the direction from Earth's centre through you -- they differ by up to
about 0.19 degrees at mid latitudes), precession of the equinox since J2000,
nutation, and -- for anything as close as the Moon -- topocentric parallax,
worth up to 0.95 degrees near the horizon. All of that already happens inside
skyfield's ``altaz()`` when it is handed the difference between two geocentric
vectors, which is precisely how
:meth:`Satellite.topocentric_state
<qsorbit.core.tracker.satellite.Satellite.topocentric_state>` has always
worked. Crucially it needs no ephemeris to do it: an ephemeris is only
required by ``observe()``, which resolves a body's position *for* you. So
supplying the position ourselves and reusing the satellite path gets every one
of those corrections for free, and this module's real job shrinks to producing
one geocentric vector per body.

**What is deliberately not corrected.** Annual aberration -- the roughly
20.5-arcsecond tilt from Earth's own orbital velocity -- is applied by
``observe()`` and therefore not by this path. That is a considered omission
rather than an oversight: it is about 1/500th of this station's measured
rotator free play, and it is dwarfed by atmospheric refraction (roughly 0.035
degrees at 25 degrees elevation, 0.09 at 10) which nothing here models either.
Refraction is the honest floor for any uncorrected pointing system, and it is
larger than every term this module could add. If a mount ever arrives that can
use the difference, refraction is the thing to model first, not aberration.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
from skyfield.constants import AU_KM
from skyfield.vectorlib import VectorFunction

from qsorbit.core.geometry import AzEl
from qsorbit.core.tracker._shared import require_timezone_aware, ts
from qsorbit.core.tracker.observer import ObserverLocation
from qsorbit.core.tracker.star_catalog import STARS, Star
from qsorbit.core.tracker.state import TopocentricState
from qsorbit.core.tracker.sun import sun_gcrs_km

#: How far away a catalogued star is placed, in kilometers.
#:
#: Stars are modeled as directions rather than places: the catalogue carries
#: no distances, and the two effects a real distance would buy -- annual
#: parallax and topocentric parallax -- are both far below anything a rotator
#: can act on (well under an arcsecond even for the nearest star). Putting
#: every star at one large fixed distance says "this is a direction" honestly,
#: rather than implying a precision the data does not have. At this value the
#: residual topocentric parallax is about 0.0003 arcseconds.
STAR_DISTANCE_KM = 1.0e12

#: Julian days per Julian year, for turning an instant into "years since
#: J2000" when applying proper motion.
DAYS_PER_YEAR = 365.25

#: The Julian date of J2000.0, the epoch the star catalogue is referred to.
J2000_JULIAN_DATE = 2451545.0

#: The interval used to measure range rate by finite difference, in seconds.
#: A :class:`~qsorbit.core.tracker.state.TopocentricState` carries a range
#: rate because Doppler correction needs one from a satellite; for a celestial
#: body it is computed honestly and is simply of no interest to anything in
#: this project today.
RANGE_RATE_INTERVAL_S = 1.0


class _GeocentricBody(VectorFunction):
    """A body whose geocentric position QSOrbit computes itself.

    Adapts a plain ``datetime -> (x, y, z) kilometers`` function into
    something skyfield will difference against an observer, which is what
    unlocks ``altaz()`` and every frame correction it performs. ``center =
    399`` declares the vector Earth-centered, the same claim
    :class:`~skyfield.sgp4lib.EarthSatellite` makes -- which is why no
    ephemeris is needed to resolve it.

    Velocity is reported as zero. Skyfield only uses it for rate products
    this module does not ask for; range rate is measured by finite
    difference in :func:`_topocentric_state` instead, so nothing reads a
    velocity that was never supplied.
    """

    center = 399

    def __init__(self, name: str, position_km) -> None:
        self._position_km = position_km
        self.target = f"qsorbit:{name}"
        self.target_name = name

    def _at(self, t):
        position_km = self._position_km(t.utc_datetime())
        return np.array(position_km, dtype=float) / AU_KM, np.zeros(3), None, None


def _topocentric_state(
    body: _GeocentricBody, observer: ObserverLocation, time: datetime
) -> TopocentricState:
    """Turn a geocentric body into what an observer sees, via skyfield.

    Deliberately the same three-step shape as
    :meth:`Satellite.topocentric_state
    <qsorbit.core.tracker.satellite.Satellite.topocentric_state>` -- difference
    against the observer, evaluate, read ``altaz()`` -- so that the frame
    handling is shared rather than reimplemented. See the module docstring for
    what that buys.
    """
    difference = body - observer.skyfield_position
    now = difference.at(ts.from_datetime(time))
    altitude, azimuth, distance = now.altaz()

    later = difference.at(ts.from_datetime(time + timedelta(seconds=RANGE_RATE_INTERVAL_S)))
    range_rate_km_s = (later.distance().km - distance.km) / RANGE_RATE_INTERVAL_S

    # Same representation edge Satellite.topocentric_state compensates for:
    # azimuth is geometrically undefined at the zenith, so floating-point
    # noise can land a hair outside [0, 360). This is not forgiving an
    # out-of-range value, which AzEl is right to reject.
    return TopocentricState(
        sky_position=AzEl(azimuth=azimuth.degrees % 360.0, elevation=altitude.degrees),
        range_km=distance.km,
        range_rate_km_s=range_rate_km_s,
    )


class SunTarget:
    """The Sun, as something the pointing path can track.

    Satisfies :class:`~qsorbit.core.tracker.target.Target` structurally --
    no subclassing, which is what the protocol exists for.

    The Sun is the one celestial target whose *purpose* is calibration
    rather than observation: point the boom at it, adjust until its shadow
    is smallest, and the difference between the computed sky position and
    the rotor's axis reading is the alignment offset, on both axes at once.
    No TLE, no timing precision, no signal strength, and verifiable by eye.

    **Never look at the Sun through anything.** Pointing an antenna at it is
    harmless; pointing an optical finder at it is not.
    """

    @property
    def name(self) -> str:
        """A human-readable name, for logs and display."""
        return "Sun"

    def topocentric_state(self, observer: ObserverLocation, time: datetime) -> TopocentricState:
        """Compute where the Sun appears from ``observer`` at ``time``.

        Args:
            observer: The ground observer's location.
            time: The instant to compute, as a timezone-aware datetime.

        Returns:
            The Sun's sky position, range, and range rate.

        Raises:
            ValueError: If ``time`` is naive (has no ``tzinfo``).
        """
        require_timezone_aware(time)
        return _topocentric_state(_GeocentricBody("Sun", sun_gcrs_km), observer, time)


class StarTarget:
    """A catalogued star, as something the pointing path can track.

    Satisfies :class:`~qsorbit.core.tracker.target.Target` structurally.

    A star is the best pointing reference this station can get short of a
    visible satellite pass: it is available every clear night, it needs no
    TLE and no schedule, and unlike the Sun it can be sighted directly. For
    *azimuth* calibration specifically, prefer a star at moderate elevation
    -- near the zenith an azimuth error shrinks by ``cos(elevation)``, so the
    same sighting error buys far less there.

    Args:
        star: The catalogue entry to track.
    """

    def __init__(self, star: Star) -> None:
        self._star = star

    @property
    def name(self) -> str:
        """A human-readable name, for logs and display."""
        return self._star.name

    @property
    def star(self) -> Star:
        """The catalogue entry this target was built from."""
        return self._star

    def topocentric_state(self, observer: ObserverLocation, time: datetime) -> TopocentricState:
        """Compute where this star appears from ``observer`` at ``time``.

        Args:
            observer: The ground observer's location.
            time: The instant to compute, as a timezone-aware datetime.

        Returns:
            The star's sky position, and a range reflecting the fixed
            :data:`STAR_DISTANCE_KM` stand-in rather than any real distance.

        Raises:
            ValueError: If ``time`` is naive (has no ``tzinfo``).
        """
        require_timezone_aware(time)
        return _topocentric_state(
            _GeocentricBody(self._star.name, self._position_km), observer, time
        )

    def _position_km(self, time: datetime) -> tuple[float, float, float]:
        """The star's geocentric direction at ``time``, as a position vector.

        Proper motion is applied here rather than baked into the catalogue,
        because it is the one thing about a star that depends on when you
        ask. Everything else -- precession, nutation, Earth rotation -- is
        skyfield's job downstream.
        """
        years = (ts.from_datetime(time).tt - J2000_JULIAN_DATE) / DAYS_PER_YEAR
        ra_hours = self._star.ra_hours + self._star.pm_ra_hours_per_year * years
        dec_degrees = self._star.dec_degrees + self._star.pm_dec_degrees_per_year * years

        right_ascension = math.radians(ra_hours * 15.0)
        declination = math.radians(dec_degrees)
        return (
            STAR_DISTANCE_KM * math.cos(declination) * math.cos(right_ascension),
            STAR_DISTANCE_KM * math.cos(declination) * math.sin(right_ascension),
            STAR_DISTANCE_KM * math.sin(declination),
        )


def celestial_target(name: str):
    """Look up a celestial target by the name a user types.

    Args:
        name: ``"sun"``, or a star's catalogue key such as ``"polaris"`` or
            ``"kaus-australis"``. Case-insensitive, and spaces are accepted
            in place of hyphens.

    Returns:
        A :class:`~qsorbit.core.tracker.target.Target`.

    Raises:
        ValueError: If no target goes by that name. The message lists what
            does, because a user who mistyped a star name has no other way
            to find out what the sixty options are.
    """
    key = name.strip().lower().replace(" ", "-")
    if key == "sun":
        return SunTarget()
    if key in STARS:
        return StarTarget(STARS[key])
    raise ValueError(
        f"No celestial target named {name!r}. Available: sun, " + ", ".join(sorted(STARS))
    )


def celestial_target_names() -> list[str]:
    """Every name :func:`celestial_target` accepts, sorted.

    Exists so a command-line parser can offer the list as choices rather
    than discovering a typo after the rotor has already connected.
    """
    return sorted([*STARS, "sun"])
