"""Closed-form lunar position, for pointing at the Moon.

The Moon is the hardest body QSOrbit computes and the only one needing a
real series. Its orbit is perturbed hard enough by the Sun that the
corrections have individual names and took two thousand years to find one
at a time -- the equation of centre at 6.29 degrees (Hipparchus), evection
at 1.27 (Ptolemy), variation at 0.66 (Tycho Brahe), the annual equation at
0.19 (Tycho again), and then fifty-odd more down to a thousandth of a
degree. Where the Sun needs two periodic terms, this needs a hundred and
twenty.

That is exactly the kind of table a person should not type. These
coefficients were extracted **mechanically** from PyMeeus rather than
transcribed, because a hand-copied table is its own authority and a single
wrong digit in a small term would sit below any tolerance worth setting.
:data:`TABLE_DIGEST` pins them: the tables describe a fixed published
theory and should never change, so an edit has to be deliberate enough to
update a checksum.

**Provenance and attribution.** The coefficients are Tables 47.A and 47.B
of Jean Meeus, *Astronomical Algorithms*, 2nd edition, chapter 47, which
are themselves a truncation of the ELP-2000/82 lunar theory. They were
taken from **PyMeeus 0.5.12**, Copyright (C) 2021 Dagoberto Salazar,
distributed under the LGPL v3 -- which permits conveying the work under
the GPL v3 that QSOrbit itself uses. See
https://github.com/architest/pymeeus.

**No ephemeris, for the reasons** :mod:`qsorbit.core.tracker.celestial`
gives. Measured against JPL's DE421 across 2000-2050 this lands within
**37 arcseconds** in direction and 42 km in distance (of about 384,400),
which is the same class of accuracy as the Sun's 39 and the star
catalogue's 40 -- and all three are far finer than atmospheric refraction,
which nothing here models.

**Frames.** Like the Sun's series this produces coordinates referred to
the mean equinox *of date*, and is rotated into GCRS by
:func:`~qsorbit.core.tracker._shared.of_date_to_gcrs` before returning.
That rotation is not optional; see :mod:`qsorbit.core.tracker.sun` for
what its absence cost.
"""

from __future__ import annotations

import math
from datetime import datetime

from qsorbit.core.tracker._shared import of_date_to_gcrs, require_timezone_aware, ts

#: Mean distance from Earth to Moon, in kilometers -- the constant term the
#: distance series corrects.
MEAN_DISTANCE_KM = 385000.56

#: Periodic terms for ecliptic longitude and distance (Meeus Table 47.A).
#:
#: Each row is ``(D, M, M', F, sigma_l, sigma_r)``: four integer multipliers
#: of the fundamental arguments, then the longitude coefficient in units of
#: 1e-6 degrees and the distance coefficient in units of 1e-3 kilometers.
#: Rows whose ``M`` multiplier is +/-1 are scaled by the eccentricity factor
#: E, and +/-2 by E squared, because those terms depend on the Sun's own
#: anomaly and Earth's orbit is slowly becoming less eccentric.
PERIODIC_TERMS_LONGITUDE_DISTANCE = (
    (0, 0, 1, 0, 6288774, -20905355),
    (2, 0, -1, 0, 1274027, -3699111),
    (2, 0, 0, 0, 658314, -2955968),
    (0, 0, 2, 0, 213618, -569925),
    (0, 1, 0, 0, -185116, 48888),
    (0, 0, 0, 2, -114332, -3149),
    (2, 0, -2, 0, 58793, 246158),
    (2, -1, -1, 0, 57066, -152138),
    (2, 0, 1, 0, 53322, -170733),
    (2, -1, 0, 0, 45758, -204586),
    (0, 1, -1, 0, -40923, -129620),
    (1, 0, 0, 0, -34720, 108743),
    (0, 1, 1, 0, -30383, 104755),
    (2, 0, 0, -2, 15327, 10321),
    (0, 0, 1, 2, -12528, 0),
    (0, 0, 1, -2, 10980, 79661),
    (4, 0, -1, 0, 10675, -34782),
    (0, 0, 3, 0, 10034, -23210),
    (4, 0, -2, 0, 8548, -21636),
    (2, 1, -1, 0, -7888, 24208),
    (2, 1, 0, 0, -6766, 30824),
    (1, 0, -1, 0, -5163, -8379),
    (1, 1, 0, 0, 4987, -16675),
    (2, -1, 1, 0, 4036, -12831),
    (2, 0, 2, 0, 3994, -10445),
    (4, 0, 0, 0, 3861, -11650),
    (2, 0, -3, 0, 3665, 14403),
    (0, 1, -2, 0, -2689, -7003),
    (2, 0, -1, 2, -2602, 0),
    (2, -1, -2, 0, 2390, 10056),
    (1, 0, 1, 0, -2348, 6322),
    (2, -2, 0, 0, 2236, -9884),
    (0, 1, 2, 0, -2120, 5751),
    (0, 2, 0, 0, -2069, 0),
    (2, -2, -1, 0, 2048, -4950),
    (2, 0, 1, -2, -1773, 4130),
    (2, 0, 0, 2, -1595, 0),
    (4, -1, -1, 0, 1215, -3958),
    (0, 0, 2, 2, -1110, 0),
    (3, 0, -1, 0, -892, 3258),
    (2, 1, 1, 0, -810, 2616),
    (4, -1, -2, 0, 759, -1897),
    (0, 2, -1, 0, -713, -2117),
    (2, 2, -1, 0, -700, 2354),
    (2, 1, -2, 0, 691, 0),
    (2, -1, 0, -2, 596, 0),
    (4, 0, 1, 0, 549, -1423),
    (0, 0, 4, 0, 537, -1117),
    (4, -1, 0, 0, 520, -1571),
    (1, 0, -2, 0, -487, -1739),
    (2, 1, 0, -2, -399, 0),
    (0, 0, 2, -2, -381, -4421),
    (1, 1, 1, 0, 351, 0),
    (3, 0, -2, 0, -340, 0),
    (4, 0, -3, 0, 330, 0),
    (2, -1, 2, 0, 327, 0),
    (0, 2, 1, 0, -323, 1165),
    (1, 1, -1, 0, 299, 0),
    (2, 0, 3, 0, 294, 0),
    (2, 0, -1, -2, 0, 8752),
)

#: Periodic terms for ecliptic latitude (Meeus Table 47.B).
#:
#: ``(D, M, M', F, sigma_b)``, the coefficient in units of 1e-6 degrees.
#: The same eccentricity scaling applies.
PERIODIC_TERMS_LATITUDE = (
    (0, 0, 0, 1, 5128122),
    (0, 0, 1, 1, 280602),
    (0, 0, 1, -1, 277693),
    (2, 0, 0, -1, 173237),
    (2, 0, -1, 1, 55413),
    (2, 0, -1, -1, 46271),
    (2, 0, 0, 1, 32573),
    (0, 0, 2, 1, 17198),
    (2, 0, 1, -1, 9266),
    (0, 0, 2, -1, 8822),
    (2, -1, 0, -1, 8216),
    (2, 0, -2, -1, 4324),
    (2, 0, 1, 1, 4200),
    (2, 1, 0, -1, -3359),
    (2, -1, -1, 1, 2463),
    (2, -1, 0, 1, 2211),
    (2, -1, -1, -1, 2065),
    (0, 1, -1, -1, -1870),
    (4, 0, -1, -1, 1828),
    (0, 1, 0, 1, -1794),
    (0, 0, 0, 3, -1749),
    (0, 1, -1, 1, -1565),
    (1, 0, 0, 1, -1491),
    (0, 1, 1, 1, -1475),
    (0, 1, 1, -1, -1410),
    (0, 1, 0, -1, -1344),
    (1, 0, 0, -1, -1335),
    (0, 0, 3, 1, 1107),
    (4, 0, 0, -1, 1021),
    (4, 0, -1, 1, 833),
    (0, 0, 1, -3, 777),
    (4, 0, -2, 1, 671),
    (2, 0, 0, -3, 607),
    (2, 0, 2, -1, 596),
    (2, -1, 1, -1, 491),
    (2, 0, -2, 1, -451),
    (0, 0, 3, -1, 439),
    (2, 0, 2, 1, 422),
    (2, 0, -3, -1, 421),
    (2, 1, -1, 1, -366),
    (2, 1, 0, 1, -351),
    (4, 0, 0, 1, 331),
    (2, -1, 1, 1, 315),
    (2, -2, 0, -1, 302),
    (0, 0, 1, 3, -283),
    (2, 1, 1, -1, -229),
    (1, 1, 0, -1, 223),
    (1, 1, 0, 1, 223),
    (0, 1, -2, -1, -220),
    (2, 1, -1, -1, -220),
    (1, 0, 1, 1, -185),
    (2, -1, -2, -1, 181),
    (0, 1, 2, 1, -177),
    (4, 0, -2, -1, 176),
    (4, -1, -1, -1, 166),
    (1, 0, 1, -1, -164),
    (4, 0, 1, -1, 132),
    (1, 0, -1, -1, -119),
    (4, -1, 0, -1, 115),
    (2, -2, 0, 1, 107),
)

#: SHA-256 prefix over both tables, so a stray edit to a coefficient cannot
#: pass unnoticed. These describe a fixed published theory: if this ever
#: needs updating, that is a decision someone should have to make on
#: purpose rather than discover afterwards.
TABLE_DIGEST = "af8ae85d6761f9ae"


def _fundamental_arguments(centuries: float):
    """The five angles every periodic term is built from, plus helpers.

    Returns ``(L', D, M, M', F, A1, A2, A3, E)`` with the angles in
    radians and reduced to a sensible range. ``L'`` is the Moon's mean
    longitude, ``D`` its mean elongation from the Sun, ``M`` the Sun's
    mean anomaly, ``M'`` the Moon's own, and ``F`` its argument of
    latitude. ``A1`` to ``A3`` cover perturbations from Venus and Jupiter
    and the flattening of the Earth; ``E`` is the eccentricity factor.
    """
    t = centuries
    mean_longitude = (
        218.3164477
        + (481267.88123421 + (-0.0015786 + (1.0 / 538841.0 - t / 65194000.0) * t) * t) * t
    )
    elongation = (
        297.8501921
        + (445267.1114034 + (-0.0018819 + (1.0 / 545868.0 - t / 113065000.0) * t) * t) * t
    )
    sun_anomaly = 357.5291092 + (35999.0502909 + (-0.0001536 + t / 24490000.0) * t) * t
    moon_anomaly = (
        134.9633964 + (477198.8675055 + (0.0087414 + (1.0 / 69699.9 + t / 14712000.0) * t) * t) * t
    )
    latitude_argument = (
        93.2720950
        + (483202.0175233 + (-0.0036539 + (-1.0 / 3526000.0 + t / 863310000.0) * t) * t) * t
    )
    venus_term = 119.75 + 131.849 * t
    jupiter_term = 53.09 + 479264.290 * t
    flattening_term = 313.45 + 481266.484 * t
    eccentricity = 1.0 + (-0.002516 - 0.0000074 * t) * t

    return (
        math.radians(mean_longitude % 360.0),
        math.radians(elongation % 360.0),
        math.radians(sun_anomaly % 360.0),
        math.radians(moon_anomaly % 360.0),
        math.radians(latitude_argument % 360.0),
        math.radians(venus_term % 360.0),
        math.radians(jupiter_term % 360.0),
        math.radians(flattening_term % 360.0),
        eccentricity,
    )


def moon_ecliptic_of_date(time: datetime) -> tuple[float, float, float]:
    """The Moon's geocentric ecliptic position, referred to the equinox of date.

    Args:
        time: The instant to compute, as a timezone-aware datetime.

    Returns:
        ``(longitude_deg, latitude_deg, distance_km)``.

    Raises:
        ValueError: If ``time`` is naive (has no ``tzinfo``).
    """
    require_timezone_aware(time)
    centuries = (ts.from_datetime(time).tt - 2451545.0) / 36525.0
    (
        mean_longitude,
        elongation,
        sun_anomaly,
        moon_anomaly,
        latitude_argument,
        venus_term,
        jupiter_term,
        flattening_term,
        eccentricity,
    ) = _fundamental_arguments(centuries)

    arguments = (elongation, sun_anomaly, moon_anomaly, latitude_argument)
    eccentricity_squared = eccentricity * eccentricity

    def scaled(coefficient: float, sun_multiplier: int) -> float:
        if abs(sun_multiplier) == 1:
            return coefficient * eccentricity
        if abs(sun_multiplier) == 2:
            return coefficient * eccentricity_squared
        return coefficient

    sigma_longitude = 0.0
    sigma_distance = 0.0
    for row in PERIODIC_TERMS_LONGITUDE_DISTANCE:
        angle = sum(m * a for m, a in zip(row[:4], arguments, strict=True) if m)
        sigma_longitude += scaled(row[4], row[1]) * math.sin(angle)
        sigma_distance += scaled(row[5], row[1]) * math.cos(angle)

    sigma_latitude = 0.0
    for row in PERIODIC_TERMS_LATITUDE:
        angle = sum(m * a for m, a in zip(row[:4], arguments, strict=True) if m)
        sigma_latitude += scaled(row[4], row[1]) * math.sin(angle)

    # Additive terms, which sit outside the tables because they come from
    # perturbations the tabulated arguments do not describe.
    sigma_longitude += (
        3958.0 * math.sin(venus_term)
        + 1962.0 * math.sin(mean_longitude - latitude_argument)
        + 318.0 * math.sin(jupiter_term)
    )
    sigma_latitude += (
        -2235.0 * math.sin(mean_longitude)
        + 382.0 * math.sin(flattening_term)
        + 175.0 * math.sin(venus_term - latitude_argument)
        + 175.0 * math.sin(venus_term + latitude_argument)
        + 127.0 * math.sin(mean_longitude - moon_anomaly)
        - 115.0 * math.sin(mean_longitude + moon_anomaly)
    )

    # float() rather than letting these through as numpy scalars: skyfield's
    # Time.tt is a numpy float, so it propagates silently through every term
    # above, and a public return value should be a plain Python number.
    return (
        float((math.degrees(mean_longitude) + sigma_longitude / 1.0e6) % 360.0),
        float(sigma_latitude / 1.0e6),
        float(MEAN_DISTANCE_KM + sigma_distance / 1.0e3),
    )


def moon_gcrs_km(time: datetime) -> tuple[float, float, float]:
    """The Moon's geocentric position at ``time``, in GCRS-frame kilometers.

    The same frame and units as :func:`~qsorbit.core.tracker.sun.sun_gcrs_km`
    and :attr:`~qsorbit.core.tracker.state.EciState.position_km`, so the
    three are directly comparable.

    Args:
        time: The instant to compute, as a timezone-aware datetime.

    Returns:
        ``(x, y, z)`` in kilometers, Earth-centered, GCRS axes.

    Raises:
        ValueError: If ``time`` is naive (has no ``tzinfo``).
    """
    longitude_deg, latitude_deg, distance_km = moon_ecliptic_of_date(time)
    t = ts.from_datetime(time)
    centuries = (t.tt - 2451545.0) / 36525.0

    obliquity = math.radians(23.439291 - 0.0130042 * centuries)
    longitude = math.radians(longitude_deg)
    latitude = math.radians(latitude_deg)

    # Ecliptic to equatorial, still referred to the equinox of date.
    of_date_km = (
        distance_km * math.cos(latitude) * math.cos(longitude),
        distance_km
        * (
            math.cos(latitude) * math.sin(longitude) * math.cos(obliquity)
            - math.sin(latitude) * math.sin(obliquity)
        ),
        distance_km
        * (
            math.cos(latitude) * math.sin(longitude) * math.sin(obliquity)
            + math.sin(latitude) * math.cos(obliquity)
        ),
    )
    return of_date_to_gcrs(of_date_km, t)
