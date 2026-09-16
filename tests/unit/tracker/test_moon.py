"""Unit tests for the closed-form lunar position.

End-to-end accuracy is checked in ``test_celestial.py`` against the DE421
reference fixture, where the Moon sits alongside the Sun and the stars under
one tolerance. What this file covers is the *module*: that its coefficient
tables are intact, and that the positions it produces obey the lunar
invariants an error in the series would violate.

Those invariants matter because the table is 120 rows of numbers taken from
elsewhere. A corrupted extraction, a truncated paste or a stray digit would
have to break at least one of: the distance staying between perigee and
apogee, the latitude staying inside the orbit's inclination, or the longitude
advancing at the rate a month is defined by.
"""

import hashlib
import math
from datetime import UTC, datetime, timedelta

import pytest

from qsorbit.core.tracker.moon import (
    MEAN_DISTANCE_KM,
    PERIODIC_TERMS_LATITUDE,
    PERIODIC_TERMS_LONGITUDE_DISTANCE,
    TABLE_DIGEST,
    moon_ecliptic_of_date,
    moon_gcrs_km,
)

#: Closest and furthest the Moon gets, in kilometers, with a little margin.
#: Perigee runs about 356,500 and apogee about 406,700.
PERIGEE_KM = 355_000.0
APOGEE_KM = 408_000.0

#: The Moon's orbit is inclined about 5.145 degrees to the ecliptic, and
#: perturbations swing that by a couple of tenths either way.
MAX_ECLIPTIC_LATITUDE_DEG = 5.4

#: Mean motion in ecliptic longitude, degrees per day: 360 / 27.32166, the
#: sidereal month.
MEAN_LONGITUDE_RATE_DEG_PER_DAY = 13.176


class TestCoefficientTables:
    """The tables are transcribed data, so they get a transcription guard.

    They describe a fixed published theory -- Meeus's Tables 47.A and 47.B,
    truncated from ELP-2000/82 -- and should never change. A digest turns
    "someone edited a coefficient" from something you find out later into
    something that fails immediately.
    """

    def test_both_tables_are_complete(self):
        assert len(PERIODIC_TERMS_LONGITUDE_DISTANCE) == 60
        assert len(PERIODIC_TERMS_LATITUDE) == 60

    def test_every_row_has_the_right_shape(self):
        # Longitude/distance rows carry two coefficients, latitude rows one.
        assert all(len(row) == 6 for row in PERIODIC_TERMS_LONGITUDE_DISTANCE)
        assert all(len(row) == 5 for row in PERIODIC_TERMS_LATITUDE)

    def test_the_argument_multipliers_are_small_integers(self):
        # Meeus's arguments are combinations like 2D - M' or 2D - 2F; a
        # multiplier outside this range means a column has shifted.
        for row in (*PERIODIC_TERMS_LONGITUDE_DISTANCE, *PERIODIC_TERMS_LATITUDE):
            for multiplier in row[:4]:
                assert isinstance(multiplier, int)
                assert -6 <= multiplier <= 6

    def test_the_tables_match_their_recorded_digest(self):
        # If this fails, either a coefficient was edited or the tables were
        # regenerated. Both are things to do deliberately: update the digest
        # in the same change, and say in the commit why the theory moved.
        rows = (
            [list(row) for row in PERIODIC_TERMS_LONGITUDE_DISTANCE],
            [list(row) for row in PERIODIC_TERMS_LATITUDE],
        )
        digest = hashlib.sha256(repr(rows).encode()).hexdigest()[:16]

        assert digest == TABLE_DIGEST

    def test_the_leading_terms_are_the_named_lunar_inequalities(self):
        # The four largest longitude terms are famous enough to check by
        # name, and they are the ones a truncation would lose first:
        # equation of centre, evection, variation, and the second harmonic
        # of the Moon's anomaly. Their arguments pin the column order too.
        leading = PERIODIC_TERMS_LONGITUDE_DISTANCE[:4]

        assert leading[0][:4] == (0, 0, 1, 0)  # M'      - equation of centre
        assert leading[1][:4] == (2, 0, -1, 0)  # 2D - M' - evection
        assert leading[2][:4] == (2, 0, 0, 0)  # 2D      - variation
        assert leading[3][:4] == (0, 0, 2, 0)  # 2M'
        # 6.29 degrees, in units of 1e-6 degrees.
        assert leading[0][4] == pytest.approx(6_288_774, rel=1e-6)


class TestMoonEclipticOfDate:
    def test_naive_datetime_is_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            moon_ecliptic_of_date(datetime(2026, 9, 16, 3, 0, 0))  # noqa: DTZ001

    def test_distance_stays_between_perigee_and_apogee(self):
        # Sampled across a full anomalistic month so both extremes are hit.
        distances = [
            moon_ecliptic_of_date(datetime(2026, 9, 1, tzinfo=UTC) + timedelta(hours=6 * n))[2]
            for n in range(112)
        ]

        assert min(distances) > PERIGEE_KM
        assert max(distances) < APOGEE_KM
        # And it actually varies -- a constant would also pass the bounds
        # above. The floor is well under the 50,000 km that separates the
        # extreme perigee from the extreme apogee, because perigee and
        # apogee distances themselves vary: one lunation rarely spans both
        # extremes, and this window measures about 36,000 km.
        assert max(distances) - min(distances) > 30_000.0

    def test_the_distance_series_brackets_its_own_mean(self):
        distances = [
            moon_ecliptic_of_date(datetime(2026, 9, 1, tzinfo=UTC) + timedelta(hours=6 * n))[2]
            for n in range(112)
        ]

        assert min(distances) < MEAN_DISTANCE_KM < max(distances)

    def test_latitude_stays_within_the_orbits_inclination(self):
        latitudes = [
            moon_ecliptic_of_date(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=8 * n))[1]
            for n in range(400)
        ]

        assert max(abs(latitude) for latitude in latitudes) < MAX_ECLIPTIC_LATITUDE_DEG
        # The Moon crosses the ecliptic twice a month, so a sample this long
        # must contain both signs -- a sign error in the latitude series
        # would otherwise be invisible to a bound alone.
        assert max(latitudes) > 4.0
        assert min(latitudes) < -4.0

    def test_longitude_advances_at_the_sidereal_rate(self):
        # This is the check a corrupted mean-longitude polynomial cannot
        # survive: a sidereal month is *defined* by this rate.
        start = datetime(2026, 9, 16, 3, 0, 0, tzinfo=UTC)
        days = 27.32166
        first = moon_ecliptic_of_date(start)[0]
        later = moon_ecliptic_of_date(start + timedelta(days=days))[0]

        # A full circuit, so the longitude should return close to where it
        # began rather than land anywhere in particular.
        difference = (later - first + 180.0) % 360.0 - 180.0
        assert abs(difference) < 3.0

    def test_daily_motion_is_about_thirteen_degrees(self):
        start = datetime(2026, 9, 16, 3, 0, 0, tzinfo=UTC)
        first = moon_ecliptic_of_date(start)[0]
        later = moon_ecliptic_of_date(start + timedelta(days=1))[0]
        advance = (later - first) % 360.0

        # Real daily motion swings roughly 12 to 15 degrees with the
        # eccentricity, so this is deliberately a band rather than a point.
        assert MEAN_LONGITUDE_RATE_DEG_PER_DAY - 2.0 < advance
        assert advance < MEAN_LONGITUDE_RATE_DEG_PER_DAY + 2.0


class TestMoonGcrsKm:
    def test_naive_datetime_is_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            moon_gcrs_km(datetime(2026, 9, 16, 3, 0, 0))  # noqa: DTZ001

    def test_the_vector_length_matches_the_computed_distance(self):
        # The ecliptic-to-equatorial conversion and the frame rotation are
        # both rotations, so neither may change the magnitude. A scaling
        # bug in either would show up here and nowhere else.
        when = datetime(2026, 9, 16, 3, 0, 0, tzinfo=UTC)
        _, _, distance_km = moon_ecliptic_of_date(when)
        vector = moon_gcrs_km(when)
        length = math.sqrt(sum(component * component for component in vector))

        assert length == pytest.approx(distance_km, rel=1e-12)

    def test_the_position_is_not_confined_to_the_equator(self):
        # The Moon's declination swings roughly +/-28 degrees. A z-component
        # stuck near zero would mean the obliquity rotation was dropped.
        when = datetime(2026, 1, 1, tzinfo=UTC)
        z_values = [
            moon_gcrs_km(when + timedelta(hours=8 * n))[2] / MEAN_DISTANCE_KM for n in range(120)
        ]

        assert max(z_values) > 0.35
        assert min(z_values) < -0.35
