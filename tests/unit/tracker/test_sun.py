"""Unit tests for the closed-form Sun position and illumination geometry."""

import math
from datetime import UTC, datetime

import pytest

from qsorbit.core.tracker.observer import ObserverLocation
from qsorbit.core.tracker.sun import (
    AU_KM,
    EARTH_RADIUS_KM,
    is_illuminated,
    sun_elevation_deg,
    sun_gcrs_km,
)


def _unit_and_distance(vector):
    distance = math.sqrt(sum(c * c for c in vector))
    return tuple(c / distance for c in vector), distance


class TestSunGcrsKm:
    def test_naive_datetime_is_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            sun_gcrs_km(datetime(2026, 8, 28, 12, 0, 0))  # noqa: DTZ001

    def test_distance_is_close_to_one_au(self):
        # Earth's orbit is nearly circular - the Sun's apparent distance
        # never strays far from 1 AU. 3% is generous against the real
        # ~1.67% eccentricity-driven range, and catches a gross unit or
        # formula error without being a photometric-precision assertion.
        _, distance_km = _unit_and_distance(
            sun_gcrs_km(datetime(2026, 8, 28, 12, 0, 0, tzinfo=UTC))
        )

        assert distance_km == pytest.approx(AU_KM, rel=0.03)

    def test_moves_over_a_quarter_year(self):
        # The Sun's apparent ecliptic longitude advances roughly
        # 360/365.25 degrees per day - over ~91 days it should have
        # moved close to a quarter of the way around, i.e. the unit
        # vector should now point in a very different direction.
        start = sun_gcrs_km(datetime(2026, 3, 20, 12, 0, 0, tzinfo=UTC))
        later = sun_gcrs_km(datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC))
        start_unit, _ = _unit_and_distance(start)
        later_unit, _ = _unit_and_distance(later)

        cos_angle = sum(a * b for a, b in zip(start_unit, later_unit, strict=True))
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))

        assert angle_deg == pytest.approx(90.0, abs=5.0)


class TestIsIlluminated:
    """Built from the geometry directly, rather than a real ephemeris,
    so each case's expected answer follows from is_illuminated's own
    contract rather than from an independent source of truth agreeing
    with it -- see the module-level shadow-test cases in the project's
    own bench verification (Session 22) for that independent check."""

    def _sun_unit(self, time):
        unit, _ = _unit_and_distance(sun_gcrs_km(time))
        return unit

    def test_satellite_on_the_sunward_side_is_illuminated(self):
        time = datetime(2026, 8, 28, 17, 0, 0, tzinfo=UTC)
        sun_unit = self._sun_unit(time)
        radius_km = EARTH_RADIUS_KM + 700.0
        position = tuple(c * radius_km for c in sun_unit)

        assert is_illuminated(position, time) is True

    def test_satellite_directly_behind_earth_is_in_shadow(self):
        time = datetime(2026, 8, 28, 17, 0, 0, tzinfo=UTC)
        sun_unit = self._sun_unit(time)
        radius_km = EARTH_RADIUS_KM + 700.0
        position = tuple(-c * radius_km for c in sun_unit)

        assert is_illuminated(position, time) is False

    def test_satellite_off_to_the_side_is_illuminated(self):
        time = datetime(2026, 8, 28, 17, 0, 0, tzinfo=UTC)
        sun_unit = self._sun_unit(time)
        # Any vector perpendicular to sun_unit.
        perp = (-sun_unit[1], sun_unit[0], 0.0)
        perp_unit, _ = _unit_and_distance(perp)
        radius_km = EARTH_RADIUS_KM + 700.0
        position = tuple(c * radius_km for c in perp_unit)

        assert is_illuminated(position, time) is True

    def test_shadow_has_a_finite_width(self):
        # Behind Earth, but offset sideways by more than Earth's radius
        # -- out of the cylinder, so illuminated even though it is on
        # the anti-solar side.
        time = datetime(2026, 8, 28, 17, 0, 0, tzinfo=UTC)
        sun_unit = self._sun_unit(time)
        perp = (-sun_unit[1], sun_unit[0], 0.0)
        perp_unit, _ = _unit_and_distance(perp)
        behind = tuple(-c * (EARTH_RADIUS_KM + 100.0) for c in sun_unit)
        far_side = tuple(
            a + b * (EARTH_RADIUS_KM + 1000.0) for a, b in zip(behind, perp_unit, strict=True)
        )
        near_side = tuple(
            a + b * (EARTH_RADIUS_KM - 1000.0) for a, b in zip(behind, perp_unit, strict=True)
        )

        assert is_illuminated(far_side, time) is True
        assert is_illuminated(near_side, time) is False


class TestSunElevationDeg:
    def test_naive_datetime_is_rejected(self):
        observer = ObserverLocation(latitude=40.0, longitude=-83.0)

        with pytest.raises(ValueError, match="timezone-aware"):
            sun_elevation_deg(observer, datetime(2026, 8, 28, 17, 0, 0))  # noqa: DTZ001

    def test_local_noon_is_higher_than_local_midnight(self):
        observer = ObserverLocation(latitude=40.0, longitude=-83.0, altitude_m=250.0)
        # -83 degrees longitude is roughly UTC-5.5h solar time; use
        # UTC instants a half-day apart so one is near local noon and
        # the other near local midnight, independent of DST.
        near_noon = datetime(2026, 8, 28, 17, 30, 0, tzinfo=UTC)
        near_midnight = datetime(2026, 8, 29, 5, 30, 0, tzinfo=UTC)

        noon_elevation = sun_elevation_deg(observer, near_noon)
        midnight_elevation = sun_elevation_deg(observer, near_midnight)

        assert noon_elevation > 30.0
        assert midnight_elevation < -20.0
        assert noon_elevation > midnight_elevation

    def test_result_is_within_the_physically_possible_range(self):
        observer = ObserverLocation(latitude=40.0, longitude=-83.0, altitude_m=250.0)

        elevation = sun_elevation_deg(observer, datetime(2026, 8, 28, 12, 0, 0, tzinfo=UTC))

        assert -90.0 <= elevation <= 90.0
