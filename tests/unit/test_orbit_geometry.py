"""Unit tests for the coarse "ever visible from this latitude" geometry.

Hand-picked numbers throughout, deliberately -- this module is pure
math with no TLE or skyfield dependency, so there is no reason to route
these tests through a fabricated TLE the way ``test_picker.py``'s own
module docstring warns against.
"""

from __future__ import annotations

import math

import pytest

from qsorbit.core.orbit_geometry import (
    MEAN_EARTH_RADIUS_KM,
    footprint_radius_deg,
    is_ever_visible_from_latitude,
    max_ground_track_latitude_deg,
)


class TestMaxGroundTrackLatitudeDeg:
    def test_prograde_orbit_reaches_its_own_inclination(self):
        assert max_ground_track_latitude_deg(51.6) == 51.6

    def test_equatorial_orbit_reaches_the_equator_only(self):
        assert max_ground_track_latitude_deg(0.0) == 0.0

    def test_polar_orbit_reaches_the_poles(self):
        assert max_ground_track_latitude_deg(90.0) == 90.0

    def test_retrograde_orbit_reaches_180_minus_inclination(self):
        # A 98-degree sun-synchronous bird tops out at 82 degrees, not
        # 98 -- it does not pass over the poles.
        assert max_ground_track_latitude_deg(98.0) == pytest.approx(82.0)

    def test_fully_retrograde_equatorial_orbit_reaches_the_equator_only(self):
        assert max_ground_track_latitude_deg(180.0) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("bad", [-0.1, 180.1])
    def test_out_of_range_inclination_is_an_error(self, bad):
        with pytest.raises(ValueError, match="inclination_deg"):
            max_ground_track_latitude_deg(bad)


class TestFootprintRadiusDeg:
    def test_matches_the_closed_form_directly(self):
        # rho = arccos(Re / (Re + h)), computed independently here
        # rather than by calling the function under test with itself.
        altitude_km = 550.0
        expected = math.degrees(
            math.acos(MEAN_EARTH_RADIUS_KM / (MEAN_EARTH_RADIUS_KM + altitude_km))
        )
        assert footprint_radius_deg(altitude_km) == pytest.approx(expected)

    def test_a_higher_orbit_has_a_wider_footprint(self):
        assert footprint_radius_deg(2000.0) > footprint_radius_deg(500.0)

    def test_a_vanishingly_low_orbit_has_a_vanishingly_small_footprint(self):
        assert footprint_radius_deg(0.001) == pytest.approx(0.0, abs=0.05)

    @pytest.mark.parametrize("bad", [0.0, -100.0])
    def test_non_positive_altitude_is_an_error(self, bad):
        with pytest.raises(ValueError, match="altitude_km"):
            footprint_radius_deg(bad)


class TestIsEverVisibleFromLatitude:
    def test_a_near_equatorial_low_orbit_never_rises_from_40_degrees_north(self):
        # The roadmap's own worked example: IO-86 is near-equatorial
        # and never rises from 40 degrees N. 0.5 degrees inclination,
        # 550 km altitude is a realistic near-equatorial LEO bird.
        assert not is_ever_visible_from_latitude(0.5, 550.0, 40.0)

    def test_the_same_orbit_is_visible_from_near_the_equator(self):
        assert is_ever_visible_from_latitude(0.5, 550.0, 10.0)

    def test_an_iss_like_inclination_reaches_mid_latitudes(self):
        # 51.6 degrees, ~400 km -- the ISS's own real numbers.
        assert is_ever_visible_from_latitude(51.6, 400.0, 40.0)

    def test_an_iss_like_inclination_does_not_reach_the_poles(self):
        assert not is_ever_visible_from_latitude(51.6, 400.0, 85.0)

    def test_a_polar_orbit_is_visible_from_everywhere(self):
        assert is_ever_visible_from_latitude(98.0, 550.0, 89.9)
        assert is_ever_visible_from_latitude(98.0, 550.0, -89.9)
        assert is_ever_visible_from_latitude(98.0, 550.0, 0.0)

    def test_visibility_is_symmetric_across_the_equator(self):
        assert is_ever_visible_from_latitude(30.0, 600.0, 25.0) == (
            is_ever_visible_from_latitude(30.0, 600.0, -25.0)
        )

    def test_right_at_the_boundary_is_visible_not_excluded(self):
        # <=, not <: a station exactly at the edge of the footprint
        # does eventually see the satellite graze its horizon.
        inclination_deg, altitude_km = 20.0, 800.0
        reach = max_ground_track_latitude_deg(inclination_deg) + footprint_radius_deg(altitude_km)
        assert is_ever_visible_from_latitude(inclination_deg, altitude_km, reach)

    def test_just_past_the_boundary_is_not_visible(self):
        inclination_deg, altitude_km = 20.0, 800.0
        reach = max_ground_track_latitude_deg(inclination_deg) + footprint_radius_deg(altitude_km)
        assert not is_ever_visible_from_latitude(inclination_deg, altitude_km, reach + 0.01)

    @pytest.mark.parametrize("bad", [-90.1, 90.1])
    def test_out_of_range_station_latitude_is_an_error(self, bad):
        with pytest.raises(ValueError, match="station_latitude_deg"):
            is_ever_visible_from_latitude(45.0, 550.0, bad)
