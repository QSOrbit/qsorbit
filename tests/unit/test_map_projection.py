"""Unit tests for the map's two projections.

Hand-derived reference points throughout -- same reasoning
``test_orbit_geometry.py``'s own module docstring gives: this module is
pure math with no TLE or skyfield dependency, so nothing here needs a
fabricated satellite to exercise.
"""

from __future__ import annotations

import pytest

from qsorbit.core.map_projection import Projection, equirectangular, orthographic, project_polyline


class TestEquirectangular:
    def test_plots_longitude_as_x_and_latitude_as_y(self):
        assert equirectangular(40.0, -83.0) == (-83.0, 40.0)

    def test_the_origin_maps_to_the_origin(self):
        assert equirectangular(0.0, 0.0) == (0.0, 0.0)

    def test_every_point_projects_somewhere(self):
        # Unlike orthographic, there is no far side of the world to
        # this projection -- even a point on the exact opposite side
        # of the planet from some reference still lands somewhere.
        assert equirectangular(-89.0, 179.9) == (179.9, -89.0)


class TestOrthographic:
    def test_the_center_point_projects_to_the_origin(self):
        assert orthographic(
            0.0, 0.0, center_latitude_deg=0.0, center_longitude_deg=0.0
        ) == pytest.approx((0.0, 0.0), abs=1e-9)

    def test_a_point_90_degrees_east_on_the_equator_sits_on_the_disks_right_edge(self):
        x, y = orthographic(0.0, 90.0, center_latitude_deg=0.0, center_longitude_deg=0.0)

        assert x == pytest.approx(1.0)
        assert y == pytest.approx(0.0, abs=1e-9)

    def test_a_point_90_degrees_west_on_the_equator_sits_on_the_disks_left_edge(self):
        x, y = orthographic(0.0, -90.0, center_latitude_deg=0.0, center_longitude_deg=0.0)

        assert x == pytest.approx(-1.0)
        assert y == pytest.approx(0.0, abs=1e-9)

    def test_the_north_pole_viewed_from_the_equator_sits_at_the_disks_top_edge(self):
        x, y = orthographic(90.0, 0.0, center_latitude_deg=0.0, center_longitude_deg=0.0)

        assert x == pytest.approx(0.0, abs=1e-9)
        assert y == pytest.approx(1.0)

    def test_the_antipode_is_not_visible(self):
        assert orthographic(0.0, 180.0, center_latitude_deg=0.0, center_longitude_deg=0.0) is None

    def test_a_point_just_past_the_visible_hemisphere_is_not_visible(self):
        # 91 degrees of angular distance from the center is just past
        # the limb (the visible hemisphere's edge sits at exactly 90).
        assert orthographic(0.0, 91.0, center_latitude_deg=0.0, center_longitude_deg=0.0) is None

    def test_a_point_just_inside_the_visible_hemisphere_is_visible(self):
        assert (
            orthographic(0.0, 89.0, center_latitude_deg=0.0, center_longitude_deg=0.0) is not None
        )

    def test_a_station_sees_its_own_location_at_the_disks_center(self):
        # The projection's whole point for this map: centered on the
        # station, the station's own position is always the origin,
        # regardless of where on Earth the station actually is.
        assert orthographic(
            40.0, -83.0, center_latitude_deg=40.0, center_longitude_deg=-83.0
        ) == pytest.approx((0.0, 0.0), abs=1e-9)

    def test_a_nearby_point_is_a_small_offset_from_center(self):
        x, y = orthographic(41.0, -83.0, center_latitude_deg=40.0, center_longitude_deg=-83.0)

        # One degree of latitude north of the center, due north on the
        # equator's own meridian -- x stays at 0, y is sin(1 degree).
        assert x == pytest.approx(0.0, abs=1e-9)
        assert y == pytest.approx(0.017452406437283512)


class TestProjectPolylineFlat:
    def test_a_track_with_no_antimeridian_crossing_is_one_segment(self):
        track = [(0.0, -10.0), (0.0, 0.0), (0.0, 10.0)]

        segments = project_polyline(track, Projection.FLAT)

        assert segments == (((-10.0, 0.0), (0.0, 0.0), (10.0, 0.0)),)

    def test_a_track_crossing_the_antimeridian_splits_into_two_segments(self):
        track = [(0.0, 170.0), (0.0, 179.0), (0.0, -179.0), (0.0, -170.0)]

        segments = project_polyline(track, Projection.FLAT)

        assert segments == (
            ((170.0, 0.0), (179.0, 0.0)),
            ((-179.0, 0.0), (-170.0, 0.0)),
        )

    def test_a_two_point_track_with_no_crossing_stays_one_segment(self):
        segments = project_polyline([(0.0, 0.0), (1.0, 1.0)], Projection.FLAT)

        assert len(segments) == 1
        assert len(segments[0]) == 2

    def test_an_empty_track_produces_no_segments(self):
        assert project_polyline([], Projection.FLAT) == ()


class TestProjectPolylineGlobe:
    def test_a_fully_visible_track_is_one_segment(self):
        track = [(0.0, -10.0), (0.0, 0.0), (0.0, 10.0)]

        segments = project_polyline(
            track, Projection.GLOBE, center_latitude_deg=0.0, center_longitude_deg=0.0
        )

        assert len(segments) == 1
        assert len(segments[0]) == 3

    def test_a_track_crossing_to_the_far_side_and_back_breaks_around_the_gap(self):
        # 0, 20, ..., 180 crosses from visible (<=90 from center) to
        # invisible (>90) partway through, then never comes back --
        # one visible run at the start, nothing after it.
        track = [(0.0, float(lon)) for lon in range(0, 181, 20)]

        segments = project_polyline(
            track, Projection.GLOBE, center_latitude_deg=0.0, center_longitude_deg=0.0
        )

        assert len(segments) == 1
        assert len(segments[0]) == 5  # lon 0, 20, 40, 60, 80

    def test_a_track_entirely_on_the_far_side_produces_no_segments(self):
        track = [(0.0, 170.0), (0.0, 175.0), (0.0, 180.0)]

        segments = project_polyline(
            track, Projection.GLOBE, center_latitude_deg=0.0, center_longitude_deg=0.0
        )

        assert segments == ()

    def test_a_single_isolated_visible_point_between_breaks_is_dropped(self):
        # Exactly at the limb on both sides of one visible sample --
        # a single point has nothing to draw a line to, so it is
        # dropped rather than kept as a lone, unconnected point.
        track = [(0.0, 179.0), (0.0, 0.0), (0.0, -179.0)]

        segments = project_polyline(
            track, Projection.GLOBE, center_latitude_deg=0.0, center_longitude_deg=180.0
        )

        assert segments == ()
