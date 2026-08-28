"""Unit tests for the horizon mask -- HorizonPoint and HorizonMask."""

import pytest

from qsorbit.core.horizon import HorizonMask, HorizonPoint


class TestHorizonPoint:
    def test_accepts_valid_values(self):
        point = HorizonPoint(azimuth_deg=111.0, min_elevation_deg=18.0)

        assert point.azimuth_deg == 111.0
        assert point.min_elevation_deg == 18.0

    def test_azimuth_360_is_rejected(self):
        # AzEl's own convention: [0, 360), not [0, 360].
        with pytest.raises(ValueError, match="azimuth_deg"):
            HorizonPoint(azimuth_deg=360.0, min_elevation_deg=0.0)

    def test_negative_azimuth_is_rejected(self):
        with pytest.raises(ValueError, match="azimuth_deg"):
            HorizonPoint(azimuth_deg=-1.0, min_elevation_deg=0.0)

    def test_negative_min_elevation_is_rejected(self):
        with pytest.raises(ValueError, match="min_elevation_deg"):
            HorizonPoint(azimuth_deg=0.0, min_elevation_deg=-0.1)

    def test_min_elevation_above_90_is_rejected(self):
        with pytest.raises(ValueError, match="min_elevation_deg"):
            HorizonPoint(azimuth_deg=0.0, min_elevation_deg=90.1)

    def test_min_elevation_90_is_allowed(self):
        # Degenerate but not invalid -- straight up.
        point = HorizonPoint(azimuth_deg=0.0, min_elevation_deg=90.0)

        assert point.min_elevation_deg == 90.0


class TestHorizonMaskConstruction:
    def test_empty_is_valid(self):
        mask = HorizonMask()

        assert mask.points == ()

    def test_single_point_is_valid(self):
        mask = HorizonMask(points=(HorizonPoint(90.0, 10.0),))

        assert len(mask.points) == 1

    def test_sorted_points_are_valid(self):
        mask = HorizonMask(
            points=(HorizonPoint(10.0, 0.0), HorizonPoint(20.0, 5.0), HorizonPoint(30.0, 0.0))
        )

        assert len(mask.points) == 3

    def test_unsorted_points_are_rejected(self):
        with pytest.raises(ValueError, match="sorted"):
            HorizonMask(points=(HorizonPoint(30.0, 0.0), HorizonPoint(10.0, 0.0)))

    def test_duplicate_azimuth_is_rejected(self):
        with pytest.raises(ValueError, match="sorted"):
            HorizonMask(points=(HorizonPoint(10.0, 0.0), HorizonPoint(10.0, 5.0)))


class TestMinElevationAt:
    def test_empty_mask_is_zero_everywhere(self):
        mask = HorizonMask()

        assert mask.min_elevation_at(0.0) == 0.0
        assert mask.min_elevation_at(180.0) == 0.0
        assert mask.min_elevation_at(359.9) == 0.0

    def test_single_point_is_constant_everywhere(self):
        mask = HorizonMask(points=(HorizonPoint(90.0, 12.0),))

        assert mask.min_elevation_at(0.0) == 12.0
        assert mask.min_elevation_at(90.0) == 12.0
        assert mask.min_elevation_at(270.0) == 12.0

    def test_exact_point_values(self):
        mask = HorizonMask(
            points=(HorizonPoint(105.0, 0.0), HorizonPoint(111.0, 18.0), HorizonPoint(117.0, 0.0))
        )

        assert mask.min_elevation_at(105.0) == 0.0
        assert mask.min_elevation_at(111.0) == pytest.approx(18.0)
        assert mask.min_elevation_at(117.0) == pytest.approx(0.0, abs=1e-9)

    def test_linear_interpolation_between_points(self):
        mask = HorizonMask(points=(HorizonPoint(100.0, 0.0), HorizonPoint(110.0, 20.0)))

        assert mask.min_elevation_at(105.0) == pytest.approx(10.0)
        assert mask.min_elevation_at(102.5) == pytest.approx(5.0)

    def test_two_obstructions_flat_baseline_between_them(self):
        # This project's own two measured obstruction points, Session
        # 22: az ~111 blocked below ~18, az ~193-199 blocked below
        # ~20-23. Everywhere outside both bumps should read exactly 0.
        mask = HorizonMask(
            points=(
                HorizonPoint(105.0, 0.0),
                HorizonPoint(111.0, 18.0),
                HorizonPoint(117.0, 0.0),
                HorizonPoint(188.0, 0.0),
                HorizonPoint(193.0, 20.0),
                HorizonPoint(199.0, 23.0),
                HorizonPoint(204.0, 0.0),
            )
        )

        assert mask.min_elevation_at(0.0) == 0.0
        assert mask.min_elevation_at(50.0) == 0.0
        assert mask.min_elevation_at(150.0) == 0.0
        assert mask.min_elevation_at(300.0) == 0.0
        assert mask.min_elevation_at(359.0) == 0.0
        assert mask.min_elevation_at(111.0) == pytest.approx(18.0)
        assert mask.min_elevation_at(196.0) == pytest.approx(21.5)

    def test_wraps_past_the_last_point_back_to_the_first(self):
        # Points must be given in ascending order (10.0 before 350.0)
        # -- the wraparound this is testing is min_elevation_at's own
        # circular interpolation from the last point back to the
        # first, not an unsorted point list, which HorizonMask rejects
        # outright (see TestHorizonMaskConstruction).
        mask = HorizonMask(points=(HorizonPoint(10.0, 20.0), HorizonPoint(350.0, 0.0)))

        # The wrapped gap from 350 through 360/0 to 10 spans 20
        # degrees; 355 is a quarter of the way across it.
        assert mask.min_elevation_at(355.0) == pytest.approx(5.0)
        assert mask.min_elevation_at(0.0) == pytest.approx(10.0)
        assert mask.min_elevation_at(5.0) == pytest.approx(15.0)

    def test_azimuth_outside_0_360_wraps_the_same_as_inside(self):
        mask = HorizonMask(points=(HorizonPoint(90.0, 12.0),))

        assert mask.min_elevation_at(450.0) == mask.min_elevation_at(90.0)
        assert mask.min_elevation_at(-10.0) == mask.min_elevation_at(350.0)
