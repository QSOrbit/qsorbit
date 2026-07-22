"""Unit tests for the Position dataclass."""

import dataclasses

import pytest

from qsorbit.core.rotor import Position


class TestValidConstruction:
    def test_typical_position(self):
        pos = Position(azimuth=180.0, elevation=45.0)
        assert pos.azimuth == 180.0
        assert pos.elevation == 45.0

    def test_azimuth_lower_bound_inclusive(self):
        assert Position(azimuth=0.0, elevation=0.0).azimuth == 0.0

    def test_azimuth_just_below_upper_bound(self):
        assert Position(azimuth=359.9, elevation=0.0).azimuth == 359.9

    def test_elevation_bounds_inclusive(self):
        assert Position(azimuth=0.0, elevation=90.0).elevation == 90.0
        assert Position(azimuth=0.0, elevation=-90.0).elevation == -90.0


class TestInvalidConstruction:
    def test_azimuth_360_rejected(self):
        with pytest.raises(ValueError, match="Azimuth"):
            Position(azimuth=360.0, elevation=0.0)

    def test_azimuth_negative_rejected(self):
        with pytest.raises(ValueError, match="Azimuth"):
            Position(azimuth=-0.1, elevation=0.0)

    def test_azimuth_wrapped_value_rejected(self):
        with pytest.raises(ValueError, match="Azimuth"):
            Position(azimuth=725.3, elevation=0.0)

    def test_elevation_above_90_rejected(self):
        with pytest.raises(ValueError, match="Elevation"):
            Position(azimuth=0.0, elevation=90.1)

    def test_elevation_below_minus_90_rejected(self):
        with pytest.raises(ValueError, match="Elevation"):
            Position(azimuth=0.0, elevation=-90.1)


class TestValueSemantics:
    def test_equality_by_value(self):
        assert Position(10.0, 20.0) == Position(10.0, 20.0)

    def test_inequality(self):
        assert Position(10.0, 20.0) != Position(10.0, 21.0)

    def test_immutable(self):
        pos = Position(10.0, 20.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            pos.azimuth = 30.0
