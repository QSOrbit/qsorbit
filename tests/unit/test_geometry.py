"""Unit tests for the AzEl sky-direction dataclass.

Validation-only — AzEl carries no math, just range rules. Mirrors the
structure of tests/unit/rotor/test_position.py, since the two types have
deliberately similar validation but different meanings.
"""

import dataclasses

import pytest

from qsorbit.core.geometry import AzEl


class TestValidConstruction:
    def test_typical_direction(self):
        d = AzEl(azimuth=180.0, elevation=45.0)
        assert d.azimuth == 180.0
        assert d.elevation == 45.0

    def test_azimuth_lower_bound_inclusive(self):
        assert AzEl(azimuth=0.0, elevation=0.0).azimuth == 0.0

    def test_azimuth_just_below_upper_bound(self):
        assert AzEl(azimuth=359.9, elevation=0.0).azimuth == 359.9

    def test_elevation_bounds_inclusive(self):
        assert AzEl(azimuth=0.0, elevation=90.0).elevation == 90.0
        assert AzEl(azimuth=0.0, elevation=-90.0).elevation == -90.0

    def test_negative_elevation_allowed(self):
        # A satellite that hasn't risen yet has a real, computable
        # position below the horizon; the tracker is expected to report it
        # rather than treat it as an error.
        assert AzEl(azimuth=0.0, elevation=-30.0).elevation == -30.0


class TestInvalidConstruction:
    def test_azimuth_360_rejected(self):
        with pytest.raises(ValueError, match="Azimuth"):
            AzEl(azimuth=360.0, elevation=0.0)

    def test_azimuth_negative_rejected(self):
        with pytest.raises(ValueError, match="Azimuth"):
            AzEl(azimuth=-0.1, elevation=0.0)

    def test_azimuth_wrapped_value_rejected(self):
        with pytest.raises(ValueError, match="Azimuth"):
            AzEl(azimuth=725.3, elevation=0.0)

    def test_elevation_above_90_rejected(self):
        # Unlike a rotor, which can be commanded past vertical in flip
        # mode, a sky elevation above 90 is geometrically meaningless.
        with pytest.raises(ValueError, match="Elevation"):
            AzEl(azimuth=0.0, elevation=90.1)

    def test_elevation_below_minus_90_rejected(self):
        with pytest.raises(ValueError, match="Elevation"):
            AzEl(azimuth=0.0, elevation=-90.1)


class TestValueSemantics:
    def test_equality_by_value(self):
        assert AzEl(10.0, 20.0) == AzEl(10.0, 20.0)

    def test_inequality(self):
        assert AzEl(10.0, 20.0) != AzEl(10.0, 21.0)

    def test_immutable(self):
        d = AzEl(10.0, 20.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.azimuth = 30.0

    def test_hashable(self):
        # Frozen dataclasses hash by value, which matters for caching and
        # for use as dict keys or set members later.
        assert len({AzEl(1.0, 2.0), AzEl(1.0, 2.0)}) == 1
