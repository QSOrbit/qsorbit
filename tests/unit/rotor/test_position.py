"""Unit tests for the Position dataclass.

Position is a *mechanical axis reading*, not a compass bearing, so its
validation is deliberately permissive: it rejects only values that
cannot have come from a working rotor. Travel limits are the pointing
layer's job, since they depend on the specific hardware.

The negative-angle cases here are not hypothetical — they come from a
real SatNOGS rotator, which reports AZ-1.5 EL2.0 when freshly homed.
An earlier version of this class rejected exactly that, which meant
QSOrbit could not read its own rotor's position.
"""

import dataclasses
import math

import pytest

from qsorbit.core.rotor import MAX_AXIS_DEGREES, Position


class TestValidConstruction:
    def test_typical_position(self):
        pos = Position(azimuth=180.0, elevation=45.0)
        assert pos.azimuth == 180.0
        assert pos.elevation == 45.0

    def test_zero(self):
        assert Position(azimuth=0.0, elevation=0.0).azimuth == 0.0

    def test_real_homed_rotor_reading(self):
        # Observed on Phil's SatNOGS rotator immediately after homing.
        # This is the regression case for the whole class of bug.
        pos = Position(azimuth=-1.5, elevation=2.0)
        assert pos.azimuth == -1.5
        assert pos.elevation == 2.0

    def test_small_negative_angles_accepted(self):
        # A rotor homed against an end-stop routinely settles just past
        # its zero.
        assert Position(azimuth=-0.1, elevation=-0.3).azimuth == -0.1

    def test_beyond_360_accepted(self):
        # On a multi-turn azimuth axis, 380 and 20 are different physical
        # places reached by different amounts of travel.
        assert Position(azimuth=380.0, elevation=0.0).azimuth == 380.0

    def test_elevation_past_vertical_accepted(self):
        # Rotors that can flip past zenith command elevation above 90.
        assert Position(azimuth=0.0, elevation=135.0).elevation == 135.0

    def test_magnitude_bound_inclusive(self):
        assert Position(azimuth=MAX_AXIS_DEGREES, elevation=0.0).azimuth == MAX_AXIS_DEGREES
        assert Position(azimuth=-MAX_AXIS_DEGREES, elevation=0.0).azimuth == -MAX_AXIS_DEGREES


class TestInvalidConstruction:
    """Only genuinely impossible readings are rejected."""

    def test_azimuth_beyond_magnitude_bound_rejected(self):
        with pytest.raises(ValueError, match="Azimuth"):
            Position(azimuth=MAX_AXIS_DEGREES + 0.1, elevation=0.0)

    def test_elevation_beyond_magnitude_bound_rejected(self):
        with pytest.raises(ValueError, match="Elevation"):
            Position(azimuth=0.0, elevation=-(MAX_AXIS_DEGREES + 0.1))

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            Position(azimuth=math.nan, elevation=0.0)
        with pytest.raises(ValueError, match="finite"):
            Position(azimuth=0.0, elevation=math.nan)

    def test_infinity_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            Position(azimuth=math.inf, elevation=0.0)
        with pytest.raises(ValueError, match="finite"):
            Position(azimuth=0.0, elevation=-math.inf)

    def test_error_says_it_is_not_a_travel_limit(self):
        # The message needs to stop a future reader from mistaking this
        # for hardware limit enforcement.
        with pytest.raises(ValueError, match="not a travel limit"):
            Position(azimuth=99999.0, elevation=0.0)


class TestValueSemantics:
    def test_equality_by_value(self):
        assert Position(10.0, 20.0) == Position(10.0, 20.0)

    def test_inequality(self):
        assert Position(10.0, 20.0) != Position(10.0, 21.0)

    def test_immutable(self):
        pos = Position(10.0, 20.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            pos.azimuth = 30.0

    def test_hashable(self):
        assert len({Position(1.0, 2.0), Position(1.0, 2.0)}) == 1
