"""Unit tests for the per-rotor capability record."""

import pytest

from qsorbit.core.rotor import (
    AzimuthWrap,
    Position,
    PositionLimitError,
    RotorCapabilities,
)


def phils_rotator(**overrides) -> RotorCapabilities:
    """The measured capabilities of Phil's rotator, unless overridden.

    Values come from the bench session against stock firmware v2.2.1 —
    see ``qsorbit-rotor-integration.md`` section 4.
    """
    fields = {
        "azimuth_min_deg": 0.0,
        "azimuth_max_deg": 360.0,
        "elevation_min_deg": 0.0,
        "elevation_max_deg": 180.0,
        "azimuth_wrap": AzimuthWrap.EXTRA_ROTATION,
        "acceptance_window_deg": 2.5,
        "rs485_turnaround_s": 0.15,
        "firmware_version": "SatNOGS-v2.2.1",
    }
    fields.update(overrides)
    return RotorCapabilities(**fields)


class TestConstruction:
    def test_typical(self):
        caps = phils_rotator()
        assert caps.azimuth_max_deg == 360.0
        assert caps.azimuth_wrap is AzimuthWrap.EXTRA_ROTATION
        assert caps.firmware_version == "SatNOGS-v2.2.1"

    def test_is_a_value_object(self):
        assert phils_rotator() == phils_rotator()

    def test_frozen(self):
        with pytest.raises(Exception):  # noqa: B017 - dataclasses raises FrozenInstanceError
            phils_rotator().azimuth_max_deg = 400.0

    def test_firmware_version_is_optional(self):
        # It is read from the rotor at connect, so a config that hasn't
        # recorded one yet is legitimate.
        assert phils_rotator(firmware_version=None).firmware_version is None

    def test_extended_travel_rotor(self):
        caps = phils_rotator(azimuth_max_deg=450.0, azimuth_wrap=AzimuthWrap.EXTENDED_TRAVEL)
        assert caps.azimuth_max_deg == 450.0


class TestValidation:
    def test_inverted_azimuth_limits_rejected(self):
        with pytest.raises(ValueError, match="Azimuth limits"):
            phils_rotator(azimuth_min_deg=300.0, azimuth_max_deg=10.0)

    def test_equal_azimuth_limits_rejected(self):
        with pytest.raises(ValueError, match="Azimuth limits"):
            phils_rotator(azimuth_min_deg=10.0, azimuth_max_deg=10.0)

    def test_inverted_elevation_limits_rejected(self):
        with pytest.raises(ValueError, match="Elevation limits"):
            phils_rotator(elevation_min_deg=90.0, elevation_max_deg=0.0)

    def test_limit_beyond_representable_axis_rejected(self):
        with pytest.raises(ValueError, match="beyond"):
            phils_rotator(azimuth_wrap=AzimuthWrap.EXTENDED_TRAVEL, azimuth_max_deg=5000.0)

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            phils_rotator(acceptance_window_deg=float("nan"))

    def test_zero_acceptance_window_rejected(self):
        # A zero window can never be satisfied: stock gains leave a
        # steady-state shortfall of 1-2 degrees on a healthy rotor.
        with pytest.raises(ValueError, match="acceptance_window_deg must be positive"):
            phils_rotator(acceptance_window_deg=0.0)

    def test_negative_acceptance_window_rejected(self):
        with pytest.raises(ValueError, match="acceptance_window_deg must be positive"):
            phils_rotator(acceptance_window_deg=-1.0)

    def test_negative_turnaround_rejected(self):
        with pytest.raises(ValueError, match="rs485_turnaround_s"):
            phils_rotator(rs485_turnaround_s=-0.1)

    def test_zero_turnaround_allowed(self):
        # Unwise on a real RS-485 link, but it is a tuning value, not a
        # safety one - and a direct TTL connection genuinely needs none.
        assert phils_rotator(rs485_turnaround_s=0.0).rs485_turnaround_s == 0.0

    def test_wrap_behaviour_must_be_declared(self):
        with pytest.raises(ValueError, match="declared per rotor"):
            phils_rotator(azimuth_wrap="extra_rotation")

    def test_extra_rotation_rotor_cannot_declare_travel_past_360(self):
        # The config-typo guard. On a wrapping rotor, commanding 380 is
        # not "20 degrees more travel" - it is a full extra rotation
        # against the cable, then a settle at 20.
        with pytest.raises(ValueError, match="EXTRA_ROTATION"):
            phils_rotator(azimuth_max_deg=450.0)


class TestCheckSetpoint:
    def test_in_range_passes(self):
        phils_rotator().check_setpoint(Position(180.0, 45.0))

    def test_limits_are_inclusive(self):
        caps = phils_rotator()
        caps.check_setpoint(Position(0.0, 0.0))
        caps.check_setpoint(Position(360.0, 180.0))

    def test_azimuth_above_max_refused(self):
        with pytest.raises(PositionLimitError, match="Azimuth"):
            phils_rotator().check_setpoint(Position(380.0, 45.0))

    def test_azimuth_below_min_refused(self):
        with pytest.raises(PositionLimitError, match="Azimuth"):
            phils_rotator().check_setpoint(Position(-10.0, 45.0))

    def test_elevation_above_max_refused(self):
        with pytest.raises(PositionLimitError, match="Elevation"):
            phils_rotator().check_setpoint(Position(180.0, 190.0))

    def test_elevation_below_min_refused(self):
        with pytest.raises(PositionLimitError, match="Elevation"):
            phils_rotator().check_setpoint(Position(180.0, -5.0))

    def test_message_says_the_command_was_not_sent(self):
        # The firmware has no limits of its own at any level, so this
        # refusal is the only guard there is - the message should say so
        # rather than reading like a report of something that happened.
        with pytest.raises(PositionLimitError, match="was not sent"):
            phils_rotator().check_setpoint(Position(720.0, 45.0))

    def test_position_a_rotor_could_report_but_not_be_sent_to(self):
        # Position itself accepts -1.5 (a homed rotor really does report
        # it). The travel limit is a separate question, answered here.
        readable = Position(-1.5, 2.0)
        with pytest.raises(PositionLimitError):
            phils_rotator().check_setpoint(readable)


class TestIsArrived:
    def test_exact_match(self):
        caps = phils_rotator()
        assert caps.is_arrived(Position(180.0, 45.0), Position(180.0, 45.0))

    def test_within_window(self):
        # The normal outcome of a healthy move: stock gains leave the
        # axis short by ~1.5 az / ~2.1 el.
        caps = phils_rotator()
        assert caps.is_arrived(Position(180.0, 45.0), Position(178.5, 42.9))

    def test_at_the_window_edge_counts_as_arrived(self):
        caps = phils_rotator()
        assert caps.is_arrived(Position(180.0, 45.0), Position(177.5, 42.5))

    def test_azimuth_outside_window(self):
        caps = phils_rotator()
        assert not caps.is_arrived(Position(180.0, 45.0), Position(174.0, 45.0))

    def test_elevation_outside_window(self):
        caps = phils_rotator()
        assert not caps.is_arrived(Position(180.0, 45.0), Position(180.0, 51.0))

    def test_overshoot_counts_the_same_as_shortfall(self):
        caps = phils_rotator()
        assert caps.is_arrived(Position(180.0, 45.0), Position(181.0, 46.0))

    def test_comparison_is_mechanical_not_modular(self):
        # A Position is an axis reading, not a compass bearing: an axis
        # at 359 and one at 1 are 358 degrees of travel apart, not 2.
        caps = phils_rotator()
        assert not caps.is_arrived(Position(1.0, 0.0), Position(359.0, 0.0))

    def test_a_wider_window_accepts_more(self):
        assert phils_rotator(acceptance_window_deg=10.0).is_arrived(
            Position(180.0, 45.0), Position(174.0, 45.0)
        )
