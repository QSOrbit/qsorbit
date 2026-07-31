"""Tests for the sky-to-rotor pointing layer.

The physics (orbit propagation, topocentric conversion) is validated
against external truth in tests/unit/tracker/. This file checks the
wiring and the seam: that compute_pointing_command() chains
Target.topocentric_state() through sky_to_rotor() into
rotor.format_set_position(), and that the conversion step is a real,
separately testable place rather than an inline construction.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from skyfield.api import load

from qsorbit.core.geometry import AzEl
from qsorbit.core.pointing import compute_pointing_command, sky_to_rotor
from qsorbit.core.rotor import Position, format_set_position
from qsorbit.core.tracker import (
    ObserverLocation,
    PropagationError,
    Satellite,
    Target,
    TopocentricState,
)

# Same TLE used throughout tests/unit/tracker/ (Vallado AIAA 2006-6753
# Appendix C example).
_TEME_EXAMPLE_TLE = """\
TEME EXAMPLE
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
"""

# GOCE's last-ever element set before re-entry, from skyfield's
# "Detecting Propagation Errors" documentation — also reused from
# tests/unit/tracker/.
_GOCE_DECAY_TLE = """\
GOCE
1 34602U 09013A   13314.96046236  .14220718  20669-5  50412-4 0   930
2 34602 096.5717 344.5256 0009826 296.2811 064.0942 16.58673376272979
"""


def _valid_time_for(sat: Satellite) -> datetime:
    """A time 3 days past the TLE epoch — inside a known-good window."""
    ts = load.timescale(builtin=True)
    epoch = sat.skyfield_satellite.epoch
    return ts.tt_jd(epoch.whole + 3.0, epoch.tt_fraction).utc_datetime()


class TestSkyToRotor:
    def test_returns_a_rotor_position(self):
        assert isinstance(sky_to_rotor(AzEl(180.0, 45.0)), Position)

    def test_currently_applies_no_correction(self):
        # No calibration data, no travel limits, no pass plan yet, so the
        # best command for a sky direction is that same direction.
        #
        # This test is DESIGNED TO FAIL when calibration, flip mode, or
        # limit handling lands. That is deliberate: it forces whoever
        # adds a correction to consciously update an assertion that says
        # "this used to pass through untouched", rather than silently
        # changing where every rotor in the field points. If you are
        # here because this test broke, that is the system working.
        result = sky_to_rotor(AzEl(123.4, 56.7))
        assert result.azimuth == 123.4
        assert result.elevation == 56.7

    def test_preserves_boundary_values(self):
        assert sky_to_rotor(AzEl(0.0, -90.0)) == Position(0.0, -90.0)
        assert sky_to_rotor(AzEl(359.9, 90.0)) == Position(359.9, 90.0)


class TestSatelliteSatisfiesTarget:
    def test_satellite_is_a_target(self):
        # Structural conformance: Satellite implements the protocol
        # without importing or subclassing it. Note isinstance works here
        # but issubclass does not — protocols with non-method members
        # (Target has a `name` property) don't support issubclass.
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        assert isinstance(sat, Target)


class TestComputePointingCommand:
    def test_matches_manually_chaining_the_pieces(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        observer = ObserverLocation(latitude=40.0, longitude=-83.0)
        time = _valid_time_for(sat)

        command = compute_pointing_command(sat, observer, time)

        expected_state = sat.topocentric_state(observer, time)
        assert command == format_set_position(sky_to_rotor(expected_state.sky_position))

    def test_command_is_well_formed_easycomm(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        observer = ObserverLocation(latitude=40.0, longitude=-83.0)

        command = compute_pointing_command(sat, observer, _valid_time_for(sat))

        assert command.startswith(b"AZ")
        assert b" EL" in command
        assert command.endswith(b"\n")

    def test_accepts_any_target_not_just_satellite(self):
        # Proves the protocol is actually load-bearing: a stand-in target
        # that reports a fixed sky position drives the pointing path with
        # no satellite involved. This is the seam a star or planet target
        # will use later.
        class FixedTarget:
            @property
            def name(self) -> str:
                return "TEST BEACON"

            def topocentric_state(self, observer, time):
                return TopocentricState(
                    sky_position=AzEl(azimuth=90.0, elevation=30.0),
                    range_km=1000.0,
                    range_rate_km_s=0.0,
                )

        observer = ObserverLocation(latitude=0.0, longitude=0.0)
        command = compute_pointing_command(
            FixedTarget(), observer, datetime(2026, 1, 1, tzinfo=UTC)
        )
        assert command == b"AZ90.0 EL30.0\n"

    def test_naive_datetime_rejected(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        observer = ObserverLocation(latitude=40.0, longitude=-83.0)
        naive = datetime(2000, 6, 30, 18, 50, 20)
        with pytest.raises(ValueError, match="timezone-aware"):
            compute_pointing_command(sat, observer, naive)

    def test_propagation_error_propagates(self):
        sat = Satellite.from_tle(_GOCE_DECAY_TLE)
        observer = ObserverLocation(latitude=0.0, longitude=0.0)
        with pytest.raises(PropagationError):
            compute_pointing_command(sat, observer, datetime(2013, 11, 9, tzinfo=UTC))
