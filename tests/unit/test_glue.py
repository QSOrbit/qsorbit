"""Integration-style test for the tracker-to-rotor glue function.

The physics (orbit propagation, topocentric conversion) is already
validated against external truth in tests/unit/tracker/. This checks
the wiring — that compute_pointing_command() correctly chains
Satellite.topocentric_state() into rotor.format_set_position() — not
new reference numbers.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from skyfield.api import load

from qsorbit.core.glue import compute_pointing_command
from qsorbit.core.rotor import format_set_position
from qsorbit.core.tracker import ObserverLocation, PropagationError, Satellite

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


class TestComputePointingCommand:
    def test_matches_manually_chaining_topocentric_state_and_format(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        observer = ObserverLocation(latitude=40.0, longitude=-83.0)
        ts = load.timescale(builtin=True)
        epoch = sat.skyfield_satellite.epoch
        t = ts.tt_jd(epoch.whole + 3.0, epoch.tt_fraction)
        time = t.utc_datetime()

        command = compute_pointing_command(sat, observer, time)

        expected_state = sat.topocentric_state(observer, time)
        assert command == format_set_position(expected_state.position)

    def test_command_is_well_formed_easycomm(self):
        sat = Satellite.from_tle(_TEME_EXAMPLE_TLE)
        observer = ObserverLocation(latitude=40.0, longitude=-83.0)
        ts = load.timescale(builtin=True)
        epoch = sat.skyfield_satellite.epoch
        t = ts.tt_jd(epoch.whole + 3.0, epoch.tt_fraction)

        command = compute_pointing_command(sat, observer, t.utc_datetime())

        assert command.startswith(b"AZ")
        assert b" EL" in command
        assert command.endswith(b"\n")

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
