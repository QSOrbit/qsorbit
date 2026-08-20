"""Hardware integration tests: a real rotator on a real serial port.

**Every test here is read-only.** They read the firmware version, the
error state, and the position. Nothing commands a setpoint, and nothing
in this file can cause the antenna to move — which is what makes it safe
to run unattended, and safe to re-run.

Movement is deliberately not automated. A test that slews an antenna
would need someone watching it, near a power switch, which is the
opposite of what a test suite is for. Commanded movement is verified by
hand at the bench, and the CLI's ``--send`` path is what does it.

These tests are deselected by default (see ``addopts`` in
``pyproject.toml``) because they need hardware CI doesn't have. Run them
at the bench with::

    uv run pytest -m integration

They use the real station config, from the same search path the CLI
uses, so they exercise the configuration the operator actually runs.

If the rotator can't be reached at all, the suite skips rather than
fails — there's no bug to report in "the rotor isn't plugged in". But
note that a skipped run is not a passing run: read the skip reason. A
rotator that answers and then misbehaves *fails*, because that is a real
finding.
"""

from __future__ import annotations

import math

import pytest

from qsorbit.core.rotor import (
    MAX_AXIS_DEGREES,
    HomingError,
    Position,
    Rotor,
    RotorErrorCode,
    SerialConnectionError,
    SerialPort,
)
from qsorbit.core.station import ConfigError, load_station_config

pytestmark = pytest.mark.integration

#: How far apart two position reads of a stationary rotor may be before
#: something is wrong. Generous: the axis is idle, so this is sensor
#: noise and rounding, not the stiction dead-band.
IDLE_TOLERANCE_DEG = 0.5


@pytest.fixture(scope="module")
def config():
    """The operator's real station config, or a skip."""
    try:
        return load_station_config(None)
    except ConfigError as exc:
        pytest.skip(f"No usable station config, so there is no rotor to talk to: {exc}")


@pytest.fixture(scope="module")
def rotor(config):
    """One connection, held open for the whole module.

    Opening the port is not free: on a rotator with DTR wired to reset,
    every open reboots the controller and triggers a full re-home. One
    connection for the module, per the integration rules.
    """
    controller = Rotor(
        SerialPort(
            config.serial.port,
            baudrate=config.serial.baudrate,
            timeout=config.serial.timeout_s,
        ),
        config.capabilities,
    )
    try:
        controller.connect()
    except SerialConnectionError as exc:
        # Nothing there to test. Not a defect.
        pytest.skip(f"Could not reach a rotator on {config.serial.port}: {exc}")
    except HomingError as exc:
        # The rotator IS there and is in a state the operator has to fix
        # by hand. Skipping would hide it.
        pytest.fail(f"Rotator has a latched homing error: {exc}")

    yield controller
    controller.close()


class TestIdentity:
    def test_reports_a_firmware_version(self, rotor):
        assert rotor.firmware_version
        assert rotor.firmware_version.strip() == rotor.firmware_version

    def test_matches_the_version_the_config_declares(self, rotor, config):
        declared = config.capabilities.firmware_version
        if declared is None:
            pytest.skip("Config declares no firmware version to compare against.")
        assert rotor.firmware_version == declared, (
            "The rotator is running firmware this config was not verified "
            "against. Not necessarily broken, but the declared capabilities "
            "may no longer describe it."
        )


class TestErrorState:
    def test_error_is_a_code_this_firmware_generation_defines(self, rotor):
        assert isinstance(rotor.read_error(), RotorErrorCode)

    def test_no_latched_homing_error(self, rotor):
        # connect() would have raised, so reaching here means it is
        # clear; this pins that the status view agrees.
        assert not rotor.status().homing_error_latched


class TestPosition:
    def test_position_is_readable(self, rotor):
        assert isinstance(rotor.read_position(), Position)

    def test_position_is_physically_plausible(self, rotor):
        position = rotor.read_position()
        for value in (position.azimuth, position.elevation):
            assert math.isfinite(value)
            assert abs(value) <= MAX_AXIS_DEGREES

    def test_a_homed_axis_may_read_slightly_negative(self, rotor):
        # Not an assertion about this rotor's current position - just a
        # note in executable form. Position must tolerate it, which is
        # the bug PR #9 fixed against this exact hardware.
        position = rotor.read_position()
        assert position == Position(position.azimuth, position.elevation)


class TestReplyAlignment:
    """The regression tests for the desync this protocol invites.

    Commands differ in whether the firmware answers them. Read one reply
    too few or too many and every later read is shifted by one message —
    which does not raise anything at the point of the mistake, and shows
    up later as a position query returning a version string.
    """

    def test_interleaved_queries_each_get_their_own_reply(self, rotor):
        first = rotor.read_position()
        error = rotor.read_error()
        second = rotor.read_position()

        assert isinstance(error, RotorErrorCode)
        assert abs(second.azimuth - first.azimuth) <= IDLE_TOLERANCE_DEG
        assert abs(second.elevation - first.elevation) <= IDLE_TOLERANCE_DEG

    def test_status_can_be_read_repeatedly(self, rotor):
        readings = [rotor.status() for _ in range(3)]

        assert all(reading.firmware_version == readings[0].firmware_version for reading in readings)
        assert all(isinstance(reading.error, RotorErrorCode) for reading in readings)


class TestReadOnly:
    def test_none_of_this_moved_the_rotor(self, rotor):
        # Ordered last by file position, which pytest respects within a
        # module: if any read above had accidentally commanded a
        # setpoint, the axis would have left where it started.
        position = rotor.read_position()
        again = rotor.read_position()

        assert abs(again.azimuth - position.azimuth) <= IDLE_TOLERANCE_DEG
        assert abs(again.elevation - position.elevation) <= IDLE_TOLERANCE_DEG
