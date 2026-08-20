"""Unit tests for the command-line interface.

The rotor is a ``MagicMock(spec=Rotor)``: scripted like a stub, but
spec'd against the real class, so a test can't keep passing against a
method the facade no longer has.

Nothing here opens a serial port, and the tests that don't use --send
assert that no rotor is built at all.
"""

import textwrap
from unittest.mock import MagicMock

import pytest

from qsorbit import __version__
from qsorbit.__main__ import build_parser, main
from qsorbit.core.rotor import Arrival, HomingError, Position, Rotor, RotorErrorCode, RotorStatus

# Vallado AIAA 2006-6753 Appendix C example, the same TLE used throughout
# tests/unit/tracker/. Times below sit a few days past its epoch, inside
# the window where SGP4 is well behaved.
TEME_EXAMPLE_TLE = """\
TEME EXAMPLE
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
"""

IN_WINDOW_TIME = "2000-07-01T18:50:00+00:00"

CONFIG = """
    [observer]
    latitude = 40.5
    longitude = -83.25
    altitude_m = 250.0

    [rotor]
    port = "COM5"
    baudrate = 19200

    [rotor.capabilities]
    azimuth_min_deg = 0.0
    azimuth_max_deg = 360.0
    elevation_min_deg = 0.0
    elevation_max_deg = 180.0
    azimuth_wrap = "extra_rotation"
    acceptance_window_deg = 2.5
    rs485_turnaround_s = 0.15
    firmware_version = "SatNOGS-v2.2.1"
"""


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "qsorbit.toml"
    path.write_text(textwrap.dedent(CONFIG), encoding="utf-8")
    return path


@pytest.fixture
def tle_path(tmp_path):
    path = tmp_path / "example.tle"
    path.write_text(TEME_EXAMPLE_TLE, encoding="utf-8")
    return path


#: What a freshly homed rotator reports: slightly past zero on both
#: axes, which is normal for an axis resting against its end-stop.
HOMED_POSITION = Position(-1.5, 2.0)


def make_rotor(
    *,
    position=HOMED_POSITION,
    error=RotorErrorCode.NO_ERROR,
    firmware="SatNOGS-v2.2.1",
    arrived=True,
    connect_raises=None,
):
    """A stand-in rotor, spec'd against the real facade."""
    rotor = MagicMock(spec=Rotor)
    rotor.firmware_version = firmware
    if connect_raises is not None:
        rotor.connect.side_effect = connect_raises
    status = RotorStatus(firmware_version=firmware, error=error, position=position)
    rotor.connect.return_value = status
    rotor.status.return_value = status
    rotor.stop.return_value = position
    rotor.wait_for_arrival.return_value = Arrival(arrived=arrived, position=position, elapsed_s=4.2)
    return rotor


@pytest.fixture
def factory():
    """A rotor factory that records its calls and hands back one mock."""
    rotor = make_rotor()
    calls = []

    def build(config, on_homing_wait):
        calls.append((config, on_homing_wait))
        return rotor

    build.rotor = rotor
    build.calls = calls
    return build


def run(argv, config_path, factory):
    return main([*argv[:0], "--config", str(config_path), *argv], rotor_factory=factory)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_no_command_is_a_usage_error(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main([])
        assert exit_info.value.code == 2

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_point_requires_a_tle(self):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(["point"])
        assert exit_info.value.code == 2

    def test_send_defaults_to_off(self):
        # The safety-relevant default, pinned so it can't be flipped by
        # accident: computing is free, moving is opt-in.
        assert build_parser().parse_args(["point", "--tle", "x"]).send is False

    def test_status_takes_no_arguments(self):
        assert build_parser().parse_args(["status"]).command == "status"


# ---------------------------------------------------------------------------
# point
# ---------------------------------------------------------------------------


class TestPoint:
    def test_prints_sky_rotor_and_command(self, config_path, tle_path, factory, capsys):
        code = run(["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME], config_path, factory)

        out = capsys.readouterr().out
        assert code == 0
        assert "TEME EXAMPLE" in out
        assert "Sky:" in out
        assert "Rotor:" in out
        assert "Command:   AZ" in out
        assert " EL" in out

    def test_says_nothing_was_sent(self, config_path, tle_path, factory, capsys):
        run(["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME], config_path, factory)

        assert "Nothing was sent" in capsys.readouterr().out

    def test_does_not_even_build_a_rotor_without_send(self, config_path, tle_path, factory):
        # Computing must not touch the serial port. On a rotator whose
        # adapter resets on open, merely connecting triggers a re-home.
        run(["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME], config_path, factory)

        assert factory.calls == []

    def test_says_no_calibration_is_applied(self, config_path, tle_path, factory, capsys):
        # sky_to_rotor is an identity today. An interface that didn't say
        # so would be implying an accuracy the software doesn't have.
        run(["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME], config_path, factory)

        assert "No alignment calibration is applied" in capsys.readouterr().out

    def test_labels_the_rotor_line_as_an_axis_command(self, config_path, tle_path, factory, capsys):
        run(["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME], config_path, factory)

        assert "(axis command)" in capsys.readouterr().out

    def test_send_moves_and_reports_arrival(self, config_path, tle_path, factory, capsys):
        code = run(
            ["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME, "--send"],
            config_path,
            factory,
        )

        out = capsys.readouterr().out
        assert code == 0
        assert factory.rotor.move_to.call_count == 1
        assert isinstance(factory.rotor.move_to.call_args.args[0], Position)
        assert "Arrived:" in out

    def test_send_closes_the_port(self, config_path, tle_path, factory):
        run(
            ["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME, "--send"],
            config_path,
            factory,
        )

        assert factory.rotor.close.call_count == 1

    def test_failure_to_settle_is_reported_and_non_zero(self, config_path, tle_path, capsys):
        rotor = make_rotor(arrived=False)
        factory = lambda config, on_wait: rotor  # noqa: E731

        code = main(
            [
                "--config",
                str(config_path),
                "point",
                "--tle",
                str(tle_path),
                "--at",
                IN_WINDOW_TIME,
                "--send",
            ],
            rotor_factory=factory,
        )

        assert code == 1
        assert "Did not settle" in capsys.readouterr().err

    def test_out_of_range_target_is_refused_before_connecting(
        self, tmp_path, tle_path, factory, capsys
    ):
        # Elevation limits of 91-92 degrees can never be satisfied by a
        # sky position, because AzEl caps elevation at 90 by definition.
        # That makes this refusal deterministic without depending on
        # where the satellite happens to be.
        config = tmp_path / "narrow.toml"
        config.write_text(
            textwrap.dedent(CONFIG)
            .replace("elevation_min_deg = 0.0", "elevation_min_deg = 91.0")
            .replace("elevation_max_deg = 180.0", "elevation_max_deg = 92.0"),
            encoding="utf-8",
        )

        code = main(
            [
                "--config",
                str(config),
                "point",
                "--tle",
                str(tle_path),
                "--at",
                IN_WINDOW_TIME,
                "--send",
            ],
            rotor_factory=factory,
        )

        assert code == 1
        assert "Out of range" in capsys.readouterr().err
        assert factory.calls == []

    def test_bad_time_is_rejected(self, config_path, tle_path, factory, capsys):
        code = run(
            ["point", "--tle", str(tle_path), "--at", "half past four"], config_path, factory
        )

        assert code == 1
        assert "ISO 8601" in capsys.readouterr().err

    def test_missing_tle_file(self, config_path, tmp_path, factory, capsys):
        code = run(["point", "--tle", str(tmp_path / "nope.tle")], config_path, factory)

        assert code == 1
        assert capsys.readouterr().err.startswith("Error:")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_reports_the_essentials(self, config_path, factory, capsys):
        code = run(["status"], config_path, factory)

        out = capsys.readouterr().out
        assert code == 0
        assert "SatNOGS-v2.2.1" in out
        assert "COM5" in out
        assert str(config_path) in out
        assert "Error:     none" in out

    def test_labels_position_as_an_axis_reading(self, config_path, factory, capsys):
        # The rotor reports -1.5 after homing. Printed without a label,
        # that reads like a compass bearing, which it is not.
        code = run(["status"], config_path, factory)

        out = capsys.readouterr().out
        assert code == 0
        assert "AZ -1.5" in out
        assert "not compass bearings" in out

    def test_shows_travel_limits_as_axis_travel(self, config_path, factory, capsys):
        run(["status"], config_path, factory)

        assert "(axis travel)" in capsys.readouterr().out

    def test_never_moves_anything(self, config_path, factory, capsys):
        run(["status"], config_path, factory)

        assert factory.rotor.move_to.call_count == 0
        assert factory.rotor.stop.call_count == 0

    def test_warns_on_a_firmware_mismatch(self, config_path, capsys):
        rotor = make_rotor(firmware="SatNOGS-v2.2")
        code = main(["--config", str(config_path), "status"], rotor_factory=lambda c, w: rotor)

        out = capsys.readouterr().out
        assert code == 0
        assert "Config declares SatNOGS-v2.2.1" in out

    def test_unhealthy_rotor_exits_non_zero(self, config_path, capsys):
        rotor = make_rotor(error=RotorErrorCode.OVER_TEMPERATURE)
        code = main(["--config", str(config_path), "status"], rotor_factory=lambda c, w: rotor)

        assert code == 1
        assert "over temperature" in capsys.readouterr().out

    def test_latched_homing_error_is_shown_with_the_fix(self, config_path, capsys):
        # status() reports it rather than raising, so the operator can
        # see the state - and the line has to say what to do about it.
        rotor = make_rotor(error=RotorErrorCode.HOMING_ERROR)
        code = main(["--config", str(config_path), "status"], rotor_factory=lambda c, w: rotor)

        assert code == 1
        assert "power-cycle the controller" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_reports_where_it_stopped(self, config_path, factory, capsys):
        code = run(["stop"], config_path, factory)

        assert code == 0
        assert factory.rotor.stop.call_count == 1
        assert "Stopped:   AZ -1.5" in capsys.readouterr().out

    def test_says_it_is_not_an_emergency_stop(self, config_path, factory, capsys):
        # It halts a converging loop by setting the setpoint to the
        # current position. It cannot stop a diverging one, and a user
        # who believes otherwise reaches for the wrong thing.
        run(["stop"], config_path, factory)

        out = capsys.readouterr().out
        assert "cannot stop a diverging one" in out
        assert "power switch" in out


# ---------------------------------------------------------------------------
# Failure paths shared by every command
# ---------------------------------------------------------------------------


class TestFailures:
    def test_homing_failure_is_its_own_message(self, config_path, capsys):
        rotor = make_rotor(connect_raises=HomingError("Power-cycle the controller."))
        code = main(["--config", str(config_path), "status"], rotor_factory=lambda c, w: rotor)

        assert code == 1
        assert "Homing failure" in capsys.readouterr().err

    def test_missing_config_lists_where_it_looked(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("APPDATA", str(tmp_path / "empty"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))

        code = main(["status"], rotor_factory=lambda c, w: make_rotor())

        assert code == 1
        assert "No station config file found" in capsys.readouterr().err

    def test_explicitly_named_config_that_is_missing(self, tmp_path, capsys):
        code = main(
            ["--config", str(tmp_path / "nope.toml"), "status"],
            rotor_factory=lambda c, w: make_rotor(),
        )

        assert code == 1
        assert "not found" in capsys.readouterr().err
