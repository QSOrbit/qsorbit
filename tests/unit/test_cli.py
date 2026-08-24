"""Unit tests for the command-line interface.

The rotor is a ``MagicMock(spec=Rotor)``: scripted like a stub, but
spec'd against the real class, so a test can't keep passing against a
method the facade no longer has.

Nothing here opens a serial port, and the tests that don't use --send
assert that no rotor is built at all.
"""

import json
import textwrap
from unittest.mock import MagicMock

import pytest

from qsorbit import __version__
from qsorbit.__main__ import DEFAULT_TUNING_OFFSET_KHZ, build_parser, main
from qsorbit.core.rotor import Arrival, HomingError, Position, Rotor, RotorErrorCode, RotorStatus
from qsorbit.core.sdr import AppliedSettings, DeviceError, DeviceInfo, TunerType

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


# ---------------------------------------------------------------------------
# SDR
# ---------------------------------------------------------------------------

V4_GAIN_STEPS_DB = (0.0, 0.9, 12.5, 32.8, 49.6)


class FakeSdr:
    """A stand-in device, written by hand rather than mocked.

    ``capture_to_file`` drives a real streaming thread through this, so
    it has to behave like a device rather than answer every attribute
    truthily the way a ``MagicMock`` would — and it caps its output so
    the reader cannot lap the writer and manufacture a drop that says
    nothing about real hardware.
    """

    def __init__(self, *, max_blocks=8, gain_db=None):
        self.index = 0
        self.is_open = False
        self.applied = None
        self.configured = []
        self.reads = 0
        self._max_blocks = max_blocks
        self._forced_gain = gain_db
        self.info = DeviceInfo(
            index=0,
            name="Generic RTL2832U OEM",
            manufacturer="RTLSDRBlog",
            product="Blog V4",
            serial="",
            tuner=TunerType.R828D,
        )

    def __enter__(self):
        self.is_open = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_open = False

    def supported_gains_db(self):
        return V4_GAIN_STEPS_DB

    def configure(self, config):
        self.configured.append(config)
        gain = self._forced_gain
        if gain is None:
            gain = 0.0 if config.uses_auto_gain else float(config.gain_db)
        self.applied = AppliedSettings(
            requested=config,
            center_hz=config.center_hz,
            sample_rate_hz=config.sample_rate_hz,
            gain_db=gain,
            manual_gain=not config.uses_auto_gain,
            ppm=config.ppm,
            agc_enabled=config.enable_agc,
        )
        return self.applied

    def read_raw(self, length):
        if self.reads >= self._max_blocks:
            raise DeviceError("fake device exhausted")
        self.reads += 1
        return bytes([self.reads % 256]) * length


def sdr_factory(device=None):
    """An SDR factory that records its calls and hands back one device."""
    device = device or FakeSdr()

    def build(config):
        build.calls.append(config)
        return device

    build.calls = []
    build.device = device
    return build


def run_sdr(argv, config_path, factory):
    return main(["--config", str(config_path), *argv], sdr_factory=factory)


class TestSdrParser:
    def test_sdr_without_a_subcommand_is_a_usage_error(self):
        with pytest.raises(SystemExit) as exit_info:
            main(["sdr"])

        assert exit_info.value.code == 2

    def test_capture_needs_somewhere_to_write(self):
        with pytest.raises(SystemExit):
            main(["sdr", "capture", "--station", "99.9", "--gain", "32.8"])

    def test_capture_needs_a_gain_decision(self):
        # No default gain, on purpose: a default is how a capture comes
        # back empty with nobody noticing.
        with pytest.raises(SystemExit):
            main(["sdr", "capture", "--station", "99.9", "--out", "x.iq"])

    def test_a_station_and_an_explicit_centre_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            main(
                [
                    "sdr",
                    "capture",
                    "--station",
                    "99.9",
                    "--center",
                    "99.65",
                    "--gain",
                    "32.8",
                    "--out",
                    "x.iq",
                ]
            )


class TestSdrInfo:
    def test_it_reports_the_device_and_its_gain_table(self, config_path, capsys):
        factory = sdr_factory()

        code = run_sdr(["sdr", "info"], config_path, factory)

        out = capsys.readouterr().out
        assert code == 0
        assert "Blog V4" in out
        assert "5 steps" in out
        assert "49.6" in out

    def test_it_closes_the_device_again(self, config_path):
        factory = sdr_factory()

        run_sdr(["sdr", "info"], config_path, factory)

        assert not factory.device.is_open


class TestSdrCapture:
    def test_it_tunes_below_the_station_by_default(self, config_path, tmp_path):
        # The off-centre discipline, made the default so it cannot be
        # forgotten: a peak at the centre of the passband is
        # indistinguishable from the receiver's own DC spike.
        factory = sdr_factory()

        run_sdr(
            [
                "sdr",
                "capture",
                "--station",
                "99.9",
                "--gain",
                "32.8",
                "--seconds",
                "0.001",
                "--rate",
                "250000",
                "--out",
                str(tmp_path / "cap.iq"),
            ],
            config_path,
            factory,
        )

        assert factory.device.configured[0].center_hz == pytest.approx(99_650_000.0)

    def test_the_offset_is_adjustable(self, config_path, tmp_path):
        factory = sdr_factory()

        run_sdr(
            [
                "sdr",
                "capture",
                "--station",
                "162.55",
                "--offset",
                "150",
                "--gain",
                "49.6",
                "--seconds",
                "0.001",
                "--rate",
                "250000",
                "--out",
                str(tmp_path / "cap.iq"),
            ],
            config_path,
            factory,
        )

        assert factory.device.configured[0].center_hz == pytest.approx(162_400_000.0)

    def test_an_explicit_centre_records_no_station(self, config_path, tmp_path):
        factory = sdr_factory()

        run_sdr(
            [
                "sdr",
                "capture",
                "--center",
                "99.65",
                "--gain",
                "32.8",
                "--seconds",
                "0.001",
                "--rate",
                "250000",
                "--out",
                str(tmp_path / "cap.iq"),
            ],
            config_path,
            factory,
        )
        meta = json.loads((tmp_path / "cap.json").read_text(encoding="utf-8"))

        assert factory.device.configured[0].center_hz == pytest.approx(99_650_000.0)
        assert "station_hz" not in meta

    def test_it_writes_the_capture_and_its_sidecar(self, config_path, tmp_path, capsys):
        factory = sdr_factory()

        code = run_sdr(
            [
                "sdr",
                "capture",
                "--station",
                "99.9",
                "--gain",
                "32.8",
                "--seconds",
                "0.001",
                "--rate",
                "250000",
                "--out",
                str(tmp_path / "cap.iq"),
            ],
            config_path,
            factory,
        )

        assert code == 0
        assert (tmp_path / "cap.iq").is_file()
        assert (tmp_path / "cap.json").is_file()
        assert "Sidecar" in capsys.readouterr().out

    def test_a_zero_gain_report_is_called_out(self, config_path, tmp_path, capsys):
        # The bring-up failure in executable form: auto gain reported
        # 0.0 dB and captured a flat noise floor, with nothing raising.
        factory = sdr_factory(FakeSdr(gain_db=0.0))

        run_sdr(
            [
                "sdr",
                "capture",
                "--station",
                "99.9",
                "--auto-gain",
                "--seconds",
                "0.001",
                "--rate",
                "250000",
                "--out",
                str(tmp_path / "cap.iq"),
            ],
            config_path,
            factory,
        )

        assert "0.0 dB" in capsys.readouterr().out

    def test_the_station_config_ppm_is_applied(self, config_path, tmp_path):
        factory = sdr_factory()

        run_sdr(
            [
                "sdr",
                "capture",
                "--center",
                "99.65",
                "--gain",
                "32.8",
                "--seconds",
                "0.001",
                "--rate",
                "250000",
                "--out",
                str(tmp_path / "cap.iq"),
            ],
            config_path,
            factory,
        )

        assert factory.device.configured[0].ppm == 0

    def test_a_bad_sample_rate_is_reported_not_traced(self, config_path, tmp_path, capsys):
        # 500 kHz falls in the RTL2832U's genuine gap between its two
        # rate windows. The operator gets a sentence, not a traceback.
        factory = sdr_factory()

        code = run_sdr(
            [
                "sdr",
                "capture",
                "--center",
                "99.65",
                "--gain",
                "32.8",
                "--rate",
                "500000",
                "--out",
                str(tmp_path / "cap.iq"),
            ],
            config_path,
            factory,
        )

        assert code == 1
        assert "sample_rate_hz" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# receive
# ---------------------------------------------------------------------------


class TestReceiveParser:
    """Parser-level checks only. The session itself is tested in test_receive.py."""

    def test_a_tle_is_required(self):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(["receive", "--downlink", "145.95", "--gain", "40"])
        assert exit_info.value.code == 2

    def test_a_downlink_is_required(self):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(["receive", "--tle", "x", "--gain", "40"])
        assert exit_info.value.code == 2

    def test_a_gain_choice_is_required(self):
        # Deliberately has no default, same as 'sdr capture': a default
        # gain is how a pass comes back silent with nobody noticing.
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(["receive", "--tle", "x", "--downlink", "145.95"])
        assert exit_info.value.code == 2

    def test_gain_and_auto_gain_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(
                ["receive", "--tle", "x", "--downlink", "145.95", "--gain", "40", "--auto-gain"]
            )
        assert exit_info.value.code == 2

    def test_send_defaults_to_off(self):
        # The safety-relevant default again, and the one that makes a
        # receive possible with no rotor connected at all.
        args = build_parser().parse_args(
            ["receive", "--tle", "x", "--downlink", "145.95", "--gain", "40"]
        )
        assert args.send is False

    def test_the_window_and_the_squelch_both_default_to_off(self):
        # The squelch default is a Chunk G decision with a reason: a mute
        # set slightly too tight makes a working receiver sound exactly
        # like a broken one, which is the last thing wanted while
        # pointing at a weak downlink for the first time.
        args = build_parser().parse_args(
            ["receive", "--tle", "x", "--downlink", "145.95", "--gain", "40"]
        )
        assert args.window is False
        assert args.squelch is False

    def test_the_tuning_offset_defaults_to_the_project_convention(self):
        args = build_parser().parse_args(
            ["receive", "--tle", "x", "--downlink", "145.95", "--gain", "40"]
        )
        assert args.offset == DEFAULT_TUNING_OFFSET_KHZ
