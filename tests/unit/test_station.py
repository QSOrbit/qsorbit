"""Unit tests for station configuration loading.

Every test writes a real TOML file to a temporary directory and loads
it, rather than constructing dicts — the parsing and the validation are
both part of what's being tested, and a config file is the one thing in
this project a user edits by hand.
"""

import sys
import textwrap
from pathlib import Path

import pytest

from qsorbit.core.rotor import AzimuthWrap
from qsorbit.core.station import (
    ConfigError,
    SerialSettings,
    StationConfig,
    candidate_config_paths,
    find_config_path,
    load_station_config,
    user_config_dir,
)

VALID_CONFIG = """
    [observer]
    latitude = 40.5
    longitude = -83.25
    altitude_m = 250.0

    [rotor]
    port = "COM5"
    baudrate = 19200
    timeout_s = 1.0

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


def write_config(tmp_path, text=VALID_CONFIG, name="qsorbit.toml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def config_without(line_starting: str) -> str:
    """VALID_CONFIG with one key removed, for missing-key tests."""
    kept = [
        line for line in VALID_CONFIG.splitlines() if not line.strip().startswith(line_starting)
    ]
    return "\n".join(kept)


class TestLoadValidConfig:
    def test_observer(self, tmp_path):
        config = load_station_config(write_config(tmp_path))

        assert config.observer.latitude == 40.5
        assert config.observer.longitude == -83.25
        assert config.observer.altitude_m == 250.0

    def test_serial(self, tmp_path):
        config = load_station_config(write_config(tmp_path))

        assert config.serial == SerialSettings(port="COM5", baudrate=19200, timeout_s=1.0)

    def test_capabilities(self, tmp_path):
        caps = load_station_config(write_config(tmp_path)).capabilities

        assert caps.azimuth_max_deg == 360.0
        assert caps.elevation_max_deg == 180.0
        assert caps.azimuth_wrap is AzimuthWrap.EXTRA_ROTATION
        assert caps.acceptance_window_deg == 2.5
        assert caps.rs485_turnaround_s == 0.15
        assert caps.firmware_version == "SatNOGS-v2.2.1"

    def test_records_where_it_came_from(self, tmp_path):
        # With three possible locations, "which config am I actually
        # using?" is a question that will come up at a bench.
        path = write_config(tmp_path)

        assert load_station_config(path).source_path == path

    def test_integers_are_accepted_for_float_fields(self, tmp_path):
        # TOML distinguishes 2 from 2.0; a config file written by hand
        # should not have to care.
        text = VALID_CONFIG.replace("acceptance_window_deg = 2.5", "acceptance_window_deg = 3")

        assert (
            load_station_config(write_config(tmp_path, text)).capabilities.acceptance_window_deg
            == 3.0
        )

    def test_is_a_station_config(self, tmp_path):
        assert isinstance(load_station_config(write_config(tmp_path)), StationConfig)


class TestShippedExample:
    def test_the_example_config_actually_loads(self):
        # config.example.toml is the first thing a new user copies. An
        # example that fails its own validation is worse than no example
        # at all, and nothing else in the suite would notice if a key
        # were renamed on one side only.
        example = Path(__file__).resolve().parents[2] / "config.example.toml"

        config = load_station_config(example)

        assert config.capabilities.azimuth_wrap is AzimuthWrap.EXTRA_ROTATION
        assert config.serial.baudrate == 19200


class TestOptionalKeys:
    def test_altitude_defaults_to_sea_level(self, tmp_path):
        config = load_station_config(write_config(tmp_path, config_without("altitude_m")))

        assert config.observer.altitude_m == 0.0

    def test_baudrate_defaults_to_the_firmware_value(self, tmp_path):
        # The stock firmware hard-codes 19200.
        config = load_station_config(write_config(tmp_path, config_without("baudrate")))

        assert config.serial.baudrate == 19200

    def test_timeout_defaults(self, tmp_path):
        config = load_station_config(write_config(tmp_path, config_without("timeout_s")))

        assert config.serial.timeout_s == 1.0

    def test_firmware_version_may_be_omitted(self, tmp_path):
        config = load_station_config(write_config(tmp_path, config_without("firmware_version")))

        assert config.capabilities.firmware_version is None


class TestMissingKeys:
    @pytest.mark.parametrize(
        "key",
        [
            "latitude",
            "longitude",
            "port",
            "azimuth_min_deg",
            "azimuth_max_deg",
            "elevation_min_deg",
            "elevation_max_deg",
            "azimuth_wrap",
            "acceptance_window_deg",
            "rs485_turnaround_s",
        ],
    )
    def test_required_key_missing(self, tmp_path, key):
        # Nothing safety-relevant is defaulted. A travel limit that
        # quietly fell back to a built-in value would be a limit the
        # operator believes is in force and isn't.
        with pytest.raises(ConfigError, match=key):
            load_station_config(write_config(tmp_path, config_without(key)))

    def test_missing_observer_section(self, tmp_path):
        text = VALID_CONFIG.split("[rotor]")[1]

        with pytest.raises(ConfigError, match=r"\[observer\]"):
            load_station_config(write_config(tmp_path, "[rotor]" + text))

    def test_missing_capabilities_section(self, tmp_path):
        text = VALID_CONFIG.split("[rotor.capabilities]")[0]

        with pytest.raises(ConfigError, match=r"\[rotor.capabilities\]"):
            load_station_config(write_config(tmp_path, text))


class TestRejectsBadValues:
    def test_unknown_key_is_an_error(self, tmp_path):
        text = VALID_CONFIG.replace("azimuth_wrap =", "azimuth_wrapping =")

        with pytest.raises(ConfigError, match="Unknown key"):
            load_station_config(write_config(tmp_path, text))

    def test_unknown_key_message_lists_the_valid_ones(self, tmp_path):
        text = VALID_CONFIG + "\n    nonsense = 1\n"

        with pytest.raises(ConfigError, match="Valid keys"):
            load_station_config(write_config(tmp_path, text))

    def test_bad_azimuth_wrap_value(self, tmp_path):
        text = VALID_CONFIG.replace('"extra_rotation"', '"sideways"')

        with pytest.raises(ConfigError, match="azimuth_wrap"):
            load_station_config(write_config(tmp_path, text))

    def test_bad_azimuth_wrap_message_explains_the_stakes(self, tmp_path):
        text = VALID_CONFIG.replace('"extra_rotation"', '"sideways"')

        with pytest.raises(ConfigError, match="full extra rotation"):
            load_station_config(write_config(tmp_path, text))

    def test_string_where_a_number_belongs(self, tmp_path):
        text = VALID_CONFIG.replace("latitude = 40.5", 'latitude = "forty"')

        with pytest.raises(ConfigError, match="must be a number"):
            load_station_config(write_config(tmp_path, text))

    def test_boolean_is_not_a_number(self, tmp_path):
        # bool is a subclass of int in Python; `true` is still not a
        # latitude.
        text = VALID_CONFIG.replace("latitude = 40.5", "latitude = true")

        with pytest.raises(ConfigError, match="must be a number"):
            load_station_config(write_config(tmp_path, text))

    def test_number_where_a_string_belongs(self, tmp_path):
        text = VALID_CONFIG.replace('port = "COM5"', "port = 5")

        with pytest.raises(ConfigError, match="must be a string"):
            load_station_config(write_config(tmp_path, text))

    def test_value_rejected_by_the_type_it_configures(self, tmp_path):
        # A wrapping rotator declaring travel past 360 is the config
        # typo that would cause a full extra rotation against the cable.
        text = VALID_CONFIG.replace("azimuth_max_deg = 360.0", "azimuth_max_deg = 450.0")

        with pytest.raises(ConfigError, match="EXTRA_ROTATION"):
            load_station_config(write_config(tmp_path, text))

    def test_out_of_range_latitude(self, tmp_path):
        text = VALID_CONFIG.replace("latitude = 40.5", "latitude = 200.0")

        with pytest.raises(ConfigError, match="latitude"):
            load_station_config(write_config(tmp_path, text))

    def test_error_names_the_file(self, tmp_path):
        text = VALID_CONFIG.replace("latitude = 40.5", "latitude = 200.0")
        path = write_config(tmp_path, text)

        with pytest.raises(ConfigError, match=str(path.name)):
            load_station_config(path)

    def test_malformed_toml(self, tmp_path):
        with pytest.raises(ConfigError, match="Could not parse"):
            load_station_config(write_config(tmp_path, "[observer\nlatitude = "))


class TestConfigDiscovery:
    def test_explicit_path_that_does_not_exist(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_station_config(tmp_path / "nope.toml")

    def test_working_directory_takes_precedence(self, tmp_path, monkeypatch):
        work = tmp_path / "work"
        work.mkdir()
        write_config(work)
        user = tmp_path / "user"
        (user / "qsorbit").mkdir(parents=True)
        (user / "qsorbit" / "config.toml").write_text("bogus", encoding="utf-8")
        monkeypatch.chdir(work)
        monkeypatch.setenv("APPDATA", str(user))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(user))

        assert find_config_path() == work / "qsorbit.toml"

    def test_falls_back_to_the_user_config_directory(self, tmp_path, monkeypatch):
        work = tmp_path / "work"
        work.mkdir()
        user = tmp_path / "user"
        (user / "qsorbit").mkdir(parents=True)
        write_config(user / "qsorbit", name="config.toml")
        monkeypatch.chdir(work)
        monkeypatch.setenv("APPDATA", str(user))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(user))

        assert find_config_path() == user / "qsorbit" / "config.toml"

    def test_no_config_anywhere(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("APPDATA", str(tmp_path / "empty"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))

        assert find_config_path() is None

    def test_missing_config_message_lists_where_it_looked(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("APPDATA", str(tmp_path / "empty"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))

        with pytest.raises(ConfigError, match="config.example.toml"):
            load_station_config()

    def test_two_candidate_paths(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        assert len(candidate_config_paths()) == 2

    def test_linux_config_dir_honours_xdg(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

        assert user_config_dir() == tmp_path / "qsorbit"

    def test_windows_config_dir_uses_appdata(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", str(tmp_path))

        assert user_config_dir() == tmp_path / "qsorbit"

    def test_macos_config_dir(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")

        assert user_config_dir().parts[-3:] == ("Library", "Application Support", "qsorbit")


class TestSerialSettings:
    def test_rejects_empty_port(self):
        with pytest.raises(ValueError, match="must not be empty"):
            SerialSettings(port="")

    def test_rejects_zero_baudrate(self):
        with pytest.raises(ValueError, match="baudrate"):
            SerialSettings(port="COM5", baudrate=0)

    def test_rejects_zero_timeout(self):
        with pytest.raises(ValueError, match="timeout_s"):
            SerialSettings(port="COM5", timeout_s=0.0)

    def test_defaults(self):
        settings = SerialSettings(port="COM5")

        assert settings.baudrate == 19200
        assert settings.timeout_s == 1.0
