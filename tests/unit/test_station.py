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

from qsorbit.core.horizon import HorizonMask, HorizonPoint
from qsorbit.core.rotor import AzimuthWrap
from qsorbit.core.station import (
    AlignmentSettings,
    ConfigError,
    PlanningSettings,
    SdrBranch,
    SdrSettings,
    SerialSettings,
    StationConfig,
    TrackingSettings,
    candidate_config_paths,
    find_config_path,
    load_station_config,
    user_config_dir,
)
from qsorbit.core.tracking_profile import CadenceError, TrackingProfile

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


SDR_SECTION = """
    [sdr]
    driver_dir = "C:\\\\Users\\\\phil\\\\dev\\\\rtlsdr-blog\\\\x64"
    device_index = 1
    ppm = -3
"""

ALIGNMENT_SECTION = """
    [rotor.alignment]
    azimuth_deg = 4.2
    elevation_deg = -1.1
"""


class TestSdrSection:
    """The [sdr] table, which is optional in a way the others are not."""

    def test_absent_section_gives_working_defaults(self, tmp_path):
        # Every config file written before Phase 2 lacks this section,
        # and an SDR is not required to point an antenna. Refusing to
        # load one would break every existing station for no gain.
        config = load_station_config(write_config(tmp_path))

        assert config.sdr == SdrSettings()
        assert config.sdr.driver_dir is None
        assert config.sdr.device_index == 0
        assert config.sdr.ppm == 0

    def test_reads_every_key(self, tmp_path):
        config = load_station_config(write_config(tmp_path, VALID_CONFIG + SDR_SECTION))

        assert config.sdr.driver_dir == r"C:\Users\phil\dev\rtlsdr-blog\x64"
        assert config.sdr.device_index == 1
        assert config.sdr.ppm == -3

    def test_an_empty_section_is_the_same_as_no_section(self, tmp_path):
        config = load_station_config(write_config(tmp_path, VALID_CONFIG + "\n[sdr]\n"))

        assert config.sdr == SdrSettings()

    def test_each_key_may_be_omitted_on_its_own(self, tmp_path):
        config = load_station_config(write_config(tmp_path, VALID_CONFIG + "\n[sdr]\nppm = 7\n"))

        assert config.sdr.ppm == 7
        assert config.sdr.driver_dir is None
        assert config.sdr.device_index == 0

    def test_unknown_key_is_an_error(self, tmp_path):
        # Same strictness as everywhere else: a misspelled driver_dir
        # that silently fell back to "search the system" would produce
        # the exact silent-mistune failure the SDR module exists to
        # prevent.
        path = write_config(tmp_path, VALID_CONFIG + '\n[sdr]\ndriver_directory = "x"\n')

        with pytest.raises(ConfigError, match="driver_directory"):
            load_station_config(path)

    def test_unknown_key_message_lists_the_valid_ones(self, tmp_path):
        path = write_config(tmp_path, VALID_CONFIG + "\n[sdr]\nnope = 1\n")

        with pytest.raises(ConfigError, match="device_index, driver_dir, ppm"):
            load_station_config(path)

    def test_a_section_that_is_not_a_table_is_an_error(self, tmp_path):
        # Written before the first table header, or TOML would read it
        # as a key inside whichever section came last.
        path = write_config(tmp_path, '\n    sdr = "somewhere"' + VALID_CONFIG)

        with pytest.raises(ConfigError, match=r"\[sdr\] in .* must be a table"):
            load_station_config(path)

    def test_a_non_integer_device_index_is_an_error(self, tmp_path):
        path = write_config(tmp_path, VALID_CONFIG + "\n[sdr]\ndevice_index = 1.5\n")

        with pytest.raises(ConfigError, match="device_index"):
            load_station_config(path)

    def test_a_negative_device_index_is_an_error(self, tmp_path):
        path = write_config(tmp_path, VALID_CONFIG + "\n[sdr]\ndevice_index = -1\n")

        with pytest.raises(ConfigError, match="device_index"):
            load_station_config(path)

    def test_an_absurd_ppm_is_an_error(self, tmp_path):
        path = write_config(tmp_path, VALID_CONFIG + "\n[sdr]\nppm = 50000\n")

        with pytest.raises(ConfigError, match="ppm"):
            load_station_config(path)

    def test_an_empty_driver_dir_is_an_error(self, tmp_path):
        # Distinct from omitting it. An empty string looks deliberate
        # and means nothing.
        path = write_config(tmp_path, VALID_CONFIG + '\n[sdr]\ndriver_dir = ""\n')

        with pytest.raises(ConfigError, match="driver_dir"):
            load_station_config(path)

    def test_the_error_names_the_file(self, tmp_path):
        path = write_config(tmp_path, VALID_CONFIG + "\n[sdr]\nppm = 50000\n")

        with pytest.raises(ConfigError, match=str(path.name)):
            load_station_config(path)


class TestSdrSettings:
    def test_defaults(self):
        settings = SdrSettings()

        assert settings.driver_dir is None
        assert settings.device_index == 0
        assert settings.ppm == 0

    def test_rejects_a_negative_device_index(self):
        with pytest.raises(ValueError, match="device_index"):
            SdrSettings(device_index=-1)

    def test_rejects_a_blank_driver_dir(self):
        with pytest.raises(ValueError, match="driver_dir"):
            SdrSettings(driver_dir="   ")

    def test_rejects_an_absurd_ppm(self):
        with pytest.raises(ValueError, match="ppm"):
            SdrSettings(ppm=5000)

    def test_accepts_a_realistic_ppm(self):
        assert SdrSettings(ppm=-12).ppm == -12

    def test_defaults_to_no_branches(self):
        # A single-dongle station, which is every station that existed
        # before branches did.
        assert SdrSettings().branches == ()

    def test_rejects_two_branches_on_one_dongle(self):
        with pytest.raises(ValueError, match="serial"):
            SdrSettings(
                branches=(
                    SdrBranch(serial="LEFT", label="A"),
                    SdrBranch(serial="LEFT", label="B"),
                )
            )

    def test_rejects_two_branches_with_one_label(self):
        with pytest.raises(ValueError, match="label"):
            SdrSettings(
                branches=(
                    SdrBranch(serial="LEFT", label="A"),
                    SdrBranch(serial="RIGHT", label="A"),
                )
            )


class TestSdrBranch:
    def test_ppm_is_optional(self):
        assert SdrBranch(serial="LEFT", label="A - Arrow V").ppm is None

    def test_rejects_a_blank_serial(self):
        # A blank serial matches every dongle whose EEPROM was never
        # flashed, which is most of them.
        with pytest.raises(ValueError, match="serial"):
            SdrBranch(serial="  ", label="A")

    def test_rejects_a_blank_label(self):
        with pytest.raises(ValueError, match="label"):
            SdrBranch(serial="LEFT", label="")

    def test_rejects_an_absurd_ppm(self):
        with pytest.raises(ValueError, match="ppm"):
            SdrBranch(serial="LEFT", label="A", ppm=5000)


class TestBranchPpm:
    """Which correction a branch actually receives."""

    def test_a_branch_without_a_ppm_falls_back_to_the_station(self):
        settings = SdrSettings(ppm=-3, branches=(SdrBranch(serial="LEFT", label="A"),))

        assert settings.ppm_for(settings.branches[0]) == -3

    def test_a_branch_with_a_ppm_overrides_the_station(self):
        settings = SdrSettings(ppm=-3, branches=(SdrBranch(serial="LEFT", label="A", ppm=7),))

        assert settings.ppm_for(settings.branches[0]) == 7

    def test_a_branch_ppm_of_zero_overrides_rather_than_falls_back(self):
        # The trap in every "use the default when unset" scheme: zero is
        # a measurement, not an absence, and a branch measured at 0 next
        # to a station default of -3 must get 0.
        settings = SdrSettings(ppm=-3, branches=(SdrBranch(serial="LEFT", label="A", ppm=0),))

        assert settings.ppm_for(settings.branches[0]) == 0

    def test_the_two_branches_can_differ(self):
        settings = SdrSettings(
            ppm=-3,
            branches=(
                SdrBranch(serial="LEFT", label="A", ppm=1),
                SdrBranch(serial="RIGHT", label="B"),
            ),
        )

        assert [settings.ppm_for(branch) for branch in settings.branches] == [1, -3]


BRANCH_SECTION = """
    [sdr]
    ppm = -3

    [[sdr.branch]]
    serial = "LEFT"
    label = "A - Arrow V"
    ppm = 1

    [[sdr.branch]]
    serial = "RIGHT"
    label = "B - Arrow H"
"""


class TestSdrBranchSection:
    """The [[sdr.branch]] array of tables, optional like [sdr] itself."""

    def test_no_branches_is_the_single_device_station(self, tmp_path):
        config = load_station_config(write_config(tmp_path, VALID_CONFIG + SDR_SECTION))

        assert config.sdr.branches == ()

    def test_reads_every_branch_in_file_order(self, tmp_path):
        # Order is meaningful, unlike [rotor.profiles]: the first branch
        # is the one a single-device command opens.
        config = load_station_config(write_config(tmp_path, VALID_CONFIG + BRANCH_SECTION))

        assert [branch.serial for branch in config.sdr.branches] == ["LEFT", "RIGHT"]
        assert [branch.label for branch in config.sdr.branches] == ["A - Arrow V", "B - Arrow H"]

    def test_reads_the_optional_per_branch_ppm(self, tmp_path):
        config = load_station_config(write_config(tmp_path, VALID_CONFIG + BRANCH_SECTION))

        assert config.sdr.branches[0].ppm == 1
        assert config.sdr.branches[1].ppm is None
        assert config.sdr.ppm_for(config.sdr.branches[1]) == -3

    def test_one_branch_is_allowed(self, tmp_path):
        path = write_config(
            tmp_path,
            VALID_CONFIG + '\n[[sdr.branch]]\nserial = "LEFT"\nlabel = "A"\n',
        )

        assert load_station_config(path).sdr.branches[0].serial == "LEFT"

    def test_a_branch_needs_a_serial(self, tmp_path):
        path = write_config(tmp_path, VALID_CONFIG + '\n[[sdr.branch]]\nlabel = "A"\n')

        with pytest.raises(ConfigError, match="serial"):
            load_station_config(path)

    def test_a_branch_needs_a_label(self, tmp_path):
        path = write_config(tmp_path, VALID_CONFIG + '\n[[sdr.branch]]\nserial = "LEFT"\n')

        with pytest.raises(ConfigError, match="label"):
            load_station_config(path)

    def test_an_unknown_branch_key_is_an_error(self, tmp_path):
        path = write_config(
            tmp_path,
            VALID_CONFIG + '\n[[sdr.branch]]\nserial = "L"\nlabel = "A"\nindex = 0\n',
        )

        with pytest.raises(ConfigError, match="index"):
            load_station_config(path)

    def test_the_error_names_which_branch(self, tmp_path):
        # With two of them, "invalid ppm" without an index sends you
        # reading both.
        path = write_config(
            tmp_path,
            VALID_CONFIG
            + '\n[[sdr.branch]]\nserial = "L"\nlabel = "A"\n'
            + '\n[[sdr.branch]]\nserial = "R"\nlabel = "B"\nppm = 50000\n',
        )

        with pytest.raises(ConfigError, match=r"sdr\.branch\[1\]"):
            load_station_config(path)

    def test_branch_is_not_a_table_keyed_section(self, tmp_path):
        # [sdr.branch] rather than [[sdr.branch]] -- one bracket short,
        # and a plausible typo. TOML reads it as a table, not a list.
        path = write_config(tmp_path, VALID_CONFIG + '\n[sdr.branch]\nserial = "L"\nlabel = "A"\n')

        with pytest.raises(ConfigError, match="array of tables"):
            load_station_config(path)

    def test_two_branches_on_one_dongle_is_an_error(self, tmp_path):
        path = write_config(
            tmp_path,
            VALID_CONFIG
            + '\n[[sdr.branch]]\nserial = "L"\nlabel = "A"\n'
            + '\n[[sdr.branch]]\nserial = "L"\nlabel = "B"\n',
        )

        with pytest.raises(ConfigError, match="serial"):
            load_station_config(path)

    def test_branches_and_device_index_together_are_refused(self, tmp_path):
        # Not resolved in some precedence order: a device_index that
        # branches override is a setting the operator believes is in
        # force and isn't, which this file refuses everywhere else.
        path = write_config(
            tmp_path,
            VALID_CONFIG
            + "\n[sdr]\ndevice_index = 1\n"
            + '\n[[sdr.branch]]\nserial = "L"\nlabel = "A"\n',
        )

        with pytest.raises(ConfigError, match="device_index"):
            load_station_config(path)

    def test_device_index_alone_is_still_fine(self, tmp_path):
        # The guard above must not fire on the single-device config
        # every existing station has.
        config = load_station_config(
            write_config(tmp_path, VALID_CONFIG + "\n[sdr]\ndevice_index = 1\n")
        )

        assert config.sdr.device_index == 1


PLANNING_SECTION = """
    [planning]
    tle_dir = "C:\\\\Users\\\\phil\\\\dev\\\\qsorbit-fixtures\\\\tle"
"""


class TestPlanningSection:
    """The [planning] table, optional like [sdr] and for the same reason."""

    def test_absent_section_gives_working_defaults(self, tmp_path):
        # Every config file written before Chunk D lacks this section,
        # and pointing an antenna doesn't require knowing where any
        # TLEs live. The Plan tab reads an unset tle_dir as "not
        # configured yet", not an error.
        config = load_station_config(write_config(tmp_path))

        assert config.planning == PlanningSettings()
        assert config.planning.tle_dir is None

    def test_reads_tle_dir(self, tmp_path):
        config = load_station_config(write_config(tmp_path, VALID_CONFIG + PLANNING_SECTION))

        assert config.planning.tle_dir == r"C:\Users\phil\dev\qsorbit-fixtures\tle"

    def test_an_empty_section_is_the_same_as_no_section(self, tmp_path):
        config = load_station_config(write_config(tmp_path, VALID_CONFIG + "\n[planning]\n"))

        assert config.planning == PlanningSettings()

    def test_unknown_key_is_an_error(self, tmp_path):
        path = write_config(tmp_path, VALID_CONFIG + '\n[planning]\ntle_directory = "x"\n')

        with pytest.raises(ConfigError, match="tle_directory"):
            load_station_config(path)

    def test_a_section_that_is_not_a_table_is_an_error(self, tmp_path):
        path = write_config(tmp_path, '\n    planning = "somewhere"' + VALID_CONFIG)

        with pytest.raises(ConfigError, match=r"\[planning\] in .* must be a table"):
            load_station_config(path)

    def test_an_empty_tle_dir_is_an_error(self, tmp_path):
        # Distinct from omitting it, same reasoning as [sdr]'s
        # driver_dir: an empty string looks deliberate and means
        # nothing.
        path = write_config(tmp_path, VALID_CONFIG + '\n[planning]\ntle_dir = ""\n')

        with pytest.raises(ConfigError, match="tle_dir"):
            load_station_config(path)

    def test_the_error_names_the_file(self, tmp_path):
        path = write_config(tmp_path, VALID_CONFIG + '\n[planning]\ntle_directory = "x"\n')

        with pytest.raises(ConfigError, match=str(path.name)):
            load_station_config(path)


class TestPlanningSettings:
    def test_defaults(self):
        assert PlanningSettings().tle_dir is None

    def test_rejects_a_blank_tle_dir(self):
        with pytest.raises(ValueError, match="tle_dir"):
            PlanningSettings(tle_dir="   ")

    def test_accepts_a_real_path(self):
        assert PlanningSettings(tle_dir="/home/phil/tle").tle_dir == "/home/phil/tle"


class TestAlignmentSection:
    """The [rotor.alignment] table, optional the same way [sdr] is."""

    def test_absent_section_gives_identity_defaults(self, tmp_path):
        # Every config file written before Chunk I lacks this section,
        # and "uncalibrated" is this feature's honest identity value -
        # refusing to load one would break every existing station.
        config = load_station_config(write_config(tmp_path))

        assert config.alignment == AlignmentSettings()
        assert config.alignment.azimuth_deg == 0.0
        assert config.alignment.elevation_deg == 0.0

    def test_reads_both_keys(self, tmp_path):
        config = load_station_config(write_config(tmp_path, VALID_CONFIG + ALIGNMENT_SECTION))

        assert config.alignment.azimuth_deg == 4.2
        assert config.alignment.elevation_deg == -1.1

    def test_an_empty_section_is_the_same_as_no_section(self, tmp_path):
        config = load_station_config(write_config(tmp_path, VALID_CONFIG + "\n[rotor.alignment]\n"))

        assert config.alignment == AlignmentSettings()

    def test_each_key_may_be_omitted_on_its_own(self, tmp_path):
        config = load_station_config(
            write_config(tmp_path, VALID_CONFIG + "\n[rotor.alignment]\nazimuth_deg = 7.0\n")
        )

        assert config.alignment.azimuth_deg == 7.0
        assert config.alignment.elevation_deg == 0.0

    def test_negative_values_are_fine(self, tmp_path):
        # Which side of true north the mast sits on is arbitrary - both
        # signs are ordinary, not a sign of a mistake.
        config = load_station_config(
            write_config(
                tmp_path,
                VALID_CONFIG + "\n[rotor.alignment]\nazimuth_deg = -12.5\nelevation_deg = -0.3\n",
            )
        )

        assert config.alignment.azimuth_deg == -12.5
        assert config.alignment.elevation_deg == -0.3

    def test_unknown_key_is_an_error(self, tmp_path):
        path = write_config(tmp_path, VALID_CONFIG + "\n[rotor.alignment]\nazimuth = 1.0\n")

        with pytest.raises(ConfigError, match="azimuth"):
            load_station_config(path)

    def test_unknown_key_names_the_nested_section(self, tmp_path):
        # [rotor.alignment], not just [alignment] - the section as it
        # actually appears in the file.
        path = write_config(tmp_path, VALID_CONFIG + "\n[rotor.alignment]\nnope = 1\n")

        with pytest.raises(ConfigError, match=r"\[rotor\.alignment\]"):
            load_station_config(path)

    def test_a_section_that_is_not_a_table_is_an_error(self, tmp_path):
        # alignment as a plain key inside [rotor], not a [rotor.alignment]
        # sub-table - a typo that looks like an omission unless caught.
        text = VALID_CONFIG.replace(
            '    port = "COM5"',
            '    port = "COM5"\n    alignment = "somewhere"',
        )
        path = write_config(tmp_path, text)

        with pytest.raises(ConfigError, match=r"\[rotor\.alignment\] in .* must be a table"):
            load_station_config(path)


class TestAlignmentSettings:
    def test_defaults(self):
        settings = AlignmentSettings()

        assert settings.azimuth_deg == 0.0
        assert settings.elevation_deg == 0.0

    def test_accepts_realistic_offsets(self):
        settings = AlignmentSettings(azimuth_deg=15.0, elevation_deg=-2.0)

        assert settings.azimuth_deg == 15.0
        assert settings.elevation_deg == -2.0


class TestHorizonSection:
    """The top-level [[horizon]] array of tables, optional like [sdr]."""

    def test_absent_section_gives_an_empty_mask(self, tmp_path):
        # Every config file written before Chunk B lacks this section,
        # and an empty mask (no obstruction anywhere) is the honest
        # identity value - refusing to load one would break every
        # existing station, the same reasoning as [rotor.alignment].
        config = load_station_config(write_config(tmp_path))

        assert config.horizon == HorizonMask()
        assert config.horizon.min_elevation_at(0.0) == 0.0

    def test_reads_points_in_order(self, tmp_path):
        text = (
            VALID_CONFIG
            + "\n[[horizon]]\nazimuth_deg = 105.0\nmin_elevation_deg = 0.0\n"
            + "\n[[horizon]]\nazimuth_deg = 111.0\nmin_elevation_deg = 18.0\n"
            + "\n[[horizon]]\nazimuth_deg = 117.0\nmin_elevation_deg = 0.0\n"
        )
        config = load_station_config(write_config(tmp_path, text))

        assert config.horizon == HorizonMask(
            points=(
                HorizonPoint(105.0, 0.0),
                HorizonPoint(111.0, 18.0),
                HorizonPoint(117.0, 0.0),
            )
        )
        assert config.horizon.min_elevation_at(111.0) == 18.0

    def test_no_horizon_entries_at_all_is_the_same_as_absent(self, tmp_path):
        config = load_station_config(write_config(tmp_path, VALID_CONFIG))

        assert config.horizon == HorizonMask()

    def test_unknown_key_is_an_error(self, tmp_path):
        text = VALID_CONFIG + "\n[[horizon]]\nazimuth_deg = 1.0\nmin_elevation = 2.0\n"
        path = write_config(tmp_path, text)

        with pytest.raises(ConfigError, match="min_elevation"):
            load_station_config(path)

    def test_unknown_key_names_the_indexed_section(self, tmp_path):
        text = VALID_CONFIG + "\n[[horizon]]\nazimuth_deg = 1.0\nnope = 2.0\n"
        path = write_config(tmp_path, text)

        with pytest.raises(ConfigError, match=r"horizon\[0\]"):
            load_station_config(path)

    def test_missing_key_is_an_error(self, tmp_path):
        text = VALID_CONFIG + "\n[[horizon]]\nazimuth_deg = 1.0\n"
        path = write_config(tmp_path, text)

        with pytest.raises(ConfigError, match="min_elevation_deg"):
            load_station_config(path)

    def test_out_of_range_value_is_an_error(self, tmp_path):
        # HorizonPoint's own validation, surfaced as a ConfigError with
        # the file name attached - same pattern as every other value
        # object this loader builds.
        text = VALID_CONFIG + "\n[[horizon]]\nazimuth_deg = 400.0\nmin_elevation_deg = 0.0\n"
        path = write_config(tmp_path, text)

        with pytest.raises(ConfigError, match="azimuth_deg"):
            load_station_config(path)

    def test_unsorted_points_are_an_error(self, tmp_path):
        # HorizonMask itself rejects an unsorted list rather than
        # silently sorting it - see its own docstring for why.
        text = (
            VALID_CONFIG
            + "\n[[horizon]]\nazimuth_deg = 200.0\nmin_elevation_deg = 10.0\n"
            + "\n[[horizon]]\nazimuth_deg = 100.0\nmin_elevation_deg = 5.0\n"
        )
        path = write_config(tmp_path, text)

        with pytest.raises(ConfigError, match="sorted"):
            load_station_config(path)

    def test_horizon_entry_that_is_not_a_table_is_an_error(self, tmp_path):
        # A bare top-level key has to precede every [section] header in
        # TOML, so this - like test_horizon_that_is_not_an_array_is_an_error
        # below - prepends rather than appends.
        text = 'horizon = ["not", "tables"]\n' + VALID_CONFIG
        path = write_config(tmp_path, text)

        with pytest.raises(ConfigError, match=r"horizon\[0\] in .* must be a table"):
            load_station_config(path)

    def test_horizon_that_is_not_an_array_is_an_error(self, tmp_path):
        text = "[horizon]\nazimuth_deg = 1.0\n" + VALID_CONFIG
        path = write_config(tmp_path, text)

        with pytest.raises(ConfigError, match=r"'horizon' in .* must be an array"):
            load_station_config(path)


# ---------------------------------------------------------------------------
# Tracking profiles (Chunk H)
# ---------------------------------------------------------------------------


PROFILES = """
    [rotor.profiles.stock]
    deadband_deg = 2.5
    interval_s = 1.0

    [rotor.profiles.tracking]
    deadband_deg = 0.25
    interval_s = 0.5
    arrival_window_deg = 1.0
"""


class TestTrackingProfiles:
    def test_a_config_without_the_section_still_loads(self, tmp_path):
        # Every config file written before Chunk H lacks it. "Nobody has
        # declared a profile" is the honest identity state.
        config = load_station_config(write_config(tmp_path))
        assert config.tracking == TrackingSettings()
        assert config.tracking.profiles == ()

    def test_the_synthesized_profile_reproduces_the_old_behaviour(self, tmp_path):
        # Before this section existed, TrackingLoop defaulted its
        # deadband to the acceptance window at a one-second tick. A
        # station that upgrades and changes nothing must track
        # identically -- the coupling is now stated, not removed.
        config = load_station_config(write_config(tmp_path))
        profile = config.tracking_profile
        assert profile.name == "stock"
        assert profile.deadband_deg == 2.5
        assert profile.interval_s == 1.0

    def test_the_synthesized_profile_reports_the_real_step(self, tmp_path):
        # The behaviour is unchanged; what is new is that the 3.0 deg
        # step it has always commanded is now visible.
        config = load_station_config(write_config(tmp_path))
        assert config.tracking_profile.commanded_step_deg == pytest.approx(3.0)

    def test_declared_profiles_are_parsed_in_file_order(self, tmp_path):
        config = load_station_config(write_config(tmp_path, VALID_CONFIG + PROFILES))
        assert [entry.name for entry in config.tracking.profiles] == ["stock", "tracking"]

    def test_the_default_active_profile_is_stock(self, tmp_path):
        config = load_station_config(write_config(tmp_path, VALID_CONFIG + PROFILES))
        assert config.tracking.active == "stock"
        assert config.tracking_profile.deadband_deg == 2.5

    def test_the_profile_key_selects_one(self, tmp_path):
        text = VALID_CONFIG.replace('port = "COM5"', 'port = "COM5"\n    profile = "tracking"')
        config = load_station_config(write_config(tmp_path, text + PROFILES))
        profile = config.tracking_profile
        assert profile.name == "tracking"
        assert profile.deadband_deg == 0.25
        assert profile.interval_s == 0.5
        assert profile.commanded_step_deg == pytest.approx(0.5)

    def test_an_arrival_window_overrides_the_capability_record(self, tmp_path):
        # 2.5 deg is a stock-gains fact. The validated set settles at
        # 0.64-0.81, so a station running it against a 2.5 deg window
        # would report arrival long before it had arrived.
        text = VALID_CONFIG.replace('port = "COM5"', 'port = "COM5"\n    profile = "tracking"')
        config = load_station_config(write_config(tmp_path, text + PROFILES))
        window = config.tracking_profile.window_against(config.capabilities.acceptance_window_deg)
        assert window == 1.0

    def test_a_profile_without_an_arrival_window_uses_the_record(self, tmp_path):
        config = load_station_config(write_config(tmp_path, VALID_CONFIG + PROFILES))
        window = config.tracking_profile.window_against(config.capabilities.acceptance_window_deg)
        assert window == 2.5

    def test_a_misspelled_active_profile_is_refused(self, tmp_path):
        text = VALID_CONFIG.replace('port = "COM5"', 'port = "COM5"\n    profile = "traking"')
        with pytest.raises(ConfigError, match="not defined"):
            load_station_config(write_config(tmp_path, text + PROFILES))

    def test_naming_a_profile_without_declaring_any_is_refused(self, tmp_path):
        # Not a silent fallback to stock: that would hide the typo.
        text = VALID_CONFIG.replace('port = "COM5"', 'port = "COM5"\n    profile = "tracking"')
        with pytest.raises(ConfigError, match="no \\[rotor.profiles\\] section"):
            load_station_config(write_config(tmp_path, text))

    def test_a_knife_edge_profile_is_refused_with_the_file_name(self, tmp_path):
        text = (
            VALID_CONFIG
            + """
    [rotor.profiles.stock]
    deadband_deg = 1.0
    interval_s = 1.0
"""
        )
        path = write_config(tmp_path, text)
        with pytest.raises(ConfigError) as exc:
            load_station_config(path)
        assert "knife edge" in str(exc.value)
        assert str(path) in str(exc.value)

    def test_an_unknown_key_in_a_profile_names_the_file_and_key(self, tmp_path):
        # Forward compatibility with the gain registers: a config that
        # declares gains before they are implemented fails loudly rather
        # than silently doing nothing.
        text = (
            VALID_CONFIG
            + """
    [rotor.profiles.stock]
    deadband_deg = 2.5
    interval_s = 1.0
    gains = 3
"""
        )
        with pytest.raises(ConfigError, match="gains"):
            load_station_config(write_config(tmp_path, text))

    @pytest.mark.parametrize("missing", ["deadband_deg", "interval_s"])
    def test_a_profile_missing_a_required_key_is_refused(self, tmp_path, missing):
        text = VALID_CONFIG + PROFILES
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith(missing))
        with pytest.raises(ConfigError, match=missing):
            load_station_config(write_config(tmp_path, text))

    def test_a_profile_that_is_not_a_table_is_refused(self, tmp_path):
        text = (
            VALID_CONFIG
            + """
    [rotor.profiles]
    stock = 2.5
"""
        )
        with pytest.raises(ConfigError, match="must be a table"):
            load_station_config(write_config(tmp_path, text))


class TestTrackingSettings:
    def test_duplicate_profile_names_are_refused(self):
        # Unreachable through TOML, which rejects a repeated table, but
        # the value object is constructible directly and the toggle
        # refers to profiles by name.
        duplicate = TrackingProfile(name="stock", deadband_deg=2.5, interval_s=1.0)
        with pytest.raises(ValueError, match="share a name"):
            TrackingSettings(profiles=(duplicate, duplicate))

    def test_a_blank_active_name_is_refused(self):
        with pytest.raises(ValueError, match="needs a name"):
            TrackingSettings(active="  ")

    def test_the_synthesized_profile_follows_the_window_it_is_given(self):
        assert TrackingSettings().active_profile(2.5).deadband_deg == 2.5
        assert TrackingSettings().active_profile(1.5).deadband_deg == 1.5

    def test_an_inherited_knife_edge_says_where_the_number_came_from(self):
        # A station whose acceptance window is a whole multiple of the
        # tick has been commanding doubled steps all along. Refusing is
        # right, but the profile's own advice names a key this operator
        # does not have, because they never wrote a profile.
        with pytest.raises(CadenceError) as exc:
            TrackingSettings().active_profile(3.0)
        message = str(exc.value)
        assert "acceptance_window_deg" in message
        assert "[rotor.profiles.stock]" in message
        assert "deadband_deg = 1.5" in message

    def test_the_inherited_knife_edge_does_not_blame_the_window(self):
        # acceptance_window_deg is a separate measurement and changing
        # it would be the wrong fix.
        with pytest.raises(CadenceError) as exc:
            TrackingSettings().active_profile(2.0)
        assert "not what was wrong" in str(exc.value)


MECHANICS_TOML = """
    azimuth_free_play_deg = 2.95
    azimuth_breakaway_pwm = 17.0
    elevation_free_play_deg = 2.55
    elevation_breakaway_pwm = 21.0
"""

SAFE_GAINS_TOML = """
    [rotor.profiles.stock]
    deadband_deg = 2.5
    interval_s = 1.0

    [rotor.profiles.tracking]
    deadband_deg = 0.25
    interval_s = 0.5
    azimuth_kp = 8.0
    azimuth_ki = 0.5
    azimuth_kd = 0.5
    elevation_kp = 10.0
    elevation_ki = 0.5
    elevation_kd = 0.3
"""


class TestRotorMechanics:
    def test_absent_by_default(self, tmp_path):
        config = load_station_config(write_config(tmp_path))
        assert config.capabilities.mechanics_measured is False

    def test_parsed_when_present(self, tmp_path):
        config = load_station_config(write_config(tmp_path, VALID_CONFIG + MECHANICS_TOML))
        assert config.capabilities.mechanics_for("azimuth") == (2.95, 17.0)
        assert config.capabilities.mechanics_for("elevation") == (2.55, 21.0)

    def test_a_partial_record_is_refused_with_the_file_name(self, tmp_path):
        text = VALID_CONFIG + "    azimuth_free_play_deg = 2.95\n"
        with pytest.raises(ConfigError, match="all-or-nothing"):
            load_station_config(write_config(tmp_path, text))

    def test_a_misspelled_key_is_refused(self, tmp_path):
        # The allow-list is what makes a typo an error rather than a
        # silently ignored line -- and a silently ignored free_play_deg
        # would leave the clamp checking against nothing.
        text = VALID_CONFIG + "    azimuth_freeplay_deg = 2.95\n"
        with pytest.raises(ConfigError, match="azimuth_freeplay_deg"):
            load_station_config(write_config(tmp_path, text))

    def test_the_shipped_example_configs_gains_pass_their_own_clamp(self, tmp_path):
        # config.example.toml is documentation people copy, and it now
        # ships live gains rather than commented-out ones. If anybody
        # edits a gain or a measurement in it without redoing the
        # arithmetic, this is what says so -- load_station_config runs
        # the clamp, so an unsafe example would fail to load at all.
        config = load_station_config(Path("config.example.toml"))
        tracking = next(p for p in config.tracking.profiles if p.name == "tracking")
        assert tracking.gains is not None
        tracking.check_against(config.capabilities)

    def test_the_shipped_example_keeps_azimuth_below_elevation(self, tmp_path):
        # The asymmetry is deliberate and easy to "tidy" back to a
        # matching pair: azimuth has the lower breakaway and the larger
        # free play, so it is the binding axis. A future editor who
        # makes them equal should have to delete this test to do it.
        from qsorbit.core.rotor import GainRegister

        config = load_station_config(Path("config.example.toml"))
        tracking = next(p for p in config.tracking.profiles if p.name == "tracking")
        gains = tracking.gains
        assert gains is not None
        assert gains[GainRegister.AZIMUTH_KI] < gains[GainRegister.ELEVATION_KI]

    def test_the_shipped_example_config_declares_them(self, tmp_path):
        # config.example.toml is documentation people copy, so the
        # measured fields being present and parseable is part of the
        # deliverable rather than an extra.
        config = load_station_config(Path("config.example.toml"))
        assert config.capabilities.mechanics_measured is True


class TestProfileGains:
    def test_gains_are_parsed_onto_the_profile(self, tmp_path):
        config = load_station_config(
            write_config(tmp_path, VALID_CONFIG + MECHANICS_TOML + SAFE_GAINS_TOML)
        )
        tracking = next(p for p in config.tracking.profiles if p.name == "tracking")
        assert tracking.azimuth_kp == 8.0
        assert tracking.elevation_kd == 0.3
        assert tracking.gains is not None

    def test_a_profile_without_gains_still_loads(self, tmp_path):
        text = (
            VALID_CONFIG
            + """
    [rotor.profiles.stock]
    deadband_deg = 2.5
    interval_s = 1.0
"""
        )
        config = load_station_config(write_config(tmp_path, text))
        assert config.tracking.profiles[0].gains is None

    def test_a_partial_gain_set_is_refused_with_the_file_name(self, tmp_path):
        text = (
            VALID_CONFIG
            + MECHANICS_TOML
            + """
    [rotor.profiles.tracking]
    deadband_deg = 0.25
    interval_s = 0.5
    azimuth_ki = 1.0
"""
        )
        with pytest.raises(ConfigError, match="all six or none"):
            load_station_config(write_config(tmp_path, text))

    def test_a_misspelled_gain_key_is_refused(self, tmp_path):
        text = (
            VALID_CONFIG
            + MECHANICS_TOML
            + """
    [rotor.profiles.tracking]
    deadband_deg = 0.25
    interval_s = 0.5
    azimuth_ki_gain = 1.0
"""
        )
        with pytest.raises(ConfigError, match="azimuth_ki_gain"):
            load_station_config(write_config(tmp_path, text))

    def test_the_clamp_refuses_an_unsafe_profile_at_load(self, tmp_path):
        # The point of checking here as well as at the push site: this
        # fails at the desk, in daylight, rather than at the rotor.
        text = (VALID_CONFIG + MECHANICS_TOML + SAFE_GAINS_TOML).replace(
            "azimuth_ki = 0.5", "azimuth_ki = 1.0"
        )
        with pytest.raises(ConfigError) as caught:
            load_station_config(write_config(tmp_path, text))
        assert "breakaway" in str(caught.value)
        assert "qsorbit.toml" in str(caught.value)

    def test_integral_gain_without_measurements_is_refused_at_load(self, tmp_path):
        text = VALID_CONFIG + SAFE_GAINS_TOML
        with pytest.raises(ConfigError, match="no free_play_deg or breakaway_pwm"):
            load_station_config(write_config(tmp_path, text))

    def test_zero_integral_gain_loads_without_measurements(self, tmp_path):
        # Nothing that can accumulate, so nothing to check against.
        text = (VALID_CONFIG + SAFE_GAINS_TOML).replace("ki = 0.5", "ki = 0.0")
        config = load_station_config(write_config(tmp_path, text))
        tracking = next(p for p in config.tracking.profiles if p.name == "tracking")
        assert tracking.gains is not None

    def test_an_inactive_profile_is_checked_too(self, tmp_path):
        # A profile nobody has selected yet is one keystroke from being
        # selected, and --rotor-profile is that keystroke. Refusing only
        # the active one would move the failure to mid-pass.
        text = (VALID_CONFIG + MECHANICS_TOML + SAFE_GAINS_TOML).replace(
            "azimuth_ki = 0.5", "azimuth_ki = 1.0"
        )
        # Left on the default active profile: the unsafe one is not
        # selected.
        assert "profile = " not in text
        with pytest.raises(ConfigError, match="breakaway"):
            load_station_config(write_config(tmp_path, text))
