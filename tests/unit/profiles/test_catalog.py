"""Unit tests for loading profiles from TOML files.

Every test writes a real TOML file to a temporary directory and loads
it, rather than constructing dicts -- the parsing and validation are
both part of what's being tested, matching test_station.py's own
reasoning for the same choice.
"""

import textwrap
from datetime import date

import pytest

from qsorbit.core.profiles.catalog import (
    DEFAULT_PROFILES_DIR,
    ProfileCatalog,
    ProfileError,
    load_profile_catalog,
)
from qsorbit.core.profiles.profile import (
    AliveRecord,
    AliveStatus,
    Mode,
    ReliabilityClass,
    SatelliteProfile,
)

VALID_PROFILE = """
    norad_id = 44909
    name = "RS-44"
    also_known_as = ["DOSAAF-85"]

    [alive]
    status = "active"
    as_of = 2026-08-25
    source = "AMSAT status board"

    [[transmitters]]
    downlink_hz = 435605000.0
    mode = "cw"
    reliability = "unconditional"
    notes = "CW beacon"
"""


def write_profile(tmp_path, text=VALID_PROFILE, name="rs44.toml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


class TestLoadProfileCatalog:
    def test_loads_a_single_valid_profile(self, tmp_path):
        write_profile(tmp_path)

        catalog = load_profile_catalog(tmp_path)

        assert len(catalog) == 1
        profile = catalog.by_norad_id(44909)
        assert profile.name == "RS-44"
        assert profile.also_known_as == ("DOSAAF-85",)
        assert profile.alive.status is AliveStatus.ACTIVE
        assert len(profile.transmitters) == 1
        assert profile.transmitters[0].mode is Mode.CW
        assert profile.transmitters[0].reliability is ReliabilityClass.UNCONDITIONAL

    def test_loads_every_toml_file_in_the_directory(self, tmp_path):
        write_profile(tmp_path, VALID_PROFILE, name="rs44.toml")
        write_profile(
            tmp_path,
            VALID_PROFILE.replace("44909", "24278").replace("RS-44", "FO-29"),
            name="fo29.toml",
        )

        catalog = load_profile_catalog(tmp_path)

        assert len(catalog) == 2
        assert catalog.by_norad_id(24278).name == "FO-29"

    def test_ignores_non_toml_files(self, tmp_path):
        write_profile(tmp_path)
        (tmp_path / "README.md").write_text("not a profile", encoding="utf-8")

        catalog = load_profile_catalog(tmp_path)

        assert len(catalog) == 1

    def test_profile_with_no_transmitters_loads(self, tmp_path):
        text = """
            norad_id = 1
            name = "Dead-Sat"

            [alive]
            status = "inactive"
            as_of = 2026-08-25
            source = "confirmed decommissioned"
        """
        write_profile(tmp_path, text, name="dead.toml")

        catalog = load_profile_catalog(tmp_path)

        assert catalog.by_norad_id(1).transmitters == ()

    def test_missing_directory_is_an_error(self, tmp_path):
        with pytest.raises(ProfileError, match="not found"):
            load_profile_catalog(tmp_path / "does-not-exist")

    def test_unknown_top_level_key_is_an_error(self, tmp_path):
        write_profile(tmp_path, VALID_PROFILE + "\nunexpected = 1\n")

        with pytest.raises(ProfileError, match="unexpected"):
            load_profile_catalog(tmp_path)

    def test_missing_norad_id_is_an_error(self, tmp_path):
        text = VALID_PROFILE.replace("norad_id = 44909\n", "")
        write_profile(tmp_path, text)

        with pytest.raises(ProfileError, match="norad_id"):
            load_profile_catalog(tmp_path)

    def test_non_integer_norad_id_is_an_error(self, tmp_path):
        text = VALID_PROFILE.replace("norad_id = 44909", 'norad_id = "44909"')
        write_profile(tmp_path, text)

        with pytest.raises(ProfileError, match="norad_id"):
            load_profile_catalog(tmp_path)

    def test_missing_alive_section_is_an_error(self, tmp_path):
        text = """
            norad_id = 1
            name = "X"
        """
        write_profile(tmp_path, text)

        with pytest.raises(ProfileError, match="alive"):
            load_profile_catalog(tmp_path)

    def test_unknown_key_in_alive_is_an_error(self, tmp_path):
        text = VALID_PROFILE.replace(
            'source = "AMSAT status board"',
            'source = "AMSAT status board"\n    checked_by = "phil"',
        )
        write_profile(tmp_path, text)

        with pytest.raises(ProfileError, match="checked_by"):
            load_profile_catalog(tmp_path)

    def test_invalid_alive_status_is_an_error(self, tmp_path):
        text = VALID_PROFILE.replace('status = "active"', 'status = "mostly-active"')
        write_profile(tmp_path, text)

        with pytest.raises(ProfileError, match="mostly-active"):
            load_profile_catalog(tmp_path)

    def test_non_date_as_of_is_an_error(self, tmp_path):
        text = VALID_PROFILE.replace("as_of = 2026-08-25", 'as_of = "2026-08-25"')
        write_profile(tmp_path, text)

        with pytest.raises(ProfileError, match="as_of"):
            load_profile_catalog(tmp_path)

    def test_invalid_mode_is_an_error(self, tmp_path):
        text = VALID_PROFILE.replace('mode = "cw"', 'mode = "morse"')
        write_profile(tmp_path, text)

        with pytest.raises(ProfileError, match="morse"):
            load_profile_catalog(tmp_path)

    def test_invalid_reliability_is_an_error(self, tmp_path):
        text = VALID_PROFILE.replace('reliability = "unconditional"', 'reliability = "always"')
        write_profile(tmp_path, text)

        with pytest.raises(ProfileError, match="always"):
            load_profile_catalog(tmp_path)

    def test_unknown_key_in_transmitter_is_an_error(self, tmp_path):
        text = VALID_PROFILE.replace(
            'notes = "CW beacon"', 'notes = "CW beacon"\n    kind = "beacon"'
        )
        write_profile(tmp_path, text)

        with pytest.raises(ProfileError, match=r"transmitters\[0\]"):
            load_profile_catalog(tmp_path)

    def test_out_of_range_transmitter_value_is_an_error(self, tmp_path):
        text = VALID_PROFILE.replace("downlink_hz = 435605000.0", "downlink_hz = -1.0")
        write_profile(tmp_path, text)

        with pytest.raises(ProfileError, match="downlink_hz"):
            load_profile_catalog(tmp_path)

    def test_transmitter_that_is_not_a_table_is_an_error(self, tmp_path):
        text = """
            norad_id = 1
            name = "X"
            transmitters = ["not", "tables"]

            [alive]
            status = "active"
            as_of = 2026-08-25
            source = "test"
        """
        write_profile(tmp_path, text)

        with pytest.raises(ProfileError, match=r"transmitters\[0\] in .* must be a table"):
            load_profile_catalog(tmp_path)

    def test_shipped_starter_set_loads_and_is_reasonably_sized(self):
        catalog = load_profile_catalog(DEFAULT_PROFILES_DIR)

        assert len(catalog) >= 10
        for profile in catalog:
            assert profile.norad_id > 0
            assert profile.name


class TestProfileCatalog:
    def _profile(self, norad_id, name):
        return SatelliteProfile(
            norad_id=norad_id,
            name=name,
            transmitters=(),
            alive=AliveRecord(status=AliveStatus.UNKNOWN, as_of=date(2026, 8, 25), source="x"),
        )

    def test_by_norad_id_returns_none_for_an_unknown_id(self, tmp_path):
        write_profile(tmp_path)
        catalog = load_profile_catalog(tmp_path)

        assert catalog.by_norad_id(1) is None

    def test_duplicate_norad_id_is_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            ProfileCatalog([self._profile(1, "A"), self._profile(1, "B")])

    def test_is_iterable(self):
        catalog = ProfileCatalog([self._profile(1, "A"), self._profile(2, "B")])

        names = sorted(profile.name for profile in catalog)

        assert names == ["A", "B"]

    def test_len(self):
        catalog = ProfileCatalog([self._profile(1, "A"), self._profile(2, "B")])

        assert len(catalog) == 2
