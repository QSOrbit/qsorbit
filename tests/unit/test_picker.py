"""Unit tests for the target picker's data layer.

The TLE and observer here are the same "TEME EXAMPLE" pair
test_pass_prediction.py and test_cli.py's TestPlan already prove
produces multiple passes in a 48-hour window from this observer -- the
pass geometry is proven elsewhere, not a fresh unverified claim.
"""

import textwrap
from datetime import UTC, date, datetime, timedelta

import pytest

from qsorbit.core.geometry import AzEl
from qsorbit.core.horizon import HorizonMask
from qsorbit.core.picker import (
    Band,
    ModeGroup,
    PickerEntry,
    PickerFilters,
    _sort_key,
    build_picker_entries,
    classify_band,
    mode_group,
    passes_filters,
    primary_transmitter,
)
from qsorbit.core.profiles import (
    AliveRecord,
    AliveStatus,
    Mode,
    ProfileCatalog,
    ReliabilityClass,
    SatelliteProfile,
    Transmitter,
)
from qsorbit.core.tracker import (
    ObserverLocation,
    Pass,
    PassEvent,
    Satellite,
    predict_passes,
)

TEME_EXAMPLE_TLE = """\
TEME EXAMPLE
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
"""

OBSERVER = ObserverLocation(latitude=40.0, longitude=-83.0, altitude_m=250.0)
NOW = datetime(2026, 8, 28, 0, 0, 0, tzinfo=UTC)


def _alive(status=AliveStatus.ACTIVE):
    return AliveRecord(status=status, as_of=date(2026, 8, 25), source="test")


def _transmitter(downlink_hz, mode, reliability):
    return Transmitter(downlink_hz=downlink_hz, mode=mode, reliability=reliability)


def _profile(norad_id=5, name="TEME EXAMPLE", transmitters=(), alive=None):
    return SatelliteProfile(
        norad_id=norad_id,
        name=name,
        transmitters=transmitters,
        alive=alive if alive is not None else _alive(),
    )


class TestClassifyBand:
    def test_two_meters(self):
        assert classify_band(145_825_000.0) is Band.TWO_METERS

    def test_seventy_cm(self):
        assert classify_band(435_640_000.0) is Band.SEVENTY_CM

    def test_two_meters_lower_edge(self):
        assert classify_band(144_000_000.0) is Band.TWO_METERS

    def test_seventy_cm_upper_edge(self):
        assert classify_band(450_000_000.0) is Band.SEVENTY_CM

    def test_out_of_band(self):
        assert classify_band(10_450_000_000.0) is Band.OTHER


class TestModeGroup:
    @pytest.mark.parametrize(
        "mode,expected",
        [
            (Mode.FM, ModeGroup.FM),
            (Mode.SSB, ModeGroup.SSB_CW),
            (Mode.CW, ModeGroup.SSB_CW),
            (Mode.AFSK1200, ModeGroup.DIGITAL),
            (Mode.BPSK, ModeGroup.DIGITAL),
            (Mode.SSTV, ModeGroup.DIGITAL),
        ],
    )
    def test_grouping(self, mode, expected):
        assert mode_group(mode) is expected


class TestPrimaryTransmitter:
    def test_none_when_no_transmitters(self):
        assert primary_transmitter(_profile(transmitters=())) is None

    def test_single_transmitter(self):
        transmitter = _transmitter(435_640_000.0, Mode.CW, ReliabilityClass.UNCONDITIONAL)
        profile = _profile(transmitters=(transmitter,))

        assert primary_transmitter(profile) is transmitter

    def test_picks_the_most_favorable_reliability(self):
        # A beacon (unconditional) and a transponder (dependent) on the
        # same bird -- same shape as RS-44's real profile.
        beacon = _transmitter(435_605_000.0, Mode.CW, ReliabilityClass.UNCONDITIONAL)
        transponder = _transmitter(435_640_000.0, Mode.SSB, ReliabilityClass.DEPENDENT)
        profile = _profile(transmitters=(transponder, beacon))

        assert primary_transmitter(profile) is beacon


class TestPassesFilters:
    def test_default_filters_pass_everything(self):
        assert passes_filters(_profile(transmitters=()), PickerFilters())

    def test_needs_transmitter_excludes_empty_profiles(self):
        filters = PickerFilters(needs_transmitter=True)

        assert not passes_filters(_profile(transmitters=()), filters)

    def test_needs_transmitter_keeps_profiles_with_one(self):
        filters = PickerFilters(needs_transmitter=True)
        profile = _profile(
            transmitters=(_transmitter(435_640_000.0, Mode.CW, ReliabilityClass.UNCONDITIONAL),)
        )

        assert passes_filters(profile, filters)

    def test_band_filter_matches_any_transmitter(self):
        filters = PickerFilters(bands=frozenset({Band.TWO_METERS}))
        profile = _profile(
            transmitters=(
                _transmitter(435_640_000.0, Mode.CW, ReliabilityClass.UNCONDITIONAL),
                _transmitter(145_825_000.0, Mode.AFSK1200, ReliabilityClass.SCHEDULED),
            )
        )

        assert passes_filters(profile, filters)

    def test_band_filter_excludes_when_no_transmitter_matches(self):
        filters = PickerFilters(bands=frozenset({Band.TWO_METERS}))
        profile = _profile(
            transmitters=(_transmitter(435_640_000.0, Mode.CW, ReliabilityClass.UNCONDITIONAL),)
        )

        assert not passes_filters(profile, filters)

    def test_mode_group_filter(self):
        filters = PickerFilters(mode_groups=frozenset({ModeGroup.FM}))
        fm_profile = _profile(
            transmitters=(_transmitter(437_800_000.0, Mode.FM, ReliabilityClass.UNCONDITIONAL),)
        )
        cw_profile = _profile(
            transmitters=(_transmitter(435_640_000.0, Mode.CW, ReliabilityClass.UNCONDITIONAL),)
        )

        assert passes_filters(fm_profile, filters)
        assert not passes_filters(cw_profile, filters)

    def test_reliability_filter(self):
        filters = PickerFilters(reliability_classes=frozenset({ReliabilityClass.UNCONDITIONAL}))
        unconditional = _profile(
            transmitters=(_transmitter(435_640_000.0, Mode.CW, ReliabilityClass.UNCONDITIONAL),)
        )
        dependent = _profile(
            transmitters=(_transmitter(435_640_000.0, Mode.SSB, ReliabilityClass.DEPENDENT),)
        )

        assert passes_filters(unconditional, filters)
        assert not passes_filters(dependent, filters)

    def test_reliability_filter_excludes_profiles_with_no_transmitters(self):
        filters = PickerFilters(reliability_classes=frozenset({ReliabilityClass.UNCONDITIONAL}))

        assert not passes_filters(_profile(transmitters=()), filters)

    def test_all_axes_combine_with_and(self):
        filters = PickerFilters(
            needs_transmitter=True,
            bands=frozenset({Band.SEVENTY_CM}),
            mode_groups=frozenset({ModeGroup.SSB_CW}),
            reliability_classes=frozenset({ReliabilityClass.UNCONDITIONAL}),
        )
        matches_everything = _profile(
            transmitters=(_transmitter(435_640_000.0, Mode.CW, ReliabilityClass.UNCONDITIONAL),)
        )
        wrong_band = _profile(
            transmitters=(_transmitter(145_825_000.0, Mode.CW, ReliabilityClass.UNCONDITIONAL),)
        )

        assert passes_filters(matches_everything, filters)
        assert not passes_filters(wrong_band, filters)


class TestBuildPickerEntries:
    def _tle_dir(self, tmp_path, text=TEME_EXAMPLE_TLE):
        directory = tmp_path / "tles"
        directory.mkdir()
        (directory / "teme.tle").write_text(textwrap.dedent(text), encoding="utf-8")
        return directory

    def test_matches_a_tle_to_its_curated_profile(self, tmp_path):
        tle_dir = self._tle_dir(tmp_path)
        catalog = ProfileCatalog([_profile(norad_id=5, name="TEME EXAMPLE")])

        entries = build_picker_entries(catalog, tle_dir, OBSERVER, HorizonMask(), NOW, hours=48.0)

        assert len(entries) == 1
        assert entries[0].profile.name == "TEME EXAMPLE"
        assert entries[0].next_pass is not None

    def test_unmatched_tle_is_skipped(self, tmp_path):
        tle_dir = self._tle_dir(tmp_path)
        catalog = ProfileCatalog([_profile(norad_id=99999, name="SOMETHING ELSE")])

        entries = build_picker_entries(catalog, tle_dir, OBSERVER, HorizonMask(), NOW, hours=48.0)

        assert entries == ()

    def test_unparseable_tle_is_skipped(self, tmp_path):
        tle_dir = tmp_path / "tles"
        tle_dir.mkdir()
        (tle_dir / "garbage.tle").write_text("not a tle\nat all\n", encoding="utf-8")
        catalog = ProfileCatalog([_profile(norad_id=5, name="TEME EXAMPLE")])

        entries = build_picker_entries(catalog, tle_dir, OBSERVER, HorizonMask(), NOW)

        assert entries == ()

    def test_no_pass_in_window_still_produces_an_entry(self, tmp_path):
        tle_dir = self._tle_dir(tmp_path)
        catalog = ProfileCatalog([_profile(norad_id=5, name="TEME EXAMPLE")])

        entries = build_picker_entries(catalog, tle_dir, OBSERVER, HorizonMask(), NOW, hours=0.001)

        assert len(entries) == 1
        assert entries[0].next_pass is None

    def test_multiple_passes_in_the_window_come_back_earliest_first(self, tmp_path):
        # Reuses test_cli.py's own proof (TestPlan.test_multiple_passes_
        # come_out_in_chronological_order) that this exact TLE/observer
        # pair produces more than one pass in a 48-hour window -- here
        # checking that build_picker_entries takes the *earliest* one
        # as next_pass, not just any of them.
        tle_dir = self._tle_dir(tmp_path)
        catalog = ProfileCatalog([_profile(norad_id=5, name="TEME EXAMPLE")])

        entries = build_picker_entries(catalog, tle_dir, OBSERVER, HorizonMask(), NOW, hours=48.0)
        all_passes = predict_passes(
            Satellite.from_file(tle_dir / "teme.tle"),
            OBSERVER,
            NOW,
            NOW + timedelta(hours=48.0),
        )

        assert len(all_passes) > 1
        assert entries[0].next_pass.aos.time == min(p.aos.time for p in all_passes)


class TestSortKey:
    """Direct tests of the private sort key, since proving order needs
    more than one distinct satellite and a real second orbit isn't
    worth the trouble for what is otherwise pure key-comparison logic.
    """

    def _pass_at(self, hour):
        event = PassEvent(time=NOW + timedelta(hours=hour), sky_position=AzEl(0.0, 10.0))
        return Pass(aos=event, los=event, tca=event, max_elevation_deg=10.0, az_track=(event,))

    def test_a_pass_sorts_before_no_pass(self):
        with_pass = PickerEntry(profile=_profile(name="Z"), next_pass=self._pass_at(1))
        without_pass = PickerEntry(profile=_profile(name="A"), next_pass=None)

        assert sorted([without_pass, with_pass], key=_sort_key) == [with_pass, without_pass]

    def test_earlier_pass_sorts_first(self):
        later = PickerEntry(profile=_profile(name="LATER"), next_pass=self._pass_at(5))
        earlier = PickerEntry(profile=_profile(name="EARLIER"), next_pass=self._pass_at(1))

        assert sorted([later, earlier], key=_sort_key) == [earlier, later]

    def test_no_pass_entries_sort_by_name(self):
        z = PickerEntry(profile=_profile(name="Z"), next_pass=None)
        a = PickerEntry(profile=_profile(name="A"), next_pass=None)

        assert sorted([z, a], key=_sort_key) == [a, z]
