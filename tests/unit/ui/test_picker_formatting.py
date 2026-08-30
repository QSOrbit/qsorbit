"""Tests for the target picker's row and staleness wording. No Qt.

Same shape as ``test_frequency_formatting``, ``test_quieting_formatting``,
and ``test_readout_formatting``: the decisions about what a picker row
*says* are the part worth testing, and they should not need a window.

Builds :class:`~qsorbit.core.picker.PickerEntry` and
:class:`~qsorbit.core.tracker.Pass` by hand rather than through a real
TLE and ``predict_passes`` -- ``test_picker.py``'s own ``TestSortKey``
does the same for the same reason: proving what this module *displays*
doesn't need a real orbit, only the shapes these functions read.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

from qsorbit.core.geometry import AzEl
from qsorbit.core.picker import Band, PickerEntry
from qsorbit.core.profiles import (
    AliveRecord,
    AliveStatus,
    CatalogManifest,
    Mode,
    ReliabilityClass,
    SatelliteProfile,
    Transmitter,
)
from qsorbit.core.tracker import Pass, PassEvent
from qsorbit.ui.picker_formatting import (
    NO_DATA_LABEL,
    PickerRowText,
    alive_status_role,
    catalogue_staleness_text,
    format_band,
    picker_row_text,
    reliability_letter,
)

NOW = datetime(2026, 8, 28, 0, 0, 0, tzinfo=UTC)
_EASTERN = timezone(timedelta(hours=-4))  # fixed offset -- portable, unlike a real zoneinfo


def _alive(status=AliveStatus.ACTIVE, as_of=date(2026, 8, 25)):
    return AliveRecord(status=status, as_of=as_of, source="test")


def _transmitter(
    downlink_hz=435_640_000.0, mode=Mode.SSB, reliability=ReliabilityClass.UNCONDITIONAL
):
    return Transmitter(downlink_hz=downlink_hz, mode=mode, reliability=reliability)


def _profile(name="RS-44", transmitters=(), alive=None):
    return SatelliteProfile(
        norad_id=44909,
        name=name,
        transmitters=transmitters,
        alive=alive if alive is not None else _alive(),
    )


def _pass(aos_hour=1.0, los_hour=1.25, max_elevation_deg=62.0):
    aos = PassEvent(time=NOW + timedelta(hours=aos_hour), sky_position=AzEl(180.0, 5.0))
    los = PassEvent(time=NOW + timedelta(hours=los_hour), sky_position=AzEl(20.0, 5.0))
    tca = PassEvent(
        time=NOW + timedelta(hours=(aos_hour + los_hour) / 2),
        sky_position=AzEl(90.0, max_elevation_deg),
    )
    return Pass(
        aos=aos, los=los, tca=tca, max_elevation_deg=max_elevation_deg, az_track=(aos, tca, los)
    )


class TestReliabilityLetter:
    def test_unconditional_is_a(self):
        assert reliability_letter(ReliabilityClass.UNCONDITIONAL) == "A"

    def test_scheduled_is_b(self):
        assert reliability_letter(ReliabilityClass.SCHEDULED) == "B"

    def test_dependent_is_c(self):
        assert reliability_letter(ReliabilityClass.DEPENDENT) == "C"


class TestAliveStatusRole:
    def test_active_is_ok(self):
        assert alive_status_role(AliveStatus.ACTIVE) == "ok"

    def test_unknown_is_warn(self):
        """UNKNOWN is the mockup's "amber = intermittent" -- unconfirmed reads as caution."""
        assert alive_status_role(AliveStatus.UNKNOWN) == "warn"

    def test_inactive_is_dim(self):
        assert alive_status_role(AliveStatus.INACTIVE) == "dim"


class TestFormatBand:
    def test_two_meters(self):
        assert format_band(Band.TWO_METERS) == "2 m"

    def test_seventy_cm(self):
        assert format_band(Band.SEVENTY_CM) == "70 cm"

    def test_other(self):
        assert format_band(Band.OTHER) == "other"


class TestPickerRowText:
    def test_a_live_satellite_with_an_upcoming_pass(self):
        entry = PickerEntry(
            profile=_profile(transmitters=(_transmitter(),)),
            next_pass=_pass(),
            visible_from_latitude=True,
        )

        text = picker_row_text(entry, local_zone=_EASTERN)

        assert text == PickerRowText(
            name="RS-44",
            status_role="ok",
            pass_text="21:00 → 21:15",
            max_elevation_text="62°",
            downlink_text="435.640",
            mode_text="SSB",
            tier_text="A",
        )

    def test_no_pass_in_the_window_shows_the_placeholder(self):
        entry = PickerEntry(
            profile=_profile(transmitters=(_transmitter(),)),
            next_pass=None,
            visible_from_latitude=True,
        )

        text = picker_row_text(entry, local_zone=_EASTERN)

        assert text.pass_text == NO_DATA_LABEL
        assert text.max_elevation_text == NO_DATA_LABEL

    def test_no_transmitter_shows_the_placeholder_for_downlink_mode_and_tier(self):
        """A profile that exists only to carry a curated alive fact.

        See ``profile.py``'s own module docstring.
        """
        entry = PickerEntry(
            profile=_profile(transmitters=()), next_pass=None, visible_from_latitude=True
        )

        text = picker_row_text(entry, local_zone=_EASTERN)

        assert text.downlink_text == NO_DATA_LABEL
        assert text.mode_text == NO_DATA_LABEL
        assert text.tier_text == NO_DATA_LABEL

    def test_a_dead_satellite_shows_dead_and_the_as_of_month_in_the_tier_cell(self):
        """Overrides the reliability letter -- see the module docstring."""
        entry = PickerEntry(
            profile=_profile(
                transmitters=(_transmitter(),),
                alive=_alive(status=AliveStatus.INACTIVE, as_of=date(2025, 6, 12)),
            ),
            next_pass=None,
            visible_from_latitude=True,
        )

        text = picker_row_text(entry, local_zone=_EASTERN)

        assert text.tier_text == "dead 2025-06"
        assert text.status_role == "dim"

    def test_an_unknown_status_reads_as_warn_not_dead(self):
        entry = PickerEntry(
            profile=_profile(
                transmitters=(_transmitter(),),
                alive=_alive(status=AliveStatus.UNKNOWN),
            ),
            next_pass=None,
            visible_from_latitude=True,
        )

        text = picker_row_text(entry, local_zone=_EASTERN)

        assert text.status_role == "warn"
        assert text.tier_text == "A"

    def test_reliability_letter_reduces_to_the_best_transmitter(self):
        """RS-44's real shape: an unconditional beacon plus a dependent transponder."""
        entry = PickerEntry(
            profile=_profile(
                transmitters=(
                    _transmitter(reliability=ReliabilityClass.DEPENDENT),
                    _transmitter(
                        downlink_hz=145_960_000.0, reliability=ReliabilityClass.UNCONDITIONAL
                    ),
                )
            ),
            next_pass=None,
            visible_from_latitude=True,
        )

        text = picker_row_text(entry, local_zone=_EASTERN)

        assert text.tier_text == "A"
        assert text.downlink_text == "145.960"


class TestCatalogueStalenessText:
    def test_no_manifest_gives_no_line(self):
        assert catalogue_staleness_text(None, date(2026, 8, 30)) is None

    def test_reports_the_shipped_date_and_age_in_days(self):
        manifest = CatalogManifest(shipped=date(2026, 8, 25))

        assert catalogue_staleness_text(manifest, date(2026, 8, 30)) == (
            "catalogue: shipped 2026-08-25 (5 d)"
        )

    def test_shipped_today_is_zero_days_old(self):
        manifest = CatalogManifest(shipped=date(2026, 8, 30))

        assert catalogue_staleness_text(manifest, date(2026, 8, 30)) == (
            "catalogue: shipped 2026-08-30 (0 d)"
        )
