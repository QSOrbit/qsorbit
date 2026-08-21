"""Unit tests for the readout window's pure formatting helpers.

Deliberately headless: this file imports qsorbit.ui.readout_formatting
only, never qsorbit.ui.readout_window, so nothing here needs PySide6
installed. See that module's docstring for why the split exists.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from qsorbit.core.geometry import AzEl
from qsorbit.core.pointing import TickOutcome, TrackSample
from qsorbit.core.rotor import Position
from qsorbit.ui.readout_formatting import (
    ReadoutText,
    format_axis_position,
    format_azel,
    format_outcome,
    format_range,
    format_rotor_as_sky,
    format_time,
    readout_text,
)

_DEFAULT_TIME = datetime(2026, 8, 21, 18, 30, 15, tzinfo=UTC)


def sample(
    *,
    time: datetime | None = None,
    sky_position: AzEl | None = None,
    range_km: float = 812.0,
    range_rate_km_s: float = -4.25,
    rotor_target: Position | None = None,
    rotor_position: Position | None = None,
    outcome: TickOutcome = TickOutcome.COMMANDED,
) -> TrackSample:
    """A TrackSample with reasonable defaults, one override at a time."""
    return TrackSample(
        time=time if time is not None else _DEFAULT_TIME,
        sky_position=sky_position if sky_position is not None else AzEl(180.0, 45.0),
        range_km=range_km,
        range_rate_km_s=range_rate_km_s,
        rotor_target=rotor_target if rotor_target is not None else Position(180.0, 45.0),
        rotor_position=rotor_position if rotor_position is not None else Position(178.5, 43.0),
        outcome=outcome,
    )


class TestFormatTime:
    def test_formats_utc_alongside_an_injected_local_zone(self):
        eastern = timezone(timedelta(hours=-4))
        instant = datetime(2026, 8, 21, 18, 30, 15, tzinfo=UTC)
        assert format_time(instant, local_zone=eastern) == "18:30:15 UTC  (14:30:15 local)"

    def test_converts_a_non_utc_source_zone_to_utc_and_to_local(self):
        source_zone = timezone(timedelta(hours=-4))
        instant = datetime(2026, 8, 21, 14, 30, 15, tzinfo=source_zone)
        pacific = timezone(timedelta(hours=-7))
        assert format_time(instant, local_zone=pacific) == "18:30:15 UTC  (11:30:15 local)"

    def test_defaults_to_the_systems_own_local_zone(self):
        # Can't assert an exact local time here - it depends on whatever
        # machine happens to run this test, which is exactly why every
        # other test in this class injects a fixed zone instead. This
        # only proves the no-argument path runs and is still labelled
        # "local" rather than silently falling back to UTC.
        instant = datetime(2026, 8, 21, 18, 30, 15, tzinfo=UTC)
        result = format_time(instant)
        assert result.startswith("18:30:15 UTC  (")
        assert result.endswith(" local)")


class TestFormatAzEl:
    def test_formats_azimuth_and_elevation(self):
        assert format_azel(AzEl(180.0, 45.0)) == "AZ 180.0  EL 45.0"

    def test_rounds_to_one_decimal(self):
        # 123.44 rounds down, 6.77 rounds up - deliberately not values
        # sitting on a rounding half-boundary, where a float's exact
        # stored value (not the decimal you typed) decides which way
        # standard rounding falls. 123.45 looked like a clean example
        # but isn't: it's stored as very slightly above 123.45, so it
        # rounds to 123.5, not 123.4 - a bug in this test, not in
        # format_azel.
        assert format_azel(AzEl(123.44, 6.77)) == "AZ 123.4  EL 6.8"


class TestFormatAxisPosition:
    def test_labels_it_as_an_axis_reading(self):
        # The whole point of a distinct function: this must never read
        # like a sky direction, even though the numbers alone would look
        # identical to format_azel's output.
        assert format_axis_position(Position(180.0, 45.0)) == "AZ 180.0  EL 45.0  (axis reading)"

    def test_handles_a_freshly_homed_negative_reading(self):
        assert format_axis_position(Position(-1.5, 2.0)) == "AZ -1.5  EL 2.0  (axis reading)"


class TestFormatRotorAsSky:
    def test_labels_it_as_uncalibrated_not_an_axis_reading(self):
        # Distinct from both format_azel (true sky) and
        # format_axis_position (raw axis) - this is a derived
        # conversion, not a measurement.
        assert format_rotor_as_sky(Position(180.0, 45.0)) == "AZ 180.0  EL 45.0  (uncalibrated)"

    def test_wraps_a_freshly_homed_reading_to_a_compass_bearing(self):
        # The whole point: AZ-1.5 isn't directly comparable to a sky
        # target's AZ 358.x without doing the mod-360 math yourself.
        # This field does that math.
        assert format_rotor_as_sky(Position(-1.5, 2.0)) == "AZ 358.5  EL 2.0  (uncalibrated)"

    def test_wraps_azimuth_past_a_full_turn(self):
        assert format_rotor_as_sky(Position(380.0, 10.0)) == "AZ 20.0  EL 10.0  (uncalibrated)"

    def test_clamps_elevation_past_vertical(self):
        assert format_rotor_as_sky(Position(10.0, 95.0)) == "AZ 10.0  EL 90.0  (uncalibrated)"


class TestFormatRange:
    def test_receding(self):
        assert format_range(812.0, 4.25) == "812 km, receding at 4.250 km/s"

    def test_approaching(self):
        assert format_range(812.0, -4.25) == "812 km, approaching at 4.250 km/s"

    def test_steady(self):
        assert format_range(812.0, 0.0) == "812 km, range steady"

    def test_range_is_rounded_to_whole_km(self):
        assert format_range(811.6, 0.0) == "812 km, range steady"


class TestFormatOutcome:
    def test_commanded(self):
        assert format_outcome(TickOutcome.COMMANDED) == "commanded"

    def test_within_deadband_reads_as_two_words(self):
        assert format_outcome(TickOutcome.WITHIN_DEADBAND) == "within deadband"

    def test_below_horizon_reads_as_two_words(self):
        assert format_outcome(TickOutcome.BELOW_HORIZON) == "below horizon"


class TestReadoutText:
    def test_returns_a_readout_text(self):
        assert isinstance(readout_text(sample(), target_name="AO-91"), ReadoutText)

    def test_carries_the_target_name_through(self):
        # TrackSample itself carries no name - the loop tracks one target
        # for its whole life, so readout_text threads it through instead.
        text = readout_text(sample(), target_name="AO-91")
        assert text.target_name == "AO-91"

    def test_every_field_matches_its_own_formatter(self):
        one_sample = sample()
        text = readout_text(one_sample, target_name="ISS")

        assert text.time == format_time(one_sample.time)
        assert text.sky_position == format_azel(one_sample.sky_position)
        assert text.rotor_axis == format_axis_position(one_sample.rotor_position)
        assert text.rotor_as_sky == format_rotor_as_sky(one_sample.rotor_position)
        assert text.range_and_rate == format_range(one_sample.range_km, one_sample.range_rate_km_s)
        assert text.outcome == format_outcome(one_sample.outcome)

    def test_rotor_as_sky_actually_uses_rotor_to_sky(self):
        # The bug this guards against: rotor_as_sky silently formatting
        # the raw reading instead of the converted one, which would
        # make rotor_to_sky() dead code with nothing left to catch it.
        text = readout_text(sample(rotor_position=Position(-1.5, 2.0)), target_name="AO-91")
        assert text.rotor_as_sky == "AZ 358.5  EL 2.0  (uncalibrated)"
        assert "-1.5" not in text.rotor_as_sky

    def test_uses_rotor_position_not_rotor_target(self):
        # The readout's whole purpose is showing sky target and rotor
        # axis position as distinct things - it must display where the
        # rotor actually is, not what was computed for it.
        text = readout_text(
            sample(
                rotor_target=Position(180.0, 45.0),
                rotor_position=Position(178.5, 43.0),
            ),
            target_name="AO-91",
        )
        assert "178.5" in text.rotor_axis
        assert "180.0" not in text.rotor_axis
