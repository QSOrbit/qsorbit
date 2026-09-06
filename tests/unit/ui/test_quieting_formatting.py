"""Unit tests for the live quieting panel's pure formatting helpers.

Deliberately headless: this file imports qsorbit.ui.quieting_formatting
only, never qsorbit.ui.quieting_widget, so nothing here needs PySide6
installed. See that module's docstring for why the split exists.
"""

from __future__ import annotations

from qsorbit.ui.quieting_formatting import (
    AWAITING_FIRST_MEASUREMENT_LABEL,
    HEARD_LABEL,
    NO_SQUELCH_LABEL,
    QuietingText,
    quieting_text,
)


class TestNoSquelch:
    def test_is_open_none_means_no_squelch_regardless_of_quieting_db(self):
        # live_squelch_open is None only when there is no squelch at all
        # (see ReceiveSession's own docs) - quieting_db being anything
        # else here would be a caller bug, not a state this function
        # should try to make sense of differently.
        result = quieting_text(None, None)
        assert result.quieting_label == NO_SQUELCH_LABEL
        assert result.gate_label == "-"
        assert result.bar_fraction == 0.0

    def test_returns_a_quieting_text(self):
        assert isinstance(quieting_text(None, None), QuietingText)


class TestAwaitingFirstMeasurement:
    def test_a_squelch_with_no_reading_yet_is_distinct_from_no_squelch(self):
        # NoiseSquelch.is_open starts False before update() has ever
        # run, so a squelch that simply hasn't measured a block yet
        # reports False, not None - this must not be mistaken for "no
        # squelch running" (None), which is a different fact.
        result = quieting_text(None, False)
        assert result.quieting_label == AWAITING_FIRST_MEASUREMENT_LABEL
        assert result.quieting_label != NO_SQUELCH_LABEL
        assert result.gate_label == "closed"
        assert result.bar_fraction == 0.0

    def test_reports_the_gate_state_even_before_a_measurement(self):
        # Contrived (is_open would not be True before update() has run
        # at least once in practice), but the function should still say
        # what it's given rather than assume.
        result = quieting_text(None, True)
        assert result.gate_label == "open"
        assert result.bar_fraction == 0.0


class TestAMeasuredReading:
    def test_formats_the_db_value_to_one_decimal(self):
        result = quieting_text(12.34, True)
        assert result.quieting_label == "12.3 dB quieting"

    def test_open_gate_reads_open(self):
        assert quieting_text(10.0, True).gate_label == "open"

    def test_closed_gate_reads_closed(self):
        assert quieting_text(-1.0, False).gate_label == "closed"

    def test_a_reading_at_the_floor_gives_an_empty_bar(self):
        result = quieting_text(-5.0, False, floor_db=-5.0, ceiling_db=20.0)
        assert result.bar_fraction == 0.0

    def test_a_reading_at_the_ceiling_gives_a_full_bar(self):
        result = quieting_text(20.0, True, floor_db=-5.0, ceiling_db=20.0)
        assert result.bar_fraction == 1.0

    def test_a_reading_halfway_gives_a_half_full_bar(self):
        result = quieting_text(7.5, True, floor_db=-5.0, ceiling_db=20.0)
        assert result.bar_fraction == 0.5

    def test_a_reading_below_the_floor_clamps_to_zero_not_negative(self):
        result = quieting_text(-20.0, False, floor_db=-5.0, ceiling_db=20.0)
        assert result.bar_fraction == 0.0

    def test_a_reading_above_the_ceiling_clamps_to_one_not_over(self):
        # Real signals can measure well above a display ceiling chosen
        # for a "solid signal" idiom - the bar should peg full, not
        # overflow or wrap.
        result = quieting_text(55.0, True, floor_db=-5.0, ceiling_db=20.0)
        assert result.bar_fraction == 1.0

    def test_default_floor_and_ceiling_are_used_when_not_given(self):
        with_defaults = quieting_text(7.5, True)
        with_explicit = quieting_text(7.5, True, floor_db=-5.0, ceiling_db=20.0)
        assert with_defaults.bar_fraction == with_explicit.bar_fraction


class TestTheEarMarker:
    """Which branch is actually reaching the speaker.

    Qt-free like everything else in this module, which matters more than
    usual here: the widget tests next door need libEGL and cannot run in
    every environment, so the rules worth arguing about live where they
    always run.
    """

    def test_a_single_branch_station_is_not_marked(self):
        # None means "no choice is being made". Marking the only panel
        # on screen would imply one, and there is nothing it could be
        # distinguished from.
        assert quieting_text(12.0, True).ear_label == ""
        assert quieting_text(12.0, True, listening=None).ear_label == ""

    def test_the_listening_branch_is_marked(self):
        assert quieting_text(12.0, True, listening=True).ear_label == HEARD_LABEL

    def test_a_branch_that_is_not_heard_is_blank(self):
        assert quieting_text(12.0, True, listening=False).ear_label == ""

    def test_the_marker_survives_having_no_squelch(self):
        # A branch can hold the speaker with no squelch attached at all.
        text = quieting_text(None, None, listening=True)

        assert text.quieting_label == NO_SQUELCH_LABEL
        assert text.ear_label == HEARD_LABEL

    def test_the_marker_survives_having_no_measurement_yet(self):
        # And it holds the speaker for the first block of every run,
        # before anything has been measured -- which is exactly when a
        # marker applied only to the measured branch would vanish.
        text = quieting_text(None, False, listening=True)

        assert text.quieting_label == AWAITING_FIRST_MEASUREMENT_LABEL
        assert text.ear_label == HEARD_LABEL

    def test_the_marker_does_not_disturb_anything_else(self):
        # The reading, the gate and the bar are what they were before
        # this parameter existed.
        without = quieting_text(12.0, True)
        with_ear = quieting_text(12.0, True, listening=True)

        assert with_ear.quieting_label == without.quieting_label
        assert with_ear.gate_label == without.gate_label
        assert with_ear.bar_fraction == without.bar_fraction
