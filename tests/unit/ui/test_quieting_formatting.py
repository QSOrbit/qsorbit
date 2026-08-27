"""Unit tests for the live quieting panel's pure formatting helpers.

Deliberately headless: this file imports qsorbit.ui.quieting_formatting
only, never qsorbit.ui.quieting_widget, so nothing here needs PySide6
installed. See that module's docstring for why the split exists.
"""

from __future__ import annotations

from qsorbit.ui.quieting_formatting import (
    AWAITING_FIRST_MEASUREMENT_LABEL,
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
