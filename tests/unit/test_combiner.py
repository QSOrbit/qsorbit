"""Tests for branch selection.

The selector is a pure function of (current choice, readings), which is
the whole reason it was written that way: every rule below is checked
without a thread, a clock, or a radio. Staleness deliberately lives
outside it -- a ``None`` reading is "no opinion" and the caller decides
what makes a reading stale -- so these tests can state the rules
directly instead of arranging for a device to stop.
"""

from __future__ import annotations

import threading

import pytest

from qsorbit.core.combiner import DEFAULT_MARGIN_DB, BranchSelector, SelectorStats


class TestMargin:
    def test_the_default_is_the_measured_one(self):
        # Measured 2026-09-06, not assumed. See the module docstring for
        # the derivation; a change here should come with new data.
        assert DEFAULT_MARGIN_DB == 3.0
        assert BranchSelector().margin_db == 3.0

    def test_a_negative_margin_is_refused(self):
        with pytest.raises(ValueError, match="must not be negative"):
            BranchSelector(margin_db=-0.1)

    def test_zero_is_allowed_and_means_no_hysteresis(self):
        # Deliberately legal: it is the control that demonstrates why
        # hysteresis is needed, so the acceptance pass can run it.
        selector = BranchSelector(margin_db=0.0)

        assert selector.choose("A", {"A": 5.0, "B": 5.01}) == "B"


class TestSwitching:
    def test_a_challenger_must_clear_the_margin(self):
        selector = BranchSelector(margin_db=3.0)

        assert selector.choose("A", {"A": 5.0, "B": 7.9}) == "A"

    def test_clearing_the_margin_moves_the_ear(self):
        selector = BranchSelector(margin_db=3.0)

        assert selector.choose("A", {"A": 5.0, "B": 8.1}) == "B"

    def test_exactly_the_margin_does_not_switch(self):
        # Strictly greater, so the boundary belongs to the incumbent.
        # An inclusive test would switch on a difference the measurement
        # cannot resolve from zero.
        selector = BranchSelector(margin_db=3.0)

        assert selector.choose("A", {"A": 5.0, "B": 8.0}) == "A"

    def test_equal_readings_never_switch(self):
        selector = BranchSelector(margin_db=3.0)

        assert selector.choose("A", {"A": 5.0, "B": 5.0}) == "A"

    def test_hysteresis_holds_in_both_directions(self):
        # The property that stops chatter: once B has the ear, A has to
        # clear the same margin to take it back. A rule that only
        # applied the margin one way would let the pair oscillate.
        selector = BranchSelector(margin_db=3.0)

        assert selector.choose("A", {"A": 5.0, "B": 9.0}) == "B"
        assert selector.choose("B", {"A": 7.0, "B": 9.0}) == "B"
        assert selector.choose("B", {"A": 12.5, "B": 9.0}) == "A"

    def test_a_worse_branch_never_takes_the_ear(self):
        selector = BranchSelector(margin_db=3.0)

        assert selector.choose("A", {"A": 12.0, "B": 2.0}) == "A"

    def test_the_best_of_several_wins(self):
        selector = BranchSelector(margin_db=3.0)

        assert selector.choose("A", {"A": 1.0, "B": 5.0, "C": 9.0}) == "C"

    def test_one_branch_alone_keeps_the_ear(self):
        # A single-branch station runs through this code too.
        selector = BranchSelector(margin_db=3.0)

        assert selector.choose("A", {"A": 5.0}) == "A"


class TestStaleReadings:
    """``None`` is no opinion, never zero."""

    def test_a_stale_challenger_is_ignored(self):
        # If None were read as 0.0 it would look like a terrible branch
        # rather than an absent one -- harmless here, and the opposite
        # of harmless in the test below.
        selector = BranchSelector(margin_db=3.0)

        assert selector.choose("A", {"A": 5.0, "B": None}) == "A"

    def test_the_ear_leaves_a_stopped_branch_regardless_of_margin(self):
        # The rule that is easy to leave out. The margin exists to stop
        # switching on noise; applying it here would hold the speaker on
        # a radio that has died, which is the one case where refusing to
        # switch is the failure. Note B is far WORSE than A's last
        # reading -- and still correct to choose.
        selector = BranchSelector(margin_db=3.0)

        assert selector.choose("A", {"A": None, "B": 1.0}) == "B"

    def test_a_stopped_branch_hands_over_to_the_best_survivor(self):
        selector = BranchSelector(margin_db=3.0)

        assert selector.choose("A", {"A": None, "B": 1.0, "C": 4.0}) == "C"

    def test_everything_stale_keeps_the_current_branch(self):
        # Nothing better to do, and switching blind would only add a
        # discontinuity to audio that is already not arriving.
        selector = BranchSelector(margin_db=3.0)

        assert selector.choose("A", {"A": None, "B": None}) == "A"

    def test_an_empty_reading_set_keeps_the_current_branch(self):
        selector = BranchSelector(margin_db=3.0)

        assert selector.choose("A", {}) == "A"


class TestStats:
    def test_it_counts_decisions_and_switches(self):
        selector = BranchSelector(margin_db=3.0)
        selector.choose("A", {"A": 5.0, "B": 5.0})
        selector.choose("A", {"A": 5.0, "B": 9.0})
        selector.choose("B", {"A": 5.0, "B": 9.0})

        stats = selector.stats
        assert stats.evaluations == 3
        assert stats.switches == 1
        assert dict(stats.evaluations_by_branch) == {"A": 1, "B": 2}

    def test_it_echoes_the_margin_in_force(self):
        # The switch count means nothing without it, so a report has to
        # be self-contained.
        assert BranchSelector(margin_db=1.5).stats.margin_db == 1.5

    def test_describe_names_switches_and_the_margin(self):
        selector = BranchSelector(margin_db=3.0)
        selector.choose("A", {"A": 5.0, "B": 9.0})

        described = selector.stats.describe()
        assert "1 switch(es)" in described
        assert "3.0 dB" in described

    def test_describe_says_so_when_it_never_ran(self):
        # Distinct from "never switched", which is a working combiner on
        # a steady signal.
        assert "never ran" in BranchSelector().stats.describe()

    def test_a_branch_that_never_held_the_ear_is_absent_rather_than_zero(self):
        # Reported per branch rather than as a share of a total: "held
        # it for none of the run" and "was never a candidate" are
        # different facts.
        selector = BranchSelector(margin_db=3.0)
        selector.choose("A", {"A": 5.0, "B": 5.0})

        assert dict(selector.stats.evaluations_by_branch) == {"A": 1}

    def test_stats_are_a_snapshot_not_a_live_view(self):
        selector = BranchSelector(margin_db=3.0)
        selector.choose("A", {"A": 5.0})
        taken = selector.stats
        selector.choose("A", {"A": 5.0})

        assert taken.evaluations == 1
        assert selector.stats.evaluations == 2

    def test_it_counts_correctly_under_concurrent_decisions(self):
        # Every demodulating thread calls choose(), so the counters are
        # written from several threads at once.
        selector = BranchSelector(margin_db=3.0)
        start = threading.Barrier(2)

        def decide():
            start.wait()
            for _ in range(500):
                selector.choose("A", {"A": 5.0, "B": 1.0})

        threads = [threading.Thread(target=decide) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10.0)

        assert selector.stats.evaluations == 1000
        assert selector.stats.switches == 0


class TestSelectorStats:
    def test_describe_reports_the_share_of_each_branch(self):
        stats = SelectorStats(
            margin_db=3.0,
            evaluations=100,
            switches=2,
            evaluations_by_branch=(("A - Arrow V", 75), ("B - Arrow H", 25)),
        )

        described = stats.describe()
        assert "A - Arrow V 75.0%" in described
        assert "B - Arrow H 25.0%" in described
