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
from datetime import UTC, datetime, timedelta

import pytest

from qsorbit.core.combiner import (
    DEFAULT_MARGIN_DB,
    BranchReading,
    BranchSelector,
    SelectorStats,
    pair_simultaneous,
)


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


# ---------------------------------------------------------------------------
# Simultaneity: which readings choose() is even allowed to compare
# ---------------------------------------------------------------------------

#: A block period at the station's 2.048 Msps with 256 KiB blocks, and the
#: half-block window the session pairs on. Real numbers, so the RS-44
#: regression below is checked at the tolerance the live run used.
_BLOCK_S = 0.064
_TOL_S = _BLOCK_S * 0.5

_T0 = datetime(2026, 9, 6, 1, 40, 0, tzinfo=UTC)


def _at(offset_s: float) -> datetime:
    return _T0 + timedelta(seconds=offset_s)


class TestSimultaneousPairing:
    """The gate that keeps a skewed pair from ever reaching the margin.

    ``pair_simultaneous`` is a pure function of readings and their block
    times, tested here without a session, a thread, or a radio -- the
    same discipline as ``choose`` itself. Staleness has already been
    applied by the caller, so a ``None`` in means a dead branch.
    """

    def test_an_aligned_challenger_is_kept(self):
        latest = {
            "A": BranchReading(quieting_db=2.0, at=_at(0.001)),
            "B": BranchReading(quieting_db=-1.0, at=_at(0.000)),
        }
        readings = pair_simultaneous("B", latest, tolerance_s=_TOL_S)
        assert readings == {"A": 2.0, "B": -1.0}

    def test_a_skewed_challenger_is_withheld(self):
        # A one-block-behind challenger is not a comparison: its reading
        # describes a different moment than the incumbent's.
        latest = {
            "A": BranchReading(quieting_db=2.0, at=_at(_BLOCK_S)),
            "B": BranchReading(quieting_db=-1.0, at=_at(0.000)),
        }
        readings = pair_simultaneous("B", latest, tolerance_s=_TOL_S)
        assert readings == {"A": None, "B": -1.0}

    def test_the_incumbent_is_never_withheld_for_skew(self):
        # The incumbent defines the reference instant, so it is always
        # kept -- nulling it would make an alive branch look dead and
        # hand the ear away on choose()'s rule 2.
        latest = {
            "A": BranchReading(quieting_db=2.0, at=_at(_BLOCK_S)),
            "B": BranchReading(quieting_db=-1.0, at=_at(0.000)),
        }
        readings = pair_simultaneous("B", latest, tolerance_s=_TOL_S)
        assert readings["B"] == -1.0

    def test_a_stopped_incumbent_keeps_every_survivor(self):
        # Nothing left to be simultaneous with, and the handoff still has
        # to happen, so simultaneity does not apply.
        latest = {
            "A": None,
            "B": BranchReading(quieting_db=-1.0, at=_at(5.0)),
            "C": BranchReading(quieting_db=4.0, at=_at(0.0)),
        }
        readings = pair_simultaneous("A", latest, tolerance_s=_TOL_S)
        assert readings == {"A": None, "B": -1.0, "C": 4.0}

    def test_a_dead_branch_stays_none(self):
        latest = {
            "A": BranchReading(quieting_db=2.0, at=_at(0.0)),
            "B": None,
        }
        readings = pair_simultaneous("A", latest, tolerance_s=_TOL_S)
        assert readings == {"A": 2.0, "B": None}

    def test_exactly_at_the_window_still_counts(self):
        latest = {
            "A": BranchReading(quieting_db=2.0, at=_at(_TOL_S)),
            "B": BranchReading(quieting_db=-1.0, at=_at(0.0)),
        }
        readings = pair_simultaneous("B", latest, tolerance_s=_TOL_S)
        assert readings["A"] == 2.0


class TestRs44SkewRegression:
    """The 2026-09-06 acceptance pass, at the block pair that switched.

    Numbers are from ``e-accept-rs44-a.csv`` (RS-44, TCA ~t=381 s), not a
    reconstruction. The ear was on branch B. Branch A rose near TCA; at
    t=377.078 A read +2.19 dB. Its *simultaneous* B block (t=377.094,
    -0.64) is 2.83 dB away -- under the 3.0 dB margin. Its *one-block
    earlier* B block (t=377.030, -0.90) is 3.09 dB away -- over it. The
    run compared A against that earlier block and switched. It must not.
    """

    A = "A - Arrow V"
    B = "B - Arrow H"

    def test_the_skewed_pair_would_have_switched(self):
        # The control: fed the skewed difference directly, choose() does
        # switch. This is what the run did, and what the guard prevents.
        selector = BranchSelector(margin_db=3.0)
        assert selector.choose(self.B, {self.A: 2.19, self.B: -0.90}) == self.A

    def test_the_guard_withholds_the_skewed_block(self):
        # A's block at 377.078; B's latest is its 377.030 block, 48 ms
        # earlier -- past the 32 ms window, so A is no comparison yet.
        latest = {
            self.A: BranchReading(quieting_db=2.19, at=_at(377.078)),
            self.B: BranchReading(quieting_db=-0.90, at=_at(377.030)),
        }
        readings = pair_simultaneous(self.B, latest, tolerance_s=_TOL_S)
        assert readings == {self.A: None, self.B: -0.90}

        selector = BranchSelector(margin_db=3.0)
        assert selector.choose(self.B, readings) == self.B

    def test_the_simultaneous_block_still_does_not_switch(self):
        # One block later B produces its 377.094 block, 16 ms from A's --
        # inside the window, a real comparison, and 2.83 dB apart, so the
        # margin correctly holds. The guard does not merely defer the
        # switch; simultaneously there was never one to make.
        latest = {
            self.A: BranchReading(quieting_db=2.19, at=_at(377.078)),
            self.B: BranchReading(quieting_db=-0.64, at=_at(377.094)),
        }
        readings = pair_simultaneous(self.B, latest, tolerance_s=_TOL_S)
        assert readings == {self.A: 2.19, self.B: -0.64}

        selector = BranchSelector(margin_db=3.0)
        assert selector.choose(self.B, readings) == self.B
