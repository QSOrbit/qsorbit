"""Unit tests for stalled-axis detection.

The detection rule is tested here without a target, an observer, a clock
or a rotor -- which is the reason :class:`StallDetector` is a separate
object from the tracking loop rather than three fields inside it.

**The free-play tests are the point of this file.** Azimuth on this
rotator has 2.95 deg of measured mechanical slop, twenty-nine times the
rotor's reporting resolution, so "the axis moved" is not evidence that
it went anywhere. Every test below that oscillates a reported position
without net displacement is that measurement, written as code.
"""

import pytest

from qsorbit.core.rotor import Position
from qsorbit.core.stall_guard import (
    DEFAULT_FREE_PLAY_DEG,
    DEFAULT_STALL_WINDOW_S,
    POSITION_REPORT_RESOLUTION_DEG,
    StallDetector,
    StallGuard,
)

# A compact policy for most tests: three ticks at a 1 s cadence, and a
# degree of slop. The real defaults get their own class below.
TIGHT = StallGuard(window_s=3.0, free_play_deg=1.0)


def pos(azimuth: float, elevation: float = 0.0) -> Position:
    return Position(azimuth=azimuth, elevation=elevation)


def drive(detector: StallDetector, pairs) -> list[bool]:
    """Feed (commanded, reported) pairs, collecting the verdict each tick."""
    return [detector.observe(commanded, reported) for commanded, reported in pairs]


def arm(d: StallDetector, axis: str) -> None:
    """Feed following motion until the detector arms that axis.

    The detector will not judge an axis it has never seen follow, so
    almost every test below has to get past that first. Arming clears
    the history, so whatever a test drives afterwards is judged on a
    window made only of its own data.
    """
    for n in range(1, 60):
        commanded = pos(2.0 * n) if axis == "azimuth" else Position(0.0, 2.0 * n)
        reported = pos(2.0 * n - 0.3) if axis == "azimuth" else Position(0.0, 2.0 * n - 0.3)
        d.observe(commanded, reported)
        if axis in d.armed_axes:
            return
    raise AssertionError(f"detector never armed {axis}")


def detector(
    guard: StallGuard = TIGHT, interval_s: float = 1.0, *, armed: bool = True
) -> StallDetector:
    d = StallDetector(guard, interval_s)
    if armed:
        arm(d, "azimuth")
        arm(d, "elevation")
    return d


class TestDetection:
    def test_a_jammed_axis_is_declared_after_the_window(self):
        # Setpoint walks away 2 deg a tick, past the 1 deg slop; the
        # axis reads the same every tick.
        verdicts = drive(detector(), [(pos(2.0 * n), pos(0.0)) for n in range(1, 6)])
        assert verdicts == [False, False, False, True, True]

    def test_a_following_axis_never_stalls(self):
        verdicts = drive(detector(), [(pos(2.0 * n), pos(2.0 * n - 0.5)) for n in range(1, 20)])
        assert not any(verdicts)

    def test_free_play_does_not_mask_a_stall(self):
        # THE test this rewrite exists for. The axis wanders across the
        # full 1 deg of declared slop every tick -- ten times the rotor's
        # reporting resolution -- and arrives exactly nowhere. The
        # previous rule, "did it move at all", called this healthy.
        wander = [0.0, 0.9, -0.1, 0.8, 0.0, 0.9, -0.1, 0.8, 0.0, 0.9]
        pairs = [(pos(2.0 * n), pos(wander[n - 1])) for n in range(1, 11)]
        assert drive(detector(), pairs)[-1] is True

    def test_real_progress_with_wobble_on_top_is_not_a_stall(self):
        # Wind sway riding on a healthy axis: it wanders, but it also
        # gets where it was going.
        wobble = [0.0, 0.6, -0.4, 0.5, -0.3, 0.4, 0.0, 0.5, -0.2, 0.3]
        pairs = [(pos(2.0 * n), pos(2.0 * n - 1.0 + wobble[n - 1])) for n in range(1, 11)]
        assert not any(drive(detector(), pairs))

    def test_an_axis_going_backwards_is_stalled(self):
        # Not following is not following, whichever way it drifts.
        pairs = [(pos(2.0 * n), pos(-0.5 * n)) for n in range(1, 6)]
        assert drive(detector(), pairs)[-1] is True

    def test_a_setpoint_moving_less_than_the_slop_is_not_judged(self):
        # Nothing has been asked for beyond what the backlash could
        # absorb, so there is no failure to detect.
        pairs = [(pos(0.2 * n), pos(0.0)) for n in range(1, 30)]
        assert not any(drive(detector(), pairs))

    def test_a_static_setpoint_never_stalls(self):
        # Inside the deadband, or below the horizon. Nothing is being
        # commanded, so nothing is failing to follow.
        assert not any(drive(detector(), [(pos(10.0), pos(7.5)) for _ in range(20)]))

    def test_nothing_commanded_yet_never_stalls(self):
        assert not any(drive(detector(), [(None, pos(0.0)) for _ in range(20)]))

    def test_either_axis_can_stall_the_detector(self):
        pairs = [(Position(2.0 * n, 2.0 * n), Position(2.0 * n, 0.0)) for n in range(1, 6)]
        assert drive(detector(), pairs)[-1] is True

    def test_the_stalled_axis_is_named(self):
        pairs = [(Position(2.0 * n, 2.0 * n), Position(2.0 * n, 0.0)) for n in range(1, 6)]
        d = detector()
        drive(d, pairs)
        assert d.stalled_axes == ("elevation",)

    def test_a_healthy_track_names_no_axes(self):
        d = detector()
        drive(d, [(pos(2.0 * n), pos(2.0 * n - 0.5)) for n in range(1, 10)])
        assert d.stalled_axes == ()

    def test_the_detector_can_actually_fire(self):
        # Canary. A guard nobody has watched refuse something is not a
        # guard -- Session 27 shipped a check that could not have failed.
        d = StallDetector(StallGuard(window_s=1.0, free_play_deg=1.0), 1.0)
        arm(d, "azimuth")
        assert any(d.observe(pos(100.0 + 2.0 * n), pos(100.0)) for n in range(1, 6))


class TestRecovery:
    def stall(self, d: StallDetector) -> None:
        drive(d, [(pos(2.0 * n), pos(0.0)) for n in range(1, 6)])
        assert d.is_stalled

    def test_a_stall_ends_when_the_axis_really_moves(self):
        d = detector()
        self.stall(d)
        assert d.observe(pos(10.0), pos(4.0)) is False
        assert not d.is_stalled

    def test_movement_inside_the_slop_is_not_recovery(self):
        # The boom rocking in its own backlash is not the axis freeing.
        d = detector()
        self.stall(d)
        assert d.observe(pos(10.0), pos(0.9)) is True
        assert d.is_stalled

    def test_a_stall_does_not_end_just_because_the_setpoint_stopped(self):
        # Once stalled the loop freezes the setpoint, so "the setpoint
        # advanced" can never be true again. A detector reusing its
        # detection test in both directions would declare recovery on
        # the next tick with the axis still jammed.
        d = detector()
        self.stall(d)
        assert all(d.observe(pos(10.0), pos(0.0)) for _ in range(20))

    def test_both_axes_must_free_before_the_setpoint_resumes(self):
        d = detector()
        drive(d, [(Position(2.0 * n, 2.0 * n), Position(0.0, 0.0)) for n in range(1, 6)])
        assert d.stalled_axes == ("azimuth", "elevation")

        assert d.observe(Position(10.0, 10.0), Position(4.0, 0.0)) is True
        assert d.observe(Position(10.0, 10.0), Position(4.0, 4.0)) is False

    def test_a_second_stall_is_counted_separately(self):
        d = detector()
        self.stall(d)
        d.observe(pos(10.0), pos(4.0))
        drive(d, [(pos(10.0 + 2.0 * n), pos(4.0)) for n in range(1, 6)])
        assert d.is_stalled
        assert d.events == 2

    def test_a_single_stall_counts_once_however_long_it_lasts(self):
        d = detector()
        drive(d, [(pos(2.0 * n), pos(0.0)) for n in range(1, 30)])
        assert d.events == 1


class TestWindowIsADuration:
    """The same policy must be the same evidence at any cadence."""

    def test_ticks_scale_with_the_cadence(self):
        guard = StallGuard(window_s=6.0)
        assert guard.ticks_for(1.0) == 6
        assert guard.ticks_for(0.5) == 12
        assert guard.ticks_for(2.0) == 3

    def test_latency_is_the_window_however_it_is_ticked(self):
        guard = StallGuard(window_s=6.0)
        assert guard.latency_s(1.0) == 6.0
        assert guard.latency_s(0.5) == 6.0

    def test_a_cadence_slower_than_the_window_still_watches_one_tick(self):
        assert StallGuard(window_s=3.0).ticks_for(10.0) == 1

    def test_latency_rounds_up_to_whole_ticks(self):
        # 5 s of window at a 2 s tick is three ticks, so detection
        # really takes 6 s -- and it is 6 that the gain clamp has to be
        # derived from, not 5.
        guard = StallGuard(window_s=5.0)
        assert guard.ticks_for(2.0) == 3
        assert guard.latency_s(2.0) == 6.0

    def test_the_detector_sizes_itself_from_the_cadence(self):
        assert StallDetector(StallGuard(window_s=6.0), 0.5).ticks == 12
        assert StallDetector(StallGuard(window_s=6.0), 1.0).ticks == 6

    def test_the_same_policy_detects_at_the_same_time_either_cadence(self):
        # Six seconds of a 1 deg/s target either way.
        slow = StallDetector(StallGuard(window_s=6.0, free_play_deg=1.0), 1.0)
        fast = StallDetector(StallGuard(window_s=6.0, free_play_deg=1.0), 0.5)
        arm(slow, "azimuth")
        arm(fast, "azimuth")
        slow_verdicts = drive(slow, [(pos(200.0 + 1.0 * n), pos(200.0)) for n in range(1, 12)])
        fast_verdicts = drive(fast, [(pos(200.0 + 0.5 * n), pos(200.0)) for n in range(1, 24)])
        assert slow_verdicts.index(True) * 1.0 == pytest.approx(
            fast_verdicts.index(True) * 0.5, abs=0.5
        )


class TestStallGuard:
    def test_the_default_window_beats_the_measured_free_play(self):
        # At a satellite's ~1 deg/s the setpoint advances 6 deg in the
        # window, against 2.95 deg of measured slop. If that ratio ever
        # drops below about 2, a jammed axis wandering in its own
        # backlash could net enough apparent progress to pass.
        advance = 1.0 * DEFAULT_STALL_WINDOW_S
        assert advance / 2.95 > 2.0

    def test_the_default_window_beats_the_proportional_runaway(self):
        # Kp 8 reaches MAX_PWM 180 at 22.5 deg of error, ~22 s at
        # 1 deg/s. Detection has to be comfortably inside that.
        assert DEFAULT_STALL_WINDOW_S < 22.5 / 3.0

    def test_the_free_play_default_is_the_measured_figure_rounded_up(self):
        assert DEFAULT_FREE_PLAY_DEG >= 2.95

    @pytest.mark.parametrize("window", [0.0, -1.0])
    def test_a_meaningless_window_is_refused(self, window):
        with pytest.raises(ValueError, match="window_s must be positive"):
            StallGuard(window_s=window)

    def test_free_play_below_the_reporting_resolution_is_refused(self):
        with pytest.raises(ValueError, match="at least the reporting resolution"):
            StallGuard(free_play_deg=POSITION_REPORT_RESOLUTION_DEG / 2.0)

    def test_a_non_positive_interval_is_refused(self):
        with pytest.raises(ValueError, match="interval_s must be positive"):
            StallGuard().ticks_for(0.0)

    def test_guards_compare_by_value(self):
        assert StallGuard(window_s=4.0) == StallGuard(window_s=4.0)
        assert StallGuard(window_s=4.0) != StallGuard(window_s=5.0)


class TestTheBenchFalsePositive:
    """The failure a real pass found that 1,871 unit tests did not.

    2026-09-02, PO-101, an 85 deg pass. The detector declared a stalled
    elevation axis **53 seconds before anything was disconnected**, on a
    rotor that was tracking correctly, and it fired 48 s before TCA --
    which is where elevation's rate first advances the setpoint past the
    3 deg gate. **A detector that trips the instant its gate opens is
    tripping on the gate**, not detecting anything.

    The cause was the same threshold on both sides of the test: at the
    shipped cadence the commanded position moves in ~3 deg jumps, so a
    window holding exactly one command has an advance a little over 3
    while a normally-lagging axis nets a little under it. Both true at
    once. The tests above never caught it because they exercise a
    perfectly following axis and a completely jammed one; this lives in
    the band between them.
    """

    def test_a_lagging_axis_at_the_gate_is_not_a_stall(self):
        # Tonight's arithmetic: one command lands in the window, the
        # axis follows with its normal stiction lag and nets slightly
        # under the threshold the advance is slightly over.
        guard = StallGuard(window_s=3.0, free_play_deg=3.0)
        d = StallDetector(guard, 1.0)
        arm(d, "azimuth")
        commanded = [0.0, 0.0, 3.2, 3.2, 3.2, 3.2]
        reported = [0.0, 0.4, 1.1, 2.0, 2.6, 2.9]
        verdicts = [d.observe(pos(c), pos(r)) for c, r in zip(commanded, reported, strict=True)]
        assert not any(verdicts), "healthy axis lagging at the gate must not stall"

    def test_the_same_advance_with_no_movement_is_a_stall(self):
        # The control for the test above: identical setpoint, axis stuck.
        guard = StallGuard(window_s=3.0, free_play_deg=3.0)
        d = StallDetector(guard, 1.0)
        arm(d, "azimuth")
        commanded = [0.0, 0.0, 3.2, 3.2, 3.2, 3.2]
        verdicts = [d.observe(pos(c), pos(0.0)) for c in commanded]
        assert verdicts[-1] is True

    def test_a_constant_lag_never_accumulates_into_a_stall(self):
        # The property the old rule lacked. Steady tracking carries a
        # fixed lag; it must not matter how long the pass runs.
        guard = StallGuard(window_s=6.0, free_play_deg=3.0)
        d = StallDetector(guard, 1.0)
        arm(d, "azimuth")
        verdicts = [d.observe(pos(0.7 * n), pos(0.7 * n - 2.1)) for n in range(1, 200)]
        assert not any(verdicts)

    def test_falling_behind_by_more_than_the_slop_is_a_stall(self):
        # Following, but shedding ground faster than the backlash can
        # explain -- a slipping drive rather than a hard jam.
        guard = StallGuard(window_s=6.0, free_play_deg=3.0)
        d = StallDetector(guard, 1.0)
        arm(d, "azimuth")
        verdicts = [d.observe(pos(1.0 * n), pos(0.2 * n)) for n in range(1, 20)]
        assert any(verdicts)


class TestOperatorMessage:
    def test_the_stall_report_is_ascii(self):
        # It reaches a Windows console, where the code page turned an em
        # dash into a stray "u" on the bench. Verify what reaches the
        # screen, not what the source looks like.
        import io as _io
        from contextlib import redirect_stdout

        from qsorbit.__main__ import _report_stall

        buf = _io.StringIO()
        with redirect_stdout(buf):
            _report_stall(("elevation",))
        buf.getvalue().encode("ascii")


# ---------------------------------------------------------------------------
# Replayed bench logs, 2026-09-02
# ---------------------------------------------------------------------------

# (commanded_az, commanded_el, reported_az, reported_el) per tick, taken
# straight off the console. `observe` sees the PREVIOUS tick's commanded
# position, which is what `replay` below reproduces.

_RUN_HEALTHY_START_AZ = [
    (1.50, 5.00, 1.60, 1.50),
    (1.50, 5.00, 1.60, 2.50),
    (1.50, 5.00, 1.60, 3.30),
    (4.50, 5.00, 1.60, 3.50),
    (4.50, 5.00, 2.10, 3.50),
    (4.50, 5.00, 2.90, 3.50),
    (7.50, 5.00, 3.00, 3.50),
    (7.50, 5.00, 4.10, 3.50),
    (7.50, 5.00, 5.30, 3.50),
    (7.50, 5.00, 6.20, 3.50),
    (7.50, 5.00, 6.20, 3.50),
    (7.50, 5.00, 6.30, 3.50),
]

_RUN_HEALTHY_START_EL = [
    (96.20, 3.50, 96.20, 3.50),
    (96.20, 3.50, 96.20, 3.50),
    (96.20, 3.50, 96.20, 3.50),
    (96.20, 6.50, 96.20, 3.50),
    (96.20, 6.50, 96.20, 4.40),
    (96.20, 6.50, 96.20, 4.50),
    (96.20, 9.50, 96.20, 4.50),
    (96.20, 9.50, 96.20, 6.40),
    (96.20, 9.50, 96.20, 7.40),
    (96.20, 9.50, 96.20, 7.40),
    (96.20, 9.50, 96.20, 7.40),
]

_RUN_REAL_STALL = [
    (1.60, 5.00, 1.50, 1.60),
    (1.60, 5.00, 1.60, 2.60),
    (1.60, 5.00, 1.60, 3.40),
    (4.60, 5.00, 1.60, 3.50),
    (4.60, 5.00, 1.70, 3.50),
    (4.60, 5.00, 3.10, 3.50),
    (7.60, 5.00, 3.00, 3.50),
    (7.60, 5.00, 5.00, 3.50),
    (7.60, 5.00, 5.60, 3.50),
    (10.60, 5.00, 6.20, 3.50),
    (10.60, 5.00, 7.80, 3.50),
    (10.60, 5.00, 8.40, 3.50),
    (13.60, 5.00, 9.50, 3.50),
    (13.60, 5.00, 10.50, 3.50),
    (13.60, 5.00, 11.30, 3.50),
    (16.60, 5.00, 12.00, 3.50),
    (16.60, 5.00, 13.70, 3.50),
    (16.60, 5.00, 14.90, 3.50),
    (19.60, 5.00, 14.90, 3.50),
    (19.60, 5.00, 17.10, 3.50),
    (19.60, 5.00, 17.70, 3.50),
    (22.60, 5.00, 18.20, 3.50),
    (22.60, 5.00, 19.60, 3.50),
    (22.60, 5.00, 20.50, 3.50),
    (25.60, 5.00, 21.40, 3.50),
    (25.60, 5.00, 22.50, 3.50),
    (25.60, 5.00, 23.50, 3.50),
    (28.60, 5.00, 24.30, 3.50),
    (28.60, 5.00, 25.70, 3.50),
    (28.60, 5.00, 26.70, 3.50),
    (31.60, 5.00, 26.80, 3.50),
    (31.60, 5.00, 28.90, 3.50),
    (31.60, 5.00, 29.90, 3.50),
    (34.60, 5.00, 30.20, 3.50),
    (34.60, 5.00, 31.80, 3.50),
    (34.60, 5.00, 32.30, 3.50),
    (37.60, 5.00, 32.00, 3.50),
    (37.60, 5.00, 32.10, 3.50),
    (37.60, 5.00, 32.20, 3.50),
    (37.60, 5.00, 32.30, 3.50),
    (37.60, 5.00, 32.40, 3.50),
    (37.60, 5.00, 32.40, 3.50),
]


def replay(rows) -> tuple[StallDetector, int | None]:
    """Feed a logged run through a detector on stock defaults."""
    d = StallDetector(StallGuard(), 1.0)
    last_commanded: Position | None = None
    fired_at: int | None = None
    for tick, (c_az, c_el, r_az, r_el) in enumerate(rows):
        if d.observe(last_commanded, Position(r_az, r_el)) and fired_at is None:
            fired_at = tick
        last_commanded = Position(c_az, c_el)
    return d, fired_at


class TestAgainstBenchLogs:
    """Real runs from 2026-09-02, replayed.

    Two of the three bench runs that evening false-positived within eight
    seconds of starting, on axes that were visibly accelerating. These
    are those runs, not a reconstruction of them -- the numbers are the
    console output. Any future change to the detection rule has to keep
    all three of these verdicts.
    """

    def test_a_standing_start_on_azimuth_is_not_a_stall(self):
        # Ran 01:07:47. Declared a stall at tick 7 while the axis was
        # accelerating through 1.60 -> 4.10 -> 5.30 -> 6.20.
        d, fired = replay(_RUN_HEALTHY_START_AZ)
        assert fired is None, "healthy azimuth acquisition must not stall"

    def test_a_standing_start_on_elevation_is_not_a_stall(self):
        # Ran 01:10:58. Same shape, and elevation breaks stiction later
        # than azimuth because it carries the boom.
        d, fired = replay(_RUN_HEALTHY_START_EL)
        assert fired is None, "healthy elevation acquisition must not stall"

    def test_the_real_stall_is_still_caught(self):
        # Ran 00:58:53. Motor leads cut around tick 33; the axis settles
        # at 32.4 and stops. The control for the two above -- a rule that
        # silenced them by going deaf would pass those and fail this.
        d, fired = replay(_RUN_REAL_STALL)
        assert fired is not None
        assert 34 <= fired <= 40, f"expected detection in the mid-to-late 30s, got {fired}"
        assert d.stalled_axes == ("azimuth",)
        assert d.events == 1

    def test_elevation_is_not_blamed_for_an_azimuth_stall(self):
        # Elevation sat at 3.50 throughout, commanded and reported. A
        # static axis is not a failing one.
        d, _ = replay(_RUN_REAL_STALL)
        assert "elevation" not in d.stalled_axes
