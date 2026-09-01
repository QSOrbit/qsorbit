"""Unit tests for stalled-axis detection.

The detection rule is tested here without a target, an observer, a clock
or a rotor -- which is the reason :class:`StallDetector` is a separate
object from the tracking loop rather than three fields inside it.
"""

import pytest

from qsorbit.core.rotor import Position
from qsorbit.core.stall_guard import (
    DEFAULT_STALL_TICKS,
    POSITION_REPORT_RESOLUTION_DEG,
    StallDetector,
    StallGuard,
)


def pos(azimuth: float, elevation: float = 0.0) -> Position:
    return Position(azimuth=azimuth, elevation=elevation)


def drive(detector: StallDetector, pairs) -> list[bool]:
    """Feed (commanded, reported) pairs, collecting the verdict each tick."""
    return [detector.observe(commanded, reported) for commanded, reported in pairs]


class TestDetection:
    def test_a_jammed_axis_is_declared_after_the_guard_s_tick_count(self):
        detector = StallDetector(StallGuard(ticks=3))
        # Setpoint walks away; the axis reads the same every tick.
        verdicts = drive(detector, [(pos(float(n)), pos(0.0)) for n in range(1, 6)])
        assert verdicts == [False, False, False, True, True]
        assert detector.is_stalled
        assert detector.events == 1

    def test_a_following_axis_never_stalls(self):
        detector = StallDetector(StallGuard(ticks=3))
        verdicts = drive(detector, [(pos(float(n)), pos(n - 0.5)) for n in range(1, 20)])
        assert not any(verdicts)
        assert detector.events == 0

    def test_a_static_setpoint_never_stalls(self):
        # The loop is inside its deadband, or the target is below the
        # horizon. Nothing is being commanded, so nothing is failing to
        # follow -- this is the case the acceptance-window formulation
        # of the test would have got wrong.
        detector = StallDetector(StallGuard(ticks=3))
        verdicts = drive(detector, [(pos(10.0), pos(7.5)) for _ in range(20)])
        assert not any(verdicts)

    def test_nothing_commanded_yet_never_stalls(self):
        detector = StallDetector(StallGuard(ticks=3))
        verdicts = drive(detector, [(None, pos(0.0)) for _ in range(20)])
        assert not any(verdicts)

    def test_movement_below_the_reported_resolution_is_not_movement(self):
        # The rotor answers AZ EL with one decimal place, so 0.04 of
        # drift is not a moving axis, it is the last digit.
        detector = StallDetector(StallGuard(ticks=3))
        verdicts = drive(detector, [(pos(float(n)), pos(0.01 * n)) for n in range(1, 6)])
        assert verdicts[-1] is True

    def test_either_axis_can_stall_the_detector(self):
        detector = StallDetector(StallGuard(ticks=3))
        # Azimuth follows perfectly; elevation is the jammed one.
        pairs = [
            (
                Position(azimuth=float(n), elevation=float(n)),
                Position(azimuth=float(n), elevation=0.0),
            )
            for n in range(1, 6)
        ]
        assert drive(detector, pairs)[-1] is True

    def test_the_stalled_axis_is_named(self):
        # "Elevation is not following" and "azimuth is not following"
        # point at different mechanical causes, and the operator is the
        # one who has to go and look.
        detector = StallDetector(StallGuard(ticks=3))
        pairs = [
            (
                Position(azimuth=float(n), elevation=float(n)),
                Position(azimuth=float(n), elevation=0.0),
            )
            for n in range(1, 6)
        ]
        drive(detector, pairs)
        assert detector.stalled_axes == ("elevation",)

    def test_a_healthy_track_names_no_axes(self):
        detector = StallDetector(StallGuard(ticks=3))
        drive(detector, [(pos(float(n)), pos(n - 0.5)) for n in range(1, 10)])
        assert detector.stalled_axes == ()

    def test_the_detector_can_actually_fire(self):
        # Canary. A guard nobody has watched refuse something is not a
        # guard -- Session 27 shipped a check that could not have failed.
        detector = StallDetector(StallGuard(ticks=1))
        fired = False
        for n in range(1, 5):
            if detector.observe(pos(float(n)), pos(0.0)):
                fired = True
        assert fired


class TestRecovery:
    def test_a_stall_ends_when_the_axis_moves_again(self):
        detector = StallDetector(StallGuard(ticks=2))
        drive(detector, [(pos(float(n)), pos(0.0)) for n in range(1, 5)])
        assert detector.is_stalled

        assert detector.observe(pos(4.0), pos(3.0)) is False
        assert not detector.is_stalled

    def test_a_stall_does_not_end_just_because_the_setpoint_stopped(self):
        # The asymmetry that matters: once stalled the loop freezes the
        # setpoint, so "the setpoint advanced" can never be true again.
        # A detector reusing the detection test in both directions would
        # declare recovery on the very next tick, with the axis still
        # jammed.
        detector = StallDetector(StallGuard(ticks=2))
        drive(detector, [(pos(float(n)), pos(0.0)) for n in range(1, 5)])
        assert detector.is_stalled

        frozen = [detector.observe(pos(4.0), pos(0.0)) for _ in range(20)]
        assert all(frozen)
        assert detector.is_stalled

    def test_recovery_needs_real_movement_not_the_last_digit(self):
        detector = StallDetector(StallGuard(ticks=2))
        drive(detector, [(pos(float(n)), pos(0.0)) for n in range(1, 5)])
        assert detector.observe(pos(4.0), pos(0.05)) is True
        assert detector.is_stalled

    def test_both_axes_must_free_before_the_setpoint_resumes(self):
        # A freed azimuth says nothing about an elevation still jammed.
        detector = StallDetector(StallGuard(ticks=2))
        drive(
            detector,
            [
                (Position(azimuth=float(n), elevation=float(n)), Position(0.0, 0.0))
                for n in range(1, 5)
            ],
        )
        assert detector.stalled_axes == ("azimuth", "elevation")

        # Azimuth frees; elevation does not.
        assert detector.observe(Position(4.0, 4.0), Position(3.0, 0.0)) is True
        assert detector.is_stalled

        # Now elevation frees too.
        assert detector.observe(Position(4.0, 4.0), Position(3.0, 3.0)) is False
        assert not detector.is_stalled

    def test_a_second_stall_is_counted_separately(self):
        detector = StallDetector(StallGuard(ticks=2))
        drive(detector, [(pos(float(n)), pos(0.0)) for n in range(1, 5)])
        detector.observe(pos(4.0), pos(3.0))  # frees
        drive(detector, [(pos(float(n)), pos(3.0)) for n in range(5, 9)])
        assert detector.is_stalled
        assert detector.events == 2

    def test_a_single_stall_counts_once_however_long_it_lasts(self):
        detector = StallDetector(StallGuard(ticks=2))
        drive(detector, [(pos(float(n)), pos(0.0)) for n in range(1, 30)])
        assert detector.events == 1


class TestStallGuard:
    def test_latency_is_ticks_times_interval(self):
        assert StallGuard(ticks=6).latency_s(0.5) == 3.0
        assert StallGuard(ticks=6).latency_s(1.0) == 6.0

    def test_the_default_policy_keeps_the_frozen_integral_under_breakaway(self):
        # The derivation the default exists for: windup over the
        # detection latency is 0.5 * Ki * rate * t^2, and it has to stay
        # below azimuth's ~17-count stiction breakaway or freezing the
        # integral is only a partial answer.
        latency = StallGuard().latency_s(0.5)
        windup_pwm = 0.5 * 1.0 * 1.0 * latency**2
        assert windup_pwm < 17.0

    @pytest.mark.parametrize("ticks", [0, -1])
    def test_a_meaningless_tick_count_is_refused(self, ticks):
        with pytest.raises(ValueError, match="ticks must be at least 1"):
            StallGuard(ticks=ticks)

    @pytest.mark.parametrize("resolution", [0.0, -0.1])
    def test_a_non_positive_resolution_is_refused(self, resolution):
        with pytest.raises(ValueError, match="resolution_deg must be positive"):
            StallGuard(resolution_deg=resolution)

    def test_the_resolution_default_is_what_the_rotor_reports(self):
        assert POSITION_REPORT_RESOLUTION_DEG == 0.1
        assert StallGuard().resolution_deg == POSITION_REPORT_RESOLUTION_DEG

    def test_the_default_tick_count(self):
        assert DEFAULT_STALL_TICKS == 6
        assert StallGuard().ticks == DEFAULT_STALL_TICKS

    def test_guards_compare_by_value(self):
        assert StallGuard(ticks=4) == StallGuard(ticks=4)
        assert StallGuard(ticks=4) != StallGuard(ticks=5)
