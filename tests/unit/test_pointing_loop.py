"""Unit tests for the tracking loop, plus rotor_to_sky().

Kept apart from test_pointing.py, which validates the sky-to-rotor seam
against real orbital math and therefore imports skyfield. Nothing here
needs an ephemeris: the target is a stub reporting scripted sky
positions, the rotor is a MagicMock spec'd against the real class, and
the clock is fake. That keeps the loop's own behaviour — cadence,
deadband, refusals — separable from whether the astronomy is right.

rotor_to_sky() lives in the same module as the loop and needs no
ephemeris either, so its tests belong here for the same reason:
importing qsorbit.core.pointing at all pulls in qsorbit.core.rotor and
qsorbit.core.tracker, and only the latter needs skyfield.

The rotor mock is spec'd rather than free-form on purpose (the Chunk F
lesson): a scripted stub that isn't spec'd keeps passing against methods
the facade no longer has.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from qsorbit.core.geometry import AzEl
from qsorbit.core.pointing import (
    DEFAULT_TICK_INTERVAL_S,
    FIRMWARE_DEADZONE_DEG,
    AlignmentOffset,
    TickOutcome,
    TrackingLoop,
    TrackSample,
    TravelGuardError,
    rotor_to_sky,
)
from qsorbit.core.rotor import (
    AzimuthWrap,
    Position,
    PositionLimitError,
    Rotor,
    RotorCapabilities,
)
from qsorbit.core.tracker import ObserverLocation, TopocentricState
from qsorbit.core.tracking_profile import CadenceError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBSERVER = ObserverLocation(latitude=40.0, longitude=-83.0)

START = datetime(2026, 8, 21, 18, 0, tzinfo=UTC)


def capabilities(**overrides) -> RotorCapabilities:
    """Phil's rotator, as declared in his station config."""
    fields = {
        "azimuth_min_deg": 0.0,
        "azimuth_max_deg": 360.0,
        "elevation_min_deg": 0.0,
        "elevation_max_deg": 180.0,
        "azimuth_wrap": AzimuthWrap.EXTRA_ROTATION,
        "acceptance_window_deg": 2.5,
        "rs485_turnaround_s": 0.15,
        "firmware_version": "SatNOGS-v2.2.1",
    }
    fields.update(overrides)
    return RotorCapabilities(**fields)


def state(azimuth: float, elevation: float, *, range_km=1000.0, range_rate=-1.5):
    """A topocentric state at a given sky direction."""
    return TopocentricState(
        sky_position=AzEl(azimuth=azimuth, elevation=elevation),
        range_km=range_km,
        range_rate_km_s=range_rate,
    )


class StubTarget:
    """A Target reporting scripted sky positions, one per call.

    Satisfies the protocol structurally, exactly as a star or planet
    target would later. When the script runs out it repeats its last
    state, so a test can run more ticks than it bothered to script.

    Args:
        states: The states to report, in order.
    """

    def __init__(self, states: list[TopocentricState]) -> None:
        self.states = list(states)
        self.asked_at: list[datetime] = []

    @property
    def name(self) -> str:
        return "STUB TARGET"

    def topocentric_state(self, observer, time) -> TopocentricState:
        self.asked_at.append(time)
        index = min(len(self.asked_at) - 1, len(self.states) - 1)
        return self.states[index]


class FakeClock:
    """A clock whose sleep() advances both its monotonic and its wall time."""

    def __init__(self) -> None:
        self.elapsed = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.elapsed += seconds

    def advance(self, seconds: float) -> None:
        """Burn time without sleeping — what a slow tick does."""
        self.elapsed += seconds

    def monotonic(self) -> float:
        return self.elapsed

    def now(self) -> datetime:
        return START + timedelta(seconds=self.elapsed)


def make_loop(
    states: list[TopocentricState],
    *,
    reported: list[Position] | Position | None = None,
    caps: RotorCapabilities | None = None,
    **kwargs,
) -> tuple[TrackingLoop, MagicMock, FakeClock]:
    """Build a loop over a stub target, a spec'd rotor mock, and a fake clock.

    Args:
        states: What the target reports, tick by tick.
        reported: What the rotor reports for its own position — one
            Position for every tick, or a list consumed in order.
        caps: The rotor's declared capabilities.
    """
    rotor = MagicMock(spec=Rotor)
    rotor.capabilities = caps or capabilities()
    if isinstance(reported, list):
        rotor.read_position.side_effect = reported
    else:
        rotor.read_position.return_value = reported or Position(0.0, 0.0)

    clock = FakeClock()
    # The deadband used to default to the acceptance window inside
    # TrackingLoop. That coupling is gone (Chunk H), so the helper states
    # it instead: every test below was written against a 2.5 deg deadband
    # and still means what it meant, but the number is now visible.
    kwargs.setdefault("deadband_deg", rotor.capabilities.acceptance_window_deg)
    loop = TrackingLoop(
        StubTarget(states),
        OBSERVER,
        rotor,
        now=clock.now,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        **kwargs,
    )
    return loop, rotor, clock


def commanded_positions(rotor: MagicMock) -> list[Position]:
    """Every position actually sent to the rotor, in order."""
    return [call.args[0] for call in rotor.move_to.call_args_list]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_deadband_must_be_stated(self):
        # It used to default to the rotor's acceptance window. Session 32
        # measured a 0.25 deg deadband beating that 2.5 deg default on
        # every metric that matters, so how hard to drive the rotor is a
        # choice, not a fact about where it settles -- and a choice has
        # to be made rather than inherited.
        rotor = MagicMock(spec=Rotor)
        rotor.capabilities = capabilities()
        with pytest.raises(TypeError, match="deadband_deg"):
            TrackingLoop(StubTarget([state(180.0, 45.0)]), OBSERVER, rotor)

    def test_deadband_no_longer_follows_the_capability_record(self):
        # The window moves, the deadband does not: they are separate
        # concepts now and nothing should re-link them.
        loop, _, _ = make_loop(
            [state(180.0, 45.0)],
            caps=capabilities(acceptance_window_deg=3.0),
            deadband_deg=0.5,
        )
        assert loop.deadband_deg == 0.5

    def test_deadband_can_be_set(self):
        loop, _, _ = make_loop([state(180.0, 45.0)], deadband_deg=0.5)
        assert loop.deadband_deg == 0.5

    def test_a_knife_edge_cadence_is_refused(self):
        # deadband == rate x interval: which tick commands is decided by
        # timing jitter, and the step silently doubles. Measured on
        # hardware as 1.15, 2.00, 1.97, 1.95 where 1.0 was configured.
        with pytest.raises(CadenceError, match="knife edge"):
            make_loop([state(180.0, 45.0)], deadband_deg=1.0, interval_s=1.0)

    def test_the_validated_cadence_is_not_a_knife_edge(self):
        # Session 32's set must survive its own guard.
        loop, _, _ = make_loop([state(180.0, 45.0)], deadband_deg=0.25, interval_s=0.5)
        assert loop.deadband_deg == 0.25

    def test_negative_deadband_is_refused(self):
        with pytest.raises(ValueError, match="deadband_deg"):
            make_loop([state(180.0, 45.0)], deadband_deg=-1.0)

    def test_non_positive_interval_is_refused(self):
        with pytest.raises(ValueError, match="interval_s"):
            make_loop([state(180.0, 45.0)], interval_s=0.0)

    def test_the_firmware_deadzone_is_only_a_floor(self):
        # Documented as the point below which a software deadband cannot
        # change anything, not as a value to use.
        assert FIRMWARE_DEADZONE_DEG == 0.2
        assert DEFAULT_TICK_INTERVAL_S == 1.0

    def test_nothing_happens_before_the_first_tick(self):
        loop, rotor, _ = make_loop([state(180.0, 45.0)])
        assert loop.latest_sample is None
        assert loop.last_commanded is None
        rotor.read_position.assert_not_called()
        rotor.move_to.assert_not_called()

    def test_alignment_offset_defaults_to_identity(self):
        loop, _, _ = make_loop([state(180.0, 45.0)])
        assert loop.alignment_offset == AlignmentOffset()
        assert loop.alignment_offset.is_identity

    def test_alignment_offset_can_be_given(self):
        offset = AlignmentOffset(azimuth_deg=3.0, elevation_deg=-1.0)
        loop, _, _ = make_loop([state(180.0, 45.0)], alignment_offset=offset)
        assert loop.alignment_offset == offset


# ---------------------------------------------------------------------------
# A single tick
# ---------------------------------------------------------------------------


class TestTick:
    def test_first_tick_always_commands(self):
        loop, rotor, _ = make_loop([state(180.0, 45.0)])

        sample = loop.tick()

        assert sample.outcome is TickOutcome.COMMANDED
        assert sample.commanded
        assert commanded_positions(rotor) == [Position(180.0, 45.0)]

    def test_alignment_offset_is_applied_to_the_commanded_position(self):
        # rotor_target - and therefore what actually gets sent - runs
        # through sky_to_rotor with the loop's own offset, not identity.
        offset = AlignmentOffset(azimuth_deg=3.0, elevation_deg=-1.0)
        loop, rotor, _ = make_loop([state(180.0, 45.0)], alignment_offset=offset)

        sample = loop.tick()

        assert sample.rotor_target == Position(183.0, 44.0)
        assert commanded_positions(rotor) == [Position(183.0, 44.0)]

    def test_sample_carries_the_whole_seam(self):
        # time, sky position, range and range rate are what dsp and the
        # UI consume; both positions are here so a display can show sky
        # and axis as the distinct things they are.
        loop, _, _ = make_loop(
            [state(123.4, 56.7, range_km=812.0, range_rate=-4.25)],
            reported=Position(120.0, 54.0),
        )

        sample = loop.tick()

        assert isinstance(sample, TrackSample)
        assert sample.time == START
        assert sample.sky_position == AzEl(123.4, 56.7)
        assert sample.range_km == 812.0
        assert sample.range_rate_km_s == -4.25
        assert sample.rotor_target == Position(123.4, 56.7)
        assert sample.rotor_position == Position(120.0, 54.0)

    def test_target_is_asked_for_the_current_time(self):
        loop, _, clock = make_loop([state(180.0, 45.0)])
        clock.advance(30.0)

        assert loop.tick().time == START + timedelta(seconds=30)

    def test_latest_sample_tracks_the_last_tick(self):
        loop, _, _ = make_loop([state(10.0, 20.0), state(90.0, 30.0)])

        first = loop.tick()
        assert loop.latest_sample is first

        second = loop.tick()
        assert loop.latest_sample is second

    def test_reads_the_rotor_once_per_tick(self):
        loop, rotor, _ = make_loop([state(180.0, 45.0)])

        loop.tick()
        loop.tick()

        assert rotor.read_position.call_count == 2

    def test_reads_the_rotor_before_commanding(self):
        # So the sample shows where the axes were when the decision was
        # made, and so the travel guard fires before a command.
        loop, rotor, _ = make_loop([state(180.0, 45.0)])
        order: list[str] = []
        rotor.read_position.side_effect = lambda: (order.append("read"), Position(0.0, 0.0))[1]
        rotor.move_to.side_effect = lambda position: order.append("move")

        loop.tick()

        assert order == ["read", "move"]

    def test_tick_does_not_sleep(self):
        # It is driven by someone else's clock: a UI timer, run(), a test.
        loop, _, clock = make_loop([state(180.0, 45.0)])

        loop.tick()

        assert clock.sleeps == []


# ---------------------------------------------------------------------------
# Deadband
# ---------------------------------------------------------------------------


class TestDeadband:
    def test_small_movement_sends_nothing(self):
        loop, rotor, _ = make_loop([state(180.0, 45.0), state(181.0, 45.5)])

        loop.tick()
        second = loop.tick()

        assert second.outcome is TickOutcome.WITHIN_DEADBAND
        assert not second.commanded
        assert commanded_positions(rotor) == [Position(180.0, 45.0)]

    def test_movement_past_the_deadband_commands(self):
        loop, rotor, _ = make_loop([state(180.0, 45.0), state(183.0, 45.0)])

        loop.tick()
        second = loop.tick()

        assert second.outcome is TickOutcome.COMMANDED
        assert commanded_positions(rotor) == [Position(180.0, 45.0), Position(183.0, 45.0)]

    def test_either_axis_can_trigger_a_command(self):
        loop, rotor, _ = make_loop([state(180.0, 45.0), state(180.5, 48.0)])

        loop.tick()
        loop.tick()

        assert commanded_positions(rotor)[-1] == Position(180.5, 48.0)

    def test_exactly_the_deadband_commands(self):
        # The window is "already arrived"; movement equal to it is not.
        loop, rotor, _ = make_loop([state(180.0, 45.0), state(182.5, 45.0)])

        loop.tick()
        loop.tick()

        assert len(commanded_positions(rotor)) == 2

    def test_small_movements_accumulate_until_they_matter(self):
        # The comparison is against the last commanded position, so a
        # target creeping a degree per tick still gets a command once the
        # total exceeds the deadband. Comparing tick to tick instead
        # would let a slow pass drift away unbounded.
        loop, rotor, _ = make_loop(
            [state(180.0 + n, 45.0) for n in range(5)],
        )

        for _ in range(5):
            loop.tick()

        assert commanded_positions(rotor) == [Position(180.0, 45.0), Position(183.0, 45.0)]

    def test_compares_against_the_command_not_the_reading(self):
        # Stock gains leave an axis 1-2 degrees short of target. Comparing
        # against the reading would read that shortfall as movement and
        # re-command the same position forever.
        loop, rotor, _ = make_loop(
            [state(180.0, 45.0), state(180.0, 45.0)],
            reported=[Position(0.0, 0.0), Position(178.0, 43.0)],
        )

        loop.tick()
        second = loop.tick()

        assert second.outcome is TickOutcome.WITHIN_DEADBAND
        assert len(commanded_positions(rotor)) == 1

    def test_zero_deadband_commands_every_tick(self):
        loop, rotor, _ = make_loop([state(180.0, 45.0), state(180.0, 45.0)], deadband_deg=0.0)

        loop.tick()
        loop.tick()

        assert len(commanded_positions(rotor)) == 2

    def test_last_commanded_reports_what_was_sent(self):
        loop, _, _ = make_loop([state(180.0, 45.0), state(181.0, 45.0)])

        loop.tick()
        loop.tick()

        assert loop.last_commanded == Position(180.0, 45.0)


# ---------------------------------------------------------------------------
# Below the horizon
# ---------------------------------------------------------------------------


class TestBelowHorizon:
    def test_below_the_horizon_commands_nothing(self):
        loop, rotor, _ = make_loop([state(180.0, -12.0)])

        sample = loop.tick()

        assert sample.outcome is TickOutcome.BELOW_HORIZON
        rotor.move_to.assert_not_called()

    def test_below_the_horizon_still_emits_a_full_sample(self):
        # Doppler correction and a display both want the numbers while
        # waiting for the target to rise.
        loop, _, _ = make_loop([state(180.0, -12.0, range_km=2400.0, range_rate=-6.0)])

        sample = loop.tick()

        assert sample.sky_position == AzEl(180.0, -12.0)
        assert sample.range_rate_km_s == -6.0
        assert sample.rotor_target == Position(180.0, -12.0)

    def test_starting_early_then_rising_commands_on_the_way_up(self):
        # The expected shape of a hand-started track: a few ticks below
        # the horizon, then the pass.
        loop, rotor, _ = make_loop([state(100.0, -5.0), state(101.0, -0.5), state(102.0, 3.0)])

        outcomes = [loop.tick().outcome for _ in range(3)]

        assert outcomes == [
            TickOutcome.BELOW_HORIZON,
            TickOutcome.BELOW_HORIZON,
            TickOutcome.COMMANDED,
        ]
        assert commanded_positions(rotor) == [Position(102.0, 3.0)]

    def test_the_horizon_is_not_the_rotors_elevation_limit(self):
        # A target at 0.0 is up, and the deadband is what decides whether
        # it is worth a command - not the sign test.
        loop, rotor, _ = make_loop([state(90.0, 0.0)])

        assert loop.tick().outcome is TickOutcome.COMMANDED
        assert commanded_positions(rotor) == [Position(90.0, 0.0)]


# ---------------------------------------------------------------------------
# Refusals and the travel guard
# ---------------------------------------------------------------------------


class TestRefusal:
    def test_a_refused_setpoint_stops_the_loop(self):
        # Above the horizon but outside declared travel means the
        # geometry is beyond what this rotor may do. Swallowing it would
        # leave the antenna parked while the app still looks like it is
        # tracking.
        loop, rotor, _ = make_loop([state(180.0, 45.0)])
        rotor.move_to.side_effect = PositionLimitError("Elevation 45.0 is outside travel")

        with pytest.raises(PositionLimitError):
            loop.tick()

    def test_a_refusal_leaves_no_record_of_a_command(self):
        loop, rotor, _ = make_loop([state(180.0, 45.0)])
        rotor.move_to.side_effect = PositionLimitError("nope")

        with pytest.raises(PositionLimitError):
            loop.tick()

        assert loop.last_commanded is None

    def test_a_refusal_propagates_out_of_run(self):
        loop, rotor, _ = make_loop([state(180.0, 45.0)])
        rotor.move_to.side_effect = PositionLimitError("nope")

        with pytest.raises(PositionLimitError):
            list(loop.run(max_ticks=5))


class TestTravelGuard:
    def test_reported_position_outside_travel_stops_the_loop(self):
        # Rule 2.6's cumulative net rotation, guarded the cheap way: the
        # axis reading already is net rotation since home, so the check
        # is that it has stayed inside declared travel.
        loop, _, _ = make_loop([state(180.0, 45.0)], reported=Position(410.0, 20.0))

        with pytest.raises(TravelGuardError, match="410.0"):
            loop.tick()

    def test_the_guard_runs_before_any_command(self):
        loop, rotor, _ = make_loop([state(180.0, 45.0)], reported=Position(-30.0, 20.0))

        with pytest.raises(TravelGuardError):
            loop.tick()

        rotor.move_to.assert_not_called()

    def test_the_guard_message_says_what_to_check(self):
        loop, _, _ = make_loop([state(180.0, 45.0)], reported=Position(0.0, 200.0))

        with pytest.raises(TravelGuardError, match="station config"):
            loop.tick()

    def test_the_guard_uses_the_declared_limits(self):
        # The same numbers check_setpoint uses, so one rotor is held to
        # one set of limits whichever direction they are checked in.
        loop, _, _ = make_loop(
            [state(180.0, 45.0)],
            reported=Position(300.0, 20.0),
            caps=capabilities(azimuth_max_deg=270.0),
        )

        with pytest.raises(TravelGuardError):
            loop.tick()

    def test_a_normal_reading_passes_the_guard(self):
        loop, _, _ = make_loop([state(180.0, 45.0)], reported=Position(179.0, 44.0))

        assert loop.tick().rotor_position == Position(179.0, 44.0)

    def test_a_healthy_reading_costs_only_one_read(self):
        """The re-read is on the suspicious path, not on every tick."""
        loop, rotor, _ = make_loop([state(180.0, 45.0)], reported=Position(179.0, 44.0))

        loop.tick()

        assert rotor.read_position.call_count == 1


class TestGuardRereads:
    """One corrupted reply must not cost a pass. Session 24, run C."""

    def test_one_impossible_reading_is_re_read_rather_than_fatal(self):
        # Exactly run C: an elevation of -8.2 that three later reads,
        # sixty seconds apart with nothing commanded in between, all
        # reported as 3.8. The first reading was false.
        loop, _, _ = make_loop(
            [state(180.0, 45.0)],
            reported=[Position(0.0, -8.2), Position(0.0, 3.8)],
        )

        sample = loop.tick()

        assert sample.rotor_position == Position(0.0, 3.8)

    def test_the_confirming_read_is_the_one_trusted(self):
        """The second reading is the sample's, not the discredited first."""
        loop, _, _ = make_loop(
            [state(180.0, 45.0)],
            reported=[Position(0.0, -8.2), Position(0.0, 3.8)],
        )

        assert loop.tick().rotor_position.elevation == 3.8

    def test_a_confirmed_divergence_still_aborts(self):
        """Aborting stays the default: driving through one is unsafe."""
        loop, rotor, _ = make_loop(
            [state(180.0, 45.0)],
            reported=[Position(0.0, -8.2), Position(0.0, -8.4)],
        )

        with pytest.raises(TravelGuardError, match="agreed"):
            loop.tick()

        rotor.move_to.assert_not_called()

    def test_the_abort_message_reports_both_readings(self):
        """Two numbers is what tells a transient from a real divergence."""
        loop, _, _ = make_loop(
            [state(180.0, 45.0)],
            reported=[Position(0.0, -8.2), Position(0.0, -8.4)],
        )

        with pytest.raises(TravelGuardError, match=r"first read said.*-8\.2"):
            loop.tick()

    def test_rereads_are_counted_even_when_the_run_survives(self):
        """The corruption rate is worth more than the completed run.

        Session 24 saw one corrupted reply and could not say whether that
        was a link falling apart or a once-an-evening event. Two in two
        attempts describes a very different link from two in twenty, and
        nothing was counting.
        """
        loop, _, _ = make_loop(
            [state(180.0, 45.0), state(181.0, 45.0)],
            reported=[
                Position(0.0, -8.2),
                Position(0.0, 3.8),
                Position(0.0, 3.9),
            ],
        )

        assert loop.guard_rereads == 0
        loop.tick()
        assert loop.guard_rereads == 1
        loop.tick()
        assert loop.guard_rereads == 1

    def test_a_freshly_homed_rotor_does_not_trip_the_guard(self):
        # The captured bytes from the first bring-up: a homed rotator
        # reports AZ-1.5 EL2.0, which is outside a declared 0-360 azimuth
        # range. Checking a *reading* as strictly as a setpoint would make
        # every track fail on its first tick. This is the same class of
        # bug as PR #9, where Position carried AzEl's constraints.
        loop, _, _ = make_loop([state(180.0, 45.0)], reported=Position(-1.5, 2.0))

        assert loop.tick().rotor_position == Position(-1.5, 2.0)

    def test_settling_past_a_declared_limit_does_not_trip_the_guard(self):
        # An axis commanded to its declared limit stops a degree or two
        # either side of it. Inside the acceptance window, that is the
        # rotor working normally.
        loop, _, _ = make_loop([state(180.0, 45.0)], reported=Position(361.5, 20.0))

        assert loop.tick().rotor_position == Position(361.5, 20.0)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    def test_yields_one_sample_per_tick(self):
        loop, _, _ = make_loop([state(180.0, 45.0)])

        samples = list(loop.run(max_ticks=3))

        assert len(samples) == 3
        assert all(isinstance(sample, TrackSample) for sample in samples)

    def test_sleeps_the_interval_between_ticks(self):
        loop, _, clock = make_loop([state(180.0, 45.0)], interval_s=2.0)

        list(loop.run(max_ticks=3))

        # Two gaps for three ticks, and no wait on the way out.
        assert clock.sleeps == [2.0, 2.0]

    def test_cadence_is_measured_from_the_start_not_the_tick(self):
        # A tick costs serial round trips. Sleeping a flat interval after
        # each one would stretch the cadence by that cost every time.
        loop, rotor, clock = make_loop([state(180.0, 45.0)], interval_s=1.0)
        rotor.read_position.side_effect = lambda: (clock.advance(0.3), Position(0.0, 0.0))[1]

        list(loop.run(max_ticks=3))

        assert clock.sleeps == [pytest.approx(0.7), pytest.approx(0.7)]
        assert clock.monotonic() == pytest.approx(2.3)

    def test_an_overrunning_tick_does_not_try_to_catch_up(self):
        # A late pointing update is worth sending; a stale one isn't.
        loop, rotor, clock = make_loop([state(180.0, 45.0)], interval_s=1.0)
        rotor.read_position.side_effect = lambda: (clock.advance(2.5), Position(0.0, 0.0))[1]

        list(loop.run(max_ticks=3))

        assert clock.sleeps == []

    def test_duration_bounds_the_run(self):
        loop, _, clock = make_loop([state(180.0, 45.0)], interval_s=1.0)

        samples = list(loop.run(duration_s=3.0))

        assert len(samples) == 3
        assert clock.monotonic() == pytest.approx(3.0)

    def test_zero_duration_ticks_nothing(self):
        loop, rotor, _ = make_loop([state(180.0, 45.0)])

        assert list(loop.run(duration_s=0.0)) == []
        rotor.read_position.assert_not_called()

    def test_the_target_is_sampled_at_each_tick_time(self):
        loop, _, _ = make_loop([state(180.0, 45.0)], interval_s=5.0)
        target = loop.target

        list(loop.run(max_ticks=3))

        assert target.asked_at == [
            START,
            START + timedelta(seconds=5),
            START + timedelta(seconds=10),
        ]

    def test_deadband_applies_across_ticks_in_a_run(self):
        loop, rotor, _ = make_loop([state(180.0, 45.0), state(180.2, 45.0), state(180.4, 45.0)])

        list(loop.run(max_ticks=3))

        assert commanded_positions(rotor) == [Position(180.0, 45.0)]


class TestCleanStop:
    def test_breaking_out_sends_nothing_further(self):
        loop, rotor, _ = make_loop([state(180.0 + n * 10, 45.0) for n in range(5)])

        for sample in loop.run(max_ticks=5):
            if sample.commanded:
                break

        assert len(commanded_positions(rotor)) == 1

    def test_stopping_does_not_stop_the_rotor(self):
        # Same call as Rotor.__exit__: a move in progress doesn't need the
        # link to finish, and SA SE is not an emergency stop.
        loop, rotor, _ = make_loop([state(180.0, 45.0)])

        for _ in loop.run(max_ticks=3):
            break

        rotor.stop.assert_not_called()

    def test_stopping_does_not_close_the_connection(self):
        # Whoever opened the port owns closing it.
        loop, rotor, _ = make_loop([state(180.0, 45.0)])

        list(loop.run(max_ticks=2))

        rotor.close.assert_not_called()
        rotor.connect.assert_not_called()

    def test_the_last_sample_survives_the_stop(self):
        # So a display still shows where things were when it ended.
        loop, _, _ = make_loop([state(180.0, 45.0)])

        list(loop.run(max_ticks=2))

        assert loop.latest_sample is not None
        assert loop.latest_sample.sky_position == AzEl(180.0, 45.0)


# ---------------------------------------------------------------------------
# rotor_to_sky()
# ---------------------------------------------------------------------------


class TestRotorToSky:
    def test_returns_an_azel(self):
        assert isinstance(rotor_to_sky(Position(180.0, 45.0)), AzEl)

    def test_an_in_range_reading_round_trips(self):
        assert rotor_to_sky(Position(123.4, 56.7)) == AzEl(123.4, 56.7)

    def test_applies_no_correction_by_default(self):
        # Mirrors TestSkyToRotor.test_applies_no_correction_by_default in
        # test_pointing.py: DESIGNED TO FAIL if the default ever stops
        # being identity, so whoever changes it has to consciously touch
        # this assertion rather than silently changing what every
        # readout window displays.
        assert rotor_to_sky(Position(90.0, 30.0)) == AzEl(90.0, 30.0)

    def test_applies_the_offset_when_one_is_given(self):
        # The inverse of TestSkyToRotor's equivalent case: subtraction,
        # not addition, going the other direction.
        offset = AlignmentOffset(azimuth_deg=5.0, elevation_deg=-2.0)
        assert rotor_to_sky(Position(95.0, 28.0), offset) == AzEl(90.0, 30.0)

    def test_offset_is_applied_before_wrap_and_clamp(self):
        # A reading that only becomes out-of-range once the offset is
        # removed still has to wrap/clamp correctly, not raise or
        # silently skip the correction.
        offset = AlignmentOffset(azimuth_deg=10.0, elevation_deg=0.0)
        # 5.0 - 10.0 = -5.0, which wraps to 355.0.
        assert rotor_to_sky(Position(5.0, 10.0), offset) == AzEl(355.0, 10.0)

    def test_a_freshly_homed_reading_wraps_to_a_compass_bearing(self):
        # The captured bytes from the first bring-up: AZ-1.5 EL2.0. A
        # negative axis reading is an ordinary homing settle, not an
        # error, and its compass direction is 358.5, not -1.5.
        assert rotor_to_sky(Position(-1.5, 2.0)) == AzEl(358.5, 2.0)

    def test_azimuth_past_360_wraps_to_the_same_bearing_as_no_extra_turns(self):
        # A rotor commanded to 380 physically ends up pointing where 20
        # does - the extra rotation is cable wrap, not a different
        # direction. See AzimuthWrap.EXTRA_ROTATION.
        assert rotor_to_sky(Position(380.0, 10.0)) == AzEl(20.0, 10.0)

    def test_a_full_extra_turn_wraps_to_the_same_bearing(self):
        assert rotor_to_sky(Position(720.0, 10.0)) == AzEl(0.0, 10.0)

    def test_elevation_past_vertical_is_clamped_not_rejected(self):
        # Phil's rotor declares elevation_max_deg=180 (it can rotate past
        # vertical); AzEl cannot represent that until flip mode exists to
        # interpret it. Clamping keeps this from raising on a legitimate
        # axis reading.
        assert rotor_to_sky(Position(10.0, 95.0)) == AzEl(10.0, 90.0)

    def test_elevation_below_the_declared_floor_is_clamped_not_rejected(self):
        assert rotor_to_sky(Position(10.0, -95.0)) == AzEl(10.0, -90.0)

    def test_elevation_boundary_values_are_not_clamped_away(self):
        assert rotor_to_sky(Position(10.0, 90.0)) == AzEl(10.0, 90.0)
        assert rotor_to_sky(Position(10.0, -90.0)) == AzEl(10.0, -90.0)

    def test_never_raises_on_a_position_the_type_itself_permits(self):
        # Position accepts anything up to +/-MAX_AXIS_DEGREES (1080).
        # rotor_to_sky is a display conversion for a live readout: it
        # must not crash the window on any reading the rotor could
        # actually report.
        for azimuth in (-1080.0, -400.0, 0.0, 400.0, 1080.0):
            for elevation in (-1080.0, -95.0, 0.0, 95.0, 1080.0):
                rotor_to_sky(Position(azimuth, elevation))
