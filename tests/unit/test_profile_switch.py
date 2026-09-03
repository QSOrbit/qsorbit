"""Unit tests for switching tracking profile while a loop is running.

Chunk H's done-when asks for a **live mid-pass switch**, so this is the
behaviour that clause rests on. Three of the four rules under test were
Phil's calls before any of it was written, and each is here because the
obvious alternative is worse:

* a switch is **queued**, applied by the loop at the top of its next
  tick, so the serial port keeps exactly one owner and no lock is
  introduced;
* a failed gain push **stops the run** rather than carrying on with a
  gain mixture nobody chose;
* a switch is **refused while an axis is stalled**, checked both when
  asked and again when applied, because a stall can arrive in the tick
  between the two.

The fixtures are shared with test_pointing_loop.py's: a stub target, a
spec'd rotor mock and a fake clock, so nothing here needs an ephemeris.
"""

from __future__ import annotations

import pytest

# Reused rather than duplicated: make_loop() already knows how to build
# a spec'd rotor mock against a stub target, and a second copy of it
# would drift from the first the moment the loop's signature changes.
from test_pointing_loop import make_loop, state

from qsorbit.core.pointing import ProfileSwitchError, TickOutcome
from qsorbit.core.rotor import GainVerificationError, Position
from qsorbit.core.stall_guard import StallGuard
from qsorbit.core.tracking_profile import TrackingProfile

STOCK = TrackingProfile(name="stock", deadband_deg=2.5, interval_s=1.0)
TRACKING = TrackingProfile(
    name="tracking",
    deadband_deg=0.25,
    interval_s=0.5,
    arrival_window_deg=1.0,
)


def make(states=None, **kwargs):
    """A loop labelled with the stock profile, matching its own numbers."""
    kwargs.setdefault("interval_s", STOCK.interval_s)
    kwargs.setdefault("deadband_deg", STOCK.deadband_deg)
    kwargs.setdefault("profile", STOCK)
    return make_loop(states or [state(100.0, 30.0)], **kwargs)


class TestLabelling:
    def test_a_loop_can_say_which_profile_it_is_running(self):
        loop, _rotor, _clock = make()
        assert loop.active_profile is STOCK

    def test_an_unlabelled_loop_is_a_real_state(self):
        # A test, or --interval on a station with no profiles declared,
        # runs a cadence no profile names. That is not missing data.
        loop, _rotor, _clock = make(profile=None)
        assert loop.active_profile is None

    def test_a_mislabelled_loop_is_refused(self):
        # A wiring mistake, not a bad value: built from one profile's
        # numbers and labelled with another's, so every later report of
        # "which profile is running" would be wrong.
        with pytest.raises(ValueError, match="but this loop was built with"):
            make(interval_s=1.0, deadband_deg=2.5, profile=TRACKING)


class TestQueueing:
    def test_a_request_does_not_take_effect_immediately(self):
        # The point of queueing: nothing has touched the port yet.
        loop, _rotor, _clock = make()
        loop.request_profile(TRACKING)
        assert loop.pending_profile is TRACKING
        assert loop.active_profile is STOCK
        assert loop.interval_s == 1.0

    def test_the_next_tick_applies_it(self):
        loop, _rotor, _clock = make()
        loop.request_profile(TRACKING)
        loop.tick()
        assert loop.pending_profile is None
        assert loop.active_profile is TRACKING

    def test_the_cadence_really_changes(self):
        # Both halves, because a switch that moved gains and not cadence
        # would ship half of a set that was validated whole.
        loop, _rotor, _clock = make()
        loop.request_profile(TRACKING)
        loop.tick()
        assert loop.interval_s == 0.5
        assert loop.deadband_deg == 0.25

    def test_a_second_request_replaces_the_first(self):
        # Nothing reached the rotor, so there is nothing to undo -- and
        # applying a superseded switch would push gains nobody wants.
        loop, _rotor, _clock = make()
        pushed = []
        loop.request_profile(TRACKING)
        loop.request_profile(STOCK)
        assert loop.pending_profile is STOCK
        assert pushed == []

    def test_the_push_runs_on_the_ticking_thread(self):
        # Recorded by when it happens, since that is the whole design:
        # the callback must fire inside tick(), not inside the request.
        seen = []
        loop, _rotor, _clock = make(on_profile_change=seen.append)
        loop.request_profile(TRACKING)
        assert seen == []
        loop.tick()
        assert seen == [TRACKING]

    def test_the_push_happens_before_anything_is_commanded(self):
        # Gains have to be in force for the move they are meant to
        # improve, not one tick late.
        order = []
        loop, rotor, _clock = make(
            states=[state(100.0, 30.0), state(140.0, 40.0)],
            on_profile_change=lambda _p: order.append("gains"),
        )
        rotor.move_to.side_effect = lambda _pos: order.append("move")
        loop.tick()
        loop.request_profile(TRACKING)
        loop.tick()
        assert order == ["move", "gains", "move"]

    def test_no_callback_is_fine(self):
        # A loop with no rotor-side pusher still switches cadence, which
        # is what a stock-only station does.
        loop, _rotor, _clock = make(on_profile_change=None)
        loop.request_profile(TRACKING)
        loop.tick()
        assert loop.active_profile is TRACKING


class TestFailedPush:
    def test_a_verification_failure_propagates_out_of_tick(self):
        # Phil's call: stop loudly. The controller is running a mixture
        # nobody chose, so every measurement after this would be
        # attributed to the wrong configuration.
        def explode(_profile):
            raise GainVerificationError("register 2 asked 1.00 got 0.00")

        loop, _rotor, _clock = make(on_profile_change=explode)
        loop.request_profile(TRACKING)
        with pytest.raises(GainVerificationError):
            loop.tick()

    def test_a_failed_switch_does_not_leave_the_switch_queued(self):
        # Otherwise the next tick would retry a push that just failed,
        # against a controller that just proved it does not accept it.
        def explode(_profile):
            raise GainVerificationError("nope")

        loop, _rotor, _clock = make(on_profile_change=explode)
        loop.request_profile(TRACKING)
        with pytest.raises(GainVerificationError):
            loop.tick()
        assert loop.pending_profile is None

    def test_a_failed_switch_leaves_the_old_profile_in_force(self):
        # The cadence must not move when the gains did not.
        def explode(_profile):
            raise GainVerificationError("nope")

        loop, _rotor, _clock = make(on_profile_change=explode)
        loop.request_profile(TRACKING)
        with pytest.raises(GainVerificationError):
            loop.tick()
        assert loop.active_profile is STOCK
        assert loop.interval_s == 1.0


FOLLOW = 12  # ticks of healthy tracking before anything jams
STEP = 3.0  # deg per tick, comfortably past the stock 2.5 deg deadband


def following(n: int) -> Position:
    """Where a healthy axis reads on tick ``n`` -- a stiction lag behind."""
    return Position(100.0 + STEP * n - 1.5, 45.0)


def moving(count: int):
    return [state(100.0 + STEP * n, 45.0) for n in range(count)]


def follows_then_jams(count: int):
    """Track cleanly, then freeze. Matching test_pointing_loop's TestStall.

    The axis has to be seen following first: the detector will not judge
    an axis it has never seen move, because a standing start is
    indistinguishable from a jam over one window -- which two bench runs
    proved the hard way on 2026-09-02.
    """
    stuck = following(FOLLOW)
    return [following(n) if n <= FOLLOW else stuck for n in range(count)]


def stalled_loop(**kwargs):
    """A loop ticked until an axis is genuinely declared stalled."""
    loop, rotor, clock = make(states=moving(40), reported=follows_then_jams(40), **kwargs)
    for _ in range(40):
        loop.tick()
        if loop.is_stalled:
            break
    assert loop.is_stalled, "fixture failed to produce a stall"
    return loop, rotor, clock


class TestStallRefusal:
    def test_requesting_while_stalled_raises(self):
        loop, _rotor, _clock = stalled_loop()
        assert loop.is_stalled
        with pytest.raises(ProfileSwitchError, match="stalled"):
            loop.request_profile(TRACKING)

    def test_the_refusal_says_which_axis_and_what_to_do(self):
        loop, _rotor, _clock = stalled_loop()
        with pytest.raises(ProfileSwitchError) as caught:
            loop.request_profile(TRACKING)
        message = str(caught.value)
        assert "Free the axis first" in message
        assert loop.profile_refusal == message

    def test_a_refusal_pushes_nothing(self):
        seen = []
        loop, _rotor, _clock = stalled_loop(on_profile_change=seen.append)
        with pytest.raises(ProfileSwitchError):
            loop.request_profile(TRACKING)
        assert seen == []

    def test_nothing_is_left_queued_by_a_refusal(self):
        loop, _rotor, _clock = stalled_loop()
        with pytest.raises(ProfileSwitchError):
            loop.request_profile(TRACKING)
        assert loop.pending_profile is None

    def test_the_refusal_message_is_printable(self):
        loop, _rotor, _clock = stalled_loop()
        with pytest.raises(ProfileSwitchError) as caught:
            loop.request_profile(TRACKING)
        str(caught.value).encode("ascii")


class TestStallRace:
    def test_a_stall_between_request_and_apply_drops_the_switch(self):
        # The one-tick race the double check exists for. An operator who
        # asked for gains half a second before an axis jammed did not
        # ask for gains on a jammed axis.
        seen = []
        loop, _rotor, _clock = stalled_loop(on_profile_change=seen.append)

        # Force the pending state directly, standing in for a request
        # that was accepted on the tick before the stall was declared.
        loop._pending_profile = TRACKING  # noqa: SLF001
        loop.tick()

        assert seen == []
        assert loop.pending_profile is None
        assert loop.active_profile is STOCK
        assert "Dropped the queued switch" in (loop.profile_refusal or "")


class TestStallDetectorRescaling:
    def test_the_detector_follows_the_new_cadence(self):
        # The guard's window is a duration, so the tick count it spans
        # has to change when the interval does.
        loop, _rotor, _clock = make(stall_guard=StallGuard(window_s=6.0, free_play_deg=3.0))
        before = loop._stall.ticks  # noqa: SLF001
        loop.request_profile(TRACKING)
        loop.tick()
        after = loop._stall.ticks  # noqa: SLF001
        assert after > before
        assert loop._stall.latency_s == pytest.approx(6.0, abs=0.5)  # noqa: SLF001

    def test_history_is_not_carried_across_a_switch(self):
        # Samples taken at the old cadence cover a different span of
        # time, so mixing them would make the window neither duration.
        loop, _rotor, _clock = make(states=[state(100.0 + n, 30.0) for n in range(6)])
        for _ in range(3):
            loop.tick()
        loop.request_profile(TRACKING)
        loop.tick()
        assert loop._stall.armed_axes == frozenset()  # noqa: SLF001


class TestOutcomesUnaffected:
    def test_a_switch_does_not_itself_command_the_rotor(self):
        loop, rotor, _clock = make(states=[state(100.0, 30.0), state(100.0, 30.0)])
        loop.tick()
        commanded = rotor.move_to.call_count
        loop.request_profile(TRACKING)
        sample = loop.tick()
        assert rotor.move_to.call_count == commanded
        assert sample.outcome is not TickOutcome.STALLED
