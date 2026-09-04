"""Tests for the background rotor ticker.

No threads and no real time in most of these. ``run_until_stopped`` is
public precisely so a test can drive the schedule directly with an
injected ``wait`` and clock, and find out what cadence the loop would
really have been ticked at rather than sleeping through real seconds to
approximate it. The two tests that do start a thread are the ones whose
whole subject is the thread.
"""

from __future__ import annotations

import threading

import pytest

from qsorbit.core.tracking_thread import TrackingThread


class FakeClock:
    """A monotonic clock that only moves when something says it did."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeLoop:
    """A TrackingLoop-shaped double. No rotor, no serial port.

    **The loop pays for its own tick**, advancing the clock by
    ``tick_cost_s``. A first draft of this file charged that cost inside
    the wait instead, which is a different physics: it made the tick's
    cost land before the tick had happened, and produced a schedule
    (0.5, 0, 0.5, 0, ...) that the real code does not produce. The
    fixture was wrong and the code was right, which is Session 20's
    lesson pointed at a test double rather than at a signal generator.

    ``interval_s`` is a plain attribute so a test can move it mid-run,
    which is what a live profile switch does to the real loop.
    """

    def __init__(
        self,
        interval_s: float = 0.5,
        *,
        clock: FakeClock | None = None,
        tick_cost_s: float = 0.0,
    ) -> None:
        self.interval_s = interval_s
        self.ticks = 0
        self.raise_on_tick: BaseException | None = None
        self._clock = clock
        self._tick_cost_s = tick_cost_s

    def tick(self) -> None:
        if self.raise_on_tick is not None:
            raise self.raise_on_tick
        self.ticks += 1
        if self._clock is not None:
            self._clock.now += self._tick_cost_s


class ScriptedWait:
    """Stands in for the stop event, and records what it was asked to wait.

    The recorded delays are the measurement these tests exist to make:
    they are what the real thread would have slept, and therefore what
    the rotor's cadence really is.
    """

    def __init__(self, clock: FakeClock, *, stop_after: int) -> None:
        self._clock = clock
        self._stop_after = stop_after
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> bool:
        self.delays.append(seconds)
        if len(self.delays) > self._stop_after:
            return True
        self._clock.now += seconds
        return False


def _driven(interval_s: float = 0.5, *, stop_after: int, tick_cost_s: float = 0.0):
    """A ticker wired to a fake clock, with the loop and the wait it uses."""
    clock = FakeClock()
    loop = FakeLoop(interval_s, clock=clock, tick_cost_s=tick_cost_s)
    wait = ScriptedWait(clock, stop_after=stop_after)
    ticker = TrackingThread(loop, wait=wait, monotonic=clock, report=lambda _message: None)
    return ticker, loop, wait


class TestCadence:
    def test_it_ticks_at_the_loops_own_interval(self):
        ticker, loop, wait = _driven(0.5, stop_after=3)

        ticker.run_until_stopped()

        assert wait.delays[:3] == [0.5, 0.5, 0.5]
        assert loop.ticks == 3

    def test_a_tick_that_takes_time_does_not_lengthen_the_period(self):
        """The whole point of the class, and it is arithmetic, not tidiness.

        A tick writes a command, sleeps out an RS-485 turnaround and
        blocks on a read -- call it 0.15 s. Scheduling with
        ``wait(interval)`` would make the real period 0.65 s, and since
        the commanded step is ``rate x tick``, the validated 0.5 deg set
        would silently become 0.65 deg at the hardware. Deadline
        scheduling absorbs the tick's cost into the following wait.
        """
        ticker, _loop, wait = _driven(0.5, stop_after=3, tick_cost_s=0.15)

        ticker.run_until_stopped()

        # The first wait is a whole interval; every later one is short
        # by exactly what the tick cost, so the period stays 0.5 s.
        assert wait.delays[0] == pytest.approx(0.5)
        assert wait.delays[1] == pytest.approx(0.35)
        assert wait.delays[2] == pytest.approx(0.35)

    def test_a_tick_slower_than_the_interval_does_not_fire_a_catch_up_burst(self):
        # A late pointing update is worth sending; a stale one is not,
        # and a burst of them would command a slew per tick. Same policy
        # as TrackingLoop.run().
        ticker, loop, wait = _driven(0.5, stop_after=3, tick_cost_s=2.0)

        ticker.run_until_stopped()

        assert wait.delays[0] == pytest.approx(0.5)
        # Clamped at zero rather than going negative and then being
        # repaid as a run of instant ticks.
        assert wait.delays[1:3] == [0.0, 0.0]
        assert loop.ticks == 3

    def test_a_profile_switch_moves_the_cadence_without_being_told(self):
        # interval_s is re-read every iteration because tick() is where
        # a queued profile is applied -- the schedule finds out the same
        # way everything else does, by reading it afterwards.
        ticker, loop, wait = _driven(1.0, stop_after=3)
        original_tick = loop.tick

        def tick_then_switch() -> None:
            original_tick()
            loop.interval_s = 0.25

        loop.tick = tick_then_switch  # type: ignore[method-assign]
        ticker.run_until_stopped()

        assert wait.delays[0] == pytest.approx(1.0)
        assert wait.delays[1] == pytest.approx(0.25)

    def test_the_first_wait_is_a_whole_interval_because_start_already_ticked(self):
        # start() ticks synchronously before the thread exists. If the
        # thread then began with a zero wait, the antenna would be
        # commanded twice in a row at startup.
        ticker, _loop, wait = _driven(0.5, stop_after=1)

        ticker.run_until_stopped()

        assert wait.delays[0] == pytest.approx(0.5)


class TestFaults:
    def test_a_failing_tick_is_recorded_rather_than_raised(self):
        ticker, loop, _wait = _driven(stop_after=5)
        loop.raise_on_tick = OSError("the port went away")

        ticker.run_until_stopped()

        assert isinstance(ticker.fault(), OSError)

    def test_a_failing_tick_is_announced_while_it_still_matters(self):
        # Printed rather than only filed, for the same reason
        # _report_stall prints: somebody can walk out and free a boom
        # during a pass, and a report read afterwards is too late.
        clock = FakeClock()
        loop = FakeLoop(clock=clock)
        loop.raise_on_tick = OSError("the port went away")
        said: list[str] = []
        ticker = TrackingThread(
            loop,
            wait=ScriptedWait(clock, stop_after=5),
            monotonic=clock,
            report=said.append,
        )

        ticker.run_until_stopped()

        assert said == ["tracking stopped: the port went away"]

    def test_a_failing_tick_stops_ticking(self):
        # Continuing to command a rotor that just failed a serial round
        # trip is a stream of errors, not a recovery.
        ticker, loop, wait = _driven(stop_after=10)
        loop.raise_on_tick = OSError("gone")

        ticker.run_until_stopped()

        assert len(wait.delays) == 1

    def test_there_is_no_fault_when_nothing_went_wrong(self):
        ticker, _loop, _wait = _driven(stop_after=2)

        ticker.run_until_stopped()

        assert ticker.fault() is None


class TestStartAndStop:
    def test_start_ticks_once_before_the_thread_exists(self):
        # Points the antenna at the target before anything streams,
        # which is what the priming tick it replaces did.
        loop = FakeLoop(interval_s=60.0)
        ticker = TrackingThread(loop, report=lambda _message: None)

        ticker.start()
        try:
            assert loop.ticks == 1
            assert ticker.ticks == 1
        finally:
            ticker.stop()

    def test_a_rotor_that_cannot_be_reached_at_all_stops_the_run(self):
        """Failing to start and failing mid-pass are different faults.

        Mid-pass is a degradation to report. At startup there is no pass
        to protect yet, and a rotor that will not answer its very first
        command is a reason not to begin.
        """
        loop = FakeLoop()
        loop.raise_on_tick = OSError("nothing on COM5")
        ticker = TrackingThread(loop, report=lambda _message: None)

        with pytest.raises(OSError, match="nothing on COM5"):
            ticker.start()

        assert not ticker.is_running

    def test_starting_twice_is_an_error(self):
        loop = FakeLoop(interval_s=60.0)
        ticker = TrackingThread(loop, report=lambda _message: None)

        ticker.start()
        try:
            with pytest.raises(RuntimeError, match="already been started"):
                ticker.start()
        finally:
            ticker.stop()

    def test_stop_ends_the_thread(self):
        ticker = TrackingThread(FakeLoop(interval_s=0.01), report=lambda _message: None)

        ticker.start()
        ticker.stop()

        assert not ticker.is_running

    def test_stopping_something_never_started_is_harmless(self):
        # stop() runs from a finally that cannot know whether start()
        # got far enough to spawn anything.
        TrackingThread(FakeLoop(), report=lambda _message: None).stop()

    def test_it_does_not_keep_the_interpreter_alive(self):
        # A daemon thread, so a fault elsewhere cannot leave a process
        # that will not exit while it holds a serial port.
        ticker = TrackingThread(FakeLoop(interval_s=60.0), report=lambda _message: None)

        ticker.start()
        try:
            alive = [t for t in threading.enumerate() if t.name == "qsorbit-rotor-tracking"]
            assert alive and all(t.daemon for t in alive)
        finally:
            ticker.stop()


class TestDescribe:
    def test_it_reports_the_cadence_the_rotor_was_actually_commanded_at(self):
        # This line is part of a bench run's measurement record: it
        # answers the question the whole class exists for.
        ticker, _loop, _wait = _driven(0.5, stop_after=3)

        ticker.run_until_stopped()

        assert ticker.describe() == "tracking: 3 tick(s) at 0.5 s"

    def test_it_says_so_when_the_ticking_stopped_early(self):
        ticker, loop, _wait = _driven(0.5, stop_after=5)
        loop.raise_on_tick = OSError("the port went away")

        ticker.run_until_stopped()

        assert ticker.describe() == (
            "tracking: STOPPED after 0 tick(s) at 0.5 s - the port went away"
        )

    def test_a_clean_run_does_not_mention_stopping(self):
        # "STOPPED" has to mean something, so it must be absent from the
        # ordinary case rather than merely differently worded.
        ticker, _loop, _wait = _driven(0.5, stop_after=2)

        ticker.run_until_stopped()

        assert "STOPPED" not in ticker.describe()
