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
import time
from datetime import UTC, datetime

import pytest

from qsorbit.core.geometry import AzEl
from qsorbit.core.pointing import TickOutcome, TrackObservation, TrackSample
from qsorbit.core.rotor import Position
from qsorbit.core.track_log import CSV_COLUMNS, TrackLog
from qsorbit.core.tracking_thread import TrackingThread

AN_INSTANT = datetime(2026, 9, 3, 22, 0, tzinfo=UTC)


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

        self.observations = 0

    def tick(self) -> TrackSample:
        if self.raise_on_tick is not None:
            raise self.raise_on_tick
        self.ticks += 1
        if self._clock is not None:
            self._clock.now += self._tick_cost_s
        # Target and position deliberately differ, and differ per tick,
        # so a CSV assertion cannot pass by reading the wrong column.
        return TrackSample(
            time=AN_INSTANT,
            sky_position=AzEl(azimuth=10.0, elevation=20.0),
            range_km=1_000.0,
            range_rate_km_s=-3.0,
            rotor_target=Position(azimuth=10.0 + self.ticks, elevation=20.0),
            rotor_position=Position(azimuth=10.0, elevation=19.0),
            outcome=TickOutcome.COMMANDED,
        )

    def observe(self) -> TrackObservation:
        self.observations += 1
        return TrackObservation(
            time=AN_INSTANT,
            rotor_target=Position(azimuth=50.0 + self.observations, elevation=60.0),
            rotor_position=Position(azimuth=50.0, elevation=59.0),
        )


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


class TestLogging:
    """The logger is opt-in, and rides the thread that already owns the port."""

    def test_without_a_log_it_never_observes_at_all(self):
        # The whole cost of this feature -- extra reads and a target
        # computation per sample -- must not land on a run that did not
        # ask for it, because this is the path where CPU is measured to
        # turn into lost USB samples.
        ticker, loop, _wait = _driven(0.5, stop_after=4)

        ticker.run_until_stopped()

        assert loop.observations == 0
        assert ticker.describe_log() is None

    def test_a_sample_due_near_a_tick_defers_to_it(self, tmp_path):
        """The port-budget rule, and it is arithmetic rather than taste.

        Every tick and every sample is one read round trip and the link
        sustains about 5.9 a second. Two free-running clocks at 0.5 s
        and 0.2 s put a tick 0.1 s behind a sample and reach six reads a
        second -- over budget, which queues reads and jitters the tick.
        A tick reads position anyway, so a sample close to one defers.
        """
        clock = FakeClock()
        loop = FakeLoop(0.5, clock=clock)
        wait = ScriptedWait(clock, stop_after=4)
        with TrackLog(tmp_path / "t.csv") as log:
            ticker = TrackingThread(
                loop, log=log, wait=wait, monotonic=clock, report=lambda _m: None
            )
            ticker.run_until_stopped()

        # Samples land at 0.2 and 0.7; the 0.4 and 0.9 slots defer to
        # the ticks at 0.5 and 1.0 rather than crowding them. That is
        # one sample plus one tick per 0.5 s -- FOUR reads per second,
        # not five, and hardware agreed: a 60 s run measured 3.9 Hz with
        # gaps alternating 0.2 and 0.3. Six would need two samples
        # between ticks and is over the link's ~5.9 transactions.
        assert wait.delays == pytest.approx([0.2, 0.3, 0.2, 0.3, 0.2])
        assert loop.ticks == 2
        assert loop.observations == 2

    def test_tick_rows_carry_the_outcome_and_observed_rows_do_not(self, tmp_path):
        # An observation decided nothing. Naming a decision on its row
        # would let a later reader filter observations in as ticks.
        path = tmp_path / "t.csv"
        clock = FakeClock()
        loop = FakeLoop(0.5, clock=clock)
        with TrackLog(path) as log:
            ticker = TrackingThread(
                loop,
                log=log,
                wait=ScriptedWait(clock, stop_after=2),
                monotonic=clock,
                report=lambda _m: None,
            )
            ticker.run_until_stopped()

        rows = path.read_text(encoding="utf-8").splitlines()
        assert rows[0] == ",".join(CSV_COLUMNS)
        assert rows[1] == "0.200,51.00,50.00,60.00,59.00,"
        assert rows[2] == "0.500,11.00,10.00,20.00,19.00,commanded"

    def test_a_log_failure_stops_logging_and_not_tracking(self, tmp_path):
        # A full disk part-way through a pass is a reason to lose the
        # record, not the antenna.
        class BrokenLog(TrackLog):
            def record(self, *args, **kwargs):
                raise OSError("no space left on device")

        clock = FakeClock()
        loop = FakeLoop(0.5, clock=clock)
        said: list[str] = []
        with BrokenLog(tmp_path / "t.csv") as log:
            ticker = TrackingThread(
                loop,
                log=log,
                wait=ScriptedWait(clock, stop_after=5),
                monotonic=clock,
                report=said.append,
            )
            ticker.run_until_stopped()

        assert ticker.fault() is None
        assert loop.ticks >= 1
        # Announced once, not every 0.2 s until the console is useless.
        assert said == ["track log stopped: no space left on device"]

    def test_the_priming_tick_is_logged_at_zero(self, tmp_path):
        # A log that starts after the priming tick begins mid-slew with
        # no record of where the antenna came from.
        path = tmp_path / "t.csv"
        loop = FakeLoop(interval_s=60.0)
        with TrackLog(path) as log:
            ticker = TrackingThread(loop, log=log, report=lambda _m: None)
            ticker.start()
            try:
                rows = path.read_text(encoding="utf-8").splitlines()
            finally:
                ticker.stop()

        assert len(rows) == 2
        assert rows[1].startswith("0.000,")
        assert rows[1].endswith(",commanded")

    def test_elapsed_is_measured_from_before_the_priming_tick(self, tmp_path):
        """A real thread and a real clock, because that is where the bug lives.

        start() records the origin, then does the priming tick -- a
        serial round trip on real hardware -- and only then spawns the
        thread. If the thread timed from its own start instead, the
        priming row would sit at 0.000 and every row after it would be
        measured from a later instant, offsetting the whole file from
        its own first line by however long that round trip took.

        Every other test here drives run_until_stopped() directly, where
        the two origins coincide and the defect is invisible. This one
        makes the priming tick slow on purpose so they cannot.
        """
        path = tmp_path / "t.csv"

        class SlowFirstTick(FakeLoop):
            def tick(self):
                if self.ticks == 0:
                    time.sleep(0.1)
                return super().tick()

        loop = SlowFirstTick(interval_s=10.0)
        with TrackLog(path) as log:
            ticker = TrackingThread(loop, log=log, sample_interval_s=0.05, report=lambda _m: None)
            ticker.start()
            try:
                time.sleep(0.2)
            finally:
                ticker.stop()

        rows = path.read_text(encoding="utf-8").splitlines()
        assert len(rows) >= 3, "expected the priming row and at least one sample"
        first_sample_s = float(rows[2].split(",")[0])
        # Timed from the thread it would be ~0.05; timed from start() it
        # carries the 0.1 s priming tick as well.
        assert first_sample_s > 0.12
