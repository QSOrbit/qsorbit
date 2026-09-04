"""Drives a :class:`~qsorbit.core.pointing.TrackingLoop` on a thread of its own.

**Why this exists as its own object rather than as work inside
:class:`~qsorbit.core.receive.ReceiveSession`.** Until Chunk E opened,
the rotor's tick was performed by ``LoopRangeRate.sample()`` on the
session's range-rate thread -- which meant the rotor was commanded at
``tracking_interval_s``, the Doppler sampler's cadence, and never at the
cadence its own tracking profile declared. Chunk H PR1 decoupled the two
*config values* and the coupling survived anyway, because it did not run
through a value: it ran through **who held the tick**.

So the tick moves out. The session's module docstring already claimed
that it "does not insist on owning" the tracking side and that the rotor
is optional; with this class in place that is true again, and
:class:`~qsorbit.core.receive.TargetRangeRate` -- which computes range
rate from the TLE and the observer, with no rotor involved -- becomes
the only range-rate source the receive path uses.

**This does not undo Chunk A PR2.** That fix moved the tick off the *GUI*
thread, because ``tick()`` writes a command, sleeps out an RS-485
turnaround and then blocks on a read, and doing that on the GUI thread
froze the interface for a sixth of a second every second. The tick stays
off the GUI thread here; it simply gets a thread of its own rather than
sharing the Doppler sampler's.

**The cadence is scheduled against a deadline, and that is load-bearing
arithmetic rather than tidiness.** The obvious loop -- ``while not
stop.wait(interval)`` -- measures from the *end* of the previous
iteration, so the true period is the interval plus however long the tick
took. On the Doppler sampler that drift is harmless. On the rotor it is
not: the commanded step is ``rate x tick``, so a 0.5 s cadence carrying
a 0.15 s serial turnaround would really command 0.65 deg steps, and the
validated set would never reach the hardware even after this class
exists. Worse, ``deadband == rate x interval`` is a knife edge that
silently doubles the step (Session 32), and the guard that refuses that
configuration checks the *configured* interval, not the achieved one.
:meth:`~qsorbit.core.pointing.TrackingLoop.run` already schedules
against a deadline for this reason; this class does the same, while
staying promptly stoppable by waiting on the stop event rather than
sleeping.

**The interval is re-read every iteration**, so a live profile switch
reaches the schedule without anything having to tell it -- the same
contract ``run()`` and ``_GuiThreadTicker`` both already follow, and the
reason :attr:`~qsorbit.core.pointing.TrackingLoop.interval_s` is a
property rather than a constructor argument that got copied.

**Failing to start and failing mid-pass are different faults and get
different answers.** :meth:`start` ticks once synchronously, on the
caller's thread, before the thread exists -- it points the antenna at
the target rather than leaving it wherever it was parked, exactly as the
priming tick it replaces did, and if the rotor cannot be talked to at
all that raises and nothing else starts. A failure *during* a pass is
recorded, reported to the operator the instant it happens, and ends this
thread only: the audio keeps playing and the run's statistics still get
printed. ``receive``'s own docstring has promised since Phase 2 that "a
rotor fault therefore does not cost you the pass", and while the tick
shared the session's thread that was only half true -- the recorded
error was re-raised out of ``stop()``, which propagated through the
caller's ``finally`` and took the whole statistics report with it. A
serial hiccup at minute 3 destroyed the measurement record of a
twenty-minute pass.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import Final

from qsorbit.core.pointing import TrackingLoop

#: How long :meth:`TrackingThread.stop` waits for the thread to end.
#: A tick is bounded by the serial port's own timeout, so the normal
#: latency here is one tick.
DEFAULT_JOIN_TIMEOUT_S: Final = 5.0


def _report(message: str) -> None:
    """Tell the operator, on stderr, while it still matters."""
    print(message, file=sys.stderr)


class TrackingThread:
    """Ticks a tracking loop on its own thread, at the loop's own cadence.

    Not for the rotor-only shell. That path ticks from a ``QTimer`` on
    the GUI thread on purpose -- with no SDR reader running there is
    nothing for a blocking read to starve, and the rotor's own measured
    cost is 18-30 ms over a whole pass -- see
    ``_run_shell_tracking_only``. This class is for the configurations
    where a radio is streaming.

    Args:
        loop: The loop to tick. Its
            :attr:`~qsorbit.core.pointing.TrackingLoop.interval_s` is
            read afresh before every wait, so a profile switch changes
            the cadence without this object being told.
        join_timeout_s: How long :meth:`stop` waits for the thread.
        report: Where a mid-pass fault is announced. Injected for tests;
            defaults to a line on stderr, matching ``_report_stall`` --
            a stopped rotor is actionable *during* a pass, so it is
            worth interrupting for rather than filing in a report
            nobody reads until afterwards.
        wait: Blocks for a number of seconds and returns ``True`` if a
            stop was requested meanwhile. Defaults to this object's own
            stop event. Injected by tests so the schedule can be driven
            without threads or real time.
        monotonic: Clock, injected for tests.
    """

    def __init__(
        self,
        loop: TrackingLoop,
        *,
        join_timeout_s: float = DEFAULT_JOIN_TIMEOUT_S,
        report: Callable[[str], None] = _report,
        wait: Callable[[float], bool] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._loop = loop
        self._join_timeout_s = join_timeout_s
        self._report = report
        self._monotonic = monotonic

        self._stop = threading.Event()
        self._wait = wait if wait is not None else self._stop.wait
        self._thread: threading.Thread | None = None

        self._lock = threading.Lock()
        self._ticks = 0
        self._error: BaseException | None = None
        self._stopped_cleanly = True

    @property
    def is_running(self) -> bool:
        """``True`` while the ticking thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def ticks(self) -> int:
        """Ticks performed, including the synchronous priming one."""
        with self._lock:
            return self._ticks

    def fault(self) -> BaseException | None:
        """Whatever stopped the ticking, or ``None`` if nothing has.

        A method rather than a property so it can be handed to a widget
        or a feed hub as a plain callable, matching
        :meth:`~qsorbit.core.receive.ReceiveSession.tracking_error` and
        ``_GuiThreadTicker.fault``, both of which a readout already
        knows how to consume.
        """
        with self._lock:
            return self._error

    def start(self) -> None:
        """Tick once here, then hand the cadence to a thread of its own.

        The first tick is synchronous and runs on the caller's thread,
        before anything is streaming. That is deliberate on two counts:
        it leaves the antenna pointed at the target rather than wherever
        it was parked, which is what you would have done by hand anyway;
        and a rotor that cannot be reached at all should stop the run
        *before* the pass rather than half a second into it, so this one
        raises rather than being recorded.

        Raises:
            RuntimeError: If already started.
            Whatever the first tick raises.
        """
        if self._thread is not None:
            raise RuntimeError("This ticker has already been started; build a new one.")

        self._loop.tick()
        with self._lock:
            self._ticks = 1

        self._thread = threading.Thread(
            target=self.run_until_stopped, name="qsorbit-rotor-tracking", daemon=True
        )
        self._thread.start()

    def run_until_stopped(self) -> None:
        """Tick on the loop's cadence until stopped or until a tick fails.

        Public so a test can drive it directly with an injected ``wait``
        and clock, rather than starting a thread and sleeping through
        real seconds to find out what it did. The thread targets exactly
        this.

        Assumes a tick has just happened, which is what :meth:`start`
        guarantees -- so the first wait here is a full interval rather
        than zero, and the antenna is not commanded twice in a row at
        startup.
        """
        next_tick_at = self._monotonic() + self._loop.interval_s
        while True:
            remaining = next_tick_at - self._monotonic()
            if self._wait(remaining if remaining > 0.0 else 0.0):
                return
            if remaining <= 0.0:
                # The tick overran its interval. Restart the schedule
                # from now rather than firing repeatedly to catch up on
                # ticks that were missed: a late pointing update is
                # worth sending, a stale one is not. Same policy as
                # TrackingLoop.run().
                next_tick_at = self._monotonic()

            try:
                self._loop.tick()
            except BaseException as exc:  # noqa: BLE001 - recorded and reported, not raised
                # Raising here would kill this thread with nobody
                # watching and leave a readout showing its last
                # plausible numbers under a dead rotor -- the silent
                # failure this project keeps meeting. Recording it where
                # the readout looks, and saying so out loud, is the
                # honest outcome.
                with self._lock:
                    self._error = exc
                self._report(f"tracking stopped: {exc}")
                return

            with self._lock:
                self._ticks += 1
            # Re-read rather than cache: a profile switch is applied
            # inside tick(), by the loop, so the schedule finds out the
            # same way everything else does.
            next_tick_at += self._loop.interval_s

    def stop(self) -> None:
        """Stop ticking, and wait for the thread to notice.

        **The rotor is deliberately not stopped**, matching
        :class:`~qsorbit.core.pointing.TrackingLoop` and
        :meth:`~qsorbit.core.rotor.Rotor.__exit__`: a move already in
        progress does not need us, and abandoning the antenna mid-slew
        is no improvement on letting it arrive.

        Safe to call on a ticker that was never started, and safe to
        call twice.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(self._join_timeout_s)
            self._stopped_cleanly = not thread.is_alive()

    def describe(self) -> str:
        """One line for the end-of-run report.

        Printed beside the receive statistics, so this is part of the
        measurement record: it says at what cadence the rotor was
        actually commanded, which is the whole question this class was
        written to answer, and it says so even when the ticking stopped
        early.
        """
        with self._lock:
            ticks = self._ticks
            error = self._error
        cadence = f"{self._loop.interval_s:.3g} s"
        if error is not None:
            return f"tracking: STOPPED after {ticks:,} tick(s) at {cadence} - {error}"
        if not self._stopped_cleanly:
            return f"tracking: {ticks:,} tick(s) at {cadence}, thread DID NOT stop cleanly"
        return f"tracking: {ticks:,} tick(s) at {cadence}"

    def __enter__(self) -> TrackingThread:
        """Start on entering a ``with`` block."""
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Stop on leaving, whether or not the body raised."""
        self.stop()
