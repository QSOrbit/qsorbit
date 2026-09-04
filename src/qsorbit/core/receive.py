"""The receive path — Phase 2's vertical slice, wired into one object.

Everything this module needs already existed and was tested in isolation.
What was missing was the wiring, and the wiring is where the interesting
failures live, because it is the only place where the tracking side and
the receiving side have to agree about anything.

**What runs where.** Three threads, and the division is not arbitrary:

``IqStream``'s reader
    Owned by :mod:`qsorbit.core.sdr.stream`. Reads the device and fans
    each block out to every subscription. Nothing here touches it.

the demodulating thread
    Owned by this module. Pulls :class:`~qsorbit.core.sdr.stream.TimedBlock`
    from the ``"audio"`` subscription, asks the Doppler tracker where the
    downlink is *at that block's midpoint*, demodulates, and writes the
    audio out. This is the closest thing here to a real-time path, so it
    shares a thread with nothing else — in particular not with the rotor,
    whose serial round trips take 0.15 s of RS-485 turnaround apiece.

the range-rate thread
    Feeds the Doppler tracker on its own cadence, from
    :class:`TargetRangeRate` -- the TLE and the observer's location,
    which is where a range rate actually comes from. **No rotor is
    involved and none ever was**: for a while this thread also held the
    rotor's tick, by way of a range-rate source that ticked the loop to
    get its number, and the side effect was that the rotor was commanded
    at *this* cadence rather than at the one its tracking profile
    declared. The tick now lives in
    :class:`~qsorbit.core.tracking_thread.TrackingThread`, which is what
    makes "this module does not insist on owning the tracking side" true
    rather than aspirational.

A ``SpectrumStream``, when one is given, gets the ``"waterfall"``
subscription and runs its own worker as it always has.

**The rotor is optional, and that is a design statement rather than a
convenience.** Doppler correction needs a range rate, and a range rate
comes from the TLE and the observer's location — not from the rotor. So
the entire radio job runs with nothing connected to COM5, and moving the
antenna is something added on top. On a bench day where several things
could be wrong at once, a rotor fault therefore does not cost you the
pass. It also matches ``point``'s standing asymmetry: computing is the
default, moving is opt-in.

**The tracker is primed before the reader starts.**
:meth:`~qsorbit.core.dsp.tuning.DopplerTracker.offset_at` raises if it
has never been given a range rate, and the demodulating thread can reach
its first block before the tracking side has produced anything. Rather
than skip those blocks and count them, :meth:`ReceiveSession.start`
takes one sample up front. Priming deletes the race; counting would only
have measured it.

**Nothing here counts what something else already counts.** Stale Doppler
queries live in :class:`~qsorbit.core.dsp.tuning.DopplerStats`, buffer
drops in :class:`~qsorbit.core.sdr.stream.StreamStats`, underruns in
:class:`~qsorbit.core.dsp.audio.AudioStats`. Duplicating any of them here
would only create two numbers with two chances to disagree — the same
reasoning :class:`~qsorbit.core.dsp.spectrum_stream.SpectrumStreamStats`
gives for not counting the IQ side's drops a second time.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Final, Protocol

from qsorbit.core.dsp.audio import AudioOutput, AudioStats
from qsorbit.core.dsp.demod import NbfmConfig, demodulate_nbfm
from qsorbit.core.dsp.iq import unpack_uint8_iq
from qsorbit.core.dsp.spectrum_stream import SpectrumStream, SpectrumStreamStats
from qsorbit.core.dsp.squelch import NoiseSquelch, SquelchStats
from qsorbit.core.dsp.tuning import DopplerStats, DopplerTracker
from qsorbit.core.sdr.stream import IqStream, StreamStats
from qsorbit.core.tracker.observer import ObserverLocation
from qsorbit.core.tracker.target import Target

#: Subscription name for the demodulating consumer.
AUDIO_SUBSCRIBER: Final = "audio"

#: Subscription name for the spectrum/waterfall consumer.
WATERFALL_SUBSCRIBER: Final = "waterfall"

#: Seconds between range-rate samples when this module drives the
#: tracking side itself. One second matches
#: :class:`~qsorbit.core.pointing.TrackingLoop`'s own default cadence and
#: :class:`~qsorbit.ui.readout_widget.ReadoutWidget`'s poll interval, so
#: headless and windowed runs feed the tracker at the same rate and their
#: measurements stay comparable.
DEFAULT_TRACKING_INTERVAL_S: Final = 1.0

#: How long :meth:`ReceiveSession.stop` waits for each of its threads.
#: The demodulating thread checks for the stop signal between blocks, so
#: normal latency is one block — about 64 ms at 2.048 Msps.
DEFAULT_JOIN_TIMEOUT_S: Final = 5.0


def _utc_now() -> datetime:
    """The current instant, timezone-aware. Matches the rest of the project."""
    return datetime.now(UTC)


class RangeRateSource(Protocol):
    """Where the Doppler tracker's range-rate samples come from.

    Declared structurally rather than as a base class, matching
    :class:`~qsorbit.core.tracker.Target` and
    :class:`~qsorbit.ui.waterfall_widget.FrameSource`: a test double
    satisfies it by having the two methods, without importing anything.

    The split between the two methods is the whole reason this protocol
    exists. :meth:`prime` must always produce a sample, because it is
    what removes the race described in the module docstring.
    :meth:`sample` is allowed to say "nothing new yet", because a source
    that merely *follows* a loop somebody else is ticking genuinely has
    nothing to report between ticks.
    """

    def prime(self) -> tuple[datetime, float]:
        """Produce one sample now, before anything starts streaming.

        Returns:
            ``(time, range_rate_km_s)``, positive when receding.
        """
        ...

    def sample(self) -> tuple[datetime, float] | None:
        """The next sample, or ``None`` if none has arrived yet."""
        ...


class TargetRangeRate:
    """Range rates computed straight from the target. No rotor involved.

    What ``receive`` uses when the antenna is not being moved. Every
    sample is available on demand, so :meth:`prime` and :meth:`sample`
    are the same operation.

    Args:
        target: What is being received from.
        observer: The ground station's location.
        now: Clock, injected for tests.
    """

    def __init__(
        self,
        target: Target,
        observer: ObserverLocation,
        *,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._target = target
        self._observer = observer
        self._now = now

    def prime(self) -> tuple[datetime, float]:
        """Compute a sample now."""
        return self.sample()

    def sample(self) -> tuple[datetime, float]:
        """Compute a sample now. Never ``None`` — nothing can be pending."""
        when = self._now()
        state = self._target.topocentric_state(self._observer, when)
        return when, state.range_rate_km_s


@dataclass(frozen=True)
class ReceiveStats:
    """What one receive session did, in the pieces that have owners.

    Args:
        blocks_demodulated: Blocks that went through
            :func:`~qsorbit.core.dsp.demod.demodulate_nbfm`.
        range_rate_updates: Samples handed to the Doppler tracker,
            including the priming one.
        stream: The IQ side, including per-consumer drop accounting.
        audio: Playback, including underruns.
        doppler: Correction range and how many queries ran on a stale
            range rate. **This is where staleness is reported** — the
            session does not keep its own count of it.
        squelch: Present only if a squelch was in use.
        spectrum: Present only if a waterfall was being fed.
        stopped_cleanly: Whether both of this module's threads exited
            within their join timeout.
    """

    blocks_demodulated: int
    range_rate_updates: int
    stream: StreamStats
    audio: AudioStats
    doppler: DopplerStats
    squelch: SquelchStats | None
    spectrum: SpectrumStreamStats | None
    stopped_cleanly: bool

    def describe(self) -> str:
        """Summarise the whole slice, one owner per section.

        Printed at the end of a bench run, so this *is* the measurement
        record. Sections that were not in use say so rather than being
        omitted: a missing line reads as zero, and "the squelch was off"
        and "the squelch never opened" are different facts.
        """
        clean = "" if self.stopped_cleanly else "receive: threads DID NOT stop cleanly\n"
        squelch = (
            self.squelch.describe()
            if self.squelch is not None
            else "squelch: off, so no quieting was measured this run."
        )
        spectrum = (
            self.spectrum.describe()
            if self.spectrum is not None
            else "spectrum: no waterfall was attached this run.\n"
        )
        return (
            f"{clean}"
            f"receive: {self.blocks_demodulated:,} block(s) demodulated, "
            f"{self.range_rate_updates:,} range-rate update(s)\n"
            f"\n--- iq ---\n{self.stream.describe()}"
            f"\n--- audio ---\n{self.audio.describe()}\n"
            f"\n--- doppler ---\n{self.doppler.describe()}\n"
            f"\n--- squelch ---\n{squelch}\n"
            f"\n--- spectrum ---\n{spectrum}"
        )


class ReceiveSession:
    """Runs the tracking side and the receive chain together.

    Usage::

        session = ReceiveSession(
            stream=IqStream(sdr),
            nbfm=nbfm_config,
            doppler=DopplerTracker(downlink_hz, applied.center_hz),
            audio=AudioOutput(nbfm_config.audio_rate_hz),
            range_rate=TargetRangeRate(satellite, observer),
        )
        with session:
            time.sleep(300)
        print(session.stats.describe())

    Args:
        stream: An :class:`~qsorbit.core.sdr.stream.IqStream` over a
            configured device. **Must not have been started or
            subscribed to** — this session subscribes once or
            twice, and subscriptions have to exist before the reader
            does. The spectrum consumer is subscribed only when a
            ``spectrum_factory`` is given, so a headless run never offers
            blocks to a consumer that will not drain them.
        nbfm: Demodulation settings. ``channel_offset_hz`` is replaced
            per block with the Doppler-corrected offset, so whatever it
            holds here is ignored; everything else is used as given.
        doppler: The tracker, built against the centre frequency the
            tuner **actually reached**.
        audio: Where the recovered audio goes.
        range_rate: Where range-rate samples come from. See
            :class:`RangeRateSource`.
        squelch: Optional noise gate, off by default — see
            :mod:`qsorbit.core.dsp.squelch` for why a mute enabled by
            default is a liability. One per session; it is stateful.
            Passing one always turns on *measurement*, whether or not
            ``mute_squelch`` also turns on muting - see that argument.
        mute_squelch: Whether a closed gate actually silences audio.
            Ignored when ``squelch`` is ``None``. Defaults to ``True``,
            matching this class's behaviour before this parameter
            existed. ``False`` measures and reports quieting exactly as
            if muting were on (:attr:`live_quieting_db`, and
            :class:`~qsorbit.core.dsp.squelch.SquelchStats` in the final
            report), without ever letting the gate's decision reach the
            speaker - see :func:`~qsorbit.core.dsp.demod.demodulate_nbfm`
            for the mechanics this threads through to.
        spectrum: Optional. When given it is started with the
            ``"waterfall"`` subscription and stopped with the session.
        tracking_interval_s: Seconds between range-rate samples.
        join_timeout_s: How long :meth:`stop` waits per thread.

    Raises:
        ValueError: If ``tracking_interval_s`` is not positive.
    """

    def __init__(
        self,
        *,
        stream: IqStream,
        nbfm: NbfmConfig,
        doppler: DopplerTracker,
        audio: AudioOutput,
        range_rate: RangeRateSource,
        squelch: NoiseSquelch | None = None,
        mute_squelch: bool = True,
        spectrum_factory: Callable[[Iterable[bytes]], SpectrumStream] | None = None,
        tracking_interval_s: float = DEFAULT_TRACKING_INTERVAL_S,
        join_timeout_s: float = DEFAULT_JOIN_TIMEOUT_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if tracking_interval_s <= 0.0:
            raise ValueError(f"tracking_interval_s must be positive, got {tracking_interval_s!r}.")

        self._stream = stream
        self._nbfm = nbfm
        self._doppler = doppler
        self._audio = audio
        self._range_rate = range_rate
        self._squelch = squelch
        self._mute_squelch = mute_squelch
        self._tracking_interval_s = tracking_interval_s
        self._join_timeout_s = join_timeout_s
        self._sleep = sleep

        # Subscribed here rather than in start(), because subscriptions
        # must exist before the reader thread does and a caller is
        # entitled to hold the waterfall subscription before starting.
        self._audio_blocks = stream.subscribe(AUDIO_SUBSCRIBER)
        # The waterfall subscription is made only when something will
        # actually drain it. It used to be unconditional, so a headless
        # run offered every block to a consumer that did not exist and
        # the bounded deque evicted them in turn: a 60-second headless
        # `receive` reported "453 block(s) dropped (118,751,232 bytes)"
        # with nothing whatever wrong (Session 24). Harmless, and it
        # reads as catastrophic data loss -- and this project's own rule
        # is that "off" and "broken" must never look the same.
        if spectrum_factory is None:
            self._waterfall_blocks = None
            self._spectrum = None
        else:
            self._waterfall_blocks = stream.subscribe(WATERFALL_SUBSCRIBER)
            self._spectrum = spectrum_factory(self._waterfall_blocks.blocks())

        self._stop = threading.Event()
        self._demod_thread: threading.Thread | None = None
        self._tracking_thread: threading.Thread | None = None
        self._error: BaseException | None = None
        # Kept apart from _error deliberately: see tracking_error().
        self._tracking_error: BaseException | None = None
        self._stopped_cleanly = True

        self._lock = threading.Lock()
        self._blocks_demodulated = 0
        self._range_rate_updates = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True`` while the demodulating thread is alive."""
        return self._demod_thread is not None and self._demod_thread.is_alive()

    @property
    def spectrum(self) -> SpectrumStream | None:
        """The spectrum stream, for a widget that needs to be handed one."""
        return self._spectrum

    def tracking_error(self) -> BaseException | None:
        """Whatever stopped the range-rate thread, or ``None`` if nothing has.

        **This is no longer a rotor fault, and the change of meaning is
        worth stating rather than leaving to be inferred.** While this
        thread also held the rotor's tick, a fault here usually *was* a
        serial fault, and a readout took it as one. The tick now belongs
        to :class:`~qsorbit.core.tracking_thread.TrackingThread`, whose
        own :meth:`~qsorbit.core.tracking_thread.TrackingThread.fault`
        is what a rotor readout should ask. What is left here is the
        Doppler side: a range-rate source that stopped producing, which
        on the receive path means propagation rather than hardware.

        **Deliberately separate from the demodulating thread's error**,
        which :meth:`stop` re-raises. They are different faults with
        different consequences -- losing range rates leaves the audio
        playing, uncorrected and drifting, which is a degradation to
        report rather than a reason to tear the session down mid-pass.

        A method rather than a property so it can be handed to a widget
        as a plain callable, without the caller having to wrap it.
        """
        return self._tracking_error

    def start(self) -> None:
        """Prime the tracker, then start everything. Starting twice is an error.

        Order matters and is the opposite of the obvious one. The tracker
        is primed **before** any thread starts, so that by the time the
        first block can possibly arrive there is already a range rate to
        correct it with. Starting the reader first and priming after
        would reintroduce exactly the race the priming exists to remove,
        just with a smaller window.
        """
        if self._demod_thread is not None:
            raise RuntimeError("This session has already been started; build a new one.")

        when, range_rate_km_s = self._range_rate.prime()
        self._doppler.update(when, range_rate_km_s)
        self._range_rate_updates = 1

        self._audio.start()
        # Started here rather than left to whichever consumer reaches its
        # first block first. Both would call the same start-if-needed
        # path, and starting it explicitly means the reader is running
        # before either consumer exists rather than as a side effect of
        # one of them.
        self._stream.start()
        if self._spectrum is not None:
            self._spectrum.start()

        self._demod_thread = threading.Thread(
            target=self._demod_loop, name="qsorbit-receive-demod", daemon=True
        )
        self._tracking_thread = threading.Thread(
            target=self._tracking_loop, name="qsorbit-receive-tracking", daemon=True
        )
        self._demod_thread.start()
        self._tracking_thread.start()

    def wait(self, timeout_s: float | None = None) -> bool:
        """Block until the demodulating thread ends, or until ``timeout_s``.

        The demodulating thread ends when the blocks stop — the device
        was unplugged, the fake source ran out, or :meth:`stop` was
        called. So a caller that would otherwise sleep out a fixed
        duration can wait on this instead and **find out promptly that
        the radio died**, rather than sitting through the rest of a pass
        with nothing arriving. Whatever it died of is then raised by
        :meth:`stop`.

        Args:
            timeout_s: Seconds to wait, or ``None`` to wait indefinitely.

        Returns:
            ``True`` if the thread has ended, ``False`` if the timeout
            expired with it still running — which is the normal outcome
            of a run that lasted its full duration.

        Raises:
            RuntimeError: If the session was never started.
        """
        thread = self._demod_thread
        if thread is None:
            raise RuntimeError("This session has not been started, so there is nothing to wait on.")
        thread.join(timeout_s)
        return not thread.is_alive()

    def stop(self) -> ReceiveStats:
        """Stop everything this session started, and return the statistics.

        Stops in the order a consumer would want: the reader first, so
        no more blocks arrive, then this module's own threads, then the
        audio device. Whatever the demodulating thread died of, if
        anything, is re-raised here — a receive session that stops
        silently is the failure mode this whole project keeps meeting.

        The rotor is deliberately **not** stopped, matching
        :class:`~qsorbit.core.pointing.TrackingLoop` and
        :meth:`~qsorbit.core.rotor.Rotor.__exit__`: a move already in
        progress does not need us, and abandoning the antenna mid-slew
        is no improvement on letting it arrive.

        Raises:
            Whatever killed the demodulating thread.
        """
        self._stop.set()
        stream_stats = self._stream.stop()
        spectrum_stats = self._spectrum.stop() if self._spectrum is not None else None

        clean = True
        for thread in (self._demod_thread, self._tracking_thread):
            if thread is not None:
                thread.join(self._join_timeout_s)
                clean = clean and not thread.is_alive()
        self._stopped_cleanly = clean

        audio_stats = self._audio.stop()
        stats = self._build_stats(stream_stats, audio_stats, spectrum_stats)

        error = self._error
        if error is not None:
            raise error
        return stats

    @property
    def stats(self) -> ReceiveStats:
        """The run's statistics. Stable once :meth:`stop` has returned."""
        return self._build_stats(
            self._stream.stats,
            self._audio.stats,
            self._spectrum.stats if self._spectrum is not None else None,
        )

    @property
    def live_quieting_db(self) -> float | None:
        """The squelch's most recent quieting measurement, for a live display.

        ``None`` when no squelch was given at all - there is nothing to
        show. Otherwise this is the number a live "quieting" readout
        polls, updating every block regardless of ``mute_squelch``: see
        that argument's docstring for why a run with muting off still
        has a real, moving number here.

        Unlike :attr:`stats` (**not** safe to call this "live" - its own
        docstring says it is stable only once :meth:`stop` has
        returned), this property is meant to be polled *while the
        session is running*, from a different thread than the one
        updating the squelch. It is deliberately not guarded by
        :attr:`_lock`: the value it reads is a single float, reassigned
        as a whole on every block by
        :meth:`~qsorbit.core.dsp.squelch.NoiseSquelch.update`, so a
        concurrent read can only ever see the value from just before or
        just after an update, never a torn one - the CPython GIL makes a
        single attribute assignment atomic. A live gauge redrawn several
        times a second tolerates being one block stale; :attr:`_lock`
        exists for the counters that end up in a report and have to add
        up exactly, which this number does not.
        """
        if self._squelch is None:
            return None
        return self._squelch.stats.last_quieting_db

    @property
    def live_squelch_open(self) -> bool | None:
        """Whether the gate is open right now, or ``None`` if there is no squelch.

        The gate's *decision*, exactly as :attr:`live_quieting_db` is its
        *measurement* - both real even when ``mute_squelch=False`` never
        lets that decision reach the speaker. Same polling contract as
        :attr:`live_quieting_db`: a single attribute read, safe enough
        for a live display, not for a report that has to add up.
        """
        if self._squelch is None:
            return None
        return self._squelch.is_open

    @property
    def live_tracked_frequency_hz(self) -> float | None:
        """The tracked downlink's true RF frequency right now, or ``None``.

        ``None`` until the tracking loop has supplied the Doppler
        tracker its first sample - :attr:`~qsorbit.core.dsp.tuning.DopplerTracker.stats`
        reports that as ``last_offset_hz is None``, and there is no
        honest frequency to report before then. Once a sample has
        landed, this is the tuner's own centre
        (:attr:`~qsorbit.core.dsp.tuning.DopplerTracker.center_hz`, fixed
        for the run) plus the most recent Doppler offset - the same two
        numbers :meth:`_demod_loop` combines every block to pick the
        demod's own ``channel_offset_hz``, so this property always
        matches where the audio the user is hearing actually sits, not
        a separately recomputed estimate.

        Same live-polling contract as :attr:`live_quieting_db`: meant to
        be read from a different thread than the one updating it, while
        the session runs. :attr:`~qsorbit.core.dsp.tuning.DopplerTracker.stats`
        already takes its own lock to hand back a consistent snapshot,
        so no additional locking is needed here.
        """
        offset_hz = self._doppler.stats.last_offset_hz
        if offset_hz is None:
            return None
        return self._doppler.center_hz + offset_hz

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ReceiveSession:
        """Start on entering a ``with`` block."""
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Stop on leaving, whether or not the body raised.

        A failure inside the ``with`` body wins: :meth:`stop` re-raises
        whatever killed the demodulating thread, and letting that replace
        the caller's own exception would hide the first fault behind a
        consequence of it.
        """
        try:
            self.stop()
        except BaseException:
            if exc_type is None:
                raise

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _demod_loop(self) -> None:
        """Demodulate every block at its own Doppler-corrected offset."""
        try:
            for block in self._audio_blocks.timed_blocks():
                if self._stop.is_set():
                    break
                # The block's MIDPOINT, not either edge: it removes a
                # systematic half-block bias for free, and TimedBlock
                # computes it so no caller can get the sign wrong.
                offset_hz = self._doppler.offset_at(block.midpoint)
                config = replace(self._nbfm, channel_offset_hz=offset_hz)
                audio = demodulate_nbfm(
                    unpack_uint8_iq(block.data),
                    config,
                    squelch=self._squelch,
                    mute=self._mute_squelch,
                )
                self._audio.write(audio)
                with self._lock:
                    self._blocks_demodulated += 1
        except BaseException as exc:  # noqa: BLE001 - re-raised from stop()
            self._error = exc

    def _tracking_loop(self) -> None:
        """Feed the Doppler tracker on a cadence until asked to stop.

        Failures here are recorded the same way the demodulating thread's
        are, rather than being allowed to kill the thread quietly. A
        tracking side that stops feeding does not stop the audio — the
        tracker extrapolates, then holds, and says how long it did so in
        :class:`~qsorbit.core.dsp.tuning.DopplerStats` — so this is a
        degradation to report rather than a reason to tear the session
        down mid-pass.
        """
        try:
            while not self._stop.wait(self._tracking_interval_s):
                pending = self._range_rate.sample()
                if pending is None:
                    continue
                when, range_rate_km_s = pending
                self._doppler.update(when, range_rate_km_s)
                with self._lock:
                    self._range_rate_updates += 1
        except BaseException as exc:  # noqa: BLE001 - re-raised from stop()
            # Recorded twice, on purpose. _error is what stop() re-raises
            # so the run cannot end silently; _tracking_error is what a
            # following readout reads, and it must not be confused with a
            # demodulation fault, which says nothing about the rotor.
            self._tracking_error = exc
            if self._error is None:
                self._error = exc

    def _build_stats(
        self,
        stream_stats: StreamStats,
        audio_stats: AudioStats,
        spectrum_stats: SpectrumStreamStats | None,
    ) -> ReceiveStats:
        """Assemble a snapshot from each owner's own accounting."""
        with self._lock:
            blocks = self._blocks_demodulated
            updates = self._range_rate_updates
        return ReceiveStats(
            blocks_demodulated=blocks,
            range_rate_updates=updates,
            stream=stream_stats,
            audio=audio_stats,
            doppler=self._doppler.stats,
            squelch=self._squelch.stats if self._squelch is not None else None,
            spectrum=spectrum_stats,
            stopped_cleanly=self._stopped_cleanly,
        )
