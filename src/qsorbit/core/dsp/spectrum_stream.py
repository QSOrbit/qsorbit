"""Spectrum frames produced off the GUI thread, at a rate a display can use.

This is the Qt half of the streaming-architecture question, settled in
Chunk F — except that nothing in this module imports Qt, which is the
whole point of how it was settled.

**The arithmetic that decided the design.** One 256 KiB block from
:class:`~qsorbit.core.sdr.stream.IqStream` is 131,072 complex samples,
64 ms at 2.048 Msps. At an ``fft_size`` of 2048 that block contains 64
non-overlapping frames, so a naive pipeline produces about **1,000
spectrum rows per second**. A waterfall panel is a few hundred rows tall
and repaints at 30 Hz at the very most. The pipeline is therefore capable
of thirty to sixty times more frames than any display can consume, and
the real question is never "how do we go fast enough" — it is *where the
excess is discarded, and whether that discard is deliberate*.

**Why the GUI thread pulls instead of being pushed to.** The conventional
Qt answer is a worker emitting a signal per frame across a queued
connection. At 1,000 frames per second that posts 1,000 events per second
onto the GUI thread's event queue, which Qt will accept without complaint:
the queue grows without bound, the UI degrades steadily, memory climbs,
and nothing ever raises. Instead the worker here fills a *bounded* buffer
and the widget drains it on its own timer, so the consumer can never be
handed more than it asked for and back-pressure is structural rather than
something a future maintainer has to remember. It also keeps this module
Qt-free, which is what makes it testable headless and reusable by a
non-UI consumer — a headless spectrum logger, or Chunk H's ``receive``
path.

**Why the FFTs are not on the reader thread.** Session 16's finding, by
name: anything on the read path is on a real-time path. A quarter-million
element list comprehension in the ctypes binding cost 6.7 ms per read and
presented as 9.5% USB loss. numpy FFTs on that same thread would
reproduce the problem deliberately. Hence a worker thread between the
reader and the GUI, and hence three threads in total: reader (USB),
worker (DSP), GUI (paint).

**Frames not computed are not frames lost**, and this module counts them
separately for the same reason
:class:`~qsorbit.core.sdr.stream.StreamStats` separates ``blocks_dropped``
from ``loss`` and :class:`~qsorbit.core.dsp.audio.AudioStats` separates
``blocks_dropped`` from ``underruns``: they have different causes and one
number would send you at the wrong one. Skipping 97% of available frames
is the design working. Dropping even one *computed* frame at the buffer
means the consumer fell behind, which is a real fault wearing the same
clothes.

Being frugal about which frames get computed is not premature
optimisation. Chunk E measured audio underruns at 0.7% under CPU pressure
from a filter rebuild, and Chunk G will run SDR, demodulation, audio and
this waterfall at once. Computing 3% of the available FFTs instead of
100% is what keeps that from becoming a problem to diagnose later.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import numpy as np

from qsorbit.core.dsp.iq import unpack_uint8_iq
from qsorbit.core.dsp.spectrum import SpectrumConfig, power_spectrum_db

#: Frames per second the worker aims to produce, when a caller does not
#: say. Chosen against what a human reading a waterfall can use rather
#: than against what the hardware can manage: at 2.048 Msps this works
#: out to roughly one frame per 256 KiB block, so a panel a few hundred
#: rows tall shows something like half a minute of history — long enough
#: to watch a Doppler slope develop, short enough to still be scrolling.
DEFAULT_FRAME_RATE_HZ: Final = 20.0

#: How many computed frames the buffer holds before it starts discarding
#: the oldest. At the default frame rate this is about three seconds of
#: history — enough to ride out a repaint hiccup or a resize, small
#: enough that a consumer which has genuinely stopped draining finds out
#: promptly instead of accumulating a backlog it will never catch up on.
DEFAULT_QUEUE_FRAMES: Final = 64

#: How long :meth:`SpectrumStream.stop` waits for the worker to notice.
#: The worker checks for the stop signal between blocks, so normal
#: latency is one block — about 64 ms at 2.048 Msps.
DEFAULT_JOIN_TIMEOUT_S: Final = 5.0

#: Bytes per complex sample in the raw stream: one I, one Q, both uint8.
#: Named rather than written as a bare ``2`` because the byte-range
#: arithmetic below is the one place where being wrong does not raise -
#: it silently frames the wrong part of the block.
BYTES_PER_SAMPLE: Final = 2

#: What :meth:`SpectrumStream.latest` subscribes under when a caller
#: never calls :meth:`SpectrumStream.subscribe` itself. Mirrors
#: :data:`~qsorbit.core.sdr.stream.DEFAULT_SUBSCRIBER_NAME`.
DEFAULT_SUBSCRIBER_NAME: Final = "default"

#: Label for a stream whose caller does not name its source. Idle today
#: and load-bearing in Chunk E, where a second dongle means a second
#: :class:`SpectrumStream` and two identically-labelled reports would be
#: unreadable at exactly the moment two devices need telling apart.
DEFAULT_SOURCE_NAME: Final = "sdr"


def _utc_now() -> datetime:
    """The current time, timezone-aware. Mirrors :mod:`qsorbit.core.pointing`."""
    return datetime.now(UTC)


@dataclass(frozen=True, eq=False)
class SpectrumFrame:
    """One power-spectrum row, and when it was made.

    Compared by identity, not by value — ``eq=False`` is deliberate.
    A generated ``__eq__`` would compare :attr:`power_db` with ``==``,
    which on a numpy array returns an *array* rather than a bool, and the
    dataclass then calls ``bool()`` on it and raises ``ValueError: the
    truth value of an array with more than one element is ambiguous``.
    That would fire from innocuous places — ``frame in some_list``, an
    ``assertEqual`` in a future test — so the trap is removed here rather
    than left for someone to trip over. Every other value object in this
    project compares by value; this one cannot, and that is why.

    Args:
        power_db: Power in dB, ``float32``, ``fft_size`` long, ordered to
            match :func:`~qsorbit.core.dsp.spectrum.frequency_axis_hz` —
            lowest frequency first. The frequency axis itself is *not*
            carried here: it is a property of the
            :class:`~qsorbit.core.dsp.spectrum.SpectrumConfig`, identical
            for every frame, and computing it per row would be waste a
            waterfall pays for several times a second.
        time: When this frame was computed, timezone-aware.

            **Not the instant its samples left the antenna.** The two
            differ by up to one block's worth of buffering plus the FFT
            itself — on the order of 70 ms at the default block size.
            That is invisible on a waterfall, which is what this is for,
            but it is not a timestamp to build anything precise on.

            Carried at all because Chunk G and H will want to lay a
            *predicted* Doppler curve, computed from the tracking loop's
            range-rate samples, over the observed trace. Both halves of
            Phase 2 meeting on one screen needs the two to share a clock,
            and adding the field now costs nothing.
    """

    power_db: np.ndarray
    time: datetime


@dataclass(frozen=True)
class SpectrumSubscriberStats:
    """What one consumer of a :class:`SpectrumStream` actually received.

    Mirrors :class:`~qsorbit.core.sdr.stream.SubscriberStats` one layer
    up, including its naming: *offered* rather than *delivered*, because
    a frame that was queued and then evicted before the consumer drained
    it was offered and not delivered, and calling that "delivered" makes
    the numbers stop adding up in a way nobody notices until they try to
    reconcile them.

    Args:
        name: The subscription's label, as given to
            :meth:`SpectrumStream.subscribe`.
        frames_offered: Frames the worker put in front of this consumer.
        frames_dropped: Frames this consumer lost because its own buffer
            was full. **Its own.** A stalled waterfall no longer costs
            the line trace a single frame, which is the entire point of
            the fan-out.
        queue_frames: This consumer's buffer depth, in frames.
    """

    name: str
    frames_offered: int
    frames_dropped: int
    queue_frames: int

    def describe(self) -> str:
        """Summarise one consumer in a single line."""
        return (
            f"{self.name}: {self.frames_offered:,} offered, "
            f"{self.frames_dropped:,} dropped, depth {self.queue_frames}"
        )


@dataclass(frozen=True)
class SpectrumStreamStats:
    """What one run produced, with the two kinds of missing frame kept apart.

    Args:
        blocks_consumed: Raw IQ blocks taken from the source.
        samples_consumed: Complex samples those blocks contained.
        frames_computed: Frames actually put through an FFT.
        frames_skipped: Frames the samples could have produced and that
            were deliberately never computed, because the display cannot
            use them. **This is the design working, not a fault**, and it
            is normally the overwhelming majority of the total.
        frames_dropped: Frames that *were* computed and then discarded
            because the buffer filled — the consumer is not draining fast
            enough. **This one is a fault.** Unlike ``frames_skipped``,
            work was done and thrown away, and a non-zero value means the
            GUI thread is behind.

            With more than one consumer this is the **worst-affected**
            one, not the sum, matching
            :attr:`~qsorbit.core.sdr.stream.StreamStats.blocks_dropped`.
            Each consumer drops out of its own buffer, so summing would
            report the stream as twice as lossy as either consumer
            actually experienced. :attr:`subscribers` carries the detail,
            and that is where to look before concluding which consumer is
            behind.
        queue_frames: Buffer depth, in frames, per consumer.
        worker_stopped_cleanly: Whether the worker exited within its join
            timeout.
        source: What this stream was framing. One device today; the field
            exists so that two do not produce two identical reports.
        subscribers: One entry per consumer, in subscription order.
            Defaults to empty so the places that build these by hand -
            tests, fixtures - keep working unchanged.

    A third possible loss needs no counter here: if the worker falls
    behind the *reader*, it is
    :attr:`~qsorbit.core.sdr.stream.StreamStats.blocks_dropped` on the
    IQ side that catches it, and duplicating the count would only give
    two chances to disagree.
    """

    blocks_consumed: int
    samples_consumed: int
    frames_computed: int
    frames_skipped: int
    frames_dropped: int
    queue_frames: int
    worker_stopped_cleanly: bool
    source: str = DEFAULT_SOURCE_NAME
    subscribers: tuple[SpectrumSubscriberStats, ...] = ()

    @property
    def frames_available(self) -> int:
        """Non-overlapping frames the consumed samples could have produced."""
        return self.frames_computed + self.frames_skipped

    @property
    def compute_fraction(self) -> float:
        """Share of available frames actually computed, in ``[0, 1]``."""
        available = self.frames_available
        if available <= 0:
            return 0.0
        return self.frames_computed / available

    def describe(self) -> str:
        """Summarise, wording the two losses so they cannot be confused."""
        clean = "" if self.worker_stopped_cleanly else "  worker DID NOT stop cleanly\n"
        # Listed only when there is more than one, for the reason
        # StreamStats.describe() gives: with a single consumer the line
        # restates the buffer line above it, and a report that says the
        # same number twice invites a hunt for a difference that is not
        # there.
        per_consumer = (
            "".join(f"  consumer {entry.describe()}\n" for entry in self.subscribers)
            if len(self.subscribers) > 1
            else ""
        )
        return (
            f"{self.blocks_consumed:,} block(s), {self.samples_consumed:,} samples "
            f"from {self.source}\n"
            f"  computed:  {self.frames_computed:,} frame(s) "
            f"({self.compute_fraction * 100:.1f}% of {self.frames_available:,} available)\n"
            f"  skipped by design: {self.frames_skipped:,} frame(s) - "
            f"the display cannot use them, not a fault\n"
            f"  dropped at buffer: {self.frames_dropped:,} frame(s) - "
            f"consumer behind, depth {self.queue_frames}\n" + per_consumer + clean
        )


def hop_for_frame_rate(config: SpectrumConfig, frame_rate_hz: float) -> int:
    """Samples to advance between frame starts, for a target frame rate.

    Args:
        config: Supplies ``sample_rate_hz`` and ``fft_size``.
        frame_rate_hz: Frames per second wanted on screen.

    Returns:
        The hop, in samples, never smaller than ``config.fft_size``.

        **The floor is deliberate and is a cap on frame rate, not on
        overlap.** A hop below ``fft_size`` would overlap frames, which
        produces *more* rows per second rather than fewer — the opposite
        of what a caller asking for a lower rate wants, and pointless for
        a caller asking for a higher one, since the samples to support it
        do not exist. Asking for more than the non-overlapping rate
        therefore gets the non-overlapping rate.

    Raises:
        ValueError: If ``frame_rate_hz`` is not a positive finite number.
    """
    if not math.isfinite(frame_rate_hz) or frame_rate_hz <= 0.0:
        raise ValueError(f"frame_rate_hz must be a positive, finite number, got {frame_rate_hz!r}.")
    return max(config.fft_size, round(config.sample_rate_hz / frame_rate_hz))


class SpectrumSubscription:
    """One consumer's independent view of a :class:`SpectrumStream`.

    Handed out by :meth:`SpectrumStream.subscribe`; never constructed
    directly. Each subscription owns its own bounded buffer, so two
    panels cannot take frames from each other.

    **This class exists because that failure actually happened.** Bench
    verification #11 (Session 24) found the waterfall and the line trace
    alternating on real hardware - one rendering while the other froze,
    with a solid band of sentinel rows where the waterfall had been
    starved. Both polled ``SpectrumStream.latest()`` on 50 ms timers and
    that method drained a single shared buffer, so whichever timer fired
    first took the whole batch. Neither widget was wrong; the API was.

    Satisfies the ``FrameSource`` protocol
    :class:`~qsorbit.ui.waterfall_widget.WaterfallWidget` and
    :class:`~qsorbit.ui.spectrum_line_widget.SpectrumLineWidget` already
    take - :attr:`config` and :meth:`latest` - so a widget is handed one
    of these instead of the stream itself and needs no idea that a
    fan-out exists. That is the Phase 2 Chunk F widget rule paying for
    itself: an element receives its feed and knows nothing about what
    contains it.

    The buffer **discards the oldest frame** when full, matching
    :class:`~qsorbit.core.sdr.stream.IqSubscription` and
    :class:`~qsorbit.core.dsp.audio.AudioOutput`: a consumer recovering
    from a repaint hiccup should see the present rather than replay a
    backlog it can never catch up on.
    """

    def __init__(self, stream: SpectrumStream, name: str, queue_frames: int) -> None:
        self._stream = stream
        self._name = name
        self._queue_frames = queue_frames
        self._frames: deque[SpectrumFrame] = deque(maxlen=queue_frames)
        self._offered = 0
        self._dropped = 0

    @property
    def name(self) -> str:
        """The label this subscription's statistics are reported under."""
        return self._name

    @property
    def config(self) -> SpectrumConfig:
        """The framing behind these frames, delegated to the stream.

        Present so a subscription satisfies ``FrameSource`` on its own,
        rather than a widget needing the stream for its axis and the
        subscription for its frames. Two objects to keep in step is two
        chances for a display to label a trace with the wrong frequency,
        which is worse than no axis because it is believed.
        """
        return self._stream.config

    @property
    def stats(self) -> SpectrumSubscriberStats:
        """What this consumer has received so far."""
        with self._stream._lock:
            return SpectrumSubscriberStats(
                name=self._name,
                frames_offered=self._offered,
                frames_dropped=self._dropped,
                queue_frames=self._queue_frames,
            )

    def latest(self) -> list[SpectrumFrame]:
        """Take this consumer's frames since its own last call, oldest first.

        Non-blocking, unlike
        :meth:`~qsorbit.core.sdr.stream.IqSubscription.blocks`, and
        deliberately so: this is drained from a ``QTimer`` on the GUI
        thread, where a blocking iterator would freeze the interface it
        is meant to be drawing. The fan-out is what changed here; the
        pull-on-a-timer shape settled in Chunk F did not.

        Returns an empty list when nothing new has arrived, which is the
        normal case whenever the consumer repaints faster than the frame
        rate. Empty is not an error and not a reason to repaint.

        Raises:
            Whatever killed the worker thread, re-raised once **this**
            consumer's buffer has drained. Frames already computed are
            handed over first and the error surfaces on the call after
            they run out, the same drain-then-raise order
            :meth:`~qsorbit.core.sdr.stream.IqSubscription.timed_blocks`
            uses. Every consumer is told, not only whichever one happens
            to drain first - see :meth:`SpectrumStream._reraise`.
        """
        with self._stream._lock:
            frames = list(self._frames)
            self._frames.clear()
        if not frames:
            self._stream._reraise()
        return frames

    def _offer(self, frame: SpectrumFrame) -> None:
        """Buffer one frame for this consumer. Called with the lock held."""
        if len(self._frames) == self._queue_frames:
            self._dropped += 1
        self._frames.append(frame)
        self._offered += 1


class SpectrumStream:
    """A worker thread turning raw IQ blocks into spectrum frames.

    Usage, wired to a real device::

        with IqStream(sdr) as iq, SpectrumStream(iq.blocks(), config) as frames:
            for frame in frames.latest():
                paint(frame)

    Two consumers, which is what the shell needs::

        spectrum = SpectrumStream(iq.blocks(), config)
        waterfall = spectrum.subscribe("waterfall")
        trace = spectrum.subscribe("spectrum-line")
        spectrum.start()
        # Each gets every frame; neither can take one from the other.

    The source is any iterable of raw byte blocks, which is what lets the
    same object be driven by :meth:`~qsorbit.core.sdr.stream.IqStream.blocks`
    at the bench and by a plain list of synthetic blocks in a test. No
    device, no Qt, and no hardware is needed to exercise everything in
    here.

    The buffer **discards the oldest frame** when full, matching
    :class:`~qsorbit.core.sdr.stream.IqStream` and
    :class:`~qsorbit.core.dsp.audio.AudioOutput`. For a waterfall this is
    plainly right: a consumer recovering from a hiccup should see the
    present, not replay a backlog it can never catch up on.

    Args:
        blocks: Raw interleaved uint8 I/Q blocks, in order.
        config: FFT size, sample rate, centre frequency and window.
        frame_rate_hz: Frames per second to aim for. See
            :func:`hop_for_frame_rate` for what happens at the extremes.
        queue_frames: Buffer depth, in frames.
        now: Clock for frame timestamps, injectable for tests. Wall clock
            rather than monotonic, deliberately: these timestamps exist
            to be compared against
            :attr:`~qsorbit.core.pointing.TrackSample.time`, which is
            also wall clock. Nothing here measures elapsed time.

    Raises:
        ValueError: If ``frame_rate_hz`` or ``queue_frames`` is unusable.
    """

    def __init__(
        self,
        blocks: Iterable[bytes],
        config: SpectrumConfig,
        *,
        frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
        queue_frames: int = DEFAULT_QUEUE_FRAMES,
        now: Callable[[], datetime] = _utc_now,
        source: str = DEFAULT_SOURCE_NAME,
    ) -> None:
        if queue_frames <= 0:
            raise ValueError(f"queue_frames must be positive, got {queue_frames!r}.")

        self._blocks = blocks
        self._config = config
        self._hop = hop_for_frame_rate(config, frame_rate_hz)
        self._queue_frames = queue_frames
        self._now = now
        self._source = source

        self._subscribers: list[SpectrumSubscription] = []
        self._default: SpectrumSubscription | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._stopped_cleanly = True

        self._blocks_consumed = 0
        self._samples_consumed = 0
        self._frames_computed = 0

        # Frame starts are tracked as absolute sample indices so the
        # cadence carries across block boundaries. Restarting the hop at
        # every block would space frames unevenly whenever the block size
        # is not a whole number of hops -- which is the normal case -- and
        # uneven spacing on a waterfall is a distorted time axis, not a
        # cosmetic problem.
        # Carried as raw bytes rather than unpacked samples, so that a
        # frame straddling a block boundary costs a small memcpy instead
        # of unpacking a whole block to keep 2,048 samples of it.
        self._next_frame_at = 0
        self._carry = b""
        self._carry_origin = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def config(self) -> SpectrumConfig:
        """The framing this stream produces, for a consumer that must label it.

        Exposed so a display can derive its frequency axis from the same
        object the frames came from, rather than being handed a second
        config that could quietly disagree. An axis that mislabels which
        frequency a trace sits at is worse than no axis, because it is
        believed.
        """
        return self._config

    @property
    def hop(self) -> int:
        """Samples between frame starts. See :func:`hop_for_frame_rate`."""
        return self._hop

    @property
    def source(self) -> str:
        """What this stream is framing, for a report that shows two."""
        return self._source

    @property
    def is_running(self) -> bool:
        """``True`` while the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the worker thread. Starting twice is an error.

        If nothing has subscribed, the implicit single-consumer
        subscription :meth:`latest` drains is created here, exactly as
        :meth:`~qsorbit.core.sdr.stream.IqStream.start` does. It has to
        exist *before* frames start arriving rather than being made on
        the first :meth:`latest` call: frames offered to an empty
        subscriber list go nowhere, so a lazily-made default would hand
        back nothing produced before the consumer first looked, which is
        silent data loss wearing an empty list.
        """
        if self._thread is not None:
            raise RuntimeError("This stream has already been started; build a new one.")
        if not self._subscribers:
            self._claim_default()
        self._thread = threading.Thread(
            target=self._work_loop,
            name=f"qsorbit-spectrum-worker-{self._source}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = DEFAULT_JOIN_TIMEOUT_S) -> SpectrumStreamStats:
        """Ask the worker to finish, wait for it, and return the statistics.

        The worker checks for the stop signal between blocks, so normal
        stop latency is one block. If the *source* blocks forever — an
        ``IqStream`` whose device stopped delivering, which librtlsdr
        gives no way to interrupt — the worker cannot notice, so this
        records the situation in
        :attr:`SpectrumStreamStats.worker_stopped_cleanly` rather than
        hanging the caller. The thread is a daemon and cannot keep the
        interpreter alive.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            self._stopped_cleanly = not thread.is_alive()
        return self.stats

    @property
    def stats(self) -> SpectrumStreamStats:
        """The run's statistics. Stable once :meth:`stop` has returned."""
        with self._lock:
            computed = self._frames_computed
            samples = self._samples_consumed
            # Built from the subscribers' private counters rather than
            # from their own stats property, which would take this same
            # lock a second time. threading.Lock is not reentrant, so
            # that reads as a tidier spelling and deadlocks.
            subscribers = tuple(
                SpectrumSubscriberStats(
                    name=sub._name,
                    frames_offered=sub._offered,
                    frames_dropped=sub._dropped,
                    queue_frames=sub._queue_frames,
                )
                for sub in self._subscribers
            )
            return SpectrumStreamStats(
                blocks_consumed=self._blocks_consumed,
                samples_consumed=samples,
                frames_computed=computed,
                frames_skipped=max(0, samples // self._config.fft_size - computed),
                frames_dropped=max((s.frames_dropped for s in subscribers), default=0),
                queue_frames=self._queue_frames,
                worker_stopped_cleanly=self._stopped_cleanly,
                source=self._source,
                subscribers=subscribers,
            )

    # ------------------------------------------------------------------
    # Consuming
    # ------------------------------------------------------------------

    def latest(self) -> list[SpectrumFrame]:
        """Take every frame buffered since the last call, oldest first.

        Returns *all* of them rather than only the newest, because a
        waterfall wants every row it can get: showing only the most
        recent would silently thin the display further than the frame
        rate already does, and the thinning would vary with how busy the
        GUI thread happened to be.

        Returns an empty list when nothing new has arrived, which is the
        normal case whenever the consumer repaints faster than the frame
        rate. Empty is not an error and not a reason to repaint.

        **This is the implicit single-consumer path.** It subscribes
        under :data:`DEFAULT_SUBSCRIBER_NAME` on first use, so a caller
        with one consumer works exactly as it did before the fan-out
        existed. It **raises** once anything has subscribed explicitly:
        an unnamed extra consumer sharing a buffer is precisely how two
        panels came to alternate on the bench, and that mistake is now
        loud rather than silent.

        Raises:
            Whatever killed the worker thread, re-raised here rather than
            vanishing with the thread. A waterfall that quietly freezes
            when the device disappears is exactly the silent failure this
            project keeps meeting; a caller driving this from a timer
            should catch, stop the timer, and show the error — the way
            :class:`~qsorbit.ui.readout_widget.ReadoutWidget` already
            does with a failing tick.

            **Frames already computed are handed over first, and the
            error surfaces on the call after they run out** — the same
            drain-then-raise order
            :meth:`~qsorbit.core.sdr.stream.IqStream.blocks` uses.
            Raising immediately would discard perfectly good frames that
            arrived before the fault, which is silent data loss inside an
            error path: the worst place to put it, because the exception
            is what everyone would be looking at instead.
        """
        return self._claim_default().latest()

    def subscribe(self, name: str) -> SpectrumSubscription:
        """Add a consumer with its own bounded buffer.

        **Allowed after the worker has started, unlike
        :meth:`~qsorbit.core.sdr.stream.IqStream.subscribe`**, and the
        divergence is deliberate rather than an oversight. That method
        refuses a late subscriber because IQ blocks are *sampled data*:
        one that joins mid-stream has a gap in it and no record of where
        it began, which is the quiet fault that module exists to make
        visible. Spectrum frames are a *display feed*. A panel opened two
        seconds into a run is simply two seconds shorter, this class
        already counts frames nobody asked for as ``frames_skipped``
        rather than as loss, and there is nothing to hide.

        The practical consequence is what settled it: the instrument
        window is built after its session has started, so refusing late
        subscription would mean either reordering that startup here - the
        change Chunk A's second half exists to make and measure on its
        own - or teaching :class:`~qsorbit.core.receive.ReceiveSession`
        which widgets exist, which is the feed hub's job a chunk early
        and against no design.

        Args:
            name: A label for this consumer, used in
                :attr:`SpectrumStreamStats.subscribers`. Must be unique
                within the stream, since the statistics are read by it.

        Returns:
            The subscription to hand that consumer.

        Raises:
            ValueError: If ``name`` is empty or already taken.
        """
        if not name:
            raise ValueError("A subscription needs a name; it is what labels the statistics.")
        with self._lock:
            if any(existing.name == name for existing in self._subscribers):
                raise ValueError(
                    f"A subscriber named {name!r} already exists. Names label the "
                    "per-consumer statistics, so two consumers sharing one would "
                    "make the report unreadable."
                )
            subscription = SpectrumSubscription(self, name, self._queue_frames)
            self._subscribers.append(subscription)
        return subscription

    def _claim_default(self) -> SpectrumSubscription:
        """Return the implicit subscription :meth:`latest` drains.

        Made by :meth:`start` when nothing has subscribed, so a
        single-consumer caller keeps every frame from the first block.
        Reachable directly too, for a stream never started.

        The cost of an implicit default is a subscription nobody drains,
        which is the shape of the phantom-drop defect this same PR
        removes from :class:`~qsorbit.core.receive.ReceiveSession`. It is
        avoided the same way there: **subscribe before starting** when
        the consumers are known, and no default is ever made. A late
        subscriber joining a stream that already has an implicit default
        will see that default's drops in the report - honestly counted,
        under a name that says what it is.

        Raises:
            RuntimeError: If anything has subscribed explicitly. Not a
                ``DeviceError`` like ``IqStream``'s equivalent, because
                nothing in this module knows a device exists and it is
                not going to start now.
        """
        with self._lock:
            if self._default is None and self._subscribers:
                raise RuntimeError(
                    "This stream has explicit subscribers, so latest() would add an "
                    "unnamed consumer whose dropped frames nothing would attribute. "
                    "Before the fan-out existed this is exactly how two panels came "
                    "to drain one buffer and alternate. Call subscribe() for this "
                    "consumer as well."
                )
            if self._default is None:
                self._default = SpectrumSubscription(
                    self, DEFAULT_SUBSCRIBER_NAME, self._queue_frames
                )
                self._subscribers.append(self._default)
            return self._default

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> SpectrumStream:
        """Start the worker on entering a ``with`` block."""
        if self._thread is None:
            self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Stop the worker on leaving, whether or not the body raised."""
        self.stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _work_loop(self) -> None:
        """The worker thread's body. Runs until stopped or until it fails."""
        try:
            for raw in self._blocks:
                if self._stop.is_set():
                    break
                self._consume(raw)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            self._error = exc

    def _consume(self, raw: bytes) -> None:
        """Compute whatever frames this block completes, and offer them.

        **Only the bytes a frame actually needs are unpacked**, which is
        the whole of this method's difference from the version it
        replaced. That one called
        :func:`~qsorbit.core.dsp.iq.unpack_uint8_iq` on the entire block
        - 131,072 complex samples at 256 KiB - and then framed about
        2,048 of them, throwing away 98% of the conversion. Measured in
        Session 19 at **2.24% of one core against the FFTs' 0.24%**: ten
        times the cost of the work it existed to feed. Same family as
        Session 16's marshalling bug, where per-sample work on a path
        that only needed a fraction of it presented as 9.5% USB loss.

        The arithmetic moved from sample indices to byte offsets to make
        that possible, since a byte range is what can be sliced before
        anything is converted. Two bytes per complex sample, hence
        :data:`BYTES_PER_SAMPLE`, and the carry is now raw bytes.
        """
        # The whole-block unpack used to catch a truncated I/Q pair for
        # free, since unpack_uint8_iq refuses an odd length. Slicing byte
        # ranges would instead floor it silently and frame whatever came
        # next, so the guarantee is kept explicitly rather than lost in
        # an optimisation.
        if len(raw) % BYTES_PER_SAMPLE != 0:
            raise ValueError(
                f"IQ block has odd length {len(raw)}; that means a truncated I/Q "
                "pair, not a valid block."
            )

        # No copy in the common case: the carry is empty whenever the hop
        # runs past the end of a block, which is every block at any frame
        # rate a display can use.
        buffer: bytes = raw if not self._carry else self._carry + bytes(raw)
        buffered_samples = len(buffer) // BYTES_PER_SAMPLE
        block_samples = len(raw) // BYTES_PER_SAMPLE
        end = self._carry_origin + buffered_samples
        fft_size = self._config.fft_size
        view = memoryview(buffer)

        computed: list[SpectrumFrame] = []
        while self._next_frame_at + fft_size <= end:
            start = (self._next_frame_at - self._carry_origin) * BYTES_PER_SAMPLE
            stop = start + fft_size * BYTES_PER_SAMPLE
            samples = unpack_uint8_iq(view[start:stop])
            power_db = power_spectrum_db(samples, self._config)
            computed.append(SpectrumFrame(power_db=power_db, time=self._now()))
            self._next_frame_at += self._hop

        # Keep only what a future frame might still need. When the hop
        # runs past the end of this block -- a low frame rate, so most
        # blocks contribute nothing -- this discards the whole buffer
        # rather than growing it, which is what stops a slow frame rate
        # from turning into a memory leak.
        keep_from = min(max(self._next_frame_at - self._carry_origin, 0), buffered_samples)
        self._carry = bytes(view[keep_from * BYTES_PER_SAMPLE :])
        self._carry_origin += keep_from

        with self._lock:
            self._blocks_consumed += 1
            # This block's own samples, not the buffer's: the carry was
            # already counted by the block it arrived in, and counting it
            # twice would inflate frames_skipped against frames_computed.
            self._samples_consumed += block_samples
            self._frames_computed += len(computed)
            # The same immutable frame goes to every consumer. No copy,
            # and no consumer can affect another's buffer.
            for frame in computed:
                for subscriber in self._subscribers:
                    subscriber._offer(frame)

    def _reraise(self) -> None:
        """Re-raise whatever killed the worker thread, if anything did.

        **The error is not cleared once raised**, unlike the
        single-consumer version this replaced, and for the reason
        :meth:`~qsorbit.core.sdr.stream.IqStream._reraise` gives one
        layer down: with several consumers, clearing means whichever one
        drained first gets the exception and every other one sees its
        frames simply stop - a silent freeze with the explanation already
        consumed by somebody else. Every consumer is entitled to be told
        why the frames stopped.

        This is a behaviour change a reader could be surprised by, and it
        is the same one Chunk H made to ``IqStream`` when it grew a
        fan-out. A caller that drains after an error now keeps seeing it
        rather than seeing an empty list.
        """
        error = self._error
        if error is not None:
            raise error
