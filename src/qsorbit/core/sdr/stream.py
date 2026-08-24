"""Continuous IQ streaming, and the arithmetic that says whether it kept up.

Two things live here, and keeping them separate is the point of the
module.

:class:`IqStream` is the streaming layer proper: a dedicated thread
looping :meth:`~qsorbit.core.sdr.device.RtlSdr.read_raw` into a bounded
buffer that a consumer drains. That shape was chosen in Session 15 over
``rtlsdr_read_async`` because the async path carries every sharp edge a
hand-written FFI has — a buffer valid only for the duration of a
callback, a ``CFUNCTYPE`` that crashes rather than raises if the GC
collects it, the GIL held inside librtlsdr's own USB thread, and a
documented cancellation race. Sync reads keep control flow in Python.

:class:`ThroughputMonitor` is the accountant, and it is deliberately a
separate object that needs no thread, no queue and no device. It exists
because the sync-read design was accepted with a **known, unmeasured
cost**: no USB transfer is in flight between one ``read_sync``
returning and the next being issued, so the device's FIFO may overflow
under sustained load. That risk was converted into a measurement rather
than an argument, and this is the instrument.

**The measurement was taken on 2026-08-22 and the design holds.** A V4
at 2.048 Msps with 256 KiB blocks lost 0.0021% of the stream over 30
seconds — 2.5 kB in 122 MB — with no read stalling by as much as a
millisecond. ``rtlsdr_read_async`` is not needed and stays where Session
15 left it: available as a contained swap behind this same interface if
a future device or platform ever needs it.

The measurement earned its keep on the way there, and not in the way
anyone expected. The first run reported 9.5% loss, which reads exactly
like the sync-read gap the whole exercise was watching for — and was
not. It was ``bytes(buffer[:n])`` in the binding building a
quarter-million-element Python list on every read, a fixed 6.7 ms
against a 64 ms block. Switching to the async path would have carried
the cost along unchanged. What gave it away was the *shape*: a bare read
loop and the full pipeline reported identical loss to four decimal
places, which no contention story explains and a fixed per-read cost
explains exactly. Hence two separate tests, and hence
:attr:`LossReport.deficits_s` being kept per-read rather than summarised
away.

**Why throughput accounting rather than the counter in test mode.** The
RTL2832U can be told to emit an incrementing 8-bit counter in place of
ADC data, which is how ``rtl_test`` detects loss. But an 8-bit counter
only recovers a gap *modulo 256*, and real losses arrive in
USB-transfer-shaped chunks that are overwhelmingly multiples of 256 — so
such a gap is invisible to it. That is why ``rtl_test`` reports "Samples
per million lost (**minimum**)": the vendor's own output string admits
it is a lower bound. The device's sample clock, by contrast, cannot lie
about how many bytes it should have produced, so comparing that against
what actually arrived is the measurement that cannot be fooled by
alignment. Test mode was considered and rejected on those grounds; it
would also have cost a nineteenth hand-written ctypes signature.

**The two losses this module reports are different faults and are
counted separately**, because reporting them as one number sends you to
fix the wrong thing:

``StreamStats.blocks_dropped``
    Blocks discarded because *our* buffer filled — the consumer could
    not keep up. Counted exactly, since we own the buffer. Swapping to
    ``rtlsdr_read_async`` would not help this at all.

``StreamStats.loss``
    Samples that never reached us across USB, inferred from the sample
    clock. This is the sync-read gap, and the one that would justify
    the async swap.

**One producer, many consumers** (Chunk H). A live session wants the
same blocks in two places at once — demodulated to audio, and framed
into a waterfall — and until this chunk that was impossible in a way
nothing reported: :meth:`IqStream.blocks` was a single generator over a
single buffer, so a second consumer *interleaved* with the first and
each quietly got roughly every other block. Both would have looked like
they were working. Chunk G worked around it by running its bench checks
as separate passes; the ``receive`` path cannot.

So the buffer is now per-consumer. :meth:`IqStream.subscribe` hands out
an :class:`IqSubscription` with its own bounded queue and its own drop
count, and the reader appends the *same* immutable ``bytes`` object to
each — the fan-out costs one deque append per subscriber inside a lock
the reader already holds, and no copy. Two consequences worth stating:

* A slow consumer drops its own blocks and nobody else's, which is why
  the drop count is per subscriber rather than one number for the
  stream. :attr:`StreamStats.blocks_dropped` keeps its old meaning by
  reporting the **worst-affected** consumer. Summing them would make a
  two-consumer stream look twice as lossy as either consumer actually
  was, which is the same "two faults reported as one number" mistake
  this module already exists to avoid.
* :meth:`IqStream.blocks` now **raises** on a second call rather than
  silently interleaving. The trap is removed at its source rather than
  left standing next to the safe thing.

**Blocks carry a timestamp** (Chunk H, same reason). Doppler correction
is recomputed at each block's midpoint, so it needs to know when a
block's samples were on the air. That timestamp has to be taken where
the block is read: deriving it in the consumer is accurate only while
the consumer keeps up, and is wrong by the whole queue depth exactly
when it does not. See :class:`TimedBlock` for what it does and does not
promise.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

from qsorbit.core.sdr.device import DEFAULT_READ_BYTES, RtlSdr
from qsorbit.core.sdr.exceptions import DeviceError

#: One complex sample is two bytes on the wire: one I, one Q, each an
#: unsigned 8-bit value. Every rate-to-byte-rate conversion goes through
#: this rather than through a bare ``* 2``.
BYTES_PER_SAMPLE: Final = 2

#: How many blocks the buffer holds before it starts discarding. At the
#: default block size this is 4 MiB, or about a second at 2.048 Msps —
#: enough to ride out a consumer hiccup, small enough that a consumer
#: which has genuinely fallen behind finds out promptly rather than
#: accumulating a stale backlog it will never catch up on.
DEFAULT_QUEUE_BLOCKS: Final = 16

#: Per-read deficits below this are reported as noise rather than as
#: stalls. Purely a **display** threshold — it changes what gets counted
#: in :attr:`LossReport.stalls`, never what gets counted as loss. Thread
#: wake-up jitter is sub-millisecond against a 64 ms block, so this sits
#: comfortably above the floor.
STALL_THRESHOLD_S: Final = 0.001

#: How long :meth:`IqStream.stop` waits for the reader to notice. One
#: block at 2.048 Msps is about 64 ms, so this is generous; it exists
#: for the pathological case described in :meth:`IqStream.stop`.
DEFAULT_JOIN_TIMEOUT_S: Final = 5.0

#: How often a waiting consumer re-checks while idle. Does not add
#: latency to a block that arrives — the reader notifies — it only
#: bounds how long a shutdown can sit unnoticed.
DEFAULT_POLL_S: Final = 0.5

#: Name of the implicit subscription :meth:`IqStream.blocks` uses, so
#: that a single-consumer stream still labels its statistics with
#: something rather than with a blank.
DEFAULT_SUBSCRIBER_NAME: Final = "default"


def _utc_now() -> datetime:
    """The current instant, timezone-aware.

    Deliberately a **wall** clock, and deliberately not the monotonic
    one :class:`ThroughputMonitor` uses. The two clocks answer different
    questions and conflating them is an easy, invisible mistake: the
    monitor measures elapsed time, where a wall clock can step
    backwards, while a block timestamp exists to be compared against
    :attr:`~qsorbit.core.pointing.TrackSample.time` and
    :attr:`~qsorbit.core.dsp.spectrum_stream.SpectrumFrame.time`, which
    are wall clock. Nothing here measures a duration.
    """
    return datetime.now(UTC)


@dataclass(frozen=True)
class TimedBlock:
    """One raw IQ block and when its samples arrived.

    Args:
        data: Raw interleaved uint8 I/Q, exactly as the device returned
            it. Shared between every subscriber rather than copied —
            ``bytes`` is immutable, so this is safe and free.
        read_at: When the read that produced this block **returned**,
            timezone-aware. The block's samples therefore span
            ``[read_at - duration_s, read_at]``, which is why
            :attr:`midpoint` subtracts rather than adds.
        duration_s: How long the block's samples represent, from the
            sample rate the device actually reported.

    **What this timestamp does not promise.** It is taken when the block
    finished arriving at the host, so it trails the antenna by however
    deep librtlsdr's USB pipeline happens to be. That lag is unknown to
    us but small and essentially constant, and a constant offset shifts
    a Doppler correction by the Doppler *rate* times the lag — at most a
    few tens of hertz at UHF against a channel sixteen kilohertz wide.
    Fine for what this is for, and not a timestamp to build anything
    precise on. Same caveat, same wording, as
    :attr:`~qsorbit.core.dsp.spectrum_stream.SpectrumFrame.time`.
    """

    data: bytes
    read_at: datetime
    duration_s: float

    @property
    def midpoint(self) -> datetime:
        """The instant halfway through this block's samples.

        The midpoint rather than either edge because it removes a
        systematic half-block bias from a per-block correction for free
        — :meth:`~qsorbit.core.dsp.tuning.DopplerTracker.offset_at` asks
        for exactly this, and computing it here rather than in each
        caller is what stops one of them getting the sign wrong.
        """
        return self.read_at - timedelta(seconds=self.duration_s / 2.0)


@dataclass(frozen=True)
class SubscriberStats:
    """What one consumer of a stream actually received.

    Args:
        name: The subscription's label, as given to
            :meth:`IqStream.subscribe`.
        blocks_offered: Blocks the reader put in front of this consumer
            — every block read while it was subscribed. Named *offered*
            rather than *delivered* on purpose: a block that is queued
            and then evicted before the consumer gets to it was offered
            and not delivered, and calling that "delivered" would make
            the two numbers below fail to add up in a way nobody would
            notice until they tried to reconcile them.
        blocks_dropped: Blocks evicted because *this* consumer's queue
            was full. Nobody else's queue is affected, which is the
            whole point of keeping the count here: a waterfall that
            stalls does not make the audio path look broken.
        queue_blocks: This consumer's buffer depth.
    """

    name: str
    blocks_offered: int
    blocks_dropped: int
    queue_blocks: int

    def describe(self) -> str:
        """Summarise one consumer in a single line."""
        return (
            f"{self.name}: {self.blocks_offered:,} offered, "
            f"{self.blocks_dropped:,} dropped, depth {self.queue_blocks}"
        )


@dataclass(frozen=True)
class LossReport:
    """How much of the stream never arrived, and in what shape.

    Args:
        reads: Reads included in the accounting. The very first read is
            deliberately excluded — see :meth:`ThroughputMonitor.record`.
        bytes_read: Bytes actually delivered across those reads.
        elapsed_s: Wall-clock time those reads spanned.
        byte_rate: Bytes per second the device's clock produces. Taken
            from the sample rate the device **reported**, never the one
            it was asked for.
        deficits_s: Per-read timing deficits, in order. Each is how much
            longer a read took than the samples it returned account for.

    A note on how to read these. :attr:`lost_bytes` is the measurement;
    :attr:`deficits_s` is the *shape* of it. The two are algebraically
    the same total — the deficits sum to the aggregate, deliberately
    unclamped so that a read which drains a backlog can offset one that
    stalled. Clamping the negatives away would double-count perfectly
    healthy buffering as loss.
    """

    reads: int
    bytes_read: int
    elapsed_s: float
    byte_rate: float
    deficits_s: tuple[float, ...]

    @property
    def expected_bytes(self) -> float:
        """How many bytes the device's clock must have produced."""
        return self.elapsed_s * self.byte_rate

    @property
    def lost_bytes(self) -> float:
        """Bytes the clock produced that never reached us.

        **Comes out slightly negative sometimes, and that is not an
        error.** The device banks samples in its own FIFO, so a run can
        end having delivered a hair more than its wall-clock span
        accounts for. Observed on real hardware: −499 bytes over a five
        second capture, or −0.12 ms.

        Two other things blur the last digit. ``byte_rate`` is built
        from the rate librtlsdr *reports*, which is nominal — the
        dongle's crystal is not the host's, and a routine few tens of
        ppm is worth a few thousand bytes over thirty seconds. And one
        late final read charges the window for samples still sitting in
        the FIFO. Treat anything under about 0.01% as the stopwatch
        rather than the device.

        A *large* negative is a different matter and means the byte rate
        is wrong — almost always the requested sample rate used where
        the actual one belonged — not that the device is generous.
        """
        return self.expected_bytes - self.bytes_read

    @property
    def loss_fraction(self) -> float:
        """:attr:`lost_bytes` as a fraction of what was expected."""
        if self.expected_bytes <= 0:
            return 0.0
        return self.lost_bytes / self.expected_bytes

    @property
    def worst_stall_s(self) -> float:
        """The largest single-read deficit."""
        return max(self.deficits_s, default=0.0)

    @property
    def stalls(self) -> int:
        """How many reads had a deficit past :data:`STALL_THRESHOLD_S`.

        The distinction this exists to draw: one big stall and a steady
        dribble produce the same :attr:`lost_bytes` and have completely
        different causes. A handful of large stalls points at something
        else on the machine; loss spread across every read points at the
        sync-read gap itself.
        """
        return sum(1 for deficit in self.deficits_s if deficit > STALL_THRESHOLD_S)

    def describe(self) -> str:
        """Return a short human-readable summary, for logs and reports."""
        return (
            f"{self.bytes_read:,} bytes over {self.elapsed_s:.2f}s "
            f"({self.reads} reads); lost {self.lost_bytes:,.0f} bytes "
            f"({self.loss_fraction * 100:.4f}%); "
            f"{self.stalls} stall(s), worst {self.worst_stall_s * 1000:.1f} ms"
        )


class ThroughputMonitor:
    """Counts what arrived against what the device's clock must have produced.

    Feed it :meth:`record` after every read. It needs no device, no
    thread and no queue, which is what lets the same instrument measure
    a bare ``read_raw`` loop and a full streaming pipeline — the two
    measurements are only comparable because the accounting is
    identical.

    Args:
        byte_rate: Bytes per second the device produces. Use
            :func:`byte_rate_for`, and give it the sample rate the
            device **reported**.
        clock: Monotonic clock, injectable for tests. Never a wall
            clock: this measures elapsed time, and a wall clock can step
            backwards.

    Raises:
        ValueError: If ``byte_rate`` is not positive.

    Thread-safety: :meth:`record` is for the reader thread alone and
    takes no lock, deliberately — a lock acquisition between the read
    returning and the timestamp being taken would contaminate the very
    number being measured. :meth:`report` is therefore only stable once
    the reader has stopped.
    """

    def __init__(self, byte_rate: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        if byte_rate <= 0:
            raise ValueError(f"byte_rate must be positive, got {byte_rate!r}.")
        self._byte_rate = float(byte_rate)
        self._clock = clock
        self._started_at: float | None = None
        self._last_at: float = 0.0
        self._bytes = 0
        self._deficits: list[float] = []

    def record(self, byte_count: int) -> None:
        """Note that a read returned ``byte_count`` bytes, just now.

        Call this immediately after the read returns and before doing
        anything else with the block, so the timestamp reflects the
        device rather than whatever the consumer did next.

        **The first call starts the clock and is otherwise ignored.**
        The interval before it contains device spin-up and the tail of
        ``reset_buffer``, which is not loss and must not be counted as
        loss.
        """
        now = self._clock()
        if self._started_at is None:
            self._started_at = now
            self._last_at = now
            return
        expected_s = byte_count / self._byte_rate
        self._deficits.append((now - self._last_at) - expected_s)
        self._last_at = now
        self._bytes += byte_count

    def report(self) -> LossReport:
        """Return the accounting so far.

        Stable once the reader thread has stopped; a snapshot taken
        while it is still running is not wrong, merely provisional.
        """
        started = self._started_at
        elapsed = 0.0 if started is None else self._last_at - started
        return LossReport(
            reads=len(self._deficits),
            bytes_read=self._bytes,
            elapsed_s=elapsed,
            byte_rate=self._byte_rate,
            deficits_s=tuple(self._deficits),
        )


def byte_rate_for(sample_rate_hz: float) -> float:
    """Return the byte rate a given sample rate implies.

    Args:
        sample_rate_hz: Complex samples per second. Pass the rate the
            device **reported** — see
            :attr:`~qsorbit.core.sdr.device.AppliedSettings.sample_rate_hz`.
            The sample clock quantises, and an accounting run against
            the requested rate measures the quantisation error as loss.

    Returns:
        Bytes per second.
    """
    return sample_rate_hz * BYTES_PER_SAMPLE


@dataclass(frozen=True)
class StreamStats:
    """What one streaming run did, with its two losses kept apart.

    Args:
        blocks_read: Blocks the reader pulled from the device.
        bytes_read: Bytes those blocks contained.
        blocks_dropped: Blocks discarded because a buffer was full —
            **our** loss, exactly counted, caused by a slow consumer.
            Fixing this means a larger buffer or a faster consumer;
            changing how the device is read would not touch it.

            With more than one consumer this is the **worst-affected**
            one, not the sum. Each consumer has its own queue and drops
            independently, so summing would report a stream as twice as
            lossy as either consumer actually experienced. The
            per-consumer detail is in :attr:`subscribers`, and that is
            where to look before concluding anything about which
            consumer is behind.
        block_bytes: Bytes requested per read.
        queue_blocks: Buffer depth in blocks, per consumer.
        reader_stopped_cleanly: Whether the reader thread exited within
            its join timeout. ``False`` means a read is wedged — see
            :meth:`IqStream.stop`.
        loss: Samples that never crossed USB — the sync-read gap, and
            the number that would justify moving to
            ``rtlsdr_read_async``.
        subscribers: One entry per consumer, in subscription order.
            Defaults to empty so that the many places which build a
            :class:`StreamStats` by hand — tests, fixtures — keep
            working unchanged.
    """

    blocks_read: int
    bytes_read: int
    blocks_dropped: int
    block_bytes: int
    queue_blocks: int
    reader_stopped_cleanly: bool
    loss: LossReport
    subscribers: tuple[SubscriberStats, ...] = ()

    @property
    def dropped_bytes(self) -> int:
        """Bytes discarded at the buffer, assuming full-size blocks."""
        return self.blocks_dropped * self.block_bytes

    def describe(self) -> str:
        """Return a summary that keeps the two faults visibly separate."""
        clean = "" if self.reader_stopped_cleanly else "  reader DID NOT stop cleanly\n"
        # Listed only when there is more than one, because for a single
        # consumer the line would restate the buffer line directly above
        # it, and a report that says the same number twice invites the
        # reader to look for a difference that is not there.
        per_consumer = (
            "".join(f"  consumer {entry.describe()}\n" for entry in self.subscribers)
            if len(self.subscribers) > 1
            else ""
        )
        return (
            f"blocks read {self.blocks_read:,} of {self.block_bytes:,} bytes\n"
            f"  at the device (USB):  {self.loss.describe()}\n"
            f"  at our buffer:        {self.blocks_dropped:,} block(s) dropped "
            f"({self.dropped_bytes:,} bytes), depth {self.queue_blocks}\n" + per_consumer + clean
        )


class IqSubscription:
    """One consumer's independent view of an :class:`IqStream`.

    Handed out by :meth:`IqStream.subscribe`; never constructed
    directly. Each subscription has its own bounded queue, so consumers
    cannot steal blocks from one another and a slow one drops only its
    own — the failure that used to happen silently when two callers
    shared a single :meth:`IqStream.blocks` generator.

    The queue **discards the oldest block** when full, matching
    :class:`IqStream`'s original policy and
    :class:`~qsorbit.core.dsp.audio.AudioOutput`'s: a consumer
    recovering from a hiccup should see the present rather than replay a
    backlog it can never catch up on.
    """

    def __init__(self, stream: IqStream, name: str, queue_blocks: int) -> None:
        self._stream = stream
        self._name = name
        self._queue_blocks = queue_blocks
        self._queue: deque[TimedBlock] = deque(maxlen=queue_blocks)
        self._offered = 0
        self._dropped = 0

    @property
    def name(self) -> str:
        """The label this subscription's statistics are reported under."""
        return self._name

    @property
    def stats(self) -> SubscriberStats:
        """What this consumer has received so far."""
        return SubscriberStats(
            name=self._name,
            blocks_offered=self._offered,
            blocks_dropped=self._dropped,
            queue_blocks=self._queue_blocks,
        )

    def blocks(self, poll_s: float = DEFAULT_POLL_S) -> Iterator[bytes]:
        """Yield this consumer's blocks as raw bytes, until the reader finishes.

        The plain form, for a consumer that does not care when a block
        arrived — a waterfall, a capture. Use :meth:`timed_blocks` where
        the time matters.

        Args:
            poll_s: See :data:`DEFAULT_POLL_S`.

        Yields:
            Raw interleaved I/Q blocks, oldest first.

        Raises:
            DeviceError: Anything the reader thread hit, re-raised once
                this consumer's queue drains.
        """
        for timed in self.timed_blocks(poll_s):
            yield timed.data

    def timed_blocks(self, poll_s: float = DEFAULT_POLL_S) -> Iterator[TimedBlock]:
        """Yield this consumer's blocks with their timestamps.

        Args:
            poll_s: See :data:`DEFAULT_POLL_S`.

        Yields:
            :class:`TimedBlock`, oldest first.

        Raises:
            DeviceError: Anything the reader thread hit is re-raised
                here once this consumer's queue drains, rather than
                vanishing with the thread. **Every** consumer is told,
                not just whichever one happens to drain first — see
                :meth:`IqStream._reraise`.
        """
        stream = self._stream
        stream._ensure_started()
        while True:
            with stream._not_empty:
                while not self._queue and not stream._finished:
                    stream._not_empty.wait(poll_s)
                if self._queue:
                    block = self._queue.popleft()
                else:
                    break
            yield block
        stream._reraise()

    def _offer(self, block: TimedBlock) -> None:
        """Put a block in this consumer's queue. Called with the lock held."""
        if len(self._queue) == self._queue_blocks:
            self._dropped += 1
        self._queue.append(block)
        self._offered += 1


class IqStream:
    """A reader thread pulling IQ blocks from a device into a bounded buffer.

    One consumer, which is most callers::

        with IqStream(sdr) as stream:
            for block in stream.blocks():
                consume(block)
                if enough:
                    break
        print(stream.stats.describe())

    Two consumers — the ``receive`` path's shape. Subscribe **before**
    starting the reader, then hand each subscription to its consumer::

        stream = IqStream(sdr)
        audio = stream.subscribe("audio")
        waterfall = stream.subscribe("waterfall")
        with stream:
            ...  # audio.timed_blocks() on one thread,
                 # waterfall.blocks() feeding a SpectrumStream on another

    The buffer **discards the oldest block** when it is full. Two
    reasons. For a live consumer — a waterfall, an audio path — stale
    samples are worth less than current ones, so a consumer recovering
    from a hiccup should see the present rather than replay a backlog.
    For a capture, any drop at all breaks contiguity, so which end is
    discarded makes no difference to correctness and the caller's job is
    to notice ``blocks_dropped`` is non-zero rather than to prefer one
    kind of hole over another. :func:`~qsorbit.core.sdr.capture.capture_to_file`
    treats a single dropped block as a failed capture for exactly that
    reason.

    Args:
        device: An open, **configured** device. Configuration is
            required because the accounting needs the sample rate the
            device actually settled on.
        block_bytes: Bytes per read. This is the parameter that governs
            the sync-read design's exposure: the loss window is the gap
            *between* reads, so halving the block size doubles how many
            gaps occur per second. If a measurement comes back bad, try
            a larger block before concluding the design is wrong.
        queue_blocks: Buffer depth, **per consumer**.
        clock: Monotonic clock for the loss accounting, injectable for
            tests. Never a wall clock — see :class:`ThroughputMonitor`.
        now: Wall clock for block timestamps, injectable for tests.
            Deliberately a second, separate clock: see :func:`_utc_now`.

    Raises:
        DeviceError: If the device is not open, or has never been
            configured.
        ValueError: If ``block_bytes`` or ``queue_blocks`` is not
            positive.
    """

    def __init__(
        self,
        device: RtlSdr,
        *,
        block_bytes: int = DEFAULT_READ_BYTES,
        queue_blocks: int = DEFAULT_QUEUE_BLOCKS,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        if block_bytes <= 0:
            raise ValueError(f"block_bytes must be positive, got {block_bytes}.")
        if queue_blocks <= 0:
            raise ValueError(f"queue_blocks must be positive, got {queue_blocks}.")
        if not device.is_open:
            raise DeviceError("The device is not open, so there is nothing to stream from.")
        applied = device.applied
        if applied is None:
            raise DeviceError(
                "The device has never been configured, so its sample rate is "
                "unknown and no loss accounting is possible. Call configure() first."
            )

        self._device = device
        self._block_bytes = block_bytes
        self._queue_blocks = queue_blocks
        self._byte_rate = byte_rate_for(applied.sample_rate_hz)
        self._monitor = ThroughputMonitor(self._byte_rate, clock=clock)
        self._now = now

        self._subscribers: list[IqSubscription] = []
        self._default: IqSubscription | None = None
        self._default_claimed = False
        self._not_empty = threading.Condition(threading.Lock())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._finished = False
        self._error: BaseException | None = None
        self._blocks_read = 0
        self._bytes_read = 0
        self._stopped_cleanly = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True`` while the reader thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def subscribe(self, name: str) -> IqSubscription:
        """Add a consumer with its own bounded queue, before the reader starts.

        Args:
            name: A label for this consumer, used in
                :attr:`StreamStats.subscribers`. Must be unique within
                the stream, since the statistics are read by it.

        Returns:
            The subscription to hand that consumer.

        Raises:
            DeviceError: If the reader has already started. A
                subscription made after the fact would silently miss
                everything read so far, and a consumer that begins
                mid-stream with no record of where it joined is exactly
                the quiet gap this module exists to make visible.
                Dynamic, mid-run subscription is not built because
                nothing needs it yet: the ``receive`` path knows both of
                its consumers before the first read, and a
                :class:`IqStream` is constructed per listening session.
            ValueError: If ``name`` is empty or already taken.
        """
        if self._thread is not None:
            raise DeviceError(
                f"Cannot subscribe {name!r} after the reader has started -- it would "
                "silently miss every block already read. Subscribe before start(), or "
                "before entering the 'with' block, which starts the reader."
            )
        if not name:
            raise ValueError("A subscription needs a name; it is what labels the statistics.")
        if any(existing.name == name for existing in self._subscribers):
            raise ValueError(
                f"A subscriber named {name!r} already exists. Names label the "
                "per-consumer statistics, so two consumers sharing one would make "
                "the report unreadable."
            )
        subscription = IqSubscription(self, name, self._queue_blocks)
        self._subscribers.append(subscription)
        return subscription

    def start(self) -> None:
        """Start the reader thread. Starting twice is an error.

        If nothing has subscribed, the implicit single-consumer
        subscription :meth:`blocks` uses is created here, so a stream
        started and left unconsumed still buffers and still counts its
        drops exactly as it did before the fan-out existed.
        """
        if self._thread is not None:
            raise DeviceError("This stream has already been started; build a new one.")
        if not self._subscribers:
            self._default = self.subscribe(DEFAULT_SUBSCRIBER_NAME)
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"qsorbit-iq-reader-{self._device.index}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = DEFAULT_JOIN_TIMEOUT_S) -> StreamStats:
        """Ask the reader to finish, wait for it, and return the statistics.

        The reader checks for the stop signal **between** reads, so
        normal stop latency is one block — about 64 ms at 2.048 Msps
        with the default size.

        The pathological case is worth knowing about: librtlsdr compiles
        its bulk transfers with an infinite timeout, so a device that
        stops delivering leaves ``read_sync`` blocked forever with no
        way to interrupt it from Python. The thread is a daemon so it
        cannot keep the interpreter alive, and this method records the
        situation in :attr:`StreamStats.reader_stopped_cleanly` rather
        than hanging the caller.

        Args:
            timeout_s: How long to wait for the reader to exit.

        Returns:
            The run's statistics.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            self._stopped_cleanly = not thread.is_alive()
        return self.stats

    @property
    def stats(self) -> StreamStats:
        """The run's statistics. Stable once :meth:`stop` has returned."""
        per_consumer = tuple(subscriber.stats for subscriber in self._subscribers)
        return StreamStats(
            blocks_read=self._blocks_read,
            bytes_read=self._bytes_read,
            # The worst-affected consumer, not the sum -- see the field's
            # docstring. With one consumer the two are the same number,
            # which is why this reads identically to how it always has.
            blocks_dropped=max((entry.blocks_dropped for entry in per_consumer), default=0),
            block_bytes=self._block_bytes,
            queue_blocks=self._queue_blocks,
            reader_stopped_cleanly=self._stopped_cleanly,
            loss=self._monitor.report(),
            subscribers=per_consumer,
        )

    # ------------------------------------------------------------------
    # Consuming
    # ------------------------------------------------------------------

    def blocks(self, poll_s: float = DEFAULT_POLL_S) -> Iterator[bytes]:
        """Yield blocks as they arrive, until the reader finishes.

        The single-consumer convenience. Starts the reader if it is not
        already running, so the common case needs no separate
        :meth:`start` call.

        Args:
            poll_s: See :data:`DEFAULT_POLL_S`.

        Returns:
            An iterator over raw interleaved I/Q blocks, oldest first.

        Raises:
            DeviceError: If called twice, or on a stream that has
                explicit subscribers. Both used to *work*, in the worst
                sense: a second consumer interleaved with the first and
                each silently received roughly every other block, with
                nothing reporting an error and both looking like they
                worked. Call :meth:`subscribe` once per consumer
                instead.

                Also re-raises anything the reader thread hit, once this
                consumer's queue drains.
        """
        return self._claim_default().blocks(poll_s)

    def timed_blocks(self, poll_s: float = DEFAULT_POLL_S) -> Iterator[TimedBlock]:
        """As :meth:`blocks`, but each block carries when it arrived.

        Args:
            poll_s: See :data:`DEFAULT_POLL_S`.

        Returns:
            An iterator over :class:`TimedBlock`, oldest first.

        Raises:
            DeviceError: Exactly as :meth:`blocks` — the two share one
                implicit subscription, so claiming either claims both.
        """
        return self._claim_default().timed_blocks(poll_s)

    def _claim_default(self) -> IqSubscription:
        """Return the implicit subscription, or explain why there isn't one.

        Deliberately **not** a generator, and neither are :meth:`blocks`
        nor :meth:`timed_blocks`: a generator function's body does not
        run until the first ``next()``, so a second ``blocks()`` call
        would sit there looking accepted and only fail later, from a
        stack frame that does not mention the mistake. Returning an
        iterator built by a plain function makes the refusal land on the
        call that caused it.
        """
        # Refused *before* starting the reader, not after: a call that is
        # going to be rejected should not leave a thread running and a
        # device being read as a side effect.
        if self._default is None and self._subscribers:
            raise DeviceError(
                "This stream has explicit subscribers, so blocks() would add an "
                "unnamed third consumer whose drops nothing would attribute. Call "
                "subscribe() for this consumer as well."
            )
        if self._thread is None:
            self.start()
        if self._default_claimed:
            raise DeviceError(
                "blocks() has already been called on this stream. A second consumer "
                "sharing one subscription would interleave -- each getting roughly "
                "every other block, with both looking like they worked. Call "
                "subscribe() once per consumer instead."
            )
        self._default_claimed = True
        return self._default

    def _ensure_started(self) -> None:
        """Start the reader if it is not running. For :class:`IqSubscription`."""
        if self._thread is None:
            self.start()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> IqStream:
        """Start the reader on entering a ``with`` block."""
        if self._thread is None:
            self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Stop the reader on leaving, whether or not the body raised."""
        self.stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        """The reader thread's body. Runs until stopped or until it fails."""
        try:
            while not self._stop.is_set():
                block = self._device.read_raw(self._block_bytes)
                # Timed before taking the lock and before anything else
                # touches the block: a lock wait here would be measured
                # as device loss.
                self._monitor.record(len(block))
                # The wall clock comes second, so the loss accounting
                # keeps the tighter of the two timestamps. This costs one
                # more clock read per block -- tens of nanoseconds against
                # a 64 ms block -- but the read path is a real-time path
                # (Session 16), so it is worth saying why it is affordable
                # rather than assuming nobody will ask.
                timed = TimedBlock(
                    data=block,
                    read_at=self._now(),
                    duration_s=len(block) / self._byte_rate,
                )
                with self._not_empty:
                    self._blocks_read += 1
                    self._bytes_read += len(block)
                    # The same immutable object goes to every consumer.
                    # No copy, and no consumer can affect another's queue.
                    for subscriber in self._subscribers:
                        subscriber._offer(timed)
                    self._not_empty.notify_all()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            self._error = exc
        finally:
            with self._not_empty:
                self._finished = True
                self._not_empty.notify_all()

    def _reraise(self) -> None:
        """Re-raise whatever killed the reader thread, if anything did.

        **The error is not cleared once raised**, unlike the
        single-consumer version this replaced. With several consumers,
        clearing it would mean whichever one drained first got the
        exception and every other one saw its stream simply end — a
        silent stop with the explanation already consumed by somebody
        else. Every consumer is entitled to be told why the blocks
        stopped.
        """
        error = self._error
        if error is not None:
            raise error
