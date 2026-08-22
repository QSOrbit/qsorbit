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
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
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
        blocks_dropped: Blocks discarded because the buffer was full —
            **our** loss, exactly counted, caused by a slow consumer.
            Fixing this means a larger buffer or a faster consumer;
            changing how the device is read would not touch it.
        block_bytes: Bytes requested per read.
        queue_blocks: Buffer depth in blocks.
        reader_stopped_cleanly: Whether the reader thread exited within
            its join timeout. ``False`` means a read is wedged — see
            :meth:`IqStream.stop`.
        loss: Samples that never crossed USB — the sync-read gap, and
            the number that would justify moving to
            ``rtlsdr_read_async``.
    """

    blocks_read: int
    bytes_read: int
    blocks_dropped: int
    block_bytes: int
    queue_blocks: int
    reader_stopped_cleanly: bool
    loss: LossReport

    @property
    def dropped_bytes(self) -> int:
        """Bytes discarded at the buffer, assuming full-size blocks."""
        return self.blocks_dropped * self.block_bytes

    def describe(self) -> str:
        """Return a summary that keeps the two faults visibly separate."""
        clean = "" if self.reader_stopped_cleanly else "  reader DID NOT stop cleanly\n"
        return (
            f"blocks read {self.blocks_read:,} of {self.block_bytes:,} bytes\n"
            f"  at the device (USB):  {self.loss.describe()}\n"
            f"  at our buffer:        {self.blocks_dropped:,} block(s) dropped "
            f"({self.dropped_bytes:,} bytes), depth {self.queue_blocks}\n" + clean
        )


class IqStream:
    """A reader thread pulling IQ blocks from a device into a bounded buffer.

    Usage::

        with IqStream(sdr) as stream:
            for block in stream.blocks():
                consume(block)
                if enough:
                    break
        print(stream.stats.describe())

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
        queue_blocks: Buffer depth.
        clock: Monotonic clock, injectable for tests.

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
        self._monitor = ThroughputMonitor(byte_rate_for(applied.sample_rate_hz), clock=clock)

        self._blocks: deque[bytes] = deque(maxlen=queue_blocks)
        self._not_empty = threading.Condition(threading.Lock())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._finished = False
        self._error: BaseException | None = None
        self._blocks_read = 0
        self._bytes_read = 0
        self._blocks_dropped = 0
        self._stopped_cleanly = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True`` while the reader thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the reader thread. Starting twice is an error."""
        if self._thread is not None:
            raise DeviceError("This stream has already been started; build a new one.")
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
        return StreamStats(
            blocks_read=self._blocks_read,
            bytes_read=self._bytes_read,
            blocks_dropped=self._blocks_dropped,
            block_bytes=self._block_bytes,
            queue_blocks=self._queue_blocks,
            reader_stopped_cleanly=self._stopped_cleanly,
            loss=self._monitor.report(),
        )

    # ------------------------------------------------------------------
    # Consuming
    # ------------------------------------------------------------------

    def blocks(self, poll_s: float = 0.5) -> Iterator[bytes]:
        """Yield blocks as they arrive, until the reader finishes.

        Starts the reader if it is not already running, so the common
        case needs no separate :meth:`start` call.

        Args:
            poll_s: How often the wait re-checks while idle. Does not
                add latency to a block that arrives — the reader
                notifies — it only bounds how long a shutdown can sit
                unnoticed.

        Yields:
            Raw interleaved I/Q blocks, oldest first.

        Raises:
            DeviceError: Anything the reader thread hit is re-raised
                here once the stream drains, rather than vanishing with
                the thread.
        """
        if self._thread is None:
            self.start()
        while True:
            with self._not_empty:
                while not self._blocks and not self._finished:
                    self._not_empty.wait(poll_s)
                if self._blocks:
                    block = self._blocks.popleft()
                else:
                    break
            yield block
        self._reraise()

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
                with self._not_empty:
                    self._blocks_read += 1
                    self._bytes_read += len(block)
                    if len(self._blocks) == self._queue_blocks:
                        self._blocks_dropped += 1
                    self._blocks.append(block)
                    self._not_empty.notify()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            self._error = exc
        finally:
            with self._not_empty:
                self._finished = True
                self._not_empty.notify_all()

    def _reraise(self) -> None:
        """Re-raise whatever killed the reader thread, if anything did."""
        error = self._error
        if error is not None:
            self._error = None
            raise error
