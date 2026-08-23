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
        queue_frames: Buffer depth, in frames.
        worker_stopped_cleanly: Whether the worker exited within its join
            timeout.

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
        return (
            f"{self.blocks_consumed:,} block(s), {self.samples_consumed:,} samples\n"
            f"  computed:  {self.frames_computed:,} frame(s) "
            f"({self.compute_fraction * 100:.1f}% of {self.frames_available:,} available)\n"
            f"  skipped by design: {self.frames_skipped:,} frame(s) - "
            f"the display cannot use them, not a fault\n"
            f"  dropped at buffer: {self.frames_dropped:,} frame(s) - "
            f"consumer behind, depth {self.queue_frames}\n" + clean
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


class SpectrumStream:
    """A worker thread turning raw IQ blocks into spectrum frames.

    Usage, wired to a real device::

        with IqStream(sdr) as iq, SpectrumStream(iq.blocks(), config) as frames:
            for frame in frames.latest():
                paint(frame)

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
    ) -> None:
        if queue_frames <= 0:
            raise ValueError(f"queue_frames must be positive, got {queue_frames!r}.")

        self._blocks = blocks
        self._config = config
        self._hop = hop_for_frame_rate(config, frame_rate_hz)
        self._queue_frames = queue_frames
        self._now = now

        self._frames: deque[SpectrumFrame] = deque(maxlen=queue_frames)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._stopped_cleanly = True

        self._blocks_consumed = 0
        self._samples_consumed = 0
        self._frames_computed = 0
        self._frames_dropped = 0

        # Frame starts are tracked as absolute sample indices so the
        # cadence carries across block boundaries. Restarting the hop at
        # every block would space frames unevenly whenever the block size
        # is not a whole number of hops -- which is the normal case -- and
        # uneven spacing on a waterfall is a distorted time axis, not a
        # cosmetic problem.
        self._next_frame_at = 0
        self._carry = np.empty(0, dtype=np.complex64)
        self._carry_origin = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def hop(self) -> int:
        """Samples between frame starts. See :func:`hop_for_frame_rate`."""
        return self._hop

    @property
    def is_running(self) -> bool:
        """``True`` while the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the worker thread. Starting twice is an error."""
        if self._thread is not None:
            raise RuntimeError("This stream has already been started; build a new one.")
        self._thread = threading.Thread(
            target=self._work_loop,
            name="qsorbit-spectrum-worker",
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
            return SpectrumStreamStats(
                blocks_consumed=self._blocks_consumed,
                samples_consumed=samples,
                frames_computed=computed,
                frames_skipped=max(0, samples // self._config.fft_size - computed),
                frames_dropped=self._frames_dropped,
                queue_frames=self._queue_frames,
                worker_stopped_cleanly=self._stopped_cleanly,
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

        Raises:
            Whatever killed the worker thread, re-raised here rather than
            vanishing with the thread. A waterfall that quietly freezes
            when the device disappears is exactly the silent failure this
            project keeps meeting; a caller driving this from a timer
            should catch, stop the timer, and show the error — the way
            :class:`~qsorbit.ui.readout_window.ReadoutWindow` already
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
        with self._lock:
            frames = list(self._frames)
            self._frames.clear()
        if not frames:
            self._reraise()
        return frames

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
                self._consume(unpack_uint8_iq(raw))
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            self._error = exc

    def _consume(self, samples: np.ndarray) -> None:
        """Compute whatever frames this block completes, and buffer them."""
        buffer = samples if self._carry.size == 0 else np.concatenate((self._carry, samples))
        end = self._carry_origin + buffer.size
        fft_size = self._config.fft_size

        computed: list[SpectrumFrame] = []
        while self._next_frame_at + fft_size <= end:
            start = self._next_frame_at - self._carry_origin
            power_db = power_spectrum_db(buffer[start : start + fft_size], self._config)
            computed.append(SpectrumFrame(power_db=power_db, time=self._now()))
            self._next_frame_at += self._hop

        # Keep only what a future frame might still need. When the hop
        # runs past the end of this block -- a low frame rate, so most
        # blocks contribute nothing -- this discards the whole buffer
        # rather than growing it, which is what stops a slow frame rate
        # from turning into a memory leak.
        keep_from = min(max(self._next_frame_at - self._carry_origin, 0), buffer.size)
        self._carry = buffer[keep_from:].copy()
        self._carry_origin += keep_from

        with self._lock:
            self._blocks_consumed += 1
            self._samples_consumed += samples.size
            self._frames_computed += len(computed)
            for frame in computed:
                if len(self._frames) == self._queue_frames:
                    self._frames_dropped += 1
                self._frames.append(frame)

    def _reraise(self) -> None:
        """Re-raise whatever killed the worker thread, if anything did."""
        error = self._error
        if error is not None:
            self._error = None
            raise error
