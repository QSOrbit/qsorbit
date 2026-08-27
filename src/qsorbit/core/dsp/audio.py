"""Audio output: play demodulated audio through an output device.

Wraps ``sounddevice``'s ``OutputStream``, and mirrors
:class:`~qsorbit.core.sdr.stream.IqStream`'s producer/consumer shape while
facing the opposite direction: there, a reader thread fills a bounded
buffer that a consumer drains; here, :meth:`AudioOutput.write` fills a
bounded buffer that PortAudio's own callback thread drains, pulling
whatever demodulated audio is ready to be heard. The two threads meet only
at a lock-protected deque, same as there.

**Import timing matters here in a way it does not for pure DSP code.**
``sounddevice`` raises ``OSError`` at *import* time — not when a stream is
opened — if the PortAudio shared library is not installed, which is the
case in this project's own cloud sandbox (confirmed while building this
module: no apt or PyPI access there to install it). If this module
imported ``sounddevice`` at the top, then ``core/dsp/__init__.py``
re-exporting :class:`AudioOutput` would make the *entire* ``core.dsp``
package fail to import anywhere PortAudio is missing. So the import is
deferred to :func:`_default_stream_factory`, called only when a real
stream is actually opened — the same reason
:mod:`qsorbit.core.sdr.librtlsdr` only calls ``ctypes.CDLL()`` inside
``load_library()``, never at module scope. This is what lets
:class:`AudioOutput`'s queueing and accounting logic be unit-tested
anywhere, with a fake stream factory standing in for PortAudio, the same
way SDR unit tests stand in a fake for the ctypes binding.

**The buffer discards the OLDEST block when full**, for the same reason
:class:`~qsorbit.core.sdr.stream.IqStream` does on the receive side: stale
audio is worth less than live audio to a real-time listener, so a
consumer recovering from a producer hiccup should hear the present rather
than a queued-up backlog working through a delay.

**Two different faults, counted separately**, mirroring
:class:`~qsorbit.core.sdr.stream.StreamStats`'s ``blocks_dropped`` versus
``loss`` split: :attr:`AudioStats.blocks_dropped` is *our* fault, a
producer writing faster than playback drains — exactly counted, since we
own the buffer. :attr:`AudioStats.underruns` is the opposite direction —
PortAudio's callback wanted audio and the buffer was empty, so silence was
substituted. They have different causes (a slow producer versus a buffer
too small or a slow consumer never actually applies here) and reporting
them as one number would send you to fix the wrong thing.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

#: Blocks the buffer holds before it starts discarding the oldest one.
#: Unlike :data:`~qsorbit.core.sdr.stream.DEFAULT_QUEUE_BLOCKS`, a block
#: here has no fixed size — it is whatever one :meth:`AudioOutput.write`
#: call was given — so this bounds queue depth in *calls*, not in time.
#: A caller writing consistently-sized blocks can reason about the time
#: this represents; one that does not should size its own writes with
#: that in mind.
DEFAULT_QUEUE_BLOCKS: Final = 8

StreamFactory = Callable[..., Any]


class AudioError(RuntimeError):
    """Raised for audio-output lifecycle misuse: writing before
    :meth:`AudioOutput.start`, or starting a stream twice.
    """


@dataclass(frozen=True)
class AudioStats:
    """What one playback run did, with its two faults kept apart.

    Args:
        blocks_written: Blocks accepted by :meth:`AudioOutput.write`.
        blocks_played: Blocks fully drained by the playback callback.
        blocks_dropped: Blocks discarded because the buffer was full —
            **our** loss: the producer (the demod chain) wrote faster
            than playback drained. Counted exactly, since we own the
            buffer. A bigger buffer or a faster producer fixes this;
            nothing about the audio device is at fault.
        frames_played: Samples handed to the audio device, including any
            silence substituted for an underrun.
        underruns: Times the playback callback needed audio and the
            buffer was empty, so silence was played instead. Points at
            the opposite problem from ``blocks_dropped``: the producer
            fell behind, or the buffer is too small for how bursty it is.
    """

    blocks_written: int
    blocks_played: int
    blocks_dropped: int
    frames_played: int
    underruns: int

    def describe(self) -> str:
        """Return a short human-readable summary, for logs and reports."""
        return (
            f"{self.blocks_written:,} block(s) written, {self.blocks_played:,} played "
            f"({self.frames_played:,} frames)\n"
            f"  buffer (writer):   {self.blocks_dropped:,} block(s) dropped, buffer full\n"
            f"  device (playback): {self.underruns:,} underrun(s), buffer empty"
        )


class AudioOutput:
    """Streams float32 mono audio to an output device.

    The system default unless ``device`` says otherwise.

    Usage::

        with AudioOutput(32_000.0) as output:
            for block in audio_blocks:
                output.write(block)
        print(output.stats.describe())

    Args:
        sample_rate_hz: Output sample rate.
        queue_blocks: Buffer depth — see :data:`DEFAULT_QUEUE_BLOCKS`.
        device: Which output device to open, in whatever form
            ``sounddevice`` itself accepts for ``OutputStream(device=...)``
            — a numeric index, a name substring, or ``None`` for
            PortAudio's own configured system default. Read once, at
            :meth:`start` — plugging in headphones mid-run does nothing,
            the same way it already did nothing before this parameter
            existed, since the stream is opened once and PortAudio does
            not migrate a running stream to a new device.
        stream_factory: Builds the underlying PortAudio stream. Injected
            for tests, the same way :class:`~qsorbit.core.sdr.device.RtlSdr`
            takes an injected ``_lib`` — a fake here means no real audio
            hardware, and no PortAudio import, is ever touched. Defaults
            to :func:`_default_stream_factory`.

    Raises:
        ValueError: If ``sample_rate_hz`` is not a positive finite
            number, or ``queue_blocks`` is not a positive integer.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        *,
        queue_blocks: int = DEFAULT_QUEUE_BLOCKS,
        device: int | str | None = None,
        stream_factory: StreamFactory | None = None,
    ) -> None:
        if sample_rate_hz <= 0.0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz!r}.")
        if queue_blocks <= 0:
            raise ValueError(f"queue_blocks must be positive, got {queue_blocks!r}.")

        self._sample_rate_hz = sample_rate_hz
        self._device = device
        self._stream_factory = stream_factory or _default_stream_factory
        self._stream: Any = None

        self._lock = threading.Lock()
        self._blocks: deque[np.ndarray] = deque(maxlen=queue_blocks)
        self._blocks_written = 0
        self._blocks_played = 0
        self._blocks_dropped = 0
        self._frames_played = 0
        self._underruns = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """``True`` while a stream is open."""
        return self._stream is not None

    def start(self) -> None:
        """Open the output stream. Starting twice is an error."""
        if self._stream is not None:
            raise AudioError("This output has already been started; build a new one.")
        stream = self._stream_factory(
            samplerate=self._sample_rate_hz,
            channels=1,
            dtype="float32",
            callback=self._callback,
            device=self._device,
        )
        stream.start()
        self._stream = stream

    @property
    def device(self) -> int | str | None:
        """The device this output was configured with — see the class docs."""
        return self._device

    def stop(self) -> AudioStats:
        """Close the stream, if open, and return the run's statistics."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return self.stats

    @property
    def stats(self) -> AudioStats:
        """The run's statistics so far."""
        with self._lock:
            return AudioStats(
                blocks_written=self._blocks_written,
                blocks_played=self._blocks_played,
                blocks_dropped=self._blocks_dropped,
                frames_played=self._frames_played,
                underruns=self._underruns,
            )

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(self, samples: np.ndarray) -> None:
        """Queue one block of mono audio for playback.

        Args:
            samples: One-dimensional audio samples, e.g. from
                :func:`~qsorbit.core.dsp.demod.demodulate_wbfm`. Cast to
                float32 if not already.

        Raises:
            AudioError: If the stream has not been started.
            ValueError: If ``samples`` is not one-dimensional.
        """
        if self._stream is None:
            raise AudioError("write() called before start() — nothing is playing.")
        if samples.ndim != 1:
            raise ValueError(f"samples must be one-dimensional, got shape {samples.shape!r}.")

        block = samples if samples.dtype == np.float32 else samples.astype(np.float32)
        with self._lock:
            if len(self._blocks) == self._blocks.maxlen:
                self._blocks_dropped += 1
            self._blocks.append(block)
            self._blocks_written += 1

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> AudioOutput:
        """Start the stream on entering a ``with`` block."""
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Stop the stream on leaving, whether or not the body raised."""
        self.stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _callback(
        self, outdata: np.ndarray, frames: int, time_info: object, status: object
    ) -> None:
        """PortAudio's pull callback: fill exactly ``frames`` samples.

        Runs on PortAudio's own thread, never the thread that calls
        :meth:`write`. Takes the lock only long enough to drain queued
        blocks — never while touching ``outdata`` past what came out of
        the queue — so a slow writer cannot make this callback itself run
        long, which is the one thing that must never happen in an audio
        callback.
        """
        filled = 0
        with self._lock:
            while filled < frames and self._blocks:
                block = self._blocks[0]
                take = min(len(block), frames - filled)
                outdata[filled : filled + take, 0] = block[:take]
                filled += take
                if take < len(block):
                    self._blocks[0] = block[take:]
                else:
                    self._blocks.popleft()
                    self._blocks_played += 1
            if filled < frames:
                self._underruns += 1
            self._frames_played += frames
        if filled < frames:
            outdata[filled:frames, 0] = 0.0


def _default_stream_factory(
    *,
    samplerate: float,
    channels: int,
    dtype: str,
    callback: Callable[..., None],
    device: int | str | None = None,
) -> Any:
    """Build a real ``sounddevice.OutputStream``.

    ``sounddevice`` is imported here, not at module scope — see the
    module docstring for why that matters.
    """
    import sounddevice

    return sounddevice.OutputStream(
        samplerate=samplerate, channels=channels, dtype=dtype, callback=callback, device=device
    )
