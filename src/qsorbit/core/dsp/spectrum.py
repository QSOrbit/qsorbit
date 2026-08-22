"""Power-spectrum frames, sized for a waterfall consumer.

A frame is one windowed FFT of :data:`SpectrumConfig.fft_size` complex
samples, turned into power in dB. :func:`frame_iq` is what turns a longer
IQ buffer into a sequence of frames — the shape a live waterfall (Chunk F)
and an offline analysis both want: "here is frame N, and here is the
frequency axis that goes with it."

Frequency, not bin index, is the unit everything downstream should reason
in. :func:`frequency_axis_hz` returns absolute frequencies (baseband offset
plus :attr:`SpectrumConfig.center_freq_hz`) in the same fftshift order as
:func:`power_spectrum_db`'s output, precisely so a caller never has to
reconstruct that mapping itself and get the shift direction wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import get_window

#: A conservative floor for the power-in-dB output. Not a physical limit —
#: it exists so a frame of exact zeros (silence, or a synthetic test signal)
#: produces a finite number instead of -inf, without meaningfully affecting
#: any frame that actually contains a signal or noise floor.
DEFAULT_FLOOR_DB: float = -200.0


@dataclass(frozen=True)
class SpectrumConfig:
    """How to turn a block of IQ samples into one spectrum frame.

    Args:
        fft_size: Samples per frame, and per FFT. Larger gives finer
            frequency resolution at the cost of coarser time resolution —
            the usual waterfall trade-off. Must be at least 4; a smaller
            FFT does not resolve enough to be useful for anything this
            module is for.
        sample_rate_hz: The IQ sample rate the frame was captured at.
            Needed to turn bin index into Hz; not touched otherwise.
        center_freq_hz: The tuner's centre frequency, if this spectrum is
            being computed from real radio samples. Defaults to 0.0 for
            baseband/synthetic analysis, where "frequency" already means
            frequency, not an offset from some other centre.
        window: Name of a window function, passed to
            :func:`scipy.signal.windows.get_window`. Checked at
            construction time — the same "what can be checked without a
            device present lives in the config" rule
            :class:`~qsorbit.core.sdr.config.SdrConfig` follows, and a
            typo'd window name is exactly the kind of mistake that should
            fail immediately rather than inside a hot loop.

    Raises:
        ValueError: If ``fft_size`` is unusable, ``sample_rate_hz`` is not
            a positive finite number, ``center_freq_hz`` is not finite, or
            ``window`` is not a name :func:`scipy.signal.windows.get_window`
            recognizes.
    """

    fft_size: int
    sample_rate_hz: float
    center_freq_hz: float = 0.0
    window: str = "hann"

    def __post_init__(self) -> None:
        if isinstance(self.fft_size, bool) or not isinstance(self.fft_size, int):
            raise ValueError(f"fft_size must be an int, got {self.fft_size!r}.")
        if self.fft_size < 4:
            raise ValueError(f"fft_size must be at least 4, got {self.fft_size}.")

        if not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError(
                f"sample_rate_hz must be a positive, finite number, got {self.sample_rate_hz!r}."
            )

        if not math.isfinite(self.center_freq_hz):
            raise ValueError(f"center_freq_hz must be finite, got {self.center_freq_hz!r}.")

        try:
            get_window(self.window, self.fft_size, fftbins=True)
        except ValueError as exc:
            raise ValueError(
                f"window {self.window!r} is not a window scipy.signal.windows.get_window "
                f"recognizes: {exc}"
            ) from exc


def _window_taps(config: SpectrumConfig) -> np.ndarray:
    """The window function's sample values, as float64.

    A free function rather than a method, matching the value-object style
    :class:`~qsorbit.core.sdr.config.SdrConfig` uses: the dataclass holds
    data and the validation that needs no device, and behaviour that
    depends on it lives beside it rather than on it.
    """
    return np.asarray(get_window(config.window, config.fft_size, fftbins=True), dtype=np.float64)


def frequency_axis_hz(config: SpectrumConfig) -> np.ndarray:
    """Return the absolute frequency, in Hz, that each output bin represents.

    Ordered to match :func:`power_spectrum_db`'s output: lowest frequency
    first, DC (or ``center_freq_hz``) at the middle bin. Computing this once
    per :class:`SpectrumConfig` rather than per frame is deliberate — the
    axis does not change frame to frame, and a caller building a waterfall
    wants it exactly once, not recomputed every row.
    """
    baseband = np.fft.fftshift(np.fft.fftfreq(config.fft_size, d=1.0 / config.sample_rate_hz))
    return baseband + config.center_freq_hz


def power_spectrum_db(
    iq: np.ndarray, config: SpectrumConfig, *, floor_db: float = DEFAULT_FLOOR_DB
) -> np.ndarray:
    """Compute one windowed power-spectrum frame, in dB.

    Args:
        iq: Exactly ``config.fft_size`` complex samples — one frame. Use
            :func:`frame_iq` to split a longer buffer into frames rather
            than slicing by hand, so the frame length is never a place a
            caller can quietly get it wrong.
        config: What FFT size, sample rate, and window to use.
        floor_db: Power below this is clamped up to it, so an all-zero
            frame (silence, or a synthetic test signal with no noise)
            returns a finite array instead of ``-inf``.

    Returns:
        Power in dB, as float32, length ``config.fft_size``, ordered to
        match :func:`frequency_axis_hz` — lowest frequency first.

    Raises:
        ValueError: If ``iq`` is not exactly one frame's worth of samples.
    """
    if iq.ndim != 1 or iq.shape[0] != config.fft_size:
        got = iq.shape[0] if iq.ndim == 1 else iq.shape
        raise ValueError(
            f"power_spectrum_db operates on exactly one frame of {config.fft_size} "
            f"samples, got {got}. Use frame_iq() to split a longer buffer."
        )

    window = _window_taps(config)
    windowed = iq * window
    spectrum = np.fft.fftshift(np.fft.fft(windowed))

    # Normalise by the window's coherent gain (its sum) so that swapping
    # window functions does not also swap the reported power level for the
    # same input signal -- a rectangular window and a Hann window should
    # agree on a tone's amplitude, and only disagree on sidelobe leakage.
    coherent_gain = window.sum()
    magnitude = np.abs(spectrum) / coherent_gain
    power = magnitude**2

    floor_power = 10.0 ** (floor_db / 10.0)
    power_db = 10.0 * np.log10(np.maximum(power, floor_power))
    return power_db.astype(np.float32)


def frame_iq(iq: np.ndarray, config: SpectrumConfig, *, hop: int | None = None):
    """Yield successive ``config.fft_size``-length frames from ``iq``.

    Args:
        iq: A complex IQ buffer, any length.
        config: Supplies the frame length (``fft_size``).
        hop: Samples to advance between frames. Defaults to
            ``config.fft_size`` (non-overlapping frames, the usual case).
            A smaller hop overlaps frames, which a waterfall can use to
            scroll more smoothly without needing more input samples per
            row.

    Yields:
        Views into ``iq``, each exactly ``config.fft_size`` samples long,
        each suitable to pass directly to :func:`power_spectrum_db`.
        Trailing samples that do not fill a complete frame are dropped —
        a waterfall consumer that wants them has to wait for the next
        buffer to arrive, the same way :class:`~qsorbit.core.sdr.stream.IqStream`
        deals in whole blocks rather than partial ones.

    Raises:
        ValueError: If ``hop`` is not positive.
    """
    step = config.fft_size if hop is None else hop
    if step <= 0:
        raise ValueError(f"hop must be positive, got {step!r}.")

    start = 0
    n = iq.shape[0]
    while start + config.fft_size <= n:
        yield iq[start : start + config.fft_size]
        start += step
