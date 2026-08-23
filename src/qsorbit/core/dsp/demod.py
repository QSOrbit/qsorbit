"""WBFM demodulation: complex IQ in, mono audio out.

Two things live here.

:func:`shift_to_baseband` is a digital mixer: it multiplies a complex IQ
buffer by a rotating unit-magnitude phasor to move a chosen frequency down
to 0 Hz. It exists because the project's own fixtures — and, per the
bring-up lessons in ``tests/fixtures/iq/README.md``, the tuning convention
worth keeping even live — deliberately place the station of interest
*away* from the tuner's centre frequency, to dodge the RTL-SDR's permanent
DC-offset spike. The discriminator below needs the station sitting at 0 Hz,
so this is the step that gets it there.

:func:`demodulate_wbfm` is the discriminator chain: a quadrature (polar)
discriminator recovers instantaneous frequency, a one-pole de-emphasis
filter undoes the transmitter's pre-emphasis, and the result is decimated
from the IQ sample rate down to an audio rate via
:func:`~qsorbit.core.dsp.decimate.decimate` — the same function
:mod:`~qsorbit.core.dsp.decimate` already uses for IQ, now reused on the
real-valued signal downstream of the discriminator.

**A deliberate simplification, worth knowing before optimising this**: the
discriminator runs at the full input sample rate rather than at some lower
intermediate "quadrature rate." A wideband FM broadcast channel occupies
roughly 200 kHz (Carson's rule: twice the sum of deviation and audio
bandwidth), so the complex IQ cannot be decimated down anywhere near an
audio rate *before* discrimination without clipping the channel itself —
only the real-valued audio that comes out of the discriminator can be
decimated that far. Running the discriminator at, say, 2.048 Msps instead
of an intermediate ~200 kHz "quad rate" costs CPU for no correctness
benefit; it is simple and correct, and cheaper only matters once this is
running against a live stream rather than an offline capture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from qsorbit.core.dsp.decimate import decimate

#: The standard deviation for wideband FM broadcast, in both the US and
#: Europe (narrowband FM, used for the satellite downlinks Chunk G will
#: demodulate, is a different signal entirely at roughly 5 kHz).
DEFAULT_DEVIATION_HZ: float = 75_000.0

#: Default target audio rate. Deliberately not 44,100 or 48,000: this
#: project's captures run at 2.048 Msps, and 2,048,000 = 2**14 * 5**3 has
#: no factor of 3 or 7 in it, so neither of those "CD-quality" rates
#: divides it evenly. 32,000 Hz does (a clean decimate-by-64), which
#: keeps the whole chain on :func:`~qsorbit.core.dsp.decimate.decimate`'s
#: integer-factor-only design rather than introducing a rational
#: resampler for a difference nobody will hear on a first-light check.
DEFAULT_AUDIO_RATE_HZ: float = 32_000.0

#: De-emphasis time constant for US broadcast FM. (Most of the rest of the
#: world uses 50 microseconds instead; pass that explicitly via
#: :class:`WbfmConfig` for a station using that convention.)
DEFAULT_DEEMPHASIS_US: float = 75.0

#: Recovered audio is clipped to this range before being returned, rather
#: than left to whatever a downstream player does with an out-of-range
#: float32 sample. A station modulated within :data:`DEFAULT_DEVIATION_HZ`
#: normalises to comfortably inside +/-1.0; only noise or a mistuned
#: capture should ever reach the clip.
AUDIO_CLIP_RANGE = (-1.0, 1.0)


def shift_to_baseband(iq: np.ndarray, offset_hz: float, sample_rate_hz: float) -> np.ndarray:
    """Digitally mix ``iq`` so that ``offset_hz`` lands at 0 Hz.

    Args:
        iq: Complex IQ samples, in the frame where ``offset_hz`` is
            measured from — typically the tuner's own baseband, where
            0 Hz means the tuned centre frequency.
        offset_hz: The frequency to move to 0 Hz, relative to ``iq``'s
            own baseband. Positive if the frequency of interest sits
            above the current 0 Hz, negative if below.
        sample_rate_hz: The IQ sample rate.

    Returns:
        ``iq`` mixed by a unit-magnitude phasor at ``-offset_hz``, as
        complex64. Same length as ``iq``.

    Raises:
        ValueError: If ``sample_rate_hz`` is not a positive finite
            number, or ``offset_hz`` is not finite.
    """
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError(
            f"sample_rate_hz must be a positive, finite number, got {sample_rate_hz!r}."
        )
    if not math.isfinite(offset_hz):
        raise ValueError(f"offset_hz must be finite, got {offset_hz!r}.")
    if offset_hz == 0.0:
        return iq.astype(np.complex64)

    n = np.arange(iq.shape[0])
    mixer = np.exp(-2j * np.pi * offset_hz * n / sample_rate_hz)
    return (iq * mixer).astype(np.complex64)


@dataclass(frozen=True)
class WbfmConfig:
    """How to demodulate one WBFM channel.

    Args:
        sample_rate_hz: The IQ sample rate ``demodulate_wbfm`` will be
            given.
        audio_rate_hz: Target audio sample rate. ``sample_rate_hz`` must
            divide evenly by this (within floating-point tolerance) —
            see :attr:`decimation_factor`.
        channel_offset_hz: Where the channel of interest sits relative to
            the IQ's own baseband. Non-zero when the capture is tuned off
            the station on purpose — see the module docstring. Passed
            straight to :func:`shift_to_baseband`.
        deviation_hz: The transmitter's peak frequency deviation. Used to
            normalise the discriminator's output so a fully-modulated
            signal lands near +/-1.0.
        de_emphasis_us: De-emphasis time constant in microseconds, or
            ``None`` to skip de-emphasis entirely (mainly useful for
            testing the discriminator in isolation).

    Raises:
        ValueError: If ``sample_rate_hz`` or ``audio_rate_hz`` is not a
            positive finite number, if ``sample_rate_hz`` does not divide
            evenly by ``audio_rate_hz``, if ``deviation_hz`` is not
            positive finite, if ``channel_offset_hz`` is not finite, or if
            ``de_emphasis_us`` is neither ``None`` nor positive finite.
    """

    sample_rate_hz: float
    audio_rate_hz: float = DEFAULT_AUDIO_RATE_HZ
    channel_offset_hz: float = 0.0
    deviation_hz: float = DEFAULT_DEVIATION_HZ
    de_emphasis_us: float | None = DEFAULT_DEEMPHASIS_US

    def __post_init__(self) -> None:
        if not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError(
                f"sample_rate_hz must be a positive, finite number, got {self.sample_rate_hz!r}."
            )
        if not math.isfinite(self.audio_rate_hz) or self.audio_rate_hz <= 0.0:
            raise ValueError(
                f"audio_rate_hz must be a positive, finite number, got {self.audio_rate_hz!r}."
            )
        raw_factor = self.sample_rate_hz / self.audio_rate_hz
        if not math.isclose(raw_factor, round(raw_factor), rel_tol=1e-6):
            raise ValueError(
                f"sample_rate_hz ({self.sample_rate_hz!r}) does not divide evenly by "
                f"audio_rate_hz ({self.audio_rate_hz!r}) -- got a factor of {raw_factor!r}. "
                "decimate() only supports integer factors; pick an audio_rate_hz that "
                "divides sample_rate_hz evenly."
            )
        if round(raw_factor) < 1:
            raise ValueError(
                f"audio_rate_hz ({self.audio_rate_hz!r}) must not exceed "
                f"sample_rate_hz ({self.sample_rate_hz!r})."
            )
        if not math.isfinite(self.channel_offset_hz):
            raise ValueError(f"channel_offset_hz must be finite, got {self.channel_offset_hz!r}.")
        if not math.isfinite(self.deviation_hz) or self.deviation_hz <= 0.0:
            raise ValueError(
                f"deviation_hz must be a positive, finite number, got {self.deviation_hz!r}."
            )
        if self.de_emphasis_us is not None and (
            not math.isfinite(self.de_emphasis_us) or self.de_emphasis_us <= 0.0
        ):
            raise ValueError(
                f"de_emphasis_us must be None or a positive finite number, "
                f"got {self.de_emphasis_us!r}."
            )

    @property
    def decimation_factor(self) -> int:
        """How much :func:`demodulate_wbfm` decimates the audio signal by."""
        return round(self.sample_rate_hz / self.audio_rate_hz)


def demodulate_wbfm(iq: np.ndarray, config: WbfmConfig) -> np.ndarray:
    """Demodulate one WBFM channel to mono audio.

    Args:
        iq: Complex IQ samples at ``config.sample_rate_hz``.
        config: How to demodulate. See :class:`WbfmConfig`.

    Returns:
        Recovered audio as float32, at ``config.audio_rate_hz``, clipped
        to :data:`AUDIO_CLIP_RANGE`. One sample shorter than
        ``len(iq) / config.decimation_factor`` would suggest, before
        :func:`~qsorbit.core.dsp.decimate.decimate`'s own edge-sample
        behaviour is accounted for -- the discriminator below needs a
        pair of samples to produce one output sample, so ``iq`` loses one
        sample before decimation ever sees it.
    """
    baseband = (
        shift_to_baseband(iq, config.channel_offset_hz, config.sample_rate_hz)
        if config.channel_offset_hz != 0.0
        else iq.astype(np.complex64)
    )

    # Polar (quadrature) discriminator: the phase advance between
    # consecutive samples is proportional to instantaneous frequency.
    # Multiplying by the conjugate of the previous sample rather than
    # dividing avoids a division by a near-zero magnitude ever being on
    # the hot path.
    phase_diff = np.angle(baseband[1:] * np.conj(baseband[:-1]))
    instantaneous_freq_hz = phase_diff * (config.sample_rate_hz / (2.0 * np.pi))
    audio = (instantaneous_freq_hz / config.deviation_hz).astype(np.float32)

    if config.de_emphasis_us is not None:
        audio = _apply_deemphasis(audio, config.sample_rate_hz, config.de_emphasis_us)

    decimated = decimate(audio, config.decimation_factor)
    return np.clip(decimated, *AUDIO_CLIP_RANGE).astype(np.float32)


def _apply_deemphasis(
    audio: np.ndarray, sample_rate_hz: float, de_emphasis_us: float
) -> np.ndarray:
    """Apply a one-pole de-emphasis low-pass, undoing the transmitter's pre-emphasis.

    The standard software-radio de-emphasis filter: a single-pole IIR,
    ``y[n] = alpha * x[n] + (1 - alpha) * y[n-1]``, with ``alpha`` set so
    the filter's own time constant matches ``de_emphasis_us``. Applied
    before decimation, at the discriminator's full output rate, so its
    corner frequency is computed against ``sample_rate_hz`` rather than
    the eventual (lower) audio rate.
    """
    tau_s = de_emphasis_us * 1e-6
    dt_s = 1.0 / sample_rate_hz
    alpha = dt_s / (tau_s + dt_s)
    return lfilter([alpha], [1.0, -(1.0 - alpha)], audio).astype(np.float32)
