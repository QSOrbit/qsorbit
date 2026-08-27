"""FM demodulation, wideband and narrowband: complex IQ in, mono audio out.

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

:func:`demodulate_nbfm` is the same chain for narrowband FM — the FM
repeaters, NOAA weather channels and satellite downlinks this project
actually exists for — and **the paragraph above inverts for it**. A
narrowband channel is about 16 kHz wide rather than 200 kHz, so decimating
the complex IQ down to a narrow intermediate rate *before* discrimination
is not merely allowed, it is mandatory. Two separate reasons, either one
sufficient:

- **Capture effect.** A discriminator run at 2.048 Msps on a 16 kHz
  channel sees the entire 2 MHz of noise and every other signal in it, and
  FM's capture effect then hands back whichever signal is loudest *in the
  whole capture* rather than the one at the requested frequency. That is
  Session 14's "ask whether our signal is present at our frequency, never
  whether it is the loudest" showing up in the DSP instead of in a
  bring-up checker.
- **Adjacent-channel rejection.** The anti-aliasing low-pass inside
  :func:`~qsorbit.core.dsp.decimate.decimate` is not just protecting the
  resample — at these rates it *is* the channel filter. NOAA weather radio
  spaces its channels 25 kHz apart, so with the default 32 kHz IF rate
  (whose filter corner lands near 12.8 kHz) the neighbour at +25 kHz is
  well into the stopband. Without it, that neighbour would alias to about
  7 kHz, directly on top of the wanted channel. Both NOAA fixtures in
  ``tests/fixtures/iq/`` have exactly such a neighbour, and a test builds
  the same situation synthetically.

The narrowband chain therefore runs shift, *then* channel filter, then
discriminate at the IF rate, then de-emphasis, then decimate to audio.
Squelch is optional and lives in :mod:`qsorbit.core.dsp.squelch`; see that
module for why it defaults to off and why it must measure the
discriminator's output rather than the finished audio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from qsorbit.core.dsp.decimate import decimate
from qsorbit.core.dsp.squelch import NoiseSquelch

#: The standard deviation for wideband FM broadcast, in both the US and
#: Europe. Narrowband FM's equivalent is
#: :data:`DEFAULT_NBFM_DEVIATION_HZ`, roughly fifteen times smaller.
#:
#: (Naming note: this constant and :data:`DEFAULT_DEEMPHASIS_US` predate
#: there being two demodulators, so they carry no ``WBFM`` in their names
#: while their narrowband counterparts carry ``NBFM``. Renaming the pair
#: for symmetry is a candidate for Chunk I; it is not done here because it
#: would churn the WBFM tests and the out-of-repo bench script for a purely
#: cosmetic gain inside a feature PR.)
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

#: Peak deviation for the narrowband FM this project actually targets:
#: amateur repeaters, the FM satellite downlinks, and NOAA weather radio
#: all run at or near 5 kHz.
DEFAULT_NBFM_DEVIATION_HZ: float = 5_000.0

#: Default intermediate rate the narrowband channel is filtered down to
#: before discrimination — see the module docstring for why narrowband
#: *must* do this and wideband must not.
#:
#: 32 kHz is chosen against three constraints at once, and it is worth
#: recording all three because moving it can break any of them silently:
#:
#: 1. **It divides this project's 2.048 Msps captures evenly** (by 64),
#:    keeping the whole chain on
#:    :func:`~qsorbit.core.dsp.decimate.decimate`'s integer-factor design.
#: 2. **Its channel filter rejects the adjacent channel.**
#:    :func:`scipy.signal.decimate` puts its corner at 0.8 of the new
#:    Nyquist, so a 32 kHz output filters at roughly 12.8 kHz: a 16 kHz
#:    wide channel (+/-8 kHz) passes intact and NOAA's neighbour at
#:    25 kHz does not. Raising this rate *widens* that filter and lets the
#:    neighbour back in.
#: 3. **It leaves the discriminator far from wrapping.** Peak phase
#:    advance per sample is ``2*pi*deviation/if_rate``, which must stay
#:    under pi or the discriminator wraps and the audio is quietly
#:    garbage. At 5 kHz deviation and 32 kHz this is 0.98 rad against a
#:    limit of 3.14. :class:`NbfmConfig` validates the limit itself.
DEFAULT_NBFM_IF_RATE_HZ: float = 32_000.0

#: Fewest IF samples :func:`demodulate_nbfm` will accept a block reducing
#: to. The discriminator itself only needs two, but the channel filter runs
#: first and :func:`scipy.signal.decimate`'s zero-phase filtering needs its
#: input to exceed an internal pad length (27 samples for the order-8
#: Chebyshev design it uses by default). That constraint applies to *each*
#: chained stage, and the last stage is the one with the fewest samples to
#: work with. Requiring 64 IF samples means the last stage always sees at
#: least 128, which clears the pad with room to spare for any decimation
#: factor and does not depend on knowing scipy's internals exactly. Real
#: blocks are far above it: a 256 KiB IQ block is 2,048 IF samples.
MIN_IF_SAMPLES: int = 64

#: De-emphasis time constant for narrowband FM — the land-mobile and
#: amateur convention, a corner near 212 Hz, ten times broadcast FM's
#: :data:`DEFAULT_DEEMPHASIS_US`. It doubles as this chain's audio
#: low-pass: at 6 dB per octave from 212 Hz it is already about 25 dB down
#: by 4 kHz, which is most of what keeps residual discriminator hiss out
#: of the speaker when the squelch is off.
DEFAULT_NBFM_DEEMPHASIS_US: float = 750.0


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

    audio = discriminate(baseband, config.sample_rate_hz, config.deviation_hz)

    if config.de_emphasis_us is not None:
        audio = _apply_deemphasis(audio, config.sample_rate_hz, config.de_emphasis_us)

    decimated = decimate(audio, config.decimation_factor)
    return np.clip(decimated, *AUDIO_CLIP_RANGE).astype(np.float32)


def discriminate(baseband: np.ndarray, sample_rate_hz: float, deviation_hz: float) -> np.ndarray:
    """Recover instantaneous frequency from a complex FM signal at baseband.

    A polar (quadrature) discriminator: the phase advance between
    consecutive samples is proportional to instantaneous frequency.
    Multiplying by the conjugate of the previous sample rather than
    dividing avoids a division by a near-zero magnitude ever being on the
    hot path.

    Shared by both demodulators rather than written twice — the same "one
    implementation, one place to have gotten it right" argument
    :mod:`~qsorbit.core.dsp.decimate` makes. Public because
    :mod:`qsorbit.core.dsp.squelch` measures this signal directly, and a
    caller assembling a custom chain needs the same access.

    Args:
        baseband: Complex samples with the channel of interest already at
            0 Hz — use :func:`shift_to_baseband` first if it is not.
        sample_rate_hz: The rate ``baseband`` is sampled at.
        deviation_hz: Peak deviation to normalise against, so a
            fully-modulated signal lands near +/-1.0.

    Returns:
        Instantaneous frequency scaled by ``deviation_hz``, as float32,
        **one sample shorter than** ``baseband`` — a phase *difference*
        needs a pair of samples to produce one output.

    Raises:
        ValueError: If ``baseband`` holds fewer than two samples, or if
            ``sample_rate_hz`` or ``deviation_hz`` is not a positive
            finite number.
    """
    if baseband.ndim != 1 or baseband.shape[0] < 2:
        raise ValueError(
            f"discriminate needs at least two samples to form a phase difference, "
            f"got shape {baseband.shape!r}."
        )
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError(
            f"sample_rate_hz must be a positive, finite number, got {sample_rate_hz!r}."
        )
    if not math.isfinite(deviation_hz) or deviation_hz <= 0.0:
        raise ValueError(f"deviation_hz must be a positive, finite number, got {deviation_hz!r}.")

    phase_diff = np.angle(baseband[1:] * np.conj(baseband[:-1]))
    instantaneous_freq_hz = phase_diff * (sample_rate_hz / (2.0 * np.pi))
    return (instantaneous_freq_hz / deviation_hz).astype(np.float32)


@dataclass(frozen=True)
class NbfmConfig:
    """How to demodulate one narrowband FM channel.

    Follows :class:`WbfmConfig`'s frozen-dataclass-plus-validation shape,
    with one extra stage in the middle: ``sample_rate_hz`` is filtered and
    decimated down to ``if_rate_hz`` *before* discrimination, which is the
    structural difference between narrowband and wideband FM. See the
    module docstring.

    Args:
        sample_rate_hz: The IQ sample rate :func:`demodulate_nbfm` will be
            given.
        if_rate_hz: Intermediate rate the channel is filtered down to
            before discrimination. ``sample_rate_hz`` must divide evenly
            by it. Defaults to :data:`DEFAULT_NBFM_IF_RATE_HZ`, whose
            docstring explains the three constraints that pick it — read
            that before changing this.
        audio_rate_hz: Target audio sample rate. ``if_rate_hz`` must
            divide evenly by it, and must not be below it. With both
            defaults this factor is 1 and the final decimation is a
            no-op copy, which is expected rather than a mistake: the
            channel filter has already brought the signal to a rate a
            speaker can use.
        channel_offset_hz: Where the channel of interest sits relative to
            the IQ's own baseband. Non-zero when the capture is tuned off
            the station on purpose — see the module docstring. Passed
            straight to :func:`shift_to_baseband`.
        deviation_hz: The transmitter's peak frequency deviation, used to
            normalise the discriminator's output.
        de_emphasis_us: De-emphasis time constant in microseconds, or
            ``None`` to skip it (mainly useful for testing the
            discriminator in isolation, and for measuring the raw
            discriminator output the way
            :mod:`qsorbit.core.dsp.squelch` does).

    Raises:
        ValueError: If any rate is not a positive finite number, if
            ``sample_rate_hz`` does not divide evenly by ``if_rate_hz``,
            if ``if_rate_hz`` does not divide evenly by ``audio_rate_hz``,
            if ``audio_rate_hz`` exceeds ``if_rate_hz``, if
            ``channel_offset_hz`` is not finite, if ``deviation_hz`` is
            not positive finite, if ``de_emphasis_us`` is neither ``None``
            nor positive finite, or if ``if_rate_hz`` is too low for
            ``deviation_hz`` (see below).

    **The deviation-versus-IF-rate check is the one worth knowing about.**
    A quadrature discriminator measures phase advance per sample, and
    phase advance is only unambiguous below pi radians. Peak advance is
    ``2*pi*deviation_hz/if_rate_hz``, so ``if_rate_hz`` must exceed
    ``2*deviation_hz`` or every modulation peak wraps and the recovered
    audio is garbage — quietly, with no exception and no obviously wrong
    number anywhere. Rejecting it at construction time is the same
    principle :class:`~qsorbit.core.sdr.config.SdrConfig` follows: check
    what can be checked before a hot loop, not inside one. Note that
    Carson's rule wants rather more headroom than the bare limit
    (``2*(deviation + audio bandwidth)``, about 16 kHz for a 5 kHz
    channel); the validation enforces only the hard aliasing floor, since
    the audio bandwidth is not a value this config knows.
    """

    sample_rate_hz: float
    if_rate_hz: float = DEFAULT_NBFM_IF_RATE_HZ
    audio_rate_hz: float = DEFAULT_AUDIO_RATE_HZ
    channel_offset_hz: float = 0.0
    deviation_hz: float = DEFAULT_NBFM_DEVIATION_HZ
    de_emphasis_us: float | None = DEFAULT_NBFM_DEEMPHASIS_US

    def __post_init__(self) -> None:
        for name in ("sample_rate_hz", "if_rate_hz", "audio_rate_hz"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive, finite number, got {value!r}.")

        channel_factor = self.sample_rate_hz / self.if_rate_hz
        if not math.isclose(channel_factor, round(channel_factor), rel_tol=1e-6):
            raise ValueError(
                f"sample_rate_hz ({self.sample_rate_hz!r}) does not divide evenly by "
                f"if_rate_hz ({self.if_rate_hz!r}) -- got a factor of {channel_factor!r}. "
                "decimate() only supports integer factors; pick an if_rate_hz that "
                "divides sample_rate_hz evenly."
            )
        if round(channel_factor) < 1:
            raise ValueError(
                f"if_rate_hz ({self.if_rate_hz!r}) must not exceed "
                f"sample_rate_hz ({self.sample_rate_hz!r})."
            )

        audio_factor = self.if_rate_hz / self.audio_rate_hz
        if not math.isclose(audio_factor, round(audio_factor), rel_tol=1e-6):
            raise ValueError(
                f"if_rate_hz ({self.if_rate_hz!r}) does not divide evenly by "
                f"audio_rate_hz ({self.audio_rate_hz!r}) -- got a factor of "
                f"{audio_factor!r}. decimate() only supports integer factors."
            )
        if round(audio_factor) < 1:
            raise ValueError(
                f"audio_rate_hz ({self.audio_rate_hz!r}) must not exceed "
                f"if_rate_hz ({self.if_rate_hz!r})."
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

        if self.if_rate_hz <= 2.0 * self.deviation_hz:
            raise ValueError(
                f"if_rate_hz ({self.if_rate_hz!r}) must be more than twice deviation_hz "
                f"({self.deviation_hz!r}), or the discriminator's phase advance exceeds pi "
                f"radians at modulation peaks and wraps -- which corrupts the audio "
                f"silently rather than raising. Carson's rule wants more headroom still: "
                f"about {2.0 * (self.deviation_hz + 3_000.0):,.0f} Hz for a voice channel."
            )

    @property
    def channel_decimation_factor(self) -> int:
        """How much the channel filter decimates the IQ by, before discrimination."""
        return round(self.sample_rate_hz / self.if_rate_hz)

    @property
    def audio_decimation_factor(self) -> int:
        """How much the recovered audio is decimated by, after discrimination."""
        return round(self.if_rate_hz / self.audio_rate_hz)


def demodulate_nbfm(
    iq: np.ndarray, config: NbfmConfig, *, squelch: NoiseSquelch | None = None, mute: bool = True
) -> np.ndarray:
    """Demodulate one narrowband FM channel to mono audio.

    Args:
        iq: Complex IQ samples at ``config.sample_rate_hz``.
        config: How to demodulate. See :class:`NbfmConfig`.
        squelch: Optional noise gate. ``None`` — the default — means no
            squelch at all, which is deliberate: see
            :mod:`qsorbit.core.dsp.squelch` for why a mute enabled by
            default is a liability rather than a convenience. When given,
            it is **stateful and must be reused across blocks**, or its
            hysteresis has nothing to remember; construct one per listening
            session, not one per block.
        mute: Whether a closed gate actually silences the returned audio.
            Ignored when ``squelch`` is ``None``. Defaults to ``True``,
            matching this function's behaviour before this parameter
            existed. Chunk I decoupled *measuring* quieting from *muting*
            on it (see :mod:`qsorbit.core.dsp.squelch`'s module
            docstring, "always measure and optionally mute") - passing a
            squelch with ``mute=False`` gets every measurement and every
            open/close decision exactly as if muting were on, with the
            gate's decision never actually applied to the audio. This is
            what lets a live quieting readout exist for a run that never
            passes ``--squelch``: :meth:`NoiseSquelch.update` still runs
            every block regardless of this flag.

    Returns:
        Recovered audio as float32, at ``config.audio_rate_hz``, clipped
        to :data:`AUDIO_CLIP_RANGE` — or an equal-length run of zeros if
        ``squelch`` is given, closed for this block, and ``mute`` is
        ``True``.

    Raises:
        ValueError: If ``iq`` is not one-dimensional, or is shorter than
            ``MIN_IF_SAMPLES * config.channel_decimation_factor`` — the
            floor the channel filter needs, which is set by scipy's
            zero-phase padding rather than by the discriminator. See
            :data:`MIN_IF_SAMPLES`.

    The squelch is measured on the discriminator's output *before*
    de-emphasis, and applied to the finished audio *after* decimation.
    Both halves of that matter and neither is arbitrary — the module
    docstring of :mod:`qsorbit.core.dsp.squelch` has the reasoning.
    """
    # Checked before the channel filter rather than after, so a caller
    # gets this message instead of scipy's "input vector x must be greater
    # than padlen" from inside decimate(). Found by a test that asserted
    # on our own wording and got scipy's.
    minimum = MIN_IF_SAMPLES * config.channel_decimation_factor
    if iq.ndim != 1 or iq.shape[0] < minimum:
        got = iq.shape[0] if iq.ndim == 1 else iq.shape
        raise ValueError(
            f"iq is too short: got {got} sample(s), and the channel filter decimates by "
            f"{config.channel_decimation_factor:,}. Supply at least {minimum:,} -- see "
            f"MIN_IF_SAMPLES for why the floor is not simply two."
        )

    baseband = (
        shift_to_baseband(iq, config.channel_offset_hz, config.sample_rate_hz)
        if config.channel_offset_hz != 0.0
        else iq.astype(np.complex64)
    )

    # The channel filter. decimate()'s anti-aliasing low-pass is doing
    # adjacent-channel rejection here, not just protecting the resample.
    channel = decimate(baseband, config.channel_decimation_factor)
    discriminated = discriminate(channel, config.if_rate_hz, config.deviation_hz)

    if squelch is not None:
        squelch.update(discriminated, config.if_rate_hz)

    audio = discriminated
    if config.de_emphasis_us is not None:
        audio = _apply_deemphasis(audio, config.if_rate_hz, config.de_emphasis_us)

    decimated = decimate(audio, config.audio_decimation_factor)
    result = np.clip(decimated, *AUDIO_CLIP_RANGE).astype(np.float32)

    if squelch is None:
        return result
    # apply() always runs, muting decision and all: its bookkeeping
    # (blocks_open, samples_passed/muted) is what makes "how much WOULD
    # this have muted" a real, measured answer rather than a guess, for
    # a run where mute=False means the gate's decision is deliberately
    # never allowed to reach the speaker.
    gated = squelch.apply(result)
    return gated if mute else result


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
