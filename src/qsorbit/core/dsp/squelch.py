"""Noise squelch: mute an FM demodulator when no carrier is present.

An FM discriminator handed no signal does not output silence — it outputs
**full-scale noise**. The phase advance between consecutive samples of pure
noise is random, and random phase advance is white noise at full amplitude.
That is why every FM radio ever built has a squelch, and why narrowband FM
needs one in a way wideband broadcast never did: a broadcast carrier is
always there, and a repeater or a satellite downlink is not.

**Why noise power rather than signal power.** The obvious squelch measures
channel power and mutes below a threshold in dB. It is rejected here for a
reason this project already paid for at the bench: Session 14 established
that *ADC level does not indicate signal presence* — an empty capture
measured noisier than a good one — and that this rig's tuner gain is a
deliberate manual setting that changes per band and per antenna. Any
absolute dB threshold on the RF would need recalibrating every time gain
moved, and would be silently wrong in between.

**Why the measurement is an absolute level anyway, and why that is not a
contradiction.** The natural next thought is to make the noise measurement
a *ratio* — noise-band power over total power — so that it, too, is
gain-independent. That was in fact the first implementation here, and real
data killed it. The reason is that
:func:`~qsorbit.core.dsp.demod.discriminate` computes ``np.angle()``, which
**discards magnitude entirely**: its output is a phase advance per sample,
normalised by peak deviation, and it does not care how strong the RF that
produced it was. The discriminator has already solved the gain problem.
Dividing by total power afterwards does not add robustness; it removes
information, because on an empty channel the total is enormous for exactly
the same reason the noise band is. Measured on this project's own NOAA
captures, the ratio form rated a genuine strong signal at 1.50 dB against
1.54 dB for an empty adjacent channel — it would have muted the signal and
opened on the noise. This is Session 14's rule biting a second time: a
verifier tested only on synthetic data will be wrong about real data in
ways the synthetic case cannot show. Every synthetic test passed with a
margin of 100 dB while the metric was, on real signals, worse than useless.

So the measurement is what an analog radio's noise squelch has always
measured: **the absolute level of the discriminator's output above the
occupied channel**, reported as "quieting" — dB below full deviation, so
larger means a quieter noise band and therefore a stronger signal. It is
the same quantity amateur radio means by "20 dB quieting" in a receiver
spec. See :func:`quieting_db` for the numbers this project measured.

**Measure before de-emphasis, always.** :meth:`NoiseSquelch.update` must be
given the raw discriminator output, not the finished audio. NBFM
de-emphasis (750 microseconds, a corner near 212 Hz) attenuates 4 kHz by
roughly 25 dB and 16 kHz by roughly 37 dB, so *after* it even pure noise
has most of its remaining power down in the voice band and measures as
heavily quieted. The metric would read "strong signal" on an empty channel.
Analog radios tap the noise detector off the discriminator ahead of the
audio filtering for exactly this reason, and
:func:`~qsorbit.core.dsp.demod.demodulate_nbfm` wires it the same way.

**It defaults to off, and that is a deliberate safety property rather than
an oversight.** A squelch is a mute, and a mute set slightly too tight
makes a *correctly working receiver* produce silence — which is
indistinguishable from a broken one. This project has a standing catalogue
of silent failure modes (the V4's mistuning, the ctypes marshalling bug
that cost 9.5% of every capture, ``sounddevice`` raising at import, a Qt
event queue degrading without complaint) and a squelch enabled by default
would be the next entry: tuned against a strong terrestrial repeater, then
pointed at a weak downlink, it would mute the very signal the pass exists
to find and report nothing. Callers opt in.

**The thresholds are calibrated against live bench runs, and the tool to
re-calibrate them ships with them.** :data:`DEFAULT_OPEN_ABOVE_DB` and
:data:`DEFAULT_CLOSE_BELOW_DB` come from measuring a real signal and a real
empty channel minutes apart on the same antenna — see :func:`quieting_db`,
which also records what happened when they were set from recordings
instead. They remain tied to the
:class:`~qsorbit.core.dsp.demod.NbfmConfig` they were measured at, because
the metric's scale depends on the IF rate and the deviation normalisation,
so a materially different config wants a fresh look.
:attr:`SquelchStats.min_quieting_db` and
:attr:`SquelchStats.max_quieting_db` record the range actually observed
during a run, which is what makes that fresh look a measurement rather than
another guess: point the receiver at an empty channel for thirty seconds
and the floor reads straight off ``max_quieting_db``.

**Known limitation, deliberately not fixed here**: the gate is hard — a
transition mutes or unmutes at a block boundary with no fade, which can
click. A few milliseconds of ramp would fix it and belongs with the other
measured-but-parked audio items in Chunk I; it does not affect whether the
squelch decides correctly, which is what this module is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

#: Where the "noise" band starts, in Hz. It must sit **above the occupied
#: channel**, not merely above the voice band: an FM channel with 5 kHz
#: deviation occupies roughly +/-8 kHz by Carson's rule, so a band starting
#: at 4 kHz would be measuring the signal's own sidebands and calling them
#: noise. Measured on the NOAA fixtures, moving this floor from 4 kHz to
#: 8 kHz is worth several dB of separation on its own. The upper end is
#: whatever Nyquist the discriminator runs at — 16 kHz at the default IF
#: rate, which leaves a clean 8 kHz-wide window between the channel edge
#: and the channel filter's own corner near 12.8 kHz.
DEFAULT_NOISE_BAND_LOW_HZ: Final = 8_000.0

#: Quieting, in dB, at or above which the gate opens. Calibrated against
#: **live** bench runs, not against recordings — see :func:`quieting_db`
#: for the numbers and for why the first attempt at this constant was
#: 6.0 dB and wrong. Sits about 3.7 dB above the noisiest empty-channel
#: block measured live, and about 3.4 dB below the quietest genuine signal
#: block, so it has real margin in both directions.
DEFAULT_OPEN_ABOVE_DB: Final = 3.0

#: Quieting, in dB, at or below which the gate closes. Lower than
#: :data:`DEFAULT_OPEN_ABOVE_DB` on purpose — the gap between them is the
#: hysteresis. A single threshold makes a signal sitting right at it
#: chatter the audio on and off several times a second, which sounds
#: materially worse than either state; the same reasoning that puts a
#: movement deadband in front of the rotor. Still about 2.2 dB clear of the
#: noisiest empty block measured live, so noise alone cannot hold the gate
#: open — and erring toward *staying open* is the deliberate direction,
#: because hiss is audible and diagnosable while a wrong mute is neither.
DEFAULT_CLOSE_BELOW_DB: Final = 1.5

#: Reported quieting is capped here. A perfectly noiseless discriminator
#: output — digital silence, or a mathematically pure synthetic tone —
#: would otherwise report a meaningless number in the hundreds of dB, set
#: by float rounding dust rather than by anything physical. Real signals
#: measured on real hardware land far below this.
MAX_QUIETING_DB: Final = 60.0

#: Mean-square power below this counts as zero, so a perfectly silent block
#: produces :data:`MAX_QUIETING_DB` rather than a ``log10(0)``. Same role as
#: :data:`~qsorbit.core.dsp.spectrum.DEFAULT_FLOOR_DB`.
POWER_FLOOR: Final = 1e-30


@dataclass(frozen=True)
class SquelchStats:
    """What one squelch run did, and what it saw while doing it.

    Args:
        blocks_evaluated: Blocks passed to :meth:`NoiseSquelch.update`.
        blocks_open: Of those, how many left the gate open.
        blocks_muted: Of those, how many left it closed. Always
            ``blocks_evaluated - blocks_open``; carried explicitly because
            a reader should not have to do the subtraction to answer "how
            much did this thing mute?"
        samples_passed: Audio samples :meth:`NoiseSquelch.apply` let
            through.
        samples_muted: Audio samples it replaced with silence. **This is
            the number that distinguishes "muted by squelch" from "chain
            broken"**, which is the whole reason the counting exists —
            silence with a large ``samples_muted`` is the squelch working,
            and silence with ``samples_muted`` at zero is a fault
            somewhere upstream.
        last_quieting_db: The most recent measurement, or ``None`` if
            nothing has been evaluated yet.
        min_quieting_db: Lowest quieting seen this run, or ``None``.
            Roughly "how quiet did the worst noise-only block look" — a
            sensible floor to set :attr:`NoiseSquelch.close_below_db` just
            above.
        max_quieting_db: Highest quieting seen this run, or ``None``.
            Roughly "how good did the signal get" — a ceiling to set
            :attr:`NoiseSquelch.open_above_db` below.
        open_above_db: The thresholds in force, echoed so a run's report
            is self-contained: the numbers mean nothing without them.
        close_below_db: See ``open_above_db``.
    """

    blocks_evaluated: int
    blocks_open: int
    blocks_muted: int
    samples_passed: int
    samples_muted: int
    last_quieting_db: float | None
    min_quieting_db: float | None
    max_quieting_db: float | None
    open_above_db: float
    close_below_db: float

    @property
    def open_fraction(self) -> float:
        """Fraction of evaluated **blocks** that left the gate open, 0.0-1.0.

        Deliberately counted in blocks rather than samples, and **not the
        same figure** as the audio percentage :meth:`describe` prints —
        the two answer different questions and will not generally agree.
        This one says how often the gate was open, which is what matters
        when judging whether a threshold is set sensibly; the audio
        percentage says how much was actually heard. Both are reported,
        and both name their unit wherever they appear, precisely so the
        two never get quoted as though they were one number.

        Returns ``0.0`` rather than raising when nothing has been
        evaluated — a run that never started did not have the squelch
        open.
        """
        if self.blocks_evaluated == 0:
            return 0.0
        return self.blocks_open / self.blocks_evaluated

    def describe(self) -> str:
        """Return a short human-readable summary, for logs and reports.

        Worded rather than tabulated, for the reason
        :meth:`~qsorbit.core.dsp.spectrum_stream.SpectrumStreamStats.describe`
        gives: attribute names get read once and output gets read every
        time. Every percentage here names the thing it is a percentage of
        — see :attr:`open_fraction` for why that is not fussiness.
        """
        total_samples = self.samples_passed + self.samples_muted
        passed_pct = 100.0 * self.samples_passed / total_samples if total_samples else 0.0
        observed = (
            "  quieting:   never measured"
            if self.last_quieting_db is None
            else (
                f"  quieting:   {self.min_quieting_db:.1f} dB min, "
                f"{self.max_quieting_db:.1f} dB max, {self.last_quieting_db:.1f} dB last"
            )
        )
        return (
            f"squelch: {self.blocks_open:,} of {self.blocks_evaluated:,} block(s) open "
            f"({100.0 * self.open_fraction:.1f}% of blocks)\n"
            f"  audio:      {self.samples_passed:,} sample(s) passed, "
            f"{self.samples_muted:,} muted ({passed_pct:.1f}% of audio passed)\n"
            f"{observed}\n"
            f"  thresholds: open at/above {self.open_above_db:.1f} dB, "
            f"close at/below {self.close_below_db:.1f} dB"
        )


def quieting_db(
    discriminated: np.ndarray,
    sample_rate_hz: float,
    *,
    noise_band_low_hz: float = DEFAULT_NOISE_BAND_LOW_HZ,
) -> float:
    """Measure how far ``discriminated``'s out-of-channel noise sits below full deviation.

    Computed in the frequency domain rather than with a high-pass filter,
    deliberately: a filter would carry state across blocks and put a
    settling transient at the start of each one, while a single real FFT
    per block is stateless, exact, and trivially testable. The cost is one
    ``rfft`` of a block that is already only a few thousand samples.

    Args:
        discriminated: The **raw discriminator output**, before
            de-emphasis. See the module docstring for why measuring after
            de-emphasis reads "strong signal" on an empty channel.
        sample_rate_hz: The rate ``discriminated`` is sampled at — the IF
            rate, not the eventual audio rate.
        noise_band_low_hz: Frequencies at or above this count as noise.

    Returns:
        Quieting in dB — ``-10*log10(mean square of the noise band)``,
        i.e. how far the out-of-channel noise sits *below* full deviation.
        Larger means a quieter noise band and therefore a stronger signal,
        matching what "20 dB quieting" means in a receiver spec. Capped at
        :data:`MAX_QUIETING_DB`.

        **The numbers this project measured**, at the default IF rate and
        noise band. Recorded fixtures first, then two live 30-second bench
        runs on the same antenna minutes apart:

        ======================  ===============  ==================
        Channel                 Fixture (median)  Live (min - max)
        ======================  ===============  ==================
        162.550 (strong)        17.4 dB           6.4 - 10.6 dB
        162.475 (strong)        10.7 dB           not run live
        162.400 (weak)           0.6 dB           not run live
        162.425 (empty)         -1.0 dB           not run live
        162.450 (empty)         -1.0 dB          -2.0 - -0.7 dB
        162.525 (empty)         -1.0 dB           not run live
        ======================  ===============  ==================

        **Read the two columns against each other — that comparison is
        the most useful thing in this docstring.** The *empty* rows agree
        between recording and live to within about half a dB. The
        *signal* row disagrees by seven to eleven dB, on the same station,
        same gain, same dongle. Thresholds set from the signal end of a
        recording are therefore worthless, and the first version of these
        defaults (6.0/3.0) was set exactly that way: it left a live margin
        of 0.4 dB on a strong local transmitter.

        The asymmetry is not bad luck, it is structural, and it tells you
        which end to calibrate from. **The empty floor is arithmetic, not
        an observation.** With no carrier the phase advance per sample is
        uniform over +/-pi, which after normalising by ``deviation_hz``
        spans ``+/-(sample_rate_hz/2)/deviation_hz`` — so the floor is
        predictable from the config alone::

            floor_db = -10*log10( (1/3) * ((sample_rate_hz/2)/deviation_hz)**2
                                  * (nyquist - noise_band_low) / nyquist )

        which gives -2.3 dB for the defaults here, against -2.0 to -0.7
        measured live. Signal level, by contrast, is a property of the
        antenna, the path and the day. So place thresholds relative to the
        floor with margin, and treat any signal-side figure as an upper
        sanity check only. It also follows that **the floor moves if
        ``if_rate_hz``, ``deviation_hz`` or the noise band moves**, which
        is why the thresholds are documented as tied to the config they
        were measured at.

        A perfectly silent block returns :data:`MAX_QUIETING_DB`, because a
        noiseless discriminator output genuinely *is* maximally quiet.
        That is deliberate rather than a gap: detecting a dead device is
        :class:`~qsorbit.core.sdr.stream.StreamStats`'s job on the receive
        side, and a second detector here could only disagree with the
        first.

    Raises:
        ValueError: If ``discriminated`` is not one-dimensional and
            non-empty, if ``sample_rate_hz`` is not a positive finite
            number, or if ``noise_band_low_hz`` is negative, non-finite,
            or at/above the Nyquist frequency (which would leave no bins
            in the noise band to measure).
    """
    if discriminated.ndim != 1 or discriminated.shape[0] == 0:
        raise ValueError(
            f"discriminated must be a non-empty one-dimensional array, "
            f"got shape {discriminated.shape!r}."
        )
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError(
            f"sample_rate_hz must be a positive, finite number, got {sample_rate_hz!r}."
        )
    nyquist_hz = sample_rate_hz / 2.0
    if not math.isfinite(noise_band_low_hz) or noise_band_low_hz < 0.0:
        raise ValueError(
            f"noise_band_low_hz must be a non-negative, finite number, got {noise_band_low_hz!r}."
        )
    if noise_band_low_hz >= nyquist_hz:
        raise ValueError(
            f"noise_band_low_hz ({noise_band_low_hz!r}) must be below the Nyquist frequency "
            f"({nyquist_hz!r}) or there are no noise bins to measure. Either lower it or "
            f"raise the sample rate the discriminator runs at."
        )

    n = discriminated.shape[0]
    power = np.abs(np.fft.rfft(discriminated)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)

    # One-sided bins carry half the energy each, and rfft is unnormalised,
    # so the factor of two and the 1/n**2 together turn a bin sum back into
    # a mean square in the input's own units -- which, downstream of
    # discriminate(), are fractions of peak deviation.
    noise_mean_square = 2.0 * float(power[freqs >= noise_band_low_hz].sum()) / (n * n)
    if noise_mean_square <= POWER_FLOOR:
        return MAX_QUIETING_DB
    return min(-10.0 * math.log10(noise_mean_square), MAX_QUIETING_DB)


class NoiseSquelch:
    """A hysteretic noise-power gate for an FM demodulator.

    Stateful on purpose, and the state is the point: hysteresis is a latch
    that has to survive from one block to the next, so this cannot be a
    pure function the way the rest of :mod:`qsorbit.core.dsp`'s
    demodulation is. It is split into two small methods rather than one so
    that the measurement (which needs the discriminator's output, at the IF
    rate, before de-emphasis) and the gating (which needs the finished
    audio, at the audio rate) each get the signal they actually want.

    Usage is normally indirect — pass one to
    :func:`~qsorbit.core.dsp.demod.demodulate_nbfm`, which calls both
    methods at the right points::

        squelch = NoiseSquelch()
        for block in blocks:
            audio = demodulate_nbfm(block, config, squelch=squelch)
        print(squelch.stats.describe())

    Args:
        open_above_db: Quieting at or above which the gate opens.
        close_below_db: Quieting at or below which it closes. Must not
            exceed ``open_above_db``; the gap between them is the
            hysteresis, and equal values mean a bare threshold with none.
        noise_band_low_hz: Where the noise band starts. See
            :data:`DEFAULT_NOISE_BAND_LOW_HZ`.

    Raises:
        ValueError: If either threshold is not finite, if
            ``close_below_db`` exceeds ``open_above_db``, or if
            ``noise_band_low_hz`` is negative or not finite.
    """

    def __init__(
        self,
        *,
        open_above_db: float = DEFAULT_OPEN_ABOVE_DB,
        close_below_db: float = DEFAULT_CLOSE_BELOW_DB,
        noise_band_low_hz: float = DEFAULT_NOISE_BAND_LOW_HZ,
    ) -> None:
        if not math.isfinite(open_above_db):
            raise ValueError(f"open_above_db must be finite, got {open_above_db!r}.")
        if not math.isfinite(close_below_db):
            raise ValueError(f"close_below_db must be finite, got {close_below_db!r}.")
        if close_below_db > open_above_db:
            raise ValueError(
                f"close_below_db ({close_below_db!r}) must not exceed open_above_db "
                f"({open_above_db!r}) -- the gate would close at a stronger signal than "
                f"it opens at, and never settle."
            )
        if not math.isfinite(noise_band_low_hz) or noise_band_low_hz < 0.0:
            raise ValueError(
                f"noise_band_low_hz must be a non-negative, finite number, "
                f"got {noise_band_low_hz!r}."
            )

        self._open_above_db = open_above_db
        self._close_below_db = close_below_db
        self._noise_band_low_hz = noise_band_low_hz

        # Starts closed: a squelch that begins open would pass one block
        # of full-scale hiss before it had measured anything.
        self._is_open = False
        self._blocks_evaluated = 0
        self._blocks_open = 0
        self._samples_passed = 0
        self._samples_muted = 0
        self._last_quieting_db: float | None = None
        self._min_quieting_db: float | None = None
        self._max_quieting_db: float | None = None

    @property
    def is_open(self) -> bool:
        """``True`` if the gate is currently passing audio."""
        return self._is_open

    @property
    def open_above_db(self) -> float:
        """Quieting at or above which the gate opens."""
        return self._open_above_db

    @property
    def close_below_db(self) -> float:
        """Quieting at or below which the gate closes."""
        return self._close_below_db

    @property
    def noise_band_low_hz(self) -> float:
        """Where the noise band starts, in Hz."""
        return self._noise_band_low_hz

    def update(self, discriminated: np.ndarray, sample_rate_hz: float) -> bool:
        """Measure one block and update the gate, returning its new state.

        Args:
            discriminated: Raw discriminator output for this block,
                **before de-emphasis** — see the module docstring.
            sample_rate_hz: The rate ``discriminated`` is sampled at.

        Returns:
            ``True`` if the gate is now open. Between the two thresholds
            the previous state is held, which is what makes this
            hysteretic rather than a bare comparison.
        """
        measured = quieting_db(
            discriminated, sample_rate_hz, noise_band_low_hz=self._noise_band_low_hz
        )

        if measured >= self._open_above_db:
            self._is_open = True
        elif measured <= self._close_below_db:
            self._is_open = False
        # Between the thresholds: hold. This branch is the hysteresis.

        self._blocks_evaluated += 1
        if self._is_open:
            self._blocks_open += 1
        self._last_quieting_db = measured
        self._min_quieting_db = (
            measured if self._min_quieting_db is None else min(self._min_quieting_db, measured)
        )
        self._max_quieting_db = (
            measured if self._max_quieting_db is None else max(self._max_quieting_db, measured)
        )
        return self._is_open

    def apply(self, audio: np.ndarray) -> np.ndarray:
        """Pass ``audio`` through, or replace it with silence if the gate is closed.

        Args:
            audio: Finished audio for the block most recently passed to
                :meth:`update`.

        Returns:
            ``audio`` unchanged when the gate is open, or an array of
            zeros of the same shape and dtype when it is closed.
        """
        if self._is_open:
            self._samples_passed += audio.shape[0]
            return audio
        self._samples_muted += audio.shape[0]
        return np.zeros_like(audio)

    @property
    def stats(self) -> SquelchStats:
        """The run's statistics so far."""
        return SquelchStats(
            blocks_evaluated=self._blocks_evaluated,
            blocks_open=self._blocks_open,
            blocks_muted=self._blocks_evaluated - self._blocks_open,
            samples_passed=self._samples_passed,
            samples_muted=self._samples_muted,
            last_quieting_db=self._last_quieting_db,
            min_quieting_db=self._min_quieting_db,
            max_quieting_db=self._max_quieting_db,
            open_above_db=self._open_above_db,
            close_below_db=self._close_below_db,
        )
