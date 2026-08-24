"""Tests for the noise squelch: the quieting measurement, the hysteretic
gate built on it, and the accounting that tells "muted" from "broken".
"""

from __future__ import annotations

import numpy as np
import pytest

from qsorbit.core.dsp import (
    DEFAULT_NOISE_BAND_LOW_HZ,
    MAX_QUIETING_DB,
    NoiseSquelch,
    SquelchStats,
    quieting_db,
)

IF_RATE_HZ = 32_000.0
DEVIATION_HZ = 5_000.0
BLOCK = 2_048

#: Peak magnitude of a discriminator's output with no carrier present, in
#: the deviation-normalised units discriminate() emits. Phase advance per
#: sample is uniform over +/-pi when the phase is random, and pi radians
#: per sample IS half the sample rate, so the peak is
#: (IF_RATE_HZ / 2) / DEVIATION_HZ -- here 3.2, not 1.0.
FULL_SCALE = (IF_RATE_HZ / 2.0) / DEVIATION_HZ


def noise_block(n: int = BLOCK, *, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """What a discriminator actually emits with no carrier present.

    Uniform over +/-:data:`FULL_SCALE` rather than the unit-variance
    Gaussian this helper used first. That original was **5.6 dB too
    quiet**, and it mattered: it made synthetic "empty channel" blocks
    measure +2.9 dB where a real empty channel measures about -1.4 dB, so
    every threshold sanity-checked against it looked healthier than it
    was. The bench found the discrepancy, not the suite -- the same shape
    as the ratio-versus-absolute mistake this module's docstring records.

    Uniform, not Gaussian, because random *phase* is uniform: a
    discriminator with nothing to lock onto reports a phase advance drawn
    flat across +/-pi, and there is no central tendency to it.
    """
    rng = np.random.default_rng(seed)
    return (rng.uniform(-FULL_SCALE, FULL_SCALE, n) * scale).astype(np.float32)


def tone_block(
    tone_hz: float = 1_000.0, n: int = BLOCK, *, rate_hz: float = IF_RATE_HZ, amplitude: float = 0.5
) -> np.ndarray:
    """A clean in-voice-band tone: what a discriminator emits on a strong signal."""
    t = np.arange(n) / rate_hz
    return (amplitude * np.sin(2 * np.pi * tone_hz * t)).astype(np.float32)


class TestQuietingMeasurement:
    """The metric is an ABSOLUTE level, not a ratio, and that is the whole
    point — see the module docstring. ``discriminate()`` computes
    ``np.angle()``, which discards magnitude, so its output is already
    gain-independent; dividing by total power afterwards removes
    information instead of adding robustness. The ratio form rated a real
    NOAA signal at 1.50 dB against 1.54 dB for an empty adjacent channel.
    """

    def test_full_scale_noise_measures_at_the_predicted_floor(self):
        # The floor is arithmetic rather than an observation -- see
        # quieting_db's docstring for the derivation. Asserting against
        # the predicted value rather than a loose bound is what would have
        # caught this helper being 5.6 dB optimistic in the first place.
        nyquist_hz = IF_RATE_HZ / 2.0
        predicted_db = -10.0 * np.log10(
            (FULL_SCALE**2 / 3.0) * (nyquist_hz - DEFAULT_NOISE_BAND_LOW_HZ) / nyquist_hz
        )

        measured = quieting_db(noise_block(), IF_RATE_HZ)

        assert measured == pytest.approx(predicted_db, abs=1.0)

    def test_the_synthetic_floor_matches_what_the_bench_measured(self):
        # Live 30 s run on an empty NOAA channel (162.450) read -2.0 to
        # -0.7 dB. Synthetic noise must land in that neighbourhood or the
        # suite is calibrating against a fiction, which is exactly what
        # happened before.
        measured = quieting_db(noise_block(), IF_RATE_HZ)

        assert -4.0 < measured < 0.0

    def test_a_clean_voice_band_signal_measures_strongly_quieted(self):
        measured = quieting_db(tone_block(), IF_RATE_HZ)

        assert measured > 20.0

    def test_a_signal_measures_higher_than_noise(self):
        assert quieting_db(tone_block(), IF_RATE_HZ) > quieting_db(noise_block(), IF_RATE_HZ)

    def test_scaling_the_discriminator_noise_up_lowers_the_quieting(self):
        # An absolute measure MUST respond to the noise level -- this is
        # the property the ratio form threw away.
        soft = quieting_db(noise_block(scale=0.01), IF_RATE_HZ)
        blaring = quieting_db(noise_block(scale=10.0), IF_RATE_HZ)

        assert soft > blaring

    def test_a_ratio_of_noise_to_total_would_not_separate_them(self):
        # The negative-space proof that the absolute form is doing real
        # work. Computed here rather than called, because the ratio metric
        # no longer exists in the module -- this test is what stops it
        # coming back on the "but a ratio is gain-independent" argument.
        def ratio_db(block):
            spectrum = np.abs(np.fft.rfft(block)) ** 2
            freqs = np.fft.rfftfreq(len(block), d=1.0 / IF_RATE_HZ)
            noise = spectrum[freqs >= DEFAULT_NOISE_BAND_LOW_HZ].sum()
            return 10.0 * np.log10(spectrum.sum() / max(noise, 1e-30))

        quiet_noise = noise_block(scale=0.01)
        loud_noise = noise_block(scale=10.0)

        # The ratio calls these identical; the shipped metric does not.
        assert ratio_db(quiet_noise) == pytest.approx(ratio_db(loud_noise), abs=0.5)
        assert quieting_db(quiet_noise, IF_RATE_HZ) - quieting_db(loud_noise, IF_RATE_HZ) > 40.0

    def test_digital_silence_reports_the_ceiling_rather_than_dividing_by_zero(self):
        # Correct rather than a gap: a noiseless discriminator output
        # genuinely is maximally quiet. Detecting a dead device is the SDR
        # layer's job, and a second detector here could only disagree.
        measured = quieting_db(np.zeros(BLOCK, dtype=np.float32), IF_RATE_HZ)

        assert measured == MAX_QUIETING_DB

    def test_a_mathematically_pure_tone_is_capped_at_the_ceiling(self):
        # Without the cap this reports a meaningless number in the
        # hundreds of dB, set by float32 rounding dust.
        assert quieting_db(tone_block(), IF_RATE_HZ) <= MAX_QUIETING_DB

    def test_noise_inside_the_channel_does_not_count_against_quieting(self):
        # The noise band starts above the occupied channel, so a 2 kHz
        # component -- squarely inside a 5 kHz-deviation channel -- is
        # signal, not hiss. Getting this floor wrong by measuring from
        # 4 kHz costs several dB of separation on real captures.
        in_channel = tone_block(tone_hz=2_000.0, amplitude=1.0)
        out_of_channel = tone_block(tone_hz=12_000.0, amplitude=1.0)

        assert quieting_db(in_channel, IF_RATE_HZ) > quieting_db(out_of_channel, IF_RATE_HZ)

    def test_a_higher_noise_band_floor_exempts_more_of_the_spectrum(self):
        signal = tone_block(tone_hz=9_000.0, amplitude=1.0)

        default_floor = quieting_db(signal, IF_RATE_HZ)
        raised_floor = quieting_db(signal, IF_RATE_HZ, noise_band_low_hz=10_000.0)

        assert raised_floor > default_floor

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_an_impossible_sample_rate(self, bad):
        with pytest.raises(ValueError, match="sample_rate_hz"):
            quieting_db(noise_block(), bad)

    def test_rejects_an_empty_block(self):
        with pytest.raises(ValueError, match="non-empty"):
            quieting_db(np.zeros(0, dtype=np.float32), IF_RATE_HZ)

    def test_rejects_a_noise_band_at_or_above_nyquist(self):
        with pytest.raises(ValueError, match="Nyquist"):
            quieting_db(noise_block(), IF_RATE_HZ, noise_band_low_hz=IF_RATE_HZ)

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_rejects_an_impossible_noise_band_floor(self, bad):
        with pytest.raises(ValueError, match="noise_band_low_hz"):
            quieting_db(noise_block(), IF_RATE_HZ, noise_band_low_hz=bad)


class TestNoiseSquelchConstruction:
    def test_defaults_are_sane_and_readable(self):
        squelch = NoiseSquelch()

        assert squelch.open_above_db > squelch.close_below_db
        assert squelch.noise_band_low_hz == DEFAULT_NOISE_BAND_LOW_HZ

    def test_starts_closed(self):
        # A squelch that began open would pass one block of full-scale
        # hiss before it had measured anything.
        assert NoiseSquelch().is_open is False

    def test_rejects_thresholds_that_could_never_settle(self):
        with pytest.raises(ValueError, match="close_below_db"):
            NoiseSquelch(open_above_db=5.0, close_below_db=9.0)

    def test_equal_thresholds_are_allowed_as_a_bare_threshold(self):
        squelch = NoiseSquelch(open_above_db=6.0, close_below_db=6.0)

        assert squelch.open_above_db == squelch.close_below_db

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_rejects_non_finite_thresholds(self, bad):
        with pytest.raises(ValueError, match="open_above_db"):
            NoiseSquelch(open_above_db=bad)


class TestNoiseSquelchGating:
    def test_a_strong_signal_opens_the_gate(self):
        squelch = NoiseSquelch()

        assert squelch.update(tone_block(), IF_RATE_HZ) is True
        assert squelch.is_open is True

    def test_noise_leaves_the_gate_closed(self):
        squelch = NoiseSquelch()

        assert squelch.update(noise_block(), IF_RATE_HZ) is False

    def test_an_open_gate_passes_audio_untouched(self):
        squelch = NoiseSquelch()
        squelch.update(tone_block(), IF_RATE_HZ)
        audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)

        assert np.array_equal(squelch.apply(audio), audio)

    def test_a_closed_gate_substitutes_silence_of_the_same_shape(self):
        squelch = NoiseSquelch()
        squelch.update(noise_block(), IF_RATE_HZ)
        audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)

        muted = squelch.apply(audio)

        assert muted.shape == audio.shape
        assert muted.dtype == audio.dtype
        assert not muted.any()

    def test_the_gate_closes_again_when_the_signal_goes_away(self):
        squelch = NoiseSquelch()
        squelch.update(tone_block(), IF_RATE_HZ)
        assert squelch.is_open is True

        squelch.update(noise_block(), IF_RATE_HZ)

        assert squelch.is_open is False


class TestHysteresis:
    """The gap between the thresholds is the whole point: a signal sitting
    between them must hold whatever state it was already in, rather than
    chattering the audio on and off several times a second.
    """

    def test_a_reading_between_the_thresholds_holds_an_open_gate_open(self):
        squelch = NoiseSquelch(open_above_db=8.0, close_below_db=5.0)
        squelch.update(tone_block(), IF_RATE_HZ)  # strongly quieted: opens
        assert squelch.is_open is True

        # Craft a block that measures between the two thresholds.
        marginal = _block_measuring_between(6.0, 8.0, 5.0)
        squelch.update(marginal, IF_RATE_HZ)

        assert squelch.is_open is True

    def test_the_same_reading_holds_a_closed_gate_closed(self):
        # Same input, opposite starting state, opposite outcome -- which is
        # what makes this hysteresis rather than a threshold with a gap.
        squelch = NoiseSquelch(open_above_db=8.0, close_below_db=5.0)
        squelch.update(noise_block(), IF_RATE_HZ)
        assert squelch.is_open is False

        marginal = _block_measuring_between(6.0, 8.0, 5.0)
        squelch.update(marginal, IF_RATE_HZ)

        assert squelch.is_open is False

    def test_without_hysteresis_the_same_sequence_would_chatter(self):
        # The negative-space check: with the gap removed, a marginal signal
        # flips state where the hysteretic gate holds it. Guards against
        # someone later "simplifying" the two thresholds into one.
        bare = NoiseSquelch(open_above_db=6.5, close_below_db=6.5)
        bare.update(noise_block(), IF_RATE_HZ)
        assert bare.is_open is False

        marginal = _block_measuring_between(7.0, 8.0, 5.0)
        bare.update(marginal, IF_RATE_HZ)

        assert bare.is_open is True


class TestSquelchAccounting:
    def test_a_fresh_squelch_reports_nothing_measured(self):
        stats = NoiseSquelch().stats

        assert stats.blocks_evaluated == 0
        assert stats.last_quieting_db is None
        assert stats.open_fraction == 0.0
        assert "never measured" in stats.describe()

    def test_open_and_muted_blocks_are_counted_separately(self):
        squelch = NoiseSquelch()
        audio = np.zeros(100, dtype=np.float32)

        squelch.update(tone_block(), IF_RATE_HZ)
        squelch.apply(audio)
        squelch.update(noise_block(), IF_RATE_HZ)
        squelch.apply(audio)

        stats = squelch.stats
        assert stats.blocks_evaluated == 2
        assert stats.blocks_open == 1
        assert stats.blocks_muted == 1
        assert stats.samples_passed == 100
        assert stats.samples_muted == 100
        assert stats.open_fraction == pytest.approx(0.5)

    def test_muted_samples_are_what_distinguishes_muted_from_broken(self):
        # The reason this counter exists at all: silence with a large
        # samples_muted is the squelch working; silence with it at zero is
        # a fault somewhere upstream. A caller must be able to tell.
        squelch = NoiseSquelch()
        squelch.update(noise_block(), IF_RATE_HZ)
        squelch.apply(np.zeros(512, dtype=np.float32))

        assert squelch.stats.samples_muted == 512
        assert squelch.stats.samples_passed == 0

    def test_the_observed_quieting_range_is_recorded_for_bench_tuning(self):
        # The thresholds ship uncalibrated, so the run has to report the
        # range it actually saw or there is nothing to calibrate against.
        squelch = NoiseSquelch()
        squelch.update(noise_block(), IF_RATE_HZ)
        squelch.update(tone_block(), IF_RATE_HZ)
        squelch.update(noise_block(seed=1), IF_RATE_HZ)

        stats = squelch.stats
        assert stats.min_quieting_db < stats.max_quieting_db
        assert stats.last_quieting_db == pytest.approx(quieting_db(noise_block(seed=1), IF_RATE_HZ))
        assert stats.min_quieting_db <= stats.last_quieting_db <= stats.max_quieting_db

    def test_describe_words_the_numbers_and_echoes_the_thresholds(self):
        squelch = NoiseSquelch(open_above_db=8.0, close_below_db=5.0)
        squelch.update(tone_block(), IF_RATE_HZ)
        squelch.apply(np.zeros(64, dtype=np.float32))

        text = squelch.stats.describe()

        assert "squelch" in text
        assert "muted" in text
        # The thresholds must travel with the numbers: the readings mean
        # nothing without knowing what they were compared against.
        assert "8.0 dB" in text
        assert "5.0 dB" in text

    def test_every_percentage_names_what_it_is_a_percentage_of(self):
        # There are two denominators here -- blocks and samples -- and
        # they do not generally agree. An unlabelled percentage is how one
        # gets quoted as the other in a report later.
        squelch = NoiseSquelch()
        squelch.update(tone_block(), IF_RATE_HZ)
        squelch.apply(np.zeros(64, dtype=np.float32))

        text = squelch.stats.describe()

        assert "% of blocks" in text
        assert "% of audio" in text

    def test_the_two_percentages_are_genuinely_different_numbers(self):
        # Guards the labelling above against becoming pointless: build a
        # run where blocks-open and audio-passed disagree, by giving the
        # open and closed blocks different sample counts.
        squelch = NoiseSquelch()
        squelch.update(tone_block(), IF_RATE_HZ)
        squelch.apply(np.zeros(1_000, dtype=np.float32))
        squelch.update(noise_block(), IF_RATE_HZ)
        squelch.apply(np.zeros(9_000, dtype=np.float32))

        stats = squelch.stats
        audio_fraction = stats.samples_passed / (stats.samples_passed + stats.samples_muted)

        assert stats.open_fraction == pytest.approx(0.5)
        assert audio_fraction == pytest.approx(0.1)

    def test_stats_is_a_snapshot_not_a_live_view(self):
        squelch = NoiseSquelch()
        squelch.update(tone_block(), IF_RATE_HZ)
        before = squelch.stats

        squelch.update(noise_block(), IF_RATE_HZ)

        assert before.blocks_evaluated == 1
        assert squelch.stats.blocks_evaluated == 2

    def test_stats_is_frozen(self):
        stats = NoiseSquelch().stats

        assert isinstance(stats, SquelchStats)
        with pytest.raises(Exception):  # noqa: B017 - dataclasses.FrozenInstanceError
            stats.blocks_open = 5  # type: ignore[misc]


def _block_measuring_between(target_db: float, open_db: float, close_db: float) -> np.ndarray:
    """Build a block whose quieting lands strictly between two thresholds.

    Scales white noise, which is the physically honest way to do it: the
    metric depends only on the noise band's absolute level, so a partly
    quieted channel *is* noise at a reduced level. Mixing in a voice-band
    tone would not move the reading at all -- which is the point of the
    metric, and was worth discovering here rather than at the bench.

    Solved directly rather than searched, since scaling noise by ``s``
    shifts quieting by exactly ``-20*log10(s)``.
    """
    base = noise_block(seed=7)
    scale = 10.0 ** ((quieting_db(base, IF_RATE_HZ) - target_db) / 20.0)
    block = (base * scale).astype(np.float32)
    measured = quieting_db(block, IF_RATE_HZ)
    assert close_db < measured < open_db, (
        f"helper failed to build a block measuring between {close_db} and {open_db} dB; "
        f"got {measured:.2f} dB"
    )
    return block
