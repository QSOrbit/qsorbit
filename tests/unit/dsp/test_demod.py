"""Tests for FM demodulation, wideband and narrowband: config validation,
the baseband mixer, the shared discriminator, and demodulator correctness
against synthetic FM signals and real captures.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qsorbit.core.dsp import (
    AUDIO_CLIP_RANGE,
    DEFAULT_AUDIO_RATE_HZ,
    DEFAULT_DEVIATION_HZ,
    DEFAULT_NBFM_DEEMPHASIS_US,
    DEFAULT_NBFM_DEVIATION_HZ,
    DEFAULT_NBFM_IF_RATE_HZ,
    NbfmConfig,
    NoiseSquelch,
    WbfmConfig,
    demodulate_nbfm,
    demodulate_wbfm,
    discriminate,
    shift_to_baseband,
    unpack_uint8_iq,
)


def synthetic_wbfm(
    audio_tone_hz: float,
    sample_rate_hz: float,
    n: int,
    *,
    deviation_hz: float = DEFAULT_DEVIATION_HZ,
    channel_offset_hz: float = 0.0,
) -> np.ndarray:
    """A synthetic WBFM signal: a pure audio tone frequency-modulated onto
    a carrier at baseband (or at ``channel_offset_hz``, if non-zero).
    """
    t = np.arange(n) / sample_rate_hz
    message = np.sin(2 * np.pi * audio_tone_hz * t)
    phase = 2 * np.pi * deviation_hz * np.cumsum(message) / sample_rate_hz
    iq = np.exp(1j * phase)
    if channel_offset_hz != 0.0:
        iq = iq * np.exp(2j * np.pi * channel_offset_hz * t)
    return iq.astype(np.complex64)


def peak_frequency_hz(audio: np.ndarray, audio_rate_hz: float) -> float:
    """The frequency of the strongest component in a real-valued signal."""
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / audio_rate_hz)
    return freqs[np.argmax(spectrum)]


class TestWbfmConfigValidation:
    def test_a_reasonable_config_is_accepted(self):
        config = WbfmConfig(sample_rate_hz=2_048_000.0)

        assert config.sample_rate_hz == 2_048_000.0
        assert config.audio_rate_hz == DEFAULT_AUDIO_RATE_HZ
        assert config.channel_offset_hz == 0.0
        assert config.deviation_hz == DEFAULT_DEVIATION_HZ
        assert config.de_emphasis_us == 75.0

    def test_is_frozen(self):
        config = WbfmConfig(sample_rate_hz=384_000.0)

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.audio_rate_hz = 8_000.0  # type: ignore[misc]

    def test_decimation_factor_is_the_sample_rate_over_the_audio_rate(self):
        config = WbfmConfig(sample_rate_hz=384_000.0, audio_rate_hz=32_000.0)

        assert config.decimation_factor == 12

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_an_impossible_sample_rate(self, bad):
        with pytest.raises(ValueError, match="sample_rate_hz"):
            WbfmConfig(sample_rate_hz=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_an_impossible_audio_rate(self, bad):
        with pytest.raises(ValueError, match="audio_rate_hz"):
            WbfmConfig(sample_rate_hz=2_048_000.0, audio_rate_hz=bad)

    def test_rejects_a_sample_rate_that_does_not_divide_evenly(self):
        # 2,048,000 / 44,100 is not an integer -- 44,100 does not divide it.
        with pytest.raises(ValueError, match="audio_rate_hz"):
            WbfmConfig(sample_rate_hz=2_048_000.0, audio_rate_hz=44_100.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_a_non_finite_channel_offset(self, bad):
        with pytest.raises(ValueError, match="channel_offset_hz"):
            WbfmConfig(sample_rate_hz=2_048_000.0, channel_offset_hz=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_an_impossible_deviation(self, bad):
        with pytest.raises(ValueError, match="deviation_hz"):
            WbfmConfig(sample_rate_hz=2_048_000.0, deviation_hz=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_an_impossible_deemphasis(self, bad):
        with pytest.raises(ValueError, match="de_emphasis_us"):
            WbfmConfig(sample_rate_hz=2_048_000.0, de_emphasis_us=bad)

    def test_none_is_a_valid_deemphasis(self):
        config = WbfmConfig(sample_rate_hz=2_048_000.0, de_emphasis_us=None)

        assert config.de_emphasis_us is None


class TestShiftToBaseband:
    def test_zero_offset_returns_the_signal_unchanged(self):
        t = np.arange(64) / 8_000.0
        iq = np.exp(2j * np.pi * 100.0 * t).astype(np.complex64)

        result = shift_to_baseband(iq, 0.0, 8_000.0)

        assert np.allclose(result, iq)

    def test_shifting_a_tone_by_its_own_frequency_lands_it_at_dc(self):
        sample_rate_hz = 8_000.0
        tone_hz = 250.0
        t = np.arange(256) / sample_rate_hz
        iq = np.exp(2j * np.pi * tone_hz * t).astype(np.complex64)

        baseband = shift_to_baseband(iq, tone_hz, sample_rate_hz)

        # A tone exactly at DC is a constant-phase phasor: the phase
        # difference between consecutive samples should be ~zero.
        phase_diff = np.angle(baseband[1:] * np.conj(baseband[:-1]))
        assert np.allclose(phase_diff, 0.0, atol=1e-6)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_an_impossible_sample_rate(self, bad):
        with pytest.raises(ValueError, match="sample_rate_hz"):
            shift_to_baseband(np.zeros(8, dtype=np.complex64), 100.0, bad)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_a_non_finite_offset(self, bad):
        with pytest.raises(ValueError, match="offset_hz"):
            shift_to_baseband(np.zeros(8, dtype=np.complex64), bad, 8_000.0)


class TestDemodulateWbfmSyntheticSignal:
    def test_recovers_the_modulating_tone_at_baseband(self):
        sample_rate_hz = 384_000.0
        audio_rate_hz = 32_000.0
        audio_tone_hz = 1_000.0
        duration_s = 0.2
        iq = synthetic_wbfm(audio_tone_hz, sample_rate_hz, int(sample_rate_hz * duration_s))

        config = WbfmConfig(
            sample_rate_hz=sample_rate_hz, audio_rate_hz=audio_rate_hz, de_emphasis_us=None
        )
        audio = demodulate_wbfm(iq, config)

        peak_hz = peak_frequency_hz(audio, audio_rate_hz)
        bin_resolution_hz = audio_rate_hz / len(audio)
        assert peak_hz == pytest.approx(audio_tone_hz, abs=bin_resolution_hz * 2)

    def test_recovered_tone_amplitude_is_close_to_the_modulating_message(self):
        # The synthetic message has peak amplitude 1.0 and is modulated at
        # exactly config.deviation_hz, so the recovered tone should come
        # back close to its original amplitude, not wildly scaled.
        sample_rate_hz = 384_000.0
        audio_tone_hz = 1_000.0
        iq = synthetic_wbfm(audio_tone_hz, sample_rate_hz, int(sample_rate_hz * 0.2))

        config = WbfmConfig(sample_rate_hz=sample_rate_hz, de_emphasis_us=None)
        audio = demodulate_wbfm(iq, config)

        assert 0.7 < np.abs(audio).max() < 1.3

    def test_recovers_the_tone_when_the_channel_sits_off_baseband(self):
        # Mirrors how this project's own captures are tuned: the channel
        # of interest is deliberately away from the IQ's own 0 Hz.
        sample_rate_hz = 384_000.0
        audio_rate_hz = 32_000.0
        audio_tone_hz = 1_000.0
        channel_offset_hz = 50_000.0
        iq = synthetic_wbfm(
            audio_tone_hz,
            sample_rate_hz,
            int(sample_rate_hz * 0.2),
            channel_offset_hz=channel_offset_hz,
        )

        config = WbfmConfig(
            sample_rate_hz=sample_rate_hz,
            audio_rate_hz=audio_rate_hz,
            channel_offset_hz=channel_offset_hz,
            de_emphasis_us=None,
        )
        audio = demodulate_wbfm(iq, config)

        peak_hz = peak_frequency_hz(audio, audio_rate_hz)
        bin_resolution_hz = audio_rate_hz / len(audio)
        assert peak_hz == pytest.approx(audio_tone_hz, abs=bin_resolution_hz * 2)

    def test_ignoring_a_real_channel_offset_recovers_the_wrong_tone(self):
        # The negative-space check for the test above: demodulating an
        # off-baseband channel as if it were centred should NOT recover
        # the original tone, which is what makes shift_to_baseband's
        # contribution here a real correctness fix rather than a no-op.
        sample_rate_hz = 384_000.0
        audio_tone_hz = 1_000.0
        channel_offset_hz = 50_000.0
        iq = synthetic_wbfm(
            audio_tone_hz,
            sample_rate_hz,
            int(sample_rate_hz * 0.2),
            channel_offset_hz=channel_offset_hz,
        )

        config = WbfmConfig(sample_rate_hz=sample_rate_hz, de_emphasis_us=None)
        audio = demodulate_wbfm(iq, config)

        peak_hz = peak_frequency_hz(audio, config.audio_rate_hz)
        assert peak_hz != pytest.approx(audio_tone_hz, abs=200.0)

    def test_output_is_clipped_to_the_documented_range(self):
        # Deliberately over-deviated (well past deviation_hz) to force
        # excursions that would otherwise land outside +/-1.0.
        sample_rate_hz = 384_000.0
        iq = synthetic_wbfm(
            1_000.0, sample_rate_hz, int(sample_rate_hz * 0.2), deviation_hz=400_000.0
        )

        config = WbfmConfig(sample_rate_hz=sample_rate_hz, de_emphasis_us=None)
        audio = demodulate_wbfm(iq, config)

        assert audio.min() >= AUDIO_CLIP_RANGE[0]
        assert audio.max() <= AUDIO_CLIP_RANGE[1]

    def test_output_dtype_is_float32(self):
        sample_rate_hz = 384_000.0
        iq = synthetic_wbfm(1_000.0, sample_rate_hz, int(sample_rate_hz * 0.1))

        config = WbfmConfig(sample_rate_hz=sample_rate_hz)
        audio = demodulate_wbfm(iq, config)

        assert audio.dtype == np.float32


class TestDeEmphasis:
    def test_deemphasis_attenuates_high_frequencies_relative_to_low(self):
        sample_rate_hz = 384_000.0
        n = int(sample_rate_hz * 0.2)
        low_hz, high_hz = 200.0, 10_000.0

        low_iq = synthetic_wbfm(low_hz, sample_rate_hz, n)
        high_iq = synthetic_wbfm(high_hz, sample_rate_hz, n)

        flat_config = WbfmConfig(sample_rate_hz=sample_rate_hz, de_emphasis_us=None)
        deemph_config = WbfmConfig(sample_rate_hz=sample_rate_hz, de_emphasis_us=75.0)

        low_flat = demodulate_wbfm(low_iq, flat_config).max()
        high_flat = demodulate_wbfm(high_iq, flat_config).max()
        low_deemph = demodulate_wbfm(low_iq, deemph_config).max()
        high_deemph = demodulate_wbfm(high_iq, deemph_config).max()

        # Without de-emphasis both tones should recover at roughly the
        # same amplitude; with it, the high tone should be knocked down
        # noticeably more than the low one relative to its own flat
        # baseline -- the signature of a low-pass, not a uniform scaling.
        assert high_deemph / high_flat < low_deemph / low_flat


class TestDemodulateWbfmAgainstRealCapture:
    """Uses the real wbfm-99.9.iq fixture from Chunk C's bring-up (see
    tests/fixtures/iq/README.md). Skips cleanly if it is not present
    locally, per that README's convention.
    """

    def test_recovers_finite_bounded_audio_with_real_content(self, load_iq_fixture):
        raw, metadata = load_iq_fixture("wbfm-99.9")
        iq = unpack_uint8_iq(raw)

        # station_hz is anchored on directly, per the Chunk D hand-off
        # note: this fixture's sidecar carries the OLD "tuning_offset_hz"
        # key, which is the OPPOSITE sign from the new station_offset_hz,
        # and station_hz itself is unambiguous either way.
        channel_offset_hz = metadata["station_hz"] - metadata["actual_center_hz"]

        config = WbfmConfig(
            sample_rate_hz=metadata["actual_sample_rate_hz"],
            channel_offset_hz=channel_offset_hz,
        )
        audio = demodulate_wbfm(iq, config)

        assert len(audio) > 0
        assert np.all(np.isfinite(audio))
        assert audio.min() >= AUDIO_CLIP_RANGE[0]
        assert audio.max() <= AUDIO_CLIP_RANGE[1]
        # Real program content, not silence or a flat DC recovery.
        assert np.std(audio) > 0.01


# ----------------------------------------------------------------------
# Narrowband FM (Chunk G)
# ----------------------------------------------------------------------


def synthetic_nbfm(
    audio_tone_hz: float,
    sample_rate_hz: float,
    n: int,
    *,
    deviation_hz: float = DEFAULT_NBFM_DEVIATION_HZ,
    channel_offset_hz: float = 0.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """A synthetic NBFM signal: an audio tone frequency-modulated onto a
    carrier at baseband (or at ``channel_offset_hz``, if non-zero).

    Identical in shape to :func:`synthetic_wbfm` above, kept separate only
    so the narrowband default deviation applies and the tests read as what
    they are. ``amplitude`` exists so two of these can be summed at
    different strengths to build an adjacent-channel scenario.
    """
    t = np.arange(n) / sample_rate_hz
    message = np.sin(2 * np.pi * audio_tone_hz * t)
    phase = 2 * np.pi * deviation_hz * np.cumsum(message) / sample_rate_hz
    iq = amplitude * np.exp(1j * phase)
    if channel_offset_hz != 0.0:
        iq = iq * np.exp(2j * np.pi * channel_offset_hz * t)
    return iq.astype(np.complex64)


class TestNbfmConfigValidation:
    def test_a_reasonable_config_is_accepted(self):
        config = NbfmConfig(sample_rate_hz=2_048_000.0)

        assert config.if_rate_hz == DEFAULT_NBFM_IF_RATE_HZ
        assert config.audio_rate_hz == DEFAULT_AUDIO_RATE_HZ
        assert config.deviation_hz == DEFAULT_NBFM_DEVIATION_HZ
        assert config.de_emphasis_us == DEFAULT_NBFM_DEEMPHASIS_US

    def test_is_frozen(self):
        config = NbfmConfig(sample_rate_hz=2_048_000.0)

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.if_rate_hz = 16_000.0  # type: ignore[misc]

    def test_the_two_decimation_factors_split_the_chain_at_the_if_rate(self):
        config = NbfmConfig(sample_rate_hz=2_048_000.0, if_rate_hz=64_000.0, audio_rate_hz=32_000.0)

        assert config.channel_decimation_factor == 32
        assert config.audio_decimation_factor == 2

    def test_the_default_audio_decimation_is_one(self):
        # Expected, not a mistake: with the default IF rate the channel
        # filter has already brought the signal to a usable audio rate.
        assert NbfmConfig(sample_rate_hz=2_048_000.0).audio_decimation_factor == 1

    @pytest.mark.parametrize(
        "name,bad",
        [
            ("sample_rate_hz", 0.0),
            ("sample_rate_hz", -1.0),
            ("sample_rate_hz", float("nan")),
            ("sample_rate_hz", float("inf")),
            ("if_rate_hz", 0.0),
            ("if_rate_hz", -1.0),
            ("if_rate_hz", float("nan")),
            ("if_rate_hz", float("inf")),
            ("audio_rate_hz", 0.0),
            ("audio_rate_hz", -1.0),
            ("audio_rate_hz", float("nan")),
            ("audio_rate_hz", float("inf")),
        ],
    )
    def test_rejects_an_impossible_rate(self, name, bad):
        kwargs = {"sample_rate_hz": 2_048_000.0, name: bad}
        with pytest.raises(ValueError, match=name):
            NbfmConfig(**kwargs)

    def test_rejects_an_if_rate_that_does_not_divide_the_sample_rate(self):
        with pytest.raises(ValueError, match="if_rate_hz"):
            NbfmConfig(sample_rate_hz=2_048_000.0, if_rate_hz=30_000.0)

    def test_rejects_an_audio_rate_that_does_not_divide_the_if_rate(self):
        with pytest.raises(ValueError, match="audio_rate_hz"):
            NbfmConfig(sample_rate_hz=2_048_000.0, if_rate_hz=32_000.0, audio_rate_hz=12_000.0)

    def test_rejects_an_audio_rate_above_the_if_rate(self):
        with pytest.raises(ValueError, match="audio_rate_hz"):
            NbfmConfig(sample_rate_hz=2_048_000.0, if_rate_hz=32_000.0, audio_rate_hz=64_000.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_a_non_finite_channel_offset(self, bad):
        with pytest.raises(ValueError, match="channel_offset_hz"):
            NbfmConfig(sample_rate_hz=2_048_000.0, channel_offset_hz=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_an_impossible_deemphasis(self, bad):
        with pytest.raises(ValueError, match="de_emphasis_us"):
            NbfmConfig(sample_rate_hz=2_048_000.0, de_emphasis_us=bad)

    def test_none_is_a_valid_deemphasis(self):
        assert NbfmConfig(sample_rate_hz=2_048_000.0, de_emphasis_us=None).de_emphasis_us is None

    def test_rejects_an_if_rate_too_low_for_the_deviation(self):
        # The check that matters most: below 2*deviation the discriminator's
        # phase advance passes pi and wraps, corrupting the audio silently.
        # 8 kHz IF against 5 kHz deviation is well inside that trap.
        with pytest.raises(ValueError, match="wraps"):
            NbfmConfig(sample_rate_hz=2_048_000.0, if_rate_hz=8_000.0, audio_rate_hz=8_000.0)

    def test_accepts_an_if_rate_just_above_the_aliasing_floor(self):
        # The boundary is 2*deviation exactly; just above it must pass, so
        # the check is a real limit rather than a vague safety margin.
        config = NbfmConfig(
            sample_rate_hz=2_048_000.0,
            if_rate_hz=16_000.0,
            audio_rate_hz=16_000.0,
            deviation_hz=5_000.0,
        )

        assert config.if_rate_hz > 2.0 * config.deviation_hz


class TestDemodulateNbfmSyntheticSignal:
    def test_recovers_the_modulating_tone_at_baseband(self):
        sample_rate_hz = 1_024_000.0
        audio_tone_hz = 1_000.0
        iq = synthetic_nbfm(audio_tone_hz, sample_rate_hz, int(sample_rate_hz * 0.5))

        config = NbfmConfig(sample_rate_hz=sample_rate_hz, de_emphasis_us=None)
        audio = demodulate_nbfm(iq, config)

        peak_hz = peak_frequency_hz(audio, config.audio_rate_hz)
        bin_resolution_hz = config.audio_rate_hz / len(audio)
        assert peak_hz == pytest.approx(audio_tone_hz, abs=bin_resolution_hz * 3)

    def test_recovers_the_tone_when_the_channel_sits_off_baseband(self):
        # Mirrors the real fixtures: the station is deliberately 250 kHz
        # away from the tuned centre, to dodge the DC spike.
        sample_rate_hz = 1_024_000.0
        audio_tone_hz = 1_000.0
        channel_offset_hz = 250_000.0
        iq = synthetic_nbfm(
            audio_tone_hz,
            sample_rate_hz,
            int(sample_rate_hz * 0.5),
            channel_offset_hz=channel_offset_hz,
        )

        config = NbfmConfig(
            sample_rate_hz=sample_rate_hz,
            channel_offset_hz=channel_offset_hz,
            de_emphasis_us=None,
        )
        audio = demodulate_nbfm(iq, config)

        peak_hz = peak_frequency_hz(audio, config.audio_rate_hz)
        bin_resolution_hz = config.audio_rate_hz / len(audio)
        assert peak_hz == pytest.approx(audio_tone_hz, abs=bin_resolution_hz * 3)

    def test_the_channel_filter_rejects_an_adjacent_channel(self):
        # The reason narrowband MUST decimate before discriminating. A
        # neighbour 25 kHz away -- NOAA's actual channel spacing, and both
        # NOAA fixtures have one -- would alias onto the wanted channel if
        # the channel filter were not doing its job.
        sample_rate_hz = 1_024_000.0
        n = int(sample_rate_hz * 0.5)
        wanted = synthetic_nbfm(1_000.0, sample_rate_hz, n)
        neighbour = synthetic_nbfm(
            2_500.0, sample_rate_hz, n, channel_offset_hz=25_000.0, amplitude=1.0
        )

        config = NbfmConfig(sample_rate_hz=sample_rate_hz, de_emphasis_us=None)
        audio = demodulate_nbfm((wanted + neighbour).astype(np.complex64), config)

        peak_hz = peak_frequency_hz(audio, config.audio_rate_hz)
        bin_resolution_hz = config.audio_rate_hz / len(audio)
        assert peak_hz == pytest.approx(1_000.0, abs=bin_resolution_hz * 3)

    def test_a_louder_adjacent_channel_does_not_capture_the_discriminator(self):
        # The sharper version of the test above, and the one that pins
        # Session 14's lesson into the DSP: our signal is not the loudest
        # thing in the capture, and it must still be the one recovered.
        sample_rate_hz = 1_024_000.0
        n = int(sample_rate_hz * 0.5)
        wanted = synthetic_nbfm(1_000.0, sample_rate_hz, n, amplitude=1.0)
        louder_neighbour = synthetic_nbfm(
            2_500.0, sample_rate_hz, n, channel_offset_hz=25_000.0, amplitude=10.0
        )

        config = NbfmConfig(sample_rate_hz=sample_rate_hz, de_emphasis_us=None)
        audio = demodulate_nbfm((wanted + louder_neighbour).astype(np.complex64), config)

        peak_hz = peak_frequency_hz(audio, config.audio_rate_hz)
        bin_resolution_hz = config.audio_rate_hz / len(audio)
        assert peak_hz == pytest.approx(1_000.0, abs=bin_resolution_hz * 3)

    def test_skipping_the_channel_filter_would_lose_to_the_louder_neighbour(self):
        # The negative-space proof that the channel filter is what saves
        # the test above, not luck. Demodulating the same mixture at the
        # full IQ rate -- i.e. the WBFM chain's structure, no channel
        # filter -- must NOT recover the wanted tone.
        sample_rate_hz = 1_024_000.0
        n = int(sample_rate_hz * 0.5)
        wanted = synthetic_nbfm(1_000.0, sample_rate_hz, n, amplitude=1.0)
        louder_neighbour = synthetic_nbfm(
            2_500.0, sample_rate_hz, n, channel_offset_hz=25_000.0, amplitude=10.0
        )
        mixture = (wanted + louder_neighbour).astype(np.complex64)

        unfiltered = WbfmConfig(
            sample_rate_hz=sample_rate_hz,
            audio_rate_hz=32_000.0,
            deviation_hz=DEFAULT_NBFM_DEVIATION_HZ,
            de_emphasis_us=None,
        )
        audio = demodulate_wbfm(mixture, unfiltered)

        peak_hz = peak_frequency_hz(audio, unfiltered.audio_rate_hz)
        assert peak_hz != pytest.approx(1_000.0, abs=100.0)

    def test_output_is_clipped_and_float32(self):
        sample_rate_hz = 1_024_000.0
        iq = synthetic_nbfm(
            1_000.0, sample_rate_hz, int(sample_rate_hz * 0.2), deviation_hz=15_000.0
        )

        config = NbfmConfig(sample_rate_hz=sample_rate_hz, de_emphasis_us=None)
        audio = demodulate_nbfm(iq, config)

        assert audio.dtype == np.float32
        assert audio.min() >= AUDIO_CLIP_RANGE[0]
        assert audio.max() <= AUDIO_CLIP_RANGE[1]

    def test_rejects_a_block_too_short_to_survive_the_channel_filter(self):
        config = NbfmConfig(sample_rate_hz=2_048_000.0)

        with pytest.raises(ValueError, match="too short"):
            demodulate_nbfm(np.zeros(16, dtype=np.complex64), config)


class TestNbfmDeEmphasis:
    def test_deemphasis_attenuates_high_audio_frequencies_more_than_low(self):
        sample_rate_hz = 1_024_000.0
        n = int(sample_rate_hz * 0.3)
        low_iq = synthetic_nbfm(200.0, sample_rate_hz, n)
        high_iq = synthetic_nbfm(3_000.0, sample_rate_hz, n)

        flat = NbfmConfig(sample_rate_hz=sample_rate_hz, de_emphasis_us=None)
        shaped = NbfmConfig(sample_rate_hz=sample_rate_hz, de_emphasis_us=750.0)

        low_ratio = demodulate_nbfm(low_iq, shaped).max() / demodulate_nbfm(low_iq, flat).max()
        high_ratio = demodulate_nbfm(high_iq, shaped).max() / demodulate_nbfm(high_iq, flat).max()

        assert high_ratio < low_ratio


class TestDiscriminate:
    def test_an_unmodulated_carrier_at_baseband_recovers_zero(self):
        carrier = np.ones(256, dtype=np.complex64)

        recovered = discriminate(carrier, 32_000.0, 5_000.0)

        assert np.allclose(recovered, 0.0, atol=1e-6)

    def test_a_constant_frequency_offset_recovers_that_offset(self):
        sample_rate_hz = 32_000.0
        offset_hz = 2_500.0
        t = np.arange(1_024) / sample_rate_hz
        signal = np.exp(2j * np.pi * offset_hz * t).astype(np.complex64)

        recovered = discriminate(signal, sample_rate_hz, 5_000.0)

        # Normalised by deviation_hz, so 2.5 kHz of 5 kHz reads as 0.5.
        assert np.median(recovered) == pytest.approx(0.5, abs=1e-3)

    def test_output_is_one_sample_shorter_than_the_input(self):
        recovered = discriminate(np.ones(100, dtype=np.complex64), 32_000.0, 5_000.0)

        assert len(recovered) == 99

    def test_rejects_a_block_too_short_to_form_a_phase_difference(self):
        with pytest.raises(ValueError, match="two samples"):
            discriminate(np.ones(1, dtype=np.complex64), 32_000.0, 5_000.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_an_impossible_deviation(self, bad):
        with pytest.raises(ValueError, match="deviation_hz"):
            discriminate(np.ones(8, dtype=np.complex64), 32_000.0, bad)


class TestNbfmWithSquelch:
    def test_a_squelch_passes_a_real_signal_through(self):
        sample_rate_hz = 1_024_000.0
        iq = synthetic_nbfm(1_000.0, sample_rate_hz, int(sample_rate_hz * 0.2))
        config = NbfmConfig(sample_rate_hz=sample_rate_hz)
        squelch = NoiseSquelch()

        audio = demodulate_nbfm(iq, config, squelch=squelch)

        assert squelch.is_open is True
        assert np.abs(audio).max() > 0.0

    def test_a_squelch_mutes_an_empty_channel(self):
        sample_rate_hz = 1_024_000.0
        rng = np.random.default_rng(3)
        n = int(sample_rate_hz * 0.2)
        noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
        config = NbfmConfig(sample_rate_hz=sample_rate_hz)
        squelch = NoiseSquelch()

        audio = demodulate_nbfm(noise, config, squelch=squelch)

        assert squelch.is_open is False
        assert not audio.any()
        assert squelch.stats.samples_muted == len(audio)

    def test_the_same_empty_channel_is_audible_without_a_squelch(self):
        # Proves the muting above is the squelch's doing and not the chain
        # producing nothing -- the exact "muted versus broken" distinction
        # the accounting exists to make.
        sample_rate_hz = 1_024_000.0
        rng = np.random.default_rng(3)
        n = int(sample_rate_hz * 0.2)
        noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
        config = NbfmConfig(sample_rate_hz=sample_rate_hz)

        audio = demodulate_nbfm(noise, config)

        assert np.abs(audio).max() > 0.0

    def test_squelch_state_carries_across_blocks(self):
        # The reason the squelch is an object passed in rather than a
        # config value: hysteresis has to remember the previous block.
        sample_rate_hz = 1_024_000.0
        n = int(sample_rate_hz * 0.2)
        iq = synthetic_nbfm(1_000.0, sample_rate_hz, n)
        config = NbfmConfig(sample_rate_hz=sample_rate_hz)
        squelch = NoiseSquelch()

        demodulate_nbfm(iq, config, squelch=squelch)
        demodulate_nbfm(iq, config, squelch=squelch)

        assert squelch.stats.blocks_evaluated == 2
        assert squelch.stats.blocks_open == 2


class TestDemodulateNbfmAgainstRealCapture:
    """Uses the real NOAA weather-radio fixtures from Chunk C's bring-up
    (see tests/fixtures/iq/README.md). Skips cleanly if absent.
    """

    @pytest.mark.parametrize("name", ["nbfm-noaa-162.550", "nbfm-noaa-162.475"])
    def test_recovers_finite_bounded_audio_with_real_content(self, load_iq_fixture, name):
        raw, metadata = load_iq_fixture(name)
        iq = unpack_uint8_iq(raw)

        # station_hz directly, never re-derived from either sidecar's
        # offset key -- the two carry opposite signs. Chunk D hand-off.
        channel_offset_hz = metadata["station_hz"] - metadata["actual_center_hz"]

        config = NbfmConfig(
            sample_rate_hz=metadata["actual_sample_rate_hz"],
            channel_offset_hz=channel_offset_hz,
        )
        audio = demodulate_nbfm(iq, config)

        assert len(audio) > 0
        assert np.all(np.isfinite(audio))
        assert audio.min() >= AUDIO_CLIP_RANGE[0]
        assert audio.max() <= AUDIO_CLIP_RANGE[1]
        assert np.std(audio) > 0.001

    def test_the_real_capture_opens_a_squelch(self, load_iq_fixture):
        # NOAA weather radio transmits continuously, so a capture of it
        # must read as quieted. If this fails, either the squelch metric
        # or the demodulator is wrong -- and the quieting figure printed
        # in the message is what tells them apart.
        raw, metadata = load_iq_fixture("nbfm-noaa-162.550")
        iq = unpack_uint8_iq(raw)
        channel_offset_hz = metadata["station_hz"] - metadata["actual_center_hz"]
        config = NbfmConfig(
            sample_rate_hz=metadata["actual_sample_rate_hz"],
            channel_offset_hz=channel_offset_hz,
        )
        squelch = NoiseSquelch()

        demodulate_nbfm(iq, config, squelch=squelch)

        assert squelch.is_open is True, (
            f"NOAA is always on, so this capture should read as quieted; "
            f"measured {squelch.stats.last_quieting_db:.1f} dB"
        )
