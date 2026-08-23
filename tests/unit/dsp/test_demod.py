"""Tests for WBFM demodulation: config validation, the baseband mixer, and
discriminator correctness against synthetic FM signals and a real capture.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qsorbit.core.dsp import (
    AUDIO_CLIP_RANGE,
    DEFAULT_AUDIO_RATE_HZ,
    DEFAULT_DEVIATION_HZ,
    WbfmConfig,
    demodulate_wbfm,
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
