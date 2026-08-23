"""Tests for integer decimation, including the multi-stage chaining that
works around scipy's own advisory for large single-call factors.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsorbit.core.dsp import decimate
from qsorbit.core.dsp.decimate import MAX_SINGLE_STAGE_FACTOR, _prime_factors, _stage_factors


def synthetic_tone(tone_hz: float, sample_rate_hz: float, n: int) -> np.ndarray:
    t = np.arange(n) / sample_rate_hz
    return np.exp(2j * np.pi * tone_hz * t).astype(np.complex64)


class TestDecimateValidation:
    def test_rejects_a_non_integer_factor(self):
        with pytest.raises(ValueError, match="factor"):
            decimate(np.zeros(100, dtype=np.complex64), 2.5)  # type: ignore[arg-type]

    def test_rejects_a_boolean_factor(self):
        with pytest.raises(ValueError, match="factor"):
            decimate(np.zeros(100, dtype=np.complex64), True)  # type: ignore[arg-type]

    def test_rejects_a_zero_or_negative_factor(self):
        with pytest.raises(ValueError, match="factor"):
            decimate(np.zeros(100, dtype=np.complex64), 0)
        with pytest.raises(ValueError, match="factor"):
            decimate(np.zeros(100, dtype=np.complex64), -3)

    def test_factor_of_one_returns_an_unmodified_copy(self):
        iq = synthetic_tone(100.0, 8_000.0, 64)

        result = decimate(iq, 1)

        assert np.array_equal(result, iq)
        assert result is not iq  # a copy, not the same array


class TestDecimateReducesLength:
    @pytest.mark.parametrize("factor", [2, 4, 10, 16, 42])
    def test_output_length_is_roughly_input_over_factor(self, factor):
        n = 8_000
        iq = synthetic_tone(50.0, 8_000.0, n)

        result = decimate(iq, factor)

        # scipy's decimate uses ceil(len/factor); allow a little slack.
        assert abs(len(result) - n / factor) <= 1

    def test_output_dtype_is_complex64(self):
        iq = synthetic_tone(50.0, 8_000.0, 8_000)

        assert decimate(iq, 4).dtype == np.complex64


class TestDecimateRealValuedInput:
    """decimate() also runs on real-valued signals -- specifically the
    audio demodulate_wbfm() produces downstream of its discriminator,
    which is why the dtype is preserved rather than hard-coded to complex.
    """

    def test_output_dtype_is_float32_for_real_input(self):
        t = np.arange(8_000) / 8_000.0
        audio = np.sin(2 * np.pi * 50.0 * t).astype(np.float32)

        assert decimate(audio, 4).dtype == np.float32

    def test_a_factor_of_one_returns_a_float32_copy(self):
        t = np.arange(64) / 8_000.0
        audio = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)

        result = decimate(audio, 1)

        assert np.array_equal(result, audio)
        assert result is not audio
        assert result.dtype == np.float32

    def test_a_low_frequency_real_tone_survives_decimation(self):
        sample_rate_hz = 48_000.0
        factor = 4
        tone_hz = 1_000.0  # well under the post-decimation Nyquist of 6 kHz
        t = np.arange(4_800) / sample_rate_hz
        audio = np.sin(2 * np.pi * tone_hz * t).astype(np.float32)

        decimated = decimate(audio, factor)
        new_rate_hz = sample_rate_hz / factor

        spectrum = np.abs(np.fft.rfft(decimated))
        freqs = np.fft.rfftfreq(len(decimated), d=1.0 / new_rate_hz)
        peak_hz = freqs[np.argmax(spectrum)]

        assert peak_hz == pytest.approx(tone_hz, abs=new_rate_hz / len(decimated) * 2)


class TestDecimatePreservesALowFrequencyTone:
    def test_a_tone_below_the_new_nyquist_survives_decimation(self):
        sample_rate_hz = 48_000.0
        factor = 4
        tone_hz = 1_000.0  # well under the post-decimation Nyquist of 6 kHz
        iq = synthetic_tone(tone_hz, sample_rate_hz, 4_800)

        decimated = decimate(iq, factor)
        new_rate_hz = sample_rate_hz / factor

        # Measure the tone's frequency in the decimated signal directly,
        # via the FFT bin with the most energy -- independent of
        # SpectrumConfig/power_spectrum_db, so this is a real cross-check
        # rather than testing the same code against itself.
        spectrum = np.fft.fftshift(np.fft.fft(decimated))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(decimated), d=1.0 / new_rate_hz))
        peak_hz = freqs[np.argmax(np.abs(spectrum))]

        assert peak_hz == pytest.approx(tone_hz, abs=new_rate_hz / len(decimated) * 2)

    def test_a_tone_above_the_new_nyquist_does_not_alias_to_a_false_low_frequency(self):
        # This is the point of anti-alias filtering: without it, a tone
        # above the new Nyquist would fold down and be indistinguishable
        # from a real low-frequency signal.
        sample_rate_hz = 48_000.0
        factor = 4  # new Nyquist = 6 kHz
        tone_hz = 20_000.0  # well above it, comfortably below the old Nyquist
        iq = synthetic_tone(tone_hz, sample_rate_hz, 4_800)

        decimated = decimate(iq, factor)
        new_rate_hz = sample_rate_hz / factor

        power = np.abs(np.fft.fft(decimated)) ** 2
        # Energy should be suppressed by the anti-alias filter, not show
        # up as a strong peak anywhere in the decimated (now-aliased-range)
        # spectrum.
        assert power.max() < 0.5 * len(decimated) ** 2  # far below a full-scale tone's own peak

        del new_rate_hz  # documents intent; not asserted on directly above


class TestStageFactors:
    def test_a_small_factor_is_a_single_stage(self):
        assert _stage_factors(4) == [4]
        assert _stage_factors(MAX_SINGLE_STAGE_FACTOR) == [MAX_SINGLE_STAGE_FACTOR]

    def test_stages_multiply_back_to_the_original_factor(self):
        for factor in (12, 40, 100, 256, 1_000):
            stages = _stage_factors(factor)
            product = 1
            for stage in stages:
                product *= stage
            assert product == factor

    def test_every_stage_is_within_the_limit_when_possible(self):
        # 100 = 2*2*5*5: groupable entirely within stages <= 13.
        for stage in _stage_factors(100):
            assert stage <= MAX_SINGLE_STAGE_FACTOR

    def test_a_prime_factor_larger_than_the_limit_is_its_own_stage(self):
        # 17 is prime and bigger than MAX_SINGLE_STAGE_FACTOR; it cannot
        # be split further, so it must appear whole.
        assert _stage_factors(17) == [17]

    def test_a_composite_of_a_large_prime_still_multiplies_back_correctly(self):
        # 34 = 2 * 17: the 17 cannot be grouped down, but the 2 can still
        # ride along as its own stage rather than being folded into it
        # unnecessarily.
        stages = _stage_factors(34)
        product = 1
        for stage in stages:
            product *= stage
        assert product == 34


class TestPrimeFactors:
    def test_a_prime_number_factors_to_itself(self):
        assert _prime_factors(17) == [17]

    def test_a_power_of_two(self):
        assert _prime_factors(16) == [2, 2, 2, 2]

    def test_a_mixed_composite(self):
        assert _prime_factors(60) == [2, 2, 3, 5]

    def test_one_has_no_prime_factors(self):
        assert _prime_factors(1) == []
