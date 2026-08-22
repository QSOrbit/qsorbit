"""Tests for spectrum framing: config validation, frequency axis, and
power-spectrum correctness against both synthetic tones and real captures.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qsorbit.core.dsp import (
    SpectrumConfig,
    frame_iq,
    frequency_axis_hz,
    power_spectrum_db,
    unpack_uint8_iq,
)


def synthetic_tone(tone_hz: float, config: SpectrumConfig) -> np.ndarray:
    """One frame of a pure complex tone at ``tone_hz`` baseband."""
    t = np.arange(config.fft_size) / config.sample_rate_hz
    return np.exp(2j * np.pi * tone_hz * t).astype(np.complex64)


class TestSpectrumConfigValidation:
    def test_a_reasonable_config_is_accepted(self):
        config = SpectrumConfig(fft_size=1024, sample_rate_hz=2_048_000.0)

        assert config.fft_size == 1024
        assert config.sample_rate_hz == 2_048_000.0
        assert config.center_freq_hz == 0.0
        assert config.window == "hann"

    def test_is_frozen(self):
        config = SpectrumConfig(fft_size=64, sample_rate_hz=8_000.0)

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.fft_size = 128  # type: ignore[misc]

    @pytest.mark.parametrize("bad", [0, 1, 2, 3, -8])
    def test_rejects_an_fft_size_too_small_to_be_useful(self, bad):
        with pytest.raises(ValueError, match="fft_size"):
            SpectrumConfig(fft_size=bad, sample_rate_hz=8_000.0)

    def test_rejects_a_non_integer_fft_size(self):
        with pytest.raises(ValueError, match="fft_size"):
            SpectrumConfig(fft_size=64.0, sample_rate_hz=8_000.0)  # type: ignore[arg-type]

    def test_rejects_a_boolean_fft_size(self):
        with pytest.raises(ValueError, match="fft_size"):
            SpectrumConfig(fft_size=True, sample_rate_hz=8_000.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_impossible_sample_rates(self, bad):
        with pytest.raises(ValueError, match="sample_rate_hz"):
            SpectrumConfig(fft_size=64, sample_rate_hz=bad)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_a_non_finite_center_freq(self, bad):
        with pytest.raises(ValueError, match="center_freq_hz"):
            SpectrumConfig(fft_size=64, sample_rate_hz=8_000.0, center_freq_hz=bad)

    def test_rejects_an_unrecognized_window_name(self):
        with pytest.raises(ValueError, match="window"):
            SpectrumConfig(fft_size=64, sample_rate_hz=8_000.0, window="not-a-real-window")

    def test_accepts_a_rectangular_window_by_its_scipy_name(self):
        # scipy calls a rectangular window "boxcar", not "rect" -- this
        # documents that the name has to match scipy's vocabulary.
        config = SpectrumConfig(fft_size=64, sample_rate_hz=8_000.0, window="boxcar")

        assert config.window == "boxcar"


class TestFrequencyAxis:
    def test_axis_length_matches_fft_size(self):
        config = SpectrumConfig(fft_size=256, sample_rate_hz=8_000.0)

        assert len(frequency_axis_hz(config)) == 256

    def test_axis_is_monotonically_increasing(self):
        config = SpectrumConfig(fft_size=256, sample_rate_hz=8_000.0)

        freqs = frequency_axis_hz(config)

        assert np.all(np.diff(freqs) > 0)

    def test_axis_spans_plus_minus_half_the_sample_rate(self):
        config = SpectrumConfig(fft_size=8, sample_rate_hz=8_000.0)

        freqs = frequency_axis_hz(config)

        assert freqs.min() == pytest.approx(-4_000.0)
        assert freqs.max() == pytest.approx(-4_000.0 + 7 * 1_000.0)

    def test_center_freq_hz_shifts_the_whole_axis(self):
        baseband = SpectrumConfig(fft_size=8, sample_rate_hz=8_000.0)
        shifted = SpectrumConfig(fft_size=8, sample_rate_hz=8_000.0, center_freq_hz=100_000_000.0)

        assert frequency_axis_hz(shifted) == pytest.approx(
            frequency_axis_hz(baseband) + 100_000_000.0
        )

    def test_dc_bin_sits_at_the_center_frequency(self):
        # fft_size even: numpy's fftfreq/fftshift convention puts the
        # zero-frequency bin at index fft_size // 2.
        config = SpectrumConfig(fft_size=8, sample_rate_hz=8_000.0, center_freq_hz=42.0)

        freqs = frequency_axis_hz(config)

        assert freqs[len(freqs) // 2] == pytest.approx(42.0)


class TestPowerSpectrumSyntheticTone:
    def test_a_bin_aligned_tone_peaks_at_exactly_its_own_bin(self):
        config = SpectrumConfig(fft_size=64, sample_rate_hz=8_000.0, window="boxcar")
        bin_resolution_hz = config.sample_rate_hz / config.fft_size  # 125 Hz
        tone_hz = 8 * bin_resolution_hz  # bin-aligned, avoids spectral leakage

        power_db = power_spectrum_db(synthetic_tone(tone_hz, config), config)
        freqs = frequency_axis_hz(config)
        peak_hz = freqs[np.argmax(power_db)]

        assert peak_hz == pytest.approx(tone_hz, abs=bin_resolution_hz / 2)

    def test_a_negative_frequency_tone_peaks_on_the_negative_side(self):
        config = SpectrumConfig(fft_size=64, sample_rate_hz=8_000.0, window="boxcar")
        bin_resolution_hz = config.sample_rate_hz / config.fft_size
        tone_hz = -8 * bin_resolution_hz

        power_db = power_spectrum_db(synthetic_tone(tone_hz, config), config)
        freqs = frequency_axis_hz(config)
        peak_hz = freqs[np.argmax(power_db)]

        assert peak_hz == pytest.approx(tone_hz, abs=bin_resolution_hz / 2)

    def test_center_freq_hz_shifts_the_reported_peak_to_match(self):
        config = SpectrumConfig(
            fft_size=64, sample_rate_hz=8_000.0, center_freq_hz=1_000_000.0, window="boxcar"
        )
        bin_resolution_hz = config.sample_rate_hz / config.fft_size
        tone_hz = 8 * bin_resolution_hz  # baseband offset

        power_db = power_spectrum_db(synthetic_tone(tone_hz, config), config)
        freqs = frequency_axis_hz(config)
        peak_hz = freqs[np.argmax(power_db)]

        assert peak_hz == pytest.approx(1_000_000.0 + tone_hz, abs=bin_resolution_hz / 2)

    def test_a_louder_tone_reports_higher_power(self):
        config = SpectrumConfig(fft_size=64, sample_rate_hz=8_000.0, window="boxcar")
        tone_hz = 8 * (config.sample_rate_hz / config.fft_size)
        quiet = 0.1 * synthetic_tone(tone_hz, config)
        loud = 1.0 * synthetic_tone(tone_hz, config)

        quiet_peak = power_spectrum_db(quiet, config).max()
        loud_peak = power_spectrum_db(loud, config).max()

        # Voltage ratio of 10x should be a 20 dB power ratio.
        assert loud_peak - quiet_peak == pytest.approx(20.0, abs=0.5)

    def test_window_choice_does_not_change_a_tones_reported_power(self):
        # This is the reason power_spectrum_db normalises by the window's
        # coherent gain: swapping windows should trade sidelobe leakage,
        # not silently rescale the reported level of the tone itself.
        bin_resolution_hz = 8_000.0 / 64
        tone_hz = 8 * bin_resolution_hz
        rect_config = SpectrumConfig(fft_size=64, sample_rate_hz=8_000.0, window="boxcar")
        hann_config = SpectrumConfig(fft_size=64, sample_rate_hz=8_000.0, window="hann")

        rect_peak = power_spectrum_db(synthetic_tone(tone_hz, rect_config), rect_config).max()
        hann_peak = power_spectrum_db(synthetic_tone(tone_hz, hann_config), hann_config).max()

        assert rect_peak == pytest.approx(hann_peak, abs=0.5)

    def test_an_all_zero_frame_is_finite_not_negative_infinity(self):
        config = SpectrumConfig(fft_size=32, sample_rate_hz=8_000.0)
        silence = np.zeros(32, dtype=np.complex64)

        power_db = power_spectrum_db(silence, config)

        assert np.all(np.isfinite(power_db))
        assert np.all(power_db <= config.fft_size)  # nowhere near a real signal's level

    def test_rejects_a_frame_of_the_wrong_length(self):
        config = SpectrumConfig(fft_size=64, sample_rate_hz=8_000.0)

        with pytest.raises(ValueError, match="64"):
            power_spectrum_db(np.zeros(32, dtype=np.complex64), config)

    def test_output_length_and_dtype(self):
        config = SpectrumConfig(fft_size=128, sample_rate_hz=8_000.0)

        power_db = power_spectrum_db(np.zeros(128, dtype=np.complex64), config)

        assert len(power_db) == 128
        assert power_db.dtype == np.float32


class TestFrameIq:
    def test_splits_an_exact_multiple_into_non_overlapping_frames(self):
        config = SpectrumConfig(fft_size=16, sample_rate_hz=8_000.0)
        iq = np.arange(48, dtype=np.complex64)

        frames = list(frame_iq(iq, config))

        assert len(frames) == 3
        assert all(len(frame) == 16 for frame in frames)
        assert frames[0][0] == 0
        assert frames[1][0] == 16
        assert frames[2][0] == 32

    def test_drops_trailing_samples_that_do_not_fill_a_frame(self):
        config = SpectrumConfig(fft_size=16, sample_rate_hz=8_000.0)
        iq = np.arange(40, dtype=np.complex64)  # two full frames, 8 left over

        frames = list(frame_iq(iq, config))

        assert len(frames) == 2

    def test_a_buffer_shorter_than_one_frame_yields_nothing(self):
        config = SpectrumConfig(fft_size=16, sample_rate_hz=8_000.0)
        iq = np.arange(8, dtype=np.complex64)

        assert list(frame_iq(iq, config)) == []

    def test_a_smaller_hop_overlaps_frames(self):
        config = SpectrumConfig(fft_size=16, sample_rate_hz=8_000.0)
        iq = np.arange(32, dtype=np.complex64)

        frames = list(frame_iq(iq, config, hop=8))

        assert len(frames) == 3  # starts at 0, 8, 16
        assert frames[1][0] == 8

    def test_rejects_a_non_positive_hop(self):
        config = SpectrumConfig(fft_size=16, sample_rate_hz=8_000.0)
        iq = np.arange(32, dtype=np.complex64)

        with pytest.raises(ValueError, match="hop"):
            list(frame_iq(iq, config, hop=0))


class TestPowerSpectrumAgainstRealCaptures:
    """Uses the real IQ fixtures from Chunk C's bring-up (see
    tests/fixtures/iq/README.md). Skips cleanly if they are not present
    locally, per that README's convention.
    """

    def test_the_broadcast_station_is_detectable_at_its_known_offset(self, load_iq_fixture):
        raw, metadata = load_iq_fixture("wbfm-99.9")

        iq = unpack_uint8_iq(raw)
        config = SpectrumConfig(
            fft_size=4096,
            sample_rate_hz=metadata["actual_sample_rate_hz"],
            center_freq_hz=metadata["actual_center_hz"],
        )
        frame = next(frame_iq(iq, config))
        power_db = power_spectrum_db(frame, config)
        freqs = frequency_axis_hz(config)

        # This fixture predates capture.py's station_offset_hz key: its
        # sidecar carries the OLD bench-script "tuning_offset_hz", which
        # is centre-minus-station -- the OPPOSITE sign. station_hz itself
        # is unambiguous either way, so anchor on that directly rather
        # than re-deriving it from a sign convention that has already
        # bitten this project once (see project-notes.md, Chunk D
        # hand-off). Do not read tuning_offset_hz here as if it were
        # station_offset_hz.
        station_hz = metadata["station_hz"]

        # A generous window around the expected bin, in case of minor
        # centre-frequency quantisation -- this is a signal-presence
        # check, not a precision-tuning check.
        bin_resolution_hz = config.sample_rate_hz / config.fft_size
        window_bins = 4
        target_bin = int(np.argmin(np.abs(freqs - station_hz)))
        lo = max(target_bin - window_bins, 0)
        hi = min(target_bin + window_bins + 1, len(power_db))

        signal_power = power_db[lo:hi].max()
        background_power = np.median(power_db)

        # "Is our signal present at our frequency", not "is our signal
        # the loudest" -- the RTL-SDR's permanent DC spike sits at the
        # tuned centre, not at the station, and would win a bare argmax.
        assert signal_power - background_power > 10.0, (
            f"expected an elevated peak near {station_hz:,.0f} Hz "
            f"(bin resolution {bin_resolution_hz:,.0f} Hz); got {signal_power:.1f} dB "
            f"there against a {background_power:.1f} dB background"
        )

    def test_frequency_axis_lines_up_with_the_fixtures_own_offset_math(self, load_iq_fixture):
        """The station's bin index, worked out purely from frequency_axis_hz,
        must fall the fixture's documented 250 kHz away from the tuned
        centre's bin index -- a check on our own bin-to-Hz bookkeeping,
        independent of what is actually in the capture.
        """
        _, metadata = load_iq_fixture("wbfm-99.9")

        config = SpectrumConfig(
            fft_size=4096,
            sample_rate_hz=metadata["actual_sample_rate_hz"],
            center_freq_hz=metadata["actual_center_hz"],
        )
        freqs = frequency_axis_hz(config)
        bin_resolution_hz = config.sample_rate_hz / config.fft_size

        center_bin = int(np.argmin(np.abs(freqs - metadata["actual_center_hz"])))
        station_bin = int(np.argmin(np.abs(freqs - metadata["station_hz"])))
        bin_distance_hz = abs(station_bin - center_bin) * bin_resolution_hz

        assert bin_distance_hz == pytest.approx(250_000.0, abs=bin_resolution_hz)
