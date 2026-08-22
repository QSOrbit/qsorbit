"""Tests for SDR device configuration and gain snapping."""

from __future__ import annotations

import dataclasses

import pytest

from qsorbit.core.sdr import (
    AUTO_GAIN,
    MAX_PPM,
    RELIABLE_MAX_SAMPLE_RATE_HZ,
    SdrConfig,
    nearest_gain_step,
)

#: The 29 steps the V4's R828D reported during bring-up. Used here as a
#: realistic table to snap against — never as a validation list in the
#: code under test, which reads the real one from the device because the
#: table belongs to the tuner chip, not to us.
V4_GAIN_STEPS_DB = (
    0.0,
    0.9,
    1.4,
    2.7,
    3.7,
    7.7,
    8.7,
    12.5,
    14.4,
    15.7,
    16.6,
    19.7,
    20.7,
    22.9,
    25.4,
    28.0,
    29.7,
    32.8,
    33.8,
    36.4,
    37.2,
    38.6,
    40.2,
    42.1,
    43.4,
    43.9,
    44.5,
    48.0,
    49.6,
)


def a_config(**overrides: object) -> SdrConfig:
    """A valid config, matching the first-light capture, with overrides."""
    values: dict[str, object] = {
        "center_hz": 99_650_000.0,
        "sample_rate_hz": 2_048_000.0,
        "gain_db": 32.8,
    }
    values.update(overrides)
    return SdrConfig(**values)  # type: ignore[arg-type]


class TestValidConfigs:
    def test_accepts_the_first_light_settings(self):
        config = a_config()

        assert config.center_hz == 99_650_000.0
        assert config.sample_rate_hz == 2_048_000.0
        assert config.gain_db == 32.8
        assert config.ppm == 0
        assert config.enable_agc is False

    def test_is_frozen(self):
        config = a_config()

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.center_hz = 1.0  # type: ignore[misc]

    def test_accepts_the_narrow_sample_rate_window(self):
        assert a_config(sample_rate_hz=250_000).sample_rate_hz == 250_000

    def test_accepts_auto_gain(self):
        config = a_config(gain_db=AUTO_GAIN)

        assert config.uses_auto_gain

    def test_a_manual_gain_is_not_auto(self):
        assert not a_config(gain_db=0.0).uses_auto_gain

    def test_zero_gain_is_a_legal_step_even_though_it_is_a_bad_idea(self):
        # 0.0 dB is on the device's own table. The config's job is to
        # reject impossible numbers, not unwise ones.
        assert a_config(gain_db=0.0).gain_db == 0.0


class TestRejectedConfigs:
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_impossible_centre_frequencies(self, bad):
        with pytest.raises(ValueError, match="center_hz"):
            a_config(center_hz=bad)

    @pytest.mark.parametrize("bad", [0, 224_000, 300_001, 899_999, 3_200_001, 10_000_000])
    def test_rejects_sample_rates_outside_the_chip_windows(self, bad):
        with pytest.raises(ValueError, match="sample_rate_hz"):
            a_config(sample_rate_hz=bad)

    def test_the_sample_rate_message_names_both_windows(self):
        with pytest.raises(ValueError) as caught:
            a_config(sample_rate_hz=500_000)

        message = str(caught.value)
        assert "225,001-300,000 Hz" in message
        assert "900,001-3,200,000 Hz" in message

    @pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
    def test_rejects_impossible_gains(self, bad):
        with pytest.raises(ValueError, match="gain_db"):
            a_config(gain_db=bad)

    def test_rejects_a_non_numeric_gain(self):
        with pytest.raises(ValueError, match="gain_db"):
            a_config(gain_db="loud")

    def test_rejects_a_boolean_gain(self):
        # bool is a subclass of int, so True would otherwise sail through
        # as 1.0 dB.
        with pytest.raises(ValueError, match="gain_db"):
            a_config(gain_db=True)

    def test_rejects_a_boolean_ppm(self):
        with pytest.raises(ValueError, match="ppm"):
            a_config(ppm=True)

    def test_rejects_a_fractional_ppm(self):
        with pytest.raises(ValueError, match="ppm"):
            a_config(ppm=1.5)

    @pytest.mark.parametrize("bad", [MAX_PPM + 1, -MAX_PPM - 1])
    def test_rejects_an_absurd_ppm(self, bad):
        with pytest.raises(ValueError, match="ppm"):
            a_config(ppm=bad)


class TestDerivedProperties:
    def test_flags_a_rate_usb_may_not_sustain(self):
        assert a_config(sample_rate_hz=RELIABLE_MAX_SAMPLE_RATE_HZ + 1).may_drop_samples

    def test_does_not_flag_the_rate_bring_up_used(self):
        assert not a_config(sample_rate_hz=2_048_000).may_drop_samples

    # The offset_from() tests that used to live here moved to
    # test_device.py alongside AppliedSettings, which is where the
    # method went: an offset has to be measured from the centre the
    # tuner actually reached, not the one it was asked for.


class TestNearestGainStep:
    def test_an_exact_step_is_returned_unchanged(self):
        assert nearest_gain_step(32.8, V4_GAIN_STEPS_DB) == 32.8

    def test_snaps_upward_to_the_closest_step(self):
        assert nearest_gain_step(32.7, V4_GAIN_STEPS_DB) == 32.8

    def test_snaps_downward_to_the_closest_step(self):
        assert nearest_gain_step(33.0, V4_GAIN_STEPS_DB) == 32.8

    def test_a_request_above_the_table_lands_on_the_maximum(self):
        assert nearest_gain_step(100.0, V4_GAIN_STEPS_DB) == 49.6

    def test_a_request_below_the_table_lands_on_the_minimum(self):
        assert nearest_gain_step(0.0, V4_GAIN_STEPS_DB) == 0.0

    def test_an_exact_tie_resolves_downward(self):
        # Midway between 0.9 and 1.4. Quiet costs signal-to-noise; loud
        # can overload the ADC and smear the whole spectrum, so the
        # cheaper mistake wins.
        assert nearest_gain_step(1.15, V4_GAIN_STEPS_DB) == 0.9

    def test_order_of_the_supported_table_does_not_matter(self):
        shuffled = tuple(reversed(V4_GAIN_STEPS_DB))

        assert nearest_gain_step(33.0, shuffled) == 32.8

    def test_an_empty_table_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="no supported gain steps"):
            nearest_gain_step(32.8, ())
