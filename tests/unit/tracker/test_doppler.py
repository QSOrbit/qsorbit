"""Unit tests for Doppler shift calculation.

This is pure arithmetic (f_observed = f_transmit * (1 - range_rate/c)),
so rather than needing an external reference, the test cases use round
fractions of the speed of light — chosen specifically so the expected
result is exact and easy to hand-verify, not something that has to be
trusted from a calculator.
"""

from __future__ import annotations

import pytest

from qsorbit.core.tracker import SPEED_OF_LIGHT_KM_S, doppler_shifted_frequency


class TestDopplerShiftedFrequency:
    def test_zero_range_rate_means_no_shift(self):
        assert doppler_shifted_frequency(145_900_000.0, 0.0) == 145_900_000.0

    def test_receding_satellite_shifts_frequency_down(self):
        # range_rate = +0.001c exactly -> factor (1 - 0.001) = 0.999
        range_rate = 0.001 * SPEED_OF_LIGHT_KM_S
        observed = doppler_shifted_frequency(100_000_000.0, range_rate)
        assert observed == pytest.approx(99_900_000.0)
        assert observed < 100_000_000.0

    def test_approaching_satellite_shifts_frequency_up(self):
        # range_rate = -0.001c exactly -> factor (1 + 0.001) = 1.001
        range_rate = -0.001 * SPEED_OF_LIGHT_KM_S
        observed = doppler_shifted_frequency(100_000_000.0, range_rate)
        assert observed == pytest.approx(100_100_000.0)
        assert observed > 100_000_000.0

    def test_realistic_leo_range_rate_is_a_small_fraction_of_a_percent(self):
        # A typical LEO max range rate is a few km/s -- for reference,
        # the ISS at zenith pass has a range rate around 7 km/s, which
        # is about 0.0023% of c. This just confirms the shift stays in
        # a physically sensible, small range for realistic inputs,
        # rather than testing a specific published number.
        transmit_frequency_hz = 145_900_000.0  # a real 2m-band satellite downlink
        observed = doppler_shifted_frequency(transmit_frequency_hz, range_rate_km_s=7.0)
        shift_hz = transmit_frequency_hz - observed
        assert 0.0 < shift_hz < 10_000.0
