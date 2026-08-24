"""Unit tests for Doppler shift calculation.

This is pure arithmetic (f_observed = f_transmit * (1 - range_rate/c)),
so rather than needing an external reference, the test cases use round
fractions of the speed of light — chosen specifically so the expected
result is exact and easy to hand-verify, not something that has to be
trusted from a calculator.

Moved here from ``tests/unit/tracker/`` in Chunk G, following the module
it covers: the arithmetic now lives at :mod:`qsorbit.core.doppler` so that
``core/dsp/`` can use it without importing skyfield. See that module's
docstring.
"""

from __future__ import annotations

import pytest

from qsorbit.core.doppler import (
    SPEED_OF_LIGHT_KM_S,
    doppler_shifted_frequency,
    downlink_receive_frequency,
    uplink_transmit_frequency,
)


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


class TestStillReachableFromTheTrackerPackage:
    """The move to ``core/doppler.py`` must not break existing imports.

    ``core/tracker/`` re-exports everything, so code (and the CLI) that
    imports from there keeps working. Worth a test rather than a promise,
    since the whole point of the re-export is that nobody has to notice
    the move.
    """

    def test_the_tracker_package_still_exports_the_primitive(self):
        from qsorbit.core.tracker import SPEED_OF_LIGHT_KM_S as tracker_c
        from qsorbit.core.tracker import doppler_shifted_frequency as tracker_fn

        assert tracker_c == SPEED_OF_LIGHT_KM_S
        assert tracker_fn is doppler_shifted_frequency

    def test_the_tracker_package_also_exports_the_named_wrappers(self):
        from qsorbit.core.tracker import downlink_receive_frequency as tracker_downlink

        assert tracker_downlink is downlink_receive_frequency


class TestDownlinkReceiveFrequency:
    """The downlink wrapper: satellite transmits, we listen.

    These tests exist because the *direction* is what gets flipped, not
    the arithmetic. Each one asserts an inequality about which way the
    frequency moved, not just a number — a sign flip changes the
    inequality and leaves the magnitude identical.
    """

    def test_a_receding_satellite_must_be_tuned_lower(self):
        tuned = downlink_receive_frequency(435_000_000.0, range_rate_km_s=7.0)

        assert tuned < 435_000_000.0

    def test_an_approaching_satellite_must_be_tuned_higher(self):
        tuned = downlink_receive_frequency(435_000_000.0, range_rate_km_s=-7.0)

        assert tuned > 435_000_000.0

    def test_it_agrees_with_the_primitive_it_wraps(self):
        # The wrapper adds a name and a direction, not a different
        # calculation. If these ever diverge, one of them is wrong.
        assert downlink_receive_frequency(145_950_000.0, 3.5) == doppler_shifted_frequency(
            145_950_000.0, 3.5
        )

    def test_the_shift_is_symmetric_about_zero_range_rate(self):
        nominal = 435_000_000.0
        up = downlink_receive_frequency(nominal, -7.0) - nominal
        down = nominal - downlink_receive_frequency(nominal, 7.0)

        assert up == pytest.approx(down)


class TestUplinkTransmitFrequencyIsReservedNotImplemented:
    """The uplink name is claimed so nobody derives it wrongly.

    The trap it guards against is specific: the uplink correction is the
    *reciprocal* of the downlink one, not its negation, and at LEO
    velocities the two agree closely enough that a wrong implementation
    would pass any tolerance test anyone thought to write.
    """

    def test_it_raises_rather_than_guessing(self):
        with pytest.raises(NotImplementedError):
            uplink_transmit_frequency(145_950_000.0, 7.0)

    def test_the_message_names_the_trap(self):
        # The reason this raises is more useful than the fact that it
        # raises, so the message has to carry it.
        with pytest.raises(NotImplementedError, match="sign"):
            uplink_transmit_frequency(145_950_000.0, 7.0)

    def test_the_reciprocal_and_the_negation_differ_but_only_barely(self):
        # Documents why a wrong uplink implementation would be invisible:
        # at 7 km/s the two candidate formulas agree to about one part in
        # 10^10, so no realistic assertion would separate them. This is
        # the evidence for reserving the name rather than trusting a
        # future reader to derive it.
        nominal, rr = 435_000_000.0, 7.0
        beta = rr / SPEED_OF_LIGHT_KM_S
        reciprocal = nominal / (1.0 - beta)
        negation = nominal * (1.0 + beta)

        assert reciprocal != negation
        assert abs(reciprocal - negation) / nominal < 1e-9
