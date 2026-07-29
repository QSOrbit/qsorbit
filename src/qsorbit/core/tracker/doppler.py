"""Doppler shift for a satellite's transmit frequency.

A satellite moving relative to a ground observer shifts the frequency
the observer receives, away from whatever frequency the satellite
actually transmitted at. This module computes that shift.
"""

from __future__ import annotations

#: Speed of light in a vacuum, in km/s. Exact by SI definition — this
#: isn't a measured/approximated value, it's how the meter is defined.
SPEED_OF_LIGHT_KM_S = 299_792.458


def doppler_shifted_frequency(transmit_frequency_hz: float, range_rate_km_s: float) -> float:
    """Compute the frequency an observer receives from a moving satellite.

    Uses the classical (non-relativistic) Doppler formula. At typical
    satellite velocities (a few km/s — far below the speed of light),
    the relativistic correction is many orders of magnitude smaller
    than any amateur receiver's tuning precision, so it isn't worth the
    added complexity here.

    Args:
        transmit_frequency_hz: The frequency the satellite transmits
            at, in Hz.
        range_rate_km_s: Rate of change of the distance between
            observer and satellite, in km/s — see
            :attr:`~qsorbit.core.tracker.state.TopocentricState.range_rate_km_s`.
            Positive means the satellite is receding (the observed
            frequency comes out lower than transmitted); negative means
            it's approaching (observed frequency comes out higher).

    Returns:
        The frequency to tune a receiver to, in Hz.
    """
    return transmit_frequency_hz * (1.0 - range_rate_km_s / SPEED_OF_LIGHT_KM_S)
