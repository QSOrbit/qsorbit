"""Satellite tracking module.

Loads two-line element sets (TLEs) and propagates satellite position
and velocity using skyfield's SGP4 implementation. Also computes
topocentric (observer-relative) az/el and Doppler shift.

The Doppler arithmetic itself moved to :mod:`qsorbit.core.doppler` in
Chunk G and is re-exported here, so existing imports from this package
are unchanged. It moved because importing anything under this package
executes this file, which imports skyfield — and ``core/dsp/`` needs the
Doppler functions while deliberately depending on nothing heavier than
numpy and scipy. See that module's docstring.
"""

from qsorbit.core.doppler import (
    SPEED_OF_LIGHT_KM_S,
    doppler_shifted_frequency,
    downlink_receive_frequency,
    uplink_transmit_frequency,
)
from qsorbit.core.tracker.exceptions import PropagationError, TleError, TrackerError
from qsorbit.core.tracker.observer import ObserverLocation
from qsorbit.core.tracker.satellite import Satellite
from qsorbit.core.tracker.state import EciState, TopocentricState
from qsorbit.core.tracker.target import Target

__all__ = [
    "SPEED_OF_LIGHT_KM_S",
    "EciState",
    "ObserverLocation",
    "PropagationError",
    "Satellite",
    "Target",
    "TleError",
    "TopocentricState",
    "TrackerError",
    "doppler_shifted_frequency",
    "downlink_receive_frequency",
    "uplink_transmit_frequency",
]
