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

Pass prediction (:mod:`qsorbit.core.tracker.pass_prediction`) and the
closed-form Sun position it can optionally use for an illumination flag
(:mod:`qsorbit.core.tracker.sun`) landed in Chunk B, re-exported here on
the same convention.
"""

from qsorbit.core.doppler import (
    SPEED_OF_LIGHT_KM_S,
    doppler_shifted_frequency,
    downlink_receive_frequency,
    uplink_transmit_frequency,
)
from qsorbit.core.tracker.exceptions import PropagationError, TleError, TrackerError
from qsorbit.core.tracker.observer import ObserverLocation
from qsorbit.core.tracker.pass_prediction import (
    DEFAULT_STEP_S,
    DEFAULT_TWILIGHT_SUN_ELEVATION_DEG,
    Pass,
    PassEvent,
    predict_passes,
)
from qsorbit.core.tracker.satellite import Satellite
from qsorbit.core.tracker.state import EciState, TopocentricState
from qsorbit.core.tracker.sun import is_illuminated, sun_elevation_deg, sun_gcrs_km
from qsorbit.core.tracker.target import Target

__all__ = [
    "DEFAULT_STEP_S",
    "DEFAULT_TWILIGHT_SUN_ELEVATION_DEG",
    "SPEED_OF_LIGHT_KM_S",
    "EciState",
    "ObserverLocation",
    "Pass",
    "PassEvent",
    "PropagationError",
    "Satellite",
    "Target",
    "TleError",
    "TopocentricState",
    "TrackerError",
    "doppler_shifted_frequency",
    "downlink_receive_frequency",
    "is_illuminated",
    "predict_passes",
    "sun_elevation_deg",
    "sun_gcrs_km",
    "uplink_transmit_frequency",
]
