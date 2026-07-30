"""Ties the tracker and rotor modules together into one pointing command.

Given a satellite, an observer, and a time, this computes where the
rotor needs to point and formats the EasyComm command that would send
it there. This is the first piece of QSOrbit that does something
end-to-end, however small — see Chunk E's notes in the Phase 1 brief.
"""

from __future__ import annotations

from datetime import datetime

from qsorbit.core.rotor import format_set_position
from qsorbit.core.tracker import ObserverLocation, Satellite


def compute_pointing_command(
    satellite: Satellite, observer: ObserverLocation, time: datetime
) -> bytes:
    """Compute the EasyComm command to point a rotor at ``satellite``.

    Args:
        satellite: The satellite to point at.
        observer: The ground station's location.
        time: The instant to compute for, as a timezone-aware datetime.

    Returns:
        The EasyComm command bytes to send to the rotor, e.g.
        ``b"AZ180.0 EL45.0\\n"``.

    Raises:
        ValueError: If ``time`` is naive (has no ``tzinfo``).
        PropagationError: If SGP4 cannot compute a valid position at
            ``time``.
    """
    state = satellite.topocentric_state(observer, time)
    return format_set_position(state.position)
