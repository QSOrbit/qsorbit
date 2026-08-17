"""Turns sky positions into rotor pointing commands.

This is the layer between "where the satellite is" and "what the rotor
gets told to do." Those two things are numerically identical today,
which is exactly why the seam needs to exist now: everything that will
eventually make them differ is hardware-specific, and none of it belongs
inside the tracker.

What is expected to land here:

* **Alignment calibration offset** — the correction measured by the
  auto-calibration routine, applied to every commanded position.
* **Flip-mode selection** — for a pass crossing near zenith, commanding
  ``azimuth + 180`` and ``180 - elevation`` instead of whipping the
  azimuth axis 180° through the top. Only available on rotors whose
  elevation travel exceeds 90°.
* **Azimuth unwrapping** — on a rotor with more than 360° of azimuth
  travel, choosing ``azimuth`` or ``azimuth + 360`` so a pass crossing
  north doesn't hit a cable-wrap stop partway through.
* **Travel limits and obstruction masks** — refusing or clamping
  positions the hardware can't reach or the horizon blocks.
* **A movement deadband** — not issuing a command for a fraction of a
  degree of change, to avoid chattering the motors.

Note that flip selection and azimuth unwrapping cannot be decided from a
single instant: they depend on the shape of the whole pass, and on where
the rotor already is. So this module is expected to grow a stateful
planner that runs once before a pass begins and a cheap per-sample
converter that follows the resulting plan. Both of those need pass
prediction (AOS/LOS), which Phase 1 defers, so today only the seam
exists — :func:`sky_to_rotor` is deliberately an identity conversion.
"""

from __future__ import annotations

from datetime import datetime

from qsorbit.core.geometry import AzEl
from qsorbit.core.rotor import Position, format_set_position
from qsorbit.core.tracker import ObserverLocation, Target


def sky_to_rotor(sky: AzEl) -> Position:
    """Convert a sky direction into a rotor command position.

    .. note::

       **No correction is currently applied.** This is a straight
       one-to-one conversion, so the rotor is commanded to the raw
       computed sky position. Any mechanical misalignment of your mast
       shows up directly as pointing error, and a pass crossing near
       zenith will swing the azimuth axis through the top rather than
       flipping. Interfaces that expose pointing to an operator should
       say so rather than implying the output is calibrated.

    It is a named function rather than an inline construction so that
    there is exactly one place for alignment offsets, flip mode,
    unwrapping, and limit handling to be added — and so no caller can
    reach the rotor without passing through it. See this module's
    docstring for what's coming.

    Args:
        sky: Where the target actually is, as seen from the observer.

    Returns:
        The position to command the rotor to.

    Raises:
        ValueError: If the resulting position is outside what a
            :class:`~qsorbit.core.rotor.Position` permits.
    """
    return Position(azimuth=sky.azimuth, elevation=sky.elevation)


def compute_pointing_command(target: Target, observer: ObserverLocation, time: datetime) -> bytes:
    """Compute the EasyComm command to point a rotor at ``target``.

    Args:
        target: What to point at — anything satisfying
            :class:`~qsorbit.core.tracker.Target`, which today means a
            :class:`~qsorbit.core.tracker.Satellite`.
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
    state = target.topocentric_state(observer, time)
    return format_set_position(sky_to_rotor(state.sky_position))
