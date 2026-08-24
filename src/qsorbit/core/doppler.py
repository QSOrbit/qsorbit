"""Doppler shift: the frequency a moving satellite is heard at, and tuned to.

**Why this lives at ``core/`` rather than in ``core/tracker/``.** It began in
``core/tracker/doppler.py``, which is where the arithmetic belongs by
subject. It moved here in Chunk G because more than one layer needs it and
``core/tracker/`` cannot be imported cheaply: importing *any* module beneath
it executes ``core/tracker/__init__.py``, which imports ``observer.py``,
which imports skyfield at module scope. So a single import of this
arithmetic from :mod:`qsorbit.core.dsp` would have made the entire DSP
package require skyfield — and ``core/dsp/`` needing nothing beyond numpy
and scipy is what makes its tests the only ones runnable in every
environment this project is developed in.

This is the same shape as :mod:`qsorbit.core.geometry`, which holds
:class:`~qsorbit.core.geometry.AzEl` as pure types that both the tracker and
the pointing layer use: physics and value types with no dependencies, shared
by layers that do have dependencies. :mod:`qsorbit.core.tracker` re-exports
everything here, so ``from qsorbit.core.tracker import
doppler_shifted_frequency`` keeps working exactly as before.

**The three named functions exist because the signs are opposite and the
mistake is invisible.** This was flagged in Session 8 and deliberately left
until something consumed it:

- :func:`doppler_shifted_frequency` is the raw primitive. It answers "what
  frequency arrives, given what was sent" and knows nothing about which end
  of a link is moving.
- :func:`downlink_receive_frequency` is that question asked from the ground:
  the satellite transmits, we listen, so a **receding** satellite must be
  tuned **lower**.
- :func:`uplink_transmit_frequency` is the mirror, reserved and
  unimplemented: the satellite listens, so to be heard on its nominal
  receive frequency we must transmit **higher** when it is receding.

Getting one of those backwards produces a receiver tuned to twice the
Doppler error — a signal that is present, visible on a waterfall, and
inaudible, with no exception anywhere. Naming the direction in the function
means a caller cannot silently pick the wrong one, which calling the
primitive directly at each site absolutely would.
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

    **Prefer :func:`downlink_receive_frequency` at a call site.** This
    primitive is direction-agnostic on purpose, which is exactly what
    makes it easy to use in the wrong direction; the named wrappers exist
    so that choice is made once, here, rather than at every caller.

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


def downlink_receive_frequency(transmit_frequency_hz: float, range_rate_km_s: float) -> float:
    """The frequency to tune a **receiver** to for a satellite downlink.

    The satellite transmits at ``transmit_frequency_hz``; this returns what
    actually arrives at the ground, which is what a receiver must be tuned
    to. A receding satellite (positive ``range_rate_km_s``) arrives
    **lower**.

    Args:
        transmit_frequency_hz: The satellite's nominal downlink frequency,
            in Hz.
        range_rate_km_s: Range rate, positive when receding — the sign
            convention :class:`~qsorbit.core.tracker.state.TopocentricState`
            and :class:`~qsorbit.core.pointing.TrackSample` both use.

    Returns:
        The frequency the downlink is received at, in Hz.
    """
    return doppler_shifted_frequency(transmit_frequency_hz, range_rate_km_s)


def uplink_transmit_frequency(
    satellite_receive_frequency_hz: float, range_rate_km_s: float
) -> float:
    """The frequency to **transmit** on so a satellite hears its nominal uplink.

    **Reserved and deliberately unimplemented.** Transmit is out of scope
    for v1 (see ``project-notes.md``, "Key decisions"), so there is nothing
    to test this against and no bench to verify it on. The name is claimed
    here anyway, with the arithmetic written down, so that whoever needs it
    implements *this* function rather than reaching for
    :func:`doppler_shifted_frequency` and guessing a sign.

    The correction runs the opposite way to :func:`downlink_receive_frequency`:
    the satellite is the listener, so a **receding** satellite must be
    transmitted to **higher**, not lower. The exact form is::

        transmit_hz = satellite_receive_frequency_hz / (1 - range_rate/c)

    **Note what that is not.** The tempting shortcut — reuse the downlink
    function with the range rate negated — gives
    ``f * (1 + range_rate/c)``, which is only the first-order expansion of
    the expression above. The two agree to about one part in 10^10 at LEO
    velocities, so the shortcut would work and would still be wrong: it
    would encode "the inverse is a sign flip" as a fact, and that stops
    being true anywhere the velocity is a meaningful fraction of c. Write
    the division.

    Raises:
        NotImplementedError: Always. See above.
    """
    raise NotImplementedError(
        "Uplink pre-correction is not implemented: transmit is out of scope for v1. "
        "The name is reserved so that nobody derives it from "
        "doppler_shifted_frequency() and picks the wrong sign -- the uplink "
        "correction is the reciprocal of the downlink one, not its negation. "
        "See this function's docstring for the arithmetic."
    )
