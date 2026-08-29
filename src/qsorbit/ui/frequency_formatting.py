"""Turning a tracked frequency into the words a Radio tab shows.

Pure functions, no Qt, matching
:mod:`qsorbit.ui.readout_formatting` and
:mod:`qsorbit.ui.quieting_formatting`: the decisions about what a
display *says* are the part worth testing, and they should not need a
window to check. The widget that shows this does nothing but move
strings into labels.

**The split into a big number and a small one is not decoration.** The
mockup shows ``435.605`` at readout size with ``.000 MHz`` small beside
it, and the break is at the kilohertz. That is where the eye needs it:
during a pass the Doppler moves the last three digits continuously and
the first six barely at all, so a single run of nine changing digits is
unreadable at a glance while a steady ``435.605`` with a blur after it
is not. Same reasoning as the waterfall's fixed brightness scale --
a display should hold still wherever holding still is honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: Shown in place of a frequency before the first tracking sample lands.
#: An em dash rather than ``0.000`` or a blank: a zero is a number
#: somebody can believe, and this project's readouts go out of their way
#: not to show a plausible value that is not a measurement.
NO_READING: Final = "-"

#: What the doppler line says before there is anything to say. The
#: distinction between "no correction yet" and "no correction needed"
#: matters at the bench -- a geostationary target legitimately shows a
#: correction of zero, and that must not look like a tracker that has
#: not started.
AWAITING_LABEL: Final = "awaiting first tracking sample"

#: Below this many Hz the shift is reported as zero rather than with a
#: sign. One hertz on a 435 MHz downlink is 2.3 parts per billion; a
#: sign on it would be noise wearing a direction.
ZERO_SHIFT_HZ: Final = 1.0

_RISING: Final = "▲"
_FALLING: Final = "▼"


@dataclass(frozen=True)
class FrequencyText:
    """Everything the frequency card displays, already decided.

    Args:
        megahertz: The steady part, in MHz to three decimals -- e.g.
            ``"435.605"``. :data:`NO_READING` when nothing has arrived.
        hertz: The moving part and its unit -- e.g. ``".000 MHz"``.
            Empty when there is no reading, so the placeholder is not
            followed by a stray unit.
        doppler: The correction line, e.g.
            ``"▲ +4,213 Hz doppler - corrected"``.
        role: Which ``role`` property the doppler line should wear, so
            the widget picks a theme token by name rather than a colour:
            ``"accent"`` once there is a reading, ``"dim"`` while there
            is not.
    """

    megahertz: str
    hertz: str
    doppler: str
    role: str


def frequency_text(live_hz: float | None, nominal_hz: float | None = None) -> FrequencyText:
    """Describe the tracked downlink for display.

    Args:
        live_hz: Where the downlink actually is right now -- the tuner's
            own centre plus the current Doppler offset, as
            :attr:`~qsorbit.core.receive.ReceiveSession.live_tracked_frequency_hz`
            reports it. ``None`` before the tracker has ever been fed,
            which is the honest state rather than an error.
        nominal_hz: The transmitter's rest frequency, from the
            satellite's profile. The difference between the two is the
            Doppler shift. ``None`` when the caller does not know it --
            a replayed capture, a bench run on an ISM frequency -- in
            which case the frequency is shown and the shift is not
            claimed.

    Returns:
        The strings and the role to show them in.
    """
    if live_hz is None:
        return FrequencyText(megahertz=NO_READING, hertz="", doppler=AWAITING_LABEL, role="dim")

    megahertz, hertz = _split_megahertz(live_hz)

    if nominal_hz is None:
        return FrequencyText(
            megahertz=megahertz,
            hertz=hertz,
            doppler="no nominal frequency, so no shift is claimed",
            role="dim",
        )

    shift_hz = live_hz - nominal_hz
    if abs(shift_hz) < ZERO_SHIFT_HZ:
        # Deliberately still says "corrected". A zero shift on a
        # geostationary target is the correction working, and the
        # station convention (Session 27) is to prove the chain against
        # exactly that case -- so this line has to read as success.
        return FrequencyText(
            megahertz=megahertz, hertz=hertz, doppler="0 Hz doppler - corrected", role="accent"
        )

    arrow = _RISING if shift_hz > 0 else _FALLING
    return FrequencyText(
        megahertz=megahertz,
        hertz=hertz,
        doppler=f"{arrow} {shift_hz:+,.0f} Hz doppler - corrected",
        role="accent",
    )


def _split_megahertz(hz: float) -> tuple[str, str]:
    """Split a frequency into its steady MHz part and its moving Hz part.

    Rounded to the hertz **before** splitting, not after. Formatting the
    two halves independently lets ``435.605999`` round up to ``435.606``
    in the first field while the second still reads ``.999``, which is a
    frequency that was never true and is off by a kilohertz in the digit
    an operator is watching.
    """
    total_hz = round(hz)
    megahertz, remainder = divmod(abs(total_hz), 1_000_000)
    kilohertz, hertz = divmod(remainder, 1_000)
    sign = "-" if total_hz < 0 else ""
    return f"{sign}{megahertz:,}.{kilohertz:03d}", f".{hertz:03d} MHz"
