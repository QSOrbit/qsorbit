"""EasyComm II protocol: command formatting and response parsing.

EasyComm II is a simple line-oriented text protocol spoken by many rotor
controllers, including the SatNOGS Arduino firmware. Commands are
space-separated tokens terminated by a newline::

    AZ180.0 EL45.0\\n    # move to azimuth 180, elevation 45
    AZ EL\\n             # query current position
    SA SE\\n             # stop azimuth and elevation motion

The rotor answers a position query with a line in the same shape as the
move command, e.g. ``AZ180.0 EL45.0\\n``.

This module is deliberately stateless: it only translates between
:class:`~qsorbit.core.rotor.position.Position` values and protocol bytes.
Connection state lives in :class:`~qsorbit.core.rotor.serial_port.SerialPort`.
"""

from __future__ import annotations

import re

from qsorbit.core.rotor.exceptions import ProtocolError
from qsorbit.core.rotor.position import Position

#: Line terminator for outgoing commands.
TERMINATOR = b"\n"

# Accepts e.g. b"AZ180.0 EL45.0", tolerating leading/trailing whitespace,
# \r\n terminators, integer or decimal angles, and an optional leading sign.
_POSITION_RE = re.compile(rb"^\s*AZ(?P<az>[-+]?\d+(?:\.\d+)?)\s+EL(?P<el>[-+]?\d+(?:\.\d+)?)\s*$")


def format_set_position(position: Position) -> bytes:
    """Format a command to move the rotor to ``position``.

    Angles are sent with one decimal place, matching the precision used
    by the SatNOGS firmware.

    Args:
        position: Target position.

    Returns:
        Command bytes, e.g. ``b"AZ180.0 EL45.0\\n"``.
    """
    return f"AZ{position.azimuth:.1f} EL{position.elevation:.1f}".encode("ascii") + TERMINATOR


def format_get_position() -> bytes:
    """Format a query for the rotor's current position.

    Returns:
        Command bytes: ``b"AZ EL\\n"``.
    """
    return b"AZ EL" + TERMINATOR


def format_stop() -> bytes:
    """Format a command to stop all rotor motion.

    Returns:
        Command bytes: ``b"SA SE\\n"``.
    """
    return b"SA SE" + TERMINATOR


def parse_position(line: bytes) -> Position:
    """Parse a rotor position report into a :class:`Position`.

    Args:
        line: A response line from the rotor, e.g. ``b"AZ180.0 EL45.0\\r\\n"``.
            Surrounding whitespace and line terminators are tolerated.

    Note that a homed rotor legitimately reports small negative angles —
    ``b"AZ-1.5 EL2.0\\n"`` is a normal reading, not an error — because a
    :class:`~qsorbit.core.rotor.position.Position` is a mechanical axis
    angle rather than a compass bearing.

    Returns:
        The reported position.

    Raises:
        ProtocolError: If the line does not match the expected format, or
            if it parses but the angles couldn't have come from a working
            rotor (non-finite, or beyond
            :data:`~qsorbit.core.rotor.position.MAX_AXIS_DEGREES`) — which
            means the rotor sent nonsense, not that the caller made a
            programming error.
    """
    match = _POSITION_RE.match(line)
    if match is None:
        raise ProtocolError(f"Could not parse rotor response: {line!r}")
    azimuth = float(match.group("az"))
    elevation = float(match.group("el"))
    try:
        return Position(azimuth=azimuth, elevation=elevation)
    except ValueError as exc:
        raise ProtocolError(f"Rotor reported an out-of-range position: {line!r} ({exc})") from exc
