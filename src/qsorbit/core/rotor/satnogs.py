"""SatNOGS rotator protocol: command formatting and response parsing.

The SatNOGS Arduino firmware describes itself as *"easycomm 3 protocol as
referred, in Hamlib"*, but what it actually implements is a
SatNOGS-specific subset with extensions of its own — ``RESET``, ``PARK``,
``RB``/``RST``, the ``IP0``-``IP8`` telemetry indices, and the ``CR``/``CW``
gain accessors are not EasyComm. It also does no bounds checking at all.
This module is therefore named for the firmware it actually talks to
rather than for the standard it cites, leaving room for a genuine
``easycomm3`` backend later.

Verified against **stock SatNOGS firmware v2.2.1**, DC motor build, by
reading ``easycomm.h`` and ``dc_motor_controller.ino`` and confirming on
hardware. See ``qsorbit-rotor-integration.md`` for the full command
reference and the rules this module exists to enforce.

Commands are line-oriented ASCII terminated by a newline::

    AZ180.0 EL45.0\\n    # move to azimuth 180, elevation 45
    AZ EL\\n             # query current position
    VE\\n                # query firmware version
    GE\\n                # query the latched error state
    SA SE\\n             # stop (converging motion only)

Two firmware behaviours shape this module's API:

**Never emit a lone axis.** On v2.2 and earlier a bare ``AZ`` falls
through to the setpoint branch, where ``isNumber("")`` returns true and
``atof("")`` is ``0.0`` — so the rotor *silently slews azimuth to zero* —
and then ``strtok_r`` returns ``NULL`` and the next line dereferences it.
``AZ10.0`` with no ``EL`` token hits the same null dereference. v2.2.1
fixed the query case, but the pair is the only form that is safe across
every firmware in the wild, so there is deliberately no way to construct
a single-axis command here.

**Some commands reply and some don't.** ``SA SE`` answers with a position
report; a set-position command answers with nothing. Writing a command
without knowing which it is desynchronizes every subsequent read — the
stop's unread reply gets returned as the answer to the *next* query. Every
formatter therefore returns a :class:`Command` carrying that fact, so a
caller cannot write bytes to the port without also knowing whether to read.

This module is stateless: it only translates between values and protocol
bytes. Connection state lives in
:class:`~qsorbit.core.rotor.serial_port.SerialPort`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum

from qsorbit.core.rotor.exceptions import ProtocolError
from qsorbit.core.rotor.position import Position

#: Line terminator for outgoing commands.
TERMINATOR = b"\n"

# Accepts e.g. b"AZ180.0 EL45.0", tolerating leading/trailing whitespace,
# \r\n terminators, integer or decimal angles, and an optional leading sign.
_POSITION_RE = re.compile(rb"^\s*AZ(?P<az>[-+]?\d+(?:\.\d+)?)\s+EL(?P<el>[-+]?\d+(?:\.\d+)?)\s*$")

# Accepts e.g. b"VESatNOGS-v2.2.1". The firmware concatenates the version
# straight onto the "VE" prefix with no separator.
_VERSION_RE = re.compile(rb"^\s*VE(?P<version>\S+)\s*$")

# Accepts e.g. b"GE1".
_ERROR_RE = re.compile(rb"^\s*GE(?P<code>\d+)\s*$")


class RotorErrorCode(IntEnum):
    """The error state reported by ``GE``.

    These look like bit flags, and the firmware's own comments treat them
    that way, but it assigns them as plain enum values — note that
    ``OVER_TEMPERATURE`` is 12, which is not a power of two and would
    read as ``HOMING_ERROR | MOTOR_ERROR`` if masked. **Compare for
    equality, never with a bitmask.**

    :attr:`HOMING_ERROR` is the one that matters operationally: the
    firmware's error handler explicitly refuses to clear it, so nothing
    sent over the serial link will recover it — not ``RESET``, not
    ``RB``. It latches until the controller is power-cycled, and should
    be surfaced to the operator as exactly that instruction rather than
    as a generic failure or something to retry.
    """

    NO_ERROR = 1
    SENSOR_ERROR = 2
    HOMING_ERROR = 4
    MOTOR_ERROR = 8
    OVER_TEMPERATURE = 12
    WDT_ERROR = 16


@dataclass(frozen=True)
class Command:
    """A protocol command, together with whether the rotor answers it.

    ``Command`` is a value object: immutable and comparable by value.
    Use :func:`bytes` to get the wire bytes.

    Args:
        data: The bytes to write, including the terminator.
        expects_reply: Whether the firmware sends a response line. A
            caller that writes ``data`` must read exactly one line when
            this is ``True`` and read nothing when it is ``False``.
            Getting it wrong doesn't fail loudly — it shifts every
            later read by one message.
    """

    data: bytes
    expects_reply: bool

    def __bytes__(self) -> bytes:
        return self.data


def format_set_position(position: Position) -> Command:
    """Format a command to move the rotor to ``position``.

    Both axes are always sent, because a lone axis is unsafe on older
    firmware — see this module's docstring. Angles are sent with one
    decimal place, matching the precision the firmware parses.

    .. warning::

       The firmware applies **no limits to a commanded setpoint** —
       neither at the command handler nor anywhere in the control loop —
       so it will happily accept ``AZ9999`` and drive toward it until
       someone cuts power. Range-check against the rotor's declared
       :class:`~qsorbit.core.rotor.capabilities.RotorCapabilities` before
       calling this.

    Args:
        position: Target position.

    Returns:
        The command, e.g. ``b"AZ180.0 EL45.0\\n"``. The firmware sends no
        reply to a set-position command.
    """
    data = f"AZ{position.azimuth:.1f} EL{position.elevation:.1f}".encode("ascii") + TERMINATOR
    return Command(data=data, expects_reply=False)


def format_get_position() -> Command:
    """Format a query for the rotor's current position.

    Both axes come back in one round trip, which is why this is the
    workhorse query rather than two single-axis reads.

    Returns:
        The command ``b"AZ EL\\n"``, which is answered with a position
        report.
    """
    return Command(data=b"AZ EL" + TERMINATOR, expects_reply=True)


def format_get_version() -> Command:
    """Format a query for the firmware version string.

    Returns:
        The command ``b"VE\\n"``, answered with e.g.
        ``b"VESatNOGS-v2.2.1\\n"``.
    """
    return Command(data=b"VE" + TERMINATOR, expects_reply=True)


def format_get_error() -> Command:
    """Format a query for the rotor's latched error state.

    Returns:
        The command ``b"GE\\n"``, answered with e.g. ``b"GE1\\n"``.
    """
    return Command(data=b"GE" + TERMINATOR, expects_reply=True)


def format_stop() -> Command:
    """Format a command to halt rotor motion.

    .. warning::

       This is **not an emergency stop** and must never be presented as
       one. It works by setting the setpoint to the current position,
       which halts a loop that is *converging*. It cannot stop a
       diverging one — a wrong-sign axis accelerates away and the error
       keeps growing on its own. The power switch is the real stop.

    Returns:
        The command ``b"SA SE\\n"``. The firmware answers it with a
        position report, which the caller must read.
    """
    return Command(data=b"SA SE" + TERMINATOR, expects_reply=True)


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


def parse_version(line: bytes) -> str:
    """Parse a ``VE`` reply into a firmware version string.

    Args:
        line: A response line, e.g. ``b"VESatNOGS-v2.2.1\\n"``.

    Returns:
        The version as reported, e.g. ``"SatNOGS-v2.2.1"``. It is
        returned verbatim rather than parsed into components: QSOrbit
        records what the rotor said and compares it to what the station
        config declares, but never infers capabilities from it. What a
        given rotor can safely be told to do is declared per rotor —
        see :class:`~qsorbit.core.rotor.capabilities.RotorCapabilities`.

    Raises:
        ProtocolError: If the line is not a version reply.
    """
    match = _VERSION_RE.match(line)
    if match is None:
        raise ProtocolError(f"Could not parse firmware version reply: {line!r}")
    return match.group("version").decode("ascii", errors="replace")


def parse_error(line: bytes) -> RotorErrorCode:
    """Parse a ``GE`` reply into a :class:`RotorErrorCode`.

    Args:
        line: A response line, e.g. ``b"GE1\\n"``.

    Returns:
        The reported error state. ``GE1`` means no error.

    Raises:
        ProtocolError: If the line is not an error reply, or reports a
            code this firmware generation doesn't define. An unknown
            code is surfaced rather than ignored — it means the rotor is
            running firmware QSOrbit hasn't been verified against, which
            the operator should know before commanding motion.
    """
    match = _ERROR_RE.match(line)
    if match is None:
        raise ProtocolError(f"Could not parse rotor error reply: {line!r}")
    code = int(match.group("code"))
    try:
        return RotorErrorCode(code)
    except ValueError as exc:
        raise ProtocolError(
            f"Rotor reported an unrecognized error code {code} in {line!r}. "
            "Expected one of: "
            + ", ".join(f"{member.value} ({member.name})" for member in RotorErrorCode)
        ) from exc
