"""Rotor control module.

Serial communication and the SatNOGS rotator protocol, plus the
per-rotor capability record that declares what a given rotator may
safely be commanded to do.

The protocol module is named for the firmware rather than for the
standard it cites: the SatNOGS controller calls itself EasyComm 3 but
implements a subset with SatNOGS-specific extensions. See
:mod:`qsorbit.core.rotor.satnogs`.
"""

from qsorbit.core.rotor.capabilities import AzimuthWrap, RotorCapabilities
from qsorbit.core.rotor.controller import Arrival, Rotor, RotorStatus
from qsorbit.core.rotor.exceptions import (
    HomingError,
    PositionLimitError,
    ProtocolError,
    RotorError,
    SerialConnectionError,
    SerialTimeoutError,
)
from qsorbit.core.rotor.position import MAX_AXIS_DEGREES, Position
from qsorbit.core.rotor.satnogs import (
    Command,
    RotorErrorCode,
    format_get_error,
    format_get_position,
    format_get_version,
    format_set_position,
    format_stop,
    parse_error,
    parse_position,
    parse_version,
)
from qsorbit.core.rotor.serial_port import SerialPort

__all__ = [
    "MAX_AXIS_DEGREES",
    "Arrival",
    "AzimuthWrap",
    "Command",
    "HomingError",
    "Position",
    "PositionLimitError",
    "ProtocolError",
    "Rotor",
    "RotorCapabilities",
    "RotorError",
    "RotorErrorCode",
    "RotorStatus",
    "SerialConnectionError",
    "SerialPort",
    "SerialTimeoutError",
    "format_get_error",
    "format_get_position",
    "format_get_version",
    "format_set_position",
    "format_stop",
    "parse_error",
    "parse_position",
    "parse_version",
]
