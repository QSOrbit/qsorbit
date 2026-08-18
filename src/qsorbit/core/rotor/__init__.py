"""Rotor control module.

Provides serial communication and EasyComm II protocol support for
Arduino-controlled antenna rotors.
"""

from qsorbit.core.rotor.easycomm import (
    format_get_position,
    format_set_position,
    format_stop,
    parse_position,
)
from qsorbit.core.rotor.exceptions import (
    ProtocolError,
    RotorError,
    SerialConnectionError,
    SerialTimeoutError,
)
from qsorbit.core.rotor.position import MAX_AXIS_DEGREES, Position
from qsorbit.core.rotor.serial_port import SerialPort

__all__ = [
    "MAX_AXIS_DEGREES",
    "Position",
    "ProtocolError",
    "RotorError",
    "SerialConnectionError",
    "SerialTimeoutError",
    "SerialPort",
    "format_get_position",
    "format_set_position",
    "format_stop",
    "parse_position",
]
