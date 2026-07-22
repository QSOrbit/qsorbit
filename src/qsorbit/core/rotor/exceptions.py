"""Exceptions for the rotor module."""


class RotorError(Exception):
    """Base exception for all rotor-related errors."""


class SerialConnectionError(RotorError):
    """Raised when the serial connection cannot be established or is lost."""


class SerialTimeoutError(RotorError):
    """Raised when a serial read operation times out."""


class ProtocolError(RotorError):
    """Raised when a rotor response cannot be parsed or is out of range."""
