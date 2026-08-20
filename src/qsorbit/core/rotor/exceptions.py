"""Exceptions for the rotor module."""


class RotorError(Exception):
    """Base exception for all rotor-related errors."""


class SerialConnectionError(RotorError):
    """Raised when the serial connection cannot be established or is lost."""


class SerialTimeoutError(RotorError):
    """Raised when a serial read operation times out."""


class ProtocolError(RotorError):
    """Raised when a rotor response cannot be parsed or is out of range."""


class HomingError(RotorError):
    """Raised when the rotator reports a latched homing failure.

    This one is distinct from every other error because of how it
    behaves: the firmware's error handler explicitly refuses to clear
    it, so nothing sent over the serial link recovers it — not
    ``RESET``, not ``RB``. Only a power cycle will.

    It is also the one error that invalidates everything else. A rotor
    that failed to home has no valid zero, so every position it reports
    afterwards is measured from nowhere. That is why connecting raises
    on this and merely reports the others.
    """


class PositionLimitError(RotorError):
    """Raised when a commanded position is outside a rotor's declared travel.

    This is a refusal, not a report: the command was **not** sent. It is
    the only guard that exists — the SatNOGS firmware applies no limits
    to a setpoint at the command handler and none in the control loop
    either, so a position that gets past this check gets attempted, no
    matter what it says.

    Distinct from :class:`ProtocolError`, which means the *rotor* said
    something unparseable, and from the ``ValueError`` raised by
    :class:`~qsorbit.core.rotor.position.Position`, which means a number
    turned up that no working rotor could produce. This one means the
    number is real and well-formed, but this particular hardware must
    not be sent there.
    """
