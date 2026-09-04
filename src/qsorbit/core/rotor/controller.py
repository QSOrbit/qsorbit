"""The rotor facade — one connection, one place that talks to hardware.

Everything above this layer works in positions; everything below works
in bytes. :class:`Rotor` owns the serial port, the command/reply
sequencing, and the safety check that the firmware doesn't do.

Four firmware behaviours shape the design, all of them learned at the
bench rather than from documentation:

**One long-lived connection.** Phil's USB adapter has no DTR-to-reset
wiring, but most do — and on those, every port open resets the
controller and triggers a full re-home. A tool that opened and closed
the port per command would slew the rotor to its end-stops on every
invocation. So :class:`Rotor` opens once and holds it.

**Silence is not failure.** ``homing()`` blocks and never services the
serial link, so after power-up or ``RESET`` nothing answers for tens of
seconds — longer if azimuth starts far from its pin. Connecting treats
that as an explicit *homing* state and waits, concluding the link is
down only after the timeout expires.

**Every setpoint is range-checked before it reaches the wire.** The
firmware applies no limits at the command handler and none in the
control loop either. ``AZ9999`` is accepted and attempted. This class is
the only thing between a bad number and the hardware.

**Arrival is decided by position, never by status.** The firmware's idle
test includes a ``speed == 0`` term, so a stalled axis reports idle just
like an arrived one.

The class takes a :class:`~qsorbit.core.rotor.capabilities.RotorCapabilities`
rather than a station config, so it stays usable without a config file
and the dependency arrow runs one way:
:mod:`qsorbit.core.station` knows about the rotor, not the reverse.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from qsorbit.core.rotor.capabilities import RotorCapabilities
from qsorbit.core.rotor.exceptions import (
    GainVerificationError,
    HomingError,
    ProtocolError,
    SerialConnectionError,
    SerialTimeoutError,
)
from qsorbit.core.rotor.position import Position
from qsorbit.core.rotor.satnogs import (
    Command,
    GainRegister,
    RotorErrorCode,
    format_get_error,
    format_get_gain,
    format_get_position,
    format_get_version,
    format_set_gain,
    format_set_position,
    format_stop,
    parse_error,
    parse_gain,
    parse_position,
    parse_version,
)
from qsorbit.core.rotor.serial_port import SerialPort

#: How long :meth:`Rotor.connect` waits for a homing controller to start
#: answering, in seconds. Homing takes seconds to tens of seconds, and
#: much longer when azimuth starts far from its pin, so this is generous
#: on purpose: the failure mode of waiting too long is a slow start, and
#: the failure mode of giving up too early is an operator convinced
#: their cable is broken.
DEFAULT_HOMING_TIMEOUT_S = 120.0

#: Gap between ``VE`` probes while waiting for homing to finish.
DEFAULT_HOMING_POLL_INTERVAL_S = 2.0

#: Gap between position reads while waiting for a move to settle.
DEFAULT_ARRIVAL_POLL_INTERVAL_S = 0.5

#: How far a gain read-back may differ from what was written.
#:
#: Both the write and the reply carry two decimal places, so anything
#: beyond half of the last digit is a real disagreement rather than
#: rounding.
GAIN_TOLERANCE = 0.005


@dataclass(frozen=True)
class RotorStatus:
    """A snapshot of what the rotator reports about itself.

    Args:
        firmware_version: The string from ``VE``, e.g.
            ``"SatNOGS-v2.2.1"``.
        error: The state from ``GE``.
        position: The axis reading from ``AZ EL``, relative to the
            rotor's homed zero. This is a *mechanical* reading, not a
            compass bearing, and no calibration offset has been applied
            to it.
    """

    firmware_version: str
    error: RotorErrorCode
    position: Position

    @property
    def healthy(self) -> bool:
        """``True`` if the rotor reports no error at all."""
        return self.error is RotorErrorCode.NO_ERROR

    @property
    def homing_error_latched(self) -> bool:
        """``True`` if homing failed and only a power cycle will clear it."""
        return self.error is RotorErrorCode.HOMING_ERROR


@dataclass(frozen=True)
class Arrival:
    """The outcome of waiting for a commanded move to settle.

    Args:
        arrived: Whether both axes came within the acceptance window.
            ``False`` is not necessarily a fault — it may mean the move
            is simply still in progress, or that an axis stalled.
        position: The last position read.
        elapsed_s: How long the wait took.
    """

    arrived: bool
    position: Position
    elapsed_s: float


class Rotor:
    """A connected SatNOGS rotator.

    Supports use as a context manager, which connects on entry and
    closes on exit::

        with Rotor(SerialPort("COM5", 19200), capabilities) as rotor:
            print(rotor.read_position())

    Args:
        port: The serial port to talk over. Injected rather than
            constructed here so tests can pass a mock and never touch
            the OS serial layer.
        capabilities: What this rotator may safely be commanded to do.
            Every setpoint is checked against it before transmission.
        homing_timeout_s: How long :meth:`connect` waits for a homing
            controller to start answering.
        homing_poll_interval_s: Gap between ``VE`` probes while waiting.
        on_homing_wait: Optional callback, invoked with the seconds
            waited so far each time a probe goes unanswered. Lets a CLI
            or UI say "still homing (12s)" instead of appearing hung.
        sleep: Injected for testing; defaults to :func:`time.sleep`.
        monotonic: Injected for testing; defaults to
            :func:`time.monotonic`.
    """

    def __init__(
        self,
        port: SerialPort,
        capabilities: RotorCapabilities,
        *,
        homing_timeout_s: float = DEFAULT_HOMING_TIMEOUT_S,
        homing_poll_interval_s: float = DEFAULT_HOMING_POLL_INTERVAL_S,
        on_homing_wait: Callable[[float], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._port = port
        self._capabilities = capabilities
        self._homing_timeout_s = homing_timeout_s
        self._homing_poll_interval_s = homing_poll_interval_s
        self._on_homing_wait = on_homing_wait
        self._sleep = sleep
        self._monotonic = monotonic
        self._firmware_version: str | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> RotorCapabilities:
        """The declared limits this rotor is being held to."""
        return self._capabilities

    @property
    def firmware_version(self) -> str | None:
        """The version reported at connect, or ``None`` before connecting."""
        return self._firmware_version

    @property
    def is_connected(self) -> bool:
        """``True`` if the serial port is open."""
        return self._port.is_open

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> RotorStatus:
        """Open the link and establish what the rotor is doing.

        The sequence is: open the port, wait for the controller to start
        answering ``VE`` (it is deaf while homing), read ``GE``, then
        read the current position.

        Returns:
            The rotor's state at the end of the sequence.

        Raises:
            SerialConnectionError: If the port can't be opened, or if
                nothing answers before ``homing_timeout_s`` expires.
            HomingError: If the rotor reports a latched homing failure.
                This one stops everything, because a rotor that failed
                to home has no valid zero — every position it reports
                afterwards is measured from nowhere. Other error states
                are returned in the status rather than raised, so a
                caller can show them and decide.
            ProtocolError: If a reply can't be parsed.
        """
        self._port.open()
        self._firmware_version = self._await_version()
        error = self.read_error()
        if error is RotorErrorCode.HOMING_ERROR:
            raise HomingError(
                "The rotator reports a latched homing failure. Power-cycle the "
                "controller — nothing sent over the serial link will clear it, "
                "including RESET. The usual cause is benign: if azimuth starts "
                "more than 360 degrees of travel from its home pin, the firmware's "
                "homing check trips before it gets there."
            )
        return RotorStatus(
            firmware_version=self._firmware_version,
            error=error,
            position=self.read_position(),
        )

    def close(self) -> None:
        """Close the serial port. Safe to call more than once."""
        self._port.close()

    def __enter__(self) -> Rotor:
        """Connect on entering a ``with`` block."""
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Close the port on exit, even if an exception occurred.

        Note that closing does **not** stop the rotor. A move already in
        progress continues to its setpoint; the controller doesn't need
        the link to finish converging.
        """
        self.close()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def read_position(self) -> Position:
        """Read both axes in one round trip.

        Returns:
            The current mechanical axis reading. Negative values are
            normal on a freshly homed rotor.

        Raises:
            SerialConnectionError: If the port is not open.
            SerialTimeoutError: If the rotor doesn't answer.
            ProtocolError: If the reply can't be parsed.
        """
        return parse_position(self._exchange_expecting_reply(format_get_position()))

    def read_error(self) -> RotorErrorCode:
        """Read the rotor's error state.

        Returns:
            The reported error, which is :attr:`RotorErrorCode.NO_ERROR`
            in normal operation.

        Raises:
            SerialConnectionError: If the port is not open.
            SerialTimeoutError: If the rotor doesn't answer.
            ProtocolError: If the reply can't be parsed, or reports a
                code this firmware generation doesn't define.
        """
        return parse_error(self._exchange_expecting_reply(format_get_error()))

    def status(self) -> RotorStatus:
        """Read a fresh snapshot of error state and position.

        Unlike :meth:`connect`, this does not wait for homing and does
        not raise on a latched homing failure — it reports it, so a
        caller can display the state rather than handle an exception.

        Returns:
            The current status. The firmware version is the one recorded
            at connect.

        Raises:
            SerialConnectionError: If the rotor was never connected.
        """
        if self._firmware_version is None:
            raise SerialConnectionError(
                "Not connected — call connect() before reading status, so the "
                "firmware version is known and homing has completed."
            )
        return RotorStatus(
            firmware_version=self._firmware_version,
            error=self.read_error(),
            position=self.read_position(),
        )

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move_to(self, position: Position) -> None:
        """Command the rotor to ``position``.

        The position is range-checked against the declared capabilities
        **before** anything is written. Both axes are always sent
        together.

        This returns as soon as the command is written; the rotor keeps
        moving. Use :meth:`wait_for_arrival` to watch it settle.

        Args:
            position: Where to point, as a mechanical axis position.

        Raises:
            PositionLimitError: If the position is outside the declared
                travel. Nothing is transmitted in that case.
            SerialConnectionError: If the port is not open.
        """
        self._capabilities.check_setpoint(position)
        self._exchange(format_set_position(position))

    def stop(self) -> Position:
        """Halt a converging move by setting the setpoint to the present position.

        .. warning::

           This is **not an emergency stop** and must not be offered as
           one. It halts a loop that is converging; it cannot stop one
           that is diverging, because a wrong-sign axis accelerates away
           and the error grows on its own. The power switch is the real
           stop.

        Returns:
            The position the rotor reports as it stops — the firmware
            answers this command, and the reply must be read or it
            becomes the answer to the next query.

        Raises:
            SerialConnectionError: If the port is not open.
            ProtocolError: If the reply can't be parsed.
        """
        return parse_position(self._exchange_expecting_reply(format_stop()))

    def read_gain(self, register: GainRegister) -> float:
        """Read one PID gain register from the controller.

        Args:
            register: Which gain to read.

        Returns:
            The value the controller reports.

        Raises:
            SerialConnectionError: If the port is not open.
            SerialTimeoutError: If the rotor doesn't answer.
            ProtocolError: If the reply can't be parsed, or reports a
                different register than the one asked for.
        """
        return parse_gain(self._exchange_expecting_reply(format_get_gain(register)), register)

    def write_gain(self, register: GainRegister, value: float) -> None:
        """Write one PID gain register, **without verifying it landed**.

        The firmware sends no reply to a gain write, so nothing about
        this call can tell you whether it worked. Prefer
        :meth:`push_gains`, which reads every register back. This exists
        for the cases that genuinely want a blind write — a diagnostic
        tool, or a caller doing its own verification.

        Args:
            register: Which gain to write.
            value: The new gain.

        Raises:
            SerialConnectionError: If the port is not open.
        """
        self._exchange(format_set_gain(register, value))

    def read_gains(self) -> dict[GainRegister, float]:
        """Read every gain register, in register order.

        The read-only counterpart to :meth:`push_gains`, and it exists
        because "what is this controller actually running" is a question
        worth answering with a measurement rather than an assumption.

        **Gains are RAM-only and survive a disconnect.** A profile that
        writes nothing therefore does not mean the controller holds its
        compiled defaults -- it means it holds whatever was last written
        to it, which may be a previous profile, a half-applied set, or a
        bench tool's experiment. Reporting the compiled defaults in that
        situation is a claim nobody checked, and it was wrong on this
        station's own rotator for a whole 543-second track.

        Costs six serial round trips, about a second, so it belongs
        where a second is affordable -- at the start of a run, not in
        the middle of a pass.

        Returns:
            Every register and the value the controller reports for it.

        Raises:
            SerialConnectionError: If the port is not open.
            SerialTimeoutError: If the rotor doesn't answer a read.
            ProtocolError: If a reply can't be parsed, or reports a
                different register than the one asked for.
        """
        return {register: self.read_gain(register) for register in GainRegister}

    def push_gains(
        self,
        gains: Mapping[GainRegister, float],
    ) -> dict[GainRegister, float]:
        """Write a set of gains and read every one of them back.

        **Every register is verified, not a sample.** Gains are RAM-only
        and re-pushed at every connect (integration rule 2.12), so a
        write that silently fails leaves the rotor tracking on compiled
        defaults while the application believes it is running a tuned
        set — and every measurement taken afterwards is attributed to the
        wrong configuration. That is a worse outcome than not pushing
        gains at all, which is why this raises rather than warning.

        **Exactly one write per exchange, and that is a firmware
        requirement rather than caution.** This used to send every write
        first and then read them all back, to pay one settle wait for
        the set instead of one per register. That is not merely slower
        to recover from -- it silently loses writes. ``easycomm_proc()``
        drains the whole serial buffer in a single call and sets
        ``char *Data = buffer`` once, at function entry; the ``CW``
        handler parses with ``strtok_r(Data, ",", &Data)``, which
        *mutates* ``Data``. So the first ``CW`` in a drain finds its
        value at ``rawData + 4`` and applies, and every later ``CW`` in
        the same drain parses from a stale pointer, fails
        ``isNumber()``, and is skipped -- with no error, and no reply,
        because gain writes never reply. Measured on hardware
        2026-09-03: six writes sent as a burst applied four and dropped
        both ``Kd`` registers, and ``rotor-pid.py``, which sends one
        command per turnaround, had been writing the same registers
        successfully for months.

        Interleaving the read is what fixes it, and it fixes it **by
        construction rather than by a sleep tuned to a guess**: a read
        is a round trip, so the host physically cannot send the next
        ``CW`` until the firmware has replied to the previous ``CR``.
        One ``CW`` per drain, guaranteed by the protocol. ``CR`` itself
        is immune -- it uses no ``strtok_r``, only ``buffer[3]``.

        It is also *cheaper* than what it replaces: the same six reads
        happen either way, and the separate settle wait is gone.

        **On what a matching read-back does and does not prove.** It
        proves what this method exists for -- that the register holds
        the value asked for, which is what "the controller is running
        the gains I chose" means. It is **not** evidence that the write
        path is healthy, because a register that already held the target
        reads back correct whether or not the write landed. That is how
        the burst defect stayed invisible: ``Kp`` was pushed at the
        firmware's own compiled defaults, so its dropped writes were
        indistinguishable from successful ones, and only ``Ki`` and
        ``Kd`` -- which differ from stock -- could ever have shown it.
        The ordering above is what guards the mechanism; the read-back
        guards the outcome.

        Args:
            gains: The registers to write and the values to write them.

        Returns:
            What each register read back, which on success equals what
            was asked for.

        Raises:
            GainVerificationError: If any register disagrees with the
                value written to it. The message names every register
                that disagreed, not just the first — one wrong register
                and six wrong registers are different faults.
            SerialConnectionError: If the port is not open.
            SerialTimeoutError: If the rotor doesn't answer a read.
            ProtocolError: If a reply can't be parsed.
        """
        # Write, verify, write, verify -- never write, write, verify.
        # See the docstring: the read's round trip is what keeps two
        # writes out of one serial drain, and two writes in one drain
        # means the second is silently discarded by the firmware.
        readback: dict[GainRegister, float] = {}
        for register, value in gains.items():
            self.write_gain(register, value)
            readback[register] = self.read_gain(register)
        wrong = [
            (register, gains[register], readback[register])
            for register in gains
            if abs(readback[register] - gains[register]) > GAIN_TOLERANCE
        ]
        if wrong:
            detail = ", ".join(
                f"{register.name} asked {asked:.2f} got {got:.2f}" for register, asked, got in wrong
            )
            raise GainVerificationError(
                f"{len(wrong)} of {len(gains)} gain register(s) did not read back as "
                f"written: {detail}. The controller is running gains nobody chose, so "
                "anything measured now would be attributed to the wrong configuration."
            )
        return readback

    def wait_for_arrival(
        self,
        target: Position,
        *,
        timeout_s: float,
        poll_interval_s: float = DEFAULT_ARRIVAL_POLL_INTERVAL_S,
    ) -> Arrival:
        """Poll the position until it reaches ``target`` or time runs out.

        Arrival means both axes are within the capability record's
        acceptance window. It is decided by comparing positions, never
        by reading rotor status: the firmware's idle test also fires on
        ``speed == 0``, so a stalled axis reports idle exactly like an
        arrived one.

        Not arriving is reported, not raised. With stock gains an axis
        stops a degree or two short of target as a matter of course, and
        the window already accounts for that — so a ``False`` here means
        something worth showing the operator, not necessarily a fault.

        Args:
            target: The position that was commanded.
            timeout_s: How long to keep polling.
            poll_interval_s: Gap between position reads.

        Returns:
            The outcome, including the last position read.

        Raises:
            SerialConnectionError: If the port is not open.
            ProtocolError: If a reply can't be parsed.
        """
        started = self._monotonic()
        while True:
            position = self.read_position()
            elapsed = self._monotonic() - started
            if self._capabilities.is_arrived(target, position):
                return Arrival(arrived=True, position=position, elapsed_s=elapsed)
            if elapsed >= timeout_s:
                return Arrival(arrived=False, position=position, elapsed_s=elapsed)
            self._sleep(poll_interval_s)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _await_version(self) -> str:
        """Probe ``VE`` until the controller answers, tolerating homing silence.

        Returns:
            The firmware version string.

        Raises:
            SerialConnectionError: If nothing parseable arrives before
                the timeout.
        """
        started = self._monotonic()
        last_unparsed: bytes | None = None
        while True:
            reply: bytes | None
            try:
                reply = self._exchange_expecting_reply(format_get_version())
            except SerialTimeoutError:
                reply = None
            if reply is not None:
                try:
                    return parse_version(reply)
                except ProtocolError:
                    # Partial or stale bytes can turn up as the controller
                    # comes out of homing. Keep the last one for the error
                    # message, but don't give up on it.
                    last_unparsed = reply
            elapsed = self._monotonic() - started
            if elapsed >= self._homing_timeout_s:
                detail = (
                    f" Last unrecognized reply was {last_unparsed!r}."
                    if last_unparsed is not None
                    else " Nothing was received at all."
                )
                raise SerialConnectionError(
                    f"No usable reply from the rotator after {elapsed:.0f}s."
                    f"{detail} A controller is deaf while homing, so some silence "
                    "is expected — but this is longer than homing should take. "
                    "Check power, cabling, the port name, and the baud rate."
                )
            if self._on_homing_wait is not None:
                self._on_homing_wait(elapsed)
            self._sleep(self._homing_poll_interval_s)

    def _exchange(self, command: Command) -> bytes | None:
        """Write a command and read its reply if it has one.

        Whether to read is carried by the command itself rather than
        decided here — that is what keeps ``SA SE``'s reply from being
        left in the buffer and returned as the answer to whatever is
        asked next.
        """
        self._port.write(command.data)
        if not command.expects_reply:
            return None
        # Half-duplex RS-485: give the transceiver time to turn around.
        # An empty reply is far more often a too-short gap than a fault.
        self._sleep(self._capabilities.rs485_turnaround_s)
        return self._port.readline()

    def _exchange_expecting_reply(self, command: Command) -> bytes:
        """Like :meth:`_exchange`, for commands known to answer."""
        reply = self._exchange(command)
        if reply is None:  # pragma: no cover - guards a programming error
            raise ProtocolError(f"Command {command.data!r} was expected to reply and did not.")
        return reply
