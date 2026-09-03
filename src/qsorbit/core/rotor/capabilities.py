"""What a particular rotor can safely be told to do.

Two builds of the same design differ in ways no protocol query can
reveal: how far the azimuth axis turns before the cable binds, whether
the boom can swing past vertical, what commanding 380° actually does,
how far short of target the axis stops. The firmware reports none of
this, and **inferring it is how hardware gets damaged** — the same
command means opposite things on two rotors of the same model.

So it is declared, per rotor, as data. The station config file supplies
these values (see :mod:`qsorbit.core.station`) and the controller reads
them; nothing in QSOrbit guesses them from a firmware version string or
a probe.

This record is also the *only* travel guard that exists. The SatNOGS
firmware's ``MIN/MAX_M1_ANGLE`` and ``MIN/MAX_M2_ANGLE`` constants are
used solely inside ``homing()`` to detect a homing fault; they are never
applied to a commanded setpoint, and there is no position limit in the
control loop at all. Below QSOrbit there is nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from qsorbit.core.rotor.exceptions import PositionLimitError
from qsorbit.core.rotor.position import MAX_AXIS_DEGREES, Position


class AzimuthWrap(Enum):
    """What the rotor does when commanded past 360°.

    This must be declared and must never be assumed. The two behaviours
    respond to the identical command in opposite ways, and picking the
    wrong one causes real physical harm — an unwanted full rotation
    against a cable, or a refusal to use travel the rotor actually has.
    """

    #: Commanding 380° spins the axis all the way around and settles at
    #: 20°. There is no travel beyond 360°; the extra rotation is pure
    #: cable wrap. Verified on Phil's rotator.
    EXTRA_ROTATION = "extra_rotation"

    #: The axis genuinely has more than 360° of travel, so 20° and 380°
    #: are different physical places. Only on such a rotor is azimuth
    #: unwrapping — commanding ``azimuth + 360`` so a pass crossing north
    #: doesn't hit a cable stop partway through — a safe strategy.
    EXTENDED_TRAVEL = "extended_travel"


@dataclass(frozen=True)
class RotorCapabilities:
    """The declared, per-rotor limits and characteristics of one rotator.

    ``RotorCapabilities`` is a value object: immutable, validated on
    construction, and comparable by value.

    Args:
        azimuth_min_deg: Lowest azimuth axis angle this rotor may be
            commanded to, in degrees from its homed zero.
        azimuth_max_deg: Highest azimuth axis angle this rotor may be
            commanded to.
        elevation_min_deg: Lowest elevation axis angle this rotor may be
            commanded to.
        elevation_max_deg: Highest elevation axis angle this rotor may be
            commanded to. Note this is *axis travel*, not how high the
            antenna can point: a boom that rotates past vertical has a
            legitimate axis angle of 135°, which points at 45° of sky
            elevation with the base turned 180° round. Sky elevation is
            capped at 90° by :class:`~qsorbit.core.geometry.AzEl`, and
            that cap is what every user-facing number obeys. Declaring
            travel past 90° here is what later permits flip mode, where
            :func:`~qsorbit.core.pointing.sky_to_rotor` reaches a target
            as ``azimuth + 180`` and ``180 - elevation`` rather than
            swinging the azimuth axis half a turn mid-pass.
        azimuth_wrap: What the rotor does when commanded past 360°. See
            :class:`AzimuthWrap`.
        acceptance_window_deg: How close to the target an axis has to
            settle before it counts as arrived. This is not a tolerance
            for sloppiness — with stock gains (``Ki = 0``) each axis
            stops when the PID output falls below what the motor needs
            to break static friction, leaving a *normal* steady-state
            shortfall of roughly 1.5° in azimuth and 2.1° in elevation.
            Treating that as a failure would report every successful
            move as an error. 2.5° is a reasonable default for this
            hardware; fine for VHF/UHF beamwidths, worth revisiting for
            anything narrower.
        rs485_turnaround_s: How long to wait between writing a command
            and reading its reply. The link is half-duplex, so the
            transceiver needs time to turn around; an empty reply is far
            more often a too-short turnaround than a real fault. 0.15 s
            was reliable on Phil's link, 0.3-0.5 s is safer on longer
            cable.
        azimuth_free_play_deg: Mechanical free play in the azimuth
            axis, degrees peak-to-peak, or ``None`` if never measured.
            **Measured, never assumed** — see
            ``bench-procedure-mechanical-slop.md`` for the protocol.
            Two numbers depend on it: the stall detector cannot declare
            a jam until the setpoint has advanced past the slop, and
            that latency is what sets how far an integral gain can wind
            up before anything notices. Declaring it is therefore the
            price of being allowed to run a non-zero ``Ki`` — see
            :func:`~qsorbit.core.tracking_profile.check_axis_gain`.
        azimuth_breakaway_pwm: The PWM count at which the azimuth motor
            breaks static friction, or ``None`` if never measured. ~17
            on Phil's rotator. This is the ceiling the wound-up integral
            must stay under: below it a frozen integral is harmless,
            above it the axis drives on its own.
        elevation_free_play_deg: The same for elevation. Usually the
            smaller of the two, because the boom's own weight takes up
            the backlash.
        elevation_breakaway_pwm: The same for elevation. Usually the
            larger, because that weight has to be lifted — ~21-26 on
            Phil's rotator, and the conservative end is the one to
            declare.
        firmware_version: The version string this configuration was
            verified against, e.g. ``"SatNOGS-v2.2.1"``, or ``None`` if
            unrecorded. The controller compares the live ``VE`` reply
            against it and warns on a mismatch. It is **not** used to
            infer any of the fields above.

    Raises:
        ValueError: If any value is non-finite, if either axis's minimum
            is not below its maximum, if a limit lies beyond
            :data:`~qsorbit.core.rotor.position.MAX_AXIS_DEGREES`, if the
            acceptance window is not positive, if the turnaround is
            negative, or if an :attr:`AzimuthWrap.EXTRA_ROTATION` rotor
            declares azimuth travel past 360°.
    """

    azimuth_min_deg: float
    azimuth_max_deg: float
    elevation_min_deg: float
    elevation_max_deg: float
    azimuth_wrap: AzimuthWrap
    acceptance_window_deg: float
    rs485_turnaround_s: float
    firmware_version: str | None = None
    azimuth_free_play_deg: float | None = None
    azimuth_breakaway_pwm: float | None = None
    elevation_free_play_deg: float | None = None
    elevation_breakaway_pwm: float | None = None

    def __post_init__(self) -> None:
        numeric = (
            ("azimuth_min_deg", self.azimuth_min_deg),
            ("azimuth_max_deg", self.azimuth_max_deg),
            ("elevation_min_deg", self.elevation_min_deg),
            ("elevation_max_deg", self.elevation_max_deg),
            ("acceptance_window_deg", self.acceptance_window_deg),
            ("rs485_turnaround_s", self.rs485_turnaround_s),
        )
        for name, value in numeric:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}.")

        if not isinstance(self.azimuth_wrap, AzimuthWrap):
            raise ValueError(
                f"azimuth_wrap must be an AzimuthWrap, got {self.azimuth_wrap!r}. "
                "It has to be declared per rotor — it cannot be inferred."
            )

        for axis, low, high in (
            ("Azimuth", self.azimuth_min_deg, self.azimuth_max_deg),
            ("Elevation", self.elevation_min_deg, self.elevation_max_deg),
        ):
            if low >= high:
                raise ValueError(f"{axis} limits must have min below max, got {low} to {high}.")
            for value in (low, high):
                if abs(value) > MAX_AXIS_DEGREES:
                    raise ValueError(
                        f"{axis} limit {value} is beyond +/-{MAX_AXIS_DEGREES} degrees, "
                        "which no rotor axis can represent."
                    )

        if self.acceptance_window_deg <= 0.0:
            raise ValueError(
                f"acceptance_window_deg must be positive, got {self.acceptance_window_deg}. "
                "A zero window can never be satisfied: stock gains leave a steady-state "
                "shortfall of 1-2 degrees on a perfectly healthy rotor."
            )
        if self.rs485_turnaround_s < 0.0:
            raise ValueError(
                f"rs485_turnaround_s must not be negative, got {self.rs485_turnaround_s}."
            )

        measured = {
            "azimuth_free_play_deg": self.azimuth_free_play_deg,
            "azimuth_breakaway_pwm": self.azimuth_breakaway_pwm,
            "elevation_free_play_deg": self.elevation_free_play_deg,
            "elevation_breakaway_pwm": self.elevation_breakaway_pwm,
        }
        for name, value in measured.items():
            if value is None:
                continue
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}.")
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}.")

        declared = {name for name, value in measured.items() if value is not None}
        if declared and len(declared) != len(measured):
            missing = sorted(set(measured) - declared)
            raise ValueError(
                f"This rotor declares {', '.join(sorted(declared))} but not "
                f"{', '.join(missing)}. The mechanical measurements are all-or-nothing: "
                "the gain clamp needs free play and breakaway together on both axes, and "
                "a partial record would let one axis be checked while the other ran "
                "unguarded. Measure the rest (bench-procedure-mechanical-slop.md) or "
                "declare none of them."
            )

        if self.azimuth_wrap is AzimuthWrap.EXTRA_ROTATION and self.azimuth_max_deg > 360.0:
            raise ValueError(
                f"azimuth_max_deg is {self.azimuth_max_deg}, but this rotor is declared "
                "EXTRA_ROTATION, meaning it has no travel past 360 degrees — commanding "
                "380 would spin it the whole way around and settle at 20. Either lower the "
                "limit to 360 or declare EXTENDED_TRAVEL if the axis really does have the "
                "extra travel."
            )

    @property
    def mechanics_measured(self) -> bool:
        """Whether this rotor's free play and breakaway are on record.

        All four values are present or none are, so any one of them
        answers the question.
        """
        return self.azimuth_free_play_deg is not None

    def mechanics_for(self, axis: str) -> tuple[float, float]:
        """One axis's ``(free_play_deg, breakaway_pwm)``.

        Args:
            axis: ``"azimuth"`` or ``"elevation"``.

        Raises:
            ValueError: If this rotor declares no mechanics, or the axis
                name is not one of the two.
        """
        if axis not in ("azimuth", "elevation"):
            raise ValueError(f"axis must be 'azimuth' or 'elevation', got {axis!r}.")
        if not self.mechanics_measured:
            raise ValueError(
                "This rotor declares no mechanical measurements, so its free play and "
                "breakaway are unknown. Nothing may be derived from them."
            )
        free_play = getattr(self, f"{axis}_free_play_deg")
        breakaway = getattr(self, f"{axis}_breakaway_pwm")
        assert free_play is not None and breakaway is not None  # noqa: S101
        return (free_play, breakaway)

    def check_setpoint(self, position: Position) -> None:
        """Raise if ``position`` is outside this rotor's declared travel.

        Call this on every setpoint before it reaches the wire. The
        firmware performs no equivalent check at any level, so this is
        the only thing standing between a bad number and the hardware.

        Args:
            position: The position about to be commanded.

        Raises:
            PositionLimitError: If either axis is outside its declared
                range.
        """
        for axis, value, low, high in (
            ("Azimuth", position.azimuth, self.azimuth_min_deg, self.azimuth_max_deg),
            ("Elevation", position.elevation, self.elevation_min_deg, self.elevation_max_deg),
        ):
            if not low <= value <= high:
                raise PositionLimitError(
                    f"{axis} {value} degrees is outside this rotor's declared travel "
                    f"of {low} to {high} degrees. The controller firmware applies no "
                    "limits of its own, so this command was not sent."
                )

    def is_arrived(self, target: Position, actual: Position) -> bool:
        """Whether ``actual`` counts as having reached ``target``.

        Arrival is decided by comparing positions, never by reading the
        rotor's ``GS`` status: the firmware's idle test is
        ``(|setpoint - input| <= POSITION_DEADZONE || speed == 0)``, and
        that second term means a **stalled** axis reports idle too.

        The comparison is a plain difference on each axis, with no
        modular arithmetic, because a :class:`Position` is a mechanical
        axis reading rather than a compass bearing — an axis at 359° and
        one at 1° are 358° of travel apart, not 2°.

        Args:
            target: The commanded position.
            actual: The position read back from the rotor.

        Returns:
            ``True`` if both axes are within
            :attr:`acceptance_window_deg` of target.
        """
        return (
            abs(actual.azimuth - target.azimuth) <= self.acceptance_window_deg
            and abs(actual.elevation - target.elevation) <= self.acceptance_window_deg
        )
