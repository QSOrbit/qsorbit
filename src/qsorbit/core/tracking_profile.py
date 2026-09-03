"""Named tracking profiles: how hard this station drives its rotor.

Session 32's gain sweep established that tracking quality is set by two
things that were validated *together*, neither of them sufficient alone:
the **controller gains** (registers 1-6, written over serial, RAM-only)
and the **cadence** QSOrbit commands at (``deadband`` and ``interval``).
This module owns the cadence half and the profile that names both; the
gain half lands with the serial work.

**Why cadence is its own concept rather than the acceptance window.**
Before this module, :class:`~qsorbit.core.pointing.TrackingLoop`
defaulted its movement deadband to
:attr:`~qsorbit.core.rotor.RotorCapabilities.acceptance_window_deg` —
one value doing two unrelated jobs. "How close does this hardware settle
before it counts as arrived" is a fact about the rotor and the gains it
is running; "how far must the target move before we bother sending a new
command" is a policy choice about how hard to drive it. They happened to
be the same number on this station, and the coincidence hid the fact
that the second was never configurable at all — nothing in the
application ever passed ``deadband_deg``.

**The floor nobody had written down.** The minimum commanded step is
``rate x interval``, whatever the deadband says. At the shipped 2.5°
deadband and a 1 s tick, a target moving 1°/s cannot cross the deadband
until the third tick, so the real step is **3.0°, not 2.5** — measured
directly off the command timestamps in Session 32. :meth:`commanded_step_deg`
is what answers "what will this configuration actually do", at the desk,
instead of after the pass.

**The knife edge.** When ``deadband`` is very close to a whole multiple
of ``rate x interval``, which tick fires the command is decided by timing
jitter rather than by geometry: the tick lands a fraction late, the
target has moved 0.98 rather than 1.00, the ``>=`` comparison fails, and
the command waits an entire extra tick. Measured as ``1.15, 2.00, 1.97,
1.95, ...`` where 1.0° was configured — a **silently doubled step**.
:func:`check_cadence` refuses to let that be configured, and is called
from both ends: by :class:`TrackingProfile` when a profile is built from
station config, and by :class:`~qsorbit.core.pointing.TrackingLoop`
itself, so a cadence that arrives from the command line or from a direct
construction cannot slip past the check the config would have applied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from qsorbit.core.rotor import GainRegister, RotorCapabilities

#: The target rate, in degrees per second, that cadence arithmetic is
#: reported against.
#:
#: A low-Earth satellite's apparent motion peaks near 1°/s around zenith.
#: That is the hardest the rotor ever has to work, so it is the honest
#: rate at which to ask "what step size will this configuration really
#: command." It is emphatically **not** a claim that the target moves at
#: 1°/s: rate sweeps from a few hundredths of a degree per second at the
#: horizon up to this near TCA, and the commanded step sweeps with it.
#: A single nominal rate is what makes the question answerable before a
#: pass rather than only after one.
NOMINAL_TRACKING_RATE_DEG_S: Final = 1.0

#: Seconds between tracking ticks when no profile says otherwise.
#:
#: The canonical definition of this project's one-second tracking
#: cadence. :data:`~qsorbit.core.pointing.DEFAULT_TICK_INTERVAL_S` binds
#: to it rather than restating it, because two constants that must agree
#: and are written down twice are two constants that will eventually
#: disagree.
DEFAULT_INTERVAL_S: Final = 1.0

#: The profile name used when station config names none.
DEFAULT_PROFILE_NAME: Final = "stock"

#: How close to a whole multiple of ``rate x interval`` a deadband may
#: sit before it counts as the knife edge, as a fraction of one step.
#:
#: Session 32 measured the failure at a 2% shortfall — the tick landed
#: late enough that the target had moved 0.98 of a configured 1.00. 5%
#: leaves room for a link that is jittering harder than that evening's
#: was, and still passes the validated set, whose ratio is 0.5.
KNIFE_EDGE_MARGIN: Final = 0.05


#: The worst-case target rate, in degrees per second, that the gain
#: clamp validates against.
#:
#: **Slow targets are the dangerous case, and the instinct is the
#: opposite.** Detection takes ``free_play / rate`` seconds, because the
#: stall detector cannot open its gate until the setpoint has advanced
#: past the slop; windup over that time is ``0.5 x Ki x rate x t^2``.
#: Substituting one into the other, the rate cancels once and inverts:
#: **windup at detection = ``0.5 x Ki x free_play^2 / rate``**. Halving
#: the rate doubles the exposure.
#:
#: 0.25 rather than the 1.0 of :data:`NOMINAL_TRACKING_RATE_DEG_S`,
#: which answers a different question. That one asks "what step will
#: this cadence command at the hardest the rotor ever works"; this one
#: asks "how slowly could the target be moving when something jams."
#: This catalogue runs 0.4-0.5 deg/s near TCA and drops well under that
#: toward the horizon, where a cable snag is not less likely. Sizing the
#: clamp to the TCA rate would exempt exactly the regime the guard
#: exists for.
DESIGN_RATE_DEG_S: Final = 0.25


class GainPolicyError(ValueError):
    """Base for the two ways a profile's gains can be refused.

    Both are :class:`ValueError` for the same reason
    :class:`CadenceError` is: they reject a value rather than report a
    failure, so
    :func:`~qsorbit.core.station.load_station_config` re-raises them as
    a :class:`~qsorbit.core.station.ConfigError` carrying the file name.
    """


class UnmeasuredMechanicsError(GainPolicyError):
    """Raised when a profile asks for integral gain on an unmeasured rotor.

    Deliberately *not* the same error as :class:`GainClampError`. "You
    have not measured this rotator yet" and "this rotator cannot safely
    run that gain" are different problems with different fixes, and an
    operator who is told the wrong one will change the wrong number.
    """


class GainClampError(GainPolicyError):
    """Raised when a profile's integral gain could wind past breakaway."""


def windup_at_detection_pwm(
    ki: float,
    free_play_deg: float,
    *,
    rate_deg_s: float = DESIGN_RATE_DEG_S,
) -> float:
    """How far the integral term winds up before a stall is detected.

    ``0.5 x Ki x free_play^2 / rate``, in PWM counts. See
    :data:`DESIGN_RATE_DEG_S` for where the shape comes from.

    Note the square: **the free-play constant matters quadratically.**
    Halving a measured 3.0 deg to 1.5 does not halve the exposure, it
    quarters it, which is why the fair-weather wind measurement is worth
    a morning.
    """
    if rate_deg_s <= 0.0:
        raise ValueError(f"rate_deg_s must be positive, got {rate_deg_s}.")
    return 0.5 * ki * free_play_deg**2 / rate_deg_s


#: Decimal places the two "safe value" helpers report to.
#:
#: They round **down**, never to nearest, and this is why they are
#: functions rather than two format strings. A limit printed to two
#: decimals and rounded up lands back over the line: 0.9767 shown as
#: "0.98" is refused a second time, which is worse than offering no
#: remedy at all. Both call sites — the refusal message and
#: ``qsorbit status`` — went wrong this way independently before the
#: arithmetic was named once here.
_SAFE_VALUE_DECIMALS: Final = 2


def _floor_to(value: float, decimals: int = _SAFE_VALUE_DECIMALS) -> float:
    """``value`` rounded down to ``decimals`` places."""
    scale = 10.0**decimals
    return math.floor(value * scale) / scale


def max_safe_ki(
    free_play_deg: float,
    breakaway_pwm: float,
    *,
    rate_deg_s: float = DESIGN_RATE_DEG_S,
) -> float:
    """The largest integral gain this axis can take, rounded down.

    Inverts :func:`windup_at_detection_pwm` for ``Ki``. Reported by
    ``qsorbit status`` so an operator can see what their rotor allows
    without first configuring something and being told no.
    """
    return _floor_to(breakaway_pwm * rate_deg_s / (0.5 * free_play_deg**2))


def max_safe_free_play_deg(
    ki: float,
    breakaway_pwm: float,
    *,
    rate_deg_s: float = DESIGN_RATE_DEG_S,
) -> float:
    """The largest free play that still permits ``ki``, rounded down.

    The other way out of a clamp refusal, and usually the better one:
    a hand-push measurement is not the disturbance the guard has to
    tolerate in service.
    """
    return _floor_to(math.sqrt(breakaway_pwm * rate_deg_s / (0.5 * ki)))


def check_axis_gain(
    ki: float,
    free_play_deg: float,
    breakaway_pwm: float,
    *,
    axis: str,
    profile_name: str,
    rate_deg_s: float = DESIGN_RATE_DEG_S,
) -> None:
    """Raise :class:`GainClampError` if ``ki`` could drive an axis on its own.

    The safety argument in one line: a stalled axis has its setpoint
    frozen by :class:`~qsorbit.core.stall_guard.StallDetector`, which
    freezes the integral too — but the integral keeps whatever it
    accumulated up to that moment, because PID_v1 has no way to
    discharge it over serial. That frozen residual is harmless **only
    while it stays below the motor's breakaway**. Above it, the axis
    drives itself with nothing commanding it, and the only recovery is a
    power cycle.

    Args:
        ki: The integral gain this profile asks for on this axis.
        free_play_deg: That axis's measured mechanical free play.
        breakaway_pwm: That axis's measured stiction breakaway.
        axis: ``"azimuth"`` or ``"elevation"``, for the message.
        profile_name: The profile being checked, for the message.
        rate_deg_s: The worst-case target rate to validate against.
    """
    if ki == 0.0:
        return
    windup = windup_at_detection_pwm(ki, free_play_deg, rate_deg_s=rate_deg_s)
    if windup <= breakaway_pwm:
        return
    safe_ki = max_safe_ki(free_play_deg, breakaway_pwm, rate_deg_s=rate_deg_s)
    safe_free_play = max_safe_free_play_deg(ki, breakaway_pwm, rate_deg_s=rate_deg_s)
    raise GainClampError(
        f"Tracking profile {profile_name!r} asks for {axis} Ki {ki:g}, which would wind "
        f"up to {windup:.1f} PWM before a stall is detected, against this rotor's "
        f"measured {axis} breakaway of {breakaway_pwm:g}. Above breakaway the axis drives "
        "itself with nothing commanding it, and stock firmware offers no way to discharge "
        "an integral over serial: the only recovery is a power cycle.\n\n"
        f"  0.5 x {ki:g} x {free_play_deg:g}^2 / {rate_deg_s:g} = {windup:.1f} PWM\n\n"
        f"Two ways out. Lower this axis's Ki to {safe_ki:.2f} or less. Or measure the free "
        f"play properly: at Ki {ki:g} anything at or under {safe_free_play:.2f} deg clears, "
        f"and {free_play_deg:g} is a hand-push figure rather than a disturbance this "
        "rotator meets in service."
    )


class CadenceError(ValueError):
    """Raised when a deadband and interval combine into a bad cadence.

    A :class:`ValueError` because it is a rejected value rather than a
    failure of anything, which also means
    :func:`~qsorbit.core.station.load_station_config` re-raises it as a
    :class:`~qsorbit.core.station.ConfigError` carrying the file name,
    the same way every other value-object rejection is reported.
    """


def ticks_per_command(
    deadband_deg: float,
    interval_s: float,
    *,
    rate_deg_s: float = NOMINAL_TRACKING_RATE_DEG_S,
) -> int:
    """How many ticks pass between commands, at ``rate_deg_s``.

    The loop accumulates ``rate x interval`` of target movement per
    tick and commands once that reaches the deadband, so this is
    ``ceil(deadband / (rate x interval))`` with a floor of one — a
    deadband of zero commands on every tick, which is a legitimate
    configuration and not a division to protect against.
    """
    step = rate_deg_s * interval_s
    if step <= 0.0:
        raise ValueError(f"rate_deg_s x interval_s must be positive, got {step}.")
    return max(1, math.ceil(deadband_deg / step))


def commanded_step_deg(
    deadband_deg: float,
    interval_s: float,
    *,
    rate_deg_s: float = NOMINAL_TRACKING_RATE_DEG_S,
) -> float:
    """The step size this cadence will actually command, at ``rate_deg_s``.

    **This is the number that matters, and it is not the deadband.** The
    shipped 2.5° deadband at a 1 s tick commands 3.0° steps against a
    1°/s target; Session 32's validated 0.25° deadband at a 0.5 s tick
    commands 0.5°. Both were measured off command timestamps before this
    function existed to predict them.
    """
    return ticks_per_command(deadband_deg, interval_s, rate_deg_s=rate_deg_s) * (
        rate_deg_s * interval_s
    )


def knife_edge_ratio(
    deadband_deg: float,
    interval_s: float,
    *,
    rate_deg_s: float = NOMINAL_TRACKING_RATE_DEG_S,
) -> float:
    """How many ticks' worth of movement the deadband is, unrounded.

    The raw ``deadband / (rate x interval)``. A value near a whole
    number at or above 1 is the knife edge: rounding decides which tick
    commands, and timing jitter decides the rounding.
    """
    step = rate_deg_s * interval_s
    if step <= 0.0:
        raise ValueError(f"rate_deg_s x interval_s must be positive, got {step}.")
    return deadband_deg / step


def check_cadence(
    deadband_deg: float,
    interval_s: float,
    *,
    rate_deg_s: float = NOMINAL_TRACKING_RATE_DEG_S,
) -> None:
    """Raise :class:`CadenceError` if this cadence sits on the knife edge.

    Called from :class:`TrackingProfile` and from
    :class:`~qsorbit.core.pointing.TrackingLoop`, so the check applies
    however the numbers arrive — station config, ``--interval`` on the
    command line, or a direct construction in a test.

    A ratio below 1 is safe by construction: the loop commands on every
    tick and no rounding decision exists to be unstable.
    """
    ratio = knife_edge_ratio(deadband_deg, interval_s, rate_deg_s=rate_deg_s)
    nearest = round(ratio)
    if nearest < 1 or abs(ratio - nearest) > KNIFE_EDGE_MARGIN:
        return
    step = rate_deg_s * interval_s
    raise CadenceError(
        f"deadband_deg {deadband_deg} is {ratio:.3f} times the per-tick movement "
        f"({rate_deg_s} deg/s x {interval_s} s = {step} deg), which is within "
        f"{KNIFE_EDGE_MARGIN:g} of a whole tick. That is the knife edge: whether a "
        "command fires on this tick or the next is decided by timing jitter rather "
        f"than by geometry, so the commanded step silently doubles to about "
        f"{2 * step:g} deg. Session 32 measured exactly this — 1.15, 2.00, 1.97, 1.95 "
        "where 1.0 was configured. Move the deadband off the multiple: halving it is "
        "the usual fix, and commands every tick."
    )


@dataclass(frozen=True)
class TrackingProfile:
    """One named way of driving this station's rotor.

    A profile is the whole validated configuration rather than a bag of
    knobs, because Session 32's set was validated as a whole and neither
    half is sufficient alone: **Ki fixes lag; cadence fixes hop.** A
    toggle that moved one and not the other would ship half of a
    measured result.

    Args:
        name: What this profile is called in station config, e.g.
            ``"stock"`` or ``"tracking"``.
        deadband_deg: How far the target must move, on either axis,
            before a new position is commanded. **Not** the acceptance
            window — see the module docstring. Zero means command every
            tick.
        interval_s: Seconds between tracking ticks.
        arrival_window_deg: How close an axis must settle to count as
            arrived while this profile is active, or ``None`` to use the
            rotor's declared
            :attr:`~qsorbit.core.rotor.RotorCapabilities.acceptance_window_deg`.
            Profile-dependent because the shortfall it describes is a
            property of the *gains*, not of the mechanism: stock gains
            (``Ki = 0``) leave 1.5-2.1° of stiction residual, and the
            validated tracking set settles at 0.64-0.81°. A station
            running tracking gains against a 2.5° window would report
            arrival long before it had arrived.
        azimuth_kp: Azimuth proportional gain to push while this profile
            is active, or ``None`` to push nothing. All six gains are
            declared together or not at all — see :attr:`gains`.
        azimuth_ki: Azimuth integral gain. **This is the one the clamp
            is about.** Session 32 measured Ki as the lever that unlocks
            everything else: the rotor parks exactly on its stiction
            threshold by construction (``Kp x error + MIN_PWM =
            breakaway`` on both axes), so nothing about step size can
            lower the lag floor and even a small integral tips the axis
            from slam-then-stall into continuous creep. It is also the
            only gain that accumulates, which is why it is the only one
            that can drive an axis after the loop has stopped commanding
            it.
        azimuth_kd: Azimuth derivative gain. Session 32 **refuted**
            raising this: Kd closes a velocity loop around a spring, and
            the boom's torsional ring-back is that spring, so the term
            pumped the oscillation it was meant to damp. Every measured
            number got worse. Declared here for completeness, not as a
            knob worth turning.
        elevation_kp: Elevation proportional gain.
        elevation_ki: Elevation integral gain.
        elevation_kd: Elevation derivative gain.

    Raises:
        ValueError: If any value is non-finite, if ``deadband_deg`` is
            negative, if ``interval_s`` or ``arrival_window_deg`` is not
            positive, if the name is blank, if any gain is negative, or
            if some but not all six gains are declared.
        CadenceError: If the cadence sits on the knife edge.
    """

    name: str
    deadband_deg: float
    interval_s: float
    arrival_window_deg: float | None = None
    azimuth_kp: float | None = None
    azimuth_ki: float | None = None
    azimuth_kd: float | None = None
    elevation_kp: float | None = None
    elevation_ki: float | None = None
    elevation_kd: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A tracking profile needs a name.")

        numeric: list[tuple[str, float]] = [
            ("deadband_deg", self.deadband_deg),
            ("interval_s", self.interval_s),
        ]
        if self.arrival_window_deg is not None:
            numeric.append(("arrival_window_deg", self.arrival_window_deg))
        for field_name, value in numeric:
            if not math.isfinite(value):
                raise ValueError(
                    f"{field_name} must be a finite number, got {value!r} "
                    f"in tracking profile {self.name!r}."
                )

        if self.deadband_deg < 0.0:
            raise ValueError(
                f"deadband_deg must not be negative, got {self.deadband_deg} "
                f"in tracking profile {self.name!r}."
            )
        if self.interval_s <= 0.0:
            raise ValueError(
                f"interval_s must be positive, got {self.interval_s} "
                f"in tracking profile {self.name!r}."
            )
        if self.arrival_window_deg is not None and self.arrival_window_deg <= 0.0:
            raise ValueError(
                f"arrival_window_deg must be positive, got {self.arrival_window_deg} "
                f"in tracking profile {self.name!r}. A zero window can never be "
                "satisfied: even the validated tracking set settles 0.6-0.8 degrees short."
            )

        gain_fields = {
            "azimuth_kp": self.azimuth_kp,
            "azimuth_ki": self.azimuth_ki,
            "azimuth_kd": self.azimuth_kd,
            "elevation_kp": self.elevation_kp,
            "elevation_ki": self.elevation_ki,
            "elevation_kd": self.elevation_kd,
        }
        for field_name, value in gain_fields.items():
            if value is None:
                continue
            if not math.isfinite(value):
                raise ValueError(
                    f"{field_name} must be a finite number, got {value!r} "
                    f"in tracking profile {self.name!r}."
                )
            if value < 0.0:
                raise ValueError(
                    f"{field_name} must not be negative, got {value} "
                    f"in tracking profile {self.name!r}."
                )

        declared = {name for name, value in gain_fields.items() if value is not None}
        if declared and len(declared) != len(gain_fields):
            missing = sorted(set(gain_fields) - declared)
            raise ValueError(
                f"Tracking profile {self.name!r} declares {', '.join(sorted(declared))} "
                f"but not {', '.join(missing)}. Gains are all six or none: the controller "
                "would keep its compiled defaults for whatever is left out, so a partial "
                "set runs a mixture nobody chose and every measurement taken under it is "
                "attributed to the wrong configuration."
            )

        check_cadence(self.deadband_deg, self.interval_s)

    @property
    def commanded_step_deg(self) -> float:
        """The step this profile really commands at the nominal rate.

        A property rather than a stored field because it is derived, and
        because storing it would let it drift from the values it is
        derived from.
        """
        return commanded_step_deg(self.deadband_deg, self.interval_s)

    def window_against(self, capabilities_window_deg: float) -> float:
        """This profile's arrival window, falling back to the rotor's own.

        Args:
            capabilities_window_deg: The rotor's declared
                ``acceptance_window_deg``, used when this profile does
                not override it.
        """
        if self.arrival_window_deg is None:
            return capabilities_window_deg
        return self.arrival_window_deg

    @property
    def gains(self) -> dict[GainRegister, float] | None:
        """The registers to push for this profile, or ``None`` for stock.

        ``None`` is not "push zeros" — it is **push nothing**, leaving
        the controller on its compiled defaults. That is what makes the
        ``stock`` profile a real baseline rather than a guess at what
        stock was: gains are RAM-only, so a controller that has not been
        written to since power-on is definitionally running the
        firmware's own values.
        """
        if self.azimuth_kp is None:
            return None
        assert self.azimuth_ki is not None  # noqa: S101
        assert self.azimuth_kd is not None  # noqa: S101
        assert self.elevation_kp is not None  # noqa: S101
        assert self.elevation_ki is not None  # noqa: S101
        assert self.elevation_kd is not None  # noqa: S101
        return {
            GainRegister.AZIMUTH_KP: self.azimuth_kp,
            GainRegister.AZIMUTH_KI: self.azimuth_ki,
            GainRegister.AZIMUTH_KD: self.azimuth_kd,
            GainRegister.ELEVATION_KP: self.elevation_kp,
            GainRegister.ELEVATION_KI: self.elevation_ki,
            GainRegister.ELEVATION_KD: self.elevation_kd,
        }

    def check_against(
        self,
        capabilities: RotorCapabilities,
        *,
        rate_deg_s: float = DESIGN_RATE_DEG_S,
    ) -> None:
        """Raise if this profile's gains are unsafe on ``capabilities``.

        Called from two ends, the same way :func:`check_cadence` is:
        by :func:`~qsorbit.core.station.load_station_config`, so a bad
        profile is refused at the desk rather than at the rotor, and
        again immediately before the gains reach the wire, so a profile
        built in code cannot slip past the check the config file would
        have applied.

        A profile with no gains, or with zero integral gain on both
        axes, passes unconditionally: there is nothing that can wind up,
        so the rotor's mechanics are irrelevant and an unmeasured
        rotator is free to run it. That is what lets ``stock`` work on
        anybody's hardware on day one.

        Args:
            capabilities: The rotor this profile would run on.
            rate_deg_s: The worst-case target rate to validate against.

        Raises:
            UnmeasuredMechanicsError: If the profile asks for non-zero
                integral gain and this rotor's free play and breakaway
                have never been measured.
            GainClampError: If the integral gain could wind past
                breakaway before a stall is detected.
        """
        gains = self.gains
        if gains is None:
            return
        integral = {
            "azimuth": gains[GainRegister.AZIMUTH_KI],
            "elevation": gains[GainRegister.ELEVATION_KI],
        }
        if not any(integral.values()):
            return

        if not capabilities.mechanics_measured:
            asked = ", ".join(f"{axis} Ki {ki:g}" for axis, ki in integral.items() if ki)
            raise UnmeasuredMechanicsError(
                f"Tracking profile {self.name!r} asks for {asked}, but this rotor declares "
                "no free_play_deg or breakaway_pwm, so there is nothing to check it "
                "against. An integral term is the only gain that accumulates, and how far "
                "it accumulates before a stall is noticed depends entirely on those two "
                "numbers, which differ between two builds of the same design and cannot "
                "be inferred from a firmware version.\n\n"
                "Measure them (bench-procedure-mechanical-slop.md) and declare them under "
                "[rotor.capabilities], or drop this profile's Ki to 0. Borrowing another "
                "rotator's figures would produce a guarantee that was never about yours."
            )

        for axis, ki in integral.items():
            free_play, breakaway = capabilities.mechanics_for(axis)
            check_axis_gain(
                ki,
                free_play,
                breakaway,
                axis=axis,
                profile_name=self.name,
                rate_deg_s=rate_deg_s,
            )
