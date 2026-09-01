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

    Raises:
        ValueError: If any value is non-finite, if ``deadband_deg`` is
            negative, if ``interval_s`` or ``arrival_window_deg`` is not
            positive, or if the name is blank.
        CadenceError: If the cadence sits on the knife edge.
    """

    name: str
    deadband_deg: float
    interval_s: float
    arrival_window_deg: float | None = None

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
