"""Detecting a rotor axis that has stopped following its setpoint.

**The hazard this exists for is not the one the roadmap named.** Phase
3's planning worried about integral windup: at ``Ki > 0`` a jammed axis
accumulates ``outputSum`` until the firmware's ``±MAX_PWM`` clamp, then
slams the instant it frees. That is real. But reading
:meth:`~qsorbit.core.pointing.TrackingLoop._decide` against the firmware
constants turns up a larger contribution that exists at **stock gains,
today, with no gain profile anywhere near it**.

The loop compares each new target against the **last commanded**
position, never against what the rotor reports — deliberately, because
comparing against the reading would chase the stiction residual forever
and re-command the same position on every tick. The consequence is that
a jammed axis keeps having its setpoint advanced as the satellite moves.
Error therefore grows at the target's rate whatever the gains are, and
the firmware's output is ``Kp × error`` clamped at ``MAX_PWM`` 180:

===================================  ======================  ==============
Contribution                         Arms at                 At 1°/s
===================================  ======================  ==============
Proportional runaway, Kp 8 azimuth   error 22.5° → full PWM  ~22 s
Proportional runaway, Kp 10 elev.    error 18° → full PWM    ~18 s
Integral windup, Ki 1.0              ``outputSum`` 180       ~19 s
===================================  ======================  ==============

The same order of magnitude, and the first two need no gain profile at
all. So the most valuable thing to do about a stall is **stop advancing
the setpoint**, which bounds the proportional term as well as the
integral one, costs no serial round trip, and cannot itself fail. Pushing
``Ki 0`` is a second, smaller action.

**Why the detector's definition avoids the acceptance window.** The
obvious test — "error exceeds the window and the axis is not moving" —
makes detection latency depend on how wide the window is, and the window
is a per-profile value. The test used here instead is *the setpoint
advanced and the axis did not follow*, which needs no window: when the
loop is not commanding, the setpoint does not advance and no stall can
be declared, which is exactly the right behaviour for a target below the
horizon or an axis sitting inside its deadband.

**Recovery is not symmetric with detection.** Once stalled, the loop
stops advancing the setpoint, so the "setpoint advanced" half of the
test can never be true again — a detector that used the same condition
in both directions would declare recovery on the very next tick. So a
stall ends only when the axis is seen to *move*, measured from where it
was when the stall was declared.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final

from qsorbit.core.rotor import Position

#: The smallest position change the rotor can report, in degrees.
#:
#: ``AZ EL`` answers with one decimal place (integration rule §1), so
#: movement below this is not observable through the command the
#: tracking loop actually uses. It is the floor on what "moved" can
#: mean, not a tolerance anyone chose.
POSITION_REPORT_RESOLUTION_DEG: Final = 0.1

#: Consecutive ticks of "commanded, did not follow" that count as a stall.
#:
#: Six, derived rather than picked:
#:
#: * At the bench-validated 0.5 s interval this is **3.0 s** of latency.
#:   Windup over that at ``Ki 1.0`` is ``0.5 × Ki × rate × t²`` = 4.5 PWM
#:   against azimuth's ~17-count stiction breakaway — so the integral
#:   frozen at detection cannot move the axis, which is what makes
#:   freezing it a safe response rather than a partial one.
#: * It leaves room for ``Ki`` up to ``34 / 3.0²`` ≈ 3.8, which covers
#:   the Ki 3.0 Session 32 measured and rejected on other grounds.
#: * Against false positives: a command issued on one tick may not
#:   produce measurable motion by the next (0.15 s of RS-485 turnaround
#:   plus motor spin-up), and six ticks is several times that margin.
#:
#: Note the latency is ``ticks × interval``, so the *same* count is a
#: different exposure at a different cadence — 6.0 s at the shipped 1 s
#: tick. That is why the gain clamp is derived from a profile's own
#: cadence rather than fixed.
DEFAULT_STALL_TICKS: Final = 6


#: The axis attribute names on :class:`~qsorbit.core.rotor.Position`.
AXES: Final = ("azimuth", "elevation")


def _delta(a: Position, b: Position, axis: str) -> float:
    """One axis's absolute difference, in degrees.

    A plain difference with no modular arithmetic, matching
    :meth:`~qsorbit.core.rotor.RotorCapabilities.is_arrived`: a
    :class:`~qsorbit.core.rotor.Position` is a mechanical axis reading
    rather than a compass bearing.
    """
    return abs(getattr(a, axis) - getattr(b, axis))


@dataclass(frozen=True)
class StallGuard:
    """How much evidence counts as a stalled axis.

    A value object, so a profile can state its own policy and the gain
    clamp can ask what latency that policy implies.

    Args:
        ticks: Consecutive ticks of the setpoint advancing without the
            axis following, before a stall is declared.
        resolution_deg: The smallest movement treated as movement.
            Defaults to what the rotor can actually report.

    Raises:
        ValueError: If ``ticks`` is below one or ``resolution_deg`` is
            not positive.
    """

    ticks: int = DEFAULT_STALL_TICKS
    resolution_deg: float = POSITION_REPORT_RESOLUTION_DEG

    def __post_init__(self) -> None:
        if self.ticks < 1:
            raise ValueError(f"ticks must be at least 1, got {self.ticks}.")
        if self.resolution_deg <= 0.0:
            raise ValueError(
                f"resolution_deg must be positive, got {self.resolution_deg}. "
                "Zero would make any reading at all count as movement, and the "
                "rotor reports one decimal place."
            )

    def latency_s(self, interval_s: float) -> float:
        """How long detection takes at a given tick interval.

        The number the gain clamp is derived from: an axis can wind up
        for exactly this long before anything notices.
        """
        return self.ticks * interval_s


class StallDetector:
    """Watches setpoint-versus-reality, tick by tick.

    Mutable and per-run, unlike :class:`StallGuard` which is policy.
    Kept separate from :class:`~qsorbit.core.pointing.TrackingLoop` so
    the detection rule can be tested without a target, an observer, a
    clock or a rotor.

    Args:
        guard: The policy to apply.
    """

    def __init__(self, guard: StallGuard | None = None) -> None:
        self._guard = guard if guard is not None else StallGuard()
        self._history: deque[tuple[Position, Position]] = deque(maxlen=self._guard.ticks + 1)
        self._stalled = False
        self._stalled_axes: tuple[str, ...] = ()
        self._stalled_at: Position | None = None
        self._events = 0

    @property
    def guard(self) -> StallGuard:
        """The policy this detector is applying."""
        return self._guard

    @property
    def is_stalled(self) -> bool:
        """Whether an axis is currently believed to be jammed."""
        return self._stalled

    @property
    def stalled_axes(self) -> tuple[str, ...]:
        """Which axes are currently stuck, e.g. ``("elevation",)``.

        Empty when nothing is stalled. Worth naming rather than
        reporting a bare boolean: "elevation is not following" and
        "azimuth is not following" point at different mechanical
        causes, and the operator is the one who has to go and look.
        """
        return self._stalled_axes

    @property
    def events(self) -> int:
        """How many distinct stalls have been declared this run.

        An instrument rather than a health score, in the same spirit as
        :attr:`~qsorbit.core.pointing.TrackingLoop.guard_rereads`: a run
        that ends with this above zero has something worth writing down,
        and the *rate* is what distinguishes a link having a bad evening
        from a boom fouling its cable every pass.
        """
        return self._events

    def observe(self, commanded: Position | None, reported: Position) -> bool:
        """Record one tick and say whether an axis is stalled.

        Args:
            commanded: The position most recently sent to the rotor, or
                ``None`` before anything has been commanded.
            reported: What the rotor says its axes read now.

        Returns:
            ``True`` if a stall is in force after this tick.
        """
        if self._stalled:
            self._check_recovery(reported)
            return self._stalled

        if commanded is None:
            # Nothing has been commanded yet, so nothing can have been
            # failed to follow.
            self._history.clear()
            return False

        self._history.append((commanded, reported))
        if len(self._history) <= self._guard.ticks:
            return False

        oldest_commanded, oldest_reported = self._history[0]
        newest_commanded, newest_reported = self._history[-1]

        # Per axis, never pooled. Taking the larger of the two movements
        # would let a healthy azimuth mask a jammed elevation, which is
        # the *likely* case rather than an exotic one: elevation carries
        # the gravity load, has the higher stiction breakaway, and
        # Session 32 measured its up and down runs as separate
        # experiments for exactly that reason.
        stalled = tuple(
            axis
            for axis in AXES
            if _delta(newest_commanded, oldest_commanded, axis) > self._guard.resolution_deg
            and _delta(newest_reported, oldest_reported, axis) <= self._guard.resolution_deg
        )
        if stalled:
            self._stalled = True
            self._stalled_axes = stalled
            self._stalled_at = reported
            self._events += 1
            self._history.clear()
        return self._stalled

    def _check_recovery(self, reported: Position) -> None:
        """Clear the stall once the axis is seen to move again.

        Measured from where the axis was when the stall was declared,
        not across a rolling window — see the module docstring on why
        recovery cannot reuse the detection test.
        """
        if self._stalled_at is None:
            return
        # Every axis that stalled has to move, not just one of them: a
        # freed azimuth says nothing about an elevation still jammed,
        # and the setpoint stays frozen while either is stuck.
        if all(
            _delta(reported, self._stalled_at, axis) > self._guard.resolution_deg
            for axis in self._stalled_axes
        ):
            self._stalled = False
            self._stalled_axes = ()
            self._stalled_at = None
            self._history.clear()
