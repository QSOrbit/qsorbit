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

**Why "did it move at all" is not the test.** The first version of this
asked whether the axis had moved more than the rotor's reporting
resolution. Measurement killed that: **azimuth on this rotator has 2.95°
of mechanical free play**, peak-to-peak, measured by logging ``IP3`` at
5.95 Hz while the boom was pushed by hand (noise floor 0.04° with hands
off, so a 70× separation). That is twenty-nine times the reporting
resolution. A jammed axis nudged by wind or cable tension wanders inside
its own slop, shows far more than 0.1° of movement, and would never be
declared stalled — in exactly the windy conditions where a cable snag is
most likely. Session 32 had already seen the azimuth axis swaying in
wind; this put a number on it.

So the test is **how far the axis has fallen behind where it was told
to go** — the setpoint's advance minus the progress actually made. Free
play oscillates about a mean and sheds the whole advance; an axis really
following carries a constant lag and sheds almost none of it; wind sway
on a healthy axis rides on top of real progress and still keeps up. The
threshold is the free play itself, a measured per-rotor fact rather than
a chosen constant.

**Comparing the progress against that threshold directly does not work,
and hardware is what showed it.** The first version did exactly that, and
on 2026-09-02 declared a stall on a healthy elevation axis 53 seconds
before anything was disconnected — because at the shipped cadence the
commanded position advances in ~3° jumps, so a window containing one
command has an advance just over the threshold while a normally-lagging
axis nets just under it. Both halves of the test true at once, on a rotor
that was tracking fine. Unit tests missed it because they exercised a
perfectly following axis and a completely jammed one; the failure lives
in the band between the two.

**And the window is a duration, not a tick count.** The evidence needed
is "the setpoint advanced further than the slop and the axis did not
follow", which is a question about degrees and therefore about seconds —
at 1°/s a 6 s window advances 6°, comfortably past 2.95° of play. A
fixed tick count would have meant 6 s at the shipped 1 s cadence and 3 s
at the bench-validated 0.5 s one, i.e. half the evidence for the same
nominal setting.

**Recovery is not symmetric with detection.** Once stalled, the loop
stops advancing the setpoint, so the "setpoint advanced" half of the
test can never be true again — a detector that used the same condition
in both directions would declare recovery on the very next tick. So a
stall ends only when the axis is seen to *move*, measured from where it
was when the stall was declared.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Final

from qsorbit.core.rotor import Position

#: The smallest position change the rotor can report, in degrees.
#:
#: ``AZ EL`` answers with one decimal place and ``IP3``/``IP4`` with two
#: (integration rule §1). This is a floor on what any movement threshold
#: can mean, not a threshold anyone should use: the free play below is
#: nearly thirty times larger and is what actually decides.
POSITION_REPORT_RESOLUTION_DEG: Final = 0.1

#: Mechanical free play in one axis, degrees peak-to-peak.
#:
#: **Measured, not chosen.** Both axes of this rotator, 2026-09-02:
#: **azimuth 2.95°, elevation 2.55°**. ``IP3``/``IP4`` logged at ~5.96 Hz
#: for 60 s while the boom was pushed by hand within normal handling
#: force, motors connected so the non-backdrivable 54:1 gearbox held the
#: output. Seven consecutive five-second buckets read 2.68–2.90° (az) and
#: 2.29–2.55° (el), against hands-off buckets of 0.00–0.05°, so the
#: figures are not an artefact of the encoder. One value covers both
#: axes: they differ by 14%, and azimuth — the larger, and the one with
#: the *lower* stiction breakaway — is the binding case. Rounded up to
#: 3.0 for headroom.
#:
#: A gravity-preload argument predicted elevation would be markedly
#: smaller, on the reasoning that a loaded axis rests against one tooth
#: flank and takes the backlash up. **That prediction was wrong in
#: magnitude** and is recorded here so nobody re-derives it: the load
#: does bias where the axis *rests* (elevation settled 0.6° from where it
#: started, azimuth 0.2°), but it does not reduce the play available when
#: something pushes.
#:
#: This is a property of one build's gear train and **another rotator
#: should measure its own** — ``rotor-motion-log.py --observe`` in the
#: project notes is the instrument. It lives here rather than in the
#: capability record only because no second rotor exists yet to disagree.
DEFAULT_FREE_PLAY_DEG: Final = 3.0

#: How long the axis is watched before a stall is declared, in seconds.
#:
#: Six, and it is squeezed from both sides:
#:
#: * **Long enough to beat the slop.** At a satellite's ~1°/s near
#:   zenith the setpoint advances 6° in this window, just over twice the
#:   measured 2.95° of free play. Much shorter and a jammed axis wandering
#:   inside its own backlash could net enough apparent progress to pass.
#: * **Short enough to matter.** The firmware's ``Kp × error`` reaches
#:   ``MAX_PWM`` at 22.5° of azimuth error, about 22 s at 1°/s, so six
#:   seconds leaves the setpoint frozen with plenty of margin.
#:
#: **The upper bound tightens sharply once ``Ki > 0``**, and the two
#: constraints stop fitting: windup is ``0.5 × Ki × rate × t²``, which at
#: ``Ki 1.0`` passes azimuth's ~17-count breakaway at **5.8 s** — shorter
#: than the window the free play demands. That conflict belongs to the
#: gain-profile work and is not resolved here; at stock's ``Ki = 0``
#: nothing accumulates and the only deadline is the proportional one.
DEFAULT_STALL_WINDOW_S: Final = 6.0

#: The axis attribute names on :class:`~qsorbit.core.rotor.Position`.
AXES: Final = ("azimuth", "elevation")


def _signed(a: Position, b: Position, axis: str) -> float:
    """One axis's signed difference ``a - b``, in degrees.

    A plain difference with no modular arithmetic, matching
    :meth:`~qsorbit.core.rotor.RotorCapabilities.is_arrived`: a
    :class:`~qsorbit.core.rotor.Position` is a mechanical axis reading
    rather than a compass bearing.
    """
    return getattr(a, axis) - getattr(b, axis)


@dataclass(frozen=True)
class StallGuard:
    """How much evidence counts as a stalled axis.

    A value object, so a profile can state its own policy and the gain
    clamp can ask what latency that policy implies at a given cadence.

    Args:
        window_s: How long the axis is watched before a stall is
            declared. A duration rather than a tick count, so the same
            policy is the same amount of evidence at any cadence.
        free_play_deg: The axis's mechanical slop, peak-to-peak. Both
            the threshold net progress must beat and the minimum
            setpoint advance before a stall can be judged at all.

    Raises:
        ValueError: If ``window_s`` is not positive, or if
            ``free_play_deg`` is below the rotor's reporting resolution
            — a threshold finer than the encoder can report is a
            threshold that means nothing.
    """

    window_s: float = DEFAULT_STALL_WINDOW_S
    free_play_deg: float = DEFAULT_FREE_PLAY_DEG

    def __post_init__(self) -> None:
        if self.window_s <= 0.0:
            raise ValueError(f"window_s must be positive, got {self.window_s}.")
        if self.free_play_deg < POSITION_REPORT_RESOLUTION_DEG:
            raise ValueError(
                f"free_play_deg must be at least the reporting resolution "
                f"({POSITION_REPORT_RESOLUTION_DEG}), got {self.free_play_deg}. "
                "A threshold finer than the rotor can report cannot be measured."
            )

    def ticks_for(self, interval_s: float) -> int:
        """How many ticks the window spans at a given cadence.

        At least one, so a cadence slower than the window still watches
        across a real interval rather than comparing a reading with
        itself.
        """
        if interval_s <= 0.0:
            raise ValueError(f"interval_s must be positive, got {interval_s}.")
        return max(1, math.ceil(self.window_s / interval_s))

    def latency_s(self, interval_s: float) -> float:
        """How long detection really takes at a given cadence.

        Rounded up to whole ticks, so it is the number the gain clamp
        should be derived from rather than :attr:`window_s` itself.
        """
        return self.ticks_for(interval_s) * interval_s


class StallDetector:
    """Watches setpoint-versus-reality, tick by tick.

    Mutable and per-run, unlike :class:`StallGuard` which is policy.
    Kept separate from :class:`~qsorbit.core.pointing.TrackingLoop` so
    the detection rule can be tested without a target, an observer, a
    clock or a rotor.

    Args:
        guard: The policy to apply.
        interval_s: The tracking cadence, which decides how many ticks
            the guard's window spans.
    """

    def __init__(self, guard: StallGuard | None = None, interval_s: float = 1.0) -> None:
        self._guard = guard if guard is not None else StallGuard()
        self._ticks = self._guard.ticks_for(interval_s)
        self._interval_s = interval_s
        self._history: deque[tuple[Position, Position]] = deque(maxlen=self._ticks + 1)
        self._stalled = False
        self._armed_axes: set[str] = set()
        self._stalled_axes: tuple[str, ...] = ()
        self._stalled_at: Position | None = None
        self._events = 0

    @property
    def guard(self) -> StallGuard:
        """The policy this detector is applying."""
        return self._guard

    def rescaled(self, interval_s: float) -> StallDetector:
        """A fresh detector with the same policy at a new cadence.

        Used when a tracking profile switch changes the interval. The
        guard's window is a **duration**, so the number of ticks it
        spans changes with the cadence and the deque has to be rebuilt.

        **History is deliberately not carried over.** The samples in it
        were taken at the old cadence, so the same number of them covers
        a different span of time; mixing the two would make the window
        neither duration. Starting clean costs one detection window of
        blindness right after a switch, which is the honest price of
        having changed the thing the window is measured in. Arming is
        lost with it, so each axis has to be seen following again before
        it is judged -- the same standing start the loop already handles
        at the beginning of every run.
        """
        return StallDetector(self._guard, interval_s)

    @property
    def ticks(self) -> int:
        """How many ticks of evidence this detector needs."""
        return self._ticks

    @property
    def latency_s(self) -> float:
        """How long detection takes at the cadence this was built for."""
        return self._guard.latency_s(self._interval_s)

    @property
    def is_stalled(self) -> bool:
        """Whether an axis is currently believed to be jammed."""
        return self._stalled

    @property
    def armed_axes(self) -> frozenset[str]:
        """Which axes have been seen to follow, and are therefore judged.

        An axis stays unarmed -- and unjudged -- until one window shows
        it making real progress. Exposed because "the guard never fired"
        and "the guard was never watching" are different states, and an
        operator chasing a missed stall needs to be able to tell them
        apart.
        """
        return frozenset(self._armed_axes)

    @property
    def stalled_axes(self) -> tuple[str, ...]:
        """Which axes are currently stuck, e.g. ``("elevation",)``.

        Empty when nothing is stalled. Worth naming rather than
        reporting a bare boolean: "elevation is not following" and
        "azimuth is not following" point at different mechanical causes,
        and the operator is the one who has to go and look.
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
            # Nothing commanded yet, so nothing can have failed to follow.
            self._history.clear()
            return False

        self._history.append((commanded, reported))
        if len(self._history) <= self._ticks:
            return False

        oldest_commanded, oldest_reported = self._history[0]
        newest_commanded, newest_reported = self._history[-1]

        # Per axis, never pooled. Taking the larger of the two movements
        # would let a healthy azimuth mask a jammed elevation, which is
        # the *likely* case rather than an exotic one: elevation carries
        # the gravity load, has the higher stiction breakaway, and
        # Session 32 measured its up and down runs as separate
        # experiments for exactly that reason.
        stalled = []
        for axis in AXES:
            verdict = self._judge(
                oldest_commanded, newest_commanded, oldest_reported, newest_reported, axis
            )
            if verdict == "armed":
                # Judging restarts from a window that lies entirely
                # after the axis was seen following, rather than one
                # still straddling its standing start.
                self._history.clear()
                return False
            if verdict == "stuck":
                stalled.append(axis)
        stalled = tuple(stalled)
        if stalled:
            self._stalled = True
            self._stalled_axes = stalled
            self._stalled_at = reported
            self._events += 1
            self._history.clear()
        return self._stalled

    def _judge(
        self,
        oldest_commanded: Position,
        newest_commanded: Position,
        oldest_reported: Position,
        newest_reported: Position,
        axis: str,
    ) -> str:
        """Classify one axis over the window: ``"ok"``, ``"armed"`` or ``"stuck"``.

        **An axis is not judged until it has been seen to follow.** A
        standing start looks exactly like a jam to the shortfall test:
        the axis has to break stiction while the setpoint is already
        walking away, so it sheds several degrees before it is properly
        moving. Two bench runs on 2026-09-02 declared a stall within
        eight seconds of starting, on axes that were visibly
        accelerating -- run A shed 3.50° during acquisition against the
        3.70° a genuine jam shed later the same evening. The numbers do
        not separate, because the difference is not inside the window:
        acquisition recovers and a jam does not.

        So the detector arms per axis on the first window showing real
        progress, and only then starts judging. The gap this leaves is
        worth stating: **an axis already jammed before the track begins
        is never flagged.** That case is visible to the operator the
        moment nothing moves, and it is not the one this guard exists
        for -- a snag developing mid-pass is.

        **The comparison is the shortfall, not the progress.** An earlier
        version asked whether net progress was under the free play, using
        the same threshold on both sides of the test — and a bench run
        caught it declaring a stall on a healthy axis 53 seconds before
        anything was touched. At the shipped cadence the commanded
        position moves in roughly 3° jumps, so a window holding exactly
        one command has ``advance`` a little over the 3° threshold while
        a normally-lagging axis nets a little under it: both halves true
        at once, on a rotor that was tracking perfectly. It fired the
        first moment the gate opened, which is the signature of a
        threshold artefact rather than a jam.

        Asking how far the axis has *fallen behind* removes it. Steady
        tracking carries a constant lag rather than a growing one, so the
        shortfall stays near zero however long the pass runs; a jammed
        axis sheds the whole advance and the shortfall grows with it.
        Progress in the wrong direction counts double, correctly — an
        axis drifting away from its setpoint is not following it.
        """
        advance = _signed(newest_commanded, oldest_commanded, axis)
        if abs(advance) <= self._guard.free_play_deg:
            # The setpoint has not asked for more than the slop, so
            # there is nothing here to fail at. Note this also means
            # detection is rate-limited: a slow target cannot advance
            # past the play inside the window, and no jam can be seen
            # until it does. That is a consequence of having 3° of
            # backlash, not something this code can decide away.
            return "ok"
        progress = _signed(newest_reported, oldest_reported, axis)
        toward = progress if advance > 0.0 else -progress

        if axis not in self._armed_axes:
            if toward > self._guard.free_play_deg:
                self._armed_axes.add(axis)
                return "armed"
            return "ok"

        shortfall = abs(advance) - toward
        return "stuck" if shortfall > self._guard.free_play_deg else "ok"

    def _check_recovery(self, reported: Position) -> None:
        """Clear the stall once the axis is seen to move again.

        Measured from where the axis was when the stall was declared,
        not across a rolling window — see the module docstring on why
        recovery cannot reuse the detection test. The threshold is the
        free play again, because an axis that has only moved within its
        own backlash has not actually gone anywhere.
        """
        if self._stalled_at is None:
            return
        # Every axis that stalled has to move, not just one of them: a
        # freed azimuth says nothing about an elevation still jammed,
        # and the setpoint stays frozen while either is stuck.
        if all(
            abs(_signed(reported, self._stalled_at, axis)) > self._guard.free_play_deg
            for axis in self._stalled_axes
        ):
            self._stalled = False
            self._stalled_axes = ()
            self._stalled_at = None
            self._history.clear()
