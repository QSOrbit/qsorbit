"""Turns sky positions into rotor pointing commands, and keeps pointing.

This is the layer between "where the satellite is" and "what the rotor
gets told to do." Those two things are numerically identical today,
which is exactly why the seam needs to exist now: everything that will
eventually make them differ is hardware-specific, and none of it belongs
inside the tracker.

It is also the orchestration layer. :class:`TrackingLoop` is the thing
that owns "we are currently tracking AO-91": it samples a
:class:`~qsorbit.core.tracker.Target` on a cadence, converts each sample
through :func:`sky_to_rotor`, and commands the rotor. Every tick emits a
:class:`TrackSample`, which is the seam the rest of the application
consumes — a display reads sky position against rotor axis position, and
Doppler correction will read range rate off the same sample.

What is still expected to land here:

* **Alignment calibration offset** — the correction measured by the
  auto-calibration routine, applied to every commanded position.
* **Flip-mode selection** — for a pass crossing near zenith, commanding
  ``azimuth + 180`` and ``180 - elevation`` instead of whipping the
  azimuth axis 180° through the top. Only available on rotors whose
  elevation travel exceeds 90°.
* **Azimuth unwrapping** — on a rotor with more than 360° of azimuth
  travel, choosing ``azimuth`` or ``azimuth + 360`` so a pass crossing
  north doesn't hit a cable-wrap stop partway through.
* **Travel limits and obstruction masks** — refusing or clamping
  positions the hardware can't reach or the horizon blocks.

Note that flip selection and azimuth unwrapping cannot be decided from a
single instant: they depend on the shape of the whole pass, and on where
the rotor already is. So this module is expected to grow a stateful
planner that runs once before a pass begins and a cheap per-sample
converter that follows the resulting plan. Both of those need pass
prediction (AOS/LOS), which is deferred, so today only the seam exists —
:func:`sky_to_rotor` is deliberately an identity conversion.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from qsorbit.core.geometry import AzEl
from qsorbit.core.rotor import Position, Rotor, format_set_position
from qsorbit.core.tracker import ObserverLocation, Target

#: How often :meth:`TrackingLoop.run` samples the target, in seconds.
#:
#: A low-Earth satellite's apparent motion peaks around 1°/s near
#: zenith, so a one-second cadence keeps the error contributed by
#: sampling alone below the rotor's own steady-state shortfall. It is
#: also fast enough for live Doppler correction to read range rate off
#: the same samples, and it matches the rate a readout display polls at.
DEFAULT_TICK_INTERVAL_S = 1.0

#: The controller firmware's own position dead-zone, in degrees.
#:
#: ``POSITION_DEADZONE`` in the SatNOGS firmware: it stops driving an
#: axis once the error falls below this. A software deadband smaller
#: than this cannot change what the hardware does, which makes it a floor
#: rather than a target — see :class:`TrackingLoop` for what the useful
#: value actually is.
FIRMWARE_DEADZONE_DEG = 0.2


class PointingError(Exception):
    """Base exception for the pointing layer."""


class TravelGuardError(PointingError):
    """Raised when the rotor *reports* a position outside its declared travel.

    Distinct from :class:`~qsorbit.core.rotor.PositionLimitError`, which
    means QSOrbit refused to *send* a position. This one means the rotor
    says it is already somewhere it should never have been — so QSOrbit's
    picture of the hardware and the hardware's own reading have diverged,
    and further commands built on that picture cannot be trusted.

    The likely causes are all worth stopping for: the axis was moved by
    something other than QSOrbit, the controller was power-cycled or
    re-homed underneath a running track, or the travel declared in the
    station config doesn't match the rotator it is pointed at.
    """


class TickOutcome(Enum):
    """What a single tracking tick did about the rotor."""

    #: A new position was sent.
    COMMANDED = "commanded"

    #: The target moved less than the deadband since the last command, so
    #: nothing was sent. The normal outcome for most ticks.
    WITHIN_DEADBAND = "within_deadband"

    #: The target is below the horizon, so nothing was sent. Expected
    #: rather than exceptional: with pass prediction deferred, a track is
    #: started by hand and usually starts before the target rises.
    BELOW_HORIZON = "below_horizon"


@dataclass(frozen=True)
class TrackSample:
    """One tick of a track: where the target is, where the rotor is, what was done.

    ``TrackSample`` is a value object: immutable and comparable by value.
    It is the seam between the tracking loop and everything watching it,
    which is why it carries more than the loop itself needs — range rate
    is here for Doppler correction, and the sky/axis pair is here so a
    display can show them as the distinct things they are.

    Args:
        time: The instant this sample was computed for, timezone-aware.
        sky_position: Where the target actually is, as astronomical
            truth — capped at 90° elevation by
            :class:`~qsorbit.core.geometry.AzEl`.
        range_km: Straight-line distance to the target, in kilometers.
        range_rate_km_s: Rate of change of range, in kilometers per
            second. Positive is receding. This is what Doppler correction
            consumes.
        rotor_target: The axis position :func:`sky_to_rotor` produced for
            :attr:`sky_position`. Computed every tick, whether or not it
            was sent.
        rotor_position: The axis position the rotor reported this tick,
            measured from wherever it homed. Not a compass bearing, and
            no calibration is applied. Expect it to trail
            :attr:`rotor_target` by a degree or two even on a healthy
            rotor — that is stiction, not an error.
        outcome: What the loop did about it.
    """

    time: datetime
    sky_position: AzEl
    range_km: float
    range_rate_km_s: float
    rotor_target: Position
    rotor_position: Position
    outcome: TickOutcome

    @property
    def commanded(self) -> bool:
        """``True`` if this tick sent a new position to the rotor."""
        return self.outcome is TickOutcome.COMMANDED


def sky_to_rotor(sky: AzEl) -> Position:
    """Convert a sky direction into a rotor command position.

    .. note::

       **No correction is currently applied.** This is a straight
       one-to-one conversion, so the rotor is commanded to the raw
       computed sky position. Any mechanical misalignment of your mast
       shows up directly as pointing error, and a pass crossing near
       zenith will swing the azimuth axis through the top rather than
       flipping. Interfaces that expose pointing to an operator should
       say so rather than implying the output is calibrated.

    It is a named function rather than an inline construction so that
    there is exactly one place for alignment offsets, flip mode,
    unwrapping, and limit handling to be added — and so no caller can
    reach the rotor without passing through it. See this module's
    docstring for what's coming.

    Args:
        sky: Where the target actually is, as seen from the observer.

    Returns:
        The position to command the rotor to.

    Raises:
        ValueError: If the resulting position is outside what a
            :class:`~qsorbit.core.rotor.Position` permits.
    """
    return Position(azimuth=sky.azimuth, elevation=sky.elevation)


def rotor_to_sky(position: Position) -> AzEl:
    """Convert a rotor axis reading into the sky direction it would mean.

    .. note::

       **No correction is currently applied**, exactly as in
       :func:`sky_to_rotor` — this is that conversion's inverse, and
       carries the same honesty. It says where an aligned, non-flipped
       installation would be pointing for this axis reading; it does
       not know this rig's actual alignment offset, because none is
       measured yet.

    A :class:`~qsorbit.core.rotor.Position` is a mechanical axis
    reading, and its domain is wider than :class:`AzEl`'s: multi-turn
    azimuth travel makes readings past 360° ordinary, a freshly homed
    axis commonly settles a degree or two past zero (``AZ-1.5 EL2.0``
    has been observed at the bench), and a rotor whose elevation travel
    passes vertical can report an axis angle ``AzEl`` cannot represent
    at all. This function never raises on a reading a real rotor can
    produce:

    * **Azimuth** is taken modulo 360°. This is not an approximation —
      the compass direction a rotor physically points depends only on
      where the axis ends up, not on how many extra turns of cable it
      took to get there.
    * **Elevation** is clamped to ``[-90.0, 90.0]``. Once flip mode
      lands (see this module's docstring), an axis reading past
      vertical will mean a real, different sky direction reached by
      way of ``azimuth + 180``, and this function will need to know
      that to convert it correctly. Until then, a reading outside
      ``AzEl``'s range means either ordinary stiction overshoot at a
      declared limit or a rotor whose extra travel this function
      doesn't yet understand — clamping is the same "be honest about a
      reading rather than crash on one" policy
      :class:`TrackingLoop`'s own travel guard already applies; it is
      not a claim that the clamped value is where the antenna truly
      points.

    This is :func:`rotor_to_sky`'s first real consumer:
    :mod:`qsorbit.ui.readout_widget` uses it to show what a
    :class:`TrackSample`'s :attr:`~TrackSample.rotor_position` means as
    a sky direction, next to :attr:`~TrackSample.sky_position` — the
    same rotor-vs-sky distinction :class:`TrackSample` already exists
    to preserve.

    Args:
        position: An axis reading, e.g. from
            :meth:`~qsorbit.core.rotor.Rotor.read_position`.

    Returns:
        The sky direction this reading would correspond to in an
        aligned, non-flipped installation.
    """
    return AzEl(
        azimuth=position.azimuth % 360.0,
        elevation=max(-90.0, min(90.0, position.elevation)),
    )


def compute_pointing_command(target: Target, observer: ObserverLocation, time: datetime) -> bytes:
    """Compute the rotor command that would point at ``target``.

    Args:
        target: What to point at — anything satisfying
            :class:`~qsorbit.core.tracker.Target`, which today means a
            :class:`~qsorbit.core.tracker.Satellite`.
        observer: The ground station's location.
        time: The instant to compute for, as a timezone-aware datetime.

    .. warning::

       The result is **not range-checked against any particular rotor.**
       It says where the target is, expressed as a command; it does not
       promise the hardware can go there. Anything that actually drives a
       rotor must check the position against that rotor's declared
       :class:`~qsorbit.core.rotor.RotorCapabilities` first, because the
       controller firmware will accept and attempt whatever it is sent.

    Returns:
        The command bytes to send to the rotor, e.g.
        ``b"AZ180.0 EL45.0\\n"``. A set-position command draws no reply
        from the firmware.

    Raises:
        ValueError: If ``time`` is naive (has no ``tzinfo``).
        PropagationError: If SGP4 cannot compute a valid position at
            ``time``.
    """
    state = target.topocentric_state(observer, time)
    return bytes(format_set_position(sky_to_rotor(state.sky_position)))


def _utc_now() -> datetime:
    """The current instant, timezone-aware. Injected so tests can fake it."""
    return datetime.now(UTC)


class TrackingLoop:
    """Keeps a rotor pointed at a moving target.

    Everything before this was a single instant: work out where the
    satellite is *now*, print a command, exit. This is what makes it a
    pass. Each tick samples the target, reads the rotor, decides whether
    the change is worth a command, and sends one if it is.

    **The deadband is the acceptance window, not the firmware's
    dead-zone.** The firmware stops driving an axis below
    :data:`FIRMWARE_DEADZONE_DEG` (0.2°), so a smaller software deadband
    cannot change what the hardware does. But the real floor is higher:
    with stock gains an axis settles one to two degrees short of target
    as a matter of course, and
    :attr:`~qsorbit.core.rotor.RotorCapabilities.acceptance_window_deg`
    is the distance within which the rotor is already considered to have
    arrived. Commanding a change smaller than that asks the motors to
    chatter over a move QSOrbit itself would call arrived, so the
    acceptance window is the default deadband.

    The comparison is against the **last commanded position**, not
    against what the rotor reports. Comparing against the reading would
    chase the stiction residual forever: the axis stops short, that
    shortfall looks like real movement, and the loop re-commands the same
    position on every tick.

    **Below the horizon is not an error.** With pass prediction deferred,
    a track is started by hand, usually a few minutes early, so the first
    ticks compute positions below the horizon. Those emit samples and
    command nothing. A refusal for a target that *is* above the horizon
    is a different matter: it means the geometry is outside what this
    rotor may do, so
    :class:`~qsorbit.core.rotor.PositionLimitError` propagates and stops
    the loop rather than leaving the antenna parked while the application
    still looks like it is tracking.

    **Cable wrap is guarded by the rotor's own reading.** Integration
    rule 2.6 asks the pointing layer to track cumulative net rotation,
    because the travel limits (~540° azimuth, 360° elevation) are
    measured from home rather than per command. The counter that rule
    asks for already exists in the hardware: the axis reading accumulates
    past a full turn — a boot has been observed reporting ``AZ -377.10``
    — and homing resets it, so the reported position *is* net rotation
    since home. Combined with the per-command check
    :meth:`~qsorbit.core.rotor.RotorCapabilities.check_setpoint` already
    performs, cable wrap is bounded by the declared travel. So no second
    counter is kept here; it could only drift from the first. What the
    loop adds instead is a check that the two have not diverged: every
    tick asserts the *reported* position is inside declared travel —
    widened by the acceptance window, because a freshly homed axis
    legitimately reports slightly past its zero — and raises
    :class:`TravelGuardError` if it isn't.

    Ending a track does **not** stop the rotor, matching
    :meth:`Rotor.__exit__ <qsorbit.core.rotor.Rotor.__exit__>`: a move
    already in progress does not need the link to finish, ``SA SE`` is
    not an emergency stop, and abandoning the antenna mid-slew is no
    improvement on letting it arrive.

    Args:
        target: What to track — anything satisfying
            :class:`~qsorbit.core.tracker.Target`.
        observer: The ground station's location.
        rotor: A connected rotor. The loop neither connects nor closes
            it; whoever owns the connection owns its lifecycle.
        interval_s: Seconds between ticks in :meth:`run`.
        deadband_deg: Minimum change, in degrees on either axis, before a
            new position is sent. Defaults to the rotor's declared
            acceptance window. Values below
            :data:`FIRMWARE_DEADZONE_DEG` are accepted but pointless.
        now: Returns the current instant, timezone-aware. Injected for
            testing.
        sleep: Injected for testing; defaults to :func:`time.sleep`.
        monotonic: Injected for testing; defaults to
            :func:`time.monotonic`.

    Raises:
        ValueError: If ``interval_s`` is not positive, or if
            ``deadband_deg`` is negative.
    """

    def __init__(
        self,
        target: Target,
        observer: ObserverLocation,
        rotor: Rotor,
        *,
        interval_s: float = DEFAULT_TICK_INTERVAL_S,
        deadband_deg: float | None = None,
        now: Callable[[], datetime] = _utc_now,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_s <= 0.0:
            raise ValueError(f"interval_s must be positive, got {interval_s}.")
        if deadband_deg is None:
            deadband_deg = rotor.capabilities.acceptance_window_deg
        if deadband_deg < 0.0:
            raise ValueError(f"deadband_deg must not be negative, got {deadband_deg}.")

        self._target = target
        self._observer = observer
        self._rotor = rotor
        self._interval_s = interval_s
        self._deadband_deg = deadband_deg
        self._now = now
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_commanded: Position | None = None
        self._latest_sample: TrackSample | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def target(self) -> Target:
        """What is being tracked."""
        return self._target

    @property
    def deadband_deg(self) -> float:
        """How far the target must move before a new command is sent."""
        return self._deadband_deg

    @property
    def latest_sample(self) -> TrackSample | None:
        """The most recent tick's sample, or ``None`` before the first tick.

        This is what a display polls. A window on its own timer can read
        it without driving the loop, and a window that drives the loop
        itself through :meth:`tick` gets the same object back.
        """
        return self._latest_sample

    @property
    def last_commanded(self) -> Position | None:
        """The last position actually sent, or ``None`` if none has been."""
        return self._last_commanded

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def tick(self) -> TrackSample:
        """Sample the target once, and command the rotor if it has moved enough.

        Deliberately free of any waiting, so it can be driven by
        something else's clock — a UI timer, a test's fake clock, or
        :meth:`run`.

        The rotor is read *before* anything is commanded, so the sample
        shows where the axes were when the decision was made, and so the
        travel guard fires before a command rather than after one.

        Returns:
            What this tick saw and did.

        Raises:
            TravelGuardError: If the rotor reports a position outside its
                declared travel.
            PositionLimitError: If a target above the horizon converts to
                a position outside declared travel. Nothing is
                transmitted, and the loop stops.
            SerialConnectionError: If the port is not open.
            ProtocolError: If a reply can't be parsed.
            PropagationError: If the target's position can't be computed.
        """
        now = self._now()
        state = self._target.topocentric_state(self._observer, now)
        rotor_position = self._rotor.read_position()
        self._guard_reported_position(rotor_position)

        rotor_target = sky_to_rotor(state.sky_position)
        outcome = self._decide(state.sky_position, rotor_target)
        if outcome is TickOutcome.COMMANDED:
            self._rotor.move_to(rotor_target)
            self._last_commanded = rotor_target

        sample = TrackSample(
            time=now,
            sky_position=state.sky_position,
            range_km=state.range_km,
            range_rate_km_s=state.range_rate_km_s,
            rotor_target=rotor_target,
            rotor_position=rotor_position,
            outcome=outcome,
        )
        self._latest_sample = sample
        return sample

    def run(
        self,
        *,
        max_ticks: int | None = None,
        duration_s: float | None = None,
    ) -> Iterator[TrackSample]:
        """Tick on a cadence, yielding every sample.

        Stopping is the caller's business: break out of the iteration and
        the loop ends there, having sent nothing further. The rotor is
        deliberately left as it is — see the class docstring.

        The cadence is measured from the start rather than from the end
        of each tick, so the serial round trips a tick costs don't make
        the interval drift longer and longer. If a tick overruns the
        interval the next one starts immediately, rather than trying to
        catch up on ticks that were missed: a late pointing update is
        worth sending, a stale one isn't. The wait happens *before* each
        tick after the first, so a bounded run doesn't sit sleeping on
        its way out.

        Args:
            max_ticks: Stop after this many ticks. ``None`` runs until
                the caller stops it.
            duration_s: Stop once this much time has elapsed. ``None``
                runs until the caller stops it.

        Yields:
            One :class:`TrackSample` per tick.
        """
        started = self._monotonic()
        next_tick_at = started
        ticks = 0
        while True:
            if max_ticks is not None and ticks >= max_ticks:
                return
            if ticks:
                remaining = next_tick_at - self._monotonic()
                if remaining > 0.0:
                    self._sleep(remaining)
                else:
                    next_tick_at = self._monotonic()
            if duration_s is not None and self._monotonic() - started >= duration_s:
                return

            yield self.tick()
            ticks += 1
            next_tick_at += self._interval_s

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _decide(self, sky: AzEl, rotor_target: Position) -> TickOutcome:
        """Decide what this tick should do about the rotor."""
        if sky.elevation < 0.0:
            return TickOutcome.BELOW_HORIZON
        if self._last_commanded is None:
            return TickOutcome.COMMANDED
        moved = max(
            abs(rotor_target.azimuth - self._last_commanded.azimuth),
            abs(rotor_target.elevation - self._last_commanded.elevation),
        )
        if moved < self._deadband_deg:
            return TickOutcome.WITHIN_DEADBAND
        return TickOutcome.COMMANDED

    def _guard_reported_position(self, position: Position) -> None:
        """Raise if the rotor says it is well outside its own declared travel.

        Deliberately **not**
        :meth:`~qsorbit.core.rotor.RotorCapabilities.check_setpoint`,
        which answers a different question. That one asks "may we send
        this?" and is exact, because a setpoint is a choice. This asks
        "is the rotor plausibly where it says it is?" about a
        *measurement*, and measurements sit outside the declared range
        for ordinary reasons: a freshly homed axis settles slightly past
        its zero and reports ``AZ -1.5``, and an axis commanded to a
        declared limit stops a degree or two either side of it. Checking
        a reading exactly would turn a healthy rotor into an error on the
        first tick.

        So the declared travel is widened by the acceptance window — the
        same allowance the capability record already makes for stiction —
        and only a reading past *that* counts as divergence. The limits
        and the window are hardware facts read from the record; how much
        slack to allow before abandoning a track is this loop's policy,
        which is why it lives here.
        """
        window = self._rotor.capabilities.acceptance_window_deg
        axes = (
            (
                "Azimuth",
                position.azimuth,
                self._rotor.capabilities.azimuth_min_deg,
                self._rotor.capabilities.azimuth_max_deg,
            ),
            (
                "Elevation",
                position.elevation,
                self._rotor.capabilities.elevation_min_deg,
                self._rotor.capabilities.elevation_max_deg,
            ),
        )
        for axis, value, low, high in axes:
            if low - window <= value <= high + window:
                continue
            raise TravelGuardError(
                f"The rotor reports an {axis.lower()} axis reading of {value} degrees, "
                f"which is outside its declared travel of {low} to {high} (plus the "
                f"{window} degree acceptance window). Tracking stopped: QSOrbit's "
                "picture of where the rotor is and the rotor's own reading have "
                "diverged, so no further command can be trusted. Check whether the "
                "axis was moved by something other than QSOrbit, whether the "
                "controller re-homed, and whether the travel declared in the station "
                "config matches this rotator."
            )
