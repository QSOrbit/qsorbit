"""Per-satellite profile data: what a satellite transmits, and how alive it is.

The config-boundary test from Session 8 draws the line for this module
exactly where it draws it for everything else: *does this change when I
point at a different satellite?* Downlink frequency, mode, and whether
anyone needs to be running a QSO through it for you to hear anything --
all yes, all live here. Where the antenna is and how far the rotor may
turn stay in :mod:`qsorbit.core.station`, which is the other side of
that same line.

Two things worth knowing before reading :class:`SatelliteProfile`:

* **Reliability is a property of a transmitter, not a satellite.**
  RS-44 carries both an unconditional CW beacon and a linear
  transponder that stays silent unless another station is working
  through it -- the same bird, two different answers to "will this be
  transmitting." A single reliability field on the profile would have
  to pick one and be wrong about the other, so it lives on
  :class:`Transmitter` instead; :meth:`SatelliteProfile.best_reliability`
  reduces that back to one number for a caller (the future target
  picker's "does this need anyone's cooperation" filter) that only
  cares about the most favorable case.
* **The tier-1 alive field is a static, curated fact, not a live
  measurement.** It is one operator's best knowledge as of a date, with
  a source -- "NOAA-19's transmitter went dark on 2025-08-19, per
  AMSAT's status board" is exactly the kind of fact this exists to
  hold, cheaply, so a pass prediction never again promises a satellite
  that no longer transmits. Tiers 2 (crowd-sourced "heard lately") and
  3 (this station's own log) are parked -- see :class:`AliveRecord`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class Mode(Enum):
    """The RF modulation a transmitter uses.

    A small, curated vocabulary rather than a free-form string, matching
    every other enumerated field a station config or profile validates
    strictly elsewhere in this project -- new modes get added here
    deliberately as the curated catalogue needs them, rather than typoed
    into existence in a TOML file with nothing to catch it.
    """

    CW = "cw"
    FM = "fm"
    SSB = "ssb"
    AFSK1200 = "afsk1200"
    BPSK = "bpsk"
    SSTV = "sstv"


class ReliabilityClass(Enum):
    """How likely a transmitter is to actually be transmitting.

    The three-tier model from Session 21: this is about *operator*
    dependency, nothing else. A beacon that only runs in sunlight (like
    AO-73's) is still :data:`UNCONDITIONAL` by this axis -- eclipse is a
    physical constraint the satellite manages on its own, not a human
    being who has to show up. That distinction is what makes this field
    useful for the target picker's first and biggest filter, "does
    receiving this require somebody to transmit."
    """

    #: Transmits whether or not anyone is working it. If the satellite
    #: is alive and above the horizon, it is transmitting -- a pass
    #: with one of these cannot come back empty for reasons outside
    #: this station's control.
    UNCONDITIONAL = "unconditional"

    #: Transmits on its own schedule or in response to its own
    #: triggers (an SSTV sequence, an announced event) -- needs nobody
    #: else's cooperation, but isn't always on the way a beacon is.
    SCHEDULED = "scheduled"

    #: Silent unless another station is actively using it -- a
    #: repeater, digipeater, or transponder. A quiet pass is
    #: indistinguishable from a fault without knowing this.
    DEPENDENT = "dependent"


class AliveStatus(Enum):
    """A curated, as-of-date judgment of whether a satellite still transmits."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AliveRecord:
    """Tier 1 of the three-tier alive-status model (Session 21): a static fact.

    ``AliveRecord`` is a value object: immutable and comparable by
    value.

    Args:
        status: The curated judgment.
        as_of: The date this judgment was last checked against a
            source -- not when the satellite launched or was last
            definitely heard, but when a human last looked. A pass
            prediction that trusts a stale record is trusting a fact
            nobody has re-checked in a while, and this is what makes
            that visible.
        source: Where the judgment came from, in enough detail to
            re-check it -- a URL, a publication, or "this station's own
            log" once tier 3 lands. Required, on the same reasoning
            station config rejects an unknown key: an alive-status
            field with no source is a belief with no way to verify it.

    Raises:
        ValueError: If ``source`` is empty.
    """

    status: AliveStatus
    as_of: date
    source: str

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError(
                "source must not be empty -- an unsourced alive status can't be checked."
            )


@dataclass(frozen=True)
class Transmitter:
    """One thing a satellite transmits (or a repeater/transponder it carries).

    ``Transmitter`` is a value object: immutable and comparable by
    value.

    Args:
        downlink_hz: Downlink frequency, in hertz. For a transponder
            with a passband rather than a single frequency, this is the
            passband's center -- the actual range belongs in ``notes``,
            since modeling a full passband is more structure than any
            profile in this chunk's curated starter set needs.
        mode: The RF modulation.
        reliability: How likely this specific transmitter is to be
            transmitting -- see :class:`ReliabilityClass`.
        uplink_hz: Uplink frequency, in hertz, for a repeater or
            transponder. ``None`` for a one-way beacon or SSTV
            transmission.
        baud: Symbol rate, for a digital mode. ``None`` where it
            doesn't apply (CW, FM voice, SSTV) or hasn't been confirmed
            against a primary source.
        notes: Free-text detail that doesn't fit a typed field --
            access tones, arming procedures, schedule caveats, the
            actual span of a transponder passband. This is deliberately
            prose rather than more enumerated fields: the target
            picker's own design goal (Session 21) is a shortlist that
            "says what you would hear and why it might be silent," and
            that reasoning reads better as a sentence than as another
            enum.

    Raises:
        ValueError: If ``downlink_hz``, ``uplink_hz``, or ``baud`` is
            not positive.
    """

    downlink_hz: float
    mode: Mode
    reliability: ReliabilityClass
    uplink_hz: float | None = None
    baud: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.downlink_hz <= 0.0:
            raise ValueError(f"downlink_hz must be positive, got {self.downlink_hz}.")
        if self.uplink_hz is not None and self.uplink_hz <= 0.0:
            raise ValueError(f"uplink_hz must be positive, got {self.uplink_hz}.")
        if self.baud is not None and self.baud <= 0.0:
            raise ValueError(f"baud must be positive, got {self.baud}.")


#: Order used by :meth:`SatelliteProfile.best_reliability` to pick the
#: single most favorable transmitter on a satellite -- lower index wins.
_RELIABILITY_ORDER = (
    ReliabilityClass.UNCONDITIONAL,
    ReliabilityClass.SCHEDULED,
    ReliabilityClass.DEPENDENT,
)


@dataclass(frozen=True)
class SatelliteProfile:
    """Everything QSOrbit needs to know about one satellite that isn't its orbit.

    ``SatelliteProfile`` is a value object: immutable and comparable by
    value. The orbit itself -- the TLE -- is deliberately not part of
    this: elements go stale in days to weeks and this profile's data
    does not, so the two are fetched and refreshed on entirely different
    schedules. Matching a profile to a TLE by ``norad_id`` is the
    caller's job, typically the CLI wiring that ties this chunk
    together.

    Args:
        norad_id: The NORAD catalog number -- the stable key this
            profile is looked up by, and the same number a TLE's own
            line 1 carries, which is what lets the two be matched.
        name: The satellite's primary display name, e.g. ``"RS-44"``.
        transmitters: What this satellite sends, and how reliably.
            May be empty for a satellite whose profile exists only to
            record its alive status (e.g. a confirmed-dead bird, kept
            around so it's excluded rather than silently absent).
        alive: The tier-1 alive-status record.
        also_known_as: Other names this satellite is published under
            -- OSCAR designations, international designators, a
            manufacturer's name. Defaults to empty.

    Raises:
        ValueError: If ``norad_id`` is not positive, or ``name`` is
            empty.
    """

    norad_id: int
    name: str
    transmitters: tuple[Transmitter, ...]
    alive: AliveRecord
    also_known_as: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.norad_id <= 0:
            raise ValueError(f"norad_id must be positive, got {self.norad_id}.")
        if not self.name.strip():
            raise ValueError("name must not be empty.")

    def best_reliability(self) -> ReliabilityClass | None:
        """The most favorable :class:`ReliabilityClass` across this profile's transmitters.

        Answers the target picker's first filter axis (Session 21):
        "does receiving *something* from this satellite require
        somebody to transmit?" -- RS-44 has both an unconditional
        beacon and a dependent transponder, and this returns
        :data:`ReliabilityClass.UNCONDITIONAL` because the beacon alone
        already answers "no."

        Returns:
            The best (lowest-dependency) reliability class among
            :attr:`transmitters`, or ``None`` if there are none.
        """
        if not self.transmitters:
            return None
        present = {transmitter.reliability for transmitter in self.transmitters}
        for candidate in _RELIABILITY_ORDER:
            if candidate in present:
                return candidate
        raise AssertionError(  # pragma: no cover - every ReliabilityClass is in _RELIABILITY_ORDER
            f"a transmitter reliability class was not in _RELIABILITY_ORDER: {present}"
        )
