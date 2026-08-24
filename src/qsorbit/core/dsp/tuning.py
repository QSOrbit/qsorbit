"""Doppler-corrected tuning: turning a tracking loop's range rate into a mixer offset.

**The tuning strategy, settled in Chunk G and measured rather than assumed.**
A satellite's downlink drifts as it passes. There are two ways to follow it:
retune the SDR's hardware centre frequency continuously, or leave the
hardware alone and follow the signal digitally within the captured
bandwidth. This module implements the second, and the reasoning is worth
keeping because "retune the radio" is the intuitive answer:

- **The excursion is tiny compared to the capture.** The worst realistic
  case measured — the ISS on 70 cm — is +/-10.4 kHz against a 2.048 MHz
  capture. The signal never approaches the edge of the window, so the
  hardware has nothing it *needs* to do.
- **The tuner PLL quantises.** Chunk C established this and it is why
  :meth:`~qsorbit.core.sdr.device.AppliedSettings.offset_from` measures
  against the centre actually reached rather than the one requested.
  Retuning every tick drags that rounding error through the audio
  continuously.
- **Retuning disturbs a real-time path.** Session 16's finding, by name:
  anything on the read path is on a real-time path, and that lesson cost
  9.5% of every capture before it was found.
- **The DC spike follows the centre frequency.** Every capture, both listen
  scripts, and ``qsorbit sdr capture`` deliberately tune off-centre for
  exactly this reason; retuning would drag the receiver's own artifact
  across the band while trying to track a signal through it.

**How often the digital shift updates, also measured.** The tracking loop
ticks once a second (:data:`~qsorbit.core.pointing.DEFAULT_TICK_INTERVAL_S`)
while IQ blocks arrive roughly every 64 ms — about sixteen blocks per tick.
Holding one mixer frequency per tick leaves a **170 Hz sawtooth with a step
once per second**; recomputing per block, by extrapolating between ticks,
collapses that to **7.8 Hz with no step at all** — 22x tighter, 10 dB less
audible-band error, for the cost of one scalar per block either way. Going
further, to a per-sample chirp within each block, buys only another 2.7 dB,
because at that point extrapolation error dominates rather than within-block
drift. So: **per block, evaluated at the block's midpoint, linearly
extrapolated from the last two ticks.**

**What this module deliberately does not do.** It never imports
:class:`~qsorbit.core.pointing.TrackSample`, and takes a plain
``(time, range_rate_km_s)`` pair instead. Importing ``core.pointing`` would
drag in ``core.rotor`` and therefore pyserial, and importing
``core.tracker`` would drag in skyfield — either of which would make the
whole of ``core/dsp/`` unimportable wherever those are missing, which is the
Session 18 ``sounddevice`` lesson applied to our own packages. Keeping
``core/dsp/`` dependent on nothing beyond numpy and scipy is what makes it
the best-tested layer in the project. The caller unpacks the two fields it
needs from a ``TrackSample``; that is one line at the call site and a whole
dependency edge avoided.

**And what it does not need**, which was measured and expected to go the
other way: the mixer needs **no phase continuity across blocks**.
:func:`~qsorbit.core.dsp.demod.shift_to_baseband` restarts its phasor at
zero for every block, and the resulting discontinuity is invisible — because
each block is demodulated independently and
:func:`~qsorbit.core.dsp.demod.discriminate` returns one fewer sample than it
is given, so the one sample pair that would straddle the boundary is never
computed. The discontinuity falls exactly in the gap the discriminator
already leaves. Measured at **-89 dBFS**, which is float32 dust, 25 dB below
this module's own residual. So ``shift_to_baseband`` stays a pure, stateless
function, and this module is the only thing here that holds state.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from qsorbit.core.doppler import downlink_receive_frequency

#: How far past its newest sample the tracker will keep extrapolating
#: before it stops trusting the slope and holds. Three missed ticks. At
#: the worst measured Doppler rate (190 Hz/s) three seconds of
#: extrapolation is under 600 Hz of error — about 12% of full-scale audio
#: on a 5 kHz-deviation channel, degraded but far from broken. Past that a
#: stale slope applied forever diverges faster than a stale value does,
#: so the tracker holds and says so rather than confidently inventing a
#: frequency from data that stopped arriving.
DEFAULT_MAX_EXTRAPOLATION_S: Final = 3.0


class DopplerError(RuntimeError):
    """Raised when a :class:`DopplerTracker` is asked for a frequency
    before it has been given any range rate to work from.

    A programming error rather than a runtime condition — it means the
    receive path started demodulating before the tracking loop produced
    its first sample — so it raises rather than guessing a frequency.
    """


@dataclass(frozen=True)
class DopplerStats:
    """What one tracker did, and the range of correction it applied.

    Args:
        updates: Range-rate samples fed in via :meth:`DopplerTracker.update`.
        queries: Times a frequency or offset was asked for.
        stale_queries: Of those, how many were further past the newest
            sample than ``max_extrapolation_s`` allows, so the slope was
            no longer trusted. **Not a cosmetic counter**: a non-zero
            value means the tracking loop stopped feeding this tracker
            while audio kept being demodulated, which degrades quietly
            and would otherwise look like the receiver drifting off for
            no reason.
        min_offset_hz: Smallest channel offset produced, or ``None``.
        max_offset_hz: Largest, or ``None``. The pair brackets how far
            the correction actually moved during a pass, which is the
            number to sanity-check against the expected Doppler
            excursion for the band.
        last_offset_hz: Most recent offset, or ``None``.
    """

    updates: int
    queries: int
    stale_queries: int
    min_offset_hz: float | None
    max_offset_hz: float | None
    last_offset_hz: float | None

    def describe(self) -> str:
        """Return a short human-readable summary, for logs and reports."""
        if self.last_offset_hz is None:
            span = "  correction: never queried"
        else:
            span = (
                f"  correction: {self.min_offset_hz / 1e3:+.2f} to "
                f"{self.max_offset_hz / 1e3:+.2f} kHz "
                f"(span {(self.max_offset_hz - self.min_offset_hz) / 1e3:.2f} kHz), "
                f"{self.last_offset_hz / 1e3:+.2f} kHz last"
            )
        stale = (
            "  all queries used a fresh range rate"
            if self.stale_queries == 0
            else (
                f"  {self.stale_queries:,} of {self.queries:,} query(ies) ran on a STALE "
                f"range rate -- the tracking loop stopped feeding this tracker"
            )
        )
        return (
            f"doppler: {self.updates:,} range-rate update(s), {self.queries:,} query(ies)\n"
            f"{span}\n{stale}"
        )


class DopplerTracker:
    """Turns a tracking loop's range-rate samples into a per-block mixer offset.

    Stateful because extrapolation needs the previous two samples; every
    other correction step in :mod:`qsorbit.core.dsp` is a pure function,
    and this one is the exception for the same reason
    :class:`~qsorbit.core.dsp.squelch.NoiseSquelch` is — it has to remember
    something between blocks.

    **Thread-safe, which the other stateful objects here are not, because
    this one genuinely spans two threads.** In the ``receive`` path the
    tracking loop calls :meth:`update` from wherever it ticks, while the
    demodulator calls :meth:`offset_at` per block from another thread.
    Without a lock the reader can see a half-written sample list — the new
    sample appended but the old one not yet dropped — and a wrong slope is
    a wrong correction that raises nothing and sounds merely a bit worse.
    The lock is held only around list and counter access, never around
    anything that blocks, so the demodulator cannot stall the tracking
    loop or the reverse. :class:`~qsorbit.core.dsp.squelch.NoiseSquelch`
    deliberately does *not* get one: it lives entirely on the
    demodulating thread, and a lock there would suggest a sharing that
    does not happen.

    Usage, one tracker per pass::

        tracker = DopplerTracker(transmit_hz=145_950_000.0, center_hz=applied.center_hz)
        # ... on each tracking-loop tick:
        tracker.update(sample.time, sample.range_rate_km_s)
        # ... on each IQ block, at the block's midpoint time:
        config = replace(base_config, channel_offset_hz=tracker.offset_at(block_mid_time))

    Args:
        transmit_hz: The satellite's nominal downlink frequency, as
            transmitted, in Hz. Not the frequency you expect to hear —
            that is what this class computes.
        center_hz: The centre frequency the tuner **actually reached**,
            from :attr:`~qsorbit.core.sdr.device.AppliedSettings.center_hz`,
            not the one requested. The PLL quantises, and an offset
            computed against a frequency the radio never reached is wrong
            by exactly the amount nobody thinks to check.
        max_extrapolation_s: See :data:`DEFAULT_MAX_EXTRAPOLATION_S`.

    Raises:
        ValueError: If ``transmit_hz`` is not a positive finite number, if
            ``center_hz`` is not finite, or if ``max_extrapolation_s`` is
            not positive and finite.
    """

    def __init__(
        self,
        transmit_hz: float,
        center_hz: float,
        *,
        max_extrapolation_s: float = DEFAULT_MAX_EXTRAPOLATION_S,
    ) -> None:
        if not math.isfinite(transmit_hz) or transmit_hz <= 0.0:
            raise ValueError(f"transmit_hz must be a positive, finite number, got {transmit_hz!r}.")
        if not math.isfinite(center_hz):
            raise ValueError(f"center_hz must be finite, got {center_hz!r}.")
        if not math.isfinite(max_extrapolation_s) or max_extrapolation_s <= 0.0:
            raise ValueError(
                f"max_extrapolation_s must be a positive, finite number, "
                f"got {max_extrapolation_s!r}."
            )

        self._transmit_hz = transmit_hz
        self._center_hz = center_hz
        self._max_extrapolation_s = max_extrapolation_s

        # Guards everything below it. A plain Lock rather than an RLock:
        # offset_at() needs frequency_at()'s arithmetic, so that
        # arithmetic lives in an unlocked private method that both public
        # entry points call while holding the lock. An RLock would have
        # worked too, and would have hidden which method owns the
        # counting -- and the counting is the part that has already been
        # wrong once (see frequency_at).
        self._lock = threading.Lock()

        # The two newest samples, oldest first, as (time, received_hz).
        self._samples: list[tuple[datetime, float]] = []
        self._updates = 0
        self._queries = 0
        self._stale_queries = 0
        self._min_offset_hz: float | None = None
        self._max_offset_hz: float | None = None
        self._last_offset_hz: float | None = None

    @property
    def transmit_hz(self) -> float:
        """The satellite's nominal downlink frequency, in Hz."""
        return self._transmit_hz

    @property
    def center_hz(self) -> float:
        """The tuner's actual centre frequency, in Hz."""
        return self._center_hz

    @property
    def has_samples(self) -> bool:
        """``True`` once at least one range rate has been supplied."""
        with self._lock:
            return bool(self._samples)

    def update(self, time: datetime, range_rate_km_s: float) -> None:
        """Supply one tracking-loop sample.

        Args:
            time: When the range rate was computed for — from
                :attr:`~qsorbit.core.pointing.TrackSample.time`. Must be
                timezone-aware and no earlier than the previous sample.
            range_rate_km_s: Positive when receding, the convention
                :class:`~qsorbit.core.tracker.state.TopocentricState` uses.

        Raises:
            ValueError: If ``range_rate_km_s`` is not finite, if ``time``
                is naive, or if ``time`` is earlier than the newest
                sample already held — time running backwards would invert
                the extrapolation slope, which is a sign flip by another
                route.
        """
        if not math.isfinite(range_rate_km_s):
            raise ValueError(f"range_rate_km_s must be finite, got {range_rate_km_s!r}.")
        if time.tzinfo is None:
            raise ValueError(
                "time must be timezone-aware; a naive datetime has no defined instant."
            )
        # Computed outside the lock: it is pure arithmetic on arguments,
        # and doing it inside would hold the lock across work the reader
        # is waiting on for no reason.
        received_hz = downlink_receive_frequency(self._transmit_hz, range_rate_km_s)

        with self._lock:
            if self._samples and time < self._samples[-1][0]:
                raise ValueError(
                    f"time went backwards: {time.isoformat()} is earlier than the previous "
                    f"sample at {self._samples[-1][0].isoformat()}. Extrapolating across that "
                    f"would invert the slope and correct the wrong way."
                )
            # Append and trim under one lock. Separately, a reader could
            # catch the list three entries long and take the wrong pair
            # as its slope.
            self._samples.append((time, received_hz))
            del self._samples[:-2]
            self._updates += 1

    def frequency_at(self, time: datetime) -> float:
        """The frequency the downlink is expected at, in Hz, at ``time``.

        Linearly extrapolated from the two newest samples. With only one
        sample there is no slope, so that sample's frequency is used
        unchanged — correct for the first tick of a pass, and the reason
        no correction is worse than a guessed slope.

        Args:
            time: The instant to evaluate at — for a block of IQ, its
                **midpoint**, which removes a systematic half-block bias
                for free.

        Returns:
            The expected received frequency in Hz.

        Raises:
            DopplerError: If no sample has been supplied yet.
        """
        with self._lock:
            return self._frequency_at(time)

    def _frequency_at(self, time: datetime) -> float:
        """:meth:`frequency_at`'s arithmetic. **The lock must be held.**"""
        if not self._samples:
            raise DopplerError(
                "No range rate has been supplied yet -- call update() with a tracking-loop "
                "sample before asking for a frequency. Demodulation started before the "
                "loop produced its first tick."
            )

        # Every public query counts exactly once, and counts here rather
        # than in each caller. offset_at() delegates to this method, so
        # counting in both would report more stale queries than queries --
        # "4 of 1 query(ies) ran on a STALE range rate", which is what an
        # earlier version of this class actually printed.
        self._queries += 1

        newest_time, newest_hz = self._samples[-1]
        elapsed_s = (time - newest_time).total_seconds()

        if len(self._samples) < 2:
            return newest_hz

        older_time, older_hz = self._samples[0]
        span_s = (newest_time - older_time).total_seconds()
        if span_s <= 0.0:
            # Two samples at the same instant carry no slope. Not an
            # error -- a loop that ticked twice in the same microsecond
            # is odd, not broken -- so fall back to the newest value.
            return newest_hz

        if elapsed_s > self._max_extrapolation_s:
            self._stale_queries += 1
            elapsed_s = self._max_extrapolation_s

        slope_hz_s = (newest_hz - older_hz) / span_s
        return newest_hz + slope_hz_s * elapsed_s

    def offset_at(self, time: datetime) -> float:
        """The channel offset to hand :class:`~qsorbit.core.dsp.demod.NbfmConfig`.

        Args:
            time: As :meth:`frequency_at` — the block's midpoint.

        Returns:
            Where the downlink sits relative to the tuner's own baseband,
            in Hz, positive when above centre. The same sense as
            :meth:`~qsorbit.core.sdr.device.AppliedSettings.offset_from`
            and as
            :attr:`~qsorbit.core.dsp.demod.NbfmConfig.channel_offset_hz`,
            deliberately: three places using one convention is what stops
            a sign being flipped in translation between them.

        Raises:
            DopplerError: If no sample has been supplied yet.
        """
        with self._lock:
            offset_hz = self._frequency_at(time) - self._center_hz
            self._last_offset_hz = offset_hz
            self._min_offset_hz = (
                offset_hz if self._min_offset_hz is None else min(self._min_offset_hz, offset_hz)
            )
            self._max_offset_hz = (
                offset_hz if self._max_offset_hz is None else max(self._max_offset_hz, offset_hz)
            )
            return offset_hz

    def is_stale_at(self, time: datetime) -> bool:
        """``True`` if ``time`` is further past the newest sample than allowed.

        Lets a caller notice a stalled tracking loop without waiting to
        read :attr:`stats` at the end of a pass.
        """
        with self._lock:
            if not self._samples:
                return True
            return (time - self._samples[-1][0]).total_seconds() > self._max_extrapolation_s

    @property
    def stats(self) -> DopplerStats:
        """The run's statistics so far.

        Taken under the lock, so the six numbers are a consistent
        snapshot rather than six reads that could straddle an update.
        """
        with self._lock:
            return DopplerStats(
                updates=self._updates,
                queries=self._queries,
                stale_queries=self._stale_queries,
                min_offset_hz=self._min_offset_hz,
                max_offset_hz=self._max_offset_hz,
                last_offset_hz=self._last_offset_hz,
            )
