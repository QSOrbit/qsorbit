"""Tests for the receive path — the vertical slice, asserted offline.

The whole chain runs here with no hardware of any kind: a fake device
handing out synthetic IQ, a stand-in for the tracking loop, a recording
stand-in for the audio device, and range rates the test dictates.

**What these tests are for, stated precisely so they are not read as
claiming more.** ``test_tuning.py`` already proves the Doppler
*arithmetic* against a real orbital profile. What was untested until now
is the **wiring** — that each block is corrected at its own midpoint,
that the correction changes as the pass does, and that the sign survives
the trip from a range rate through a tracker into a demodulator config.
So the carrier here sits at a known offset and the range rates are
scripted; the assertions are about which way the correction moves and
whether it moves at all, not about recovering a modulated signal. A real
pass is Chunk H's bench day, and nothing here substitutes for it.

**Two things have to be dated deliberately or these tests assert
nothing.** Block timestamps come from an injected clock, because a real
one would date the blocks at whatever instant the suite happened to run
while the range-rate samples sit in 2026 — making the extrapolation run
forwards or backwards depending on the hour. And the audio double is a
synchronisation barrier as well as a recorder: stepping the fake device
proves a block reached the *queue*, not that another thread demodulated
it.

**No sleeps anywhere.** The fake device parks on an event the test
controls, exactly as ``tests/unit/sdr/test_stream.py`` does, so "the
session has demodulated three blocks" is a state the test arranges rather
than one it waits for.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from qsorbit.core.dsp.demod import NbfmConfig
from qsorbit.core.dsp.squelch import NoiseSquelch
from qsorbit.core.dsp.tuning import DopplerTracker
from qsorbit.core.geometry import AzEl
from qsorbit.core.receive import (
    AUDIO_SUBSCRIBER,
    WATERFALL_SUBSCRIBER,
    LoopRangeRate,
    ReceiveSession,
    TargetRangeRate,
)
from qsorbit.core.sdr import AppliedSettings, DeviceError, IqStream, SdrConfig
from qsorbit.core.tracker.state import TopocentricState

#: Small rates, so a "block" is a few thousand samples rather than a
#: hundred thousand and the tests stay quick. The ratios are what the
#: chain cares about; the absolute numbers only have to be legal.
SAMPLE_RATE_HZ = 256_000.0
IF_RATE_HZ = 32_000.0
AUDIO_RATE_HZ = 32_000.0
BLOCK_SAMPLES = 8_192
BLOCK_BYTES = BLOCK_SAMPLES * 2

#: The downlink under test, and where the tuner sits relative to it.
DOWNLINK_HZ = 145_950_000.0
TUNING_OFFSET_HZ = 50_000.0
CENTER_HZ = DOWNLINK_HZ - TUNING_OFFSET_HZ

AN_INSTANT = datetime(2026, 8, 24, 18, 30, 0, tzinfo=UTC)


class BlockClock:
    """The wall clock ``IqStream`` stamps blocks with, advanced by hand.

    **Injected rather than left real, and the tests below do not work
    without it.** A block's timestamp is what the Doppler tracker
    extrapolates to, so a real clock would put block times at whatever
    instant the suite happened to run and the extrapolation would be
    against range-rate samples dated somewhere else entirely — sometimes
    forwards, sometimes backwards, depending on the hour. That is not a
    slow test or a flaky one, it is a test whose *direction* depends on
    when it runs.
    """

    def __init__(self, start: datetime = AN_INSTANT, step_s: float = 1.0) -> None:
        self._now = start
        self._step = timedelta(seconds=step_s)

    def __call__(self) -> datetime:
        self._now += self._step
        return self._now


def an_nbfm_config(**overrides) -> NbfmConfig:
    defaults = {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "if_rate_hz": IF_RATE_HZ,
        "audio_rate_hz": AUDIO_RATE_HZ,
    }
    return NbfmConfig(**{**defaults, **overrides})


def applied_settings() -> AppliedSettings:
    config = SdrConfig(center_hz=CENTER_HZ, sample_rate_hz=SAMPLE_RATE_HZ, gain_db=32.8)
    return AppliedSettings(
        requested=config,
        center_hz=CENTER_HZ,
        sample_rate_hz=SAMPLE_RATE_HZ,
        gain_db=32.8,
        manual_gain=True,
        ppm=0,
        agc_enabled=False,
    )


def carrier_block(offset_hz: float, index: int) -> bytes:
    """One block of uint8 IQ holding a carrier at ``offset_hz`` from centre.

    Phase is advanced across blocks by ``index`` so consecutive blocks
    join up rather than each restarting at zero — a discontinuity every
    block would be a broadband click the discriminator would faithfully
    reproduce, and would make any audio assertion meaningless.
    """
    start = index * BLOCK_SAMPLES
    n = np.arange(start, start + BLOCK_SAMPLES, dtype=np.float64)
    phase = 2.0 * np.pi * offset_hz * n / SAMPLE_RATE_HZ
    # 100 rather than 127 so the tone has headroom and nothing clips at
    # the ADC's rails, which would add harmonics we did not ask for.
    i = np.round(127.5 + 100.0 * np.cos(phase)).astype(np.uint8)
    q = np.round(127.5 + 100.0 * np.sin(phase)).astype(np.uint8)
    interleaved = np.empty(BLOCK_SAMPLES * 2, dtype=np.uint8)
    interleaved[0::2] = i
    interleaved[1::2] = q
    return interleaved.tobytes()


def noise_block(index: int, *, seed: int = 7) -> bytes:
    """One block of uint8 IQ holding broadband noise, no carrier at all.

    Where :func:`carrier_block` is a clean tone -- deliberately, so it
    reliably opens a squelch on the first block -- this is its opposite:
    a channel with nothing in it, for tests that need the gate to stay
    *closed*. ``index`` seeds the generator so consecutive blocks differ
    (a repeating block would not be noise), while staying reproducible
    run to run.
    """
    rng = np.random.default_rng(seed + index)
    return rng.integers(0, 256, size=BLOCK_SAMPLES * 2, dtype=np.uint8).tobytes()


class SteppedFakeDevice:
    """A device the test advances one block at a time.

    Each block carries a carrier at whatever offset the test asks for, so
    a sequence of steps can walk a signal across the passband the way a
    real Doppler shift does. ``block_fn``, when given, overrides *what*
    each block contains (its content, not its timing) -- used by the
    squelch tests below, which need a block that stays a closed channel
    rather than a carrier that would open the gate on the first read.
    """

    def __init__(
        self, offsets_hz: list[float], *, block_fn: Callable[[int], bytes] | None = None
    ) -> None:
        self.index = 0
        self.is_open = True
        self.applied = applied_settings()
        self.reads = 0
        self.done = False
        self._offsets = offsets_hz
        self._block_fn = block_fn
        self.allow = threading.Event()
        self.ready = threading.Event()

    def read_raw(self, length: int) -> bytes:
        self.ready.set()
        self.allow.wait(5.0)
        self.allow.clear()
        if self.done or self.reads >= len(self._offsets):
            raise DeviceError("stepped fake device exhausted")
        if self._block_fn is not None:
            block = self._block_fn(self.reads)
        else:
            block = carrier_block(self._offsets[self.reads], self.reads)
        self.reads += 1
        return block

    def step(self) -> None:
        self.ready.clear()
        self.allow.set()
        assert self.ready.wait(5.0), "the reader never came back for another block"

    def finish(self) -> None:
        self.done = True
        self.allow.set()


class RecordingAudio:
    """An AudioOutput-shaped double that keeps what was written to it.

    Also the tests' **synchronisation barrier**, which is the less
    obvious half of its job. ``SteppedFakeDevice.step()`` guarantees a
    block reached the queue, not that anything demodulated it — those
    are different threads — so a test that reads the Doppler statistics
    straight after a step is reading whatever the demodulating thread
    happened to have finished. :meth:`wait_for` waits on the audio
    actually arriving, which is the end of the chain and therefore the
    only honest place to say "that block is done".
    """

    def __init__(self) -> None:
        self.blocks: list[np.ndarray] = []
        self.started = False
        self.stopped = False
        self._condition = threading.Condition()

    def start(self) -> None:
        self.started = True

    def write(self, samples: np.ndarray) -> None:
        with self._condition:
            self.blocks.append(samples)
            self._condition.notify_all()

    def wait_for(self, count: int, timeout_s: float = 5.0) -> bool:
        """Block until ``count`` blocks have been demodulated and written."""
        with self._condition:
            return self._condition.wait_for(lambda: len(self.blocks) >= count, timeout_s)

    def stop(self):
        self.stopped = True
        return self.stats

    @property
    def stats(self):
        from qsorbit.core.dsp.audio import AudioStats

        return AudioStats(
            blocks_written=len(self.blocks),
            blocks_played=len(self.blocks),
            blocks_dropped=0,
            frames_played=sum(block.size for block in self.blocks),
            underruns=0,
        )


class ScriptedRangeRate:
    """A range-rate source the test drives by hand."""

    def __init__(self, samples: list[tuple[datetime, float]]) -> None:
        self._samples = list(samples)
        self.primed = False
        self.pending: list[tuple[datetime, float]] = []

    def prime(self) -> tuple[datetime, float]:
        self.primed = True
        return self._samples[0]

    def sample(self) -> tuple[datetime, float] | None:
        if not self.pending:
            return None
        return self.pending.pop(0)


class FakeTarget:
    """A target whose range rate the test dictates. Satisfies ``Target``."""

    name = "FAKE-1"

    def __init__(self, range_rate_km_s: float = -3.0) -> None:
        self.range_rate_km_s = range_rate_km_s
        self.calls = 0

    def topocentric_state(self, observer: object, time: datetime) -> TopocentricState:
        self.calls += 1
        return TopocentricState(
            sky_position=AzEl(azimuth=120.0, elevation=30.0),
            range_km=1_000.0,
            range_rate_km_s=self.range_rate_km_s,
        )


def a_session(
    device: SteppedFakeDevice, source, *, clock: BlockClock | None = None, **overrides
) -> tuple[ReceiveSession, RecordingAudio]:
    """Build a session over a stepped device, with everything faked."""
    audio = RecordingAudio()
    stream = IqStream(device, block_bytes=BLOCK_BYTES, queue_blocks=8, now=clock or BlockClock())
    session = ReceiveSession(
        stream=stream,
        nbfm=an_nbfm_config(),
        doppler=DopplerTracker(DOWNLINK_HZ, CENTER_HZ),
        audio=audio,
        range_rate=source,
        **overrides,
    )
    return session, audio


def quietly_stop(session: ReceiveSession) -> None:
    """Stop a session whose fake device has run out, ignoring that fact.

    A session reports why its blocks stopped, which is the whole point
    of :meth:`ReceiveSession.stop` re-raising — but for a test about
    something else entirely, the fake running dry is the teardown rather
    than the subject.
    """
    try:
        session.stop()
    except DeviceError:
        pass


class TestTargetRangeRate:
    def test_it_computes_a_sample_from_the_target_with_no_rotor_anywhere(self):
        target = FakeTarget(range_rate_km_s=-4.5)
        source = TargetRangeRate(target, observer=object(), now=lambda: AN_INSTANT)

        when, range_rate = source.sample()

        assert when == AN_INSTANT
        assert range_rate == -4.5

    def test_prime_and_sample_are_the_same_operation(self):
        target = FakeTarget()
        source = TargetRangeRate(target, observer=object(), now=lambda: AN_INSTANT)

        assert source.prime() == source.sample()


class FakeLoop:
    """A TrackingLoop-shaped double, so no rotor is involved."""

    def __init__(self) -> None:
        self.ticks = 0
        self.latest_sample = None
        self._time = AN_INSTANT

    def tick(self):
        self.ticks += 1
        self._time += timedelta(seconds=1)
        self.latest_sample = _sample_at(self._time, -3.0)
        return self.latest_sample

    def publish(self, when: datetime, range_rate_km_s: float) -> None:
        """Simulate somebody else — a readout widget — ticking the loop."""
        self.latest_sample = _sample_at(when, range_rate_km_s)


def _sample_at(when: datetime, range_rate_km_s: float):
    from qsorbit.core.pointing import TickOutcome, TrackSample
    from qsorbit.core.rotor import Position

    return TrackSample(
        time=when,
        sky_position=AzEl(azimuth=120.0, elevation=30.0),
        range_km=1_000.0,
        range_rate_km_s=range_rate_km_s,
        rotor_target=Position(azimuth=120.0, elevation=30.0),
        rotor_position=Position(azimuth=120.0, elevation=30.0),
        outcome=TickOutcome.COMMANDED,
    )


class TestLoopRangeRate:
    def test_driving_ticks_the_loop(self):
        loop = FakeLoop()
        source = LoopRangeRate(loop, drive=True)

        source.sample()
        source.sample()

        assert loop.ticks == 2

    def test_following_never_ticks_the_loop(self):
        # Two things ticking one loop would double the rotor's serial
        # traffic and interleave two streams of commands. With a window
        # open, ReadoutWidget owns the tick.
        loop = FakeLoop()
        source = LoopRangeRate(loop, drive=False)
        source.prime()
        loop.publish(AN_INSTANT + timedelta(seconds=5), -2.0)

        source.sample()
        source.sample()

        assert loop.ticks == 1  # the priming tick, and only that

    def test_following_reports_a_sample_once_and_then_nothing_until_it_moves(self):
        loop = FakeLoop()
        source = LoopRangeRate(loop, drive=False)
        source.prime()
        loop.publish(AN_INSTANT + timedelta(seconds=5), -2.0)

        first = source.sample()
        second = source.sample()

        assert first == (AN_INSTANT + timedelta(seconds=5), -2.0)
        assert second is None

    def test_priming_ticks_even_when_following(self):
        # Priming has to produce a sample and, before the window is up,
        # there is nobody else to produce one. The alternative is
        # starting a pass with no correction for the first second.
        loop = FakeLoop()

        LoopRangeRate(loop, drive=False).prime()

        assert loop.ticks == 1


class TestSessionWiring:
    def test_it_subscribes_twice_under_the_documented_names(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, -3.0)])
        session, _ = a_session(device, source)

        names = [entry.name for entry in session.stats.stream.subscribers]

        assert names == [AUDIO_SUBSCRIBER, WATERFALL_SUBSCRIBER]

    def test_the_tracker_is_primed_before_anything_streams(self):
        # offset_at() raises if it has never been given a range rate, and
        # the demodulating thread can reach its first block before the
        # tracking side produces anything. Priming removes the race
        # rather than instrumenting it.
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, -3.0)])
        session, _ = a_session(device, source)

        session.start()
        try:
            assert source.primed
            assert session.stats.doppler.updates == 1
        finally:
            device.finish()
            quietly_stop(session)

    def test_starting_twice_is_refused(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, -3.0)])
        session, _ = a_session(device, source)
        session.start()
        try:
            with pytest.raises(RuntimeError, match="already been started"):
                session.start()
        finally:
            device.finish()
            quietly_stop(session)

    def test_a_non_positive_tracking_interval_is_refused(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, -3.0)])

        with pytest.raises(ValueError, match="tracking_interval_s"):
            a_session(device, source, tracking_interval_s=0.0)


class TestDemodulation:
    def test_every_block_is_demodulated_and_written_out(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ] * 3)
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source)

        session.start()
        try:
            for _ in range(3):
                device.step()
            assert audio.wait_for(3), "not every block was demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        assert len(audio.blocks) == 3
        assert session.stats.blocks_demodulated == 3

    def test_the_recovered_audio_is_the_right_length_and_finite(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source)

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        recovered = audio.blocks[0]
        # 8192 samples at 256 kHz filtered to 32 kHz is 1024 IF samples,
        # and the audio decimation factor is 1 with these rates -- but
        # the result is 1023, not 1024. discriminate() works on adjacent
        # PAIRS, so it drops the one straddling the block boundary. That
        # is the same behaviour Session 20 measured as making phase
        # continuity across blocks unnecessary (-89 dBFS), so it is
        # asserted here rather than rounded past.
        assert recovered.shape == (BLOCK_SAMPLES // 8 - 1,)
        assert np.all(np.isfinite(recovered))

    def test_a_device_failure_reaches_the_caller_rather_than_dying_quietly(self):
        # A receive session that stops for a reason nobody is told is
        # the failure mode this project keeps meeting.
        device = SteppedFakeDevice([])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, _ = a_session(device, source)

        session.start()
        device.finish()
        assert session.wait(5.0), "the demodulating thread never noticed the stream ending"

        with pytest.raises(DeviceError, match="exhausted"):
            session.stop()


class TestDopplerFollowsThePass:
    def run_a_turnover(self) -> tuple[float, float, ReceiveSession]:
        """Demodulate one block approaching, then one receding.

        Blocks land at AN_INSTANT +1 s and +2 s; the range-rate samples
        sit at AN_INSTANT and +1.5 s, so the second block is genuinely
        extrapolating forward from a slope that has turned over. Both
        halves have to be dated deliberately or the test asserts nothing
        about direction.
        """
        device = SteppedFakeDevice([TUNING_OFFSET_HZ] * 2)
        source = ScriptedRangeRate([(AN_INSTANT, -6.0)])
        session, audio = a_session(device, source, clock=BlockClock(step_s=1.0))

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the first block was never demodulated"
            first = session.stats.doppler.last_offset_hz

            # The pass turns over: approaching becomes receding.
            session._doppler.update(AN_INSTANT + timedelta(seconds=1.5), +6.0)
            device.step()
            assert audio.wait_for(2), "the second block was never demodulated"
            second = session.stats.doppler.last_offset_hz
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()
        return first, second, session

    def test_the_correction_tracks_a_range_rate_that_changes_between_blocks(self):
        # The point of the whole chunk: the correction has to FOLLOW,
        # not merely be applied once. A pass goes from approaching to
        # receding, so the offset must sweep in the same direction.
        first, second, _ = self.run_a_turnover()

        # Approaching puts the downlink HIGH, receding puts it LOW. If
        # the sign were flipped this assertion is the one that catches
        # it, and a flipped Doppler sign is invisible in the audio.
        assert first > TUNING_OFFSET_HZ
        assert second < TUNING_OFFSET_HZ

    def test_the_offset_range_spans_the_pass_rather_than_one_instant(self):
        _, _, session = self.run_a_turnover()

        stats = session.stats.doppler
        assert stats.min_offset_hz is not None
        assert stats.max_offset_hz is not None
        assert stats.max_offset_hz - stats.min_offset_hz > 1_000.0


class TestStatsPresentation:
    def test_describe_names_every_section_including_the_absent_ones(self):
        # "The squelch was off" and "the squelch never opened" are
        # different facts, and an omitted line reads as the second.
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source)

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        text = session.stats.describe()

        assert "--- iq ---" in text
        assert "--- audio ---" in text
        assert "--- doppler ---" in text
        assert "squelch: off" in text
        assert "no waterfall was attached" in text


class TestLiveQuieting:
    """Chunk I: :attr:`ReceiveSession.live_quieting_db` and
    :attr:`live_squelch_open`, and the ``mute_squelch`` wiring they sit
    beside.

    ``test_demod.py`` already proves the decoupling arithmetic --
    ``mute=False`` still measures and decides, it just does not silence.
    What is untested until now is that :class:`ReceiveSession` actually
    threads its ``mute_squelch`` constructor argument down to that call,
    and that the two live properties read the real squelch rather than a
    stale or mismatched copy of it. So these tests are about the wiring,
    exactly as the rest of this module states its own scope.
    """

    def test_no_squelch_means_no_live_reading_at_all(self):
        # "No squelch" and "a squelch that has not opened yet" have to
        # read differently, or a caller cannot tell them apart.
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source)

        assert session.live_quieting_db is None
        assert session.live_squelch_open is None

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        assert session.live_quieting_db is None
        assert session.live_squelch_open is None

    def test_a_strong_signal_opens_the_gate_and_reports_a_live_measurement(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        squelch = NoiseSquelch()
        session, audio = a_session(device, source, squelch=squelch)

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        assert session.live_quieting_db is not None
        assert session.live_squelch_open is True
        # mute_squelch defaults to True, but the gate opened on this very
        # block -- apply() runs after update() inside the same call, so an
        # opening block is never muted against itself.
        assert np.abs(audio.blocks[0]).max() > 0.0

    def test_an_empty_channel_stays_closed_and_mutes_by_default(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ], block_fn=noise_block)
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        squelch = NoiseSquelch()
        session, audio = a_session(device, source, squelch=squelch)

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        assert session.live_quieting_db is not None
        assert session.live_squelch_open is False
        assert not audio.blocks[0].any()

    def test_mute_squelch_false_still_reports_the_closed_decision_but_lets_audio_through(self):
        # The point of the whole item: the live readout has to be honest
        # about a gate that WOULD have muted, even on a run where
        # mute_squelch=False means nothing actually gets silenced.
        device = SteppedFakeDevice([TUNING_OFFSET_HZ], block_fn=noise_block)
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        squelch = NoiseSquelch()
        session, audio = a_session(device, source, squelch=squelch, mute_squelch=False)

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        assert session.live_quieting_db is not None
        assert session.live_squelch_open is False
        assert np.abs(audio.blocks[0]).max() > 0.0
