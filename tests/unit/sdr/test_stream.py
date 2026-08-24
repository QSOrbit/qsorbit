"""Tests for the streaming layer and its loss accounting.

Two halves, tested very differently.

:class:`~qsorbit.core.sdr.stream.ThroughputMonitor` is pure arithmetic
over an injected clock, so it is tested exactly — with the real numbers
from bring-up (2.048 Msps, 256 KiB blocks, 64 ms per read) rather than
round invented ones, so a failure reads in units the bench uses.

:class:`~qsorbit.core.sdr.stream.IqStream` runs a real thread, and
thread tests that lean on ``sleep`` are how a suite becomes flaky. So
the fake device here is **event-driven**: it hands out a fixed number of
blocks and then parks on an event the test controls. That makes "the
reader got ahead of the consumer" a thing the test *arranges* rather
than a thing it races for.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from itertools import islice

import pytest

from qsorbit.core.sdr import (
    STALL_THRESHOLD_S,
    AppliedSettings,
    DeviceError,
    IqStream,
    SdrConfig,
    ThroughputMonitor,
    TimedBlock,
    byte_rate_for,
)

#: Bring-up's actual numbers, so failures read in bench units.
SAMPLE_RATE_HZ = 2_048_000
BYTE_RATE = 4_096_000.0
BLOCK_BYTES = 262_144
BLOCK_SECONDS = 0.064


def a_config(**overrides) -> SdrConfig:
    defaults = {
        "center_hz": 99_650_000,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "gain_db": 32.8,
    }
    return SdrConfig(**{**defaults, **overrides})


def applied_settings(sample_rate_hz: float = SAMPLE_RATE_HZ) -> AppliedSettings:
    config = a_config()
    return AppliedSettings(
        requested=config,
        center_hz=config.center_hz,
        sample_rate_hz=sample_rate_hz,
        gain_db=32.8,
        manual_gain=True,
        ppm=0,
        agc_enabled=False,
    )


class Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeDevice:
    """A stand-in :class:`~qsorbit.core.sdr.RtlSdr` for the reader thread.

    Hands out a fixed number of blocks and then raises, which is how a
    real device behaves when it stops delivering and is what makes
    these tests deterministic without a single ``sleep``: the reader
    always terminates on its own, so a test can wait for
    :attr:`drained` and then read the statistics knowing nothing more
    is coming.

    Every block is filled with its own index, so ordering and
    drop-policy assertions can name blocks without the test holding
    payloads.
    """

    def __init__(
        self,
        *,
        blocks: int = 4,
        block_bytes: int = BLOCK_BYTES,
        fail_after: int | None = None,
        sample_rate_hz: float = SAMPLE_RATE_HZ,
        is_open: bool = True,
        applied: AppliedSettings | None = None,
    ) -> None:
        self.index = 0
        self.is_open = is_open
        self.applied = applied if applied is not None else applied_settings(sample_rate_hz)
        self._total = blocks
        self._block_bytes = block_bytes
        self._fail_after = fail_after
        self.reads = 0
        #: Set once every block has been handed out.
        self.drained = threading.Event()

    def read_raw(self, length: int) -> bytes:
        if self._fail_after is not None and self.reads >= self._fail_after:
            raise DeviceError("the fake device gave up")
        if self.reads >= self._total:
            raise DeviceError("fake device exhausted")
        block = bytes([self.reads % 256]) * min(length, self._block_bytes)
        self.reads += 1
        if self.reads >= self._total:
            self.drained.set()
        return block


class TestByteRateFor:
    def test_two_bytes_per_complex_sample(self):
        assert byte_rate_for(SAMPLE_RATE_HZ) == BYTE_RATE

    def test_it_takes_the_actual_rate_not_a_rounded_one(self):
        # The sample clock quantises; accounting against a tidied-up
        # rate measures the quantisation as loss.
        assert byte_rate_for(2_047_992) == 4_095_984


class TestThroughputMonitorArithmetic:
    def test_the_first_read_starts_the_clock_and_is_not_counted(self):
        # It spans device spin-up and the tail of reset_buffer, which is
        # not loss and must not be reported as loss.
        clock = Clock()
        monitor = ThroughputMonitor(BYTE_RATE, clock=clock)

        monitor.record(BLOCK_BYTES)
        report = monitor.report()

        assert report.reads == 0
        assert report.bytes_read == 0
        assert report.elapsed_s == 0.0

    def test_reads_arriving_exactly_on_time_lose_nothing(self):
        clock = Clock()
        monitor = ThroughputMonitor(BYTE_RATE, clock=clock)

        for _ in range(4):
            monitor.record(BLOCK_BYTES)
            clock.advance(BLOCK_SECONDS)
        report = monitor.report()

        assert report.reads == 3
        assert report.bytes_read == 3 * BLOCK_BYTES
        assert report.lost_bytes == pytest.approx(0.0)
        assert report.loss_fraction == pytest.approx(0.0)

    def test_a_stalled_read_is_counted_as_the_samples_it_missed(self):
        clock = Clock()
        monitor = ThroughputMonitor(BYTE_RATE, clock=clock)

        monitor.record(BLOCK_BYTES)  # starts the clock
        for advance in (BLOCK_SECONDS, BLOCK_SECONDS, BLOCK_SECONDS + 0.1):
            clock.advance(advance)
            monitor.record(BLOCK_BYTES)
        report = monitor.report()

        # 100 ms of samples at 4.096 MB/s.
        assert report.lost_bytes == pytest.approx(409_600.0)
        assert report.worst_stall_s == pytest.approx(0.1)

    def test_the_deficits_sum_to_the_aggregate(self):
        # The built-in self-check: the per-read figures are the shape of
        # the same number the aggregate reports, so if these two ever
        # disagree the accounting is broken rather than the device.
        clock = Clock()
        monitor = ThroughputMonitor(BYTE_RATE, clock=clock)

        monitor.record(BLOCK_BYTES)
        for advance in (0.070, 0.055, 0.200, 0.064):
            clock.advance(advance)
            monitor.record(BLOCK_BYTES)
        report = monitor.report()

        assert sum(report.deficits_s) * BYTE_RATE == pytest.approx(report.lost_bytes)

    def test_a_read_that_drains_a_backlog_offsets_one_that_stalled(self):
        # Deficits are deliberately unclamped. The device buffers, so a
        # fast read after a slow one is recovery, not a second event —
        # clamping the negatives away would report healthy buffering as
        # loss.
        clock = Clock()
        monitor = ThroughputMonitor(BYTE_RATE, clock=clock)

        monitor.record(BLOCK_BYTES)
        clock.advance(BLOCK_SECONDS + 0.05)
        monitor.record(BLOCK_BYTES)
        clock.advance(BLOCK_SECONDS - 0.05)
        monitor.record(BLOCK_BYTES)

        assert monitor.report().lost_bytes == pytest.approx(0.0)

    def test_short_reads_are_accounted_by_what_they_actually_returned(self):
        clock = Clock()
        monitor = ThroughputMonitor(BYTE_RATE, clock=clock)

        monitor.record(BLOCK_BYTES)
        clock.advance(BLOCK_SECONDS)
        monitor.record(BLOCK_BYTES // 2)
        report = monitor.report()

        # Half a block delivered in a whole block's worth of time.
        assert report.lost_bytes == pytest.approx(BLOCK_BYTES / 2)

    def test_a_non_positive_byte_rate_is_refused(self):
        with pytest.raises(ValueError, match="byte_rate"):
            ThroughputMonitor(0)


class TestLossReportPresentation:
    def test_stalls_count_only_deficits_past_the_display_threshold(self):
        clock = Clock()
        monitor = ThroughputMonitor(BYTE_RATE, clock=clock)

        monitor.record(BLOCK_BYTES)
        for advance in (BLOCK_SECONDS, BLOCK_SECONDS + STALL_THRESHOLD_S * 2):
            clock.advance(advance)
            monitor.record(BLOCK_BYTES)

        assert monitor.report().stalls == 1

    def test_an_empty_run_reports_no_loss_rather_than_dividing_by_zero(self):
        assert ThroughputMonitor(BYTE_RATE).report().loss_fraction == 0.0

    def test_describe_mentions_the_loss_and_the_worst_stall(self):
        clock = Clock()
        monitor = ThroughputMonitor(BYTE_RATE, clock=clock)
        monitor.record(BLOCK_BYTES)
        clock.advance(BLOCK_SECONDS + 0.1)
        monitor.record(BLOCK_BYTES)

        text = monitor.report().describe()

        assert "lost" in text
        assert "stall" in text


class TestIqStreamConstruction:
    def test_a_closed_device_is_refused(self):
        with pytest.raises(DeviceError, match="not open"):
            IqStream(FakeDevice(is_open=False))

    def test_an_unconfigured_device_is_refused(self):
        # Without a configure() there is no actual sample rate, and
        # without that there is no accounting worth reporting.
        device = FakeDevice()
        device.applied = None

        with pytest.raises(DeviceError, match="never been configured"):
            IqStream(device)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_block_size_is_refused(self, bad):
        with pytest.raises(ValueError, match="block_bytes"):
            IqStream(FakeDevice(), block_bytes=bad)

    @pytest.mark.parametrize("bad", [0, -4])
    def test_a_non_positive_queue_depth_is_refused(self, bad):
        with pytest.raises(ValueError, match="queue_blocks"):
            IqStream(FakeDevice(), queue_blocks=bad)

    def test_the_accounting_uses_the_rate_the_device_reported(self):
        # Not the requested one. a_config() asks for 2_048_000; the
        # device here settled on something else.
        device = FakeDevice(sample_rate_hz=2_047_992)
        stream = IqStream(device)

        assert stream.stats.loss.byte_rate == pytest.approx(4_095_984.0)


class TestIqStreamReading:
    def test_blocks_arrive_in_order(self):
        device = FakeDevice(blocks=4)
        stream = IqStream(device, block_bytes=BLOCK_BYTES, queue_blocks=8)

        collected = []
        with stream:
            for block in stream.blocks(poll_s=0.05):
                collected.append(block[0])
                if len(collected) == 4:
                    break

        assert collected == [0, 1, 2, 3]

    def test_the_reader_starts_on_first_use_without_an_explicit_start(self):
        device = FakeDevice(blocks=1)
        stream = IqStream(device, queue_blocks=4)

        first = next(stream.blocks(poll_s=0.05))
        stream.stop()

        assert first[0] == 0
        assert stream.stats.blocks_read == 1

    def test_starting_twice_is_refused(self):
        stream = IqStream(FakeDevice(blocks=1), queue_blocks=4)
        stream.start()
        try:
            with pytest.raises(DeviceError, match="already been started"):
                stream.start()
        finally:
            stream.stop()

    def test_a_reader_failure_reaches_the_consumer(self):
        # A thread that dies quietly is the worst possible outcome: the
        # stream simply stops and nothing says why.
        device = FakeDevice(blocks=8, fail_after=2)
        stream = IqStream(device, queue_blocks=8)

        with pytest.raises(DeviceError, match="gave up"), stream:
            list(stream.blocks(poll_s=0.05))

    def test_stopping_reports_a_clean_shutdown(self):
        device = FakeDevice(blocks=2)
        stream = IqStream(device, queue_blocks=4)
        stream.start()
        device.drained.wait(5.0)

        stats = stream.stop()

        assert stats.reader_stopped_cleanly
        assert not stream.is_running


def drained_stream(blocks: int, queue_blocks: int) -> IqStream:
    """Run a stream to exhaustion with nothing consuming, and stop it.

    Filling the buffer past its depth is *arranged* rather than raced:
    the reader runs to the end of the fake's data before any consumer
    exists, so the drop count is a fact about the buffer depth and not
    about how the two threads happened to interleave.
    """
    device = FakeDevice(blocks=blocks)
    stream = IqStream(device, block_bytes=BLOCK_BYTES, queue_blocks=queue_blocks)
    stream.start()
    device.drained.wait(5.0)
    stream.stop()
    return stream


class TestIqStreamDropPolicy:
    def test_the_oldest_blocks_are_discarded_when_the_buffer_fills(self):
        stream = drained_stream(blocks=6, queue_blocks=2)

        survivors = []
        # The stream ends the way the fake device does — by failing —
        # so draining it surfaces that. What matters here is *which*
        # blocks were still in the buffer.
        with pytest.raises(DeviceError, match="exhausted"):
            for block in stream.blocks(poll_s=0.05):
                survivors.append(block[0])

        # Six produced into a buffer of two: the two most recent are the
        # survivors. Were the policy drop-newest, this would be [0, 1].
        assert survivors == [4, 5]

    def test_the_dropped_blocks_are_counted(self):
        stats = drained_stream(blocks=6, queue_blocks=2).stats

        assert stats.blocks_read == 6
        assert stats.blocks_dropped == 4
        assert stats.dropped_bytes == 4 * BLOCK_BYTES

    def test_a_buffer_drop_is_not_reported_as_device_loss(self):
        # The distinction the whole module exists to preserve: a slow
        # consumer and a slow USB path are different faults, and only
        # one of them would be fixed by rewriting the reader. Every
        # block here reached us; four were discarded afterwards.
        stats = drained_stream(blocks=6, queue_blocks=2).stats

        assert stats.blocks_dropped == 4
        # The monitor saw all six reads; the first only started its clock.
        assert stats.loss.bytes_read == stats.bytes_read - BLOCK_BYTES

    def test_describe_keeps_the_two_faults_apart(self):
        text = drained_stream(blocks=3, queue_blocks=1).stats.describe()

        assert "at the device (USB)" in text
        assert "at our buffer" in text


#: A fixed instant for the timestamp tests. Any real datetime would do;
#: a named constant keeps the arithmetic visible in the assertions.
AN_INSTANT = datetime(2026, 8, 24, 18, 30, 0, tzinfo=UTC)


class SteppedDevice:
    """A device the test advances one read at a time.

    :class:`FakeDevice` is deterministic about *how many* blocks exist;
    this is deterministic about *when* each one lands. The reader parks
    inside ``read_raw`` until the test releases it, and does not park
    again until the block it just returned has been offered to every
    subscriber — so "three blocks have been read and nothing has
    consumed them" is a state the test arranges rather than one it
    races for.

    That distinction is what makes it possible to give two consumers
    *different* drop counts on purpose, which is the property the
    fan-out exists to provide and the one that cannot be tested by
    letting a stream run to exhaustion.
    """

    def __init__(
        self,
        *,
        block_bytes: int = BLOCK_BYTES,
        sample_rate_hz: float = SAMPLE_RATE_HZ,
    ) -> None:
        self.index = 0
        self.is_open = True
        self.applied = applied_settings(sample_rate_hz)
        self.reads = 0
        self.done = False
        self._block_bytes = block_bytes
        #: Released by the test to let exactly one read through.
        self.allow = threading.Event()
        #: Set when the reader comes back for another block, which it
        #: only does after queueing the previous one.
        self.ready = threading.Event()

    def read_raw(self, length: int) -> bytes:
        self.ready.set()
        self.allow.wait(5.0)
        self.allow.clear()
        if self.done:
            raise DeviceError("stepped device closed")
        block = bytes([self.reads % 256]) * min(length, self._block_bytes)
        self.reads += 1
        return block

    def step(self) -> None:
        """Let exactly one block through, and wait until it is queued."""
        self.ready.clear()
        self.allow.set()
        assert self.ready.wait(5.0), "the reader never came back for another block"

    def finish(self) -> None:
        """Let the reader out of its wait so the thread can exit."""
        self.done = True
        self.allow.set()


class TestTimedBlock:
    def test_the_midpoint_is_half_a_block_before_the_read_returned(self):
        # read_at is when the read RETURNED, so the samples are behind
        # it, not ahead of it. Adding instead of subtracting here would
        # be a whole block of standing Doppler error and nothing would
        # raise.
        block = TimedBlock(data=b"", read_at=AN_INSTANT, duration_s=0.064)

        assert block.midpoint == AN_INSTANT - timedelta(seconds=0.032)

    def test_a_zero_length_block_has_its_read_time_as_its_midpoint(self):
        block = TimedBlock(data=b"", read_at=AN_INSTANT, duration_s=0.0)

        assert block.midpoint == AN_INSTANT


class TestBlockTimestamps:
    def test_a_block_is_timestamped_when_its_read_returned(self):
        device = FakeDevice(blocks=1)
        stream = IqStream(device, block_bytes=BLOCK_BYTES, now=lambda: AN_INSTANT)

        first = next(stream.timed_blocks(poll_s=0.05))
        stream.stop()

        assert first.read_at == AN_INSTANT

    def test_the_duration_comes_from_the_sample_rate_the_device_reported(self):
        # 256 KiB at 2.048 Msps is 64 ms. Bring-up's own numbers, so a
        # failure reads in bench units.
        device = FakeDevice(blocks=1)
        stream = IqStream(device, block_bytes=BLOCK_BYTES, now=lambda: AN_INSTANT)

        first = next(stream.timed_blocks(poll_s=0.05))
        stream.stop()

        assert first.duration_s == pytest.approx(BLOCK_SECONDS)
        assert first.midpoint == AN_INSTANT - timedelta(seconds=BLOCK_SECONDS / 2)

    def test_a_short_read_is_timed_by_what_it_returned_not_by_what_was_asked_for(self):
        # The sibling of "account against the rate the device reported,
        # not the one requested". A block half the requested size covers
        # half the time, and timing it by block_bytes would put its
        # midpoint 16 ms too early with nothing to notice.
        device = FakeDevice(blocks=1, block_bytes=BLOCK_BYTES // 2)
        stream = IqStream(device, block_bytes=BLOCK_BYTES, now=lambda: AN_INSTANT)

        first = next(stream.timed_blocks(poll_s=0.05))
        stream.stop()

        assert first.duration_s == pytest.approx(BLOCK_SECONDS / 2)

    def test_blocks_and_timed_blocks_are_two_views_of_one_subscription(self):
        # Not two consumers. Claiming either claims both, because a
        # caller who wanted both really wants one stream of blocks and
        # sometimes the time with it.
        stream = IqStream(FakeDevice(blocks=2), queue_blocks=4)
        stream.blocks(poll_s=0.05)
        try:
            with pytest.raises(DeviceError, match="already been called"):
                stream.timed_blocks(poll_s=0.05)
        finally:
            stream.stop()


class TestSubscriptionRefusals:
    def test_calling_blocks_twice_is_refused(self):
        # The whole reason the fan-out exists. This used to "work": two
        # generators over one buffer, each getting roughly every other
        # block, both looking fine.
        stream = IqStream(FakeDevice(blocks=4), queue_blocks=4)
        stream.blocks(poll_s=0.05)
        try:
            with pytest.raises(DeviceError, match="already been called"):
                stream.blocks(poll_s=0.05)
        finally:
            stream.stop()

    def test_the_refusal_lands_on_the_call_not_on_the_first_next(self):
        # blocks() is deliberately not a generator function. If it were,
        # its body would not run until the first next(), so the second
        # call would sit there looking accepted and fail later from a
        # frame that does not mention the mistake. Asserting the refusal
        # without iterating is what pins that down.
        stream = IqStream(FakeDevice(blocks=4), queue_blocks=4)
        stream.blocks(poll_s=0.05)
        try:
            with pytest.raises(DeviceError):
                stream.blocks(poll_s=0.05)  # never iterated
        finally:
            stream.stop()

    def test_blocks_is_refused_once_there_are_explicit_subscribers(self):
        stream = IqStream(FakeDevice(blocks=4), queue_blocks=4)
        stream.subscribe("audio")
        stream.start()
        try:
            with pytest.raises(DeviceError, match="explicit subscribers"):
                stream.blocks(poll_s=0.05)
        finally:
            stream.stop()

    def test_the_refusal_does_not_start_the_reader_as_a_side_effect(self):
        # blocks() starts the reader on first use, so the refusal has to
        # come first. A rejected call that leaves a thread running and a
        # device being read is a worse outcome than the mistake it was
        # rejecting.
        stream = IqStream(FakeDevice(blocks=4), queue_blocks=4)
        stream.subscribe("audio")

        with pytest.raises(DeviceError, match="explicit subscribers"):
            stream.blocks(poll_s=0.05)

        assert not stream.is_running

    def test_subscribing_after_the_reader_started_is_refused(self):
        # A late subscriber silently misses everything already read.
        # Refusing is the honest answer while nothing needs dynamic
        # subscription; the alternative is a consumer that begins
        # mid-stream with no record of where it joined.
        stream = IqStream(FakeDevice(blocks=4), queue_blocks=4)
        stream.start()
        try:
            with pytest.raises(DeviceError, match="after the reader has started"):
                stream.subscribe("late")
        finally:
            stream.stop()

    def test_two_subscribers_may_not_share_a_name(self):
        stream = IqStream(FakeDevice(blocks=4), queue_blocks=4)
        stream.subscribe("audio")

        with pytest.raises(ValueError, match="already exists"):
            stream.subscribe("audio")

    def test_a_subscription_needs_a_name(self):
        stream = IqStream(FakeDevice(blocks=4), queue_blocks=4)

        with pytest.raises(ValueError, match="needs a name"):
            stream.subscribe("")


class TestFanOut:
    def test_every_subscriber_gets_every_block_in_order(self):
        device = FakeDevice(blocks=4)
        stream = IqStream(device, block_bytes=BLOCK_BYTES, queue_blocks=8)
        audio = stream.subscribe("audio")
        waterfall = stream.subscribe("waterfall")

        stream.start()
        device.drained.wait(5.0)
        stream.stop()

        assert [block[0] for block in islice(audio.blocks(poll_s=0.05), 4)] == [0, 1, 2, 3]
        assert [block[0] for block in islice(waterfall.blocks(poll_s=0.05), 4)] == [0, 1, 2, 3]

    def test_one_subscriber_draining_does_not_steal_from_another(self):
        # The exact failure the fan-out replaces: before it, a second
        # consumer took blocks out from under the first and neither had
        # any way to tell.
        device = FakeDevice(blocks=3)
        stream = IqStream(device, block_bytes=BLOCK_BYTES, queue_blocks=8)
        audio = stream.subscribe("audio")
        waterfall = stream.subscribe("waterfall")

        stream.start()
        device.drained.wait(5.0)
        stream.stop()

        list(islice(audio.blocks(poll_s=0.05), 3))

        assert [block[0] for block in islice(waterfall.blocks(poll_s=0.05), 3)] == [0, 1, 2]

    def test_subscribers_share_the_block_object_rather_than_a_copy(self):
        # Blocks are 256 KiB and immutable. Copying per consumer would
        # put a memcpy of that size on the read path, which Session 16
        # established is a real-time path.
        device = FakeDevice(blocks=1)
        stream = IqStream(device, block_bytes=BLOCK_BYTES, queue_blocks=4)
        audio = stream.subscribe("audio")
        waterfall = stream.subscribe("waterfall")

        stream.start()
        device.drained.wait(5.0)
        stream.stop()

        assert next(audio.blocks(poll_s=0.05)) is next(waterfall.blocks(poll_s=0.05))

    def test_a_reader_failure_reaches_every_consumer_not_just_the_first(self):
        # A stream that stops for a reason only one consumer is told
        # about is the silent-failure shape this project keeps meeting.
        device = FakeDevice(blocks=2, fail_after=2)
        stream = IqStream(device, queue_blocks=8)
        audio = stream.subscribe("audio")
        waterfall = stream.subscribe("waterfall")
        stream.start()

        with pytest.raises(DeviceError, match="gave up"):
            list(audio.blocks(poll_s=0.05))
        with pytest.raises(DeviceError, match="gave up"):
            list(waterfall.blocks(poll_s=0.05))


class TestPerConsumerDropAccounting:
    def stepped_stream(self) -> tuple[SteppedDevice, IqStream]:
        """Two consumers over a two-block buffer, advanced by hand."""
        device = SteppedDevice()
        stream = IqStream(device, block_bytes=BLOCK_BYTES, queue_blocks=2)
        return device, stream

    def test_a_slow_consumer_drops_its_own_blocks_and_nobody_elses(self):
        device, stream = self.stepped_stream()
        audio = stream.subscribe("audio")
        waterfall = stream.subscribe("waterfall")
        stream.start()
        try:
            for _ in range(3):
                device.step()
            # Both are one block behind a two-deep buffer at this point.
            # Audio catches up; the waterfall does not.
            list(islice(audio.blocks(poll_s=0.05), 2))
            for _ in range(2):
                device.step()
        finally:
            device.finish()
            stream.stop()

        assert audio.stats.blocks_dropped == 1
        assert waterfall.stats.blocks_dropped == 3

    def test_the_stream_reports_the_worst_consumer_not_the_sum(self):
        # Summing would say four blocks were lost when no consumer lost
        # more than three, which is the "two faults reported as one
        # number" mistake this module exists to avoid.
        device, stream = self.stepped_stream()
        audio = stream.subscribe("audio")
        stream.subscribe("waterfall")
        stream.start()
        try:
            for _ in range(3):
                device.step()
            list(islice(audio.blocks(poll_s=0.05), 2))
            for _ in range(2):
                device.step()
        finally:
            device.finish()
            stats = stream.stop()

        assert stats.blocks_dropped == 3
        assert {entry.name: entry.blocks_dropped for entry in stats.subscribers} == {
            "audio": 1,
            "waterfall": 3,
        }

    def test_offered_counts_every_block_put_in_front_of_a_consumer(self):
        device, stream = self.stepped_stream()
        audio = stream.subscribe("audio")
        stream.start()
        try:
            for _ in range(5):
                device.step()
        finally:
            device.finish()
            stream.stop()

        # Five offered, three of them evicted before anyone read them.
        assert audio.stats.blocks_offered == 5
        assert audio.stats.blocks_dropped == 3


class TestStatsPresentation:
    def test_a_single_consumer_stream_still_reports_one_buffer_line(self):
        # The default subscription exists but is not itemised: a second
        # line restating the number directly above it invites a reader
        # to hunt for a difference that is not there.
        text = drained_stream(blocks=3, queue_blocks=1).stats.describe()

        assert "at our buffer" in text
        assert "consumer" not in text

    def test_several_consumers_are_itemised_by_name(self):
        device = FakeDevice(blocks=3)
        stream = IqStream(device, block_bytes=BLOCK_BYTES, queue_blocks=1)
        stream.subscribe("audio")
        stream.subscribe("waterfall")
        stream.start()
        device.drained.wait(5.0)
        text = stream.stop().describe()

        assert "consumer audio:" in text
        assert "consumer waterfall:" in text

    def test_a_stream_left_unconsumed_still_counts_its_drops(self):
        # Guards the implicit subscription: a stream started with no
        # consumer at all must still buffer and still count, exactly as
        # it did before the fan-out existed.
        stats = drained_stream(blocks=6, queue_blocks=2).stats

        assert stats.blocks_dropped == 4
        assert [entry.name for entry in stats.subscribers] == ["default"]
