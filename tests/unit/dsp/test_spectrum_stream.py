"""Tests for the spectrum streaming pipeline.

Everything here runs with no device, no Qt and no hardware: the source is
an iterable of raw byte blocks, so a list of synthetic blocks stands in
for :meth:`~qsorbit.core.sdr.stream.IqStream.blocks` exactly. The clock
is injected for the same reason Chunk A's loop injects one — asserting
against a real clock makes a test that passes on a fast machine and
flakes on a slow one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from qsorbit.core.dsp.iq import IQ_ZERO_OFFSET, unpack_uint8_iq
from qsorbit.core.dsp.spectrum import SpectrumConfig, frequency_axis_hz, power_spectrum_db
from qsorbit.core.dsp.spectrum_stream import (
    BYTES_PER_SAMPLE,
    DEFAULT_FRAME_RATE_HZ,
    SpectrumFrame,
    SpectrumStream,
    SpectrumStreamStats,
    hop_for_frame_rate,
)

FFT_SIZE = 64
SAMPLE_RATE = 64_000.0


def make_config(**overrides) -> SpectrumConfig:
    """A small config, sized so tests stay fast and hand-checkable."""
    kwargs = {"fft_size": FFT_SIZE, "sample_rate_hz": SAMPLE_RATE}
    kwargs.update(overrides)
    return SpectrumConfig(**kwargs)


def tone_block(n_samples: int, *, freq_hz: float = 0.0, rate_hz: float = SAMPLE_RATE) -> bytes:
    """Raw uint8 interleaved I/Q carrying a complex tone at ``freq_hz``.

    Built the way the RTL-SDR actually delivers samples — offset binary,
    127.5 as zero — so the block travels the real
    :func:`~qsorbit.core.dsp.iq.unpack_uint8_iq` path rather than
    sidestepping it.
    """
    t = np.arange(n_samples, dtype=np.float64) / rate_hz
    wave = np.exp(2j * np.pi * freq_hz * t) * 0.5
    interleaved = np.empty(n_samples * 2, dtype=np.float64)
    interleaved[0::2] = wave.real
    interleaved[1::2] = wave.imag
    return np.round(interleaved * IQ_ZERO_OFFSET + IQ_ZERO_OFFSET).astype(np.uint8).tobytes()


def silent_block(n_samples: int) -> bytes:
    """A block of exact zeros, in the wire format."""
    return np.full(n_samples * 2, int(IQ_ZERO_OFFSET), dtype=np.uint8).tobytes()


class FakeClock:
    """A clock that advances a fixed step per call. Never the real one."""

    def __init__(self, step_s: float = 1.0) -> None:
        self._now = datetime(2026, 8, 23, 12, 0, 0, tzinfo=UTC)
        self._step = timedelta(seconds=step_s)

    def __call__(self) -> datetime:
        now = self._now
        self._now += self._step
        return now


def drain(stream: SpectrumStream, *, timeout_s: float = 5.0) -> list[SpectrumFrame]:
    """Run a stream to completion over a finite source and collect every frame.

    Polls rather than sleeping a fixed amount: a fixed sleep is how a
    threading test becomes a machine-speed lottery.
    """
    stream.start()
    thread = stream._thread
    assert thread is not None
    thread.join(timeout_s)
    assert not thread.is_alive(), "worker did not finish within the timeout"
    return stream.latest()


# ----------------------------------------------------------------------
# hop_for_frame_rate
# ----------------------------------------------------------------------


def test_hop_matches_requested_frame_rate():
    config = make_config()
    # 64,000 samples/s at 10 frames/s wants a frame start every 6,400.
    assert hop_for_frame_rate(config, 10.0) == 6_400


def test_hop_never_goes_below_fft_size():
    """A rate higher than the non-overlapping rate is capped, not overlapped."""
    config = make_config()
    # The non-overlapping rate here is 1000 frames/s; ask for ten times that.
    assert hop_for_frame_rate(config, 10_000.0) == FFT_SIZE


def test_hop_at_realistic_bench_settings():
    """The numbers the design argument was actually built on."""
    config = SpectrumConfig(fft_size=2048, sample_rate_hz=2_048_000.0)
    assert hop_for_frame_rate(config, 20.0) == 102_400
    # One 256 KiB block is 131,072 samples, so this is a bit over one
    # frame per block -- the cadence the chunk's design note claims.
    assert 131_072 / 102_400 == pytest.approx(1.28, abs=0.01)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_hop_rejects_unusable_frame_rates(bad):
    with pytest.raises(ValueError, match="frame_rate_hz"):
        hop_for_frame_rate(make_config(), bad)


# ----------------------------------------------------------------------
# Frame production
# ----------------------------------------------------------------------


def test_produces_expected_frame_count():
    config = make_config()
    # hop == fft_size, so every sample is used: 4 blocks x 64 samples.
    stream = SpectrumStream([silent_block(FFT_SIZE)] * 4, config, frame_rate_hz=1_000.0)
    frames = drain(stream)
    assert len(frames) == 4


def test_skips_frames_it_was_told_not_to_compute():
    """The design's whole point: most available frames are never computed."""
    config = make_config()
    # 10 frames/s from a 64 kHz stream: one frame per 6,400 samples,
    # against 100 available frames per 6,400 samples.
    stream = SpectrumStream([silent_block(6_400)] * 4, config, frame_rate_hz=10.0)
    frames = drain(stream)
    stats = stream.stop()
    assert len(frames) == 4
    assert stats.frames_computed == 4
    assert stats.frames_available == 400
    assert stats.frames_skipped == 396
    assert stats.compute_fraction == pytest.approx(0.01)


def test_frames_carry_the_configured_length_and_dtype():
    config = make_config()
    frames = drain(SpectrumStream([silent_block(FFT_SIZE)], config, frame_rate_hz=1_000.0))
    assert frames[0].power_db.shape == (FFT_SIZE,)
    assert frames[0].power_db.dtype == np.float32


def test_frame_finds_a_tone_at_the_right_frequency():
    """Presence at the expected frequency, never 'the loudest bin'.

    Session 14's bring-up rule, and the same check
    ``test_spectrum.py`` applies to the real captured fixture: an
    RTL-SDR's permanent DC spike wins a bare argmax, so a test that
    trusts argmax passes on a mistuned radio.
    """
    config = make_config(center_freq_hz=100_000_000.0)
    tone_hz = 8_000.0
    frames = drain(
        SpectrumStream([tone_block(FFT_SIZE, freq_hz=tone_hz)], config, frame_rate_hz=1_000.0)
    )

    axis = frequency_axis_hz(config)
    expected = config.center_freq_hz + tone_hz
    bin_index = int(np.argmin(np.abs(axis - expected)))
    power = frames[0].power_db
    assert power[bin_index] > np.median(power) + 20.0


def test_frame_cadence_carries_across_block_boundaries():
    """Frame starts stay evenly spaced when blocks are not whole hops.

    The failure this guards against is subtle and would look fine at a
    glance: restarting the hop per block bunches frames up near each
    block's start, which on a waterfall is a distorted time axis rather
    than a visible bug.
    """
    config = make_config()
    hop = 100
    rate = SAMPLE_RATE / hop
    # 150-sample blocks against a 100-sample hop: no block is a whole
    # number of hops, so a per-block reset would give exactly one frame
    # per block (5), where correct behaviour gives one per 100 samples.
    stream = SpectrumStream([silent_block(150)] * 5, config, frame_rate_hz=rate)
    assert stream.hop == hop
    frames = drain(stream)
    assert len(frames) == 7  # 750 samples / 100, less the trailing partial


def test_low_frame_rate_does_not_grow_the_carry_buffer():
    """A hop longer than a block must discard, not accumulate.

    Without the discard this is a slow memory leak that only shows up on
    a long pass -- exactly the kind of thing a short test never sees, so
    the internal buffer is asserted on directly.
    """
    config = make_config()
    stream = SpectrumStream([silent_block(100)] * 20, config, frame_rate_hz=100.0)
    assert stream.hop == 640  # much longer than a 100-sample block
    drain(stream)
    # Counted in bytes now that the carry is raw rather than unpacked --
    # the point of the assertion is unchanged, only its units.
    assert len(stream._carry) // BYTES_PER_SAMPLE < 640


# ----------------------------------------------------------------------
# Timestamps
# ----------------------------------------------------------------------


def test_frames_are_timestamped_from_the_injected_clock():
    config = make_config()
    clock = FakeClock(step_s=0.05)
    frames = drain(
        SpectrumStream([silent_block(FFT_SIZE)] * 3, config, frame_rate_hz=1_000.0, now=clock)
    )
    times = [frame.time for frame in frames]
    assert times == sorted(times)
    assert (times[-1] - times[0]).total_seconds() == pytest.approx(0.10)
    assert all(t.tzinfo is not None for t in times)


# ----------------------------------------------------------------------
# The two losses, kept apart
# ----------------------------------------------------------------------


def test_drops_oldest_when_the_consumer_never_drains():
    config = make_config()
    stream = SpectrumStream(
        [silent_block(FFT_SIZE)] * 10, config, frame_rate_hz=1_000.0, queue_frames=4
    )
    drain(stream)
    stats = stream.stop()
    assert stats.frames_computed == 10
    assert stats.frames_dropped == 6
    assert stats.frames_skipped == 0


def test_a_dropped_frame_is_not_a_skipped_frame():
    """The distinction this module exists to preserve.

    A run that skips heavily and drops nothing is healthy; a run that
    drops anything is not. One combined counter would report these two
    as the same number and send you at the wrong thing.
    """
    config = make_config()
    healthy = SpectrumStream([silent_block(6_400)] * 2, config, frame_rate_hz=10.0, queue_frames=64)
    drain(healthy)
    healthy_stats = healthy.stop()

    behind = SpectrumStream(
        [silent_block(FFT_SIZE)] * 20, config, frame_rate_hz=1_000.0, queue_frames=2
    )
    drain(behind)
    behind_stats = behind.stop()

    assert healthy_stats.frames_skipped > 0
    assert healthy_stats.frames_dropped == 0
    assert behind_stats.frames_dropped > 0
    assert behind_stats.frames_skipped == 0


def test_describe_words_the_two_losses_unmistakably():
    stats = SpectrumStreamStats(
        blocks_consumed=4,
        samples_consumed=25_600,
        frames_computed=4,
        frames_skipped=396,
        frames_dropped=0,
        queue_frames=64,
        worker_stopped_cleanly=True,
    )
    text = stats.describe()
    assert "skipped by design" in text
    assert "not a fault" in text
    assert "dropped at buffer" in text
    assert "consumer behind" in text


def test_describe_calls_out_an_unclean_stop():
    stats = SpectrumStreamStats(
        blocks_consumed=1,
        samples_consumed=64,
        frames_computed=1,
        frames_skipped=0,
        frames_dropped=0,
        queue_frames=64,
        worker_stopped_cleanly=False,
    )
    assert "DID NOT stop cleanly" in stats.describe()


def test_compute_fraction_is_zero_before_anything_runs():
    stream = SpectrumStream([], make_config())
    assert stream.stats.compute_fraction == 0.0
    assert stream.stats.frames_available == 0


# ----------------------------------------------------------------------
# Consuming
# ----------------------------------------------------------------------


def test_latest_returns_every_frame_since_the_last_call_oldest_first():
    config = make_config()
    clock = FakeClock(step_s=1.0)
    stream = SpectrumStream([silent_block(FFT_SIZE)] * 3, config, frame_rate_hz=1_000.0, now=clock)
    frames = drain(stream)
    assert [f.time for f in frames] == sorted(f.time for f in frames)
    # Drained once; a second call sees nothing new.
    assert stream.latest() == []


def test_latest_is_empty_not_an_error_when_nothing_has_arrived():
    stream = SpectrumStream([], make_config())
    assert stream.latest() == []


# ----------------------------------------------------------------------
# Failure propagation
# ----------------------------------------------------------------------


def test_a_failing_source_surfaces_in_the_consumer():
    """A waterfall that silently freezes when the device dies is the bug."""

    def exploding_source():
        yield silent_block(FFT_SIZE)
        raise OSError("device went away")

    stream = SpectrumStream(exploding_source(), make_config(), frame_rate_hz=1_000.0)
    stream.start()
    stream._thread.join(5.0)
    # Good frames first, error second: raising straight away would throw
    # away the frame that arrived before the fault.
    assert len(stream.latest()) == 1
    with pytest.raises(OSError, match="device went away"):
        stream.latest()


def test_frames_from_before_a_failure_are_not_discarded():
    """The drain-then-raise order, asserted directly rather than implied."""

    def exploding_source():
        for _ in range(3):
            yield silent_block(FFT_SIZE)
        raise OSError("device went away")

    stream = SpectrumStream(exploding_source(), make_config(), frame_rate_hz=1_000.0)
    stream.start()
    stream._thread.join(5.0)
    assert len(stream.latest()) == 3
    with pytest.raises(OSError):
        stream.latest()


def test_the_error_keeps_being_raised_rather_than_clearing():
    """Changed contract, and deliberately: the fan-out made it necessary.

    The single-consumer version cleared the error once raised, so a
    second drain saw an empty list. With several consumers that means
    whichever one drains first gets the exception and every other one
    watches its frames simply stop -- a silent freeze with the
    explanation already consumed by somebody else. ``IqStream`` made
    exactly this change when it grew its own fan-out in Chunk H.
    """

    def exploding_source():
        raise OSError("boom")
        yield  # pragma: no cover - unreachable, makes this a generator

    stream = SpectrumStream(exploding_source(), make_config())
    stream.start()
    stream._thread.join(5.0)
    with pytest.raises(OSError):
        stream.latest()
    with pytest.raises(OSError):
        stream.latest()


def test_every_consumer_is_told_why_the_frames_stopped():
    """The reason the error is no longer cleared, asserted directly."""

    def exploding_source():
        raise OSError("device went away")
        yield  # pragma: no cover - unreachable, makes this a generator

    stream = SpectrumStream(exploding_source(), make_config())
    waterfall = stream.subscribe("waterfall")
    trace = stream.subscribe("spectrum-line")
    stream.start()
    stream._thread.join(5.0)
    with pytest.raises(OSError, match="device went away"):
        waterfall.latest()
    with pytest.raises(OSError, match="device went away"):
        trace.latest()


# ----------------------------------------------------------------------
# Fan-out
# ----------------------------------------------------------------------


def run_to_completion(stream: SpectrumStream, *, timeout_s: float = 5.0) -> None:
    """Start a stream over a finite source and wait for the worker to end."""
    stream.start()
    thread = stream._thread
    assert thread is not None
    thread.join(timeout_s)
    assert not thread.is_alive(), "worker did not finish within the timeout"


def test_two_consumers_each_receive_every_frame():
    """Bench verification #11, as a test that would have caught it.

    Two spectrum panels alternated on real hardware because both drained
    one shared buffer and whichever timer fired first took the batch.
    Neither widget was wrong, so no widget test could have found it --
    only one asserting that two consumers of one stream both get
    everything.
    """
    config = make_config()
    stream = SpectrumStream([silent_block(FFT_SIZE)] * 5, config, frame_rate_hz=1_000.0)
    waterfall = stream.subscribe("waterfall")
    trace = stream.subscribe("spectrum-line")
    run_to_completion(stream)

    from_waterfall = waterfall.latest()
    from_trace = trace.latest()
    assert len(from_waterfall) == 5
    assert len(from_trace) == 5
    # The same frame objects, not copies: one immutable frame is offered
    # to every consumer, so a five-panel Custom tab costs one FFT.
    assert [id(f) for f in from_waterfall] == [id(f) for f in from_trace]


def test_a_slow_consumer_drops_only_its_own_frames():
    """The whole point of per-consumer buffers, stated as an assertion."""
    config = make_config()
    stream = SpectrumStream([silent_block(FFT_SIZE)] * 10, config, frame_rate_hz=1_000.0)
    healthy = stream.subscribe("healthy")
    stalled = stream.subscribe("stalled")
    # The stalled one never drains; give it a buffer far too small.
    stalled._queue_frames = 2
    stalled._frames = type(stalled._frames)(maxlen=2)
    run_to_completion(stream)

    assert len(healthy.latest()) == 10
    assert healthy.stats.frames_dropped == 0
    assert stalled.stats.frames_dropped == 8


def test_a_late_subscriber_joins_without_disturbing_anyone():
    """Allowed here, unlike IqStream, because a display feed has no gap."""
    config = make_config()
    stream = SpectrumStream([silent_block(FFT_SIZE)] * 3, config, frame_rate_hz=1_000.0)
    early = stream.subscribe("early")
    run_to_completion(stream)
    late = stream.subscribe("late")

    assert len(early.latest()) == 3
    # Nothing was produced after it joined, and that is reported as
    # having been offered nothing -- not as loss.
    assert late.latest() == []
    assert late.stats.frames_offered == 0
    assert late.stats.frames_dropped == 0


def test_latest_refuses_once_something_has_subscribed_explicitly():
    """The #11 mistake, made loud instead of silent."""
    stream = SpectrumStream([silent_block(FFT_SIZE)], make_config())
    stream.subscribe("waterfall")
    with pytest.raises(RuntimeError, match="subscribe"):
        stream.latest()


def test_a_subscription_needs_a_name_and_a_unique_one():
    stream = SpectrumStream([], make_config())
    with pytest.raises(ValueError, match="needs a name"):
        stream.subscribe("")
    stream.subscribe("waterfall")
    with pytest.raises(ValueError, match="already exists"):
        stream.subscribe("waterfall")


def test_a_subscription_reports_the_streams_framing():
    """It satisfies the widgets' FrameSource protocol on its own."""
    config = make_config()
    stream = SpectrumStream([], config)
    assert stream.subscribe("waterfall").config is config


def test_stats_name_each_consumer_and_report_the_worst_drop():
    config = make_config()
    stream = SpectrumStream([silent_block(FFT_SIZE)] * 10, config, frame_rate_hz=1_000.0)
    healthy = stream.subscribe("healthy")
    stalled = stream.subscribe("stalled")
    stalled._queue_frames = 2
    stalled._frames = type(stalled._frames)(maxlen=2)
    run_to_completion(stream)
    healthy.latest()
    stats = stream.stop()

    assert [s.name for s in stats.subscribers] == ["healthy", "stalled"]
    # The worst consumer, not the sum: summing would report the stream as
    # twice as lossy as either consumer actually experienced.
    assert stats.frames_dropped == 8
    text = stats.describe()
    assert "consumer healthy:" in text
    assert "consumer stalled:" in text


def test_one_consumer_is_not_listed_separately():
    """A report that says the same number twice invites a wrong hunt."""
    config = make_config()
    stream = SpectrumStream([silent_block(FFT_SIZE)] * 2, config, frame_rate_hz=1_000.0)
    run_to_completion(stream)
    # Matched with the colon: the buffer line above already ends in the
    # words "consumer behind", which a bare substring would catch.
    assert "consumer default:" not in stream.stop().describe()


def test_the_source_label_distinguishes_two_streams():
    """Idle until Chunk E, when a second dongle means a second stream."""
    config = make_config()
    assert SpectrumStream([], config).source == "sdr"
    branch_b = SpectrumStream([], config, source="sdr1")
    assert branch_b.source == "sdr1"
    assert "sdr1" in branch_b.stats.describe()


# ----------------------------------------------------------------------
# Unpacking only what a frame needs
# ----------------------------------------------------------------------


def test_byte_range_unpacking_gives_the_same_frames_as_whole_block():
    """The optimisation must not move a single dB.

    Session 19 measured whole-block unpacking at 2.24% of a core against
    the FFTs' 0.24%, and the fix is to convert only the bytes each frame
    needs. That is a hot-path rewrite of index arithmetic, so the thing
    worth asserting is not that it is faster but that it is identical:
    computed here against the old whole-block path, done by hand.
    """
    config = make_config()
    block = tone_block(FFT_SIZE * 4, freq_hz=8_000.0)
    stream = SpectrumStream([block], config, frame_rate_hz=1_000.0)
    frames = drain(stream)

    whole = unpack_uint8_iq(block)
    hop = stream.hop
    expected = [
        power_spectrum_db(whole[start : start + FFT_SIZE], config)
        for start in range(0, len(whole) - FFT_SIZE + 1, hop)
    ]
    assert len(frames) == len(expected)
    for frame, want in zip(frames, expected, strict=True):
        np.testing.assert_array_equal(frame.power_db, want)


def test_a_truncated_iq_pair_is_still_refused():
    """The whole-block unpack used to catch this for free.

    Slicing byte ranges would floor an odd length and quietly frame
    whatever came next, so the guarantee is kept explicitly rather than
    lost inside an optimisation.
    """
    stream = SpectrumStream([silent_block(FFT_SIZE)[:-1]], make_config())
    stream.start()
    stream._thread.join(5.0)
    with pytest.raises(ValueError, match="odd length"):
        stream.latest()


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_starting_twice_is_an_error():
    stream = SpectrumStream([], make_config())
    stream.start()
    with pytest.raises(RuntimeError, match="already been started"):
        stream.start()


def test_context_manager_starts_and_stops():
    config = make_config()
    with SpectrumStream([silent_block(FFT_SIZE)] * 2, config, frame_rate_hz=1_000.0) as stream:
        assert stream.is_running or stream.stats.blocks_consumed > 0
    assert not stream.is_running
    assert stream.stats.worker_stopped_cleanly


def test_stop_reports_a_clean_exit_over_a_finite_source():
    config = make_config()
    stream = SpectrumStream([silent_block(FFT_SIZE)], config, frame_rate_hz=1_000.0)
    drain(stream)
    assert stream.stop().worker_stopped_cleanly


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_an_unusable_queue_depth(bad):
    with pytest.raises(ValueError, match="queue_frames"):
        SpectrumStream([], make_config(), queue_frames=bad)


def test_default_frame_rate_is_sane_for_a_real_capture():
    """Guards the default against becoming quietly useless.

    Bounds both directions, per Session 16: an assertion with only an
    upper bound passes when the value collapses to zero.
    """
    config = SpectrumConfig(fft_size=2048, sample_rate_hz=2_048_000.0)
    rows_per_second = config.sample_rate_hz / hop_for_frame_rate(config, DEFAULT_FRAME_RATE_HZ)
    assert 5.0 < rows_per_second < 60.0


# ----------------------------------------------------------------------
# Against a real capture
# ----------------------------------------------------------------------


def test_real_broadcast_capture_streams_through_to_a_detectable_station(load_iq_fixture):
    """The whole pipeline, on real hardware data, in streaming shape.

    Synthetic blocks prove the arithmetic; this proves the arithmetic
    survives contact with a real dongle's output. Session 14's rule
    applies and is the point of the test: ask *is our signal present at
    our frequency*, never *is our signal the loudest*. Getting that
    backwards once already misreported a correctly tuned radio as
    mistuned.

    Anchored on ``station_hz`` and ``actual_center_hz`` directly. The
    offset is deliberately **not** re-derived from this sidecar's
    ``tuning_offset_hz``, which carries the opposite sign from the
    ``station_offset_hz`` key newer sidecars use — the trap Chunk D
    handed forward, and the same avoidance both prior DSP PRs used.
    """
    raw, meta = load_iq_fixture("wbfm-99.9")
    block_bytes = 262_144
    blocks = [raw[i : i + block_bytes] for i in range(0, len(raw), block_bytes)]

    config = SpectrumConfig(
        fft_size=2048,
        sample_rate_hz=meta["actual_sample_rate_hz"],
        center_freq_hz=meta["actual_center_hz"],
    )
    stream = SpectrumStream(blocks, config, frame_rate_hz=20.0, queue_frames=256)
    frames = drain(stream, timeout_s=30.0)
    stats = stream.stop()

    # The capture is 2 seconds; at 20 frames/s that is about 40 rows.
    # Bounded both directions, per Session 16: an upper bound alone
    # passes when the count collapses to zero.
    assert 35 <= len(frames) <= 45
    assert stats.frames_dropped == 0
    assert 0.01 < stats.compute_fraction < 0.05

    axis = frequency_axis_hz(config)
    average_db = np.mean(np.stack([frame.power_db for frame in frames]), axis=0)
    median_db = float(np.median(average_db))

    near_station = np.abs(axis - meta["station_hz"]) <= meta["tolerance_hz"]
    assert near_station.any()
    assert float(average_db[near_station].max()) > median_db + 15.0

    # And it is genuinely the station rather than the receiver's own DC
    # spike: the capture is tuned 250 kHz off deliberately, so the two
    # are far apart and the detection window excludes the centre.
    assert not near_station[int(np.argmin(np.abs(axis - config.center_freq_hz)))]
