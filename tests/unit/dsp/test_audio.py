"""Tests for AudioOutput: queueing, oldest-discard, and underrun accounting.

All of these inject a fake stream factory, so none of them touch real
PortAudio -- which matters here specifically, since importing
``sounddevice`` for real raises ``OSError`` wherever PortAudio is not
installed (this project's own sandbox included). See the module docstring
in ``core/dsp/audio.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsorbit.core.dsp.audio import AudioError, AudioOutput


class FakeStream:
    """Stands in for sounddevice.OutputStream. Never calls the callback
    on its own -- tests call AudioOutput's private _callback directly,
    the same way :class:`~qsorbit.core.sdr.stream.IqStream`'s own tests
    drive its reader loop directly rather than waiting on a real thread.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def fake_stream_factory(**kwargs):
    return FakeStream(**kwargs)


def pull(output: AudioOutput, frames: int) -> np.ndarray:
    """Drive AudioOutput's callback as PortAudio would, and return what
    it filled.
    """
    outdata = np.zeros((frames, 1), dtype=np.float32)
    output._callback(outdata, frames, None, None)
    return outdata[:, 0]


class TestLifecycle:
    def test_start_opens_and_starts_the_stream(self):
        output = AudioOutput(32_000.0, stream_factory=fake_stream_factory)

        output.start()

        assert output.is_open
        assert output._stream.started

    def test_starting_twice_is_an_error(self):
        output = AudioOutput(32_000.0, stream_factory=fake_stream_factory)
        output.start()

        with pytest.raises(AudioError, match="already been started"):
            output.start()

    def test_stop_closes_the_stream_and_returns_stats(self):
        output = AudioOutput(32_000.0, stream_factory=fake_stream_factory)
        output.start()
        stream = output._stream

        stats = output.stop()

        assert stream.stopped
        assert stream.closed
        assert not output.is_open
        assert stats.blocks_written == 0

    def test_stop_without_start_is_a_no_op(self):
        output = AudioOutput(32_000.0, stream_factory=fake_stream_factory)

        stats = output.stop()

        assert stats.blocks_written == 0

    def test_write_before_start_is_an_error(self):
        output = AudioOutput(32_000.0, stream_factory=fake_stream_factory)

        with pytest.raises(AudioError, match="before start"):
            output.write(np.zeros(10, dtype=np.float32))

    def test_context_manager_starts_and_stops(self):
        with AudioOutput(32_000.0, stream_factory=fake_stream_factory) as output:
            assert output.is_open
            stream = output._stream

        assert stream.stopped
        assert stream.closed

    def test_rejects_a_non_positive_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate_hz"):
            AudioOutput(0.0, stream_factory=fake_stream_factory)

    def test_rejects_a_non_positive_queue_depth(self):
        with pytest.raises(ValueError, match="queue_blocks"):
            AudioOutput(32_000.0, queue_blocks=0, stream_factory=fake_stream_factory)


class TestWriteAndPlayback:
    def test_a_written_block_is_played_back_exactly(self):
        output = AudioOutput(32_000.0, stream_factory=fake_stream_factory)
        output.start()
        block = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        output.write(block)
        played = pull(output, 4)

        assert np.allclose(played, block)
        assert output.stats.blocks_played == 1

    def test_a_non_float32_block_is_converted(self):
        output = AudioOutput(32_000.0, stream_factory=fake_stream_factory)
        output.start()
        block = np.array([0.5, -0.5], dtype=np.float64)

        output.write(block)
        played = pull(output, 2)

        assert np.allclose(played, block)

    def test_rejects_a_multi_dimensional_block(self):
        output = AudioOutput(32_000.0, stream_factory=fake_stream_factory)
        output.start()

        with pytest.raises(ValueError, match="one-dimensional"):
            output.write(np.zeros((4, 2), dtype=np.float32))

    def test_a_callback_can_span_more_than_one_written_block(self):
        output = AudioOutput(32_000.0, stream_factory=fake_stream_factory)
        output.start()
        output.write(np.array([1.0, 2.0], dtype=np.float32))
        output.write(np.array([3.0, 4.0], dtype=np.float32))

        played = pull(output, 4)

        assert np.array_equal(played, [1.0, 2.0, 3.0, 4.0])
        assert output.stats.blocks_played == 2

    def test_a_callback_can_take_only_part_of_a_written_block(self):
        output = AudioOutput(32_000.0, stream_factory=fake_stream_factory)
        output.start()
        output.write(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

        first = pull(output, 2)
        second = pull(output, 2)

        assert np.array_equal(first, [1.0, 2.0])
        assert np.array_equal(second, [3.0, 4.0])
        # The block is only "played" once fully drained, on the second pull.
        assert output.stats.blocks_played == 1

    def test_an_underrun_pads_with_silence_and_is_counted(self):
        output = AudioOutput(32_000.0, stream_factory=fake_stream_factory)
        output.start()
        output.write(np.array([1.0, 2.0], dtype=np.float32))

        played = pull(output, 4)  # more than was written

        assert np.array_equal(played, [1.0, 2.0, 0.0, 0.0])
        assert output.stats.underruns == 1

    def test_a_pull_with_nothing_queued_is_silence_and_an_underrun(self):
        output = AudioOutput(32_000.0, stream_factory=fake_stream_factory)
        output.start()

        played = pull(output, 4)

        assert np.array_equal(played, [0.0, 0.0, 0.0, 0.0])
        assert output.stats.underruns == 1

    def test_frames_played_counts_every_callback_including_underruns(self):
        output = AudioOutput(32_000.0, stream_factory=fake_stream_factory)
        output.start()
        output.write(np.zeros(2, dtype=np.float32))

        pull(output, 2)
        pull(output, 5)  # underrun, nothing queued

        assert output.stats.frames_played == 7


class TestOldestBlockIsDiscardedWhenFull:
    def test_writing_past_capacity_drops_the_oldest_block(self):
        output = AudioOutput(32_000.0, queue_blocks=2, stream_factory=fake_stream_factory)
        output.start()

        output.write(np.array([1.0], dtype=np.float32))
        output.write(np.array([2.0], dtype=np.float32))
        output.write(np.array([3.0], dtype=np.float32))  # buffer full: drops [1.0]

        played = pull(output, 2)

        assert np.array_equal(played, [2.0, 3.0])
        assert output.stats.blocks_dropped == 1

    def test_blocks_written_counts_every_write_including_dropped_ones(self):
        output = AudioOutput(32_000.0, queue_blocks=1, stream_factory=fake_stream_factory)
        output.start()

        output.write(np.array([1.0], dtype=np.float32))
        output.write(np.array([2.0], dtype=np.float32))

        assert output.stats.blocks_written == 2
        assert output.stats.blocks_dropped == 1


class TestAudioStatsDescribe:
    def test_describe_mentions_both_kinds_of_fault(self):
        output = AudioOutput(32_000.0, stream_factory=fake_stream_factory)
        output.start()

        text = output.stats.describe()

        assert "dropped" in text
        assert "underrun" in text
