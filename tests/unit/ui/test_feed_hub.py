"""Tests for the feed hub.

No Qt anywhere in this file, deliberately. The hub is Qt-free (see its
module docstring), so its feed accounting can be checked by arithmetic
rather than by looking at panels — which is the whole lesson of Session
25: *a per-consumer drop count on a bounded buffer proves the consumer
drained*, and that turns "did both panels keep working" from an eyeball
judgement into a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsorbit.core.dsp.iq import IQ_ZERO_OFFSET
from qsorbit.core.dsp.spectrum import SpectrumConfig
from qsorbit.core.dsp.spectrum_stream import SpectrumStream
from qsorbit.ui.feed_hub import FeedHub, QuietingFeed, RotorFeed, TrackedFrequencyFeed

FFT_SIZE = 64
SAMPLE_RATE = 64_000.0


def make_config(**overrides) -> SpectrumConfig:
    kwargs = {"fft_size": FFT_SIZE, "sample_rate_hz": SAMPLE_RATE}
    kwargs.update(overrides)
    return SpectrumConfig(**kwargs)


def silent_block(n_samples: int) -> bytes:
    """A block of exact zeros, in the RTL-SDR's offset-binary wire format."""
    return np.full(n_samples * 2, int(IQ_ZERO_OFFSET), dtype=np.uint8).tobytes()


def make_stream(blocks: int = 8) -> SpectrumStream:
    """A finite stream: hop == fft_size, so one frame per block."""
    return SpectrumStream([silent_block(FFT_SIZE)] * blocks, make_config(), frame_rate_hz=1_000.0)


def run_to_completion(stream: SpectrumStream, *, timeout_s: float = 5.0) -> None:
    """Start the worker and wait for the finite source to run out.

    Polls the thread rather than sleeping a fixed amount, matching
    ``test_spectrum_stream.drain`` — a fixed sleep is how a threading
    test becomes a machine-speed lottery.
    """
    stream.start()
    thread = stream._thread
    assert thread is not None
    thread.join(timeout_s)
    assert not thread.is_alive(), "worker did not finish within the timeout"


class FakeRadio:
    """Stands in for a ReceiveSession's three live levels.

    Satisfies ``RadioSource`` structurally, which is the point of
    declaring that protocol rather than importing the session: no
    stream, no audio device and no rotor are needed to check that the
    hub forwards a float.
    """

    def __init__(self) -> None:
        self.live_quieting_db: float | None = -3.0
        self.live_squelch_open: bool | None = True
        self.live_tracked_frequency_hz: float | None = 435_605_000.0


class FakeLoop:
    """The little of a TrackingLoop that a RotorFeed touches."""

    def __init__(self, name: str = "RS-44") -> None:
        self.target = type("Target", (), {"name": name})()
        self.latest_sample = None


# ----------------------------------------------------------------------
# Streams are claimed: a new, independent feed every call
# ----------------------------------------------------------------------


def test_each_claim_returns_a_distinct_feed():
    hub = FeedHub(spectrum=make_stream())
    first = hub.spectrum("waterfall")
    second = hub.spectrum("spectrum-line")
    assert first is not second


def test_two_feeds_each_receive_every_frame():
    """The arithmetic that proves nothing was stolen.

    Bench verification #11 (Session 24) found two panels alternating on
    real hardware because both drained one shared buffer. This is that
    failure expressed as a number: with a buffer deeper than the run,
    *each* consumer must see every frame and drop none. A consumer that
    had frames taken from it would show fewer offered; one that never
    drained would show ``offered - depth`` dropped.
    """
    stream = make_stream(blocks=8)
    hub = FeedHub(spectrum=stream)
    waterfall = hub.spectrum("waterfall")
    line = hub.spectrum("spectrum-line")

    run_to_completion(stream)

    assert len(waterfall.latest()) == 8
    assert len(line.latest()) == 8
    assert waterfall.stats.frames_dropped == 0
    assert line.stats.frames_dropped == 0
    assert waterfall.stats.frames_offered == line.stats.frames_offered == 8


def test_a_repeated_name_is_made_unique_rather_than_refused():
    """What makes a duplicated widget possible at all.

    ``SpectrumStream.subscribe`` raises on a name collision, correctly —
    two consumers sharing a label make the statistics unreadable. But
    the Custom tab builds widgets from a config file, and a second
    ``waterfall`` is exactly what a user will write there, with nobody
    in that path to invent a distinct name.
    """
    hub = FeedHub(spectrum=make_stream())
    first = hub.spectrum("waterfall")
    second = hub.spectrum("waterfall")
    third = hub.spectrum("waterfall")

    assert first.name == "waterfall"
    assert second.name == "waterfall-2"
    assert third.name == "waterfall-3"
    assert hub.claimed == ("waterfall", "waterfall-2", "waterfall-3")


def test_a_duplicated_feed_is_independent_not_shared():
    """A second instance must be a real second consumer, not an alias.

    The standing Phase 3 rule is that every widget must work as a second
    instance in the Custom tab. A duplicate that quietly shared the
    first one's buffer would satisfy the *name* check above while
    reproducing the exact frame-stealing bug Chunk A fixed.
    """
    stream = make_stream(blocks=5)
    hub = FeedHub(spectrum=stream)
    first = hub.spectrum("waterfall")
    second = hub.spectrum("waterfall")

    run_to_completion(stream)

    assert len(first.latest()) == 5
    # If the second were an alias of the first, the first's drain above
    # would have emptied it.
    assert len(second.latest()) == 5


def test_claiming_before_the_stream_starts_is_the_supported_order():
    """The Chunk A stall fix depends on this.

    The shell must finish building — QApplication, widgets, a realised
    window — before anything streams, because Session 24 measured a
    single ~1.03 s stall per run caused by doing it the other way round.
    So every feed has to be claimable while the stream is still stopped.
    """
    stream = make_stream(blocks=4)
    hub = FeedHub(spectrum=stream)
    assert not stream.is_running
    feed = hub.spectrum("waterfall")

    run_to_completion(stream)

    assert len(feed.latest()) == 4


def test_claiming_a_feed_does_not_start_anything():
    """The hub hands out views; it never owns a lifetime."""
    stream = make_stream()
    hub = FeedHub(spectrum=stream)
    hub.spectrum("waterfall")
    assert not stream.is_running


def test_claiming_without_a_stream_is_an_error_naming_the_way_out():
    hub = FeedHub()
    assert not hub.has_spectrum
    with pytest.raises(RuntimeError, match="has_spectrum"):
        hub.spectrum("waterfall")


def test_an_empty_feed_name_is_refused():
    hub = FeedHub(spectrum=make_stream())
    with pytest.raises(ValueError, match="name"):
        hub.spectrum("")


# ----------------------------------------------------------------------
# Levels are read: the same feed every call
# ----------------------------------------------------------------------


def test_a_level_feed_is_the_same_object_every_call():
    """The asymmetry, asserted rather than merely documented.

    Reading a level does not consume it, so there is nothing to divide
    and no reason to hand out a second one. If this ever starts
    returning new objects, the module's central claim has quietly
    stopped being true.
    """
    hub = FeedHub(radio=FakeRadio(), tracking=FakeLoop())
    assert hub.quieting is hub.quieting
    assert hub.tracked_frequency is hub.tracked_frequency
    assert hub.rotor is hub.rotor


def test_level_feeds_are_absent_rather_than_empty_when_nothing_is_attached():
    """ "Off" and "broken" must not look the same.

    A caller gets ``None`` and can draw *no SDR attached*, instead of a
    feed that reports ``None`` forever and a panel that looks like a
    working instrument watching a dead band.
    """
    hub = FeedHub()
    assert hub.quieting is None
    assert hub.tracked_frequency is None
    assert hub.rotor is None
    assert not hub.has_spectrum


def test_quieting_feed_forwards_the_live_values():
    radio = FakeRadio()
    feed = FeedHub(radio=radio).quieting
    assert isinstance(feed, QuietingFeed)
    assert feed.live_quieting_db == -3.0
    assert feed.live_squelch_open is True

    radio.live_quieting_db = -11.5
    radio.live_squelch_open = False
    assert feed.live_quieting_db == -11.5
    assert feed.live_squelch_open is False


def test_tracked_frequency_feed_forwards_the_live_value():
    radio = FakeRadio()
    feed = FeedHub(radio=radio).tracked_frequency
    assert isinstance(feed, TrackedFrequencyFeed)
    assert feed.live_tracked_frequency_hz == 435_605_000.0

    radio.live_tracked_frequency_hz = None
    assert feed.live_tracked_frequency_hz is None


def test_a_level_feed_does_not_expose_the_session_behind_it():
    """A panel that draws one bar must not be able to stop the receiver.

    The widget rule is that an element knows nothing about what contains
    it, and a ReceiveSession is very much a container. Two forwarded
    properties is the price of that being true by construction.
    """
    feed = FeedHub(radio=FakeRadio()).quieting
    assert not hasattr(feed, "stop")
    assert not hasattr(feed, "start")


def test_rotor_feed_carries_the_loop_and_its_fault_source():
    loop = FakeLoop()
    fault = RuntimeError("rotor went away")
    hub = FeedHub(tracking=loop, tracking_fault=lambda: fault)
    feed = hub.rotor
    assert isinstance(feed, RotorFeed)
    assert feed.loop is loop
    assert feed.fault() is fault
    assert feed.target_name == "RS-44"


def test_rotor_feed_defaults_to_reporting_no_fault():
    feed = FeedHub(tracking=FakeLoop()).rotor
    assert feed is not None
    assert feed.fault() is None


def test_tracking_fault_is_ignored_when_there_is_no_rotor():
    hub = FeedHub(tracking_fault=lambda: RuntimeError("nothing to report about"))
    assert hub.rotor is None


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def test_describe_names_what_is_attached_and_what_is_not():
    empty = FeedHub().describe()
    assert "spectrum no" in empty
    assert "radio no" in empty
    assert "rotor no" in empty

    full = FeedHub(spectrum=make_stream(), radio=FakeRadio(), tracking=FakeLoop()).describe()
    assert "spectrum yes" in full
    assert "radio yes" in full
    assert "rotor yes" in full
