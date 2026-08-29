"""The feed hub: one owner of the live sources, many independent feeds.

**Two kinds of feed, and the difference is real rather than stylistic.**
This module's whole shape follows from one observation made at the end
of Chunk C PR1 and deliberately left unanswered until now: *rotor state
is a level, not a stream.*

A spectrum frame is **consumed**. There is exactly one of each, and a
consumer that takes one has taken it from everybody else — which is not
a hypothetical, it is bench verification #11 (Session 24), where the
waterfall and the line trace alternated on real hardware because both
drained one shared buffer. The fix was
:meth:`~qsorbit.core.dsp.spectrum_stream.SpectrumStream.subscribe`, a
per-consumer bounded buffer, and it cost a chunk.

A rotor position is **not consumed**. Two readouts reading the same
latest sample cannot steal from each other, because reading a float does
not remove it. The same is true of the live quieting figure, the
squelch's open/closed decision, and the tracked downlink frequency:
every one of them is a value that is simply *there*, replaced whenever
something upstream replaces it.

So the hub does not offer one uniform API. Streams are **claimed** — a
verb, taking a name, returning something new each time. Levels are
**read** — a property, returning the same feed however often it is
asked. Wrapping a float in a subscription would have implied a
per-consumer buffer that does not exist and a stealing hazard that
cannot happen, and would have been the mistake Session 20 already
recorded in another form: *check whether a property is already
guaranteed upstream before adding machinery to guarantee it.*

**What the hub is for.** Widgets already receive their feeds and know
nothing about their container — the rule adopted in Session 19 and
honoured from Chunk F onward. What was missing was somebody to hand
those feeds out. Until now that job lived in
:func:`qsorbit.__main__._show_instruments`, which subscribed two
spectrum consumers by hand and passed a
:class:`~qsorbit.core.receive.ReceiveSession` straight into a widget.
That works for exactly one window with exactly one set of panels. A tab
that can be duplicated needs a source of feeds that can be asked again,
and needs it to not matter who asks.

**Nothing here starts or stops anything.** The hub is handed live
objects and hands out views of them; the session owns its own lifetime,
as it has since Chunk H, and the rotor owns its connection. This is not
tidiness — it is what keeps the Chunk A stall fix intact. A hub that
started the stream when its first feed was claimed would put the SDR
reader thread behind whichever widget was constructed first, which is
precisely the two-lines-in-the-wrong-order fault that cost 1.03 s of
every windowed run until Session 25 found it. The hub is therefore
constructible, and every feed claimable, **before anything streams** —
:meth:`~qsorbit.core.dsp.spectrum_stream.SpectrumStream.subscribe`
already guarantees that, and this module does nothing to weaken it.

**No Qt.** Same split PR1 settled for the theme system, and for the same
reason: :mod:`qsorbit.ui.theme` and :mod:`qsorbit.ui.theme_qss` are
importable without PySide6 and only :mod:`qsorbit.ui.theme_manager`
needs it. A hub that cannot be tested without a display is a hub whose
feed accounting is only ever checked by looking at panels, which is the
eyeball judgement Session 25 replaced with arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from qsorbit.core.dsp.spectrum_stream import SpectrumStream, SpectrumSubscription
from qsorbit.core.pointing import TrackingLoop


def _no_fault() -> None:
    """The default tracking-fault source: nothing is watching the ticker.

    Matches :func:`qsorbit.ui.readout_widget._no_fault`, whose docstring
    explains why returning ``None`` forever is honest rather than
    optimistic here.
    """
    return None


class RadioSource(Protocol):
    """The live levels a receive session publishes while it runs.

    Declared structurally rather than importing
    :class:`~qsorbit.core.receive.ReceiveSession` as a type, matching
    :class:`~qsorbit.ui.waterfall_widget.FrameSource`,
    :class:`~qsorbit.ui.quieting_widget.QuietingSource` and
    :class:`~qsorbit.core.tracker.Target`: a test double satisfies it by
    having the three properties, without building a real session's
    stream, audio device and rotor.

    All three are the live-polling contract
    :attr:`~qsorbit.core.receive.ReceiveSession.live_quieting_db`
    describes — a single attribute read, safe to poll from a different
    thread than the one updating it, tolerant of being one block stale,
    and explicitly *not* the reporting numbers, which have to add up.
    """

    @property
    def live_quieting_db(self) -> float | None:
        """See :attr:`~qsorbit.core.receive.ReceiveSession.live_quieting_db`."""
        ...

    @property
    def live_squelch_open(self) -> bool | None:
        """See :attr:`~qsorbit.core.receive.ReceiveSession.live_squelch_open`."""
        ...

    @property
    def live_tracked_frequency_hz(self) -> float | None:
        """See :attr:`~qsorbit.core.receive.ReceiveSession.live_tracked_frequency_hz`."""
        ...


class QuietingFeed:
    """A level feed: the squelch's live measurement and its decision.

    Satisfies :class:`~qsorbit.ui.quieting_widget.QuietingSource`, so a
    :class:`~qsorbit.ui.quieting_widget.QuietingWidget` takes one of
    these unchanged.

    **A wrapper rather than the session itself**, even though the session
    already satisfies the protocol. Handing a widget the whole session
    would give a panel that draws one bar the ability to call
    :meth:`~qsorbit.core.receive.ReceiveSession.stop`, and the widget
    rule is that an element knows nothing about what contains it — a
    session is very much a container. Two forwarded properties is a
    cheap price for that staying true by construction rather than by
    everyone remembering.

    Args:
        source: Where the levels come from.
    """

    __slots__ = ("_source",)

    def __init__(self, source: RadioSource) -> None:
        self._source = source

    @property
    def live_quieting_db(self) -> float | None:
        """How far the channel is quieted, in dB, or ``None`` if unmeasured."""
        return self._source.live_quieting_db

    @property
    def live_squelch_open(self) -> bool | None:
        """Whether the gate is open right now, or ``None`` if there is no squelch."""
        return self._source.live_squelch_open


class TrackedFrequencyFeed:
    """A level feed: where the tracked downlink actually sits right now.

    Satisfies
    :class:`~qsorbit.ui.zoom_controller.TrackedFrequencySource`, so a
    :class:`~qsorbit.ui.zoom_controller.ZoomController` takes one of
    these unchanged and can drive its frequency lock without either
    spectrum widget needing to know that tracking exists.

    Args:
        source: Where the level comes from.
    """

    __slots__ = ("_source",)

    def __init__(self, source: RadioSource) -> None:
        self._source = source

    @property
    def live_tracked_frequency_hz(self) -> float | None:
        """The tracked downlink's true RF frequency, or ``None`` before the first sample."""
        return self._source.live_tracked_frequency_hz


class RotorFeed:
    """A level feed: the tracking loop's latest sample, and whether it died.

    Bundles the two things
    :class:`~qsorbit.ui.readout_widget.ReadoutWidget` needs — the loop to
    read and the fault callable to ask — so a tab does not have to know
    that a readout without a fault source will happily show frozen
    numbers under a dead rotor for the rest of a pass.

    **This feed does not tick.** Exactly one thing ticks the loop, and
    since Chunk A PR2 that is the session's tracking thread, on a thread
    nobody is looking at. Reading
    :attr:`~qsorbit.core.pointing.TrackingLoop.latest_sample` touches no
    serial port, which is the whole reason rotor state can be a level at
    all: if reading it cost an RS-485 round trip, two readouts would be
    two rounds of traffic and the asymmetry in this module would be a
    lie.

    Args:
        loop: The tracking loop to read. Something else ticks it.
        fault: Asked whether the thing ticking the loop has died, and
            handed whatever killed it.
    """

    __slots__ = ("_fault", "_loop")

    def __init__(
        self,
        loop: TrackingLoop,
        *,
        fault: Callable[[], BaseException | None] = _no_fault,
    ) -> None:
        self._loop = loop
        self._fault = fault

    @property
    def loop(self) -> TrackingLoop:
        """The loop to read. Reading is free; ticking belongs to somebody else."""
        return self._loop

    @property
    def fault(self) -> Callable[[], BaseException | None]:
        """Asked on every repaint whether the ticker has died."""
        return self._fault

    @property
    def target_name(self) -> str:
        """What is being tracked, for a window title or a top bar."""
        return self._loop.target.name


class FeedHub:
    """Owns the live sources and hands out feeds onto them.

    Every source is optional, and that is the point rather than a
    convenience. A rotor-only evening, a receiver with nothing attached
    to COM5, and a pass with both are the same shell with different tabs
    live — the same asymmetry ``receive`` has had since Chunk H, where a
    rotor fault costs you the antenna pointing and nothing else.

    **Streams are claimed, levels are read.** See the module docstring
    for why. In practice:

    - :meth:`spectrum` is a method taking a name and returns a **new**
      independent subscription every call. Two calls give two feeds that
      cannot take frames from each other.
    - :attr:`quieting`, :attr:`tracked_frequency` and :attr:`rotor` are
      properties returning the **same** feed every call, because there is
      nothing to divide.

    Absence is reported as ``None`` from the level properties and by
    :attr:`has_spectrum` for the stream, so a caller can build a
    placeholder that says *no SDR attached* rather than an empty panel.
    "Off" and "broken" must never look the same — the rule this project
    wrote down after a healthy headless run reported 453 dropped blocks.

    Args:
        spectrum: The spectrum stream to fan out, or ``None`` when no SDR
            is attached. **Need not have been started**, and claiming
            feeds before it starts is the intended order: it is what lets
            the shell finish building before anything streams, which is
            the Chunk A stall fix. Late subscription also works — see
            :meth:`~qsorbit.core.dsp.spectrum_stream.SpectrumStream.subscribe`
            — but relying on it would put the graphics stack back in
            front of a running reader thread.
        radio: The live receive levels, or ``None`` when nothing is being
            received. Supplies both :attr:`quieting` and
            :attr:`tracked_frequency`, since one session publishes both
            and splitting them would let a caller hand a widget a
            quieting feed from one run and a frequency feed from another.
        tracking: The tracking loop, or ``None`` when no rotor is
            connected.
        tracking_fault: Asked whether whatever ticks ``tracking`` has
            died. Ignored when ``tracking`` is ``None``. Defaults to a
            source that never reports one, for a caller that ticks the
            loop itself.
    """

    def __init__(
        self,
        *,
        spectrum: SpectrumStream | None = None,
        radio: RadioSource | None = None,
        tracking: TrackingLoop | None = None,
        tracking_fault: Callable[[], BaseException | None] = _no_fault,
    ) -> None:
        self._spectrum = spectrum
        self._radio = radio
        self._claimed: list[str] = []

        self._quieting = QuietingFeed(radio) if radio is not None else None
        self._tracked_frequency = TrackedFrequencyFeed(radio) if radio is not None else None
        self._rotor = RotorFeed(tracking, fault=tracking_fault) if tracking is not None else None

    # ------------------------------------------------------------------
    # Streams — claimed
    # ------------------------------------------------------------------

    @property
    def has_spectrum(self) -> bool:
        """Whether there is a spectrum stream to claim feeds from."""
        return self._spectrum is not None

    @property
    def claimed(self) -> tuple[str, ...]:
        """Every spectrum feed name handed out, in the order claimed.

        The names that will label the per-consumer rows of
        :class:`~qsorbit.core.dsp.spectrum_stream.SpectrumStreamStats` at
        the end of a run. Exposed because that report is the evidence
        for "no widget stole frames from any other" — Session 25 proved
        the fan-out by reading 2,407 offered and 0 dropped per consumer
        rather than by watching two panels and judging whether either
        looked frozen — and a report you cannot map back onto the panels
        that produced it is not evidence of anything.
        """
        return tuple(self._claimed)

    def spectrum(self, name: str) -> SpectrumSubscription:
        """Claim an independent spectrum feed. A new one every call.

        Satisfies :class:`~qsorbit.ui.waterfall_widget.FrameSource`
        directly, so the returned object goes straight into a
        :class:`~qsorbit.ui.waterfall_widget.WaterfallWidget` or a
        :class:`~qsorbit.ui.spectrum_line_widget.SpectrumLineWidget` with
        no adapter and no change to either widget.

        **A repeated name is made unique rather than refused**, and that
        is what makes a duplicated widget possible.
        :meth:`~qsorbit.core.dsp.spectrum_stream.SpectrumStream.subscribe`
        raises on a collision, correctly — two consumers sharing a label
        would make the statistics unreadable. But the Custom tab builds
        its widgets from a list in a config file, so a second
        ``"waterfall"`` is exactly what a user will ask for, and there is
        nobody in that path to invent a distinct name. The hub invents
        one: the second claim of ``"waterfall"`` is reported as
        ``"waterfall-2"``, the third as ``"waterfall-3"``. The statistics
        stay readable *and* the second instance works, which the widget
        rule requires — every widget must work as a second instance in
        the Custom tab, or the design is wrong.

        Args:
            name: What this consumer should be called in the run's
                per-consumer statistics. Suffixed with ``-2``, ``-3`` and
                so on if already taken.

        Returns:
            The subscription to hand that widget.

        Raises:
            RuntimeError: If there is no spectrum stream. Check
                :attr:`has_spectrum` first and build a placeholder
                instead — a panel drawing nothing and a panel with no
                radio behind it must not look the same.
            ValueError: If ``name`` is empty.
        """
        if self._spectrum is None:
            raise RuntimeError(
                "This hub has no spectrum stream, so there is nothing to claim a feed "
                "from. Check has_spectrum and show a placeholder instead."
            )
        if not name:
            raise ValueError("A feed needs a name; it is what labels the statistics.")

        unique = name
        suffix = 1
        while unique in self._claimed:
            suffix += 1
            unique = f"{name}-{suffix}"

        subscription = self._spectrum.subscribe(unique)
        self._claimed.append(unique)
        return subscription

    # ------------------------------------------------------------------
    # Levels — read
    # ------------------------------------------------------------------

    @property
    def quieting(self) -> QuietingFeed | None:
        """The squelch's live measurement and decision, or ``None``.

        The same feed every call. Nothing is consumed by reading it, so
        every quieting panel in the application can share one — see the
        module docstring.
        """
        return self._quieting

    @property
    def tracked_frequency(self) -> TrackedFrequencyFeed | None:
        """Where the tracked downlink sits right now, or ``None``.

        The same feed every call, for the same reason as
        :attr:`quieting`. Note that the *zoom controller* reading it is
        not shared: each spectrum group builds its own, so a gesture on
        one pair of panels does not move another's.
        """
        return self._tracked_frequency

    @property
    def rotor(self) -> RotorFeed | None:
        """The tracking loop's latest sample and fault state, or ``None``.

        The same feed every call. Reading
        :attr:`~qsorbit.core.pointing.TrackingLoop.latest_sample` touches
        no serial port, so two readouts cost exactly what one does.
        """
        return self._rotor

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """One line naming what is attached and what is not.

        Printed when the shell opens. A tab that is dark because no
        hardware is attached and a tab that is dark because something
        broke look identical on screen for the first few seconds, and
        this is what tells them apart without waiting.
        """
        parts = (
            f"spectrum {'yes' if self.has_spectrum else 'no'}",
            f"radio {'yes' if self._radio is not None else 'no'}",
            f"rotor {'yes' if self._rotor is not None else 'no'}",
        )
        return "feeds: " + ", ".join(parts)
