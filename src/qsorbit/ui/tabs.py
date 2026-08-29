"""The shell's built-in tabs.

Each tab is a plain :class:`QWidget` handed a
:class:`~qsorbit.ui.feed_hub.FeedHub` and a
:class:`~qsorbit.ui.theme_manager.ThemeManager`, and builds its own
widget instances from feeds it claims itself. **No tab shares a widget
with another tab**, which is the Phase 2 Chunk F convention finally
carrying real weight: a Qt widget has exactly one parent, so a Custom
tab showing "the waterfall" cannot be the Radio tab's waterfall
re-parented -- it has to be a second instance with a second feed. Every
tab here is therefore built the way the Custom tab will have to build
things in PR3, rather than being a special case the config-driven path
then has to imitate.

**Zoom is per-tab, not per-application.** Phil's call at PR2 kickoff:
each spectrum group owns its own
:class:`~qsorbit.ui.zoom_controller.ZoomController` and
:class:`~qsorbit.ui.waterfall_render.WaterfallScale`, so a gesture on
the Radio tab does not move a second waterfall in the Custom tab. A
duplicate that always showed exactly what the original showed would be a
weak answer to why anyone would want one. Within a tab the controller is
shared by the line trace and the waterfall, as it has been since Chunk
I, so those two stay locked together -- which is the pairing a person
actually gestures at.

**Where a feed is missing, a tab says so in words.** Not an empty panel:
"off" and "broken" must never look the same, and an instrument drawing
nothing looks identical to one whose hardware died. See
:class:`~qsorbit.ui.cards.Placeholder`.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from qsorbit.core.dsp.spectrum import frequency_axis_hz
from qsorbit.ui.cards import Card, Placeholder
from qsorbit.ui.feed_hub import FeedHub
from qsorbit.ui.frequency_widget import FrequencyWidget
from qsorbit.ui.quieting_widget import QuietingWidget
from qsorbit.ui.readout_widget import ReadoutWidget
from qsorbit.ui.spectrum_line_widget import SpectrumLineWidget
from qsorbit.ui.theme_manager import ThemeManager
from qsorbit.ui.waterfall_render import WaterfallScale
from qsorbit.ui.waterfall_widget import WaterfallWidget
from qsorbit.ui.zoom_controller import ZoomController
from qsorbit.ui.zoom_controls_widget import ZoomControlsWidget

#: Width of the Radio tab's right-hand column of small cards. Wide
#: enough for the frequency readout at its 30 px size, which is the
#: widest thing that has to fit.
SIDE_COLUMN_WIDTH: Final = 300

#: Width of the Rotor tab's left-hand column.
ROTOR_COLUMN_WIDTH: Final = 320

#: Feed names claimed by a Radio tab, in claim order. A second Radio
#: tab -- or a Custom tab asking for the same widgets -- gets
#: ``waterfall-2`` and so on, without either tab knowing it happened.
WATERFALL_FEED: Final = "waterfall"
SPECTRUM_LINE_FEED: Final = "spectrum-line"


def _column(*, spacing: int = 10) -> tuple[QWidget, QVBoxLayout]:
    """A bare vertical column widget and its layout."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    return widget, layout


class RadioTab(QWidget):
    """Spectrum, waterfall, zoom, and the live receive levels.

    Args:
        hub: Where the feeds come from.
        themes: The active theme, passed to every widget that draws its
            own pixels.
        nominal_hz: The transmitter's rest frequency, for the Doppler
            line on the frequency card. ``None`` when unknown.
    """

    def __init__(
        self,
        hub: FeedHub,
        *,
        themes: ThemeManager,
        nominal_hz: float | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        left, left_layout = _column()
        right, right_layout = _column()
        right.setFixedWidth(SIDE_COLUMN_WIDTH)

        if hub.has_spectrum:
            # Claimed before any widget is constructed and long before
            # anything streams -- see FeedHub.spectrum on why that
            # ordering is the Chunk A stall fix rather than a style
            # choice.
            line_feed = hub.spectrum(SPECTRUM_LINE_FEED)
            waterfall_feed = hub.spectrum(WATERFALL_FEED)

            # From the feed's own config rather than the stream's: a
            # subscription reports the framing it is handing over, so
            # the axis and the frames cannot come from two objects that
            # quietly disagree.
            axis = frequency_axis_hz(line_feed.config)
            tracked = hub.tracked_frequency
            zoom = ZoomController(
                float(axis[0]),
                float(axis[-1]),
                tracked_frequency_source=tracked,
                parent=self,
            )
            # Shared between this tab's two spectrum panels so the line
            # trace and the waterfall's colour ramp agree on what "loud"
            # means. Not shared with any other tab.
            scale = WaterfallScale()

            left_layout.addWidget(
                Card(
                    "Spectrum",
                    SpectrumLineWidget(line_feed, themes=themes, zoom=zoom, scale=scale),
                    themes=themes,
                    index=0,
                )
            )
            waterfall_body, waterfall_layout = _column(spacing=8)
            waterfall_layout.addWidget(
                WaterfallWidget(waterfall_feed, themes=themes, zoom=zoom, scale=scale), 1
            )
            waterfall_layout.addWidget(ZoomControlsWidget(zoom))
            left_layout.addWidget(
                Card("Waterfall", waterfall_body, themes=themes, index=1, stretch=True), 1
            )
        else:
            left_layout.addWidget(
                Card(
                    "Spectrum",
                    Placeholder(
                        "No SDR attached, so there is no spectrum to draw. "
                        "Start the shell with --receive, or plug in a dongle "
                        "and start it again."
                    ),
                    themes=themes,
                    index=0,
                    stretch=True,
                ),
                1,
            )

        tracked = hub.tracked_frequency
        if tracked is not None:
            right_layout.addWidget(
                Card(
                    "Frequency",
                    FrequencyWidget(tracked, nominal_hz=nominal_hz),
                    themes=themes,
                    index=0,
                )
            )
        quieting = hub.quieting
        if quieting is not None:
            right_layout.addWidget(
                Card("Quieting / squelch", QuietingWidget(quieting), themes=themes, index=1)
            )
        if tracked is None and quieting is None:
            right_layout.addWidget(
                Card(
                    "Receiver",
                    Placeholder(
                        "Nothing is being received, so there are no live levels.",
                        compact=True,
                    ),
                    themes=themes,
                    index=0,
                )
            )

        # Present, greyed, and honest about why. The mockup carries this
        # card with branch B greyed out until Chunk E's second SDR
        # exists; leaving it out entirely would make the Radio tab look
        # finished when it is not, and inventing a meter with nothing
        # behind it would be worse than either.
        right_layout.addWidget(
            Card(
                "Branches",
                Placeholder(
                    "Single SDR. Branch B lights up with Chunk E's second dongle.",
                    compact=True,
                ),
                themes=themes,
                index=2,
            )
        )
        right_layout.addStretch(1)

        layout.addWidget(left, 1)
        layout.addWidget(right)


class RotorTab(QWidget):
    """Where the antenna is pointing, and where the sky target is.

    The pass arc the mockup draws over a horizon mask is explicitly
    later work -- the roadmap's own Chunk C line reads "Rotor (position
    readout; pass arc later)" -- so this tab shows the readout and says
    plainly what is not here yet rather than drawing an empty circle.
    """

    def __init__(
        self,
        hub: FeedHub,
        *,
        themes: ThemeManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        left, left_layout = _column()
        left.setFixedWidth(ROTOR_COLUMN_WIDTH)

        rotor = hub.rotor
        if rotor is not None:
            left_layout.addWidget(
                Card(
                    "Position",
                    ReadoutWidget(rotor.loop, fault=rotor.fault),
                    themes=themes,
                    index=0,
                )
            )
        else:
            left_layout.addWidget(
                Card(
                    "Position",
                    Placeholder(
                        "No rotor connected. The receiver runs without one - "
                        "Doppler correction comes from the orbit and your "
                        "location, not from the antenna position.",
                        compact=True,
                    ),
                    themes=themes,
                    index=0,
                )
            )
        left_layout.addStretch(1)

        layout.addWidget(left)
        layout.addWidget(
            Card(
                "Sky view",
                Placeholder(
                    "The pass arc over this station's horizon mask arrives "
                    "later in Chunk C. The mask itself already exists - Chunk "
                    "B reads it from [[horizon]] in your station config."
                ),
                themes=themes,
                index=1,
                stretch=True,
            ),
            1,
        )


class PlanTab(QWidget):
    """The target picker's home. Populated in Chunk D."""

    def __init__(self, *, themes: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(
            Card(
                "Plan",
                Placeholder(
                    "The target picker and the ground-track map arrive in "
                    "Chunk D. Everything they need already works from the "
                    "command line: try `qsorbit plan`."
                ),
                themes=themes,
                index=0,
                stretch=True,
            ),
            1,
        )


class DecodeTab(QWidget):
    """A placeholder, and gated. Filled by the stretch Chunk G, if it runs."""

    def __init__(self, *, themes: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(
            Card(
                "Decode",
                Placeholder(
                    "Reserved for the CW decoder - the stretch Chunk G, which "
                    "runs only if the phase does not run long. A decoder adds "
                    "read load to the receive path, so it stayed gated on the "
                    "USB stall being fixed rather than merely diagnosed."
                ),
                themes=themes,
                index=0,
                stretch=True,
            ),
            1,
        )
