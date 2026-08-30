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

from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout, QWidget

from qsorbit.core.dsp.spectrum import frequency_axis_hz
from qsorbit.core.horizon import HorizonMask
from qsorbit.core.profiles import CatalogManifest, ProfileCatalog
from qsorbit.core.tracker import ObserverLocation
from qsorbit.ui.cards import Card, Placeholder
from qsorbit.ui.custom_tab import (
    KNOWN_WIDGETS,
    CustomTabConfig,
    custom_tab_config_path,
)
from qsorbit.ui.feed_hub import FeedHub
from qsorbit.ui.frequency_widget import FrequencyWidget
from qsorbit.ui.picker_widget import PickerWidget
from qsorbit.ui.quieting_widget import QuietingWidget
from qsorbit.ui.readout_widget import ReadoutWidget
from qsorbit.ui.spectrum_line_widget import SpectrumLineWidget
from qsorbit.ui.theme_manager import ThemeManager
from qsorbit.ui.waterfall_render import WaterfallScale
from qsorbit.ui.waterfall_widget import WaterfallWidget
from qsorbit.ui.zoom_controller import ZoomController
from qsorbit.ui.zoom_controls_widget import ZoomControlsWidget

#: **Minimum** width of the Radio tab's right-hand column of small
#: cards, not a fixed one. See :data:`ROTOR_COLUMN_WIDTH` for what a
#: fixed width cost.
SIDE_COLUMN_WIDTH: Final = 300

#: **Minimum** width of the Rotor tab's left-hand column.
#:
#: These were ``setFixedWidth`` when the shell shipped, and both were
#: guesses about content this module does not own. With a real rotor
#: attached the readout's value column needed 249 px and got 142, so
#: four of its six rows were clipped mid-word on screen -- "39131 km,
#: approaching at 0." with the rate itself gone, which is a readout
#: quietly dropping the number it exists to show.
#:
#: The mocked rotor used while building the shell reported no sample at
#: all, so every value was a one-character placeholder and nothing was
#: ever too wide. The bug needed real hardware to appear, and a fixed
#: width is what made it possible: **a container must not assert how
#: wide somebody else's content is.** A minimum lets a column start at a
#: sensible size and grow to whatever it is actually given to show.
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
        right.setMinimumWidth(SIDE_COLUMN_WIDTH)

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
        left.setMinimumWidth(ROTOR_COLUMN_WIDTH)

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
    """The target picker's home.

    The table and its filter chips shipped in Chunk D PR2; the
    ground-track map is PR3's, and this tab still shows a placeholder
    where it will go.

    Args:
        themes: Passed to every widget that draws its own pixels.
        catalog: The curated profile catalogue the picker matches TLEs
            against.
        catalog_manifest: The catalogue's optional shipped-date
            manifest, for the picker's staleness line.
        tle_dir: This station's configured TLE directory
            (``[planning] tle_dir`` in station config), or ``None`` if
            it has not been set -- in which case this tab shows a
            placeholder naming the config key rather than a picker with
            nothing to search, the same "off, not broken" convention
            :class:`RadioTab` and :class:`RotorTab` already follow for
            a missing feed.
        observer: This station's location, for pass prediction.
        horizon: This station's own horizon mask, applied the same way
            ``qsorbit plan`` applies it.
    """

    def __init__(
        self,
        *,
        themes: ThemeManager,
        catalog: ProfileCatalog,
        catalog_manifest: CatalogManifest | None,
        tle_dir: str | None,
        observer: ObserverLocation,
        horizon: HorizonMask,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        if tle_dir is None:
            content: QWidget = Placeholder(
                "No TLE directory configured. Set tle_dir under [planning] in "
                "your station config to light up the target picker here -- "
                "everything it needs already works from the command line: "
                "try `qsorbit plan --tle-dir PATH`."
            )
        else:
            content = PickerWidget(catalog, catalog_manifest, tle_dir, observer, horizon)

        layout.addWidget(Card("Plan", content, themes=themes, index=0, stretch=True), 1)


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


class CustomTab(QWidget):
    """A grid of widgets named by a config file, not by code.

    Built the same way every other tab in this module is built --
    feeds are claimed here, one call per cell, so a config asking for
    ``"waterfall"`` twice gets two independent instances through
    :meth:`~qsorbit.ui.feed_hub.FeedHub.spectrum`'s own ``-2``, ``-3``
    suffixing. Nothing about *how* a cell is built differs from
    :class:`RadioTab` or :class:`RotorTab`; what differs is that the
    list of cells comes from a file instead of from this module's own
    code -- this class is the thing every other tab's docstring has
    been promising exists.

    **Every cell is independent**, deliberately unlike the Radio tab's
    waterfall and spectrum line, which share one
    :class:`~qsorbit.ui.zoom_controller.ZoomController` so a gesture on
    one moves the other. A config-driven grid has no guarantee two
    spectrum cells are even related to each other -- a user might ask
    for two waterfalls, or a waterfall with no spectrum line anywhere
    in the tab -- so nothing here invents a pairing nobody asked for.
    Each spectrum cell gets its own controller and its own zoom
    controls beneath it, same as a lone waterfall on the Radio tab
    would if the Radio tab ever built one alone.

    **A missing config and a broken one read differently.**
    ``config=None`` with ``error=None`` means nobody has written a
    :func:`~qsorbit.ui.custom_tab.custom_tab_config_path` file yet --
    the normal state for a fresh install. ``config=None`` with
    ``error`` set means one exists and failed to load, and ``error``
    is shown verbatim: the caller already ran it through
    :func:`~qsorbit.ui.custom_tab.load_custom_tab_config` and caught
    :class:`~qsorbit.ui.custom_tab.CustomTabConfigError`. "Off" and
    "broken" must still read differently even for a tab that is
    entirely optional.

    Args:
        hub: Where every cell's feed comes from.
        themes: Passed to every widget that draws its own pixels.
        config: The validated config, or ``None`` if it is missing or
            broken.
        error: Why ``config`` is ``None``, in words. ``None`` means
            there simply is no file yet, rather than a load failure.
        nominal_hz: Threaded through to any ``frequency`` cell, same
            meaning as :class:`RadioTab`'s own parameter.
    """

    def __init__(
        self,
        hub: FeedHub,
        *,
        themes: ThemeManager,
        config: CustomTabConfig | None,
        error: str | None = None,
        nominal_hz: float | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        if config is None:
            if error is not None:
                message = error
            else:
                path = custom_tab_config_path()
                message = (
                    f"No {path.name} found at {path}. Create one to build a "
                    "grid from named widgets: "
                    f"{', '.join(sorted(KNOWN_WIDGETS))}. See "
                    "custom_tab.example.toml in the repo for the format."
                )
            layout.addWidget(
                Card("Custom", Placeholder(message), themes=themes, index=0, stretch=True), 1
            )
            return

        grid = QGridLayout()
        grid.setSpacing(10)
        for cell_index, name in enumerate(config.widgets):
            row, column = divmod(cell_index, config.columns)
            grid.addWidget(
                self._build_cell(name, cell_index, hub=hub, themes=themes, nominal_hz=nominal_hz),
                row,
                column,
            )
        grid_host = QWidget(self)
        grid_host.setLayout(grid)
        layout.addWidget(grid_host)
        layout.addStretch(1)

    def _build_cell(
        self,
        name: str,
        index: int,
        *,
        hub: FeedHub,
        themes: ThemeManager,
        nominal_hz: float | None,
    ) -> Card:
        """One grid cell: a card wrapping one named widget.

        Mirrors the per-widget placeholder logic :class:`RadioTab` and
        :class:`RotorTab` already use for a missing feed -- an empty
        cell and a cell whose hardware died must not look the same, so
        an absent feed is always a placeholder naming what is missing
        rather than a blank space in the grid.

        ``name`` is guaranteed to be a member of
        :data:`~qsorbit.ui.custom_tab.KNOWN_WIDGETS` --
        :func:`~qsorbit.ui.custom_tab.load_custom_tab_config` already
        rejected anything outside that set, so the fall-through branch
        below is only ever reached for ``"rotor_readout"``.
        """
        title = name.replace("_", " ").title()

        if name in ("waterfall", "spectrum_line"):
            if not hub.has_spectrum:
                return Card(
                    title,
                    Placeholder("No SDR attached, so there is no spectrum to draw.", compact=True),
                    themes=themes,
                    index=index,
                )
            feed_name = WATERFALL_FEED if name == "waterfall" else SPECTRUM_LINE_FEED
            feed = hub.spectrum(feed_name)
            axis = frequency_axis_hz(feed.config)
            zoom = ZoomController(
                float(axis[0]),
                float(axis[-1]),
                tracked_frequency_source=hub.tracked_frequency,
                parent=self,
            )
            scale = WaterfallScale()
            body, body_layout = _column(spacing=8)
            if name == "waterfall":
                body_layout.addWidget(
                    WaterfallWidget(feed, themes=themes, zoom=zoom, scale=scale), 1
                )
            else:
                body_layout.addWidget(
                    SpectrumLineWidget(feed, themes=themes, zoom=zoom, scale=scale)
                )
            body_layout.addWidget(ZoomControlsWidget(zoom))
            return Card(title, body, themes=themes, index=index, stretch=(name == "waterfall"))

        if name == "quieting":
            quieting = hub.quieting
            if quieting is None:
                return Card(
                    title,
                    Placeholder("Nothing is being received.", compact=True),
                    themes=themes,
                    index=index,
                )
            return Card(title, QuietingWidget(quieting), themes=themes, index=index)

        if name == "frequency":
            tracked = hub.tracked_frequency
            if tracked is None:
                return Card(
                    title,
                    Placeholder("Nothing is being received.", compact=True),
                    themes=themes,
                    index=index,
                )
            return Card(
                title,
                FrequencyWidget(tracked, nominal_hz=nominal_hz),
                themes=themes,
                index=index,
            )

        rotor = hub.rotor
        if rotor is None:
            return Card(
                title,
                Placeholder("No rotor connected.", compact=True),
                themes=themes,
                index=index,
            )
        return Card(title, ReadoutWidget(rotor.loop, fault=rotor.fault), themes=themes, index=index)
