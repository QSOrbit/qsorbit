"""The map: ground tracks and visibility footprints for the picker's current selection.

Chunk D PR3, the last piece of Chunk D's own done-when clause -- see
``phase-3-roadmap.md``. Same split as every other panel in this
package: :mod:`qsorbit.core.map_projection`,
:mod:`qsorbit.core.coastlines`,
:mod:`qsorbit.core.tracker.ground_track`, and
:mod:`qsorbit.core.orbit_geometry` do the geometry with no Qt; this
module owns a toggle, a coordinate transform, and a ``paintEvent``.

**Fed by another widget, not by the hub.** Every other panel in this
package receives its feed from :mod:`qsorbit.ui.feed_hub` -- a live
hardware source, claimed once, read or subscribed to from then on. The
map's own feed is different in kind: "which satellites to draw" is the
picker's *current filtered selection*, a UI-level fact about what
chips are toggled on :class:`~qsorbit.ui.picker_widget.PickerWidget`'s
own table, not a hardware reading the hub has any business carrying.
:class:`~qsorbit.ui.tabs.PlanTab` wires
:attr:`~qsorbit.ui.picker_widget.PickerWidget.entries_changed` straight
to :meth:`MapWidget.set_visible_entries`, so the two widgets still
know nothing about each other -- one just happens to feed the other,
the same "widgets receive feeds and know nothing about their
container" convention applied sibling-to-sibling instead of hub-to-widget.
Chunk D's roadmap entry also asks for a hub-fed Custom-tab instance;
that scope was deliberately cut from this PR (see ``project-notes.md``)
and is its own small follow-up.

**Two projections, one set of geometry.** Every point this widget
draws -- coastlines, the station marker, a ground track, a footprint
ring -- is computed once in plain latitude/longitude and handed to
:func:`~qsorbit.core.map_projection.project_polyline` (or, for the
single-point station marker,
:func:`~qsorbit.core.map_projection.equirectangular` /
:func:`~qsorbit.core.map_projection.orthographic` directly), so
switching the toggle never touches the underlying geometry, only which
projection function converts it to a picture.

**The footprint is stroked, not filled.** The mockup's own note calls
it a "shaded circle." A filled circle is trivial when the ring stays
in one unbroken piece, which is the common case, but
:func:`~qsorbit.core.map_projection.project_polyline` can hand back a
footprint ring broken into several pieces -- straddling the
antimeridian on the flat map, or clipped by the globe's own limb for a
satellite near the edge of the station's current view -- and filling a
broken ring as one polygon would either self-intersect or silently
close a gap that is not really there. A stroked ring degrades to
exactly the pieces that are actually correct in both cases, which
matters more here than matching the mockup's fill pixel-for-pixel.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from qsorbit.core.coastlines import load_coastlines
from qsorbit.core.map_projection import Projection, equirectangular, orthographic, project_polyline
from qsorbit.core.orbit_geometry import footprint_circle, footprint_radius_deg
from qsorbit.core.picker import PickerEntry
from qsorbit.core.tracker import ObserverLocation, Satellite, load_satellites_by_norad_id
from qsorbit.core.tracker.ground_track import ground_track
from qsorbit.ui.theme_manager import ThemeManager, accent_bar_color, theme_color

#: The station marker's glyph, matching the shell mockup's own "⌂ = station" note.
_STATION_GLYPH = "⌂"

#: Pen width for a ground track, in pixels.
_TRACK_WIDTH_PX = 2

#: Pen width for a footprint ring, in pixels -- thinner than the track
#: it belongs to, so the two read as "the line, and its footprint"
#: rather than two equally-weighted marks.
_FOOTPRINT_WIDTH_PX = 1


def _utc_now() -> datetime:
    """The current instant, timezone-aware. Injected so tests can fake it."""
    return datetime.now(UTC)


def _make_toggle_chip(label: str) -> QPushButton:
    """A checkable projection-toggle button, matching
    :class:`~qsorbit.ui.picker_widget.PickerWidget`'s own filter-chip
    look -- but exclusive (exactly one of flat/globe is ever checked),
    not independent, so this module keeps its own copy rather than
    importing that module's private helper for a subtly different
    contract.
    """
    chip = QPushButton(label)
    chip.setCheckable(True)
    return chip


def _fit_transform(rect: QRectF, projection: Projection) -> tuple[float, float, float]:
    """The scale and screen-pixel center to fit this canvas's active projection into ``rect``.

    :data:`Projection.FLAT` fills a 360-by-180-degree extent (longitude
    by latitude, a 2:1 aspect ratio); :data:`Projection.GLOBE` fills the
    unit disk, 2 units across on every axis. Fitting either one into a
    canvas rect of whatever size the layout actually gave this widget
    uses the tighter of the two axis constraints, so the whole shape
    stays visible and undistorted -- letterboxed on the loose axis
    rather than stretched to fill it.

    Args:
        rect: The canvas's own paint rect, in screen pixels.
        projection: Which projection is active -- selects the extent
            being fit.

    Returns:
        ``(scale, center_x, center_y)``. ``scale`` converts a
        projection-space unit to a screen-pixel offset (also the
        globe boundary circle's own pixel radius, when
        ``projection`` is :data:`Projection.GLOBE`); ``(center_x,
        center_y)`` is the screen-pixel origin that projection-space
        ``(0, 0)`` maps to -- this rect's own center, so the map stays
        centered regardless of the canvas's exact width and height.
    """
    if projection is Projection.FLAT:
        extent_x, extent_y = 360.0, 180.0
    else:
        extent_x, extent_y = 2.0, 2.0
    scale = min(rect.width() / extent_x, rect.height() / extent_y)
    center = rect.center()
    return scale, center.x(), center.y()


def _to_screen(
    point: tuple[float, float], scale: float, center_x: float, center_y: float
) -> QPointF:
    """One projection-space ``(x, y)`` point to a screen pixel.

    Flips the y axis: every projection in :mod:`qsorbit.core.map_projection`
    uses the map convention (``y`` grows north), while Qt's screen
    coordinates grow downward -- the single sign flip
    :mod:`qsorbit.core.map_projection`'s own module docstring leaves to
    "whichever widget is actually painting pixels," which is here.
    """
    x, y = point
    return QPointF(center_x + x * scale, center_y - y * scale)


def _project_point(
    latitude_deg: float,
    longitude_deg: float,
    projection: Projection,
    center_latitude_deg: float,
    center_longitude_deg: float,
) -> tuple[float, float] | None:
    """Dispatch a single point to whichever projection is active.

    The same per-point math :func:`~qsorbit.core.map_projection.project_polyline`
    applies to a whole track, exposed here for the station marker and a
    satellite's current position -- neither one a polyline, so neither
    needs that function's own antimeridian/limb splitting.

    Returns:
        ``(x, y)`` in that projection's own native units, or ``None`` if
        ``projection`` is :data:`Projection.GLOBE` and the point is on
        the far side of the globe from the view center -- matching
        :func:`~qsorbit.core.map_projection.orthographic`'s own
        contract, since :data:`Projection.FLAT` never returns ``None``.
    """
    if projection is Projection.FLAT:
        return equirectangular(latitude_deg, longitude_deg)
    return orthographic(
        latitude_deg,
        longitude_deg,
        center_latitude_deg=center_latitude_deg,
        center_longitude_deg=center_longitude_deg,
    )


def _segment_path(
    segment: tuple[tuple[float, float], ...], scale: float, center_x: float, center_y: float
) -> QPainterPath:
    """Build a drawable path from one :func:`~qsorbit.core.map_projection.project_polyline` segment.

    Every point in ``segment`` is already safe to connect to its
    neighbours with a straight line -- that splitting is
    ``project_polyline``'s own job, not this one's -- so this just
    walks the segment through :func:`_to_screen` and traces it.
    """
    path = QPainterPath()
    first_x, first_y = segment[0]
    path.moveTo(_to_screen((first_x, first_y), scale, center_x, center_y))
    for point in segment[1:]:
        path.lineTo(_to_screen(point, scale, center_x, center_y))
    return path


class MapWidget(QWidget):
    """Ground tracks and current visibility footprints for the picker's filtered entries.

    Args:
        themes: Provides the active theme and its change notification --
            every colour this widget draws comes from here, never a
            literal (see the Phase 3 standing rule, enforced by
            ``tests/unit/ui/test_no_hardcoded_colours.py``).
        tle_dir: Directory of this station's ``*.tle`` files -- the
            same directory :class:`~qsorbit.ui.picker_widget.PickerWidget`
            was given, so a satellite the picker names by NORAD id can
            be found and propagated again here.
        observer: This station's location -- the globe projection's
            own center, and where the station marker is drawn.
        now: Returns the current instant, timezone-aware. Injected for
            testing, matching :class:`~qsorbit.ui.picker_widget.PickerWidget`'s
            own convention.
        parent: The owning widget, or ``None``.
    """

    def __init__(
        self,
        *,
        themes: ThemeManager,
        tle_dir: str | Path,
        observer: ObserverLocation,
        now: Callable[[], datetime] = _utc_now,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._themes = themes
        self._tle_dir = tle_dir
        self._observer = observer
        self._now = now
        self._projection = Projection.FLAT
        self._coastlines = load_coastlines()
        self._entries: tuple[PickerEntry, ...] = ()
        self._satellites: dict[int, Satellite] = {}

        themes.changed.connect(self._on_theme_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        toggle_row = QHBoxLayout()
        toggle_row.addStretch(1)
        self._flat_chip = _make_toggle_chip("flat")
        self._globe_chip = _make_toggle_chip("globe")
        self._flat_chip.setChecked(True)
        self._toggle_group = QButtonGroup(self)
        self._toggle_group.setExclusive(True)
        self._toggle_group.addButton(self._flat_chip)
        self._toggle_group.addButton(self._globe_chip)
        self._flat_chip.toggled.connect(self._on_projection_toggled)
        toggle_row.addWidget(self._flat_chip)
        toggle_row.addWidget(self._globe_chip)
        layout.addLayout(toggle_row)

        self._canvas = _MapCanvas(self)
        layout.addWidget(self._canvas, 1)

    def set_visible_entries(self, entries: tuple[PickerEntry, ...]) -> None:
        """Redraw for exactly these picker entries -- the map's own feed.

        Args:
            entries: The satellites to draw a ground track and
                footprint for. Loads a fresh :class:`~qsorbit.core.
                tracker.satellite.Satellite` per entry from ``tle_dir``
                whenever the requested set of NORAD ids changes, and
                reuses what is already loaded otherwise -- toggling an
                unrelated filter chip (say, a mode group) re-renders
                without a new directory walk if the set of satellites
                it leaves visible happens not to change.
        """
        self._entries = entries
        norad_ids = frozenset(entry.profile.norad_id for entry in entries)
        if norad_ids != frozenset(self._satellites):
            self._satellites = load_satellites_by_norad_id(self._tle_dir, norad_ids)
        self._canvas.update()

    def _on_projection_toggled(self, flat_checked: bool) -> None:
        self._projection = Projection.FLAT if flat_checked else Projection.GLOBE
        self._canvas.update()

    def _on_theme_changed(self, _theme) -> None:
        self._canvas.update()


class _MapCanvas(QWidget):
    """The actual paintEvent, split from :class:`MapWidget` so its layout can hold
    the toggle row above this without hand-carving a reserved rect out of one
    shared ``paintEvent`` -- the same reason
    :class:`~qsorbit.ui.zoom_controls_widget.ZoomControlsWidget` and
    :class:`~qsorbit.ui.waterfall_widget.WaterfallWidget` are two widgets rather
    than one, just composed one level further in here because the toggle is
    this map's own state and nothing else's.

    **This canvas states its own size, and that is not decoration.**
    A ``QWidget`` that only implements ``paintEvent`` has no opinion
    about how big it needs to be, so ``minimumSizeHint()`` comes back
    empty and a layout is free to give it nothing at all. Inside a
    non-stretch :class:`~qsorbit.ui.cards.Card` -- whose size policy is
    ``(Preferred, Minimum)`` -- "nothing at all" is exactly what
    happened: the Plan tab rendered a titled card with a working
    flat/globe toggle above a map squeezed to a sliver, on a real
    screen, while twenty passing tests said nothing (they call
    ``grab()`` on the widget standalone, where the *test* supplies the
    size).

    Same defect class as the Rotor tab's clipped readouts fixed in #42:
    a widget's size is the widget's own business, and the container has
    no way to know it. So the numbers live here rather than in
    :class:`~qsorbit.ui.tabs.PlanTab`.
    """

    #: The map's preferred size, and the figures are not invented:
    #: ``qsorbit-shell-mockup.html`` draws this panel as
    #: ``viewBox="0 0 720 360"``, so the widget's stated size and the
    #: reference design agree by construction rather than by
    #: coincidence. 2:1 is also equirectangular's natural ratio.
    PREFERRED_SIZE: Final = QSize(720, 360)

    #: The floor. Small enough that a cramped window still lays out,
    #: large enough that a coastline is still recognisably a coastline
    #: -- the Natural Earth 110m data is trimmed to two decimal places,
    #: so below roughly this size its points start landing on the same
    #: pixel and the outline stops meaning anything.
    MINIMUM_SIZE: Final = QSize(360, 180)

    def __init__(self, owner: MapWidget) -> None:
        super().__init__(owner)
        self._owner = owner

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt's spelling
        """The size this canvas would like, absent other constraints."""
        return QSize(self.PREFERRED_SIZE)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt's spelling
        """The size below which this canvas stops being a map."""
        return QSize(self.MINIMUM_SIZE)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's spelling
        theme = self._owner._themes.current
        rect = QRectF(self.rect())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(rect, theme_color(theme, "inset"))

        projection = self._owner._projection
        observer = self._owner._observer
        scale, center_x, center_y = _fit_transform(rect, projection)

        if projection is Projection.GLOBE:
            painter.setPen(QPen(theme_color(theme, "edge"), 1))
            painter.drawEllipse(QPointF(center_x, center_y), scale, scale)
            center_lat, center_lon = observer.latitude, observer.longitude
        else:
            center_lat, center_lon = 0.0, 0.0

        self._paint_coastlines(
            painter, theme, projection, center_lat, center_lon, scale, center_x, center_y
        )
        self._paint_station_marker(painter, theme, projection, observer, scale, center_x, center_y)

        for index, entry in enumerate(self._owner._entries):
            satellite = self._owner._satellites.get(entry.profile.norad_id)
            if satellite is None:
                # Named by the picker (it matched a curated profile) but
                # its TLE didn't parse on this widget's own independent
                # load -- see load_satellites_by_norad_id's own "silently
                # absent" contract. Nothing to propagate, so nothing to
                # draw for this one entry; the rest of the map still
                # renders.
                continue
            color = accent_bar_color(theme, index)
            self._paint_satellite(
                painter,
                satellite,
                color,
                projection,
                center_lat,
                center_lon,
                scale,
                center_x,
                center_y,
            )

    def _paint_coastlines(
        self, painter, theme, projection, center_lat, center_lon, scale, center_x, center_y
    ) -> None:
        painter.setPen(QPen(theme_color(theme, "dim"), 1))
        for coastline in self._owner._coastlines:
            for segment in project_polyline(
                coastline,
                projection,
                center_latitude_deg=center_lat,
                center_longitude_deg=center_lon,
            ):
                painter.drawPath(_segment_path(segment, scale, center_x, center_y))

    def _paint_station_marker(
        self, painter, theme, projection, observer, scale, center_x, center_y
    ) -> None:
        point = _project_point(
            observer.latitude, observer.longitude, projection, observer.latitude, observer.longitude
        )
        if point is None:
            return
        screen = _to_screen(point, scale, center_x, center_y)
        painter.setPen(theme_color(theme, "text"))
        painter.drawText(screen, _STATION_GLYPH)

    def _paint_satellite(
        self,
        painter,
        satellite,
        color,
        projection,
        center_lat,
        center_lon,
        scale,
        center_x,
        center_y,
    ) -> None:
        now = self._owner._now()
        track = ground_track(satellite, now)
        track_points = tuple((point.latitude_deg, point.longitude_deg) for point in track)

        track_pen = QPen(color, _TRACK_WIDTH_PX)
        painter.setPen(track_pen)
        for segment in project_polyline(
            track_points,
            projection,
            center_latitude_deg=center_lat,
            center_longitude_deg=center_lon,
        ):
            painter.drawPath(_segment_path(segment, scale, center_x, center_y))

        current = satellite.subpoint_at(now)
        current_point = _project_point(
            current.latitude_deg, current.longitude_deg, projection, center_lat, center_lon
        )
        if current_point is None:
            # Currently on the far side of this station-centered globe
            # view -- expected for most entries most of the time; the
            # track above still shows where its own arc passes through
            # view, if any of it does.
            return
        screen = _to_screen(current_point, scale, center_x, center_y)
        # The marker dot is the only filled thing this widget draws, and
        # its brush is scoped rather than reset afterwards. A bare
        # setBrush(NoBrush) on the way out would work until somebody
        # adds an early return between here and there -- and the leak
        # this replaces happened by omission in the first place, so the
        # fix should not be another line that has to be remembered.
        #
        # What leaked: the brush stayed set through the footprint ring
        # below and through the *next* satellite's track on the following
        # iteration, so both were filled. Every track became a solid
        # blob, which is the opposite of this map's stated design -- a
        # ring is stroked precisely so that a segment split by the
        # antimeridian or the globe's limb degrades to the pieces that
        # are actually correct, instead of closing itself into a polygon
        # that was never there.
        painter.save()
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(screen, 3.0, 3.0)
        painter.setPen(color)
        painter.drawText(screen + QPointF(6.0, -4.0), satellite.name)
        painter.restore()

        footprint_deg = footprint_radius_deg(satellite.mean_altitude_km)
        ring = footprint_circle(current.latitude_deg, current.longitude_deg, footprint_deg)
        closed_ring = (*ring, ring[0])
        painter.setPen(QPen(color, _FOOTPRINT_WIDTH_PX, Qt.PenStyle.DashLine))
        for segment in project_polyline(
            closed_ring, projection, center_latitude_deg=center_lat, center_longitude_deg=center_lon
        ):
            painter.drawPath(_segment_path(segment, scale, center_x, center_y))
