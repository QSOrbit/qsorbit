"""The target picker: a filterable shortlist of what's worth pointing at right now.

The Plan tab's centrepiece (Chunk D). Everything this widget shows comes
from :mod:`qsorbit.core.picker` (which satellites, filtered how) and
:mod:`qsorbit.ui.picker_formatting` (what a row and the staleness line
say) -- this module owns a table, a row of checkable filter chips, and a
refresh button, and does no judgment of its own beyond wiring those
together. Same split every other panel in this package follows.

**Recompute is manual, not polled.** Every other feed-backed widget here
(:class:`~qsorbit.ui.quieting_widget.QuietingWidget`,
:class:`~qsorbit.ui.frequency_widget.FrequencyWidget`) polls a plain
property on a timer because that read costs nothing.
:func:`~qsorbit.core.picker.build_picker_entries` does not: it walks
every ``*.tle`` file in the configured directory and runs a full SGP4
pass search per matched satellite, which is real CPU work on the GUI
thread. Phil's call (Chunk D PR2): a **Refresh** button, matching the
CLI's own explicit ``--refresh-catalogue`` rather than a background
timer that could stutter the shell on an interval nobody asked for.
Filtering, by contrast, is instant -- toggling a chip re-renders the
table from the entries already in hand without recomputing anything.

**The widget does not decide whether it exists.** Same convention
:class:`~qsorbit.ui.tabs.RadioTab` and
:class:`~qsorbit.ui.tabs.RotorTab` already follow for a missing feed:
whether a TLE directory is configured at all is
:class:`~qsorbit.ui.tabs.PlanTab`'s call, made once at construction by
choosing between this widget and a
:class:`~qsorbit.ui.cards.Placeholder`. This widget only handles the
narrower case of a *configured* directory that turns out to be missing
or empty when refreshed -- see :meth:`PickerWidget.refresh`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, tzinfo
from pathlib import Path
from typing import Final

from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qsorbit.core.horizon import HorizonMask
from qsorbit.core.picker import (
    DEFAULT_LOOKAHEAD_HOURS,
    Band,
    ModeGroup,
    PickerEntry,
    PickerFilters,
    build_picker_entries,
    passes_filters,
)
from qsorbit.core.profiles import CatalogManifest, ProfileCatalog, ReliabilityClass
from qsorbit.core.tracker import ObserverLocation
from qsorbit.ui.picker_formatting import catalogue_staleness_text, format_band, picker_row_text

#: Column headers, in table order. The leading blank header is the
#: status-dot column -- a dot needs no label, the mockup gives it none
#: either.
_COLUMN_HEADERS: Final = (
    "",
    "satellite",
    "next pass (local)",
    "max el",
    "downlink",
    "mode",
    "tier",
)

#: The status dot's glyph. A filled circle in a QLabel wearing an
#: existing ``role`` (``"ok"``/``"warn"``/``"dim"``) rather than a
#: coloured swatch widget -- reuses the selectors
#: :mod:`qsorbit.ui.theme_qss` already emits for
#: :func:`~qsorbit.ui.picker_formatting.alive_status_role`, so this
#: widget needed no new QSS of its own for the dot.
_STATUS_DOT: Final = "●"


def _utc_now() -> datetime:
    """The current instant, timezone-aware. Injected so tests can fake it."""
    return datetime.now(UTC)


def _make_chip(label: str) -> QPushButton:
    """A checkable filter-chip button, unchecked (no restriction) by default."""
    chip = QPushButton(label)
    chip.setCheckable(True)
    return chip


class PickerWidget(QWidget):
    """A refreshable, filterable table of curated satellites and their next pass.

    Args:
        catalog: The curated profile catalogue to match TLEs against.
        manifest: The catalogue's optional shipped-date manifest, for
            the staleness line -- ``None`` if the catalogue directory
            carries none.
        tle_dir: Directory of this station's ``*.tle`` files. Assumed
            configured -- a station with none should never reach this
            widget at all; see the module docstring.
        observer: This station's location.
        horizon: This station's own horizon mask, applied to every pass
            search exactly as ``qsorbit plan`` applies it.
        hours: How far ahead each refresh looks. Defaults to
            :data:`~qsorbit.core.picker.DEFAULT_LOOKAHEAD_HOURS`.
        local_zone: The zone pass times are shown in. Defaults to the
            system's configured zone -- see
            :func:`~qsorbit.ui.picker_formatting.picker_row_text`'s
            identical parameter.
        now: Returns the current instant, timezone-aware. Injected for
            testing, matching
            :class:`~qsorbit.core.pointing.TrackingLoop`'s own
            convention.
    """

    def __init__(
        self,
        catalog: ProfileCatalog,
        manifest: CatalogManifest | None,
        tle_dir: str | Path,
        observer: ObserverLocation,
        horizon: HorizonMask,
        *,
        hours: float = DEFAULT_LOOKAHEAD_HOURS,
        local_zone: tzinfo | None = None,
        now: Callable[[], datetime] = _utc_now,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._catalog = catalog
        self._manifest = manifest
        self._tle_dir = tle_dir
        self._observer = observer
        self._horizon = horizon
        self._hours = hours
        self._local_zone = local_zone
        self._now = now
        self._entries: tuple[PickerEntry, ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        self._status_label = QLabel("")
        self._status_label.setProperty("role", "dim")
        top_row.addWidget(self._status_label)
        top_row.addStretch(1)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh)
        top_row.addWidget(refresh_button)
        layout.addLayout(top_row)

        chip_row = QHBoxLayout()
        self._needs_transmitter_chip = _make_chip("has transmitter")
        chip_row.addWidget(self._needs_transmitter_chip)

        self._band_chips: dict[Band, QPushButton] = {
            band: _make_chip(format_band(band)) for band in (Band.SEVENTY_CM, Band.TWO_METERS)
        }
        self._mode_chips: dict[ModeGroup, QPushButton] = {
            ModeGroup.FM: _make_chip("FM"),
            ModeGroup.SSB_CW: _make_chip("SSB/CW"),
            ModeGroup.DIGITAL: _make_chip("digital"),
        }
        self._reliability_chips: dict[ReliabilityClass, QPushButton] = {
            ReliabilityClass.UNCONDITIONAL: _make_chip("reliability A"),
            ReliabilityClass.SCHEDULED: _make_chip("B"),
            ReliabilityClass.DEPENDENT: _make_chip("C"),
        }
        all_chips = (
            (self._needs_transmitter_chip,)
            + tuple(self._band_chips.values())
            + tuple(self._mode_chips.values())
            + tuple(self._reliability_chips.values())
        )
        for chip in all_chips[1:]:
            chip_row.addWidget(chip)
        for chip in all_chips:
            chip.toggled.connect(self._on_filters_changed)
        chip_row.addStretch(1)
        layout.addLayout(chip_row)

        self._table = QTableWidget(0, len(_COLUMN_HEADERS))
        self._table.setHorizontalHeaderLabels(list(_COLUMN_HEADERS))
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table, 1)

        self.refresh()

    def refresh(self) -> None:
        """Recompute this station's picker entries and re-render the table.

        A missing or empty TLE directory is not an error -- it shows
        plainly in the status line and an empty table, the same
        "off, not broken" honesty
        :class:`~qsorbit.ui.cards.Placeholder` gives a missing feed --
        rather than raising out of a button click.
        """
        if not Path(self._tle_dir).is_dir():
            self._entries = ()
            self._status_label.setText(f"TLE directory not found: {self._tle_dir}")
            self._status_label.setProperty("role", "warn")
            self._restyle(self._status_label)
            self._render_table()
            return

        today: date = self._now().date()
        self._entries = build_picker_entries(
            self._catalog,
            self._tle_dir,
            self._observer,
            self._horizon,
            self._now(),
            hours=self._hours,
        )
        staleness = catalogue_staleness_text(self._manifest, today)
        self._status_label.setText(staleness or "")
        self._status_label.setProperty("role", "dim")
        self._restyle(self._status_label)
        self._render_table()

    def _on_filters_changed(self, _checked: bool) -> None:
        self._render_table()

    def _collect_filters(self) -> PickerFilters:
        return PickerFilters(
            needs_transmitter=self._needs_transmitter_chip.isChecked(),
            bands=frozenset(band for band, chip in self._band_chips.items() if chip.isChecked()),
            mode_groups=frozenset(
                group for group, chip in self._mode_chips.items() if chip.isChecked()
            ),
            reliability_classes=frozenset(
                rc for rc, chip in self._reliability_chips.items() if chip.isChecked()
            ),
        )

    def _render_table(self) -> None:
        filters = self._collect_filters()
        visible = [entry for entry in self._entries if passes_filters(entry.profile, filters)]

        self._table.setRowCount(len(visible))
        for row, entry in enumerate(visible):
            text = picker_row_text(entry, local_zone=self._local_zone)

            dot = QLabel(_STATUS_DOT)
            dot.setProperty("role", text.status_role)
            self._table.setCellWidget(row, 0, dot)

            cells = (
                text.name,
                text.pass_text,
                text.max_elevation_text,
                text.downlink_text,
                text.mode_text,
                text.tier_text,
            )
            for column, value in enumerate(cells, start=1):
                self._table.setItem(row, column, QTableWidgetItem(value))

    @staticmethod
    def _restyle(widget: QWidget) -> None:
        """Force the stylesheet to re-evaluate ``widget``'s ``role`` property.

        Matches :class:`~qsorbit.ui.frequency_widget.FrequencyWidget`'s
        own comment: Qt only re-applies a property selector when told
        to, and this widget's status label changes ``role`` between
        ``"dim"`` (a normal staleness line) and ``"warn"`` (a bad TLE
        directory) after its first show.
        """
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
