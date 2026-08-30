"""Custom tab configuration: a widget list from a file, not from code.

**Custom tab v1, per the roadmap**: a config-file-defined grid with
persisted layout, no in-app editing. "Persisted" turns out to mean
nothing more than "it's a file" -- there is no drag-and-drop state to
save, so the whole feature is *load one small TOML, build a grid from
it*. That is deliberately the same shape as :mod:`qsorbit.ui.theme`:
one file, strict validation, a name-the-file-and-the-key error message
for whoever is editing it by hand.

**One real difference from every other config file this project
loads.** Station config refuses to start the application at all on any
error -- a misread travel limit is a safety issue. A theme file is
looked up from a whole directory, and a bad one is silently skipped so
the rest of the catalogue still loads. This file is neither: there is
exactly one of it, and getting it wrong should cost you the Custom tab,
not the whole shell. :func:`load_custom_tab_config` raises on any
problem, same as :func:`~qsorbit.ui.theme.load_theme` -- the caller
(:mod:`qsorbit.__main__`) is the one that decides a raised
:class:`CustomTabConfigError` becomes a placeholder in one tab rather
than a refusal to open, the same "off and broken must never look the
same" rule this project applies everywhere, aimed at a file that is
entirely optional in the first place.

**Where the file lives.** Beside the themes this build ships with, not
wherever ``config.toml`` was actually found: :func:`custom_tab_config_path`
resolves through :func:`~qsorbit.core.station.user_config_dir` directly,
exactly the way :func:`~qsorbit.ui.theme.app_themes_dir` does. A station
config loaded from ``./qsorbit.toml`` for a one-off test does not also
relocate the Custom tab's layout.

**Widget names are a closed set, checked at load time.** ``next_pass``
is in the mockup's own sample and does not exist yet -- it needs Chunk
D's pass prediction wired into the shell -- so it is not offered here.
:data:`KNOWN_WIDGETS` is exactly the five widgets the built-in tabs
already build from a hub feed today. A name outside that set is a load
error naming the file, not a cell that silently goes missing and gets
discovered by counting.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

#: The widget kinds v1 knows how to build in a Custom-tab grid cell --
#: exactly the widgets :mod:`qsorbit.ui.tabs`'s built-in tabs already
#: build from a hub feed. Audio-device selection and ``next_pass`` are
#: not in this set because neither is wired into the shell yet.
KNOWN_WIDGETS: Final[frozenset[str]] = frozenset(
    {"waterfall", "spectrum_line", "quieting", "frequency", "rotor_readout"}
)

#: The only layout v1 builds. A single-member set rather than a bare
#: string check so a second layout (rows? free-form?) has an obvious
#: place to be added later without restructuring the error message.
KNOWN_LAYOUTS: Final[frozenset[str]] = frozenset({"grid"})

#: The file's name, beside ``config.toml`` and the ``themes/`` directory
#: in the same per-user config directory. See :func:`custom_tab_config_path`.
CONFIG_FILENAME: Final = "custom_tab.toml"


class CustomTabConfigError(Exception):
    """Raised when ``custom_tab.toml`` is present but malformed.

    Unlike :class:`~qsorbit.core.station.ConfigError`, raising this
    never stops the application from opening -- see the module
    docstring. The message still names the file and the key at fault,
    for the same reason station config's errors do: this is read by
    someone editing a text file by hand.
    """


@dataclass(frozen=True)
class CustomTabConfig:
    """A validated ``custom_tab.toml``: a grid of named widgets.

    Args:
        columns: How many widgets wide the grid is. Widgets fill
            row-major -- the first ``columns`` entries of
            :attr:`widgets` are the top row, and so on.
        widgets: One widget kind per cell, in fill order. Every entry
            is a member of :data:`KNOWN_WIDGETS` -- checked once here,
            at load time, rather than left for whatever builds the grid
            to discover cell by cell.
    """

    columns: int
    widgets: tuple[str, ...]


def custom_tab_config_path() -> Path:
    """Where ``custom_tab.toml`` is looked for.

    Always :func:`~qsorbit.core.station.user_config_dir` plus
    :data:`CONFIG_FILENAME` -- there is no ``--custom-tab-config`` flag
    and no current-working-directory fallback, unlike station config's
    three-way search. One file, one place, matching how
    :func:`~qsorbit.ui.theme.app_themes_dir` resolves the shared
    ``themes/`` directory rather than following ``--config``.

    Returns:
        The path this build will look for. Not created and may not
        exist -- an absent file is the normal "you haven't written one
        yet" case, not an error.
    """
    from qsorbit.core.station import user_config_dir

    return user_config_dir() / CONFIG_FILENAME


def load_custom_tab_config(path: str | Path) -> CustomTabConfig:
    """Load and validate one ``custom_tab.toml``.

    Args:
        path: The file to read.

    Returns:
        The validated configuration.

    Raises:
        CustomTabConfigError: If the file is missing, is not valid
            TOML, has a missing or unknown key, ``layout`` is not
            ``"grid"``, ``columns`` is not a positive integer,
            ``widgets`` is empty, or any entry in ``widgets`` is not a
            string naming a member of :data:`KNOWN_WIDGETS`.
    """
    resolved = Path(path)
    try:
        with resolved.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as error:
        raise CustomTabConfigError(f"No custom tab config at {resolved}.") from error
    except tomllib.TOMLDecodeError as error:
        raise CustomTabConfigError(f"{resolved} is not valid TOML: {error}") from error
    except OSError as error:  # pragma: no cover - permissions, a directory, a bad mount
        raise CustomTabConfigError(f"Could not read {resolved}: {error}") from error

    _reject_unknown_keys(data, {"layout", "columns", "widgets"}, "top level", resolved)

    layout = _require(data, "layout", "top level", resolved)
    if layout not in KNOWN_LAYOUTS:
        valid = ", ".join(f"'{name}'" for name in sorted(KNOWN_LAYOUTS))
        raise CustomTabConfigError(
            f"'layout' in {resolved} must be one of {valid} -- v1 builds nothing "
            f"else -- got {layout!r}."
        )

    columns = _require(data, "columns", "top level", resolved)
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise CustomTabConfigError(
            f"'columns' in {resolved} must be a positive integer, got {columns!r}."
        )

    raw_widgets = _require(data, "widgets", "top level", resolved)
    if not isinstance(raw_widgets, list) or not raw_widgets:
        raise CustomTabConfigError(
            f"'widgets' in {resolved} must be a non-empty array of widget names."
        )

    widgets: list[str] = []
    valid_names = ", ".join(f"'{name}'" for name in sorted(KNOWN_WIDGETS))
    for index, name in enumerate(raw_widgets):
        if not isinstance(name, str):
            raise CustomTabConfigError(
                f"widgets[{index}] in {resolved} must be a string, got {name!r}."
            )
        if name not in KNOWN_WIDGETS:
            raise CustomTabConfigError(
                f"widgets[{index}] in {resolved} names {name!r}, which this build "
                f"does not know how to draw. Known widgets: {valid_names}."
            )
        widgets.append(name)

    return CustomTabConfig(columns=columns, widgets=tuple(widgets))


# ---------------------------------------------------------------------------
# Validation helpers
#
# Duplicated from qsorbit.core.station and qsorbit.ui.theme rather than
# shared -- both of those modules made the same call independently, and
# a third copy keeps this module importable without either of them.
# ---------------------------------------------------------------------------


def _reject_unknown_keys(
    table: dict[str, Any], allowed: set[str], section: str, path: Path
) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise CustomTabConfigError(
            f"Unknown key{'s' if len(unknown) > 1 else ''} in [{section}] of {path}: "
            f"{', '.join(unknown)}. Valid keys: {', '.join(sorted(allowed))}."
        )


def _require(table: dict[str, Any], key: str, section: str, path: Path) -> Any:
    if key not in table:
        raise CustomTabConfigError(f"Missing required key '{key}' in [{section}] of {path}.")
    return table[key]
