"""Loading per-satellite profiles from TOML data files.

Mirrors :mod:`qsorbit.core.station`'s TOML-and-``tomllib`` approach and
its strict validation -- missing keys and unknown keys are both errors,
nothing is silently defaulted. A misspelled ``reliablity`` typo that
quietly fell back to some default would be exactly the kind of error
this project has already been burned by elsewhere, applied to data that
decides what a shortlist tells someone to point at.

One file per satellite, named for readability rather than parsed for
meaning -- ``rs44.toml``, not ``44909.toml`` -- with ``norad_id`` as the
key inside the file that actually matters.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from qsorbit.core.profiles.profile import (
    AliveRecord,
    AliveStatus,
    Mode,
    ReliabilityClass,
    SatelliteProfile,
    Transmitter,
)

#: The curated starter set shipped with QSOrbit.
DEFAULT_PROFILES_DIR = Path(__file__).parent / "data"

#: A catalogue-level manifest, optional, living beside the profile files
#: it describes. Reserved -- load_profile_catalog() skips it rather than
#: trying to parse it as a profile.
CATALOG_MANIFEST_FILENAME = "CATALOG.toml"


class ProfileError(Exception):
    """Raised when a profile file is missing, malformed, or incomplete.

    The message always names the file and the specific key at fault,
    for the same reason :class:`~qsorbit.core.station.ConfigError`'s
    docstring gives: read by someone editing a text file by hand.
    """


class ProfileCatalog:
    """A loaded set of satellite profiles, indexed by NORAD catalog number.

    Args:
        profiles: The profiles to index.

    Raises:
        ValueError: If two profiles share a ``norad_id`` -- the catalog
            number is this catalog's key, and a collision almost
            certainly means one profile's file has the wrong number in
            it rather than that two different satellites really share
            an ID.
    """

    def __init__(self, profiles: Iterable[SatelliteProfile]) -> None:
        self._by_norad_id: dict[int, SatelliteProfile] = {}
        for profile in profiles:
            existing = self._by_norad_id.get(profile.norad_id)
            if existing is not None:
                raise ValueError(
                    f"duplicate norad_id {profile.norad_id}: {existing.name!r} and "
                    f"{profile.name!r} both claim it."
                )
            self._by_norad_id[profile.norad_id] = profile

    def by_norad_id(self, norad_id: int) -> SatelliteProfile | None:
        """The profile for ``norad_id``, or ``None`` if this catalog has none."""
        return self._by_norad_id.get(norad_id)

    def __len__(self) -> int:
        return len(self._by_norad_id)

    def __iter__(self) -> Iterator[SatelliteProfile]:
        return iter(self._by_norad_id.values())


def load_profile_catalog(directory: str | Path = DEFAULT_PROFILES_DIR) -> ProfileCatalog:
    """Load every ``*.toml`` profile in ``directory``.

    Args:
        directory: Where to look. Defaults to :data:`DEFAULT_PROFILES_DIR`,
            the curated starter set shipped with QSOrbit.

    Returns:
        The loaded catalog.

    Raises:
        ProfileError: If ``directory`` doesn't exist, or any file in it
            can't be read, can't be parsed, or fails validation.
        ValueError: If two profiles in ``directory`` share a
            ``norad_id``.
    """
    resolved = Path(directory)
    if not resolved.is_dir():
        raise ProfileError(f"Profile directory not found: {resolved}")
    profiles = [
        _load_profile_file(path)
        for path in sorted(resolved.glob("*.toml"))
        if path.name != CATALOG_MANIFEST_FILENAME
    ]
    return ProfileCatalog(profiles)


@dataclass(frozen=True)
class CatalogManifest:
    """Catalogue-level metadata: when this directory's curated profile set was last revised.

    Distinct from any single profile's ``alive.as_of`` -- that is when
    one satellite's status was last checked; this is when the *set
    itself* (which satellites are curated, at all) was last revised.
    Optional: a directory of profiles with no ``CATALOG.toml`` simply
    has no catalogue-level staleness to report, which is why
    :func:`load_catalog_manifest` returns ``None`` rather than raising
    when the file is simply absent -- an older or hand-rolled
    ``--profiles-dir`` predating this feature is the expected case, not
    an error.

    Args:
        shipped: The date this profile set was last revised.
    """

    shipped: date


def load_catalog_manifest(directory: str | Path = DEFAULT_PROFILES_DIR) -> CatalogManifest | None:
    """Load ``directory``'s catalogue manifest, if it has one.

    Args:
        directory: Where to look. Defaults to :data:`DEFAULT_PROFILES_DIR`,
            the curated starter set shipped with QSOrbit.

    Returns:
        The loaded manifest, or ``None`` if ``directory`` has no
        :data:`CATALOG_MANIFEST_FILENAME` -- missing is fine, same
        "no file yet" tolerance as ``custom_tab.toml``'s loader.

    Raises:
        ProfileError: If the manifest file exists but can't be read,
            can't be parsed, or fails validation.
    """
    path = Path(directory) / CATALOG_MANIFEST_FILENAME
    if not path.is_file():
        return None

    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ProfileError(f"Could not parse {path}: {exc}") from exc
    except OSError as exc:
        raise ProfileError(f"Could not read {path}: {exc}") from exc

    _reject_unknown_keys(data, {"shipped"}, section="top level", path=path)
    shipped = _require(data, "shipped", "top level", path)
    if not isinstance(shipped, date):
        raise ProfileError(
            f"'shipped' in {path} must be a TOML date (e.g. 2026-08-28), got {shipped!r}."
        )

    return CatalogManifest(shipped=shipped)


def _load_profile_file(path: Path) -> SatelliteProfile:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ProfileError(f"Could not parse {path}: {exc}") from exc
    except OSError as exc:
        raise ProfileError(f"Could not read {path}: {exc}") from exc

    _reject_unknown_keys(
        data,
        {"norad_id", "name", "also_known_as", "alive", "transmitters"},
        section="top level",
        path=path,
    )

    norad_id_raw = _require(data, "norad_id", "top level", path)
    if isinstance(norad_id_raw, bool) or not isinstance(norad_id_raw, int):
        raise ProfileError(f"'norad_id' in {path} must be an integer, got {norad_id_raw!r}.")

    name = _require(data, "name", "top level", path)
    if not isinstance(name, str):
        raise ProfileError(f"'name' in {path} must be a string, got {name!r}.")

    also_known_as_raw = data.get("also_known_as", [])
    if not isinstance(also_known_as_raw, list) or not all(
        isinstance(item, str) for item in also_known_as_raw
    ):
        raise ProfileError(f"'also_known_as' in {path} must be an array of strings.")

    alive_table = _require(data, "alive", "top level", path)
    if not isinstance(alive_table, dict):
        raise ProfileError(f"[alive] in {path} must be a table, got {type(alive_table).__name__}.")
    alive = _load_alive(alive_table, path)

    transmitters_raw = data.get("transmitters", [])
    if not isinstance(transmitters_raw, list):
        raise ProfileError(f"'transmitters' in {path} must be an array of tables.")
    transmitters = tuple(
        _load_transmitter(table, index, path) for index, table in enumerate(transmitters_raw)
    )

    try:
        return SatelliteProfile(
            norad_id=norad_id_raw,
            name=name,
            transmitters=transmitters,
            alive=alive,
            also_known_as=tuple(also_known_as_raw),
        )
    except ValueError as exc:
        raise ProfileError(f"Invalid value in {path}: {exc}") from exc


def _load_alive(table: dict[str, Any], path: Path) -> AliveRecord:
    _reject_unknown_keys(table, {"status", "as_of", "source"}, section="alive", path=path)

    status_raw = _require(table, "status", "alive", path)
    if not isinstance(status_raw, str):
        raise ProfileError(f"'status' in [alive] of {path} must be a string, got {status_raw!r}.")
    try:
        status = AliveStatus(status_raw)
    except ValueError as exc:
        valid = ", ".join(f"'{member.value}'" for member in AliveStatus)
        raise ProfileError(
            f"'status' in [alive] of {path} must be one of {valid}, got '{status_raw}'."
        ) from exc

    as_of = _require(table, "as_of", "alive", path)
    if not isinstance(as_of, date):
        raise ProfileError(
            f"'as_of' in [alive] of {path} must be a TOML date (e.g. 2026-08-25), got {as_of!r}."
        )

    source = _require(table, "source", "alive", path)
    if not isinstance(source, str):
        raise ProfileError(f"'source' in [alive] of {path} must be a string, got {source!r}.")

    try:
        return AliveRecord(status=status, as_of=as_of, source=source)
    except ValueError as exc:
        raise ProfileError(f"Invalid value in [alive] of {path}: {exc}") from exc


def _load_transmitter(table: Any, index: int, path: Path) -> Transmitter:
    section = f"transmitters[{index}]"
    if not isinstance(table, dict):
        raise ProfileError(f"{section} in {path} must be a table, got {type(table).__name__}.")
    _reject_unknown_keys(
        table,
        {"downlink_hz", "mode", "reliability", "uplink_hz", "baud", "notes"},
        section=section,
        path=path,
    )

    downlink_hz = _require_number(table, "downlink_hz", section, path)

    mode_raw = _require(table, "mode", section, path)
    if not isinstance(mode_raw, str):
        raise ProfileError(f"'mode' in {section} of {path} must be a string, got {mode_raw!r}.")
    try:
        mode = Mode(mode_raw)
    except ValueError as exc:
        valid = ", ".join(f"'{member.value}'" for member in Mode)
        raise ProfileError(
            f"'mode' in {section} of {path} must be one of {valid}, got '{mode_raw}'."
        ) from exc

    reliability_raw = _require(table, "reliability", section, path)
    if not isinstance(reliability_raw, str):
        raise ProfileError(
            f"'reliability' in {section} of {path} must be a string, got {reliability_raw!r}."
        )
    try:
        reliability = ReliabilityClass(reliability_raw)
    except ValueError as exc:
        valid = ", ".join(f"'{member.value}'" for member in ReliabilityClass)
        raise ProfileError(
            f"'reliability' in {section} of {path} must be one of {valid}, got '{reliability_raw}'."
        ) from exc

    uplink_hz_raw = table.get("uplink_hz")
    uplink_hz = (
        _as_number(uplink_hz_raw, "uplink_hz", section, path) if uplink_hz_raw is not None else None
    )
    baud_raw = table.get("baud")
    baud = _as_number(baud_raw, "baud", section, path) if baud_raw is not None else None

    notes = table.get("notes", "")
    if not isinstance(notes, str):
        raise ProfileError(f"'notes' in {section} of {path} must be a string, got {notes!r}.")

    try:
        return Transmitter(
            downlink_hz=downlink_hz,
            mode=mode,
            reliability=reliability,
            uplink_hz=uplink_hz,
            baud=baud,
            notes=notes,
        )
    except ValueError as exc:
        raise ProfileError(f"Invalid value in {section} of {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Table and key helpers -- deliberately parallel to qsorbit.core.station's
# ---------------------------------------------------------------------------


def _reject_unknown_keys(
    table: dict[str, Any], allowed: set[str], *, section: str, path: Path
) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ProfileError(
            f"Unknown key{'s' if len(unknown) > 1 else ''} in [{section}] of {path}: "
            f"{', '.join(unknown)}. Valid keys: {', '.join(sorted(allowed))}."
        )


def _require(table: dict[str, Any], key: str, section: str, path: Path) -> Any:
    if key not in table:
        raise ProfileError(f"Missing required key '{key}' in [{section}] of {path}.")
    return table[key]


def _as_number(value: Any, key: str, section: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"'{key}' in [{section}] of {path} must be a number, got {value!r}.")
    return float(value)


def _require_number(table: dict[str, Any], key: str, section: str, path: Path) -> float:
    return _as_number(_require(table, key, section, path), key, section, path)
