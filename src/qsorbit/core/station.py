"""Station configuration — the things that don't change per satellite.

The dividing line, decided early and worth restating: *does this change
when I point at a different satellite?* If no, it belongs here — where
the antenna is, which serial port the rotator is on, how far that
rotator may turn. If yes, it belongs in :mod:`qsorbit.core.profiles`
with the frequencies, mode, and decoder settings.

Config is read from a TOML file with :mod:`tomllib` from the standard
library. Read-only is all Phase 1 needs, and it avoids a dependency.

**Where the file is looked for**, in order of precedence:

1. An explicit path, from ``--config`` on the command line.
2. ``qsorbit.toml`` in the current working directory — handy for
   keeping a per-site config beside a set of notes, or for testing an
   alternative rotator without touching the real one.
3. The platform's user config directory:

   * Windows: ``%APPDATA%\\qsorbit\\config.toml``
   * macOS: ``~/Library/Application Support/qsorbit/config.toml``
   * Linux and other Unix: ``$XDG_CONFIG_HOME/qsorbit/config.toml``,
     falling back to ``~/.config/qsorbit/config.toml``

Validation here is deliberately strict — missing keys and unknown keys
are both errors, and nothing is silently defaulted. A typo in
``azimuth_max_deg`` that quietly fell back to some built-in value would
be a travel limit the operator believes is in force and isn't, on
hardware whose firmware enforces nothing of its own. Better to refuse to
start.

This module depends on :mod:`qsorbit.core.rotor`,
:mod:`qsorbit.core.sdr` and :mod:`qsorbit.core.tracker`, never the other
way round. The rotor
controller takes a :class:`~qsorbit.core.rotor.RotorCapabilities`, not a
:class:`StationConfig` — so it stays usable without a config file, and
the dependency arrow only points one way.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qsorbit.core.horizon import HorizonMask, HorizonPoint
from qsorbit.core.rotor import AzimuthWrap, RotorCapabilities
from qsorbit.core.sdr import MAX_PPM
from qsorbit.core.tracker import ObserverLocation

#: Name of the config file looked for in the current working directory.
LOCAL_CONFIG_FILENAME = "qsorbit.toml"

#: Name of the config file inside the platform user config directory.
USER_CONFIG_FILENAME = "config.toml"

#: Directory name used inside the platform user config directory.
APP_DIR_NAME = "qsorbit"

#: Default baud rate. The SatNOGS firmware hard-codes ``#define BAUDRATE
#: 19200``, so this is not a guess — but it is still overridable, because
#: a modified build could differ.
DEFAULT_BAUDRATE = 19200

#: Default serial read timeout, in seconds. Distinct from the RS-485
#: turnaround in :class:`~qsorbit.core.rotor.RotorCapabilities`: this one
#: bounds how long a read blocks, that one paces the gap between writing
#: and reading.
DEFAULT_TIMEOUT_S = 1.0

#: Default RTL-SDR device index. Zero is the only device on a
#: single-dongle station, which is every station until Phase 3.
DEFAULT_DEVICE_INDEX = 0

#: Default crystal correction, in parts per million. Zero is an honest
#: default in a way a default gain would not be: an uncalibrated dongle
#: really is uncorrected, and the error it leaves is a few kHz at VHF,
#: which is visible rather than silent.
DEFAULT_PPM = 0


class ConfigError(Exception):
    """Raised when a station config file is missing, malformed, or incomplete.

    The message always names the file and the specific key at fault.
    Configuration errors are read by someone editing a text file by
    hand, often at a bench with cold fingers, so "unknown key
    ``azimuth_wrapping`` in ``[rotor.capabilities]``" beats a stack
    trace.
    """


@dataclass(frozen=True)
class SerialSettings:
    """How to reach the rotator's serial port.

    Args:
        port: Port name, e.g. ``"COM5"`` on Windows or
            ``"/dev/ttyUSB0"`` on Linux.
        baudrate: Baud rate. Defaults to :data:`DEFAULT_BAUDRATE`.
        timeout_s: Read timeout in seconds. Defaults to
            :data:`DEFAULT_TIMEOUT_S`.

    Raises:
        ValueError: If ``port`` is empty, or a numeric field is not
            positive.
    """

    port: str
    baudrate: int = DEFAULT_BAUDRATE
    timeout_s: float = DEFAULT_TIMEOUT_S

    def __post_init__(self) -> None:
        if not self.port:
            raise ValueError("Serial port name must not be empty.")
        if self.baudrate <= 0:
            raise ValueError(f"baudrate must be positive, got {self.baudrate}.")
        if self.timeout_s <= 0.0:
            raise ValueError(f"timeout_s must be positive, got {self.timeout_s}.")


@dataclass(frozen=True)
class SdrSettings:
    """How to reach this station's SDR.

    Everything here passes the config-boundary test — none of it changes
    when you point at a different satellite. Note what is therefore
    *absent*: centre frequency, sample rate and gain all vary per
    satellite and per band, so they belong with the profile that asks
    for them, not here. Gain in particular is a per-pass judgement (the
    bring-up used 32.8 dB for FM broadcast and 49.6 dB for NOAA weather
    radio on the same evening), and freezing it into station config
    would make it look like a property of the station.

    Args:
        driver_dir: Directory containing librtlsdr. Effectively required
            on Windows, where neither the working directory nor ``PATH``
            is searched for a DLL — point it at the ``x64`` folder of
            the RTL-SDR Blog driver release. Leave unset on Linux and
            macOS, where the package manager puts the library somewhere
            the loader already looks.
        device_index: Which device to open when more than one is
            attached.
        ppm: Crystal frequency correction in parts per million. A
            property of this particular dongle's oscillator, which is
            exactly why it lives with the station rather than the
            satellite.

    Raises:
        ValueError: If a value is outside what any device could use.
    """

    driver_dir: str | None = None
    device_index: int = DEFAULT_DEVICE_INDEX
    ppm: int = DEFAULT_PPM

    def __post_init__(self) -> None:
        if self.driver_dir is not None and not self.driver_dir.strip():
            raise ValueError("driver_dir must be a path or omitted entirely, not an empty string.")
        if self.device_index < 0:
            raise ValueError(f"device_index must not be negative, got {self.device_index}.")
        if abs(self.ppm) > MAX_PPM:
            raise ValueError(
                f"ppm must be within +/-{MAX_PPM}, got {self.ppm}. A real dongle is "
                "out by single or low double digits."
            )


@dataclass(frozen=True)
class AlignmentSettings:
    """This station's measured alignment offset, if one has been recorded.

    Storage only — see :class:`~qsorbit.core.pointing.AlignmentOffset`
    for the arithmetic it feeds and the sign convention it uses. This
    class exists separately, rather than reusing that one directly,
    because this module's own rule (see the module docstring) is that
    it depends on :mod:`qsorbit.core.rotor`, :mod:`qsorbit.core.sdr` and
    :mod:`qsorbit.core.tracker` and never the other way round — so a
    config value type lives here, and whoever needs the pointing-layer
    type builds one from these two floats.

    Args:
        azimuth_deg: How far the rotor's home azimuth sits clockwise of
            true north, in degrees. Defaults to 0.0 — no correction,
            i.e. uncalibrated, which is the honest default for a
            section most stations will not have measured yet.
        elevation_deg: The equivalent constant for elevation. Defaults
            to 0.0.
    """

    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0


@dataclass(frozen=True)
class PlanningSettings:
    """Where to find this station's TLEs, for the Plan tab's pass predictions.

    Passes the same config-boundary test every other section here does:
    a directory of TLEs is a property of the station, not of any one
    satellite, and it does not change when you point at a different
    bird -- it's the same directory whichever satellite you're
    searching for a pass on. Optional, same reasoning as ``[sdr]``:
    every config file written before Chunk D lacks this section, and
    "no TLE directory set" is the honest identity state -- the Plan
    tab shows a placeholder rather than guessing at a path.

    Args:
        tle_dir: Directory of ``*.tle`` files to search for upcoming
            passes, same format ``qsorbit plan --tle-dir`` already
            reads. ``None`` means unset.

    Raises:
        ValueError: If ``tle_dir`` is an empty string rather than
            omitted entirely.
    """

    tle_dir: str | None = None

    def __post_init__(self) -> None:
        if self.tle_dir is not None and not self.tle_dir.strip():
            raise ValueError("tle_dir must be a path or omitted entirely, not an empty string.")


@dataclass(frozen=True)
class StationConfig:
    """Everything QSOrbit needs to know about one ground station.

    Args:
        observer: Where the antenna is.
        serial: How to reach the rotator.
        capabilities: What that rotator may safely be commanded to do.
        sdr: How to reach the SDR. Defaults to a plain
            :class:`SdrSettings`, so a config file with no ``[sdr]``
            section stays valid — every station that predates Phase 2
            has one, and an SDR is not required to point an antenna.
        alignment: This station's measured alignment offset. Defaults
            to a plain :class:`AlignmentSettings` (0.0, 0.0), so a
            config file with no ``[rotor.alignment]`` section stays
            valid — every station that predates Chunk I has one, and
            "uncalibrated" is this offset's honest identity value, not
            a placeholder standing in for a required measurement.
        horizon: What this station's own site blocks. Defaults to an
            empty :class:`~qsorbit.core.horizon.HorizonMask` (no
            obstruction anywhere), so a config file with no
            ``[[horizon]]`` entries — every station that predates
            Chunk B — stays valid, and "nobody has measured this yet"
            is the honest identity state rather than an omission.
        planning: Where this station's TLEs live. Defaults to a plain
            :class:`PlanningSettings` (``tle_dir=None``), so a config
            file with no ``[planning]`` section stays valid — every
            station that predates Chunk D has one, and the Plan tab
            reads an unset directory as "not configured yet" rather
            than an error.
        source_path: The file this was loaded from, or ``None`` if it
            was constructed directly. Carried so error messages and
            ``status`` output can say which config is in force — with
            three possible locations, "which file am I actually using?"
            is a question that will come up.
    """

    observer: ObserverLocation
    serial: SerialSettings
    capabilities: RotorCapabilities
    sdr: SdrSettings = field(default_factory=SdrSettings)
    alignment: AlignmentSettings = field(default_factory=AlignmentSettings)
    horizon: HorizonMask = field(default_factory=HorizonMask)
    planning: PlanningSettings = field(default_factory=PlanningSettings)
    source_path: Path | None = None


def user_config_dir() -> Path:
    """Return the platform's per-user config directory for QSOrbit.

    Resolved from environment variables rather than a dependency on
    ``platformdirs``; Phase 1 only needs to find one file.

    Returns:
        The directory that would contain :data:`USER_CONFIG_FILENAME`.
        It is not created and may not exist.
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_DIR_NAME
        return Path.home() / "AppData" / "Roaming" / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / APP_DIR_NAME
    return Path.home() / ".config" / APP_DIR_NAME


def candidate_config_paths() -> list[Path]:
    """Return the paths searched for a config file, highest precedence first.

    Returns:
        ``[./qsorbit.toml, <user config dir>/config.toml]``. An explicit
        ``--config`` path is not part of this list — it bypasses the
        search entirely.
    """
    return [Path.cwd() / LOCAL_CONFIG_FILENAME, user_config_dir() / USER_CONFIG_FILENAME]


def find_config_path() -> Path | None:
    """Return the first config file that exists, or ``None`` if there is none.

    Returns:
        The highest-precedence existing path from
        :func:`candidate_config_paths`.
    """
    for path in candidate_config_paths():
        if path.is_file():
            return path
    return None


def load_station_config(path: str | Path | None = None) -> StationConfig:
    """Load and validate a station config file.

    Args:
        path: An explicit config path, typically from ``--config``. When
            ``None``, :func:`find_config_path` searches the standard
            locations.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If no config file is found, the file can't be read
            or parsed, a required key is missing, an unknown key is
            present, a value has the wrong type, or a value is rejected
            by the type it configures (an azimuth limit past 360° on a
            wrapping rotor, say).
    """
    if path is not None:
        resolved = Path(path)
        if not resolved.is_file():
            raise ConfigError(f"Config file not found: {resolved}")
    else:
        found = find_config_path()
        if found is None:
            searched = "\n  ".join(str(candidate) for candidate in candidate_config_paths())
            raise ConfigError(
                "No station config file found. Looked in:\n  "
                f"{searched}\n"
                "Copy config.example.toml to one of those locations and fill in "
                "your station's values, or pass --config with an explicit path."
            )
        resolved = found

    try:
        with resolved.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Could not parse {resolved}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read {resolved}: {exc}") from exc

    _reject_unknown_keys(
        data,
        {"observer", "rotor", "sdr", "horizon", "planning"},
        section="top level",
        path=resolved,
    )

    observer_table = _require_table(data, "observer", path=resolved)
    rotor_table = _require_table(data, "rotor", path=resolved)
    # Optional, unlike the two above: a station that only points an
    # antenna is a complete station, and every config file written
    # before Phase 2 lacks this section.
    sdr_table = _optional_table(data, "sdr", path=resolved)
    # Optional, same reasoning as [sdr]: every config file written
    # before Chunk D lacks this section, and "no TLE directory set" is
    # this feature's honest identity state.
    planning_table = _optional_table(data, "planning", path=resolved)
    capabilities_table = _require_table(rotor_table, "capabilities", path=resolved, parent="rotor")
    # Optional, same reasoning as [sdr]: every config file written
    # before Chunk I lacks this section, and "no offset recorded" is
    # this feature's honest identity state, not an omission.
    alignment_table = _optional_table(rotor_table, "alignment", path=resolved, parent="rotor")

    _reject_unknown_keys(
        observer_table,
        {"latitude", "longitude", "altitude_m"},
        section="observer",
        path=resolved,
    )
    _reject_unknown_keys(
        rotor_table,
        {"port", "baudrate", "timeout_s", "capabilities", "alignment"},
        section="rotor",
        path=resolved,
    )
    _reject_unknown_keys(
        alignment_table,
        {"azimuth_deg", "elevation_deg"},
        section="rotor.alignment",
        path=resolved,
    )
    _reject_unknown_keys(
        capabilities_table,
        {
            "azimuth_min_deg",
            "azimuth_max_deg",
            "elevation_min_deg",
            "elevation_max_deg",
            "azimuth_wrap",
            "acceptance_window_deg",
            "rs485_turnaround_s",
            "firmware_version",
        },
        section="rotor.capabilities",
        path=resolved,
    )
    _reject_unknown_keys(
        sdr_table,
        {"driver_dir", "device_index", "ppm"},
        section="sdr",
        path=resolved,
    )
    _reject_unknown_keys(
        planning_table,
        {"tle_dir"},
        section="planning",
        path=resolved,
    )
    horizon_points = _require_horizon_points(data, resolved)

    try:
        observer = ObserverLocation(
            latitude=_require_float(observer_table, "latitude", "observer", resolved),
            longitude=_require_float(observer_table, "longitude", "observer", resolved),
            altitude_m=_optional_float(observer_table, "altitude_m", "observer", resolved, 0.0),
        )
        serial_settings = SerialSettings(
            port=_require_str(rotor_table, "port", "rotor", resolved),
            baudrate=_optional_int(rotor_table, "baudrate", "rotor", resolved, DEFAULT_BAUDRATE),
            timeout_s=_optional_float(
                rotor_table, "timeout_s", "rotor", resolved, DEFAULT_TIMEOUT_S
            ),
        )
        capabilities = RotorCapabilities(
            azimuth_min_deg=_require_float(
                capabilities_table, "azimuth_min_deg", "rotor.capabilities", resolved
            ),
            azimuth_max_deg=_require_float(
                capabilities_table, "azimuth_max_deg", "rotor.capabilities", resolved
            ),
            elevation_min_deg=_require_float(
                capabilities_table, "elevation_min_deg", "rotor.capabilities", resolved
            ),
            elevation_max_deg=_require_float(
                capabilities_table, "elevation_max_deg", "rotor.capabilities", resolved
            ),
            azimuth_wrap=_require_azimuth_wrap(capabilities_table, resolved),
            acceptance_window_deg=_require_float(
                capabilities_table, "acceptance_window_deg", "rotor.capabilities", resolved
            ),
            rs485_turnaround_s=_require_float(
                capabilities_table, "rs485_turnaround_s", "rotor.capabilities", resolved
            ),
            firmware_version=_optional_str(
                capabilities_table, "firmware_version", "rotor.capabilities", resolved, None
            ),
        )
        sdr_settings = SdrSettings(
            driver_dir=_optional_str(sdr_table, "driver_dir", "sdr", resolved, None),
            device_index=_optional_int(
                sdr_table, "device_index", "sdr", resolved, DEFAULT_DEVICE_INDEX
            ),
            ppm=_optional_int(sdr_table, "ppm", "sdr", resolved, DEFAULT_PPM),
        )
        alignment_settings = AlignmentSettings(
            azimuth_deg=_optional_float(
                alignment_table, "azimuth_deg", "rotor.alignment", resolved, 0.0
            ),
            elevation_deg=_optional_float(
                alignment_table, "elevation_deg", "rotor.alignment", resolved, 0.0
            ),
        )
        planning_settings = PlanningSettings(
            tle_dir=_optional_str(planning_table, "tle_dir", "planning", resolved, None),
        )
        horizon = HorizonMask(points=horizon_points)
    except ValueError as exc:
        # The value objects do the real range checking; re-raised as a
        # ConfigError so the operator gets the file name alongside it.
        raise ConfigError(f"Invalid value in {resolved}: {exc}") from exc

    return StationConfig(
        observer=observer,
        serial=serial_settings,
        capabilities=capabilities,
        sdr=sdr_settings,
        alignment=alignment_settings,
        horizon=horizon,
        planning=planning_settings,
        source_path=resolved,
    )


# ---------------------------------------------------------------------------
# Table and key helpers
# ---------------------------------------------------------------------------


def _reject_unknown_keys(
    table: dict[str, Any], allowed: set[str], *, section: str, path: Path
) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(
            f"Unknown key{'s' if len(unknown) > 1 else ''} in [{section}] of {path}: "
            f"{', '.join(unknown)}. Valid keys: {', '.join(sorted(allowed))}. "
            "Nothing is silently ignored here — a misspelled travel limit would "
            "be a limit you believe is in force and isn't."
        )


def _require_table(
    data: dict[str, Any], key: str, *, path: Path, parent: str | None = None
) -> dict[str, Any]:
    section = f"{parent}.{key}" if parent else key
    if key not in data:
        raise ConfigError(f"Missing required section [{section}] in {path}.")
    value = data[key]
    if not isinstance(value, dict):
        raise ConfigError(f"[{section}] in {path} must be a table, got {type(value).__name__}.")
    return value


def _optional_table(
    data: dict[str, Any], key: str, *, path: Path, parent: str | None = None
) -> dict[str, Any]:
    """Return table ``key``, or an empty one if it is absent.

    An absent optional section and a present-but-empty one are the same
    thing here, so both come back as ``{}`` and every key inside falls
    to its default. A section that is present but is *not* a table is
    still an error — that is a typo, not an omission.

    Args:
        parent: For a nested section like ``[rotor.alignment]``, the
            enclosing table's name, so the error message names the
            section the way it appears in the file rather than just
            its last component. Mirrors :func:`_require_table`'s own
            parameter of the same name.
    """
    section = f"{parent}.{key}" if parent else key
    if key not in data:
        return {}
    value = data[key]
    if not isinstance(value, dict):
        raise ConfigError(f"[{section}] in {path} must be a table, got {type(value).__name__}.")
    return value


def _require(table: dict[str, Any], key: str, section: str, path: Path) -> Any:
    if key not in table:
        raise ConfigError(f"Missing required key '{key}' in [{section}] of {path}.")
    return table[key]


def _as_float(value: Any, key: str, section: str, path: Path) -> float:
    # bool is a subclass of int in Python, and `true` is not a number.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{key}' in [{section}] of {path} must be a number, got {value!r}.")
    return float(value)


def _require_float(table: dict[str, Any], key: str, section: str, path: Path) -> float:
    return _as_float(_require(table, key, section, path), key, section, path)


def _optional_float(
    table: dict[str, Any], key: str, section: str, path: Path, default: float
) -> float:
    if key not in table:
        return default
    return _as_float(table[key], key, section, path)


def _require_str(table: dict[str, Any], key: str, section: str, path: Path) -> str:
    value = _require(table, key, section, path)
    if not isinstance(value, str):
        raise ConfigError(f"'{key}' in [{section}] of {path} must be a string, got {value!r}.")
    return value


def _optional_str(
    table: dict[str, Any], key: str, section: str, path: Path, default: str | None
) -> str | None:
    if key not in table:
        return default
    return _require_str(table, key, section, path)


def _optional_int(table: dict[str, Any], key: str, section: str, path: Path, default: int) -> int:
    if key not in table:
        return default
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{key}' in [{section}] of {path} must be an integer, got {value!r}.")
    return value


def _require_azimuth_wrap(table: dict[str, Any], path: Path) -> AzimuthWrap:
    raw = _require_str(table, "azimuth_wrap", "rotor.capabilities", path)
    try:
        return AzimuthWrap(raw)
    except ValueError as exc:
        valid = ", ".join(f"'{member.value}'" for member in AzimuthWrap)
        raise ConfigError(
            f"'azimuth_wrap' in [rotor.capabilities] of {path} must be one of {valid}, "
            f"got '{raw}'. This one cannot be guessed: on a rotator that wraps, "
            "commanding 380 degrees means a full extra rotation against the cable, "
            "while on one with extended travel it means 20 degrees more travel."
        ) from exc


def _require_horizon_points(data: dict[str, Any], path: Path) -> tuple[HorizonPoint, ...]:
    """Parse the top-level ``[[horizon]]`` array of tables, if present.

    Not a table-keyed section like the rest of this file's helpers
    handle -- ``[[horizon]]`` is TOML's array-of-tables syntax, one
    entry per measured point, so this walks ``data["horizon"]``
    directly rather than going through :func:`_optional_table`.
    """
    raw = data.get("horizon", [])
    if not isinstance(raw, list):
        raise ConfigError(
            f"'horizon' in {path} must be an array of tables ('[[horizon]]' entries), "
            f"got {type(raw).__name__}."
        )
    points = []
    for index, entry in enumerate(raw):
        section = f"horizon[{index}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{section} in {path} must be a table, got {type(entry).__name__}.")
        _reject_unknown_keys(
            entry, {"azimuth_deg", "min_elevation_deg"}, section=section, path=path
        )
        azimuth_deg = _require_float(entry, "azimuth_deg", section, path)
        min_elevation_deg = _require_float(entry, "min_elevation_deg", section, path)
        try:
            points.append(
                HorizonPoint(azimuth_deg=azimuth_deg, min_elevation_deg=min_elevation_deg)
            )
        except ValueError as exc:
            raise ConfigError(f"Invalid value in {section} of {path}: {exc}") from exc
    return tuple(points)
