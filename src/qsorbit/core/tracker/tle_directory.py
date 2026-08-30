"""Loading Satellite objects from a directory of TLE files, keyed by NORAD id.

:func:`~qsorbit.core.picker.build_picker_entries` already walks a TLE
directory this same way -- glob ``*.tle``, parse each, skip anything
that fails -- but it keeps only the derived
:class:`~qsorbit.core.picker.PickerEntry` once it has computed
``next_pass`` and ``visible_from_latitude``, discarding the propagated
:class:`~qsorbit.core.tracker.satellite.Satellite` itself. The map
needs the opposite: the ``Satellite`` objects themselves, to keep
propagating them for a ground track and a footprint. Rather than
changing ``build_picker_entries()``'s own already-shipped return
contract to also hand those back, this is a second, independent walk
over the same directory -- one more directory scan on a refresh the map
draws far less often than the picker recomputes passes, not a cost
worth reworking tested code to avoid.
"""

from __future__ import annotations

from pathlib import Path

from qsorbit.core.tracker.exceptions import TrackerError
from qsorbit.core.tracker.satellite import Satellite


def load_satellites_by_norad_id(
    tle_dir: str | Path, norad_ids: frozenset[int]
) -> dict[int, Satellite]:
    """Load every TLE in ``tle_dir`` whose NORAD id is in ``norad_ids``.

    Args:
        tle_dir: Directory of ``*.tle`` files.
        norad_ids: Which satellites to keep. An empty set returns an
            empty dict without walking the directory's contents.

    Returns:
        A dict keyed by NORAD id, one entry per requested id that was
        actually found and successfully parsed. A requested id with no
        matching TLE, or a TLE that fails to parse, is silently absent
        -- the same "an unmatched or unparseable TLE is the expected
        case" reasoning :func:`~qsorbit.core.picker.build_picker_entries`
        uses for the same walk.
    """
    if not norad_ids:
        return {}
    satellites: dict[int, Satellite] = {}
    for tle_path in sorted(Path(tle_dir).glob("*.tle")):
        try:
            satellite = Satellite.from_file(tle_path)
        except TrackerError:
            continue
        if satellite.norad_id in norad_ids:
            satellites[satellite.norad_id] = satellite
    return satellites
