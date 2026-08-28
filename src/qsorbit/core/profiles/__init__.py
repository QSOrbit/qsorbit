"""Per-satellite profiles: what a satellite transmits, and how alive it is.

See :mod:`qsorbit.core.profiles.profile` for the data model and the
reasoning behind it, and :mod:`qsorbit.core.profiles.catalog` for
loading the curated starter set this package ships in ``data/``.
"""

from qsorbit.core.profiles.catalog import (
    DEFAULT_PROFILES_DIR,
    ProfileCatalog,
    ProfileError,
    load_profile_catalog,
)
from qsorbit.core.profiles.profile import (
    AliveRecord,
    AliveStatus,
    Mode,
    ReliabilityClass,
    SatelliteProfile,
    Transmitter,
)

__all__ = [
    "DEFAULT_PROFILES_DIR",
    "AliveRecord",
    "AliveStatus",
    "Mode",
    "ProfileCatalog",
    "ProfileError",
    "ReliabilityClass",
    "SatelliteProfile",
    "Transmitter",
    "load_profile_catalog",
]
