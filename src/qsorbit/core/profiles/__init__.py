"""Per-satellite profiles: what a satellite transmits, and how alive it is.

See :mod:`qsorbit.core.profiles.profile` for the data model and the
reasoning behind it, :mod:`qsorbit.core.profiles.catalog` for loading
the curated starter set this package ships in ``data/`` (plus its
optional catalogue-level manifest), and
:mod:`qsorbit.core.profiles.catalog_source` for the (currently
unconfigured) network-refresh plumbing.
"""

from qsorbit.core.profiles.catalog import (
    CATALOG_MANIFEST_FILENAME,
    DEFAULT_PROFILES_DIR,
    CatalogManifest,
    ProfileCatalog,
    ProfileError,
    load_catalog_manifest,
    load_profile_catalog,
)
from qsorbit.core.profiles.catalog_source import (
    CatalogRefreshUnavailable,
    NotConfiguredCatalogSource,
    ProfileCatalogSource,
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
    "CATALOG_MANIFEST_FILENAME",
    "DEFAULT_PROFILES_DIR",
    "AliveRecord",
    "AliveStatus",
    "CatalogManifest",
    "CatalogRefreshUnavailable",
    "Mode",
    "NotConfiguredCatalogSource",
    "ProfileCatalog",
    "ProfileCatalogSource",
    "ProfileError",
    "ReliabilityClass",
    "SatelliteProfile",
    "Transmitter",
    "load_catalog_manifest",
    "load_profile_catalog",
]
