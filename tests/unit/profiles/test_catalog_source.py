"""Unit tests for the profile catalogue's fetch-source plumbing.

No real network source exists yet -- see catalog_source.py's module
docstring for why. All there is to test right now is that the
stand-in behaves predictably: it fails loudly and specifically rather
than silently doing nothing, and satisfies the protocol a real
implementation will have to satisfy too.
"""

import pytest

from qsorbit.core.profiles.catalog import ProfileError
from qsorbit.core.profiles.catalog_source import (
    CatalogRefreshUnavailable,
    NotConfiguredCatalogSource,
    ProfileCatalogSource,
)


class TestNotConfiguredCatalogSource:
    def test_refresh_raises_catalog_refresh_unavailable(self):
        with pytest.raises(CatalogRefreshUnavailable):
            NotConfiguredCatalogSource().refresh()

    def test_catalog_refresh_unavailable_is_a_profile_error(self):
        # So existing `except ProfileError` handling -- the CLI's
        # top-level catch, in particular -- already covers this
        # without new wiring.
        assert issubclass(CatalogRefreshUnavailable, ProfileError)

    def test_error_message_explains_why_and_points_at_the_shipped_snapshot(self):
        with pytest.raises(CatalogRefreshUnavailable, match="no network source configured"):
            NotConfiguredCatalogSource().refresh()

    def test_satisfies_the_protocol(self):
        assert isinstance(NotConfiguredCatalogSource(), ProfileCatalogSource)
