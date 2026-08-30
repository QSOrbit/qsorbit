"""A pluggable source for refreshing the profile catalogue over the network.

The Chunk D planning decision calls for the profile catalogue to ship
in-repo plus support an "optional network refresh... the same shape as
the deferred TLE-catalog fetch, so the two can share plumbing when
that lands" -- but names no actual source for either. Picking a real
one (a URL is a real, permanent external dependency someone has to
host and maintain) is a decision for Phil to make deliberately, not
one to guess at while writing this module.

So this ships the *shape* rather than a guess: :class:`ProfileCatalogSource`,
a protocol any future implementation can satisfy, and
:class:`NotConfiguredCatalogSource`, the default that makes "no source
configured yet" an explicit, catchable state instead of a silent
no-op or a made-up endpoint. The picker's "refresh" action (later in
Chunk D) and the CLI's ``--refresh-catalogue`` flag both wire to this
today and will start working the moment a real source lands, with no
further caller-side change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from qsorbit.core.profiles.catalog import ProfileCatalog, ProfileError


class CatalogRefreshUnavailable(ProfileError):
    """Raised by :meth:`ProfileCatalogSource.refresh` when no real source is wired up yet.

    A subclass of :class:`~qsorbit.core.profiles.catalog.ProfileError`
    on purpose, so every place that already catches ``ProfileError``
    (the CLI's top-level handler, in particular) covers this too
    without new wiring.
    """


@runtime_checkable
class ProfileCatalogSource(Protocol):
    """Something that can fetch an updated profile catalogue over the network."""

    def refresh(self) -> ProfileCatalog:
        """Fetch and return an updated catalogue.

        Raises:
            CatalogRefreshUnavailable: If this source isn't configured
                yet, or -- once a real implementation exists -- on a
                genuine fetch failure (network, parsing, or otherwise).
        """
        ...


class NotConfiguredCatalogSource:
    """The default :class:`ProfileCatalogSource`: refresh isn't wired to anything yet.

    Every call to :meth:`refresh` raises :class:`CatalogRefreshUnavailable`
    with the same message, so a caller gets a clear, stable answer
    rather than a fetch that silently does nothing.
    """

    def refresh(self) -> ProfileCatalog:
        raise CatalogRefreshUnavailable(
            "Catalogue refresh has no network source configured yet -- showing "
            "the shipped snapshot. Tracked alongside the deferred TLE-catalog "
            "fetch (see phase-3-roadmap.md)."
        )
