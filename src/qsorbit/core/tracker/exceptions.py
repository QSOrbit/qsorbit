"""Exceptions for the tracker module."""


class TrackerError(Exception):
    """Base exception for all tracker-related errors."""


class TleError(TrackerError):
    """Raised when a two-line element set cannot be parsed."""


class PropagationError(TrackerError):
    """Raised when SGP4 cannot compute a valid position at the requested time.

    This happens when the requested time is far enough from the TLE's
    epoch that the underlying orbital elements no longer describe a
    physically sensible orbit — for example, the satellite has since
    decayed and re-entered. SGP4 signals this internally by returning
    NaN position/velocity with an explanatory message rather than
    raising; this exception surfaces that message instead of letting
    NaNs propagate silently into the rest of the app.
    """
