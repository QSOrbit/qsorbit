"""Exceptions for the SDR module.

The split that matters here, and the one the Phase 2 brief asks for by
name, is **"no device" versus "device misbehaving"**. They look similar
at a call site and mean completely different things to whoever is
standing at the bench: one is "plug it in", the other is "something is
wrong". A third case sits above both — the *driver* could not be loaded
at all, which is neither, and which on Windows is by far the most likely
thing to go wrong first.
"""


class SdrError(Exception):
    """Base exception for all SDR-related errors."""


class DriverError(SdrError):
    """Raised when the librtlsdr shared library cannot be found or loaded.

    This is a **setup** fault, not a device fault. Nothing is wrong with
    the dongle; QSOrbit could not get as far as asking it anything.

    On Windows this is the first thing that goes wrong for a new user,
    and for a reason that is not obvious: since Python 3.8 (bpo-36085)
    the interpreter no longer searches the working directory or ``PATH``
    when resolving a DLL, so the vendor's own instruction — "copy
    ``rtlsdr.dll`` next to your software" — does nothing. The directory
    has to be registered explicitly. See
    :func:`~qsorbit.core.sdr.librtlsdr.register_driver_directory`.
    """


class DriverMismatchError(DriverError):
    """Raised when the loaded librtlsdr cannot correctly drive this device.

    Specifically: an RTL-SDR Blog V4 was found, but the library that got
    loaded is demonstrably **not** the Blog fork the V4 requires.

    This exists because the failure it prevents is silent. A V4 driven by
    a stock librtlsdr enumerates, opens, accepts a frequency, and streams
    samples — it simply tunes somewhere other than where it was told,
    with nothing anywhere reporting an error. Refusing to open is the
    only way that fault ever announces itself.

    The test is narrow on purpose: it fires only when the device is a
    V4 *and* the loaded library exports a symbol the Blog fork does not
    have. A non-Blog library driving a non-V4 dongle is fine and is not
    touched by this.
    """


class DeviceNotFoundError(SdrError):
    """Raised when no RTL-SDR is present at the requested device index.

    The library loaded and answered; there is simply no hardware there.
    Nothing to diagnose beyond "is it plugged in, and is it the index
    you meant" — so this is deliberately distinct from
    :class:`DeviceError`, which means something *is* there and it went
    wrong.

    The integration suite treats this as a skip, never a failure: there
    is no bug to report in "the dongle isn't connected".
    """


class DeviceError(SdrError):
    """Raised when a device is present but a librtlsdr call failed.

    This is the "device misbehaving" half of the pair. It also covers
    the most common real-world case that *looks* like a hardware fault
    and isn't: another program (SDR#, SDR++, a stray ``rtl_test``) still
    holds the device open. The message says so, because at a bench that
    guess is right more often than not.
    """
