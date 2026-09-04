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
    """Raised when nothing attached matches the device that was asked for.

    Two ways to ask, and this covers both: an index the enumeration
    does not reach, and an EEPROM serial no attached device carries.

    The library loaded and answered; there is simply no hardware there.
    Nothing to diagnose beyond "is it plugged in, and is it the one you
    meant" — so this is deliberately distinct from :class:`DeviceError`,
    which means something *is* there and it went wrong, and from
    :class:`AmbiguousDeviceError`, which means too many things are.

    The integration suite treats this as a skip, never a failure: there
    is no bug to report in "the dongle isn't connected".
    """


class AmbiguousDeviceError(SdrError):
    """Raised when more than one attached device carries the same serial.

    Not a hypothetical. RTL-SDR Blog V4s ship with the serial
    ``00000001`` burned into every one of them, so the first thing that
    happens when a second identical stick is plugged in is that both
    answer to the same name. Until one is reflashed there is no way to
    say which is which — and picking "the first match" would quietly
    hand back whichever happened to enumerate first, which is the
    unstable thing addressing by serial exists to escape.

    So this refuses rather than guesses, for the same reason
    :class:`DriverMismatchError` does: the alternative failure is
    silent, and looks exactly like success.
    """


class DeviceError(SdrError):
    """Raised when a device is present but a librtlsdr call failed.

    This is the "device misbehaving" half of the pair. It also covers
    the most common real-world case that *looks* like a hardware fault
    and isn't: another program (SDR#, SDR++, a stray ``rtl_test``) still
    holds the device open. The message says so, because at a bench that
    guess is right more often than not.
    """
