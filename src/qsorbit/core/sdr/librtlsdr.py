"""A direct ctypes binding to the RTL-SDR Blog ``librtlsdr``.

This is the SDR module's thin wrapper — the same role
:class:`~qsorbit.core.rotor.SerialPort` plays for the rotor. It knows
how to load the shared library, how each C function is shaped, and how
that library reports failure. It holds no policy: it does not decide
what a sensible gain is, it does not remember what the device is tuned
to, and it does not order the configuration calls. That is
:class:`~qsorbit.core.sdr.device.RtlSdr`'s job.

**Why a hand-written binding at all** (decided during bring-up,
2026-08-22): the RTL-SDR Blog V4 requires the Blog fork of librtlsdr —
a stock build appears to work and silently mistunes. ``pyrtlsdr`` binds
``rtlsdr_set_dithering`` unconditionally at import, and the Blog fork
does not export that symbol, so importing it dies outright. SoapySDR
links against whatever librtlsdr it was built against, which very
likely reintroduces the same silent mistune underneath an abstraction
layer. The project already made this call once, in Session 8, when it
dropped hamlib and spoke the rotator's protocol directly: a
compatibility layer you do not control becomes the thing that breaks.

**Windows DLL resolution is the platform requirement to know about.**
Since Python 3.8 (bpo-36085) neither the working directory nor ``PATH``
is searched when a DLL is resolved. The vendor's install procedure —
"copy ``rtlsdr.dll`` next to your software" — therefore does nothing
from Python, and no wrapper library can fix that for us. The directory
must be registered with :func:`os.add_dll_directory`, which is the
application's job and is why ``driver_dir`` exists in the station
config.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any, Final

from qsorbit.core.sdr.exceptions import DeviceError, DeviceNotFoundError, DriverError

#: Library file names tried in order when no explicit driver directory is
#: given. Windows is listed for completeness, but on Windows a bare name
#: only resolves once a directory has been registered — see the module
#: docstring.
LIBRARY_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "win32": ("rtlsdr.dll", "librtlsdr.dll"),
    "darwin": ("librtlsdr.dylib", "librtlsdr.0.dylib"),
    "default": ("librtlsdr.so.0", "librtlsdr.so"),
}

#: ``rtlsdr_read_sync`` requires its buffer length to be a multiple of
#: this. Not a suggestion — a length that is not is rejected.
READ_BLOCK_MULTIPLE: Final = 512

#: Symbol the RTL-SDR Blog fork does **not** export, and stock/osmocom
#: builds from 2018 onward do. Its presence therefore proves the loaded
#: library is not the Blog fork. Its *absence* proves less — an old
#: osmocom build also lacks it — so this is used only as a red flag, and
#: only when the device present is one that needs the Blog fork.
#:
#: This is the same symbol collision that rules out ``pyrtlsdr``, turned
#: around and used as a diagnostic.
NON_BLOG_MARKER_SYMBOL: Final = "rtlsdr_set_dithering"

#: ``rtlsdr_set_freq_correction`` returns this when the requested ppm is
#: already in force. It is a "nothing to do", not a failure.
PPM_UNCHANGED: Final = -2

#: Handles returned by :func:`os.add_dll_directory`, kept alive for the
#: life of the process and keyed by resolved directory.
#:
#: They are deliberately never closed. Closing one un-registers the
#: directory, and any later load — including one librtlsdr itself
#: triggers — would then fail. Registering the same directory twice is
#: harmless but leaks a handle, so this doubles as an idempotence guard.
_registered_directories: dict[Path, Any] = {}


#: A handle to an open device. Opaque: it is a ``rtlsdr_dev_t *`` on the
#: other side and is only ever handed straight back to the library.
DeviceHandle = ctypes.c_void_p


class TunerType(IntEnum):
    """Tuner chips librtlsdr can report, with its own numbering.

    Worth surfacing because it is a cheap sanity check on what actually
    got loaded and found: an RTL-SDR Blog V4 is an
    :attr:`R828D`, and a V4 reporting anything else means something
    upstream of us is confused.
    """

    UNKNOWN = 0
    E4000 = 1
    FC0012 = 2
    FC0013 = 3
    FC2580 = 4
    R820T = 5
    R828D = 6


def register_driver_directory(directory: str | os.PathLike[str]) -> None:
    """Make ``directory`` searchable for DLL loads, on Windows.

    A no-op on platforms without :func:`os.add_dll_directory`. Calling
    it repeatedly with the same directory is safe and does nothing after
    the first time.

    Args:
        directory: The folder containing ``rtlsdr.dll`` and its
            companions (``msvcr100.dll``, ``pthreadVC2.dll``), e.g. the
            ``x64`` folder from the RTL-SDR Blog driver release.

    Raises:
        DriverError: If the directory does not exist, or the OS refuses
            to register it.
    """
    resolved = Path(directory).expanduser().resolve()
    if resolved in _registered_directories:
        return
    if not resolved.is_dir():
        raise DriverError(
            f"Driver directory not found: {resolved}. This should be the folder "
            "containing rtlsdr.dll from the RTL-SDR Blog driver release — the "
            "'x64' folder on 64-bit Windows."
        )
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        # Not Windows. Nothing to register; the loader's own search path
        # applies and an explicit full path still works.
        return
    try:
        _registered_directories[resolved] = add_dll_directory(str(resolved))
    except OSError as exc:
        raise DriverError(f"Could not register driver directory {resolved}: {exc}") from exc


def load_library(driver_dir: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    """Load librtlsdr and return the raw :class:`ctypes.CDLL`.

    Args:
        driver_dir: Directory holding the library. When given, the
            directory is registered first and the library is loaded from
            an explicit full path — the only reliable way on Windows.
            When ``None``, the platform's own search is used, which is
            the normal case on Linux and macOS where librtlsdr is
            installed by a package manager.

    Returns:
        The loaded library.

    Raises:
        DriverError: If the library cannot be found or cannot be loaded.
            The message distinguishes the two, because they have
            completely different fixes.
    """
    if driver_dir is not None:
        register_driver_directory(driver_dir)
        directory = Path(driver_dir).expanduser().resolve()
        candidates = [directory / name for name in _candidate_names()]
        existing = [path for path in candidates if path.is_file()]
        if not existing:
            tried = ", ".join(path.name for path in candidates)
            raise DriverError(
                f"No librtlsdr found in {directory}. Looked for: {tried}. "
                "On Windows this should be the 'x64' folder from the RTL-SDR Blog "
                "driver release (not the vendor page's stale link — check the "
                "GitHub releases)."
            )
        return _load_from(str(existing[0]), origin=str(existing[0]))

    for name in _candidate_names():
        try:
            return _load_from(name, origin=name)
        except DriverError:
            continue

    found = ctypes.util.find_library("rtlsdr")
    if found:
        return _load_from(found, origin=found)

    raise DriverError(
        "Could not find librtlsdr anywhere on the library search path. "
        "Install it (package 'librtlsdr' on most Linux distributions), or set "
        "driver_dir in the [sdr] section of your station config to the folder "
        "containing it. On Windows a driver directory is effectively mandatory: "
        "since Python 3.8 the working directory and PATH are not searched."
    )


def _candidate_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return LIBRARY_NAMES["win32"]
    if sys.platform == "darwin":
        return LIBRARY_NAMES["darwin"]
    return LIBRARY_NAMES["default"]


def _load_from(target: str, *, origin: str) -> ctypes.CDLL:
    try:
        return ctypes.CDLL(target)
    except OSError as exc:
        raise DriverError(
            f"Found {origin} but could not load it: {exc}. The usual cause is an "
            "architecture mismatch — a 32-bit DLL under 64-bit Python, or the "
            "reverse — or a missing companion DLL that ships alongside it "
            "(msvcr100.dll, pthreadVC2.dll)."
        ) from exc


class LibRtlSdr:
    """Typed access to the librtlsdr C functions.

    Roughly eighteen functions, which is all of Phase 2. Each method is
    a direct translation of one C call plus the error mapping that call
    needs; there is no state here beyond the loaded library itself.
    Device handles are passed in and out rather than held, so this class
    stays trivially fakeable — the facade above it is what owns a
    device's lifetime.

    Args:
        library: A loaded librtlsdr. Pass one built by
            :func:`load_library`, or, in tests, any object exposing the
            same symbols.

    Raises:
        DriverError: If the library is missing a symbol this binding
            requires, which means it is not librtlsdr at all.
    """

    def __init__(self, library: ctypes.CDLL) -> None:
        self._lib = library
        self._bind()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, driver_dir: str | os.PathLike[str] | None = None) -> LibRtlSdr:
        """Load librtlsdr and wrap it.

        Args:
            driver_dir: See :func:`load_library`.

        Returns:
            A binding ready to use.

        Raises:
            DriverError: If the library cannot be loaded or is not
                librtlsdr.
        """
        return cls(load_library(driver_dir))

    def _bind(self) -> None:
        dev_p = ctypes.c_void_p
        int_p = ctypes.POINTER(ctypes.c_int)
        char_p = ctypes.c_char_p
        u32 = ctypes.c_uint32
        c_int = ctypes.c_int

        signatures: tuple[tuple[str, Any, tuple[Any, ...]], ...] = (
            ("rtlsdr_get_device_count", u32, ()),
            ("rtlsdr_get_device_name", char_p, (u32,)),
            ("rtlsdr_get_device_usb_strings", c_int, (u32, char_p, char_p, char_p)),
            ("rtlsdr_open", c_int, (ctypes.POINTER(dev_p), u32)),
            ("rtlsdr_close", c_int, (dev_p,)),
            ("rtlsdr_set_sample_rate", c_int, (dev_p, u32)),
            ("rtlsdr_get_sample_rate", u32, (dev_p,)),
            ("rtlsdr_set_center_freq", c_int, (dev_p, u32)),
            ("rtlsdr_get_center_freq", u32, (dev_p,)),
            ("rtlsdr_set_freq_correction", c_int, (dev_p, c_int)),
            ("rtlsdr_get_freq_correction", c_int, (dev_p,)),
            ("rtlsdr_set_tuner_gain_mode", c_int, (dev_p, c_int)),
            ("rtlsdr_set_tuner_gain", c_int, (dev_p, c_int)),
            ("rtlsdr_get_tuner_gain", c_int, (dev_p,)),
            ("rtlsdr_get_tuner_gains", c_int, (dev_p, int_p)),
            ("rtlsdr_get_tuner_type", c_int, (dev_p,)),
            ("rtlsdr_set_agc_mode", c_int, (dev_p, c_int)),
            ("rtlsdr_reset_buffer", c_int, (dev_p,)),
            ("rtlsdr_read_sync", c_int, (dev_p, ctypes.c_void_p, c_int, int_p)),
        )

        self._fn: dict[str, Any] = {}
        for name, restype, argtypes in signatures:
            try:
                function = getattr(self._lib, name)
            except AttributeError as exc:
                raise DriverError(
                    f"The loaded library does not export {name!r}, so it is not a "
                    "librtlsdr this binding can use. Check that driver_dir points at "
                    "the RTL-SDR Blog driver release and not at some other DLL that "
                    "happens to share the name."
                ) from exc
            function.restype = restype
            function.argtypes = list(argtypes)
            self._fn[name] = function

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def exports_dithering(self) -> bool:
        """``True`` if the loaded library exports :data:`NON_BLOG_MARKER_SYMBOL`.

        Which means it is **not** the RTL-SDR Blog fork. The converse
        does not hold — an old stock build lacks the symbol too — so
        this is only ever read as a red flag, never as a certificate.
        """
        return hasattr(self._lib, NON_BLOG_MARKER_SYMBOL)

    # ------------------------------------------------------------------
    # Enumeration — no device handle needed
    # ------------------------------------------------------------------

    def device_count(self) -> int:
        """Return how many RTL-SDR devices the library can see."""
        return int(self._fn["rtlsdr_get_device_count"]())

    def device_name(self, index: int) -> str:
        """Return the device's name string, or ``""`` if there is none.

        Note that this is the *generic* name — a Blog V4 reports
        ``"Generic RTL2832U OEM"`` like anything else. V4-ness lives in
        the USB strings, not here.
        """
        raw = self._fn["rtlsdr_get_device_name"](index)
        return raw.decode("utf-8", errors="replace") if raw else ""

    def usb_strings(self, index: int) -> tuple[str, str, str]:
        """Return ``(manufacturer, product, serial)`` from the device's EEPROM.

        These are what actually identify a Blog V4: manufacturer
        ``RTLSDRBlog``, product ``Blog V4``. The driver's own V4
        detection matches on exactly these strings.

        Args:
            index: Device index.

        Returns:
            The three strings, each possibly empty.

        Raises:
            DeviceNotFoundError: If there is no device at ``index``.
        """
        # librtlsdr writes up to 256 bytes into each buffer.
        manufacturer = ctypes.create_string_buffer(256)
        product = ctypes.create_string_buffer(256)
        serial = ctypes.create_string_buffer(256)
        result = self._fn["rtlsdr_get_device_usb_strings"](index, manufacturer, product, serial)
        if result != 0:
            raise DeviceNotFoundError(
                f"No RTL-SDR at index {index} (reading its USB strings returned {result})."
            )
        return (
            manufacturer.value.decode("utf-8", errors="replace"),
            product.value.decode("utf-8", errors="replace"),
            serial.value.decode("utf-8", errors="replace"),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, index: int) -> ctypes.c_void_p:
        """Open the device at ``index`` and return its handle.

        Raises:
            DeviceNotFoundError: If no device exists at that index.
            DeviceError: If the device exists but will not open — most
                often because another program still has it.
        """
        count = self.device_count()
        if index >= count:
            raise DeviceNotFoundError(
                f"No RTL-SDR at index {index}: the library sees {count} device(s). "
                "Check it is plugged in, and that the USB driver is bound (Zadig "
                "on Windows)."
            )
        handle = ctypes.c_void_p()
        result = self._fn["rtlsdr_open"](ctypes.byref(handle), index)
        if result != 0 or not handle:
            raise DeviceError(
                f"Found an RTL-SDR at index {index} but could not open it "
                f"(rc={result}). The usual cause is that something else already "
                "has it open — close SDR#, SDR++, rtl_tcp or any stray rtl_test "
                "and try again."
            )
        return handle

    def close(self, handle: ctypes.c_void_p) -> None:
        """Close a device handle. Safe to call with a null handle."""
        if not handle:
            return
        self._fn["rtlsdr_close"](handle)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_sample_rate(self, handle: ctypes.c_void_p, rate_hz: int) -> None:
        """Set the sample rate in Hz."""
        self._check(self._fn["rtlsdr_set_sample_rate"](handle, int(rate_hz)), "set sample rate")

    def get_sample_rate(self, handle: ctypes.c_void_p) -> int:
        """Return the sample rate the device is *actually* running at.

        Not the requested one. The sample clock quantises, and every
        spectrum calculation downstream has to use this value or its
        frequency axis is quietly wrong.
        """
        return int(self._fn["rtlsdr_get_sample_rate"](handle))

    def set_center_freq(self, handle: ctypes.c_void_p, freq_hz: int) -> None:
        """Set the centre frequency in Hz."""
        self._check(
            self._fn["rtlsdr_set_center_freq"](handle, int(freq_hz)), "set centre frequency"
        )

    def get_center_freq(self, handle: ctypes.c_void_p) -> int:
        """Return the centre frequency the device is *actually* tuned to.

        The tuner PLL quantises, so this can differ from what was asked
        for. Read it, never assume it.
        """
        return int(self._fn["rtlsdr_get_center_freq"](handle))

    def set_freq_correction(self, handle: ctypes.c_void_p, ppm: int) -> None:
        """Set the crystal correction in parts per million.

        A return of :data:`PPM_UNCHANGED` means the value was already in
        force, which is success with nothing to do — treating it as a
        failure would make setting ppm=0 on a fresh device an error.
        """
        result = self._fn["rtlsdr_set_freq_correction"](handle, int(ppm))
        if result == PPM_UNCHANGED:
            return
        self._check(result, "set frequency correction")

    def get_freq_correction(self, handle: ctypes.c_void_p) -> int:
        """Return the crystal correction currently in force, in ppm."""
        return int(self._fn["rtlsdr_get_freq_correction"](handle))

    def set_tuner_gain_mode(self, handle: ctypes.c_void_p, manual: bool) -> None:
        """Select manual or automatic tuner gain.

        Args:
            manual: ``True`` for manual gain (what you almost always
                want — see :class:`~qsorbit.core.sdr.config.AutoGain`),
                ``False`` to let the tuner decide.
        """
        self._check(
            self._fn["rtlsdr_set_tuner_gain_mode"](handle, 1 if manual else 0),
            "set tuner gain mode",
        )

    def set_tuner_gain(self, handle: ctypes.c_void_p, gain_db: float) -> None:
        """Set the tuner gain, in dB.

        librtlsdr works in tenths of a dB, which is converted here so
        nothing above this layer has to remember it. Only has an effect
        in manual gain mode.
        """
        tenths = int(round(gain_db * 10))
        self._check(self._fn["rtlsdr_set_tuner_gain"](handle, tenths), "set tuner gain")

    def get_tuner_gain(self, handle: ctypes.c_void_p) -> float:
        """Return the tuner gain currently in force, in dB.

        In automatic mode this reports whatever the tuner settled on,
        and on the V4 during bring-up that was 0.0 dB with an empty
        capture behind it. A zero here is worth treating as suspicious
        rather than as a valid quiet setting.
        """
        return self._fn["rtlsdr_get_tuner_gain"](handle) / 10.0

    def tuner_gains(self, handle: ctypes.c_void_p) -> tuple[float, ...]:
        """Return every gain step this tuner supports, in dB, ascending.

        Two calls, as librtlsdr requires: a null pointer asks how many
        there are, then a buffer of that size collects them.

        Raises:
            DeviceError: If the device will not report its gain table.
        """
        count = self._fn["rtlsdr_get_tuner_gains"](handle, None)
        if count <= 0:
            raise DeviceError(
                f"The device reported {count} supported gain steps. A working tuner "
                "reports at least one; this usually means the tuner was not "
                "identified at open."
            )
        buffer = (ctypes.c_int * count)()
        written = self._fn["rtlsdr_get_tuner_gains"](handle, buffer)
        if written != count:
            raise DeviceError(
                f"The device said it had {count} gain steps and then returned "
                f"{written}. Something is wrong with the device or the library."
            )
        return tuple(value / 10.0 for value in buffer)

    def tuner_type(self, handle: ctypes.c_void_p) -> TunerType:
        """Return which tuner chip the device reports."""
        raw = int(self._fn["rtlsdr_get_tuner_type"](handle))
        try:
            return TunerType(raw)
        except ValueError:
            return TunerType.UNKNOWN

    def set_agc_mode(self, handle: ctypes.c_void_p, enabled: bool) -> None:
        """Enable or disable the RTL2832's digital AGC.

        Separate from tuner gain, and easily confused with it. This one
        operates on the already-digitised signal; the tuner gain is
        analogue and ahead of the ADC.
        """
        self._check(self._fn["rtlsdr_set_agc_mode"](handle, 1 if enabled else 0), "set AGC mode")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def reset_buffer(self, handle: ctypes.c_void_p) -> None:
        """Discard whatever the device has already buffered.

        Required before the first read, and after any retune: without
        it the next read returns samples captured at the *previous*
        settings, which is a particularly nasty way to be wrong because
        the data looks entirely plausible.
        """
        self._check(self._fn["rtlsdr_reset_buffer"](handle), "reset buffer")

    def read_sync(self, handle: ctypes.c_void_p, length: int) -> bytes:
        """Read up to ``length`` bytes of raw interleaved I/Q, blocking.

        Args:
            handle: An open device.
            length: How many bytes to read. Must be a positive multiple
                of :data:`READ_BLOCK_MULTIPLE`.

        Returns:
            The bytes actually read, which may be fewer than requested.
            Raw uint8, interleaved I, Q, I, Q — two bytes per complex
            sample, offset binary with 127.5 as zero. Converting that to
            complex numbers belongs downstream in ``core/dsp``, not
            here.

        Raises:
            ValueError: If ``length`` is not a positive multiple of 512.
            DeviceError: If the read fails or returns nothing.

        **This method is on a real-time path and its cost is measured.**
        No USB transfer is in flight between one synchronous read
        returning and the next being issued, so every microsecond spent
        here is a microsecond the device's FIFO spends overflowing —
        time turns directly into lost samples, with nothing anywhere
        reporting an error. See
        :class:`~qsorbit.core.sdr.stream.ThroughputMonitor`, and the
        bench measurement in ``tests/integration/test_sdr_streaming.py``
        that exists to keep this honest.
        """
        if length <= 0 or length % READ_BLOCK_MULTIPLE != 0:
            raise ValueError(
                f"length must be a positive multiple of {READ_BLOCK_MULTIPLE}, got {length}."
            )
        buffer = (ctypes.c_ubyte * length)()
        n_read = ctypes.c_int()
        self._check(
            self._fn["rtlsdr_read_sync"](handle, buffer, length, ctypes.byref(n_read)),
            "read samples",
        )
        if n_read.value <= 0:
            raise DeviceError(
                "The device accepted a read and then returned no samples. It is "
                "present but not streaming; a reconnect or power cycle is the "
                "usual fix."
            )
        # NOT bytes(buffer[:n]), which is the obvious spelling and is a
        # trap: slicing a ctypes array builds a Python *list* of that
        # many integers, then walks it again to make bytes. At a
        # 262,144-byte block that measured 3.8 ms in Claude's sandbox
        # and roughly 6.7 ms on Phil's machine, against a block that is
        # only 64 ms of samples — which is where 2026-08-22's first
        # streaming measurement lost 9.5% of the stream. A memoryview
        # slice is one memcpy and measured 0.008 ms, a ~400x difference.
        #
        # Reusing a single buffer across reads was also measured and
        # deliberately not done: it saves a further 0.009 ms, which is
        # not worth reasoning about aliasing between the reader thread
        # and its consumer for.
        return bytes(memoryview(buffer)[: n_read.value])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _check(result: int, what: str) -> None:
        """Turn librtlsdr's integer return into an exception, or nothing.

        Every call in this library returns 0 for success and a negative
        libusb error code otherwise. There is no error string to fetch,
        so the code is all the detail there is.
        """
        if result != 0:
            raise DeviceError(f"librtlsdr could not {what} (rc={result}).")
