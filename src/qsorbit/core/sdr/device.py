"""The RTL-SDR facade: open it, configure it, read from it.

This is to :mod:`qsorbit.core.sdr.librtlsdr` what
:class:`~qsorbit.core.rotor.Rotor` is to
:class:`~qsorbit.core.rotor.SerialPort` — it owns a device's lifetime,
knows the order things have to happen in, and turns a pile of C calls
into something a caller can reason about. The binding underneath stays
dumb and therefore stays easy to fake.

Two habits are baked in here because bring-up proved they matter:

**Read back what actually happened.** The tuner PLL and the sample clock
both quantise, so what a device is doing is not what it was asked to do.
:class:`AppliedSettings` reports the actual values and how far they sit
from the request, rather than letting a caller assume.

**Refuse to open a V4 through the wrong library.** A Blog V4 driven by a
stock librtlsdr enumerates, opens, and streams — at the wrong frequency,
with nothing reporting an error. That is the failure this whole module
exists to prevent, so it is a refusal rather than a warning.

**Address a device by serial, never by a stored index.** An index is
enumeration order and nothing more; unplugging some other dongle
renumbers it, and a stored one then opens a different radio without
saying so. :func:`index_for_serial` resolves one at the moment of
opening, and refuses rather than guesses when two sticks answer to the
same name — which, out of the box, they do.
"""

from __future__ import annotations

from dataclasses import dataclass

from qsorbit.core.sdr.config import AutoGain, SdrConfig, nearest_gain_step
from qsorbit.core.sdr.exceptions import (
    AmbiguousDeviceError,
    DeviceError,
    DeviceNotFoundError,
    DriverMismatchError,
)
from qsorbit.core.sdr.librtlsdr import DeviceHandle, LibRtlSdr, TunerType

#: EEPROM strings that identify an RTL-SDR Blog V4. The Blog driver's own
#: V4 detection matches on exactly these, which is why the device name is
#: useless for the job — every dongle reports "Generic RTL2832U OEM".
BLOG_V4_MANUFACTURER: str = "RTLSDRBlog"
BLOG_V4_PRODUCT: str = "Blog V4"

#: A default read size: 256 KiB, which is 128 Ki complex samples and about
#: 64 ms at 2.048 Msps. Big enough that the per-call overhead disappears,
#: small enough to stay responsive.
DEFAULT_READ_BYTES: int = 262_144


def _device_label(manufacturer: str, product: str, name: str) -> str:
    """Return the friendliest true name for a device, for one-line output.

    Shared by the two records below so that a device does not describe
    itself one way before it is opened and another way after.
    """
    if manufacturer == BLOG_V4_MANUFACTURER and product == BLOG_V4_PRODUCT:
        return "RTL-SDR Blog V4"
    return name or "unknown device"


@dataclass(frozen=True)
class DeviceInfo:
    """What a device says about itself, read before it is configured.

    Args:
        index: The device index it was found at.
        name: The library's generic name for it, e.g.
            ``"Generic RTL2832U OEM"``. Not distinguishing.
        manufacturer: EEPROM manufacturer string.
        product: EEPROM product string.
        serial: EEPROM serial string. Blank on many dongles, and the
            field :func:`index_for_serial` addresses a device by when
            there is more than one attached.
        tuner: Which tuner chip the library reports.
    """

    index: int
    name: str
    manufacturer: str
    product: str
    serial: str
    tuner: TunerType

    @property
    def is_blog_v4(self) -> bool:
        """``True`` if the EEPROM identifies this as an RTL-SDR Blog V4."""
        return self.manufacturer == BLOG_V4_MANUFACTURER and self.product == BLOG_V4_PRODUCT

    def describe(self) -> str:
        """Return a one-line human description, for logs and CLI output."""
        label = _device_label(self.manufacturer, self.product, self.name)
        serial = f", serial {self.serial}" if self.serial else ""
        return f"[{self.index}] {label} ({self.tuner.name} tuner{serial})"


@dataclass(frozen=True)
class AttachedDevice:
    """One device the library can see, identified without opening it.

    Deliberately smaller than :class:`DeviceInfo`, and the difference is
    the entire point of having two records: :attr:`DeviceInfo.tuner` can
    only be read through an open handle, so building one costs an
    exclusive claim on the device. Every field here comes from
    ``rtlsdr_get_device_name`` and ``rtlsdr_get_device_usb_strings``,
    neither of which needs a handle — so the whole bus can be
    enumerated while something else is already streaming from one of
    them, which is exactly the situation a second receive branch
    creates.

    Args:
        index: Enumeration order, and only that. Not an address worth
            writing down: it changes with which ports are populated and
            with what was plugged in when, which is why
            :func:`index_for_serial` exists at all.
        name: The library's generic name, e.g. ``"Generic RTL2832U
            OEM"``. Not distinguishing — every dongle reports it.
        manufacturer: EEPROM manufacturer string.
        product: EEPROM product string.
        serial: EEPROM serial string. Blank on many dongles and
            identical on many more: both V4s used here shipped as
            ``00000001`` and had to be reflashed before either could be
            named.
    """

    index: int
    name: str
    manufacturer: str
    product: str
    serial: str

    def describe(self) -> str:
        """Return a one-line human description, for logs and CLI output."""
        label = _device_label(self.manufacturer, self.product, self.name)
        return f"[{self.index}] {label}, serial {self.serial or '(blank)'}"


def attached_devices(lib: LibRtlSdr) -> tuple[AttachedDevice, ...]:
    """Enumerate every RTL-SDR the library can see, opening none of them.

    Args:
        lib: A loaded binding.

    Returns:
        One record per device, in enumeration order.

    A device whose USB strings will not read is reported with blank
    strings rather than skipped. It is genuinely attached — the count
    includes it, and on Windows an unbound WinUSB driver produces
    exactly this — so dropping it would turn "one device would not
    answer" into "no device has that serial", which sends whoever is at
    the bench looking in the wrong place entirely.
    """
    devices: list[AttachedDevice] = []
    for index in range(lib.device_count()):
        try:
            manufacturer, product, serial = lib.usb_strings(index)
        except DeviceNotFoundError:
            manufacturer, product, serial = ("", "", "")
        devices.append(
            AttachedDevice(
                index=index,
                name=lib.device_name(index),
                manufacturer=manufacturer,
                product=product,
                serial=serial,
            )
        )
    return tuple(devices)


def index_for_serial(lib: LibRtlSdr, serial: str) -> int:
    """Return the index of the attached device carrying EEPROM ``serial``.

    Args:
        lib: A loaded binding.
        serial: The serial to look for. Matched exactly — it is read
            from the dongle's EEPROM, not normalised by anything here.

    Returns:
        The index that device currently enumerates at. Resolve it again
        next time rather than storing it: it is enumeration order, and
        unplugging some *other* dongle renumbers it.

    Raises:
        DeviceNotFoundError: If nothing attached carries that serial.
            The message lists what is attached, because the useful next
            question is always "then what did I flash it as".
        AmbiguousDeviceError: If more than one device carries it.

    librtlsdr has ``rtlsdr_get_index_by_serial`` and it is deliberately
    not bound: it collapses "no such serial", "no devices at all" and
    "two devices match" into negative return codes, and it cannot say
    what it *did* find. Walking the enumeration costs two extra calls
    per attached device and buys an error message that names the
    alternatives — which, at a bench with two identical sticks, is the
    whole difference between a fix and a guess.
    """
    devices = attached_devices(lib)
    matches = [device for device in devices if device.serial == serial]
    if len(matches) > 1:
        where = ", ".join(f"index {device.index}" for device in matches)
        raise AmbiguousDeviceError(
            f"{len(matches)} attached devices report serial {serial!r} ({where}), "
            "so there is no way to say which one was meant. Serials are not "
            "unique out of the box — every RTL-SDR Blog V4 ships as "
            "'00000001' — so give each stick its own with "
            "'rtl_eeprom -d <index> -s <name>', one plugged in at a time, and "
            "replug afterwards for the new string to be read."
        )
    if not matches:
        raise DeviceNotFoundError(_no_such_serial_message(serial, devices))
    return matches[0].index


def _no_such_serial_message(serial: str, devices: tuple[AttachedDevice, ...]) -> str:
    if not devices:
        return (
            f"No RTL-SDR with serial {serial!r}: the library sees no devices at "
            "all. Check it is plugged in, and that the USB driver is bound "
            "(Zadig on Windows)."
        )
    listing = "\n  ".join(device.describe() for device in devices)
    return (
        f"No RTL-SDR with serial {serial!r}. Attached right now:\n  {listing}\n"
        "Serials are case-sensitive, and are read from the dongle's EEPROM "
        "rather than from anything QSOrbit stores — so this is asking what the "
        "stick is called, not what the config calls it."
    )


@dataclass(frozen=True)
class AppliedSettings:
    """What the device actually did, next to what it was asked for.

    Args:
        requested: The configuration that was applied.
        center_hz: The centre frequency actually tuned.
        sample_rate_hz: The sample rate actually running.
        gain_db: The tuner gain actually in force.
        manual_gain: Whether the tuner is in manual gain mode.
        ppm: The crystal correction actually in force.
        agc_enabled: Whether the RTL2832's digital AGC is on.

    **Two of these are echoed from the request rather than read back**,
    which matters given that reading back is the whole point of this
    class: ``manual_gain`` and ``agc_enabled``. librtlsdr exposes no
    getter for either — there is no ``rtlsdr_get_agc_mode``, and no way
    to ask which gain mode is in force — so there is nothing to read.
    Every other field here is what the hardware reported.
    """

    requested: SdrConfig
    center_hz: float
    sample_rate_hz: float
    gain_db: float
    manual_gain: bool
    ppm: int
    agc_enabled: bool

    @property
    def center_error_hz(self) -> float:
        """How far the actual centre sits from the requested one, in Hz.

        Small non-zero values are normal — the PLL quantises. A large
        one is the signature of the silent-mistune failure, and is worth
        looking at before blaming an antenna.
        """
        return self.center_hz - self.requested.center_hz

    @property
    def sample_rate_error_hz(self) -> float:
        """How far the actual sample rate sits from the requested one, in Hz."""
        return self.sample_rate_hz - self.requested.sample_rate_hz

    def offset_from(self, station_hz: float) -> float:
        """Return where a signal at ``station_hz`` lands in this capture.

        Args:
            station_hz: The frequency of interest, in Hz.

        Returns:
            Its offset from the centre actually tuned, in Hz. Positive
            means above centre.

        Measured from :attr:`center_hz` and not from the requested
        frequency, which is the entire reason this lives on the applied
        settings rather than on the config: the PLL quantises, and an
        offset computed against a frequency the tuner never reached is
        wrong by exactly the amount nobody thinks to check.

        Small enough to look unnecessary, and here anyway, because
        getting this sign backwards is the easiest mistake in the area
        and because bring-up established that captures are made
        deliberately *off*-centre — a peak at DC cannot be told apart
        from the RTL-SDR's permanent DC-offset spike, so it proves
        nothing.
        """
        return station_hz - self.center_hz

    @property
    def reports_zero_gain(self) -> bool:
        """``True`` if the tuner claims 0.0 dB of gain.

        Nearly always means the capture will be empty. Observed on the
        V4 in automatic gain mode during bring-up: 0.0 dB reported, flat
        noise floor captured, no error anywhere. Worth checking before a
        capture rather than after.
        """
        return self.gain_db <= 0.0


class RtlSdr:
    """One RTL-SDR, opened and configured.

    Supports use as a context manager::

        with RtlSdr(driver_dir=r"C:\\Users\\me\\dev\\rtlsdr-blog\\x64") as sdr:
            applied = sdr.configure(config)
            raw = sdr.read_raw()

    Args:
        index: Which device to open, when more than one is attached.
        driver_dir: Directory holding librtlsdr. Required in practice on
            Windows; optional elsewhere. See
            :func:`~qsorbit.core.sdr.librtlsdr.load_library`.
        _lib: A pre-built binding. Intended for unit testing — pass a
            fake here and no real library is ever loaded, the same way
            :class:`~qsorbit.core.rotor.SerialPort` takes an injected
            serial instance.
    """

    def __init__(
        self,
        index: int = 0,
        *,
        driver_dir: str | None = None,
        _lib: LibRtlSdr | None = None,
    ) -> None:
        self._index = index
        self._driver_dir = driver_dir
        self._lib = _lib
        self._handle: DeviceHandle | None = None
        self._info: DeviceInfo | None = None
        self._applied: AppliedSettings | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def index(self) -> int:
        """The device index this instance addresses."""
        return self._index

    @property
    def is_open(self) -> bool:
        """``True`` if a device handle is currently held."""
        return self._handle is not None

    @property
    def info(self) -> DeviceInfo | None:
        """What the open device said about itself, or ``None`` if closed."""
        return self._info

    @property
    def applied(self) -> AppliedSettings | None:
        """The most recent applied configuration, or ``None`` if never configured."""
        return self._applied

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> DeviceInfo:
        """Open the device and identify it.

        Loads the library if it has not been loaded already, reads the
        device's USB strings, opens it, and checks that the library is
        one that can actually drive it.

        Returns:
            What the device says about itself.

        Raises:
            DriverError: If librtlsdr cannot be loaded.
            DriverMismatchError: If the device is a Blog V4 and the
                loaded library is demonstrably not the Blog fork.
            DeviceNotFoundError: If there is no device at this index.
            DeviceError: If the device is there but will not open.
        """
        if self._info is not None:
            return self._info

        if self._lib is None:
            self._lib = LibRtlSdr.load(self._driver_dir)

        manufacturer, product, serial = self._lib.usb_strings(self._index)
        # Checked before opening: a mismatch is not something we want to
        # have a live handle for, and the USB strings do not need one.
        if (
            manufacturer == BLOG_V4_MANUFACTURER
            and product == BLOG_V4_PRODUCT
            and self._lib.exports_dithering
        ):
            raise DriverMismatchError(
                "This is an RTL-SDR Blog V4, but the librtlsdr that loaded is not "
                "the Blog fork it requires (it exports rtlsdr_set_dithering, which "
                "the Blog fork does not). A V4 on a stock librtlsdr opens and "
                "streams normally while tuning somewhere other than where it is "
                "told, so this is refused rather than warned about. Point "
                "driver_dir at the RTL-SDR Blog driver release."
            )

        handle = self._lib.open(self._index)
        try:
            info = DeviceInfo(
                index=self._index,
                name=self._lib.device_name(self._index),
                manufacturer=manufacturer,
                product=product,
                serial=serial,
                tuner=self._lib.tuner_type(handle),
            )
        except Exception:
            # Never leak a handle because identification failed.
            self._lib.close(handle)
            raise

        self._handle = handle
        self._info = info
        return info

    def close(self) -> None:
        """Close the device if it is open. Safe to call repeatedly."""
        if self._handle is not None and self._lib is not None:
            self._lib.close(self._handle)
        self._handle = None
        self._info = None
        self._applied = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def supported_gains_db(self) -> tuple[float, ...]:
        """Return the gain steps this tuner offers, in dB.

        Read from the device rather than hardcoded, because the table
        belongs to the tuner chip. It doubles as a fingerprint of what
        got loaded: a V4's R828D reports 29 steps from 0.0 to 49.6 dB,
        and a library reporting some other table found different
        hardware than you think.
        """
        return self._require_lib().tuner_gains(self._require_handle())

    def configure(self, config: SdrConfig) -> AppliedSettings:
        """Apply ``config`` to the open device and report what it did.

        The call order is deliberate and is not arbitrary:

        1. **Sample rate first.** It determines the usable bandwidth,
           and librtlsdr recomputes tuner bandwidth from it.
        2. **Then the ppm correction.** Setting it re-applies the
           current frequency internally, so it has to come before the
           frequency we actually want, or it silently re-tunes off the
           old one.
        3. **Then the centre frequency.**
        4. **Then gain mode, then gain.** Setting a gain value has no
           effect while the tuner is in automatic mode.
        5. **Then the digital AGC**, which is a separate control.
        6. **Then reset the buffer**, discarding everything captured at
           the old settings. Skipping this returns plausible-looking
           samples from the previous configuration.

        Args:
            config: What to apply.

        Returns:
            The actual settings, alongside the requested ones.

        Raises:
            DeviceError: If the device rejects a setting or the device
                is not open.
        """
        lib = self._require_lib()
        handle = self._require_handle()

        lib.set_sample_rate(handle, int(config.sample_rate_hz))
        lib.set_freq_correction(handle, config.ppm)
        lib.set_center_freq(handle, int(config.center_hz))

        gain = config.gain_db
        manual = not isinstance(gain, AutoGain)
        lib.set_tuner_gain_mode(handle, manual)
        if not isinstance(gain, AutoGain):
            step = nearest_gain_step(float(gain), lib.tuner_gains(handle))
            lib.set_tuner_gain(handle, step)

        lib.set_agc_mode(handle, config.enable_agc)
        lib.reset_buffer(handle)

        applied = AppliedSettings(
            requested=config,
            center_hz=float(lib.get_center_freq(handle)),
            sample_rate_hz=float(lib.get_sample_rate(handle)),
            gain_db=lib.get_tuner_gain(handle),
            manual_gain=manual,
            ppm=lib.get_freq_correction(handle),
            agc_enabled=config.enable_agc,
        )
        self._applied = applied
        return applied

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_raw(self, length: int = DEFAULT_READ_BYTES) -> bytes:
        """Read one block of raw interleaved I/Q, blocking until it arrives.

        Args:
            length: Bytes to read. Must be a positive multiple of 512.

        Returns:
            Raw uint8 samples, interleaved I, Q, I, Q — two bytes per
            complex sample, offset binary with 127.5 as zero. This layer
            deliberately does not convert to complex numbers: that
            belongs in ``core/dsp``, and keeping the raw form here is
            what lets a captured file and a live stream be the same
            thing to everything downstream.

        Raises:
            ValueError: If ``length`` is not a positive multiple of 512.
            DeviceError: If the device is not open, or the read fails.
        """
        return self._require_lib().read_sync(self._require_handle(), length)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> RtlSdr:
        """Open the device on entering a ``with`` block."""
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Close the device on exiting a ``with`` block, even on error."""
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_lib(self) -> LibRtlSdr:
        if self._lib is None:
            raise DeviceError("No device is open — call open() first.")
        return self._lib

    def _require_handle(self) -> DeviceHandle:
        if self._handle is None:
            raise DeviceError("No device is open — call open() first.")
        return self._handle
