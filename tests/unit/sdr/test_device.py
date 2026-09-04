"""Tests for the RTL-SDR facade.

The fake library here is written by hand rather than mocked, for two
reasons. It records call *order*, which is the part of ``configure()``
that is easy to get wrong and impossible to notice — a gain set before
the mode, or a buffer never reset, both look fine and quietly produce
wrong data. And a ``MagicMock`` would make ``exports_dithering`` truthy
by accident, which would fire the driver-mismatch refusal on every test.
"""

from __future__ import annotations

import pytest

from qsorbit.core.sdr import (
    AUTO_GAIN,
    AmbiguousDeviceError,
    DeviceError,
    DeviceInfo,
    DeviceNotFoundError,
    DriverMismatchError,
    RtlSdr,
    SdrConfig,
    TunerType,
    attached_devices,
    index_for_serial,
)

V4_GAIN_STEPS_DB = (0.0, 0.9, 1.4, 12.5, 25.4, 32.8, 33.8, 49.6)


class FakeLib:
    """A stand-in :class:`~qsorbit.core.sdr.LibRtlSdr` with a fake device.

    Records every call in order, quantises the centre frequency the way
    a real PLL does, and reports 0.0 dB in automatic gain mode — which
    is what the V4 actually did during bring-up.
    """

    def __init__(
        self,
        *,
        manufacturer: str = "RTLSDRBlog",
        product: str = "Blog V4",
        exports_dithering: bool = False,
        gains: tuple[float, ...] = V4_GAIN_STEPS_DB,
    ) -> None:
        self.exports_dithering = exports_dithering
        self._manufacturer = manufacturer
        self._product = product
        self._gains = gains
        self.calls: list[str] = []
        self.closed = 0
        self.center_hz = 0
        self.sample_rate_hz = 0
        self.gain_db = 0.0
        self.manual = False
        self.ppm = 0
        self.read_returns = b""

    def _record(self, name: str) -> None:
        self.calls.append(name)

    def usb_strings(self, index: int) -> tuple[str, str, str]:
        self._record("usb_strings")
        return (self._manufacturer, self._product, "00000001")

    def device_name(self, index: int) -> str:
        return "Generic RTL2832U OEM"

    def device_count(self) -> int:
        return 1

    def open(self, index: int) -> object:
        self._record("open")
        return object()

    def close(self, handle: object) -> None:
        self._record("close")
        self.closed += 1

    def tuner_type(self, handle: object) -> TunerType:
        return TunerType.R828D

    def set_sample_rate(self, handle: object, rate_hz: int) -> None:
        self._record("set_sample_rate")
        # A real clock divides 28.8 MHz, so exact rates are luck.
        self.sample_rate_hz = rate_hz - (rate_hz % 8)

    def get_sample_rate(self, handle: object) -> int:
        return self.sample_rate_hz

    def set_center_freq(self, handle: object, freq_hz: int) -> None:
        self._record("set_center_freq")
        self.center_hz = freq_hz - (freq_hz % 100)

    def get_center_freq(self, handle: object) -> int:
        return self.center_hz

    def set_freq_correction(self, handle: object, ppm: int) -> None:
        self._record("set_freq_correction")
        self.ppm = ppm

    def get_freq_correction(self, handle: object) -> int:
        return self.ppm

    def set_tuner_gain_mode(self, handle: object, manual: bool) -> None:
        self._record("set_tuner_gain_mode")
        self.manual = manual
        if not manual:
            self.gain_db = 0.0

    def set_tuner_gain(self, handle: object, gain_db: float) -> None:
        self._record("set_tuner_gain")
        if gain_db not in self._gains:
            raise AssertionError(f"{gain_db} is not a step this tuner offers")
        self.gain_db = gain_db

    def get_tuner_gain(self, handle: object) -> float:
        return self.gain_db

    def tuner_gains(self, handle: object) -> tuple[float, ...]:
        return self._gains

    def set_agc_mode(self, handle: object, enabled: bool) -> None:
        self._record("set_agc_mode")

    def reset_buffer(self, handle: object) -> None:
        self._record("reset_buffer")

    def read_sync(self, handle: object, length: int) -> bytes:
        self._record("read_sync")
        return self.read_returns or bytes(length)


def a_config(**overrides: object) -> SdrConfig:
    values: dict[str, object] = {
        "center_hz": 99_650_000.0,
        "sample_rate_hz": 2_048_000.0,
        "gain_db": 32.8,
    }
    values.update(overrides)
    return SdrConfig(**values)  # type: ignore[arg-type]


@pytest.fixture
def lib() -> FakeLib:
    return FakeLib()


@pytest.fixture
def sdr(lib: FakeLib) -> RtlSdr:
    device = RtlSdr(_lib=lib)  # type: ignore[arg-type]
    device.open()
    return device


class TestLifecycle:
    def test_starts_closed(self, lib):
        assert not RtlSdr(_lib=lib).is_open  # type: ignore[arg-type]

    def test_open_reports_the_device(self, sdr):
        assert sdr.is_open
        assert sdr.info is not None
        assert sdr.info.is_blog_v4

    def test_open_twice_does_not_open_twice(self, lib, sdr):
        sdr.open()

        assert lib.calls.count("open") == 1

    def test_close_releases_the_handle(self, lib, sdr):
        sdr.close()

        assert not sdr.is_open
        assert lib.closed == 1

    def test_close_is_safe_to_repeat(self, lib, sdr):
        sdr.close()
        sdr.close()

        assert lib.closed == 1

    def test_close_forgets_the_applied_settings(self, sdr):
        sdr.configure(a_config())
        sdr.close()

        assert sdr.applied is None

    def test_works_as_a_context_manager(self, lib):
        with RtlSdr(_lib=lib) as device:  # type: ignore[arg-type]
            assert device.is_open

        assert lib.closed == 1

    def test_the_context_manager_closes_after_an_error(self, lib):
        with pytest.raises(RuntimeError), RtlSdr(_lib=lib):  # type: ignore[arg-type]
            raise RuntimeError("boom")

        assert lib.closed == 1

    def test_reading_before_opening_is_refused(self, lib):
        with pytest.raises(DeviceError, match="No device is open"):
            RtlSdr(_lib=lib).read_raw(1024)  # type: ignore[arg-type]

    def test_configuring_before_opening_is_refused(self, lib):
        with pytest.raises(DeviceError, match="No device is open"):
            RtlSdr(_lib=lib).configure(a_config())  # type: ignore[arg-type]


class TestDriverMismatch:
    def test_a_v4_on_a_non_blog_library_is_refused(self):
        lib = FakeLib(exports_dithering=True)

        with pytest.raises(DriverMismatchError, match="rtlsdr_set_dithering"):
            RtlSdr(_lib=lib).open()  # type: ignore[arg-type]

    def test_the_refusal_explains_why_it_is_not_a_warning(self):
        lib = FakeLib(exports_dithering=True)

        with pytest.raises(DriverMismatchError, match="tuning somewhere other than"):
            RtlSdr(_lib=lib).open()  # type: ignore[arg-type]

    def test_it_refuses_before_opening_the_device(self):
        lib = FakeLib(exports_dithering=True)

        with pytest.raises(DriverMismatchError):
            RtlSdr(_lib=lib).open()  # type: ignore[arg-type]

        assert "open" not in lib.calls

    def test_a_non_v4_on_a_non_blog_library_is_allowed(self):
        # A stock librtlsdr driving a stock dongle is a perfectly normal
        # setup. The guard is narrow on purpose.
        lib = FakeLib(manufacturer="Realtek", product="RTL2838UHIDIR", exports_dithering=True)

        assert RtlSdr(_lib=lib).open().is_blog_v4 is False  # type: ignore[arg-type]

    def test_a_v4_on_a_blog_library_is_allowed(self, sdr):
        assert sdr.is_open


class TestConfigure:
    def test_applies_the_settings_in_the_order_the_hardware_needs(self, lib, sdr):
        lib.calls.clear()
        sdr.configure(a_config())

        assert lib.calls == [
            "set_sample_rate",
            "set_freq_correction",
            "set_center_freq",
            "set_tuner_gain_mode",
            "set_tuner_gain",
            "set_agc_mode",
            "reset_buffer",
        ]

    def test_the_ppm_correction_is_applied_before_the_frequency(self, lib, sdr):
        # Setting ppm re-applies the current frequency internally, so
        # doing it afterwards silently re-tunes off the previous one.
        lib.calls.clear()
        sdr.configure(a_config(ppm=12))

        assert lib.calls.index("set_freq_correction") < lib.calls.index("set_center_freq")

    def test_the_gain_mode_is_set_before_the_gain(self, lib, sdr):
        # A gain value is ignored while the tuner is in automatic mode.
        lib.calls.clear()
        sdr.configure(a_config())

        assert lib.calls.index("set_tuner_gain_mode") < lib.calls.index("set_tuner_gain")

    def test_the_buffer_is_reset_last(self, lib, sdr):
        # Anything captured at the old settings has to be discarded, and
        # it has to happen after the last setting changes.
        lib.calls.clear()
        sdr.configure(a_config())

        assert lib.calls[-1] == "reset_buffer"

    def test_reports_the_actual_centre_not_the_requested_one(self, sdr):
        applied = sdr.configure(a_config(center_hz=99_650_050))

        assert applied.center_hz == 99_650_000.0
        assert applied.center_error_hz == -50.0

    def test_reports_the_actual_sample_rate(self, sdr):
        applied = sdr.configure(a_config(sample_rate_hz=2_048_004))

        assert applied.sample_rate_hz == 2_048_000.0
        assert applied.sample_rate_error_hz == -4.0

    def test_keeps_the_requested_config_alongside_the_actual(self, sdr):
        config = a_config()
        applied = sdr.configure(config)

        assert applied.requested is config

    def test_offset_of_a_station_above_centre_is_positive(self, sdr):
        # First light's arrangement: centred 250 kHz below 99.9 MHz, so
        # the station must appear at +250 kHz. Getting this sign
        # backwards is the easiest available mistake.
        applied = sdr.configure(a_config(center_hz=99_650_000))

        assert applied.offset_from(99_900_000) == pytest.approx(250_000.0)

    def test_offset_of_a_station_below_centre_is_negative(self, sdr):
        applied = sdr.configure(a_config(center_hz=99_650_000))

        assert applied.offset_from(99_500_000) == pytest.approx(-150_000.0)

    def test_the_offset_is_measured_from_where_the_tuner_landed(self, sdr):
        # Why this method is on AppliedSettings and not on SdrConfig.
        # The fake quantises to 100 Hz exactly as a real PLL does, so
        # asking for 99,650,050 lands on 99,650,000 - and an offset
        # measured from the request is wrong by that difference, in a
        # direction nothing downstream can detect.
        applied = sdr.configure(a_config(center_hz=99_650_050))

        assert applied.offset_from(99_900_000) == pytest.approx(250_000.0)
        assert applied.offset_from(99_900_000) != pytest.approx(249_950.0)

    def test_remembers_the_last_applied_settings(self, sdr):
        applied = sdr.configure(a_config())

        assert sdr.applied is applied

    def test_a_manual_gain_is_snapped_to_a_real_step(self, sdr):
        # The fake device rejects a gain that is not on its table, the
        # way the real one does, so an unsnapped value would raise here.
        applied = sdr.configure(a_config(gain_db=33.0))

        assert applied.gain_db == 32.8
        assert applied.manual_gain

    def test_auto_gain_never_sets_a_gain_value(self, lib, sdr):
        lib.calls.clear()
        sdr.configure(a_config(gain_db=AUTO_GAIN))

        assert "set_tuner_gain" not in lib.calls

    def test_auto_gain_is_reported_as_such(self, sdr):
        applied = sdr.configure(a_config(gain_db=AUTO_GAIN))

        assert not applied.manual_gain

    def test_auto_gain_trips_the_zero_gain_warning(self, sdr):
        # The bring-up trap in executable form: automatic mode reported
        # 0.0 dB and captured nothing, with no error anywhere.
        applied = sdr.configure(a_config(gain_db=AUTO_GAIN))

        assert applied.reports_zero_gain

    def test_a_working_manual_gain_does_not_trip_it(self, sdr):
        assert not sdr.configure(a_config()).reports_zero_gain

    def test_can_be_reconfigured_mid_session(self, lib, sdr):
        sdr.configure(a_config(center_hz=99_650_000))
        applied = sdr.configure(a_config(center_hz=162_300_000, gain_db=49.6))

        assert applied.center_hz == 162_300_000.0
        assert applied.gain_db == 49.6

    def test_a_retune_resets_the_buffer_again(self, lib, sdr):
        sdr.configure(a_config())
        lib.calls.clear()
        sdr.configure(a_config(center_hz=162_300_000))

        assert lib.calls.count("reset_buffer") == 1

    def test_supported_gains_come_from_the_device(self, sdr):
        assert sdr.supported_gains_db() == V4_GAIN_STEPS_DB


class TestReading:
    def test_returns_what_the_device_produced(self, lib, sdr):
        lib.read_returns = bytes(range(256)) * 4

        assert sdr.read_raw(1024) == bytes(range(256)) * 4

    def test_reads_the_default_block_size_when_not_told(self, lib, sdr):
        assert len(sdr.read_raw()) == 262_144


class TestDeviceInfo:
    def test_identifies_a_blog_v4_from_its_eeprom_strings(self):
        info = DeviceInfo(0, "Generic RTL2832U OEM", "RTLSDRBlog", "Blog V4", "", TunerType.R828D)

        assert info.is_blog_v4

    def test_the_generic_name_does_not_identify_a_v4(self):
        # Every dongle reports the same name; V4-ness lives in the USB
        # strings alone.
        info = DeviceInfo(
            0, "Generic RTL2832U OEM", "Realtek", "RTL2838UHIDIR", "", TunerType.R828D
        )

        assert not info.is_blog_v4

    def test_describe_names_the_v4(self):
        info = DeviceInfo(0, "Generic RTL2832U OEM", "RTLSDRBlog", "Blog V4", "", TunerType.R828D)

        assert "RTL-SDR Blog V4" in info.describe()
        assert "R828D" in info.describe()

    def test_describe_includes_a_serial_when_there_is_one(self):
        info = DeviceInfo(1, "x", "RTLSDRBlog", "Blog V4", "00000001", TunerType.R828D)

        assert "serial 00000001" in info.describe()

    def test_describe_omits_an_empty_serial(self):
        info = DeviceInfo(1, "x", "RTLSDRBlog", "Blog V4", "", TunerType.R828D)

        assert "serial" not in info.describe()

    def test_describe_falls_back_to_the_device_name(self):
        info = DeviceInfo(
            0, "Generic RTL2832U OEM", "Realtek", "RTL2838UHIDIR", "", TunerType.R820T
        )

        assert "Generic RTL2832U OEM" in info.describe()


# ---------------------------------------------------------------------------
# Addressing a device by serial
# ---------------------------------------------------------------------------


class FakeBus:
    """A stand-in library that only knows how to enumerate.

    Separate from :class:`FakeLib` above, which models one device and
    records the call order through it. This one models *several*, and
    none of them open — which is the whole claim ``attached_devices``
    makes about itself.
    """

    def __init__(self, devices: tuple[tuple[str, str, str], ...] = (), unreadable: int = -1):
        # Each entry is (manufacturer, product, serial).
        self._devices = devices
        self._unreadable = unreadable
        self.opened: list[int] = []

    def device_count(self) -> int:
        return len(self._devices)

    def device_name(self, index: int) -> str:
        return "Generic RTL2832U OEM"

    def usb_strings(self, index: int) -> tuple[str, str, str]:
        if index == self._unreadable:
            raise DeviceNotFoundError(f"No RTL-SDR at index {index}")
        return self._devices[index]

    def open(self, index: int) -> object:
        self.opened.append(index)
        return object()


V4 = ("RTLSDRBlog", "Blog V4")


def bus(*serials: str, unreadable: int = -1) -> FakeBus:
    return FakeBus(tuple((*V4, serial) for serial in serials), unreadable=unreadable)


class TestAttachedDevices:
    def test_it_reports_one_record_per_device_in_enumeration_order(self):
        found = attached_devices(bus("LEFT", "RIGHT"))

        assert [device.index for device in found] == [0, 1]
        assert [device.serial for device in found] == ["LEFT", "RIGHT"]

    def test_an_empty_bus_is_an_empty_tuple_not_an_error(self):
        assert attached_devices(bus()) == ()

    def test_it_opens_nothing(self):
        # The reason this function exists rather than reusing
        # DeviceInfo: enumerating must not claim a device another
        # branch is already streaming from.
        enumeration = bus("LEFT", "RIGHT")

        attached_devices(enumeration)

        assert enumeration.opened == []

    def test_a_device_whose_strings_will_not_read_is_still_listed(self):
        # Counted by the library but unreadable — an unbound WinUSB
        # driver does exactly this. Dropping it would turn "one device
        # would not answer" into "no device has that serial".
        found = attached_devices(bus("LEFT", "RIGHT", unreadable=0))

        assert len(found) == 2
        assert found[0].serial == ""
        assert found[1].serial == "RIGHT"

    def test_describe_names_the_v4_and_its_serial(self):
        described = attached_devices(bus("LEFT"))[0].describe()

        assert "RTL-SDR Blog V4" in described
        assert "serial LEFT" in described

    def test_describe_says_blank_rather_than_leaving_a_gap(self):
        described = attached_devices(bus(""))[0].describe()

        assert "(blank)" in described


class TestIndexForSerial:
    def test_it_finds_the_device_carrying_that_serial(self):
        assert index_for_serial(bus("LEFT", "RIGHT"), "RIGHT") == 1

    def test_it_does_not_settle_for_the_first_device(self):
        # The failure this whole feature exists to prevent: ignoring the
        # serial and opening index 0 looks correct whenever the wanted
        # device happens to be first, so the test asks for the one that
        # is not.
        assert index_for_serial(bus("RIGHT", "LEFT"), "LEFT") == 1

    def test_matching_is_exact(self):
        with pytest.raises(DeviceNotFoundError):
            index_for_serial(bus("LEFT"), "left")

    def test_an_unknown_serial_lists_what_is_actually_attached(self):
        with pytest.raises(DeviceNotFoundError) as error:
            index_for_serial(bus("LEFT", "RIGHT"), "MIDDLE")

        message = str(error.value)
        assert "LEFT" in message
        assert "RIGHT" in message

    def test_an_empty_bus_says_so_rather_than_listing_nothing(self):
        with pytest.raises(DeviceNotFoundError, match="no devices at all"):
            index_for_serial(bus(), "LEFT")

    def test_two_devices_with_one_serial_is_refused_not_guessed(self):
        # Both V4s ship as 00000001. Picking the first match would hand
        # back whichever enumerated first, which is the unstable thing
        # addressing by serial exists to escape.
        with pytest.raises(AmbiguousDeviceError) as error:
            index_for_serial(bus("00000001", "00000001"), "00000001")

        assert "rtl_eeprom" in str(error.value)

    def test_the_ambiguous_message_names_both_indices(self):
        with pytest.raises(AmbiguousDeviceError) as error:
            index_for_serial(bus("00000001", "LEFT", "00000001"), "00000001")

        message = str(error.value)
        assert "index 0" in message
        assert "index 2" in message

    def test_a_blank_serial_can_still_be_ambiguous(self):
        # Most dongles report a blank serial, so this is the common
        # shape of the mistake rather than an edge case.
        with pytest.raises(AmbiguousDeviceError):
            index_for_serial(bus("", ""), "")
