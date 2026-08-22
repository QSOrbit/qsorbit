"""Tests for the ctypes binding to librtlsdr.

These use a stand-in library object rather than a real DLL, so they run
anywhere. That has a real limit worth stating: they prove the binding
calls the right function with the right arguments and maps the return
value correctly, but they cannot prove the ``ctypes`` *signatures* are
right, because nothing here marshals across a C boundary. The signatures
are verified against real hardware by the integration suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qsorbit.core.sdr import (
    READ_BLOCK_MULTIPLE,
    DeviceError,
    DeviceNotFoundError,
    DriverError,
    LibRtlSdr,
    TunerType,
    register_driver_directory,
)
from qsorbit.core.sdr.librtlsdr import NON_BLOG_MARKER_SYMBOL, PPM_UNCHANGED

V4_GAIN_TENTHS = (0, 9, 14, 27, 37, 77, 87, 125, 328, 496)


class FakeSymbol:
    """Stands in for a ctypes function pointer.

    Accepts ``restype``/``argtypes`` assignment the way a real one does,
    records how it was called, and returns whatever it was told to.
    """

    def __init__(self, name: str, owner: FakeLibrary) -> None:
        self.name = name
        self._owner = owner
        self.restype: object = None
        self.argtypes: list[object] = []

    def __call__(self, *args: object) -> object:
        self._owner.calls.append((self.name, args))
        result = self._owner.returns.get(self.name, 0)
        return result(*args) if callable(result) else result


class FakeLibrary:
    """A stand-in for a loaded ``ctypes.CDLL``.

    Args:
        missing: Symbol names to pretend are absent, so the binding's
            "this is not librtlsdr" path can be exercised.
        exports_dithering: Whether to expose the symbol whose presence
            means the library is not the RTL-SDR Blog fork.
    """

    def __init__(self, *, missing: tuple[str, ...] = (), exports_dithering: bool = False) -> None:
        self._missing = set(missing)
        if not exports_dithering:
            self._missing.add(NON_BLOG_MARKER_SYMBOL)
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.returns: dict[str, object] = {}
        self._symbols: dict[str, FakeSymbol] = {}

    def __getattr__(self, name: str) -> FakeSymbol:
        if not name.startswith("rtlsdr_") or name in self._missing:
            raise AttributeError(name)
        if name not in self._symbols:
            self._symbols[name] = FakeSymbol(name, self)
        return self._symbols[name]

    def called(self) -> list[str]:
        """The names of the calls made, in order."""
        return [name for name, _ in self.calls]


@pytest.fixture
def library() -> FakeLibrary:
    return FakeLibrary()


@pytest.fixture
def binding(library: FakeLibrary) -> LibRtlSdr:
    return LibRtlSdr(library)


HANDLE = object()


class TestBinding:
    def test_binds_every_function_it_needs(self, library):
        LibRtlSdr(library)

        # Setting restype on each is what proves it looked them all up.
        assert library.rtlsdr_read_sync.restype is not None
        assert library.rtlsdr_open.argtypes

    def test_a_library_missing_a_symbol_is_refused(self):
        with pytest.raises(DriverError, match="rtlsdr_read_sync"):
            LibRtlSdr(FakeLibrary(missing=("rtlsdr_read_sync",)))

    def test_the_refusal_says_it_is_not_a_usable_librtlsdr(self):
        with pytest.raises(DriverError, match="not a librtlsdr this binding can use"):
            LibRtlSdr(FakeLibrary(missing=("rtlsdr_open",)))


class TestDitheringMarker:
    def test_a_blog_style_library_does_not_export_it(self, binding):
        assert not binding.exports_dithering

    def test_a_stock_style_library_does_export_it(self):
        assert LibRtlSdr(FakeLibrary(exports_dithering=True)).exports_dithering


class TestEnumeration:
    def test_reports_the_device_count(self, library, binding):
        library.returns["rtlsdr_get_device_count"] = 2

        assert binding.device_count() == 2

    def test_decodes_the_device_name(self, library, binding):
        library.returns["rtlsdr_get_device_name"] = b"Generic RTL2832U OEM"

        assert binding.device_name(0) == "Generic RTL2832U OEM"

    def test_a_null_device_name_becomes_an_empty_string(self, library, binding):
        library.returns["rtlsdr_get_device_name"] = None

        assert binding.device_name(0) == ""

    def test_reads_the_usb_strings(self, library, binding):
        def fill(index, manufacturer, product, serial):
            manufacturer.value = b"RTLSDRBlog"
            product.value = b"Blog V4"
            serial.value = b"00000001"
            return 0

        library.returns["rtlsdr_get_device_usb_strings"] = fill

        assert binding.usb_strings(0) == ("RTLSDRBlog", "Blog V4", "00000001")

    def test_missing_device_is_not_found_rather_than_broken(self, library, binding):
        library.returns["rtlsdr_get_device_usb_strings"] = -1

        with pytest.raises(DeviceNotFoundError, match="index 3"):
            binding.usb_strings(3)


class TestOpen:
    def test_an_index_past_the_device_count_is_not_found(self, library, binding):
        library.returns["rtlsdr_get_device_count"] = 1

        with pytest.raises(DeviceNotFoundError, match="sees 1 device"):
            binding.open(1)

    def test_a_device_that_will_not_open_is_a_device_error(self, library, binding):
        library.returns["rtlsdr_get_device_count"] = 1
        library.returns["rtlsdr_open"] = -6

        with pytest.raises(DeviceError, match="could not open"):
            binding.open(0)

    def test_the_open_failure_names_the_likely_culprit(self, library, binding):
        library.returns["rtlsdr_get_device_count"] = 1
        library.returns["rtlsdr_open"] = -6

        with pytest.raises(DeviceError, match="SDR#"):
            binding.open(0)

    def test_closing_a_null_handle_does_nothing(self, library, binding):
        binding.close(None)

        assert "rtlsdr_close" not in library.called()


class TestConfiguration:
    def test_sets_the_sample_rate(self, library, binding):
        binding.set_sample_rate(HANDLE, 2_048_000)

        assert ("rtlsdr_set_sample_rate", (HANDLE, 2_048_000)) in library.calls

    def test_a_rejected_sample_rate_raises(self, library, binding):
        library.returns["rtlsdr_set_sample_rate"] = -1

        with pytest.raises(DeviceError, match="set sample rate"):
            binding.set_sample_rate(HANDLE, 2_048_000)

    def test_reads_back_the_actual_centre_frequency(self, library, binding):
        library.returns["rtlsdr_get_center_freq"] = 99_649_950

        assert binding.get_center_freq(HANDLE) == 99_649_950

    def test_an_unchanged_ppm_is_success_not_failure(self, library, binding):
        # librtlsdr returns -2 for "already that value". Treating it as
        # an error would make ppm=0 on a fresh device fail every time.
        library.returns["rtlsdr_set_freq_correction"] = PPM_UNCHANGED

        binding.set_freq_correction(HANDLE, 0)

    def test_a_genuinely_failed_ppm_still_raises(self, library, binding):
        library.returns["rtlsdr_set_freq_correction"] = -1

        with pytest.raises(DeviceError, match="frequency correction"):
            binding.set_freq_correction(HANDLE, 12)

    def test_manual_gain_mode_sends_one(self, library, binding):
        binding.set_tuner_gain_mode(HANDLE, True)

        assert ("rtlsdr_set_tuner_gain_mode", (HANDLE, 1)) in library.calls

    def test_auto_gain_mode_sends_zero(self, library, binding):
        binding.set_tuner_gain_mode(HANDLE, False)

        assert ("rtlsdr_set_tuner_gain_mode", (HANDLE, 0)) in library.calls

    def test_gain_is_converted_to_tenths_of_a_db(self, library, binding):
        binding.set_tuner_gain(HANDLE, 32.8)

        assert ("rtlsdr_set_tuner_gain", (HANDLE, 328)) in library.calls

    def test_gain_is_converted_back_from_tenths(self, library, binding):
        library.returns["rtlsdr_get_tuner_gain"] = 496

        assert binding.get_tuner_gain(HANDLE) == pytest.approx(49.6)

    def test_reads_the_gain_table_in_two_passes(self, library, binding):
        def gains(handle, buffer):
            if buffer is None:
                return len(V4_GAIN_TENTHS)
            for slot, value in enumerate(V4_GAIN_TENTHS):
                buffer[slot] = value
            return len(V4_GAIN_TENTHS)

        library.returns["rtlsdr_get_tuner_gains"] = gains

        assert binding.tuner_gains(HANDLE) == pytest.approx(
            tuple(value / 10.0 for value in V4_GAIN_TENTHS)
        )

    def test_the_first_gain_pass_asks_with_a_null_pointer(self, library, binding):
        def gains(handle, buffer):
            if buffer is None:
                return len(V4_GAIN_TENTHS)
            for slot, value in enumerate(V4_GAIN_TENTHS):
                buffer[slot] = value
            return len(V4_GAIN_TENTHS)

        library.returns["rtlsdr_get_tuner_gains"] = gains
        binding.tuner_gains(HANDLE)

        first_call = next(call for name, call in library.calls if name == "rtlsdr_get_tuner_gains")
        assert first_call[1] is None

    def test_a_tuner_reporting_no_gains_is_a_device_error(self, library, binding):
        library.returns["rtlsdr_get_tuner_gains"] = 0

        with pytest.raises(DeviceError, match="0 supported gain steps"):
            binding.tuner_gains(HANDLE)

    def test_a_gain_table_that_changes_size_mid_query_is_refused(self, library, binding):
        sizes = iter([4, 3])
        library.returns["rtlsdr_get_tuner_gains"] = lambda handle, buffer: next(sizes)

        with pytest.raises(DeviceError, match="said it had 4"):
            binding.tuner_gains(HANDLE)

    def test_maps_the_tuner_type(self, library, binding):
        library.returns["rtlsdr_get_tuner_type"] = 6

        assert binding.tuner_type(HANDLE) is TunerType.R828D

    def test_an_unrecognised_tuner_type_is_unknown_rather_than_an_error(self, library, binding):
        library.returns["rtlsdr_get_tuner_type"] = 99

        assert binding.tuner_type(HANDLE) is TunerType.UNKNOWN


class TestReading:
    @pytest.mark.parametrize("bad", [0, -512, 1, 1000, READ_BLOCK_MULTIPLE - 1])
    def test_rejects_a_length_that_is_not_a_positive_multiple_of_512(self, binding, bad):
        with pytest.raises(ValueError, match="multiple of 512"):
            binding.read_sync(HANDLE, bad)

    def test_returns_the_bytes_the_device_produced(self, library, binding):
        def read(handle, buffer, length, n_read):
            for index in range(length):
                buffer[index] = index % 256
            n_read._obj.value = length
            return 0

        library.returns["rtlsdr_read_sync"] = read

        assert len(binding.read_sync(HANDLE, 1024)) == 1024

    def test_a_short_read_returns_only_what_arrived(self, library, binding):
        def read(handle, buffer, length, n_read):
            n_read._obj.value = 512
            return 0

        library.returns["rtlsdr_read_sync"] = read

        assert len(binding.read_sync(HANDLE, 1024)) == 512

    def test_a_failed_read_raises(self, library, binding):
        library.returns["rtlsdr_read_sync"] = -8

        with pytest.raises(DeviceError, match="read samples"):
            binding.read_sync(HANDLE, 1024)

    def test_a_read_that_returns_nothing_raises(self, library, binding):
        # Distinct from a failed read: the call succeeded and produced no
        # samples, which means the device is present but not streaming.
        library.returns["rtlsdr_read_sync"] = lambda *args: 0

        with pytest.raises(DeviceError, match="not streaming"):
            binding.read_sync(HANDLE, 1024)

    def test_resets_the_buffer(self, library, binding):
        binding.reset_buffer(HANDLE)

        assert "rtlsdr_reset_buffer" in library.called()


class TestDriverDirectory:
    def test_a_missing_directory_is_a_driver_error(self, tmp_path):
        with pytest.raises(DriverError, match="Driver directory not found"):
            register_driver_directory(tmp_path / "nope")

    def test_the_message_points_at_the_blog_release(self, tmp_path):
        with pytest.raises(DriverError, match="RTL-SDR Blog driver release"):
            register_driver_directory(tmp_path / "nope")

    def test_a_real_directory_is_accepted(self, tmp_path):
        register_driver_directory(tmp_path)

    def test_registering_twice_is_harmless(self, tmp_path):
        register_driver_directory(tmp_path)
        register_driver_directory(tmp_path)

    def test_a_directory_with_no_library_in_it_is_reported(self, tmp_path):
        from qsorbit.core.sdr import load_library

        with pytest.raises(DriverError, match="No librtlsdr found"):
            load_library(tmp_path)

    def test_the_no_library_message_lists_what_it_looked_for(self, tmp_path):
        from qsorbit.core.sdr import load_library

        with pytest.raises(DriverError) as caught:
            load_library(tmp_path)

        assert "Looked for:" in str(caught.value)
        assert str(Path(tmp_path)) in str(caught.value)
