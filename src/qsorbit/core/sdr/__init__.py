"""SDR device module.

An RTL-SDR, driven through a hand-written ``ctypes`` binding to the
RTL-SDR Blog ``librtlsdr``. The layering mirrors the rotor module
exactly, because the problem has the same shape — a thin wrapper over
something foreign, a facade that owns lifetime and ordering, and
stateless helpers beside them:

===========================  ==========================================
``rotor``                    ``sdr``
===========================  ==========================================
:mod:`~qsorbit.core.rotor.serial_port`   :mod:`~qsorbit.core.sdr.librtlsdr`
:mod:`~qsorbit.core.rotor.controller`    :mod:`~qsorbit.core.sdr.device`
:mod:`~qsorbit.core.rotor.capabilities`  :mod:`~qsorbit.core.sdr.config`
===========================  ==========================================

The module is named for what it does rather than for the library it
binds, the same way the rotor's protocol module is named ``satnogs``
rather than ``easycomm``: a second SDR family would arrive as another
backend here, not as a rename.
"""

from qsorbit.core.sdr.capture import (
    IQ_FORMAT_DESCRIPTION,
    SIDECAR_VERSION,
    CaptureResult,
    capture_to_file,
)
from qsorbit.core.sdr.config import (
    AUTO_GAIN,
    MAX_PPM,
    RELIABLE_MAX_SAMPLE_RATE_HZ,
    SAMPLE_RATE_WINDOWS_HZ,
    AutoGain,
    SdrConfig,
    nearest_gain_step,
)
from qsorbit.core.sdr.device import (
    BLOG_V4_MANUFACTURER,
    BLOG_V4_PRODUCT,
    DEFAULT_READ_BYTES,
    AppliedSettings,
    AttachedDevice,
    DeviceInfo,
    RtlSdr,
    attached_devices,
    index_for_serial,
)
from qsorbit.core.sdr.exceptions import (
    AmbiguousDeviceError,
    DeviceError,
    DeviceNotFoundError,
    DriverError,
    DriverMismatchError,
    SdrError,
)
from qsorbit.core.sdr.librtlsdr import (
    READ_BLOCK_MULTIPLE,
    DeviceHandle,
    LibRtlSdr,
    TunerType,
    load_library,
    register_driver_directory,
)
from qsorbit.core.sdr.stream import (
    BYTES_PER_SAMPLE,
    DEFAULT_POLL_S,
    DEFAULT_QUEUE_BLOCKS,
    DEFAULT_SUBSCRIBER_NAME,
    STALL_THRESHOLD_S,
    IqStream,
    IqSubscription,
    LossReport,
    StreamStats,
    SubscriberStats,
    ThroughputMonitor,
    TimedBlock,
    byte_rate_for,
)

__all__ = [
    "AUTO_GAIN",
    "BLOG_V4_MANUFACTURER",
    "BLOG_V4_PRODUCT",
    "BYTES_PER_SAMPLE",
    "DEFAULT_POLL_S",
    "DEFAULT_QUEUE_BLOCKS",
    "DEFAULT_READ_BYTES",
    "DEFAULT_SUBSCRIBER_NAME",
    "IQ_FORMAT_DESCRIPTION",
    "MAX_PPM",
    "READ_BLOCK_MULTIPLE",
    "RELIABLE_MAX_SAMPLE_RATE_HZ",
    "SAMPLE_RATE_WINDOWS_HZ",
    "SIDECAR_VERSION",
    "STALL_THRESHOLD_S",
    "AmbiguousDeviceError",
    "AppliedSettings",
    "AttachedDevice",
    "AutoGain",
    "CaptureResult",
    "DeviceError",
    "DeviceHandle",
    "DeviceInfo",
    "DeviceNotFoundError",
    "DriverError",
    "DriverMismatchError",
    "IqStream",
    "IqSubscription",
    "LibRtlSdr",
    "LossReport",
    "RtlSdr",
    "SdrConfig",
    "SdrError",
    "StreamStats",
    "SubscriberStats",
    "ThroughputMonitor",
    "TimedBlock",
    "TunerType",
    "attached_devices",
    "byte_rate_for",
    "capture_to_file",
    "index_for_serial",
    "load_library",
    "nearest_gain_step",
    "register_driver_directory",
]
