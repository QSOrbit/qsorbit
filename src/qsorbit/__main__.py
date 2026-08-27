"""Command-line entry point, run as ``uv run qsorbit`` (or ``python -m qsorbit``).

Six subcommands: ``point``, ``goto``, ``status``, ``stop``, ``sdr``
(itself split into ``info`` and ``capture``), and ``receive`` — the whole
vertical slice, tracking and receiving a pass together. ``goto`` is
``point``'s raw-axis sibling: it sends AZ/EL numbers you typed, rather
than working them out from a TLE, which is what calibration needs.

**Computing is the default; moving is opt-in.** ``point`` works out where
the rotor would have to go and prints it, without opening the serial port
at all. Only ``--send`` moves anything, and ``receive`` follows the same
rule — it demodulates a whole pass with nothing connected to the serial
port, because Doppler correction needs the TLE and the observer rather
than the rotor. That asymmetry is deliberate — the
controller firmware applies no position limits at any level, so a command
that reaches it is a command it attempts, and the cost of an accidental
slew is measured in cable and gearbox rather than in a retry.

Two things this interface has to be honest about, because both are easy
to misread:

* **Alignment calibration is applied only if your config has one.**
  ``[rotor.alignment]`` defaults to no correction, in which case the
  rotor is commanded to the raw computed sky position and any
  mechanical misalignment of the mast shows up directly as pointing
  error - true of every config file written before Chunk I. On a
  portable rig re-aimed by hand at every setup, that offset needs
  re-measuring and re-entering each time; there is still no
  auto-calibration routine to do the measuring for you.
* **Positions read back from the rotor are axis readings**, measured from
  wherever it homed — not compass bearings, and not sky elevation. They
  are labelled as such wherever they appear.

Output is deliberately plain ASCII. A degree symbol renders fine in most
terminals and then raises ``UnicodeEncodeError`` in the one legacy Windows
code page someone is actually using.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime

from qsorbit import __version__
from qsorbit.core.dsp import (
    DEFAULT_AUDIO_RATE_HZ,
    DEFAULT_CLOSE_BELOW_DB,
    DEFAULT_NBFM_DEVIATION_HZ,
    DEFAULT_NBFM_IF_RATE_HZ,
    DEFAULT_OPEN_ABOVE_DB,
    AudioOutput,
    DopplerTracker,
    NbfmConfig,
    NoiseSquelch,
    SpectrumConfig,
    SpectrumStream,
)
from qsorbit.core.pointing import AlignmentOffset, TrackingLoop, sky_to_rotor
from qsorbit.core.receive import (
    DEFAULT_TRACKING_INTERVAL_S,
    LoopRangeRate,
    ReceiveSession,
    TargetRangeRate,
)
from qsorbit.core.rotor import (
    HomingError,
    Position,
    PositionLimitError,
    Rotor,
    RotorError,
    RotorErrorCode,
    SerialPort,
    format_set_position,
)
from qsorbit.core.sdr import (
    AUTO_GAIN,
    AppliedSettings,
    IqStream,
    RtlSdr,
    SdrConfig,
    SdrError,
    capture_to_file,
)
from qsorbit.core.station import ConfigError, StationConfig, load_station_config
from qsorbit.core.tracker import Satellite, TrackerError

#: How long ``point --send`` waits for the rotor to settle, in seconds.
DEFAULT_ARRIVAL_TIMEOUT_S = 90.0

#: Default capture rate — what all of Phase 2's bring-up used.
DEFAULT_SAMPLE_RATE_HZ = 2_048_000

#: FFT size for ``receive``'s waterfall. Matches what Chunk F's bench
#: runs used, so a trace on screen during a pass is directly comparable
#: with the broadcast-FM runs the display was verified against.
RECEIVE_FFT_SIZE = 2048

#: How far below the signal of interest ``sdr capture`` tunes by default,
#: in kHz. Not a stylistic choice: the RTL-SDR has a permanent DC-offset
#: spike at the centre of its passband, so a signal tuned to exactly the
#: centre lands on top of an artifact and its presence proves nothing.
#: Tuning off-centre is what makes "did we receive it" answerable, and
#: making it the default is what stops it being forgotten.
DEFAULT_TUNING_OFFSET_KHZ = 250.0

#: Printed by ``sdr capture`` when the capture has a hole in it.
NON_CONTIGUOUS_WARNING = (
    "WARNING: blocks were dropped, so this capture is NOT contiguous. The "
    "samples are real but there is a gap in them, which will produce wrong "
    "and plausible-looking results in any analysis. Do not use it as a test "
    "fixture. See the sidecar for the count."
)

#: Printed wherever a commanded position is shown and the station's
#: config carries no alignment offset - which is every config file
#: written before Chunk I, and the honest default after it too, since
#: "uncalibrated" is what [rotor.alignment]'s identity value means.
UNCALIBRATED_NOTE = (
    "No alignment calibration is applied: the rotor is commanded to the raw "
    "computed sky position, so any mechanical misalignment shows up directly "
    "as pointing error."
)


def _alignment_offset(config: StationConfig) -> AlignmentOffset:
    """Build the pointing layer's :class:`AlignmentOffset` from station config.

    A one-line adapter, not a coincidence of naming: :mod:`core.station`
    deliberately does not import :mod:`core.pointing` (see that module's
    own dependency-direction rule), so nothing lower than this entry
    point can build the type :func:`~qsorbit.core.pointing.sky_to_rotor`
    actually wants. Every caller that needs one calls this rather than
    reaching into ``config.alignment`` themselves.
    """
    return AlignmentOffset(
        azimuth_deg=config.alignment.azimuth_deg,
        elevation_deg=config.alignment.elevation_deg,
    )


def _alignment_note(offset: AlignmentOffset) -> str:
    """Describe whether alignment correction is in effect, and what it is.

    :data:`UNCALIBRATED_NOTE` when ``offset`` corrects nothing - the
    default, and every config file written before Chunk I. Otherwise
    names the correction that was actually applied, so an operator who
    measured one sees it confirmed rather than told it doesn't exist -
    the same "say what's actually true" rule ``UNCALIBRATED_NOTE``
    itself exists to follow.
    """
    if offset.is_identity:
        return UNCALIBRATED_NOTE
    return (
        f"Alignment offset applied: AZ {offset.azimuth_deg:+.1f}  EL "
        f"{offset.elevation_deg:+.1f} (from [rotor.alignment] in station config). "
        "Flip mode and azimuth unwrapping are still not applied."
    )


def _squelch_status_line(*, mute: bool, open_above_db: float, close_below_db: float) -> str:
    """Describe the squelch's live configuration, for the startup banner.

    A NoiseSquelch is now built unconditionally (Chunk I: "always
    measure, optionally mute" - see demodulate_nbfm's own ``mute``
    argument and ReceiveSession's ``mute_squelch``) so the live quieting
    readout has something to show even on a run where ``--squelch`` was
    never given. This line is what tells an operator which of those two
    things is actually true here: the gate is always deciding, and
    ``--squelch`` is what lets that decision reach the speaker.
    """
    thresholds = f"open at/above {open_above_db:.1f} dB, close at/below {close_below_db:.1f} dB"
    if mute:
        return f"Squelch:   muting enabled, {thresholds}"
    return f"Squelch:   muting off - always measuring for the live quieting readout, {thresholds}"


def _parse_audio_device(value: str | None) -> int | str | None:
    """Interpret ``--audio-device`` into what ``AudioOutput`` wants.

    ``sounddevice`` itself accepts either a numeric index or a name
    substring for ``OutputStream(device=...)`` - see ``query_devices()``
    for what a string resolves against. The flag is always typed as
    ``str`` by argparse (a name like ``"2"`` and an index ``2`` need to
    stay distinguishable from each other only in *this* function, not in
    two different argparse types), so this is where that split actually
    happens: a purely numeric value becomes an ``int`` index, anything
    else is passed through as a name substring, and an unset flag stays
    ``None`` - PortAudio's own "system default" sentinel.
    """
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _spectrum_factory(
    window: bool, spectrum_config: SpectrumConfig
) -> Callable[[Iterable[bytes]], SpectrumStream] | None:
    """Build the spectrum pipeline's factory, or ``None`` to skip it entirely.

    Handing :class:`~qsorbit.core.receive.ReceiveSession` a factory
    commits to running :class:`~qsorbit.core.dsp.spectrum_stream.SpectrumStream`'s
    worker thread - unpacking IQ and computing FFTs continuously for as
    long as the session runs - so with no window open to show a frame to
    (``window=False``), returning ``None`` means that pipeline never
    starts at all, rather than starting it and discarding everything it
    computes. Measured at 98.2% of computed frames discarded on a
    headless run: this makes the honest headless figure 0%.
    """
    if not window:
        return None
    return lambda blocks: SpectrumStream(blocks, spectrum_config)


RotorFactory = Callable[[StationConfig, Callable[[float], None]], Rotor]
SdrFactory = Callable[[StationConfig], RtlSdr]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured parser. Separate from :func:`main` so tests can
        inspect it without running a command.
    """
    parser = argparse.ArgumentParser(
        prog="qsorbit",
        description="Satellite tracking and rotor control for amateur radio.",
    )
    parser.add_argument("--version", action="version", version=f"QSOrbit {__version__}")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=(
            "Station config file. Defaults to ./qsorbit.toml, then the per-user "
            "config directory. See config.example.toml."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    point = subcommands.add_parser(
        "point",
        help="Work out where to point at a satellite, and optionally go there.",
        description=(
            "Computes the rotor position for a satellite at a given time and "
            "prints it. Nothing is transmitted, and the serial port is not "
            "opened, unless --send is given."
        ),
    )
    point.add_argument(
        "--tle",
        required=True,
        metavar="PATH",
        help="File holding one TLE: two element lines, optionally preceded by a name line.",
    )
    point.add_argument(
        "--at",
        metavar="TIME",
        help=(
            "ISO 8601 instant to compute for, e.g. 2026-08-20T18:30:00+00:00. "
            "Defaults to now. A time with no zone offset is read as UTC."
        ),
    )
    point.add_argument(
        "--send",
        action="store_true",
        help="Actually move the rotor. Without this, nothing is transmitted.",
    )
    point.add_argument(
        "--arrival-timeout",
        type=float,
        default=DEFAULT_ARRIVAL_TIMEOUT_S,
        metavar="SECONDS",
        help=f"How long to wait for the move to settle (default {DEFAULT_ARRIVAL_TIMEOUT_S:.0f}).",
    )

    goto = subcommands.add_parser(
        "goto",
        help="Send the rotor straight to an axis position - no TLE, no satellite.",
        description=(
            "Command the rotor directly to AZ/EL axis coordinates, the same "
            "range-checked and --send-gated way 'point' commands a computed "
            "one. For calibration and manual pointing, where the position "
            "is typed rather than worked out from an orbit."
        ),
    )
    goto.add_argument(
        "--az",
        type=float,
        required=True,
        metavar="DEGREES",
        help="Azimuth axis position to send.",
    )
    goto.add_argument(
        "--el",
        type=float,
        required=True,
        metavar="DEGREES",
        help="Elevation axis position to send.",
    )
    goto.add_argument(
        "--send",
        action="store_true",
        help="Actually move the rotor. Without this, nothing is transmitted.",
    )
    goto.add_argument(
        "--arrival-timeout",
        type=float,
        default=DEFAULT_ARRIVAL_TIMEOUT_S,
        metavar="SECONDS",
        help=f"How long to wait for the move to settle (default {DEFAULT_ARRIVAL_TIMEOUT_S:.0f}).",
    )

    subcommands.add_parser(
        "status",
        help="Read the rotor's firmware version, error state, and position.",
        description=(
            "Read-only. Connects, reports what the rotator says about itself, "
            "and has no way to move anything."
        ),
    )

    subcommands.add_parser(
        "stop",
        help="Halt a converging move by setting the setpoint to the current position.",
        description=(
            "NOT an emergency stop. It halts a move that is converging; it "
            "cannot stop one that is diverging. The power switch is the real stop."
        ),
    )

    _add_sdr_commands(subcommands)
    _add_receive_command(subcommands)
    return parser


def _add_receive_command(subcommands: argparse._SubParsersAction) -> None:
    """Add ``receive``: the whole slice, tracking and receiving together.

    Split out for the same reason the ``sdr`` group is: it has more
    options than everything else here put together, and inlining it
    would bury the commands above it.
    """
    receive = subcommands.add_parser(
        "receive",
        help="Follow a satellite and demodulate its FM downlink.",
        description=(
            "Runs the tracking loop and the receive chain together: the SDR "
            "streams, NBFM demodulates, and the Doppler correction follows the "
            "pass using range rates computed from the TLE. The rotor is NOT "
            "moved unless --send is given, and no window opens unless --window "
            "is. Doppler correction needs the TLE and your location, not the "
            "rotor, so the whole radio job works with nothing on the serial port."
        ),
    )
    receive.add_argument(
        "--tle",
        required=True,
        metavar="PATH",
        help="File holding one TLE: two element lines, optionally preceded by a name line.",
    )
    receive.add_argument(
        "--downlink",
        type=float,
        required=True,
        metavar="MHZ",
        help=(
            "The satellite's nominal downlink in MHz, as transmitted - not the "
            "frequency you expect to hear. Doppler is what makes those differ, "
            "and computing it is this command's job."
        ),
    )
    receive.add_argument(
        "--offset",
        type=float,
        default=DEFAULT_TUNING_OFFSET_KHZ,
        metavar="KHZ",
        help=(
            f"How far below the downlink to place the tuner, in kHz (default "
            f"{DEFAULT_TUNING_OFFSET_KHZ:.0f}). Same reason as 'sdr capture': the "
            "receiver's DC spike sits at the centre of its own passband."
        ),
    )
    receive.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        metavar="SPS",
        help=f"IQ sample rate (default {DEFAULT_SAMPLE_RATE_HZ:,}).",
    )
    receive.add_argument(
        "--if-rate",
        type=float,
        default=DEFAULT_NBFM_IF_RATE_HZ,
        metavar="HZ",
        help=(
            f"Channel-filter output rate (default {DEFAULT_NBFM_IF_RATE_HZ:,.0f}). "
            "Must divide --rate evenly and be more than twice --deviation."
        ),
    )
    receive.add_argument(
        "--audio-rate",
        type=float,
        default=DEFAULT_AUDIO_RATE_HZ,
        metavar="HZ",
        help=f"Playback sample rate (default {DEFAULT_AUDIO_RATE_HZ:,.0f}).",
    )
    receive.add_argument(
        "--deviation",
        type=float,
        default=DEFAULT_NBFM_DEVIATION_HZ,
        metavar="HZ",
        help=(
            f"Transmitter peak deviation (default {DEFAULT_NBFM_DEVIATION_HZ:,.0f}, "
            "which suits amateur FM). Wider modes need both this and --if-rate raised."
        ),
    )
    receive.add_argument(
        "--seconds",
        type=float,
        default=None,
        metavar="S",
        help="Stop automatically after this long. Default: run until Ctrl-C.",
    )
    receive.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_TRACKING_INTERVAL_S,
        metavar="S",
        help=f"Seconds between tracking updates (default {DEFAULT_TRACKING_INTERVAL_S:.0f}).",
    )
    receive.add_argument(
        "--send",
        action="store_true",
        help=(
            "Actually move the rotor to follow the pass. Without this nothing "
            "is transmitted to the controller and the serial port is not opened."
        ),
    )
    receive.add_argument(
        "--window",
        action="store_true",
        help=(
            "Open the instrument window - waterfall and live quieting, plus the "
            "readout when --send is given."
        ),
    )
    receive.add_argument(
        "--squelch",
        action="store_true",
        help=(
            "Mute the noise squelch's closed gate. Off by default: a mute set "
            "slightly too tight makes a working receiver indistinguishable from "
            "a broken one. The gate is always measured and shown live either way "
            "- see --window - this only controls whether it silences audio."
        ),
    )
    receive.add_argument(
        "--audio-device",
        type=str,
        default=None,
        metavar="DEVICE",
        help=(
            "Output device: a numeric index or a name substring, matching "
            "sounddevice's own OutputStream(device=...). Defaults to the system's "
            "configured default. To see what's available, run "
            "'python -c \"import sounddevice as sd; print(sd.query_devices())\"'."
        ),
    )
    receive.add_argument(
        "--squelch-open",
        type=float,
        default=DEFAULT_OPEN_ABOVE_DB,
        metavar="DB",
        help=f"Quieting at/above which the gate opens (default {DEFAULT_OPEN_ABOVE_DB:.1f}).",
    )
    receive.add_argument(
        "--squelch-close",
        type=float,
        default=DEFAULT_CLOSE_BELOW_DB,
        metavar="DB",
        help=f"Quieting at/below which it closes (default {DEFAULT_CLOSE_BELOW_DB:.1f}).",
    )
    gain = receive.add_mutually_exclusive_group(required=True)
    gain.add_argument(
        "--gain",
        type=float,
        metavar="DB",
        help=(
            "Tuner gain in dB. Required, and deliberately has no default: a "
            "default gain is how a pass comes back silent with nobody noticing."
        ),
    )
    gain.add_argument(
        "--auto-gain",
        action="store_true",
        help="Let the tuner choose. Rarely what you want - during bring-up it reported 0.0 dB.",
    )


def _add_sdr_commands(subcommands: argparse._SubParsersAction) -> None:
    """Add the ``sdr`` group: ``info`` and ``capture``.

    Split out because the group has its own nested subcommands and
    inlining it would bury the rotor commands.
    """
    sdr = subcommands.add_parser(
        "sdr",
        help="Inspect the SDR, or capture IQ from it.",
        description="Receiver-side commands. Nothing here moves the rotor.",
    )
    sdr_commands = sdr.add_subparsers(dest="sdr_command", required=True, metavar="COMMAND")

    sdr_commands.add_parser(
        "info",
        help="Report what SDR is attached and what it can do.",
        description=(
            "Read-only. Opens the device, reports how it identifies itself and "
            "which gain steps its tuner offers, and closes it again. The gain "
            "table doubles as a fingerprint of which librtlsdr got loaded."
        ),
    )

    capture = sdr_commands.add_parser(
        "capture",
        help="Record raw IQ to a file, with a JSON sidecar.",
        description=(
            "Captures raw uint8 interleaved I/Q - the same format rtl_sdr.exe "
            "writes - alongside a sidecar recording what the device actually "
            "did. By default the tuner is placed below the signal of interest "
            "rather than on it, because a peak at the centre of the passband "
            "cannot be told apart from the receiver's own DC spike."
        ),
    )
    target = capture.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--station",
        type=float,
        metavar="MHZ",
        help=(
            "Signal of interest in MHz. The tuner is placed --offset kHz below "
            "it, and both the frequency and its offset are recorded in the sidecar."
        ),
    )
    target.add_argument(
        "--center",
        type=float,
        metavar="MHZ",
        help="Tune here directly, in MHz. No signal of interest is recorded.",
    )
    capture.add_argument(
        "--offset",
        type=float,
        default=DEFAULT_TUNING_OFFSET_KHZ,
        metavar="KHZ",
        help=(
            f"How far below --station to tune, in kHz (default "
            f"{DEFAULT_TUNING_OFFSET_KHZ:.0f}). Ignored with --center."
        ),
    )
    capture.add_argument(
        "--seconds", type=float, default=2.0, metavar="S", help="Duration (default 2)."
    )
    capture.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        metavar="SPS",
        help=f"Sample rate in samples per second (default {DEFAULT_SAMPLE_RATE_HZ:,}).",
    )
    gain = capture.add_mutually_exclusive_group(required=True)
    gain.add_argument(
        "--gain",
        type=float,
        metavar="DB",
        help=(
            "Tuner gain in dB, snapped to the nearest step the device offers. "
            "Required, and deliberately has no default: a default gain is how a "
            "capture comes back empty with nobody noticing."
        ),
    )
    gain.add_argument(
        "--auto-gain",
        action="store_true",
        help=(
            "Let the tuner choose. Rarely what you want - during bring-up this "
            "reported 0.0 dB and captured a flat noise floor, with no error."
        ),
    )
    capture.add_argument(
        "--out", required=True, metavar="PATH", help="Where to write the .iq file."
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    rotor_factory: RotorFactory | None = None,
    sdr_factory: SdrFactory | None = None,
) -> int:
    """Run the CLI.

    Args:
        argv: Command-line arguments, defaulting to :data:`sys.argv`.
        rotor_factory: Builds the :class:`~qsorbit.core.rotor.Rotor` from
            a config and a homing-progress callback. Injected by tests so
            no serial port is ever opened.
        sdr_factory: Builds the :class:`~qsorbit.core.sdr.RtlSdr` from a
            config. Injected by tests so no device is ever opened.

    Returns:
        A process exit code: 0 on success, 1 on a failure the operator
        needs to read. Argparse exits with 2 on a usage error.
    """
    args = build_parser().parse_args(argv)
    factory = rotor_factory or _open_rotor

    try:
        config = load_station_config(args.config)
        if args.command == "point":
            return _command_point(args, config, factory)
        if args.command == "goto":
            return _command_goto(args, config, factory)
        if args.command == "status":
            return _command_status(config, factory)
        if args.command == "sdr":
            return _command_sdr(args, config, sdr_factory or _open_sdr)
        if args.command == "receive":
            return _command_receive(args, config, factory, sdr_factory or _open_sdr)
        return _command_stop(config, factory)
    except HomingError as exc:
        # Its own state rather than a generic failure: nothing sent over
        # the link clears it, so "try again" would be the wrong advice.
        print(f"Homing failure: {exc}", file=sys.stderr)
        return 1
    except (ConfigError, RotorError, SdrError, TrackerError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _command_point(args: argparse.Namespace, config: StationConfig, factory: RotorFactory) -> int:
    # Time first: it is the cheapest thing to get wrong, and a typo in
    # --at should not wait behind reading and parsing a TLE.
    when = _parse_time(args.at)
    satellite = Satellite.from_file(args.tle)
    state = satellite.topocentric_state(config.observer, when)
    sky = state.sky_position
    offset = _alignment_offset(config)
    target = sky_to_rotor(sky, offset)

    print(f"Target:    {satellite.name}")
    print(f"Time:      {when.isoformat()}")
    print(f"Sky:       az {sky.azimuth:.1f}  el {sky.elevation:.1f}  (degrees)")
    print(f"Range:     {state.range_km:.0f} km, {_range_description(state.range_rate_km_s)}")
    print(f"Rotor:     {_format_position(target)}  (axis command)")
    print(f"Command:   {bytes(format_set_position(target)).decode('ascii').strip()}")
    print()
    print(_alignment_note(offset))

    if sky.elevation < 0.0:
        print(
            f"The target is below the horizon ({sky.elevation:.1f} degrees), so "
            "it is not observable from here at this time."
        )

    try:
        config.capabilities.check_setpoint(target)
    except PositionLimitError as exc:
        print(f"\nOut of range: {exc}", file=sys.stderr)
        return 1

    if not args.send:
        print("\nNothing was sent. Re-run with --send to move the rotor.")
        return 0

    return _send_and_wait(config, factory, target, args.arrival_timeout)


def _command_goto(args: argparse.Namespace, config: StationConfig, factory: RotorFactory) -> int:
    """``goto``: send the rotor straight to a typed AZ/EL, no TLE involved.

    ``point``'s raw-axis sibling. Everything past "here is the target
    position" is identical between the two - the range check, the
    --send gate, connecting, moving, and reporting arrival - which is
    why both call :func:`_send_and_wait` rather than each doing it
    themselves. What differs is only where ``target`` comes from: a
    TLE and a time there, typed numbers here.
    """
    target = Position(azimuth=args.az, elevation=args.el)

    print(f"Rotor:     {_format_position(target)}  (axis command)")
    print(f"Command:   {bytes(format_set_position(target)).decode('ascii').strip()}")

    try:
        config.capabilities.check_setpoint(target)
    except PositionLimitError as exc:
        print(f"\nOut of range: {exc}", file=sys.stderr)
        return 1

    if not args.send:
        print("\nNothing was sent. Re-run with --send to move the rotor.")
        return 0

    return _send_and_wait(config, factory, target, args.arrival_timeout)


def _send_and_wait(
    config: StationConfig, factory: RotorFactory, target: Position, arrival_timeout_s: float
) -> int:
    """Connect, move to ``target``, wait for arrival, and report the outcome.

    Shared by ``point --send`` and ``goto --send``: once a target
    :class:`~qsorbit.core.rotor.Position` exists, sending it and
    reporting arrival is identical whether that position came from a
    TLE and a time, or was typed directly.
    """
    with _Connected(config, factory) as rotor:
        print(f"\nConnected: {rotor.firmware_version}")
        rotor.move_to(target)
        print(f"Sent:      {_format_position(target)}")
        arrival = rotor.wait_for_arrival(target, timeout_s=arrival_timeout_s)
        if arrival.arrived:
            print(f"Arrived:   {_format_position(arrival.position)} after {arrival.elapsed_s:.1f}s")
            return 0
        print(
            f"Did not settle within {arrival_timeout_s:.0f}s. Last reading "
            f"{_format_position(arrival.position)}, target {_format_position(target)}.",
            file=sys.stderr,
        )
        return 1


def _command_status(config: StationConfig, factory: RotorFactory) -> int:
    with _Connected(config, factory) as rotor:
        status = rotor.status()
        capabilities = config.capabilities

        print(f"QSOrbit {__version__}")
        print(f"Config:    {config.source_path}")
        print(f"Port:      {config.serial.port} at {config.serial.baudrate} baud")
        print(f"Firmware:  {status.firmware_version}")
        if (
            capabilities.firmware_version is not None
            and capabilities.firmware_version != status.firmware_version
        ):
            print(
                f"           Config declares {capabilities.firmware_version}. "
                "QSOrbit was verified against the declared version; behaviour on "
                "this one is untested."
            )
        print(f"Error:     {_describe_error(status.error)}")
        print(f"Position:  {_format_position(status.position)}")
        print(
            "           Axis readings, measured from where the rotor homed - "
            "not compass bearings, and not sky elevation."
        )
        print(
            f"Limits:    AZ {capabilities.azimuth_min_deg:.1f} to "
            f"{capabilities.azimuth_max_deg:.1f}, "
            f"EL {capabilities.elevation_min_deg:.1f} to "
            f"{capabilities.elevation_max_deg:.1f}  (axis travel)"
        )
        print(f"Wrap:      {capabilities.azimuth_wrap.value}")
        print(f"Window:    {capabilities.acceptance_window_deg:.1f} degrees acceptance")
        offset = _alignment_offset(config)
        if offset.is_identity:
            print("Alignment: none recorded (see [rotor.alignment] in config.example.toml)")
        else:
            print(
                f"Alignment: AZ {offset.azimuth_deg:+.1f}  EL {offset.elevation_deg:+.1f}  "
                "(from [rotor.alignment])"
            )
        print()
        print(_alignment_note(offset))
        return 0 if status.healthy else 1


def _command_sdr(args: argparse.Namespace, config: StationConfig, factory: SdrFactory) -> int:
    """Dispatch the ``sdr`` group."""
    if args.sdr_command == "info":
        return _command_sdr_info(config, factory)
    return _command_sdr_capture(args, config, factory)


def _command_sdr_info(config: StationConfig, factory: SdrFactory) -> int:
    with factory(config) as sdr:
        info = sdr.info
        gains = sdr.supported_gains_db()

        print(f"QSOrbit {__version__}")
        print(f"Config:    {config.source_path}")
        print(f"Driver:    {config.sdr.driver_dir or 'system library search path'}")
        print(f"Device:    {info.describe()}")
        print(f"Gains:     {len(gains)} steps, {min(gains)} to {max(gains)} dB")
        print(f"           {' '.join(str(step) for step in gains)}")
        print(f"Ppm:       {config.sdr.ppm}")
        print()
        print(
            "The gain table belongs to the tuner chip, so it doubles as a "
            "fingerprint: a V4's R828D reports 29 steps topping out at 49.6 dB, "
            "and a different table means a different library was loaded than "
            "the one intended."
        )
        return 0


def _command_sdr_capture(
    args: argparse.Namespace, config: StationConfig, factory: SdrFactory
) -> int:
    station_hz = args.station * 1e6 if args.station is not None else None
    if station_hz is not None:
        center_hz = station_hz - args.offset * 1e3
    else:
        center_hz = args.center * 1e6

    sdr_config = SdrConfig(
        center_hz=center_hz,
        sample_rate_hz=args.rate,
        gain_db=AUTO_GAIN if args.auto_gain else args.gain,
        ppm=config.sdr.ppm,
    )
    if sdr_config.may_drop_samples:
        print(
            f"Note: {args.rate:,.0f} sps is above what USB reliably sustains on "
            "many machines. If the capture comes back short, this is the first "
            "thing to look at."
        )

    with factory(config) as sdr:
        print(f"Device:    {sdr.info.describe()}")
        result = capture_to_file(
            sdr,
            sdr_config,
            args.out,
            seconds=args.seconds,
            station_hz=station_hz,
        )

    print(result.describe())
    print(f"Sidecar:   {result.sidecar_path}")
    if result.applied.reports_zero_gain:
        print(
            "\nWARNING: the tuner reports 0.0 dB of gain, which nearly always "
            "means the capture is empty. Set a manual gain."
        )
    if not result.is_contiguous:
        print(f"\n{NON_CONTIGUOUS_WARNING}", file=sys.stderr)
        return 1
    return 0


def _command_receive(
    args: argparse.Namespace,
    config: StationConfig,
    rotor_factory: RotorFactory,
    sdr_factory: SdrFactory,
) -> int:
    """Run the vertical slice: track, stream, demodulate, correct, play.

    Reads in the order that fails cheapest first — the TLE before the
    radio, the radio before the rotor — so a typo in ``--tle`` does not
    wait behind opening a USB device, and a rotor that will not home does
    not cost the SDR configuration that already succeeded.
    """
    satellite = Satellite.from_file(args.tle)
    downlink_hz = args.downlink * 1e6
    center_hz = downlink_hz - args.offset * 1e3

    sdr_config = SdrConfig(
        center_hz=center_hz,
        sample_rate_hz=args.rate,
        gain_db=AUTO_GAIN if args.auto_gain else args.gain,
        ppm=config.sdr.ppm,
    )

    with sdr_factory(config) as sdr:
        applied = sdr.configure(sdr_config)
        print(f"Target:    {satellite.name}")
        print(f"Device:    {sdr.info.describe()}")
        print(
            f"Tuned:     {applied.center_hz / 1e6:.4f} MHz, downlink "
            f"{applied.offset_from(downlink_hz) / 1e3:+.1f} kHz from centre "
            f"(before Doppler)"
        )
        if applied.reports_zero_gain:
            print(
                "\nWARNING: the tuner reports 0.0 dB of gain, which nearly "
                "always means you will hear nothing. Set a manual gain."
            )

        # channel_offset_hz is left at its default here on purpose: the
        # session replaces it on every block with the Doppler-corrected
        # value, and seeding it with a static offset would only invite
        # someone to believe the static one mattered.
        nbfm = NbfmConfig(
            sample_rate_hz=applied.sample_rate_hz,
            if_rate_hz=args.if_rate,
            audio_rate_hz=args.audio_rate,
            deviation_hz=args.deviation,
        )
        print(
            f"Chain:     {applied.sample_rate_hz:,.0f} -> {nbfm.if_rate_hz:,.0f} Hz IF "
            f"(/{nbfm.channel_decimation_factor}) -> {nbfm.audio_rate_hz:,.0f} Hz audio "
            f"(/{nbfm.audio_decimation_factor})"
        )

        # Against the centre the tuner ACTUALLY reached, never the one it
        # was asked for: the PLL quantises, and an offset computed from
        # the requested frequency is wrong by exactly the amount nobody
        # thinks to check.
        doppler = DopplerTracker(downlink_hz, applied.center_hz)
        # Built unconditionally now - see _squelch_status_line's own
        # docstring for why. --squelch only decides whether its decision
        # reaches the speaker (ReceiveSession's mute_squelch below).
        squelch = NoiseSquelch(open_above_db=args.squelch_open, close_below_db=args.squelch_close)
        print(
            _squelch_status_line(
                mute=args.squelch,
                open_above_db=args.squelch_open,
                close_below_db=args.squelch_close,
            )
        )

        if not args.send:
            print(
                "Rotor:     not being moved. Doppler correction needs the TLE and "
                "your location, not the rotor, so this is a complete receive."
            )
            return _run_receive(args, config, satellite, applied, nbfm, doppler, squelch, sdr)

        with _Connected(config, rotor_factory) as rotor:
            print(f"Rotor:     connected, {rotor.firmware_version}")
            loop = TrackingLoop(
                satellite,
                config.observer,
                rotor,
                interval_s=args.interval,
                alignment_offset=_alignment_offset(config),
            )
            return _run_receive(
                args, config, satellite, applied, nbfm, doppler, squelch, sdr, loop=loop
            )


def _run_receive(
    args: argparse.Namespace,
    config: StationConfig,
    satellite: Satellite,
    applied: AppliedSettings,
    nbfm: NbfmConfig,
    doppler: DopplerTracker,
    squelch: NoiseSquelch,
    sdr: RtlSdr,
    *,
    loop: TrackingLoop | None = None,
) -> int:
    """Build and run the session. Split out so the rotor's ``with`` stays thin."""
    stream = IqStream(sdr)
    spectrum_config = SpectrumConfig(
        fft_size=RECEIVE_FFT_SIZE,
        sample_rate_hz=applied.sample_rate_hz,
        center_freq_hz=applied.center_hz,
    )

    session = ReceiveSession(
        stream=stream,
        nbfm=nbfm,
        doppler=doppler,
        audio=AudioOutput(nbfm.audio_rate_hz, device=_parse_audio_device(args.audio_device)),
        range_rate=(
            # With a window, ReadoutWidget ticks the loop on the GUI
            # thread as Chunk F proved; the session follows. Headless,
            # nobody else is ticking, so the session drives.
            LoopRangeRate(loop, drive=not args.window)
            if loop is not None
            else TargetRangeRate(satellite, config.observer)
        ),
        squelch=squelch,
        # args.squelch is "let the gate's decision reach the speaker" -
        # the gate itself is always deciding now, see
        # _squelch_status_line.
        mute_squelch=args.squelch,
        spectrum_factory=_spectrum_factory(args.window, spectrum_config),
        tracking_interval_s=args.interval,
    )

    print(
        "Receiving - Ctrl-C to stop."
        if args.seconds is None
        else f"Receiving for {args.seconds:.0f}s."
    )

    session.start()
    try:
        if args.window:
            _show_instruments(args, satellite, session, loop)
        elif session.wait(args.seconds):
            # The demodulating thread ended before the clock did, which
            # means the blocks stopped arriving. Sleeping out the rest of
            # the run would have hidden that behind a normal-looking
            # exit; session.stop() below raises whatever caused it.
            print("The stream ended early - see the error below.", file=sys.stderr)
    except KeyboardInterrupt:
        print()  # the ^C the terminal echoed deserves its own line
    finally:
        stats = session.stop()

    print()
    print(stats.describe())
    return 0


def _show_instruments(
    args: argparse.Namespace,
    satellite: Satellite,
    session: ReceiveSession,
    loop: TrackingLoop | None,
) -> None:
    """Open the instrument window and run Qt's event loop until it closes.

    **Qt is imported here and nowhere above**, because importing PySide6
    at module scope would make the entire CLI unusable anywhere it is not
    installed. Same reasoning as ``core/dsp/audio.py`` deferring
    ``sounddevice`` and ``core/sdr/librtlsdr.py`` deferring its
    ``CDLL()`` — a dependency that can fail merely by being imported
    belongs inside the function that actually needs it.

    The readout is present only when there is a loop to drive, and when
    it is present **it owns the tick** — which is why the session was
    built with ``drive=False`` in that case. Two things ticking one loop
    would double the rotor's serial traffic.
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from qsorbit.ui.instrument_window import InstrumentWindow
    from qsorbit.ui.quieting_widget import QuietingWidget
    from qsorbit.ui.readout_widget import ReadoutWidget
    from qsorbit.ui.waterfall_widget import WaterfallWidget

    app = QApplication.instance() or QApplication([])
    window = InstrumentWindow(
        readout=(
            ReadoutWidget(loop, poll_interval_ms=int(args.interval * 1000))
            if loop is not None
            else None
        ),
        # Always attached, regardless of --squelch: the session's squelch
        # is now unconditional (see _squelch_status_line), so there is
        # always a live reading to show, muted or not.
        quieting=QuietingWidget(session),
        waterfall=WaterfallWidget(session.spectrum),
        title=f"QSOrbit - receiving {satellite.name}",
    )
    window.show()
    if args.seconds is not None:
        # Honoured rather than ignored: a --seconds that silently did
        # nothing under --window is precisely the sort of quiet
        # disagreement between a flag and its behaviour this project
        # keeps finding the hard way.
        QTimer.singleShot(int(args.seconds * 1000), app.quit)
    app.exec()


def _command_stop(config: StationConfig, factory: RotorFactory) -> int:
    with _Connected(config, factory) as rotor:
        position = rotor.stop()
        print(f"Stopped:   {_format_position(position)}")
        print(
            "The setpoint is now the current position. This halts a converging "
            "move only - it cannot stop a diverging one. Use the power switch "
            "for that."
        )
        return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Connected:
    """Connects a rotor on entry and closes it on exit.

    A thin wrapper rather than ``with Rotor(...)`` directly, so the
    factory stays injectable and homing progress reaches the terminal.
    """

    def __init__(self, config: StationConfig, factory: RotorFactory) -> None:
        self._config = config
        self._factory = factory
        self._rotor: Rotor | None = None

    def __enter__(self) -> Rotor:
        rotor = self._factory(self._config, _report_homing_wait)
        self._rotor = rotor
        rotor.connect()
        return rotor

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._rotor is not None:
            self._rotor.close()


def _open_rotor(config: StationConfig, on_homing_wait: Callable[[float], None]) -> Rotor:
    """Build a :class:`Rotor` over a real serial port."""
    port = SerialPort(
        config.serial.port,
        baudrate=config.serial.baudrate,
        timeout=config.serial.timeout_s,
    )
    return Rotor(port, config.capabilities, on_homing_wait=on_homing_wait)


def _open_sdr(config: StationConfig) -> RtlSdr:
    """Build an :class:`RtlSdr` from the station's ``[sdr]`` settings."""
    return RtlSdr(config.sdr.device_index, driver_dir=config.sdr.driver_dir)


def _report_homing_wait(elapsed_s: float) -> None:
    """Say something while the controller is deaf, so it doesn't look hung."""
    print(f"Waiting for the controller ({elapsed_s:.0f}s) - it is deaf while homing.")


def _parse_time(text: str | None) -> datetime:
    """Parse an ISO 8601 instant, defaulting to now and assuming UTC.

    Args:
        text: The ``--at`` value, or ``None`` for "now".

    Returns:
        A timezone-aware datetime.

    Raises:
        ValueError: If ``text`` isn't a recognizable ISO 8601 instant.
    """
    if text is None:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"Could not read {text!r} as a time. Use ISO 8601, for example "
            "2026-08-20T18:30:00+00:00."
        ) from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _format_position(position: Position) -> str:
    """Format a rotor axis position for display."""
    return f"AZ {position.azimuth:.1f}  EL {position.elevation:.1f}"


def _range_description(range_rate_km_s: float) -> str:
    """Describe a range rate in words, since the sign alone is easy to misread."""
    if range_rate_km_s > 0:
        return f"receding at {range_rate_km_s:.3f} km/s"
    if range_rate_km_s < 0:
        return f"approaching at {abs(range_rate_km_s):.3f} km/s"
    return "range steady"


def _describe_error(error: RotorErrorCode) -> str:
    """Describe an error code, with the operator's next step where there is one."""
    if error is RotorErrorCode.NO_ERROR:
        return "none"
    if error is RotorErrorCode.HOMING_ERROR:
        return "homing failed - power-cycle the controller; nothing over serial clears it"
    return error.name.lower().replace("_", " ")


if __name__ == "__main__":
    sys.exit(main())
