# Changelog

All notable changes to QSOrbit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `TrackingLoop`, which watches a target continuously and commands the rotor to follow it, re-commanding only once the pointing error exceeds a deadband so it doesn't chatter the antenna in place. Bench-verified tracking the sun on real hardware for 20 minutes with only 2 commands issued.
- Below-horizon targets are now handled as a normal tracking state rather than an error — the loop keeps sampling and simply commands nothing until the target rises.
- A live readout window showing the sky target and the rotor's actual axis position as distinct values, updating roughly once a second while a track runs. Its time row shows the local system clock alongside UTC, so there's no mental time-zone math while watching a pass.
- `rotor_to_sky()`, converting a rotor axis reading into the sky direction it currently means, so the readout can show that alongside the sky target without doing the mod-360 arithmetic by eye.
- An SDR device layer for the RTL-SDR: open a device, tune it, set gain and crystal correction, and read raw IQ from it. Built as a direct binding to the RTL-SDR Blog `librtlsdr`, so it adds no new dependency.
- QSOrbit now refuses to open an RTL-SDR Blog V4 through a driver that cannot correctly tune it, rather than streaming samples from the wrong frequency in silence.
- Tuner gain is snapped to a step the device actually offers, and every setting is read back from the hardware afterwards, so what a capture says it was tuned to is what the radio was really doing.
- An optional `[sdr]` section in station config, for the driver directory, device index, and crystal correction. Existing config files keep working untouched.
- Continuous IQ streaming from the SDR: a reader thread feeds a bounded buffer that a consumer drains, so receiving no longer means one blocking read at a time.
- Streaming reports two kinds of loss separately — samples that never arrived from the device, and blocks discarded because the buffer filled. They have different causes and different fixes, and a single number would send you after the wrong one.
- `qsorbit sdr capture`, which records raw IQ to a file alongside a JSON sidecar describing what the radio actually did. By default it tunes below the signal of interest rather than onto it, because a peak at the centre of the passband can't be told apart from the receiver's own artifact.
- A capture that lost blocks says so in its own metadata and exits non-zero, so a recording with a gap in it can't quietly become a test fixture.
- `qsorbit sdr info`, reporting the attached device and the gain steps its tuner offers.
- Power-spectrum framing for the DSP layer: turn a block of IQ samples into a windowed FFT frame in dB, sized for a waterfall consumer, with a frequency axis reported in absolute Hz so a caller never has to reconstruct the bin-to-frequency mapping itself.
- Integer decimation for IQ samples, chaining automatically into smaller stages for downsampling factors large enough that a single filter design would be numerically unreliable.
- `numpy` and `scipy` as dependencies, for the DSP layer's FFT and filtering.
- WBFM demodulation: a quadrature discriminator recovers audio from a captured or streamed channel, a de-emphasis filter undoes the transmitter's pre-emphasis, and the result is decimated down to an audio rate. Bench-verified against a real FM broadcast capture.
- A digital baseband mixer, needed because this project's own captures (and the tuning convention worth keeping live) deliberately place the station away from the tuner's centre frequency, to dodge the RTL-SDR's permanent DC-offset spike — the discriminator needs the station sitting at 0 Hz, so demodulation shifts it there first.
- Audio output via `sounddevice`: streams demodulated audio to the system's default output device, with the same bounded-buffer, oldest-block-discarded shape the SDR streaming layer uses, facing the speaker instead of the dongle. Reports buffer-full drops and playback underruns separately, since they point at opposite problems.
- `sounddevice` as a dependency, for audio output.
- Spectrum frames are now produced continuously on a background worker, at a rate a display can actually use rather than the rate the radio can produce them. A live stream is capable of about a thousand spectrum rows a second and a screen can show a few dozen, so the frames nobody would ever see are never computed in the first place.
- The spectrum pipeline reports frames it deliberately skipped separately from frames it computed and then had to throw away. The first is the design working as intended; the second means whatever is drawing them has fallen behind, and only one of those is worth chasing.
- A live waterfall panel, showing the received spectrum scrolling in real time beside the rotor/sky readout in the same window. Narrow signals survive being squeezed onto a panel's pixels: where many frequency bins share one pixel the display takes the strongest of them rather than the average, so a satellite's carrier stays visible instead of being blended into the quiet either side of it.
- Both of the waterfall's scales are fixed rather than adapting as it runs. Brightness maps a set dB window, so a signal appearing, fading, or dropping out actually changes what you see instead of being tracked by a scale that moves with it. The time axis is fixed from the first frame for the same reason: a display whose vertical scale is still settling makes a Doppler slope look like it is changing angle when nothing about the signal has.
- A labelled frequency axis under the waterfall, so a trace can be read as a frequency rather than just a position. Tick spacing follows the span and thins out as the window narrows, and the labels are derived from the same settings the frames were computed with, so they cannot drift out of step with what is being drawn.
- Narrowband FM demodulation — the mode the FM satellites, amateur repeaters, and NOAA weather radio all use. Unlike wideband broadcast FM, a narrowband channel is filtered down to a narrow intermediate rate *before* the discriminator sees it, which is what stops a louder signal elsewhere in the capture being recovered instead of the one you asked for.
- The narrowband channel filter rejects the adjacent channel as well as protecting the resample: a neighbouring transmission 25 kHz away — the spacing NOAA weather radio actually uses — would otherwise fold directly on top of the wanted one.
- An optional noise squelch, which mutes the hiss an FM demodulator produces when no carrier is present. It measures how far the noise above the channel sits below full deviation, so it works the same whether the tuner gain is high or low, and it opens and closes at different levels so a marginal signal cannot chatter the audio on and off.
- The squelch is off unless asked for, and reports how long it spent muted along with the range of signal quality it saw. A mute that is set slightly too tight makes a working receiver sound exactly like a broken one, so it has to be possible to tell the two apart without guessing — and the reported range is what a threshold gets tuned from.

### Changed

- The helper that works out where a signal sits within a capture now measures from the frequency the tuner actually reached rather than the one it was asked for. The two differ by however much the tuner rounded, and nothing downstream could have noticed.
- The rotor/sky readout is now a panel rather than a window in its own right, so it can share a window with the waterfall — or run without it. Each panel feeds itself, which means looking at a spectrum no longer needs a rotor connected, and watching the rotor no longer needs an SDR.
- Decimation now preserves whether its input was real or complex, rather than always returning complex samples — needed so the same decimation code can be reused on the real-valued audio signal downstream of the WBFM discriminator, not only on IQ.

<!--
When adding entries, group them under these headings as needed:

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security
-->
