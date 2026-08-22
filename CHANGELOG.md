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

<!--
When adding entries, group them under these headings as needed:

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security
-->
