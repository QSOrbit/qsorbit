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
- Doppler-corrected tuning: the receiver now follows a satellite's downlink as it drifts during a pass. The SDR's hardware frequency stays put and the correction is applied digitally within the captured bandwidth, which avoids retuning a radio whose tuner rounds every request and whose own DC artifact moves with it.
- The correction is recomputed for every block of samples rather than once per tracking update, by extrapolating between updates. Holding one value between updates leaves a small step in the audio once a second; recomputing removes it, and costs nothing extra.
- Doppler correction reports the range of correction it applied over a pass, and says plainly when it was working from a stale figure because the tracking loop stopped feeding it — a correction quietly drifting on old data otherwise looks like the receiver wandering off for no reason.
- Separately named functions for downlink and uplink Doppler correction, so a caller picks a direction by name instead of by sign. The uplink one is reserved and raises rather than guessing: its correction is the reciprocal of the downlink's, not its negation, and the two agree closely enough at orbital speeds that a wrong implementation would pass any tolerance anyone thought to write.
- One radio can now feed several parts of the application at once — hearing a satellite while watching its trace on the waterfall no longer means choosing one or the other. Each consumer gets its own buffer, so a display that falls behind drops its own frames and leaves the audio alone.
- Every block of samples now carries the time it arrived, recorded as it comes off the radio rather than worked out later by whatever consumes it. Doppler correction depends on knowing when a block was on the air, and a time derived further downstream is only right for as long as nothing is running late.
- `qsorbit receive`, the command this whole phase has been building towards: it follows a satellite through a pass and plays its FM downlink, tracking the rotor, correcting for Doppler, and showing the spectrum, all at once. Until now each of those existed on its own and could only be exercised one at a time.
- Doppler correction now follows a live pass rather than a recording. The frequency the downlink is expected at is recomputed for every block of samples from the satellite's motion, so a signal that drifts by several kilohertz over a ten-minute pass stays centred in the receiver instead of sliding out of the channel.
- The receiver runs without a rotor connected, and moves one only when asked. Following the Doppler needs the satellite's orbit and your location, not the antenna position, so a rotor that will not connect costs you the antenna pointing and nothing else.
- Audio and the waterfall can now be watched at the same time, from one radio, which is what makes a pass diagnosable while it happens: a downlink is visible as a sloping trace even during the seconds when it is too weak to hear.
- One radio's spectrum can now feed several panels at once, each with its own buffer. A display that falls behind drops its own frames and leaves every other one untouched, which is the same arrangement the raw sample stream has had since the receiver was built.
- `qsorbit plan`, a new command that answers "what's worth pointing at tonight": it checks every TLE you have against a curated catalogue of satellite profiles, filters out anything that never clears your own horizon, and prints what each pass would actually sound like — frequency, mode, and how reliably that transmitter tends to run.
- A curated catalogue of thirteen satellite profiles, shipped with the app: NORAD catalog number, transmitters (frequency, mode, and whether each is an unconditional beacon, a scheduled event, or dependent on another operator being active), and a current alive/inactive status with its source.
- Pass prediction: acquisition of signal, time of closest approach, loss of signal, and the azimuth track between them, for a satellite over a given search window.
- An optional horizon mask in station config (`[[horizon]]`) describing what your own site actually blocks, as a handful of measured azimuth/elevation points. Pass prediction filters out anything that never clears it. A station with no mask sees the plain geometric horizon, same as before this existed.
- A per-pass naked-eye visibility flag, from a closed-form check of whether the satellite is sunlit and the sky is dark enough at the observer to see it.
- Themes, as files. A theme is one small TOML file naming eleven colours and the waterfall's colour ramp, and QSOrbit ships eight: Deep Space, Daylight, Earth, Mars, Luna, Night Ops, LCARS and WOPR. Light and dark are both first-class, because the station operates outdoors in daylight and on into the night.
- Night Ops is red on black throughout, so a visual-pass evening costs no dark adaptation.
- Themes can also change shape and typography, not just colour — border treatment, corner radii and font — which is what LCARS and WOPR are for. Two fonts ship with the app so both look right on a machine that has never seen them.
- Your own theme is the same kind of file, dropped into a `themes/` folder beside your config. Give it the same name as one of the shipped themes and yours is used instead, so "start from Deep Space and change two colours" needs nothing but a copy and an edit.
- A theme can name its author, a description and a URL, so a theme someone shares stays attributable to whoever made it.
- Theme files declare a format version, so a theme written for a later QSOrbit says so plainly instead of failing on a key name that looks like a typo.
- A theme asking for a border-and-typography style this version doesn't have still loads and uses its colours, rather than refusing entirely. It says once that it's drawing in the plain style so a downloaded theme that looks unlike its screenshot isn't a mystery.
- `qsorbit receive --theme` picks the theme for the instrument window.
- Ctrl+T in the instrument window cycles through the installed themes (Ctrl+Shift+T steps back), so a theme can be judged against a live waterfall rather than by relaunching.
- `qsorbit shell`, the application window: Radio, Rotor, Plan and Decode tabs above one live radio, with the theme picker in the top bar and the local and UTC clocks beside it. Every part of it is optional - with nothing attached it opens and each tab says what it is waiting for, which is what an evening with no sky looks like.
- Panels no longer subscribe to anything themselves. A feed hub owns the radio, the rotor and the spectrum, and hands each panel its own independent view - so two waterfalls can watch the same radio without taking frames from each other, which is what will make a custom tab possible.
- A live frequency readout showing where the tracked downlink actually sits, with the Doppler shift under it. The megahertz are shown large and the hertz small, because during a pass the last three digits move continuously and the first six barely at all.
- A tab whose hardware is absent says so in words rather than showing an empty instrument. A panel drawing nothing and a panel whose radio died look identical, and only one of them is a fault.
- A run now reports what the waterfall spent repainting - how many repaints, at what size and rate, and how much of each went into assembling the image versus drawing it to the window - printed beside the receive statistics so the two can be compared. Every other layer already reported its own accounting; the display was the one that did not, and it turned out to matter. **A large waterfall panel costs the receive path samples**: on the bench, a maximized shell lost 0.6% of its USB samples against 0.02% for the same shell windowed. Panel area is what drives it - repainting less often was measured and does not help - so until the mechanism is understood, run the shell windowed during a pass.

### Fixed

- The Rotor tab no longer cuts its own readings off mid-word. With a rotor connected, four of the six rows were clipped by a column that had been given a fixed width - the range row read "39131 km, approaching at 0." with the rate itself missing, which is a readout dropping the number it exists to show. Columns now size to what they are actually holding.

### Changed

- The quieting panel now stacks its bar above its labels instead of running them in a row, so it stays readable in a narrow column. In the old layout its text was cut off mid-word when placed beside the spectrum.
- Cards, panels and headings are drawn from the theme's own tokens throughout the new window, including the LCARS accent bars, which take their colours from whichever accent the active theme sets rather than from a fixed palette. A theme of your own that asks for the LCARS style gets bars in its own key.


### Changed

- Doppler arithmetic moved to its own module so the signal-processing layer can use it without pulling in the satellite-propagation library. Existing imports are unchanged.
- The waterfall's colour ramp is now part of the theme rather than fixed in the code, so switching theme restyles it along with everything else. Every ramp still runs monotonically from dark to bright (or bright to dark, for the light theme), so brighter always means stronger and no band of the scale can look like a signal that is not there.

- The helper that works out where a signal sits within a capture now measures from the frequency the tuner actually reached rather than the one it was asked for. The two differ by however much the tuner rounded, and nothing downstream could have noticed.
- The rotor/sky readout is now a panel rather than a window in its own right, so it can share a window with the waterfall — or run without it. Each panel feeds itself, which means looking at a spectrum no longer needs a rotor connected, and watching the rotor no longer needs an SDR.
- Decimation now preserves whether its input was real or complex, rather than always returning complex samples — needed so the same decimation code can be reused on the real-valued audio signal downstream of the WBFM discriminator, not only on IQ.
- Asking a stream for its blocks twice is now refused rather than quietly answered. It used to appear to work, with each caller receiving roughly every other block and neither having any way to notice.
- Buffer drops are now reported per consumer as well as for the stream as a whole, so a stalled waterfall can be told apart from a stalled audio path instead of both being one number.
- Doppler correction is now safe to share between the part of the application that tracks the satellite and the part that demodulates it, which are no longer the same thread.
- The spectrum pipeline converts only the samples each frame actually needs, rather than converting a whole block of them and discarding about 98% of the result. It was costing roughly 2.3% of a processor core to feed work that costs 0.2%, and that conversion sits on the same path the radio is read on, where time spent turns directly into samples lost.

### Fixed

- The marker showing the receiver's own centre-frequency artifact is no longer always yellow. It follows the theme like everything else, which matters most in Night Ops, where a bright yellow line across the spectrum undid the whole point of a theme designed to preserve dark adaptation.
- The waterfall and the spectrum line trace no longer take frames from each other. With both panels open, one would update while the other sat frozen, and they alternated: each was emptying a buffer they shared, so whichever asked first got everything that had arrived. Each panel now has its own feed and both show every frame.
- Receiving without a window no longer reports a large number of dropped blocks. Nothing was being lost — the spectrum consumer was created whether or not anything would ever look at it, and blocks queued for it were discarded unread — but a 60-second run announcing 453 dropped blocks and 118 MB reads as catastrophic data loss. Off and broken should never look the same.
- The receiver no longer loses about a second of samples every time it opens its window. It was building the window while the radio was already streaming, so standing up the graphics stack starved the reader exactly once per run — which is where essentially all of the "USB loss" this project has been reporting for months actually came from. Measured at 1.0331 seconds before and 0.001 seconds after, over runs from twenty seconds to twenty minutes.
- The window no longer freezes while the rotor is being read. Reading the antenna's position means writing a command, waiting for the RS-485 bus to turn around, and then waiting for a reply, and all of that was happening on the thread that draws — once a second during a pass, and for as long as the port's whole timeout whenever a reply went missing. It now happens on a background thread, and the readout shows what that thread most recently found.
- A rotor that answers slowly can no longer hold up a read indefinitely. The configured timeout was being applied to each byte rather than to the reply as a whole, so a controller trickling characters just faster than the timeout was never actually timing out.
- A reply that arrives incomplete is now treated as a timeout instead of being parsed. A truncated "EL 3." reads as a perfectly plausible 3.0 degrees rather than as the 3.8 it was going to say, and nothing downstream could have noticed.
- A single corrupted reply from the rotor no longer ends a pass. One bad reading used to abort the track, and passes are ten-minute appointments that do not come back. The position is now read a second time, and the track stops only if the second reading agrees that something is wrong — which a transient does not. How often this happened is reported, since the rate is worth knowing.

### Added

- A Custom tab: a grid of widgets built from a config file (`custom_tab.toml`) rather than from code -- the same widgets the built-in tabs already show, in whatever combination and repetition you list, each cell getting its own independent feed. With no file present the tab says so and every other tab works exactly as it would otherwise; a bad file costs only this one tab, with the specific problem named right there rather than only on the console.

### Added

- The curated profile catalogue can now carry a catalogue-level manifest (`CATALOG.toml`, optional, beside the profile files) recording when the curated set itself was last revised, distinct from any one satellite's own alive-status date. `qsorbit plan` prints it when present.
- `qsorbit plan --refresh-catalogue`, for fetching an updated catalogue over the network ahead of planning. No real source is wired up yet -- it fails with a clear, specific error rather than silently using the shipped snapshot -- but the interface any future source will satisfy is in place now, shared with the still-deferred TLE-catalog fetch.

### Added

- The Plan tab now shows a real target picker: a filterable, refreshable table of curated satellites and their next pass, matched against this station's own TLEs and horizon mask exactly as `qsorbit plan` computes them. Four filter axes -- needs a transmitter, band, modulation, and reliability class -- combine as chips above the table; a fifth, "visible from this latitude," is deliberately not here yet and lands in its own follow-up PR with its own geometry. Recompute is a manual Refresh button rather than a timer, since matching TLEs and predicting passes is real CPU work, not a cheap property read.
- An optional `[planning]` section in station config, with a `tle_dir` key naming where the Plan tab should look for this station's TLEs. A station that hasn't set one sees a placeholder naming the config key instead of an empty table with no explanation.
- The target picker's fifth filter axis: "visible from here," a chip that excludes satellites whose orbit can geometrically never rise above this station's horizon at all -- an equatorial bird will never show up for a station at 60 degrees north, no matter how long the search window runs. Whether an orbit reaches a given latitude is computed from its inclination and mean altitude alone (ground-track reach plus flat-horizon footprint radius), a permanent fact about the orbit and the station, not a per-pass prediction.

### Added

- The Plan tab now shows a ground-track map above the target picker, GPredict-style: a ±90-minute track and a current visibility footprint for each satellite the picker's own filters currently leave visible, redrawn the moment a filter chip changes since the map is fed straight from the picker's own selection rather than a timer. Two projections sit behind a flat/globe toggle -- flat is the whole world at once with a straightforward stretch near the poles; globe is centered on this station and shows only the hemisphere it can actually see, which is where a satellite's ground track reads as the curve it geometrically is rather than the sawtooth a flat map draws near the poles. A footprint is drawn as a ring rather than a shaded disc, since the ring degrades correctly to just the pieces that are still connected wherever a track or footprint gets split by the antimeridian or the globe's own horizon; a filled shape would either self-intersect or paper over a gap that was never really there.
- Natural Earth's 1:110m public-domain coastline data, vendored into the app so the map has a world to draw the tracks over without a network fetch. Trimmed to two-decimal-place coordinates -- well below both the data's own resolution and anything a small on-screen map can show a difference for.
- `Satellite.subpoint_at()` and `ground_track()`, giving a satellite's ground-track position at a single instant or across a span of them -- the propagation the map needs and the picker's own pass prediction never did.
- `footprint_circle()`, the ring of points a satellite's current visibility footprint traces on the ground, from the same footprint-radius geometry the "visible from here" filter chip already uses.
- `qsorbit.core.map_projection`: the flat (equirectangular) and globe (orthographic, station-centered) projections themselves, plus the antimeridian- and horizon-aware splitting that turns a projected track into safe-to-draw line segments -- pure geometry, tested the same way the rest of this project's own orbit math is, against hand-derived reference points rather than a rendered picture.
- This PR closes out Chunk D: the target picker now answers "what should I point at right now" from live pass prediction filtered through the real horizon and the curated catalogue, says how stale its data is, and the map shows tracks and footprints for exactly what the picker currently has selected.


<!--
When adding entries, group them under these headings as needed:

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security
-->
