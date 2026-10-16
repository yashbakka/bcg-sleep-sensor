# Results

All numbers below come from recordings in this project or from the synthetic test in `software/`. Where something didn't work, it says so.

## Summary

| Metric | Result | Notes |
|---|---|---|
| Sensor capacitance | ~2.0 nF | Measured with a multimeter. Constant under pressure, which confirms the film behaves as a charge source |
| Heart rate, autocorrelation | 59.6 bpm | 18-minute blind sleep test. Apple Watch read 58 bpm |
| Heart rate, template detector | 66.2 bpm | Same recording. About 8 bpm high (was 87.8 bpm before the fixes listed below) |
| Respiratory rate | ~14-15 breaths/min | Consistent across a full overnight recording. On synthetic data: 14.0 reported vs 14.4 true |
| 50 Hz mains rejection | ~10,000× | After adding the grounded Faraday shield, confirmed by FFT |
| Sleep staging | Wake / REM / Light / Deep | Runs on respiration and movement only. Not validated against a reference |
| RMSSD (HRV) | 85-170 ms | Resting physiological range is 20-70 ms. Detection-limited, see below |

## Heart rate

Two methods are in this project and they behave differently.

**Autocorrelation** (an early analysis script, not included in this repo) estimates the beat period from how the signal repeats, then counts one peak per period. On the blind sleep test it read 59.6 bpm against the watch's 58. Its weakness: it has no way to tell when there is no heartbeat. On a synthetic file with breathing and noise but no heartbeat at all, it still reported 60 bpm.

**Template matching** (`bcg_paalasmaa.py`, the detector in `run_overnight.py`) learns the shape of one heartbeat and accepts an interval only if two copies of that shape reconstruct the signal well. It is meant to give beat-level timing for HRV. On the same recording it reads 66.2 bpm. The remaining error comes from secondary waves inside a beat sometimes being accepted as beats.

## Synthetic test

`make_synthetic.py` builds 15-minute recordings tuned to match the real captures (beat width 102 vs 116 ms, signal-to-noise 2.4 vs 2.7, same noise spectrum), with every true heartbeat time saved. `evaluate_synthetic.py` runs the real pipeline on them and scores it.

HR per 5-minute window, true value in brackets:

| File | Truth | Original code | Current code |
|---|---|---|---|
| easy (clean beats) | 62, 62, 62 | failed on all three | 62, 62, 62 |
| realistic (matches real data) | 65, 69, 62 | 140, 68, 142 | 66, 69, 63 |
| hard (weak beats) | 65, 68, 62 | 151, 130, 151 | no output |
| no heartbeat | none | 9 epochs wrongly trusted | 10 epochs wrongly trusted |

On the realistic file the current code finds 61% of beats and invents 30. On the hard file it now reports nothing rather than a wrong number. The no-heartbeat file still produces false readings: nothing in the pipeline checks whether a heartbeat is present.

## Fixes made to the detector

1. **Spectral seed switched off.** The original code narrowed its beat search around the peak of the power spectrum. That peak often sat at twice the true heart rate, so the search window excluded the real interval and HR came out near 140. This is what caused the 87.8 bpm reading on real data.
2. **Autocorrelation seed added.** The search window is now set from autocorrelation of the raw signal, which found the right period on real data where the spectrum didn't.
3. **Outlier-based acceptance gate.** The old gate rejected a fixed 40% of beat pairs regardless of quality. The new one rejects only pairs that stand out from the rest. On the clean synthetic file, beats found went from 59% to 93% with this change alone, and to 97% with the new seed as well.
4. **Crash fix.** A window with no candidate beats caused a `ValueError` instead of a clean error.

## RMSSD: the open problem

RMSSD needs each heartbeat timed to the millisecond. The cardiac complex here is real and repeatable, but it arrives as a broad ~100 ms hump rather than a sharp spike.

The cause is mechanical. The mattress and foam between body and sensor act as a mechanical low-pass filter and round off the heartbeat impulse before it reaches the film. Widening the analog low-pass filter made no difference, because the sharpness was already gone before the electronics.

The synthetic test measures how much this costs. At the noise level of the real recordings, timing scatters by about 14 ms even on correctly detected beats. That alone sets an RMSSD error floor of about 35 ms, the same size as the true value (34 ms). On the clean synthetic file the scatter drops to 7 ms and reported RMSSD is 38 ms against a true 34, so the method works when the signal is sharp enough. On real data it isn't.

Two possible ways forward, neither tried yet:

1. **Hardware:** a stiffer sensor assembly (film bonded to a rigid carrier) under the upper torso, with less foam in between, to keep the impulse sharp.
2. **Reference data:** a chest strap or ECG recorded at the same time as the BCG. Nothing on real data can be scored beat by beat without one.

## Sleep staging

Staging uses breathing regularity and movement only, following published work that reaches about 84% REM/NREM accuracy on this sensor type without heartbeat features. It runs across a full night and produces a hypnogram. It has not been compared against any reference, so treat its output as experimental.

## Known limitations

- No check for whether a heartbeat is present. A test based on autocorrelation strength was tried and removed: the no-heartbeat file scored the same as real data.
- Movement detection flagged none of the movement bursts in the synthetic files. The synthetic bursts may not look like real movement, so this is unconfirmed either way.
- The detector's preprocessing (from the Paalasmaa paper) removes everything below about 1.6 Hz, including the ~1 Hz heartbeat fundamental. This may be part of why secondary waves get accepted. Not yet tested.
