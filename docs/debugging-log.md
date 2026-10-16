# Debugging Log

The bugs that cost real time, written down so nobody loses the same days again. Each entry has the symptom, the cause and the fix.

---

### 1. No output at all, for days

**Symptom:** the circuit appeared dead. Pressing the sensor produced nothing at the output, across many test sessions.

**Cause:** the dual op-amp was wired with the second op-amp's pins mirrored relative to the datasheet. That starves half the chip of supply, puts the rail voltage on a signal input and reverses the feedback path.

**Fix:** correct the chip orientation. The check that finds this kind of fault quickly is to probe every signal node's resting DC voltage. In a healthy circuit every node sits at the mid-rail reference. A node stuck near 0 V or the supply rail points at the broken stage.

---

### 2. A decoupling cap killed the signal

**Symptom:** with the reference buffer not yet fitted, a bulk capacitor was added on the reference rail to stiffen it, and the sensor signal vanished.

**Cause:** the sensor's second terminal connects to the reference rail. A bulk capacitor there gives the sensor's charge a path to ground instead of onto the feedback capacitor, so there's nothing left to amplify.

**Fix:** remove the capacitor and use an op-amp buffer, which stiffens the rail without giving the charge somewhere to go. Never put bulk capacitance directly on the reference rail.

---

### 3. Higher gain gave a smaller spike

**Symptom:** at high gain, a firm press produced a smaller output spike than the same press at unity gain. It was first put down to inconsistent finger pressure.

**Cause:** the resting baseline had drifted well below mid-rail, from op-amp offset amplified by the gain stage plus some leakage at the high-value feedback node. With the operating point off-centre, high gain drove the output into clipping. The pressure explanation didn't hold up.

**Fix:** fit the reference buffer and keep the operating point centred. Probing the charge-amp output before the gain stage showed the underlying signal had been fine all along.

---

### 4. 50 Hz mains hum

**Symptom:** strong 50 Hz hum on the output, worst on a breadboard.

**Cause:** the high-impedance input node acts as an antenna for electric-field noise from mains wiring. Swapping the gain resistor for a larger value made the hum scale with gain, which showed the coupling enters at or before the gain stage.

**Fix:** a grounded conductive shield (Faraday cage) around the sensor, insulated from the electrodes and soldered to circuit ground. A pressure-contact ground brought the hum straight back; soldering fixed it. Mains pickup dropped by about 10,000×, confirmed by FFT.

---

### 5. Heart rate reading too high

**Symptom:** a simple peak-counting detector reported HR about 8-10 bpm above the true value.

**Cause:** the BCG waveform has secondary waves either side of the main peak. A plain peak finder counts them too, so it double-counts within each beat.

**Fix:** estimate the period with autocorrelation or FFT first, and count one peak per period.

---

### 6. Misleading baseline readings

**Symptom:** confusing, inconsistent baseline voltages during early bench testing.

**Cause:** one capture instrument's analog ground was misconnected, and AC-coupled scope views hid the real DC operating point.

**Fix:** a single common ground for every probe, and DC coupling so the true operating point is visible.

---

### 7. Template detector locked onto twice the heart rate

**Symptom:** on real data, two of four 5-minute windows reported about 146 bpm, and the night's summary showed 88 bpm against an Apple Watch reading of 58. On a clean synthetic file with a known HR of 62, detection failed completely.

**Cause:** the detector narrowed its search to ±35% around the peak of the power spectrum. That peak often sat at a harmonic, twice the true rate, so the real interval fell outside the search window. Each 30 s epoch then took the median gap inside it, which came out at a believable-looking 86-91 bpm and passed the trust gate. The failure never showed up as an obviously wrong number.

**Fix:** switch the spectral seed off and set the search window from autocorrelation instead. Found by building synthetic recordings with known beat times; the real data alone would never have shown it.

---

### 8. The new seed locked onto a harmonic too

**Symptom:** the first autocorrelation seed made real-data HR worse (89 bpm), even though autocorrelation on its own had matched the Apple Watch.

**Cause:** it was computed on the detector's preprocessed signal. That preprocessing removes everything below about 1.6 Hz, which takes out the ~1 Hz heartbeat fundamental and leaves only its harmonics.

**Fix:** compute the seed on the raw signal. Real-data HR dropped to 66 bpm and the realistic synthetic file came within 1 bpm of the truth.
