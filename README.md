# BCG Sleep Sensor
Contactless BCG-based sensor for sleep monitoring. Uses a PVDF piezoelectric film, a charge-amp analog front-end, and a Python DSP pipeline to extract heart rate (HR), respiratory rate (RR), and sleep stage (Wake/REM/Light NREM/Deep NREM) from oscilloscope-captured data.
The Piezoelectric strip is placed underneath a mattress/mattress topper (depending on thickness of material) and picks up on tiny body recoil due to each heartbeat (known as Ballistocardiography, or BCG). It requires zero skin contact and does not have to be worn.

**Status:** Heart Rate (HR) has been checked against a reference (Apple Watch, with a built in PPG) on an 18-minute blind sleep test. Autocorrelation on the raw signal matches the watch to within 2 bpm; the template beat detector in this repo still reads about 8 bpm high on the same recording. Respiratory Rate (RR) is robust. Sleep staging works, but is unvalidated as the user was not wearing any smart appliances while sleeping. HRV (RMSSD) was explored in depth, but in spite of multiple approaches and algorithms, I was not able to get accurate HRV readings. This was due to hardware limitations, I will go into detail regarding this in [`docs/results.md`](docs/results.md).

## How it works:
```
PVDF sensor -> Charge amplifier (HPF ~0.17 Hz) -> Gain stage -> Low-Pass filter (~34 Hz) -> ADC protection -> Python DSP
```

The sensor is treated as a **charge source**, to avoid accounting for its own capacitance shifting due to temperature changes. This also makes it so the circuit does not have to be altered if a replacement piezoelectric strip with its own internal capacitance is used. The charge generated is forced onto a feedback capacitor and voltage spikes are calculated across this capacitor.

## Results:

| Metric | Result | Notes |
|---|---|---|
| Heart rate (autocorrelation) | 59.6 bpm | Apple Watch read 58 bpm on the same blind sleep test |
| Heart rate (template detector) | 66.2 bpm | Same recording. Reads high; see [`docs/results.md`](docs/results.md) |
| Respiratory rate | ~14-15 breaths/min | Robust across a full overnight recording |
| 50 Hz mains rejection | ~10,000× | After adding a grounded Faraday shield around the sensor |
| Sleep staging | Wake/REM/Light/Deep | Working, but **not yet validated** against a reference standard |
| HRV (RMSSD) | 85-170 ms measured | Physiological range is 20-70 ms. Detection-limited, see below. |

The pipeline is also tested against synthetic recordings where every heartbeat time is known. Full numbers and methodology in [`docs/results.md`](docs/results.md).

### The open problem: HRV

RMSSD (Root Mean Square of Successive Differences) needs beat timing accuracy to the millisecond. The cardiac complex measured is a real, repeatable ~100ms bump. The real issue is mechanical smearing: the mattress and foam between body and sensor act as a mechanical low-pass filter, rounding the sharp heartbeat impulse before it reaches the film. What arrives is a broad hump with multiple small jitters near the true J-peak, and the template matcher cannot reliably pick which jitter is the real one. Widening the analog bandwidth produced no improvement, confirming the sharpness was lost mechanically, not electronically.

The synthetic test puts a number on this. At the noise level of the real recordings, beat timing scatters by about 14 ms even on correctly detected beats, which sets an RMSSD error floor of about 35 ms. That is the same size as the HRV being measured, so no detection algorithm can recover it from this signal.

## Repo Structure


```
hardware/
  kicad/                 KiCad schematic source
  schematic-printable/   exported schematic (PDF)
  BOM.md                 as-built bill of materials, with substitution reasoning
software/
  run_overnight.py       main overnight processing pipeline
  bcg_paalasmaa.py       template-matched beat detection (HRV attempt)
  sleep_staging.py       Wake/REM/Light/Deep classifier
  make_synthetic.py      builds synthetic recordings with known heartbeat times
  evaluate_synthetic.py  scores the pipeline against those recordings
docs/
  design-decisions.md    charge-amp vs voltage-mode, component selection method
  debugging-log.md       bugs that cost real time, and how they were found
  results.md             full metrics and limitations
sample-data/
  sample_capture.csv     40 s example capture
requirements.txt
```

## Running the DSP pipeline

```bash
pip install -r requirements.txt
python software/run_overnight.py sample-data/sample_capture.csv
```

The sample rate is read from the file. To run the synthetic test, from inside `software/`:

```bash
python make_synthetic.py
python evaluate_synthetic.py
```

## Hardware summary

- **Sensor:** PVDF piezoelectric film (~2 nF), under-mattress placement
- **Front end:** charge amplifier → ×69 gain → low-pass filter → ADC guard, single MCP602 dual op-amp
- **Reference rail:** 1.65 V, buffered with an OPA333 precision op-amp
- **Shielding:** grounded conductive Faraday cage around the sensor, cut 50 Hz pickup ~10,000×
- **Capture:** PicoScope 4262 (16-bit, 2500 Hz) for overnight recordings; Saleae logic analyzer analog input (1250 Hz) for short captures, including the sample file

NOTE: The ADC guardrail was not implemented during testing as output was directly streamed into an Oscilloscope (PicoScope 4262)

Full schematic and BOM in [`hardware/`](hardware/).

## License

MIT, see [LICENSE](LICENSE).
