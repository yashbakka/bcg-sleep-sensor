"""
make_synthetic.py

Builds fake under-mattress BCG recordings where every answer is known in
advance: each heartbeat's exact time, the true HR, the true RMSSD, the true
breathing rate, and when the sleeper moved. Run the real pipeline on these
files and compare its output to the truth with evaluate_synthetic.py.

The fake signal is built to look like the real captures (analog*.csv):
  - Beat shape: an I-J-K complex (small dip, big J peak, dip after) that is
    then blurred by a "mattress" low-pass filter, giving the ~100 ms blunt
    hump seen in the real data.
  - Each beat varies a little: size changes with breathing and at random,
    and the I/K waves shift by a few ms, the way real beats do.
  - Beat timing: HR drifts slowly through the recording, speeds up and slows
    down with each breath (respiratory sinus arrhythmia), plus random
    beat-to-beat variation. This is where the known RMSSD comes from.
  - Breathing: ~14 breaths/min, much larger than the heartbeat, and not a
    pure sine wave, so it has harmonics that land in the heart-rate band
    (the same problem the real data has).
  - Noise: colored noise scaled so the beat-to-noise ratio matches what was
    measured on the real files (SNR about 2.7 on analog.csv).
  - Movement: a few bursts of large noise, like rolling over.
  - Saved like the Saleae files: 1250 Hz, rounded to 1 mV steps.

Four files are written by default:
  synth_easy.csv       clean beats (SNR 8), steady HR, no movement
  synth_realistic.csv  matches the real data (SNR 2.7, HR drift, movement)
  synth_hard.csv       weaker beats (SNR 1.5)
  synth_nobeat.csv     same breathing and noise, NO heartbeat at all

Each file gets two truth files next to it:
  <name>_truth_beats.csv    the exact time of every heartbeat
  <name>_truth_epochs.csv   true HR, RMSSD, RR and movement per 30 s epoch

Usage:
  py make_synthetic.py                 (all four files, 15 min each)
  py make_synthetic.py --minutes 30
  py make_synthetic.py --only realistic
"""

import argparse
import csv

import numpy as np
from scipy import signal as sg

FS = 1250.0
EPOCH_S = 30.0

PRESETS = {
    #            snr   hr drift (bpm path)   movement bursts   beats on
    "easy":      dict(snr=8.0, hr_path=(62, 62, 62), n_moves=0, beats=True),
    "realistic": dict(snr=2.7, hr_path=(62, 70, 58), n_moves=3, beats=True),
    "hard":      dict(snr=1.5, hr_path=(62, 70, 58), n_moves=3, beats=True),
    "nobeat":    dict(snr=2.7, hr_path=(62, 70, 58), n_moves=3, beats=False),
}

# HRV settings shared by all presets (true RMSSD comes out around 35-45 ms)
RSA_MS = 30.0         # how far IBI swings with each breath (ms, peak)
RANDOM_IBI_MS = 18.0  # random beat-to-beat variation (ms, std)

# Breathing
RR_BPM = 14.0             # breaths per minute (mean)
RESP_AMP_V = 0.045        # breathing size at the sensor (V, peak); ~150 mV p-p like analog.csv

# Beat
J_AMP_V = 0.036           # J-peak height before blurring (V); ends up ~10 mV in the 2-12 Hz band
MATTRESS_LP_HZ = 5.0      # mechanical blur: lower = blunter peak (5 Hz gives ~100 ms like the real data)


def heartbeat_times(dur_s, hr_path, resp_phase_fn, rng):
    """Return true beat times (s). HR follows hr_path (start, middle, end bpm),
    IBI is modulated by breathing (RSA) plus random variation."""
    times = [0.8]
    ar = 0.0
    while True:
        t = times[-1]
        if t >= dur_s - 1.0:
            break
        frac = t / dur_s
        if frac < 0.5:
            hr = hr_path[0] + (hr_path[1] - hr_path[0]) * (frac / 0.5)
        else:
            hr = hr_path[1] + (hr_path[2] - hr_path[1]) * ((frac - 0.5) / 0.5)
        base_ibi = 60.0 / hr
        rsa = (RSA_MS / 1000.0) * np.sin(resp_phase_fn(t))
        # mildly correlated random variation (AR(1)), like real beat-to-beat noise
        ar = 0.3 * ar + rng.normal(0, RANDOM_IBI_MS / 1000.0)
        times.append(t + base_ibi + rsa + ar)
    return np.array(times[:-1])


def beat_waveform(n_half, fs, rng, amp):
    """One sharp H-I-J-K-L complex centred on the J peak (index n_half).
    Small random shifts in the side waves give beat-to-beat shape variation."""
    tt = (np.arange(2 * n_half + 1) - n_half) / fs
    comps = [  # (offset s, relative amp, width s)
        (-0.100, 0.20, 0.025),   # H
        (-0.055, -0.45, 0.025),  # I
        (0.000, 1.00, 0.050),    # J
        (0.065, -0.55, 0.0275),  # K
        (0.140, 0.25, 0.035),    # L
    ]
    w = np.zeros_like(tt)
    for off, a, s in comps:
        if off != 0.0:
            off = off + rng.normal(0, 0.004)          # side waves wander a few ms
            a = a * (1 + rng.normal(0, 0.15))
        s = s * (1 + rng.normal(0, 0.10))
        w += a * np.exp(-((tt - off) ** 2) / (2 * s ** 2))
    return amp * w


def colored_noise(n, fs, rng):
    """Noise with the same spectral slope as the real sensor floor: power
    falling roughly as 1/f^2.4 between ~0.5 and 15 Hz, then flattening, and
    cut off by the analog low-pass (~34 Hz) like the real board."""
    white = rng.normal(0, 1, n)
    F = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / fs)
    shape = 1.0 / np.maximum(f, 0.5) ** 1.2
    shape += 0.04 * shape[np.searchsorted(f, 15.0)] * (f > 0)   # floor above ~15 Hz
    x = np.fft.irfft(F * shape, n)
    sos = sg.butter(2, 34.0, btype="low", fs=fs, output="sos")   # analog LPF
    x = sg.sosfilt(sos, x)
    return x / x.std()


def bandpass(x, fs, lo, hi):
    sos = sg.butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sg.sosfiltfilt(sos, x)


def build(name, minutes, seed):
    p = PRESETS[name]
    rng = np.random.default_rng(seed)
    dur = minutes * 60.0
    n = int(dur * FS)
    t = np.arange(n) / FS

    # --- breathing: slowly wandering rate, non-sinusoidal shape ---
    rr_hz = (RR_BPM / 60.0) * (1 + 0.08 * np.sin(2 * np.pi * t / 240.0))
    phase = 2 * np.pi * np.cumsum(rr_hz) / FS

    def resp_phase_fn(tq):
        return np.interp(tq, t, phase)

    resp = RESP_AMP_V * (np.sin(phase) + 0.35 * np.sin(2 * phase + 0.6)
                         + 0.15 * np.sin(3 * phase + 1.1))

    # --- heartbeats ---
    beats = heartbeat_times(dur, p["hr_path"], resp_phase_fn, rng)
    card = np.zeros(n)
    n_half = int(0.40 * FS)
    if p["beats"]:
        for bt in beats:
            # size follows breathing (+-25%) and random (+-15%)
            amp = J_AMP_V * (1 + 0.25 * np.sin(resp_phase_fn(bt))) \
                * (1 + rng.normal(0, 0.15))
            w = beat_waveform(n_half, FS, rng, amp)
            c = int(round(bt * FS))
            a, b = c - n_half, c + n_half + 1
            if a < 0 or b > n:
                continue
            card[a:b] += w
        # mattress = mechanical low-pass: blurs the sharp J into a blunt hump.
        # Zero-phase, so the true beat time (J centre) is not shifted.
        sos = sg.butter(2, MATTRESS_LP_HZ, btype="low", fs=FS, output="sos")
        card = sg.sosfiltfilt(sos, card)

    # --- noise, scaled so beat SNR in the 2-12 Hz band hits the target ---
    noise = colored_noise(n, FS, rng)
    ref = card if p["beats"] else None
    if ref is None:
        # no beats: build a reference beat train only to set the noise level,
        # so this file has exactly the same noise as 'realistic'
        tmp = np.zeros(n)
        rng2 = np.random.default_rng(seed + 999)  # separate stream
        for bt in beats:
            w = beat_waveform(n_half, FS, rng2, J_AMP_V)
            c = int(round(bt * FS)); a, b = c - n_half, c + n_half + 1
            if 0 <= a and b <= n:
                tmp[a:b] += w
        sos = sg.butter(2, MATTRESS_LP_HZ, btype="low", fs=FS, output="sos")
        ref = sg.sosfiltfilt(sos, tmp)
    ref_bp = bandpass(ref, FS, 2, 12)
    idx = np.clip((beats * FS).astype(int), 0, n - 1)
    beat_peak = float(np.median(ref_bp[idx]))
    noise_bp_std = float(np.std(bandpass(noise, FS, 2, 12)))
    noise *= (beat_peak / p["snr"]) / noise_bp_std

    # --- movement bursts (rolling over etc.) ---
    move_mask = np.zeros(n, bool)
    if p["n_moves"]:
        starts = np.linspace(0.2, 0.8, p["n_moves"]) * dur
        for s0 in starts + rng.uniform(-20, 20, p["n_moves"]):
            length = rng.uniform(5, 12)
            a, b = int(s0 * FS), int((s0 + length) * FS)
            # body shifts are mostly slow, large motion: same 1/f-type shape as
            # the noise floor (a broadband burst does not match the real spectrum)
            burst = colored_noise(b - a, FS, rng)
            env = np.hanning(b - a)
            card_scale = 4 * J_AMP_V
            noise[a:b] += card_scale * env * burst
            noise[a:b] += 0.05 * env * np.sin(2 * np.pi * 0.5 * t[a:b])  # slow shift
            move_mask[a:b] = True

    x = 1.52 + resp + card + noise
    x = np.round(x, 3)  # Saleae-style 1 mV steps
    vref = np.round(1.65 + rng.normal(0, 0.0005, n), 3)

    # --- write signal (PicoScope-style 3-line header; works with both loaders) ---
    fname = f"synth_{name}.csv"
    with open(fname, "w", newline="") as f:
        f.write("Time,Channel A,Channel B\n(s),(V),(V)\n\n")
        np.savetxt(f, np.column_stack([t, x, vref]), delimiter=",",
                   fmt=["%.6f", "%.3f", "%.3f"])

    # --- truth: beats ---
    with open(f"synth_{name}_truth_beats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beat_time_s", "during_movement"])
        for bt in beats:
            i = min(int(bt * FS), n - 1)
            w.writerow([f"{bt:.6f}", bool(move_mask[i]) if p["beats"] else ""])

    # --- truth: per-epoch ---
    rows = []
    for e0 in np.arange(0, dur - EPOCH_S + 1e-9, EPOCH_S):
        e1 = e0 + EPOCH_S
        b = beats[(beats >= e0) & (beats < e1)]
        ibi = np.diff(b) * 1000.0
        hr = 60000.0 / np.median(ibi) if len(ibi) else np.nan
        rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2))) if len(ibi) > 2 else np.nan
        m = (t >= e0) & (t < e1)
        rows.append(dict(
            t_start_s=round(float(e0), 1),
            true_hr_bpm=round(hr, 2) if p["beats"] else "",
            true_rmssd_ms=round(rmssd, 2) if p["beats"] else "",
            true_rr_bpm=round(float(np.mean(rr_hz[m]) * 60.0), 2),
            movement=bool(move_mask[m].any()),
        ))
    with open(f"synth_{name}_truth_epochs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    ibi_all = np.diff(beats) * 1000
    print(f"{fname}: {minutes:.0f} min, {len(beats)} beats"
          f"{'' if p['beats'] else ' (NOT drawn into the signal)'}, "
          f"SNR {p['snr']}, true RMSSD over whole file "
          f"{np.sqrt(np.mean(np.diff(ibi_all) ** 2)):.1f} ms, "
          f"movement bursts {p['n_moves']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--only", choices=list(PRESETS), default=None)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    names = [a.only] if a.only else list(PRESETS)
    for i, nm in enumerate(names):
        build(nm, a.minutes, a.seed + i)


if __name__ == "__main__":
    main()
