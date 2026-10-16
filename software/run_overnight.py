"""
Process an overnight BCG CSV into a per-epoch results CSV (HR, RR, movement).

Beats are detected once per long window (default 5 min) and then sliced into
30 s epochs. Detecting per 30 s epoch fails: that is too short to bootstrap a
reliable template, so the detector either fails or locks onto a harmonic.
Epochs whose HR sits near 2x or 0.5x the night's median are flagged.

Usage:
    py run_overnight.py night_data.csv --test     first 5 min
    py run_overnight.py night_data.csv            full file

Needs bcg_paalasmaa.py in the same folder.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy import signal as sg

_here = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(_here, "bcg_paalasmaa.py")):
    print("\nERROR: 'bcg_paalasmaa.py' is missing. Put it in this folder:\n"
          f"  {_here}\n")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("\nERROR: pandas not installed. Run:  pip install pandas\n")
    sys.exit(1)

import bcg_paalasmaa as pa


RR_BAND = (0.1, 0.5)        # 6-30 breaths/min
# Movement bands flank the cardiac band (2-12 Hz) so a strong heartbeat is not
# counted as movement. Body movement shows up as slow displacement plus
# broadband friction against the mattress.
MOVE_BAND_LOW = (0.3, 1.0)     # slow body shifts
MOVE_BAND_HIGH = (18.0, 40.0)  # friction, above cardiac
HR_MIN, HR_MAX = 40.0, 150.0   # above 150 on a sleeping adult is a harmonic


def _bandpass(x, fs, lo, hi, order=4):
    sos = sg.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sg.sosfiltfilt(sos, x)


def load_big_csv(path, fs_override=None, col=1):
    print(f"[1/3] Loading {path} ... (a few minutes for a big file)")
    t0 = time.time()
    df = pd.read_csv(path, skiprows=3, header=None, usecols=[0, col],
                     names=["t", "x"], engine="c", on_bad_lines="skip")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df = df.dropna()
    t = df["t"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    del df
    if len(t) < 100:
        print("\nERROR: almost no data rows read. Check the file / --col.\n")
        sys.exit(1)
    if fs_override:
        fs = float(fs_override)
    else:
        dt = float(np.median(np.diff(t[:1000])))
        if dt <= 0:
            print("\nERROR: could not infer sample rate. Use --fs 3000\n")
            sys.exit(1)
        fs = 1.0 / dt
    print(f"      {len(x)/1e6:.1f}M samples in {time.time()-t0:.0f}s | "
          f"fs={fs:.0f} Hz | {(t[-1]-t[0])/60:.1f} min")
    return t, x, fs


def beats_for_window(seg, fs, timing_method="xcorr", bank_k=None):
    """Template beat detection over one long window.

    Returns (beat_times_s, result), or (None, None) if too few beats were
    found, meaning the sensor was not coupled to the body in that window."""
    try:
        from dataclasses import replace as _replace
        cfg = pa.Config(timing_method=timing_method)
        # Acceptance gate. The default pctile=45 rejects 55% of candidates by
        # construction. Fully open (pctile=100) lets weak windows accept noise
        # and I/K waves, inflating median HR from 70 to 84. pctile=60 recovered
        # most coverage (30 -> 43 beats/min) with HR steady at 70.
        cfg = _replace(cfg,
                       auto_residual_pctile=60.0,
                       residual_frac=0.75,
                       amp_ratio_max=2.5)
        if bank_k:
            cfg = _replace(cfg, timing_method="xcorr_bank",
                           bank_k=bank_k, bank_min_per_template=3)
        res, det = pa.analyze(seg, fs=fs, cfg=cfg, return_detail=True)
        if "error" in res:
            return None, None
        bt = det["beat_times"]

        # at least 25 beats/min, well under any sleeping HR, to count as coupled
        dur_s = len(seg) / fs
        min_beats = max(10, int(25.0 * dur_s / 60.0))
        if len(bt) < min_beats:
            return None, None
        return bt, res
    except Exception:
        return None, None


def hr_from_beat_times(bt, e0, e1):
    """HR from beats inside [e0, e1), using the median interval."""
    b = bt[(bt >= e0) & (bt < e1)]
    if len(b) < 3:
        return None, len(b)
    ibi = np.diff(b)
    ibi = ibi[(ibi > 60.0 / HR_MAX) & (ibi < 60.0 / HR_MIN)]
    if len(ibi) < 2:
        return None, len(b)
    return float(60.0 / np.median(ibi)), len(b)


def hrv_from_beat_times(bt, e0, e1):
    """RMSSD and SDNN from beats inside [e0, e1), after NN correction.

    BCG HRV is detection-limited: the broadened J-peak jitters beat timing by
    tens of ms and RMSSD squares that error, so expect 85-100 ms where a true
    resting value is 20-70 ms. Trend only."""
    b = bt[(bt >= e0) & (bt < e1)]
    if len(b) < 5:
        return None, None, 0
    ibi = np.diff(b) * 1000.0   # ms
    ibi = ibi[(ibi >= 60000.0 / HR_MAX) & (ibi <= 60000.0 / HR_MIN)]
    if len(ibi) < 4:
        return None, None, len(ibi)
    med = np.median(ibi)
    keep = np.abs(ibi - med) <= 0.25 * med
    ibi_c = ibi[keep]
    if len(ibi_c) < 3:
        return None, None, len(ibi_c)
    diffs = np.diff(ibi_c)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    sdnn = float(np.std(ibi_c, ddof=1))
    return rmssd, sdnn, len(ibi_c)


def rr_for_epoch(x, fs):
    resp = _bandpass(x - x.mean(), fs, *RR_BAND)
    nper = min(len(resp), int(fs * 30))
    if nper < int(fs * 8):
        return None, False
    f, p = sg.welch(resp, fs=fs, nperseg=nper)
    band = (f >= RR_BAND[0]) & (f <= RR_BAND[1])
    if not np.any(band):
        return None, False
    rr = float(f[band][int(np.argmax(p[band]))] * 60.0)
    zc = np.where(np.diff(np.signbit(resp)))[0]
    dur = len(resp) / fs
    rr_zc = (len(zc) / 2.0) / dur * 60.0 if dur > 0 else None
    agree = rr_zc is not None and abs(rr - rr_zc) < 3.0
    return rr, bool(agree)


def compute_movement_reference(x, fs, max_samples=200):
    """Movement-band reference level sampled across the whole recording.
    Using only the first few epochs fails if they happen to be quiet."""
    lo = _bandpass(x - x.mean(), fs, *MOVE_BAND_LOW)
    hi_hz = min(MOVE_BAND_HIGH[1], fs / 2 - 1)
    hi = (_bandpass(x - x.mean(), fs, MOVE_BAND_HIGH[0], hi_hz)
          if hi_hz > MOVE_BAND_HIGH[0] else np.zeros_like(x))
    mv = lo + hi
    w = max(1, int(1.0 * fs))
    n = len(mv) // w
    if n == 0:
        return 1e-9
    idx = np.linspace(0, n - 1, min(n, max_samples)).astype(int)
    rms = np.array([np.sqrt(np.mean(mv[i * w:(i + 1) * w] ** 2)) for i in idx])
    return float(np.median(rms))


def movement_for_epoch(x, fs, ref_med, thresh=3.0, win_s=1.0):
    """Fraction of 1 s windows above thresh x ref_med in the movement bands."""
    lo = _bandpass(x - x.mean(), fs, *MOVE_BAND_LOW)
    hi_hz = min(MOVE_BAND_HIGH[1], fs / 2 - 1)
    hi = (_bandpass(x - x.mean(), fs, MOVE_BAND_HIGH[0], hi_hz)
          if hi_hz > MOVE_BAND_HIGH[0] else np.zeros_like(x))
    mv = lo + hi
    w = max(1, int(win_s * fs))
    n = len(mv) // w
    if n == 0:
        return 0.0, 0.0
    rms = np.array([np.sqrt(np.mean(mv[i * w:(i + 1) * w] ** 2)) for i in range(n)])
    med = ref_med if ref_med else float(np.median(rms))
    return float(np.mean(rms > thresh * med)), float(np.median(rms))


def main():
    ap = argparse.ArgumentParser(
        description="Overnight BCG -> per-epoch HR / RR / movement")
    ap.add_argument("csv")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--epoch", type=float, default=30.0)
    ap.add_argument("--window", type=float, default=300.0,
                    help="beat-detection window (s); keep this in minutes")
    ap.add_argument("--fs", type=float, default=None)
    ap.add_argument("--col", type=int, default=1)
    ap.add_argument("--test", action="store_true", help="first 5 minutes only")
    ap.add_argument("--t0", type=float, default=None,
                    help="start time in seconds (skip into the recording)")
    ap.add_argument("--t1", type=float, default=None,
                    help="end time in seconds")
    ap.add_argument("--bank", type=int, default=None, metavar="K",
                    help="use a bank of K shape-clustered templates "
                         "instead of a single template (try --bank 6)")
    ap.add_argument("--hrv", action="store_true",
                    help="also compute HRV (RMSSD/SDNN). Detection-limited and "
                         "inflated; trend only, not clinical.")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"\nERROR: file not found: {args.csv}\n")
        sys.exit(1)

    t, x, fs = load_big_csv(args.csv, args.fs, args.col)
    t_start = args.t0 if args.t0 is not None else t[0]
    t_end = args.t1 if args.t1 is not None else t[-1]
    if args.test:
        t_end = min(t_end, t_start + 300.0)
        print("      test mode: first 5 minutes of the selected range only")

    rows = []
    # movement reference computed once over the selected range
    sel = (t >= t_start) & (t < t_end)
    print("      computing movement baseline over the whole recording ...")
    ref_med = compute_movement_reference(x[sel], fs)

    starts = np.arange(t_start, t_end, args.window)
    print(f"\n[2/3] {len(starts)} window(s) of {args.window/60:.0f} min. "
          f"Beats detected per window, then sliced into {args.epoch:.0f}s epochs.")
    tproc = time.time()

    for wi, wstart in enumerate(starts):
        wstop = min(wstart + args.window, t_end)
        m = (t >= wstart) & (t < wstop)
        if m.sum() < int(fs * 30):
            continue
        seg = x[m]
        seg_t = np.arange(len(seg)) / fs   # seconds from window start

        bt, res = beats_for_window(seg, fs, bank_k=args.bank)
        win_hr = res.get("mean_hr_bpm") if res else None

        n_ep = int((wstop - wstart) // args.epoch)
        for ei in range(n_ep):
            e0 = ei * args.epoch
            e1 = e0 + args.epoch
            abs_start = wstart + e0

            em = (seg_t >= e0) & (seg_t < e1)
            xe = seg[em]
            if len(xe) < int(fs * 5):
                continue

            hr, nbeats = (None, 0)
            rmssd, sdnn, n_ibi = (None, None, 0)
            if bt is not None:
                hr, nbeats = hr_from_beat_times(bt, e0, e1)
                if args.hrv:
                    rmssd, sdnn, n_ibi = hrv_from_beat_times(bt, e0, e1)

            rr, rr_ok = rr_for_epoch(xe, fs)
            mvf, _ = movement_for_epoch(xe, fs, ref_med)

            row = {
                "t_start_s": round(float(abs_start), 1),
                "t_start_min": round(float(abs_start) / 60.0, 2),
                "hr_bpm": round(hr, 1) if hr else "",
                "n_beats": nbeats,
                "window_hr": round(win_hr, 1) if win_hr else "",
                "rr_bpm": round(rr, 1) if rr else "",
                "rr_ok": rr_ok,
                "movement_frac": round(mvf, 3),
            }
            if args.hrv:
                row["rmssd_ms"] = round(rmssd, 1) if rmssd else ""
                row["sdnn_ms"] = round(sdnn, 1) if sdnn else ""
                row["n_ibi"] = n_ibi
            rows.append(row)

        el = time.time() - tproc
        eta = (el / (wi + 1)) * (len(starts) - wi - 1)
        nb = 0 if bt is None else len(bt)
        wh = f"{win_hr:.0f}" if win_hr else "FAILED"
        print(f"      window {wi+1}/{len(starts)} ({wstop/60:.0f} min) | "
              f"beats={nb} | winHR={wh} | elapsed {el/60:.1f}m ETA {eta/60:.1f}m")

    if not rows:
        print("\nERROR: no epochs produced.\n")
        sys.exit(1)

    # Trust gate: an epoch must pass all three checks.
    #  (a) physiological bounds: above ~110 bpm during sleep is an artifact,
    #      and a median inflated by bad epochs can't catch that on its own
    #  (b) harmonic check against the median of high-beat-count epochs only,
    #      since the overall median is circular when many epochs are wrong
    #  (c) enough beats to be a real rhythm
    SLEEP_HR_MIN, SLEEP_HR_MAX = 40.0, 110.0

    high_conf = [r for r in rows
                 if r["n_beats"] >= 20 and r["hr_bpm"] != ""
                 and SLEEP_HR_MIN <= r["hr_bpm"] <= SLEEP_HR_MAX]
    if high_conf:
        ref_med = float(np.median([r["hr_bpm"] for r in high_conf]))
    else:
        plausible = [r["hr_bpm"] for r in rows
                     if r["hr_bpm"] != ""
                     and SLEEP_HR_MIN <= r["hr_bpm"] <= SLEEP_HR_MAX]
        ref_med = float(np.median(plausible)) if plausible else None

    for r in rows:
        h = r["hr_bpm"]
        ok = False
        if h != "" and ref_med:
            in_bounds = SLEEP_HR_MIN <= h <= SLEEP_HR_MAX          # (a)
            near_double = abs(h - 2 * ref_med) < 0.15 * 2 * ref_med  # (b)
            near_half = abs(h - 0.5 * ref_med) < 0.15 * 0.5 * ref_med
            enough = r["n_beats"] >= 10                              # (c)
            ok = in_bounds and enough and (not near_double) and (not near_half)
        r["hr_trustworthy"] = ok
    night_med = ref_med

    print(f"\n[3/3] Writing {args.out} ...")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    good = [r["hr_bpm"] for r in rows if r["hr_trustworthy"]]
    rrs = [r["rr_bpm"] for r in rows if r["rr_bpm"] != ""]
    moves = [r["movement_frac"] for r in rows]
    print("\n" + "=" * 58)
    print(f"  DONE - {len(rows)} epochs of {args.epoch:.0f}s -> {args.out}")
    print("=" * 58)
    if good:
        g = np.array(good)
        print(f"  HR (trustworthy): median {np.median(g):.1f} bpm")
        print(f"                    IQR {np.percentile(g,25):.0f}-{np.percentile(g,75):.0f}, "
              f"range {g.min():.0f}-{g.max():.0f}")
        pct = 100.0 * len(good) / len(rows)
        print(f"                    {len(good)}/{len(rows)} epochs usable "
              f"({pct:.0f}% coverage)")
        # a genuine sleeping HR clusters tightly; a wide IQR means artifacts
        iqr = np.percentile(g, 75) - np.percentile(g, 25)
        if iqr > 20:
            print(f"  note: IQR is {iqr:.0f} bpm, wide for sleep.")
            print( "        A genuine sleeping HR clusters tightly, so some")
            print( "        epochs are probably still detection artifacts.")
        if pct < 50:
            print(f"  note: only {pct:.0f}% of the night produced a usable")
            print( "        heartbeat, so the sensor was poorly coupled to the")
            print( "        body. Check its positioning.")
    if args.hrv:
        hrvs = [r["rmssd_ms"] for r in rows
                if r.get("rmssd_ms") not in ("", None) and r["hr_trustworthy"]]
        if hrvs:
            print(f"\n  HRV RMSSD       : median {np.median(hrvs):.1f} ms "
                  f"(from {len(hrvs)} trustworthy epochs)")
            print( "  caveat: BCG HRV is detection-limited. The broadened")
            print( "          J-peak jitters beat timing by tens of ms and")
            print( "          RMSSD squares that error. Expect 85-100 ms where")
            print( "          truth is 20-70 ms. Trend only.")
    if rrs:
        print(f"\n  RR              : median {np.median(rrs):.1f} breaths/min")
    print(f"  Movement        : {np.mean(moves)*100:.1f}% avg, "
          f"{sum(1 for m in moves if m > 0.2)} heavy epochs")
    print()


if __name__ == "__main__":
    main()
