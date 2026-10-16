"""
evaluate_synthetic.py

Scores the real pipeline against a synthetic file whose answers are known.

It does two things:
  1. Runs run_overnight.py on the file (with --hrv), exactly as you would on a
     real night, and compares every 30 s epoch to the truth: was HR right, was
     RMSSD right, was the epoch wrongly trusted.
  2. Runs the same beat detector directly (run_overnight.beats_for_window, same
     settings) and compares every detected beat to the true beat times: how many
     were missed, how many were invented, and how far off the timing was.

Usage (same folder as run_overnight.py, bcg_paalasmaa.py and the synth files):
  py evaluate_synthetic.py synth_realistic.csv
  py evaluate_synthetic.py synth_easy.csv synth_realistic.csv synth_hard.csv synth_nobeat.csv
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

import run_overnight as ro

MATCH_MS = 150.0      # a detection within +-150 ms of a true beat counts as a hit
WINDOW_S = 300.0      # same 5-minute detection windows as run_overnight.py


def run_pipeline(csv_path):
    out = csv_path.replace(".csv", "_results.csv")
    cmd = [sys.executable, "run_overnight.py", csv_path, "--hrv", "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"run_overnight.py failed on {csv_path}")
    return pd.read_csv(out)


def detect_all_beats(csv_path):
    """Beat times from the same detector and settings run_overnight.py uses."""
    t, x, fs = ro.load_big_csv(csv_path)
    beats, windows = [], []
    for w0 in np.arange(t[0], t[-1], WINDOW_S):
        m = (t >= w0) & (t < w0 + WINDOW_S)
        if m.sum() < int(fs * 30):
            continue
        bt, res = ro.beats_for_window(x[m], fs)
        hr = res.get("mean_hr_bpm") if res else None
        windows.append((w0, hr, 0 if bt is None else len(bt)))
        if bt is not None:
            beats.extend(list(np.asarray(bt) + w0))
    return np.array(sorted(beats)), windows


def score_beats(det, truth):
    true_t = truth.beat_time_s.to_numpy()
    moving = truth.during_movement.astype(str).str.lower().eq("true").to_numpy()
    keep = ~moving
    tt = true_t[keep]

    errs, used = [], set()
    for b in det:
        k = int(np.argmin(np.abs(true_t - b))) if len(true_t) else -1
        if k >= 0 and abs(true_t[k] - b) * 1000 <= MATCH_MS and k not in used:
            used.add(k)
            if keep[k]:
                errs.append((b - true_t[k]) * 1000)
    errs = np.array(errs)
    n_extra = len(det) - len(used)
    n_missed = int(keep.sum() - sum(1 for k in used if keep[k]))
    return errs, n_missed, n_extra, len(tt)


def report(csv_path):
    name = os.path.basename(csv_path).replace(".csv", "")
    truth_b = pd.read_csv(f"{name}_truth_beats.csv")
    truth_e = pd.read_csv(f"{name}_truth_epochs.csv")
    has_beats = truth_e.true_hr_bpm.notna().any()

    print("\n" + "=" * 66)
    print(f"  {name}")
    print("=" * 66)

    # ---------------- beat level ----------------
    det, windows = detect_all_beats(csv_path)
    print("  5-minute detection windows (truth HR in brackets):")
    for w0, hr, nb in windows:
        seg = truth_e[(truth_e.t_start_s >= w0) & (truth_e.t_start_s < w0 + WINDOW_S)]
        thr = f"{seg.true_hr_bpm.median():.0f}" if has_beats else "no beats"
        shown = f"{hr:.0f}" if hr else "FAILED"
        flag = ""
        if hr and has_beats and abs(hr - seg.true_hr_bpm.median()) > 10:
            flag = "   <-- WRONG"
        print(f"    {w0/60:4.0f}-{(w0+WINDOW_S)/60:<3.0f}min  detector HR {shown:>6}  "
              f"[{thr}]  beats {nb}{flag}")

    if has_beats:
        errs, missed, extra, n_true = score_beats(det, truth_b)
        print(f"\n  Beats (outside movement): {n_true} true, {len(det)} detected")
        print(f"    found    : {len(errs)} ({100*len(errs)/n_true:.0f}%)")
        print(f"    missed   : {missed} ({100*missed/n_true:.0f}%)")
        print(f"    invented : {extra} detections with no true beat nearby")
        if len(errs) > 5:
            bias = np.median(errs)
            mad = 1.4826 * np.median(np.abs(errs - bias))
            print(f"    timing   : constant offset {bias:+.1f} ms (harmless), "
                  f"scatter {np.std(errs):.1f} ms (robust {mad:.1f} ms)")
            print(f"    -> RMSSD floor from this scatter: about "
                  f"{np.sqrt(6)*mad:.0f} ms even with a perfect beat count")
    else:
        print(f"\n  NO heartbeat in this file. Detector still produced "
              f"{len(det)} 'beats'.")

    # ---------------- epoch level (the real run_overnight.py output) ----------------
    res = run_pipeline(csv_path)
    m = res.merge(truth_e, on="t_start_s", how="inner")
    trusted = m[m.hr_trustworthy == True]
    print(f"\n  run_overnight.py epochs: {len(m)}, marked trustworthy: "
          f"{len(trusted)} ({100*len(trusted)/max(1,len(m)):.0f}%)")
    if not has_beats:
        print(f"    every trusted epoch here is a FALSE ALARM "
              f"(median reported HR {trusted.hr_bpm.median():.0f} bpm)"
              if len(trusted) else "    good: nothing trusted")
        return
    if len(trusted):
        err = (trusted.hr_bpm - trusted.true_hr_bpm).abs()
        bad = (err > 10).sum()
        print(f"    HR error on trusted epochs: median {err.median():.1f} bpm, "
              f"{bad} epoch(s) off by >10 bpm")
        r = trusted.dropna(subset=["rmssd_ms"])
        r = r[r.rmssd_ms != ""]
        if len(r):
            meas = pd.to_numeric(r.rmssd_ms).median()
            true = r.true_rmssd_ms.median()
            print(f"    RMSSD: reported median {meas:.0f} ms vs TRUE {true:.0f} ms "
                  f"({meas/true:.1f}x)")
    rr = pd.to_numeric(m.rr_bpm, errors="coerce")
    print(f"    RR: reported median {rr.median():.1f} vs true "
          f"{m.true_rr_bpm.median():.1f} breaths/min")
    mv_true = m.movement.astype(str).str.lower().eq("true")
    print(f"    movement: {int(mv_true.sum())} epochs truly had movement; "
          f"pipeline flagged {(m.movement_frac > 0.2).sum()} as heavy movement "
          f"({(mv_true & (m.movement_frac > 0.2)).sum()} of them correct)")


if __name__ == "__main__":
    files = sys.argv[1:] or ["synth_easy.csv", "synth_realistic.csv",
                             "synth_hard.csv", "synth_nobeat.csv"]
    for f in files:
        report(f)
    print()
