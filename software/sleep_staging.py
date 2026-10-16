"""
Sleep staging (wake / REM / light / deep) from BCG respiration and movement.

HRV is not used: under-mattress BCG beat timing jitters by tens of ms, which
swamps the autonomic differences between stages. NREM shows regular, repeatable
breathing; REM shows irregular breathing with low movement (muscle atonia).

Unsupervised by default, so results are estimates until validated against a
reference device.

Usage:
    py sleep_staging.py night_data.csv --fs 2500
    py sleep_staging.py night_data.csv --fs 2500 --out stages.csv --plot
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy import signal as sg

try:
    import pandas as pd
except ImportError:
    print("\nERROR: pandas not installed. Run:  pip install pandas\n")
    sys.exit(1)


RESP_BAND = (0.1, 0.5)          # 6-30 breaths/min
# Movement band sits above both respiration (0.1-0.5 Hz) and cardiac (up to
# ~12 Hz). A 0.3-1.0 Hz band overlaps respiration, so irregular breathing (the
# REM signature) leaks in and gets scored as wake.
MOVE_BAND = (20.0, 45.0)


def _bandpass(x, fs, lo, hi, order=4):
    hi = min(hi, fs / 2 - 1)
    if hi <= lo:
        return np.zeros_like(x)
    sos = sg.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sg.sosfiltfilt(sos, x)


def load_csv(path, fs_override=None, col=1):
    print(f"[1/4] Loading {path} ...")
    t0 = time.time()
    df = pd.read_csv(path, skiprows=3, header=None, usecols=[0, col],
                     names=["t", "x"], engine="c", on_bad_lines="skip")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df = df.dropna()
    t = df["t"].to_numpy(float)
    x = df["x"].to_numpy(float)
    del df
    fs = fs_override or 1.0 / float(np.median(np.diff(t[:1000])))
    print(f"      {len(x)/1e6:.1f}M samples in {time.time()-t0:.0f}s | "
          f"fs={fs:.0f} Hz | {(t[-1]-t[0])/60:.1f} min")
    return t, x, fs


# Respiration
def respiratory_waveform(x, fs):
    """Bandpass respiration, find the dominant frequency, then re-filter
    narrowly around it to get a peak-detectable waveform."""
    resp = _bandpass(x - x.mean(), fs, *RESP_BAND)
    nper = min(len(resp), int(fs * 30))
    if nper < int(fs * 8):
        return resp, None
    f, p = sg.welch(resp, fs=fs, nperseg=nper)
    band = (f >= RESP_BAND[0]) & (f <= RESP_BAND[1])
    if not np.any(band):
        return resp, None
    f_dom = float(f[band][int(np.argmax(p[band]))])
    if f_dom <= 0:
        return resp, None
    lo = max(0.05, f_dom * 0.5)
    hi = min(RESP_BAND[1] * 1.5, f_dom * 1.8)
    clean = _bandpass(x - x.mean(), fs, lo, hi)
    return clean, f_dom


def breath_intervals(resp_clean, fs, f_dom):
    """Breath-to-breath intervals (s) from peaks of the respiratory wave."""
    if f_dom is None or f_dom <= 0:
        return np.array([])
    min_dist = int(0.6 / f_dom * fs)   # at least 60% of the expected breath period
    pk, _ = sg.find_peaks(resp_clean, distance=max(1, min_dist))
    if len(pk) < 3:
        return np.array([])
    return np.diff(pk) / fs


def spectral_entropy(x, fs, band):
    """Normalised spectral entropy within band. Low = regular breathing."""
    nper = min(len(x), int(fs * 30))
    if nper < int(fs * 8):
        return np.nan
    f, p = sg.welch(x, fs=fs, nperseg=nper)
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m) or p[m].sum() <= 0:
        return np.nan
    pn = p[m] / p[m].sum()
    pn = pn[pn > 0]
    return float(-np.sum(pn * np.log(pn)) / np.log(len(pn)))


# Feature extraction
def extract_features(t, x, fs, epoch_s=30.0):
    print(f"\n[2/4] Extracting features per {epoch_s:.0f}s epoch ...")
    n_ep = int((t[-1] - t[0]) // epoch_s)
    n_per = int(epoch_s * fs)

    # movement reference from the whole recording
    mv = np.abs(_bandpass(x - x.mean(), fs, *MOVE_BAND))
    w = int(fs)
    nw = len(mv) // w
    idx = np.linspace(0, nw - 1, min(nw, 300)).astype(int)
    ref_rms = float(np.median([np.sqrt(np.mean(mv[i*w:(i+1)*w] ** 2)) for i in idx]))

    feats = []
    resp_waves = []
    for i in range(n_ep):
        a, b = i * n_per, (i + 1) * n_per
        if b > len(x):
            break
        xe = x[a:b]

        clean, f_dom = respiratory_waveform(xe, fs)
        ibi = breath_intervals(clean, fs, f_dom)

        rr = 60.0 * f_dom if f_dom else np.nan
        if len(ibi) >= 3:
            d = np.diff(ibi)
            resp_rmssd = float(np.sqrt(np.mean(d ** 2)) * 1000)   # ms
            resp_sdnn = float(np.std(ibi, ddof=1) * 1000)
            resp_cv = float(np.std(ibi) / np.mean(ibi)) if np.mean(ibi) > 0 else np.nan
        else:
            resp_rmssd = resp_sdnn = resp_cv = np.nan

        ent = spectral_entropy(xe - xe.mean(), fs, RESP_BAND)

        # respiratory amplitude: separates real REM (normal amplitude) from a
        # decoupled sensor (faint breathing), which look alike on shape alone
        resp_amp = float(np.sqrt(np.mean(clean ** 2))) if len(clean) else 0.0

        mv_e = np.abs(_bandpass(xe - xe.mean(), fs, *MOVE_BAND))
        move_int = float(np.mean(mv_e))
        ww = int(fs)
        nn = len(mv_e) // ww
        rms = np.array([np.sqrt(np.mean(mv_e[j*ww:(j+1)*ww] ** 2)) for j in range(nn)]) \
            if nn else np.array([0.0])
        move_frac = float(np.mean(rms > 3.0 * ref_rms))

        # downsampled wave for cross-epoch similarity
        ds = max(1, len(clean) // 300)
        resp_waves.append(clean[::ds][:300])

        feats.append({
            "epoch": i,
            "t_start_min": round(i * epoch_s / 60.0, 2),
            "resp_rate": rr,
            "resp_amp": resp_amp,
            "resp_rmssd": resp_rmssd,
            "resp_sdnn": resp_sdnn,
            "resp_cv": resp_cv,
            "resp_entropy": ent,
            "move_intensity": move_int,
            "move_frac": move_frac,
        })
        if (i + 1) % 200 == 0:
            print(f"      {i+1}/{n_ep} epochs")

    # similarity to neighbouring epochs: REM breaks it, NREM preserves it
    L = min(len(w_) for w_ in resp_waves) if resp_waves else 0
    for i, f_ in enumerate(feats):
        sims = []
        for j in (i - 2, i - 1, i + 1, i + 2):
            if 0 <= j < len(resp_waves):
                a_ = resp_waves[i][:L]
                b_ = resp_waves[j][:L]
                if len(a_) > 10 and np.std(a_) > 0 and np.std(b_) > 0:
                    sims.append(abs(float(np.corrcoef(a_, b_)[0, 1])))
        f_["resp_similarity"] = float(np.mean(sims)) if sims else np.nan

    print(f"      done: {len(feats)} epochs")
    return feats


# Staging
def stage_epochs(feats, smooth_epochs=5, four_stage=True):
    """Stage epochs into WAKE / REM / LIGHT / DEEP.

    REM, light and deep sit on one breathing-regularity axis: REM at the
    irregular end (with low movement), deep at the regular end. REM/NREM is
    the more reliable split; light vs deep is a softer distinction."""
    print("\n[3/4] Staging (unsupervised) ...")

    def col(name):
        return np.array([f.get(name, np.nan) for f in feats], dtype=float)

    def z(v):
        v = v.copy()
        m = np.isfinite(v)
        if m.sum() < 3:
            return np.zeros_like(v)
        mu, sd = np.nanmean(v[m]), np.nanstd(v[m])
        if sd == 0:
            return np.zeros_like(v)
        out = np.full_like(v, np.nan)
        out[m] = (v[m] - mu) / sd
        return out

    # irregularity axis: higher = more irregular breathing
    z_var = np.nanmean(np.vstack([z(col("resp_rmssd")),
                                  z(col("resp_sdnn")),
                                  z(col("resp_cv"))]), axis=0)
    z_ent = z(col("resp_entropy"))
    z_sim = -z(col("resp_similarity"))     # low similarity = irregular
    irregularity = np.nanmean(np.vstack([1.0 * z_var,
                                         0.8 * z_ent,
                                         1.2 * z_sim]), axis=0)

    z_mov = z(col("move_intensity"))

    # REM: irregular breathing with low movement
    rem_score = irregularity - 0.6 * z_mov

    # temporal smoothing, since stages persist for minutes
    def smooth(v):
        valid = np.isfinite(v)
        if valid.sum() <= smooth_epochs:
            return v
        k = np.ones(smooth_epochs) / smooth_epochs
        filled = np.where(valid, v, np.nanmedian(v[valid]))
        return np.convolve(filled, k, mode="same")

    rem_score = smooth(rem_score)
    irregularity = smooth(irregularity)

    n = len(feats)
    mv = col("move_frac")
    mvi = col("move_intensity")

    # signal-quality gate: faint respiration means a decoupled sensor, which
    # would otherwise look like REM. Mark it unknown instead of staging it.
    amp = col("resp_amp")
    amp_ref = float(np.nanmedian(amp[np.isfinite(amp)])) if np.isfinite(amp).any() else 0.0
    poor = np.isfinite(amp) & (amp < 0.35 * amp_ref) if amp_ref > 0 else np.zeros(n, bool)
    poor |= ~np.isfinite(col("resp_rmssd"))     # no breaths detected at all

    # wake: 15% of an epoch (~4-5 s of motion) is a lot for someone asleep
    mv_med = float(np.nanmedian(mvi[np.isfinite(mvi)])) if np.isfinite(mvi).any() else 0.0
    wake = (mv > 0.15) | (np.isfinite(mvi) & (mvi > 5.0 * mv_med) if mv_med > 0
                          else np.zeros(n, bool))
    wake &= ~poor    # a decoupled epoch is unknown, not wake

    # REM: top of rem_score among good, non-wake epochs
    sleep_mask = (~wake) & (~poor) & np.isfinite(rem_score)
    if sleep_mask.sum() < 10:
        rem_thr = np.inf
    else:
        rem_thr = float(np.percentile(rem_score[sleep_mask], 78))

    # REM latency: suppress REM in the first 60 min
    epoch_min = np.array([f["t_start_min"] for f in feats], dtype=float)
    rem_allowed = epoch_min >= 60.0

    is_rem = sleep_mask & (rem_score >= rem_thr) & rem_allowed

    # within NREM, the most regular 25% is deep
    nrem_mask = (~wake) & (~poor) & (~is_rem) & np.isfinite(irregularity)
    if four_stage and nrem_mask.sum() >= 20:
        deep_thr = float(np.percentile(irregularity[nrem_mask], 25))
        is_deep = nrem_mask & (irregularity <= deep_thr)
    else:
        is_deep = np.zeros(n, dtype=bool)

    stages = []
    for i, f_ in enumerate(feats):
        if poor[i]:
            s = "UNKNOWN"
        elif wake[i]:
            s = "WAKE"
        elif is_rem[i]:
            s = "REM"
        elif is_deep[i]:
            s = "DEEP"
        else:
            s = "LIGHT"
        f_["rem_score"] = round(float(rem_score[i]), 3) if np.isfinite(rem_score[i]) else ""
        f_["irregularity"] = round(float(irregularity[i]), 3) if np.isfinite(irregularity[i]) else ""
        f_["stage"] = s
        stages.append(s)

    print(f"      {'stage':8s} {'epochs':>7} {'%':>5} {'minutes':>8}")
    for name in ("UNKNOWN", "WAKE", "REM", "LIGHT", "DEEP"):
        c = stages.count(name)
        if c:
            print(f"      {name:8s} {c:>7} {100*c/n:>4.0f}% {c*0.5:>7.0f}")

    n_poor = int(poor.sum())
    if n_poor:
        print(f"\n      {100*n_poor/n:.0f}% of the night had a respiratory signal too")
        print( "      weak to stage (sensor not coupled to the body). Those epochs")
        print( "      are marked unknown rather than guessed. This is the same")
        print( "      coupling problem the cardiac analysis found.")

    # sanity checks against normal sleep architecture
    n_staged = n - n_poor
    if n_staged > 0:
        deep_pct = 100 * stages.count("DEEP") / n_staged
        rem_pct = 100 * stages.count("REM") / n_staged
        print()
        if not (10 <= rem_pct <= 30):
            print(f"      NOTE: REM at {rem_pct:.0f}% of staged sleep is outside the usual 20-25%.")
        if not (8 <= deep_pct <= 25):
            print(f"      NOTE: DEEP at {deep_pct:.0f}% of staged sleep is outside the usual 13-23%.")

    # deep sleep should cluster early in the night
    deep_idx = [i for i, s in enumerate(stages) if s == "DEEP"]
    if deep_idx:
        first_third = sum(1 for i in deep_idx if i < n / 3)
        frac = 100.0 * first_third / len(deep_idx)
        print(f"      {frac:.0f}% of DEEP sleep falls in the first third of the night")
        print( "      (it should be the majority; if this is low, the DEEP calls")
        print( "      are suspect).")

    return feats


def main():
    ap = argparse.ArgumentParser(
        description="REM/NREM staging from BCG respiration + movement (no HRV)")
    ap.add_argument("csv")
    ap.add_argument("--fs", type=float, default=None)
    ap.add_argument("--col", type=int, default=1)
    ap.add_argument("--epoch", type=float, default=30.0)
    ap.add_argument("--out", default="stages.csv")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"\nERROR: file not found: {args.csv}\n")
        sys.exit(1)

    t, x, fs = load_csv(args.csv, args.fs, args.col)
    feats = extract_features(t, x, fs, args.epoch)
    if not feats:
        print("No epochs extracted.")
        sys.exit(1)
    feats = stage_epochs(feats)

    print(f"\n[4/4] Writing {args.out} ...")
    keys = ["epoch", "t_start_min", "stage", "rem_score", "irregularity",
            "resp_rate", "resp_amp", "resp_rmssd", "resp_sdnn", "resp_cv",
            "resp_entropy", "resp_similarity", "move_intensity", "move_frac"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in feats:
            w.writerow({k: (round(v, 4) if isinstance(v, float) and np.isfinite(v)
                            else ("" if isinstance(v, float) else v))
                        for k, v in r.items() if k in keys})

    # text hypnogram
    print("\n  HYPNOGRAM (W=wake  R=REM  -=light  #=deep  ?=no usable signal)")
    sym = {"WAKE": "W", "REM": "R", "LIGHT": "-", "DEEP": "#", "UNKNOWN": "?"}
    line = ""
    for i, f_ in enumerate(feats):
        line += sym[f_["stage"]]
        if (i + 1) % 100 == 0:
            print(f"   {(i-99)*0.5:6.0f}m |{line}")
            line = ""
    if line:
        print(f"   {(len(feats)-len(line))*0.5:6.0f}m |{line}")

    print("\n  Validate this. Unsupervised staging finds structure in the data")
    print( "  and labels clusters by their physiological signature, but")
    print( "  without a reference device it cannot be verified.")
    print( "  Light vs deep is a difference of degree on a continuum rather")
    print( "  than a distinct signature like REM, so it is less reliable")
    print( "  than the REM/NREM split. Compare against a wearable.\n")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            tm = [f_["t_start_min"] for f_ in feats]
            lvl = {"DEEP": 0, "LIGHT": 1, "REM": 2, "WAKE": 3, "UNKNOWN": 4}
            stg = [lvl[f_["stage"]] for f_ in feats]
            fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
            ax[0].step(tm, stg, where="post", lw=1)
            ax[0].set_yticks([0, 1, 2, 3, 4])
            ax[0].set_yticklabels(["DEEP", "LIGHT", "REM", "WAKE", "no signal"])
            ax[0].invert_yaxis()
            ax[0].set_title("Hypnogram (respiration + movement, no HRV)")
            ax[1].plot(tm, [f_["irregularity"] if f_["irregularity"] != "" else np.nan
                            for f_ in feats], lw=0.8, color="purple")
            ax[1].set_ylabel("breathing\nirregularity")
            ax[2].plot(tm, [f_["resp_rate"] for f_ in feats], lw=0.8, color="teal")
            ax[2].set_ylabel("resp rate"); ax[2].set_xlabel("minutes")
            plt.tight_layout(); plt.savefig("hypnogram.png", dpi=90)
            print("  saved hypnogram.png\n")
        except ImportError:
            print("  (matplotlib not installed; skipping plot)\n")


if __name__ == "__main__":
    main()
