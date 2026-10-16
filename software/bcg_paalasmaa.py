"""
Beat-to-beat interval (IBI) detection for under-mattress BCG.

Implements Paalasmaa, Toivonen & Partinen, "Adaptive Heartbeat Modeling for
Beat-to-Beat Heart Rate Measurement in Ballistocardiograms," IEEE JBHI
19(6):1945-1952, 2015.

The signal is modelled as overlapping copies of one learned heartbeat shape
(the template mu). An interval (ta, tb) is accepted only if two amplitude-fit
copies of mu reconstruct that stretch with low residual and similar amplitudes.
Ambiguous stretches are skipped rather than guessed.

Dependencies: numpy, scipy. matplotlib only for --plot.
"""

from __future__ import annotations
import glob
import os
import numpy as np
from dataclasses import dataclass, field, replace
from scipy import signal as sg
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform


# Configuration
@dataclass
class Config:
    # interval search range (s): 0.4-2.0 s = 30-150 bpm
    min_ibi: float = 0.40
    max_ibi: float = 2.00

    # preprocessing
    gauss_sigma_s: float = 0.10            # respiration removal (Gaussian high-pass)
    lp_corner_hz: float | None = 25.0      # anti-noise low-pass, None to skip

    # template bootstrap
    boot_segment_s: float = 15.0
    boot_window_s: float = 0.40
    boot_cluster_size: int = 4

    # template; keep ~one cardiac cycle or the residual picks up phantom neighbours
    template_len_s: float = 0.80
    hvs_lp_hz: float = 10.0

    # candidate detection
    xcorr_mid_s: float = 0.40

    # acceptance gates
    residual_frac: float = 0.55            # accept if mse/msig below this (paper: 0.20)
    auto_residual: bool = True             # calibrate threshold from this recording
    # "mad": threshold = median + auto_residual_mad_k * MAD. Rejects outliers only.
    # "percentile": threshold = auto_residual_pctile of observed ratios. Rejects
    #   (100 - pctile)% of pairs by construction, even on clean data.
    auto_residual_mode: str = "mad"
    auto_residual_pctile: float = 45.0
    auto_residual_mad_k: float = 3.0
    amp_ratio_max: float = 2.00            # reject if larger amp > 2x smaller

    # template re-estimation interval (accepted beats)
    reestimate_every: int = 20

    # HRV reporting clamps
    hrv_min_ms: float = 300.0
    hrv_max_ms: float = 1500.0
    nn_reject_frac: float = 0.25           # drop IBIs >25% off the local median

    # beat timing: 'apex', 'xcorr', 'slope' or 'xcorr_bank'
    timing_method: str = "xcorr"
    bank_k: int = 6
    bank_min_per_template: int = 2

    # spectral seed: narrows min/max_ibi around the PSD peak before detection.
    # Off by default; it can lock onto 2x the true rate and exclude the real
    # interval entirely. The spectral cross-check in the output runs either way.
    use_spectral_seed: bool = False
    spectral_band_hz: tuple = (0.6, 2.2)
    spectral_seed_halfwidth: float = 0.35
    spectral_disagreement_bpm: float = 6.0

    # autocorrelation seed: narrows min/max_ibi around the beat period found by
    # autocorrelation of the raw 0.8-12 Hz band. Uses the raw signal because
    # preprocessing strips the ~1 Hz cardiac fundamental. Skipped if the
    # spectral seed is on.
    use_acf_seed: bool = True
    acf_seed_bpm: tuple = (40.0, 120.0)
    acf_seed_halfwidth: float = 0.35

    # frequency-domain HRV (unreliable under 5 min)
    fd_hrv_min_reliable_s: float = 300.0
    fd_resample_hz: float = 4.0
    fd_vlf_band: tuple = (0.003, 0.04)
    fd_lf_band: tuple = (0.04, 0.15)
    fd_hf_band: tuple = (0.15, 0.40)


# Loading
def load_picoscope_csvs(paths, channel_col=1):
    """Load PicoScope CSVs (3 header rows) and concatenate in order.
    Non-finite samples are interpolated. Returns (t, x, fs)."""
    if isinstance(paths, str):
        paths = [paths]
    seg_t, seg_x, t_off = [], [], 0.0
    for p in paths:
        raw = np.genfromtxt(p, delimiter=",", skip_header=3)
        tt = raw[:, 0]
        xx = raw[:, channel_col]
        m = np.isfinite(tt)
        tt, xx = tt[m], xx[m]
        bad = ~np.isfinite(xx)
        if bad.any():
            xx[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), xx[~bad])
        seg_t.append(tt + t_off)
        seg_x.append(xx)
        t_off += tt[-1] + (tt[1] - tt[0])
    t = np.concatenate(seg_t)
    x = np.concatenate(seg_x)
    fs = 1.0 / np.median(np.diff(seg_t[0]))
    return t, x, fs


# Preprocessing
def preprocess(x, fs, cfg: Config):
    """Remove DC and respiration, optionally low-pass."""
    x = np.asarray(x, float) - np.mean(x)
    sigma = cfg.gauss_sigma_s * fs
    resp = _gaussian_filter1d(x, sigma)
    y = x - resp
    if cfg.lp_corner_hz is not None:
        sos = sg.butter(4, cfg.lp_corner_hz, btype="low", fs=fs, output="sos")
        y = sg.sosfiltfilt(sos, y)
    return y


def _gaussian_filter1d(x, sigma):
    if sigma < 1e-6:
        return x.copy()
    radius = int(4 * sigma + 0.5)
    k = np.arange(-radius, radius + 1)
    g = np.exp(-(k ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return np.convolve(x, g, mode="same")


def acf_period(y, fs, cfg: Config):
    """Beat period (s) from autocorrelation of the 0.8-12 Hz band, or None."""
    sos = sg.butter(3, [0.8, 12.0], btype="band", fs=fs, output="sos")
    yy = sg.sosfiltfilt(sos, y - np.mean(y))
    if np.std(yy) == 0:
        return None
    ac = sg.correlate(yy, yy, mode="full", method="fft")[len(yy) - 1:]
    ac = ac / ac[0]
    lo = int(fs * 60.0 / cfg.acf_seed_bpm[1])
    hi = int(fs * 60.0 / cfg.acf_seed_bpm[0])
    if hi >= len(ac) or hi <= lo:
        return None
    return (lo + int(np.argmax(ac[lo:hi]))) / fs


# Spectral analysis
def spectral_hr_estimate(y, fs, cfg: Config):
    """Window-averaged cardiac rate from the Welch PSD peak."""
    lo, hi = cfg.spectral_band_hz
    nper = min(len(y), int(fs * 8))
    if nper < int(fs * 2):
        return {"error": "signal too short for spectral estimate"}
    freqs, power = sg.welch(y, fs=fs, nperseg=nper)
    band = (freqs >= lo) & (freqs <= hi)
    if not np.any(band):
        return {"error": "no energy in cardiac band"}
    pk_idx = np.argmax(power[band])
    peak_hz = freqs[band][pk_idx]
    peak_power = power[band][pk_idx]
    band_power_frac = float(peak_power / (power[band].sum() + 1e-30))
    return {
        "peak_hz": float(peak_hz),
        "peak_bpm": float(peak_hz * 60.0),
        "band_power_frac": band_power_frac,
        "freqs": freqs,
        "power": power,
    }


def frequency_domain_hrv(beat_times_s, cfg: Config):
    """LF/HF band power from the IBI series, spline-resampled to an even grid."""
    bt = np.asarray(beat_times_s, float)
    if len(bt) < 4:
        return {"error": "too few beats for frequency-domain HRV"}
    ibi_s = np.diff(bt)
    ibi_t = bt[1:]

    duration = bt[-1] - bt[0]
    reliable = duration >= cfg.fd_hrv_min_reliable_s

    from scipy.interpolate import CubicSpline
    if len(ibi_t) < 4:
        return {"error": "too few intervals for spline resampling"}
    cs = CubicSpline(ibi_t, ibi_s)
    grid_fs = cfg.fd_resample_hz
    grid_t = np.arange(ibi_t[0], ibi_t[-1], 1.0 / grid_fs)
    if len(grid_t) < 8:
        return {"error": "resampled grid too short"}
    grid_ibi = cs(grid_t) * 1000.0
    grid_ibi = grid_ibi - np.mean(grid_ibi)

    nper = min(len(grid_ibi), int(grid_fs * 60))
    freqs, power = sg.welch(grid_ibi, fs=grid_fs, nperseg=max(8, nper))

    def band_power(lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        if not np.any(m):
            return 0.0
        _trap = getattr(np, "trapezoid", None) or np.trapz
        return float(_trap(power[m], freqs[m]))

    vlf = band_power(*cfg.fd_vlf_band)
    lf = band_power(*cfg.fd_lf_band)
    hf = band_power(*cfg.fd_hf_band)
    lf_hf = float(lf / hf) if hf > 1e-9 else float("inf")
    total_lh = lf + hf
    lf_nu = float(100.0 * lf / total_lh) if total_lh > 1e-9 else float("nan")
    hf_nu = float(100.0 * hf / total_lh) if total_lh > 1e-9 else float("nan")

    return {
        "vlf_power_ms2": vlf,
        "lf_power_ms2": lf,
        "hf_power_ms2": hf,
        "lf_hf_ratio": lf_hf,
        "lf_nu": lf_nu,
        "hf_nu": hf_nu,
        "fd_hrv_reliable": bool(reliable),
        "recording_duration_s": float(duration),
        "fd_hrv_min_reliable_s": cfg.fd_hrv_min_reliable_s,
        "freqs": freqs,
        "power": power,
    }


# Template bootstrap
def _bootstrap_template(y, fs, cfg: Config, seg_start=0):
    """Seed mu from the densest cluster of short windows in a 15 s segment,
    re-centred on each member's local peak. Returns mu or None."""
    seg_n = int(cfg.boot_segment_s * fs)
    seg = y[seg_start: seg_start + seg_n]
    if len(seg) < seg_n // 2:
        return None

    win_n = int(cfg.boot_window_s * fs)
    half = win_n // 2

    dseg = np.diff(seg)
    ext, _ = sg.find_peaks(np.abs(dseg), distance=max(1, int(0.12 * fs)))
    ext = ext[(ext - half >= 0) & (ext + half < len(seg))]
    if len(ext) < cfg.boot_cluster_size + 1:
        return None

    W = np.array([seg[c - half: c + half] for c in ext])
    Wn = W - W.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Wn, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Wn = Wn / norms

    if len(Wn) < 2:
        return None
    D = pdist(Wn, metric="correlation")
    Z = linkage(D, method="complete")
    best_idx = _densest_cluster(Z, squareform(D), cfg.boot_cluster_size)
    if best_idx is None or len(best_idx) < cfg.boot_cluster_size:
        return None

    L = int(cfg.template_len_s * fs)
    half_L = L // 2
    stack = []
    for ci in best_idx:
        c = ext[ci]
        r = int(0.30 * fs)
        a0, b0 = max(0, c - r), min(len(seg), c + r)
        pk = a0 + int(np.argmax(seg[a0:b0]))
        a, b = pk - half_L, pk + half_L
        if a >= 0 and b < len(seg):
            w = seg[a:b]
            n = np.linalg.norm(w - w.mean())
            if n > 0:
                stack.append((w - w.mean()) / n)
    if len(stack) < 3:
        return None
    mu = np.mean(np.array(stack), axis=0)
    mu -= mu.mean()
    return mu


def _densest_cluster(Z, Dsq, k):
    """Indices of the tightest cluster with at least k members."""
    n = Dsq.shape[0]
    best, best_diam = None, np.inf
    for thr in np.unique(Z[:, 2]):
        labels = fcluster(Z, t=thr, criterion="distance")
        for lab in np.unique(labels):
            members = np.flatnonzero(labels == lab)
            if len(members) >= k:
                diam = Dsq[np.ix_(members, members)].max()
                if diam < best_diam:
                    best_diam, best = diam, members
        if best is not None:
            break
    return best


def _extract_between_central_minima(shape, hvs):
    """Cut shape to the span between the two HVS minima around centre."""
    c = len(hvs) // 2
    left = c
    for i in range(c, 0, -1):
        if hvs[i - 1] > hvs[i] and hvs[i] <= hvs[min(i + 1, len(hvs) - 1)]:
            left = i
            break
    right = c
    for i in range(c, len(hvs) - 1):
        if hvs[i + 1] > hvs[i] and hvs[i] <= hvs[max(i - 1, 0)]:
            right = i
            break
    if right - left < int(0.2 * len(shape)):
        w = len(shape) // 2
        left, right = max(0, c - w // 2), min(len(shape), c + w // 2)
    return shape[left:right]


# Candidate positions and interval acceptance
def _candidate_positions(y, mu, fs, cfg: Config):
    """Local maxima of the cross-correlation between y and the middle of mu."""
    mid_n = int(cfg.xcorr_mid_s * fs)
    anchor = _mu_anchor(mu)
    c0 = max(0, anchor - mid_n // 2)
    mid = mu[c0: c0 + mid_n]
    mid = mid - mid.mean()
    if np.linalg.norm(mid) == 0:
        return np.array([], dtype=int), np.zeros_like(y)
    xc = sg.correlate(y, mid, mode="same")
    cands, _ = sg.find_peaks(xc, distance=int(cfg.min_ibi * fs))
    return cands, xc


def _mu_anchor(mu):
    """Fiducial offset within mu (its argmax). Used everywhere mu is placed."""
    return int(np.argmax(mu))


def _fit_amplitude(y, mu, t0):
    """Least-squares amplitude of mu anchored at t0. Returns (z, (a, b))."""
    L = len(mu)
    off = _mu_anchor(mu)
    a, b = t0 - off, t0 - off + L
    if a < 0 or b > len(y):
        return None, None
    seg = y[a:b]
    denom = np.dot(mu, mu)
    if denom == 0:
        return 0.0, (a, b)
    z = np.dot(seg, mu) / denom
    return z, (a, b)


def _interval_residual(y, mu, ta, tb, cfg: Config):
    """Fit two copies of mu at ta and tb over the two-beat region.
    Returns (mse, msig, amp_ratio) or None if out of bounds."""
    L = len(mu)
    off = _mu_anchor(mu)
    za, ra = _fit_amplitude(y, mu, ta)
    zb, rb = _fit_amplitude(y, mu, tb)
    if za is None or zb is None:
        return None

    region_start = ta
    region_end = min(ta + 2 * (tb - ta), len(y))
    if region_end - region_start < 2:
        return None

    recon = np.zeros(region_end - region_start)
    idx = np.arange(region_start, region_end)
    ia = idx - (ta - off)
    va = (ia >= 0) & (ia < L)
    recon[va] += za * mu[ia[va]]
    ib = idx - (tb - off)
    vb = (ib >= 0) & (ib < L)
    recon[vb] += zb * mu[ib[vb]]

    obs = y[region_start:region_end]
    eps = obs - recon
    mse = float(np.mean(eps ** 2))
    msig = float(np.mean(obs ** 2))
    amps = sorted([abs(za), abs(zb)])
    amp_ratio = amps[1] / amps[0] if amps[0] > 1e-12 else np.inf
    return mse, msig, amp_ratio


def _detect_intervals(y, mu, fs, cfg: Config):
    """Greedy interval scan with periodic template re-estimation.
    Returns (beats, intervals, mu)."""
    cands, xc = _candidate_positions(y, mu, fs, cfg)
    if len(cands) == 0:
        return [], [], mu
    min_n = int(cfg.min_ibi * fs)
    max_n = int(cfg.max_ibi * fs)

    # calibrate the residual threshold from this recording
    resid_thr = cfg.residual_frac
    if cfg.auto_residual:
        ratios = []
        for i in range(len(cands) - 1):
            ta_i = int(cands[i])
            opts = cands[(cands >= ta_i + min_n) & (cands <= ta_i + max_n)]
            if len(opts):
                r = _interval_residual(y, mu, ta_i, int(opts[0]), cfg)
                if r:
                    ratios.append(r[0] / r[1])
        if len(ratios) >= 5:
            if cfg.auto_residual_mode == "mad":
                med = float(np.median(ratios))
                mad = 1.4826 * float(np.median(np.abs(np.asarray(ratios) - med)))
                resid_thr = med + cfg.auto_residual_mad_k * mad
            else:
                resid_thr = float(np.percentile(ratios, cfg.auto_residual_pctile))
            resid_thr = min(resid_thr, cfg.residual_frac)

    beats = [int(cands[0])]
    intervals = []
    ta = int(cands[0])
    accepted_since_reest = 0
    guard = 0
    n_total = len(y)

    def next_candidate_after(pos):
        nxt = cands[cands > pos]
        return int(nxt[0]) if len(nxt) else None

    while ta < n_total - min_n and guard < 200000:
        guard += 1
        lo, hi = ta + min_n, ta + max_n
        tb_opts = cands[(cands >= lo) & (cands <= hi)]

        # lowest-residual tb that passes both gates
        best_tb, best_mse_ratio = None, np.inf
        for tb in tb_opts:
            r = _interval_residual(y, mu, ta, int(tb), cfg)
            if r is None:
                continue
            mse, msig, amp_ratio = r
            ratio = mse / msig if msig > 0 else np.inf
            if ratio < resid_thr and amp_ratio < cfg.amp_ratio_max:
                if ratio < best_mse_ratio:
                    best_mse_ratio, best_tb = ratio, int(tb)

        if best_tb is not None:
            intervals.append((ta, best_tb))
            beats.append(best_tb)
            ta = best_tb
            accepted_since_reest += 1
        else:
            # skip ta without bridging the gap, so no giant IBI enters RMSSD
            nxt = next_candidate_after(ta)
            if nxt is None:
                break
            ta = nxt

        if accepted_since_reest >= cfg.reestimate_every:
            mu_new = _template_from_beats(y, beats, fs, cfg)
            if mu_new is not None:
                mu = mu_new
                cands, xc = _candidate_positions(y, mu, fs, cfg)
            accepted_since_reest = 0

    return sorted(set(beats)), intervals, mu


def _template_from_beats(y, beats, fs, cfg: Config):
    """Re-estimate mu as the normalised mean shape around accepted beats."""
    L = int(cfg.template_len_s * fs)
    off = L // 2
    stack = []
    for b in beats:
        a, e = b - off, b - off + L
        if a >= 0 and e < len(y):
            w = y[a:e]
            n = np.linalg.norm(w - w.mean())
            if n > 0:
                stack.append((w - w.mean()) / n)
    if len(stack) < 3:
        return None
    mu = np.mean(np.array(stack), axis=0)
    mu -= mu.mean()
    return mu


# Beat timing refinement
def _parabolic_refine(y, idx):
    """Sub-sample peak location by parabolic interpolation."""
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)
    y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if denom == 0:
        return float(idx)
    return idx + 0.5 * (y0 - y2) / denom


def _time_apex(y, b, fs, mu=None):
    """Fiducial = sub-sample signal apex within +-60 ms of b."""
    r = int(0.06 * fs)
    a, e = max(0, b - r), min(len(y), b + r)
    loc = a + int(np.argmax(y[a:e]))
    return _parabolic_refine(y, loc)


def _time_xcorr(y, b, fs, mu):
    """Fiducial = sub-sample lag of max cross-correlation with mu near b."""
    anchor = int(np.argmax(mu))
    L = len(mu)
    slide = int(0.12 * fs)
    a = max(0, b - anchor - slide)
    e = min(len(y), b - anchor + L + slide)
    seg = y[a:e]
    if len(seg) < L + 2:
        return _time_apex(y, b, fs)
    m = mu - mu.mean()
    xc = sg.correlate(seg - seg.mean(), m, mode="valid")
    k = int(np.argmax(xc))
    if 0 < k < len(xc) - 1:
        y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]
        den = y0 - 2 * y1 + y2
        k = k + (0.5 * (y0 - y2) / den if den != 0 else 0.0)
    return a + k + anchor


def _time_slope(y, b, fs, mu=None):
    """Fiducial = steepest point of the rising edge into the peak near b."""
    r = int(0.06 * fs)
    a, e = max(0, b - r), min(len(y), b + r)
    seg = y[a:e]
    apex = int(np.argmax(seg))
    if apex < 2:
        return _time_apex(y, b, fs)
    d = np.diff(seg[:apex + 1])
    loc = a + int(np.argmax(d))
    if a < loc < e - 2:
        dd = np.diff(y[max(1, loc - 1):loc + 2])
        di = loc - a
        if 0 < di < len(d) - 1:
            y0, y1, y2 = d[di - 1], d[di], d[di + 1]
            den = y0 - 2 * y1 + y2
            return loc + (0.5 * (y0 - y2) / den if den != 0 else 0.0)
    return float(loc)


_TIMING_METHODS = {"apex": _time_apex, "xcorr": _time_xcorr, "slope": _time_slope}


def _build_template_bank(y, beats, fs, cfg):
    """Cluster accepted beats into up to bank_k shape groups, one template each."""
    L = int(cfg.template_len_s * fs)
    half = L // 2
    stack = []
    for b in beats:
        a, e = b - half, b - half + L
        if a >= 0 and e < len(y):
            w = y[a:e]
            n = np.linalg.norm(w - w.mean())
            if n > 0:
                stack.append((w - w.mean()) / n)
    if len(stack) < cfg.bank_k * 2:
        if len(stack) >= 2:
            mu = np.mean(np.array(stack), axis=0); mu -= mu.mean()
            return [mu]
        return None
    X = np.array(stack)
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    D = pdist(X, metric="correlation")
    Z = linkage(D, method="average")
    labels = fcluster(Z, t=cfg.bank_k, criterion="maxclust")
    bank = []
    for lab in np.unique(labels):
        grp = X[labels == lab]
        if len(grp) >= cfg.bank_min_per_template:
            mu = grp.mean(axis=0); mu -= mu.mean()
            bank.append(mu)
    return bank if bank else None


def _time_xcorr_bank(y, b, fs, bank):
    """Fiducial from whichever bank template correlates best with the beat."""
    best_time, best_corr = None, -np.inf
    for mu in bank:
        anchor = int(np.argmax(mu)); L = len(mu)
        slide = int(0.12 * fs)
        a = max(0, b - anchor - slide)
        e = min(len(y), b - anchor + L + slide)
        seg = y[a:e]
        if len(seg) < L + 2:
            continue
        m = mu - mu.mean()
        xc = sg.correlate(seg - seg.mean(), m, mode="valid")
        k = int(np.argmax(xc))
        peak_corr = xc[k] / (np.linalg.norm(seg) * np.linalg.norm(m) + 1e-12)
        if peak_corr > best_corr:
            kk = float(k)
            if 0 < k < len(xc) - 1:
                y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]
                den = y0 - 2 * y1 + y2
                kk = k + (0.5 * (y0 - y2) / den if den != 0 else 0.0)
            best_corr = peak_corr
            best_time = a + kk + anchor
    return best_time


# HRV metrics
def hrv_metrics(ibi_ms, cfg: Config):
    """Time-domain HRV after a physiological clamp and NN artifact correction."""
    ibi = np.asarray(ibi_ms, float)
    n_raw = len(ibi)

    keep = (ibi >= cfg.hrv_min_ms) & (ibi <= cfg.hrv_max_ms)
    ibi = ibi[keep]
    if len(ibi) < 3:
        return {"n_intervals": int(len(ibi)), "error": "too few intervals"}

    med = _running_median(ibi, win=11)
    ok = np.abs(ibi - med) <= cfg.nn_reject_frac * med
    ibi_c = ibi[ok]
    if len(ibi_c) < 3:
        ibi_c = ibi

    diffs = np.diff(ibi_c)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    sdnn = float(np.std(ibi_c, ddof=1))
    pnn50 = float(100.0 * np.mean(np.abs(diffs) > 50.0))
    return {
        "n_intervals": int(len(ibi_c)),
        "n_intervals_raw": int(n_raw),
        "n_rejected": int(n_raw - len(ibi_c)),
        "mean_ibi_ms": float(np.mean(ibi_c)),
        "mean_hr_bpm": float(60000.0 / np.mean(ibi_c)),
        "rmssd_ms": rmssd,
        "sdnn_ms": sdnn,
        "pnn50_pct": pnn50,
        "ln_rmssd": float(np.log(rmssd)) if rmssd > 0 else float("nan"),
    }


def _running_median(x, win=11):
    """Centred running median, reflect-padded."""
    n = len(x)
    h = win // 2
    xp = np.pad(x, h, mode="reflect")
    return np.array([np.median(xp[i:i + win]) for i in range(n)])


# Top-level driver
def analyze(paths_or_signal, fs=None, cfg: Config = None,
            t0_s=0.0, t1_s=None, channel_col=1, return_detail=False):
    """Run the full pipeline on CSV paths or a raw signal array (needs fs).
    Returns a metrics dict, plus a detail dict if return_detail."""
    cfg = cfg or Config()

    if isinstance(paths_or_signal, np.ndarray):
        assert fs is not None, "fs required when passing a raw signal array"
        x = paths_or_signal.astype(float)
        t = np.arange(len(x)) / fs
    else:
        t, x, fs = load_picoscope_csvs(paths_or_signal, channel_col=channel_col)

    if t1_s is None:
        t1_s = t[-1]
    win = (t >= t0_s) & (t <= t1_s)
    t_w, x_w = t[win], x[win]

    y = preprocess(x_w, fs, cfg)

    # spectral estimate: always computed for the cross-check, only narrows the
    # search range when use_spectral_seed is on
    spec = spectral_hr_estimate(y, fs, cfg)
    if cfg.use_spectral_seed:
        if "error" not in spec and spec["peak_hz"] > 0:
            period = 1.0 / spec["peak_hz"]
            cfg = replace(cfg,
                          min_ibi=max(0.30, period * (1 - cfg.spectral_seed_halfwidth)),
                          max_ibi=min(2.00, period * (1 + cfg.spectral_seed_halfwidth)))

    if cfg.use_acf_seed and not cfg.use_spectral_seed:
        period = acf_period(x_w, fs, cfg)
        if period:
            cfg = replace(cfg,
                          min_ibi=max(0.30, period * (1 - cfg.acf_seed_halfwidth)),
                          max_ibi=min(2.00, period * (1 + cfg.acf_seed_halfwidth)))

    # bootstrap template, trying later segments if the first fails
    mu = None
    for off in range(0, max(1, len(y) - int(cfg.boot_segment_s * fs)),
                      int(cfg.boot_segment_s * fs)):
        mu = _bootstrap_template(y, fs, cfg, seg_start=off)
        if mu is not None and np.linalg.norm(mu) > 0:
            break
    if mu is None:
        return {"error": "could not bootstrap a heartbeat template"}

    beats, intervals, mu = _detect_intervals(y, mu, fs, cfg)
    if len(intervals) < 2:
        return {"error": "no intervals detected"}

    if cfg.timing_method == "xcorr_bank":
        bank = _build_template_bank(y, beats, fs, cfg)
        if bank and len(bank) >= 1:
            timing_fn = lambda yy, bb, ff, mm: _time_xcorr_bank(yy, bb, ff, bank)
        else:
            timing_fn = _time_xcorr
    else:
        timing_fn = _TIMING_METHODS.get(cfg.timing_method, _time_apex)
    refine_cache = {}

    def refined_time(b):
        if b not in refine_cache:
            idx = timing_fn(y, b, fs, mu)
            if idx is None:
                idx = _time_apex(y, b, fs)
            refine_cache[b] = t_w[0] + idx / fs
        return refine_cache[b]

    # IBIs only from accepted consecutive pairs; skipped gaps are not bridged
    ibi_ms = np.array([(refined_time(tb) - refined_time(ta)) * 1000.0
                       for (ta, tb) in intervals])
    beat_times = np.array(sorted({refined_time(b) for b in beats}))

    out = hrv_metrics(ibi_ms, cfg)
    out["coverage_pct"] = float(100.0 * len(intervals) /
                                max(1, len(beats) - 1))
    out["detected_intervals"] = int(len(intervals))
    out["fs_hz"] = float(fs)
    out["window_s"] = float(t_w[-1] - t_w[0])

    # spectral cross-check against the time-domain HR
    if "error" not in spec:
        out["spectral_hr_bpm"] = spec["peak_bpm"]
        out["spectral_band_power_frac"] = spec["band_power_frac"]
        if "mean_hr_bpm" in out:
            diff = abs(out["mean_hr_bpm"] - spec["peak_bpm"])
            out["spectral_hr_diff_bpm"] = float(diff)
            out["spectral_check_ok"] = bool(diff <= cfg.spectral_disagreement_bpm)
            if diff > cfg.spectral_disagreement_bpm:
                out["spectral_warning"] = (
                    f"Time-domain HR ({out['mean_hr_bpm']:.1f} bpm) disagrees with "
                    f"independent spectral HR ({spec['peak_bpm']:.1f} bpm) by "
                    f"{diff:.1f} bpm; detector may have locked onto a harmonic "
                    f"or be missing/doubling beats. Treat RMSSD with caution."
                )
    else:
        out["spectral_hr_bpm"] = None

    if len(beat_times) >= 4:
        fdh = frequency_domain_hrv(beat_times, cfg)
        if "error" not in fdh:
            out["fd_hrv"] = {k: v for k, v in fdh.items()
                             if k not in ("freqs", "power")}

    if return_detail:
        detail = dict(beat_times=beat_times, ibi_ms=ibi_ms, intervals=intervals,
                      fs=fs, y=y, t=t_w, mu=mu)
        if "error" not in spec:
            detail["spectral_freqs"] = spec["freqs"]
            detail["spectral_power"] = spec["power"]
        return out, detail
    return out


# CLI
if __name__ == "__main__":
    import sys, argparse, json
    ap = argparse.ArgumentParser(description="Paalasmaa BCG beat-to-beat / HRV.")
    ap.add_argument("csv", nargs="+", help="PicoScope CSV file(s), in time order")
    ap.add_argument("--t0", type=float, default=0.0, help="window start (s)")
    ap.add_argument("--t1", type=float, default=None, help="window end (s)")
    ap.add_argument("--col", type=int, default=1, help="BCG channel column (1=A)")
    ap.add_argument("--plot", action="store_true", help="save detection plot")
    args = ap.parse_args()

    res, det = analyze(sorted(args.csv), t0_s=args.t0, t1_s=args.t1,
                       channel_col=args.col, return_detail=True)
    print(json.dumps(res, indent=2))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        y, t, bt = det["y"], det["t"], det["beat_times"]
        fig, ax = plt.subplots(2, 1, figsize=(15, 7))
        ax[0].plot(t, y * 1000, lw=0.6, color="navy")
        ax[0].plot(bt, np.interp(bt, t, y) * 1000, "o", ms=7, mfc="none",
                   mec="red", mew=1.8)
        ax[0].set_title("Paalasmaa beat detections"); ax[0].set_ylabel("mV")
        ibi = det["ibi_ms"]
        ax[1].plot(bt[1:], ibi, "o-", ms=4)
        ax[1].set_title("IBI tachogram"); ax[1].set_xlabel("s"); ax[1].set_ylabel("ms")
        plt.tight_layout(); plt.savefig("bcg_paalasmaa_detection.png", dpi=90)
        print("saved bcg_paalasmaa_detection.png")
