"""Milestone 2 - Event Detection (pure signal functions, no file I/O).

Detect switch-on / switch-off events in a PAC4200 active-power signal.

Design:
  - Threshold relative to the OFF-baseline, not the peak, so it catches loads
    from 6 W (LED) to 1.4 kW (toaster).
  - Smooth -> threshold -> fill short gaps (keeps the mixer's two plateaus and
    noisy small loads as one region) -> keep regions above a minimum duration.
  - detect_active_region(): the single longest ON region (used for fingerprints
    and single-cycle feature extraction).
  - detect_all_regions(): ALL qualifying ON regions, so multi-cycle lab files
    (water heater, fans cycling several times) yield multiple real samples.
  - Negative-power safety: detection uses |P| above baseline, so generation
    (PV) does not crash the detector (PV is excluded from the classifier anyway).
"""

import numpy as np


def _smooth(x, win):
    if win <= 1:
        return x
    return np.convolve(x, np.ones(win) / win, mode="same")


def _regions(df, baseline_secs, k_noise, abs_floor_w, min_on_secs,
             max_gap_secs, smooth_secs):
    """Return (list_of_regions, info). Each region is (on_idx, off_idx)."""
    p = df["p_total_w"].fillna(0).values
    t = df["t_sec"].values
    sps = max(1, round(1.0 / np.median(np.diff(t)))) if len(t) > 1 else 5

    n_base = max(3, int(baseline_secs * sps))
    base_seg = p[:n_base]
    baseline = float(np.median(base_seg))
    noise = float(np.std(base_seg))
    threshold = max(k_noise * noise, abs_floor_w)  # relative to baseline magnitude

    win = max(1, int(smooth_secs * sps) | 1)
    p_dev = np.abs(_smooth(p, win) - baseline)     # deviation from baseline (sign-safe)
    above = p_dev > threshold

    # fill short gaps
    max_gap = int(max_gap_secs * sps)
    filled = above.copy()
    n = len(above)
    i = 0
    while i < n:
        if not filled[i]:
            j = i
            while j < n and not filled[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                filled[i:j] = True
            i = j
        else:
            i += 1

    # collect regions >= min duration
    min_on = int(min_on_secs * sps)
    regions = []
    i = 0
    while i < n:
        if filled[i]:
            j = i
            while j < n and filled[j]:
                j += 1
            if (j - i) >= min_on:
                regions.append((i, j - 1))
            i = j
        else:
            i += 1

    info = {"baseline_w": round(baseline, 3), "noise_w": round(noise, 3),
            "threshold_w": round(threshold, 3), "sps": sps,
            "n_candidate_regions": len(regions)}
    return regions, info


def detect_all_regions(df, baseline_secs=3.0, k_noise=6.0, abs_floor_w=3.0,
                       min_on_secs=3.0, max_gap_secs=2.0, smooth_secs=0.6):
    """All qualifying ON regions as a list of dicts (on_idx, off_idx, on_t, off_t)."""
    regions, info = _regions(df, baseline_secs, k_noise, abs_floor_w,
                             min_on_secs, max_gap_secs, smooth_secs)
    t = df["t_sec"].values
    out = []
    for on_i, off_i in regions:
        out.append({"on_idx": int(on_i), "off_idx": int(off_i),
                    "on_t": round(float(t[on_i]), 2),
                    "off_t": round(float(t[off_i]), 2),
                    "baseline_w": info["baseline_w"], "sps": info["sps"]})
    return out


def detect_active_region(df, **kw):
    """The single longest ON region (dict), or None-filled dict if none."""
    regions, info = _regions(
        df,
        kw.get("baseline_secs", 3.0), kw.get("k_noise", 6.0),
        kw.get("abs_floor_w", 3.0), kw.get("min_on_secs", 3.0),
        kw.get("max_gap_secs", 2.0), kw.get("smooth_secs", 0.6))
    res = dict(info)
    res.update({"on_idx": None, "off_idx": None, "on_t": None, "off_t": None})
    if regions:
        on_i, off_i = max(regions, key=lambda r: r[1] - r[0])
        t = df["t_sec"].values
        res.update({"on_idx": int(on_i), "off_idx": int(off_i),
                    "on_t": round(float(t[on_i]), 2),
                    "off_t": round(float(t[off_i]), 2)})
    return res
