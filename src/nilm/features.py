"""M2 Stage 2: Feature Extraction (data-driven, multi-cycle).

For each recording, detect the active region(s) and compute 12 electrical
features per region. KEY CHANGE for the lab data: real files that cycle several
times yield ONE feature row PER cycle, so the real test set grows from ~6 to
~30 samples (much less coarse evaluation).

Feature set (12): p_mean, q_mean, q_sign, qp_ratio, thd_i_mean, inrush_ratio,
p_std, thd_u_mean, settling_s, crest_factor, n_state_changes, s_mean.
Excluded: power factor (train/test mismatch), energy (recording-length leak).
"""

import glob
import numpy as np
import pandas as pd

from . import paths, registry, events

FEATURE_COLS = [
    "p_mean", "q_mean", "q_sign", "qp_ratio", "thd_i_mean",
    "inrush_ratio", "p_std", "thd_u_mean", "settling_s", "crest_factor",
    "n_state_changes", "s_mean",
]


def _count_state_changes(p_on, sps, p_mean, min_dwell_s=2.0, rel_jump=0.25):
    if len(p_on) < 3 * sps or p_mean < 1e-6:
        return 0
    win = max(1, int(min_dwell_s * sps))
    n_blocks = len(p_on) // win
    if n_blocks < 2:
        return 0
    bm = [np.nanmean(p_on[b * win:(b + 1) * win]) for b in range(n_blocks)]
    return sum(1 for a, b in zip(bm[:-1], bm[1:]) if abs(b - a) > rel_jump * p_mean)


def features_for_region(df, region, settle_s=1.0):
    """12-feature dict for one ON region (region from events.detect_*)."""
    sps = region["sps"]
    on_i, off_i = region["on_idx"], region["off_idx"]
    p = df["p_total_w"].values; q = df["q_total_var"].values
    s = df["s_total_va"].values; i_a = df["i_l1_a"].values
    thd_i = df["thd_i_l1_percent"].values; thd_u = df["thd_u_l1_percent"].values

    settle = on_i + int(settle_s * sps)
    ss = slice(settle, off_i) if settle < off_i else slice(on_i, off_i)

    def nm(x):
        v = x[ss]; v = v[~np.isnan(v)]
        return float(v.mean()) if len(v) else 0.0

    p_mean, q_mean, s_mean = nm(p), nm(q), nm(s)
    thd_i_mean, thd_u_mean = nm(thd_i), nm(thd_u)
    p_ss = p[ss]; p_ss = p_ss[~np.isnan(p_ss)]
    p_std = float(p_ss.std()) if len(p_ss) else 0.0

    q_sign = 1 if q_mean > 5 else (-1 if q_mean < -5 else 0)
    qp_ratio = float(q_mean / p_mean) if abs(p_mean) > 1e-6 else 0.0

    iw = p[on_i:on_i + sps]; iw = iw[~np.isnan(iw)]
    inrush_peak = float(iw.max()) if len(iw) else p_mean
    inrush_ratio = float(inrush_peak / p_mean) if abs(p_mean) > 1e-6 else 1.0

    target = 0.9 * p_mean
    seg = p[on_i:off_i]; reach = np.where(seg >= target)[0]
    settling_s = float(reach[0] / sps) if len(reach) else 0.0

    i_ss = i_a[ss]; i_ss = i_ss[~np.isnan(i_ss)]
    crest = float(i_ss.max() / i_ss.mean()) if len(i_ss) and i_ss.mean() > 1e-6 else 1.0

    nsc = _count_state_changes(p[on_i:off_i], sps, p_mean)

    return {"p_mean": round(p_mean, 3), "q_mean": round(q_mean, 3),
            "q_sign": q_sign, "qp_ratio": round(qp_ratio, 4),
            "thd_i_mean": round(thd_i_mean, 3), "inrush_ratio": round(inrush_ratio, 3),
            "p_std": round(p_std, 3), "thd_u_mean": round(thd_u_mean, 3),
            "settling_s": round(settling_s, 2), "crest_factor": round(crest, 3),
            "n_state_changes": int(nsc), "s_mean": round(s_mean, 3)}


def extract_features(df, settle_s=1.0):
    """Single main-region features (used for synthetic single-cycle files)."""
    r = events.detect_active_region(df)
    if r["on_idx"] is None:
        return None
    return features_for_region(df, r, settle_s)


def features_all_cycles(df, settle_s=1.0, min_cycle_s=4.0):
    """One feature dict per ON cycle (used for real multi-cycle files)."""
    out = []
    for r in events.detect_all_regions(df):
        if (r["off_t"] - r["on_t"]) < min_cycle_s:
            continue
        out.append(features_for_region(df, r, settle_s))
    return out


# ----------------------------------------------------------------------
def build_feature_table(file_label_pairs, settle_s=1.0):
    """Single-region table: one row per file. (Used for synthetic data.)"""
    rows = []
    for path, label in file_label_pairs:
        feats = extract_features(registry.load_pac(path), settle_s)
        if feats is None:
            continue
        feats["label"] = label
        feats["source_file"] = str(path).split("/")[-1]
        rows.append(feats)
    return pd.DataFrame(rows)


def real_feature_table(settle_s=1.0):
    """Real appliances, ONE ROW PER CYCLE (the multi-sample test set)."""
    rows = []
    for device, path in registry.discover().items():
        df = registry.load_pac(path)
        cyc = features_all_cycles(df, settle_s)
        for k, feats in enumerate(cyc):
            feats = dict(feats)
            feats["label"] = device
            feats["source_file"] = path.name
            feats["cycle"] = k
            rows.append(feats)
    return pd.DataFrame(rows)


def real_pairs():
    """One (path, device) per appliance (single-region; kept for compatibility)."""
    return [(p, d) for d, p in registry.discover().items()]


def synth_pairs(split="training", per_device=None):
    pairs = []
    tag = "train" if split == "training" else "test"
    for device in registry.discover():
        slug = registry.slug(device)
        files = sorted(glob.glob(str(paths.SYNTH / split / f"pac4200_{slug}_{tag}_*.csv")))
        if per_device:
            files = files[:per_device]
        pairs.extend([(f, device) for f in files])
    return pairs


def _demo():
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print("REAL feature table (one row per cycle):")
    real = real_feature_table()
    print(f"  {len(real)} real samples from {real['label'].nunique()} appliances")
    print(real.groupby("label").size().to_string())


if __name__ == "__main__":
    _demo()
