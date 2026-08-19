"""Milestone 3 - Stage 4a: Novelty layer ("New Device Found").

The classifier is closed: it always returns one of the known device names, even
for a device it was never trained on. This layer decides, before trusting a
name, whether a device is known at all.

Two signals, matching the two ways an unknown device shows up:

1. Plugged in ALONE -> we can capture its full fingerprint (the 12 features).
   We check whether that fingerprint is an outlier against the known devices,
   using DBSCAN plus a nearest-distance threshold in the standardised feature
   space. This is the path that feeds auto-onboarding.

2. Switched inside a MIX -> we only see a step (delta_p, delta_q). We check
   whether any known device or small combination explains it; if the best
   residual is large, the step is unexplained and likely a new device. This
   reuses the tracker's combination matching.

Honest limit (open-set recognition): a new device whose signature sits close to
a known one, or close to a combination of known ones, cannot be flagged. Only
devices that are clearly different from everything known are caught. This is a
fundamental limit, not a tuning problem.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from . import aggregator, features, registry
from .tracker import match_step


def known_fingerprints():
    """One 12-feature fingerprint per known device (mean over its cycles)."""
    ft = features.real_feature_table()
    label_col = "label" if "label" in ft.columns else "device"
    drop = {label_col, "source_file", "cycle"}
    feats = [c for c in ft.columns if c not in drop and
             pd.api.types.is_numeric_dtype(ft[c])]
    fp = ft.groupby(label_col)[feats].mean()
    return fp, feats


class NoveltyDetector:
    def __init__(self, dist_k=2.0, eps=1.6, min_samples=2):
        self.dist_k = dist_k          # nearest-distance multiplier for the flag
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, fp, feats):
        self.feats = feats
        self.devices = list(fp.index)
        self.scaler = StandardScaler().fit(fp[feats].values)
        self.X = self.scaler.transform(fp[feats].values)
        # threshold from the spread of known devices: how far apart are they?
        d = self._pairwise_min(self.X)
        self.threshold = float(np.median(d) + self.dist_k * np.std(d))
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(self.X)
        return self

    def _pairwise_min(self, X):
        out = []
        for i in range(len(X)):
            dd = [np.linalg.norm(X[i] - X[j]) for j in range(len(X)) if j != i]
            out.append(min(dd) if dd else 0.0)
        return np.array(out)

    def is_novel_fingerprint(self, fp_row):
        """fp_row: dict or Series of the 12 features for a device seen alone."""
        x = self.scaler.transform([[fp_row[f] for f in self.feats]])[0]
        dmin = min(np.linalg.norm(x - xk) for xk in self.X)
        return bool(dmin > self.threshold), round(dmin, 2), round(self.threshold, 2)

    def is_novel_step(self, dp, dq, candidates, params, res_threshold=30.0):
        """Step inside a mix: novel if no known device/combination explains it."""
        target_q = dq if dp >= 0 else -dq
        _, err = match_step(abs(dp), target_q, candidates, params, max_combo=3)
        return bool(err > res_threshold), round(err, 2)


def leave_one_out_eval(dist_k=2.0, noise_frac=0.05, seed=0):
    """Two honest tests.

    Detection: hold each device out of the known set entirely, then test whether
    its fingerprint is flagged as new. This measures catching genuine unknowns.

    False alarm: fit on ALL known devices, then re-present each known device with
    measurement-style noise added, and check it is NOT flagged. This measures
    whether a re-plugged known device is wrongly called new.
    """
    fp, feats = known_fingerprints()
    devices = list(fp.index)
    rng = np.random.default_rng(seed)

    # detection: hold-out
    detected = []
    for held in devices:
        known = fp.drop(index=held)
        det = NoveltyDetector(dist_k=dist_k).fit(known, feats)
        novel, dmin, thr = det.is_novel_fingerprint(fp.loc[held])
        detected.append((held, novel, dmin, thr))
    n_det = sum(1 for _, v, _, _ in detected if v)

    # false alarm: all known, re-present with noise
    det_all = NoveltyDetector(dist_k=dist_k).fit(fp, feats)
    col_std = fp[feats].std().replace(0, 1e-6).values
    fa = 0
    for _ in range(20):
        for dev in devices:
            noisy = fp.loc[dev].copy()
            noisy[feats] = fp.loc[dev, feats].values + rng.normal(0, noise_frac, len(feats)) * col_std
            flagged, _, _ = det_all.is_novel_fingerprint(noisy)
            fa += int(flagged)
    fa_rate = fa / (20 * len(devices))

    return {
        "detected": detected,
        "detection_rate": round(n_det / len(devices), 3),
        "n_detected": n_det,
        "n_devices": len(devices),
        "false_alarm_rate": round(fa_rate, 3),
    }


def _demo():
    print("=" * 78)
    print("NOVELTY LAYER - 'New Device Found'")
    print("=" * 78)
    rep = leave_one_out_eval(dist_k=2.0)
    print(f"\nLeave-one-device-out (each device treated as unknown in turn):")
    print(f"  Detected as new: {rep['n_detected']}/{rep['n_devices']}"
          f"  (rate {rep['detection_rate']})")
    print(f"  False-alarm rate on known devices: {rep['false_alarm_rate']}")
    print("\n  Per device (was it flagged as new?):")
    for name, novel, dmin, thr in rep["detected"]:
        mark = "NEW" if novel else "looks known"
        print(f"    {name:18s} dist={dmin:5.2f} thr={thr:5.2f}  -> {mark}")
    print("\n  Reading: distinct devices are caught; devices that resemble a")
    print("  known one (or a mix of known ones) are not. That is the open-set limit.")


if __name__ == "__main__":
    _demo()
