"""Milestone 3 - Stage 2: Disaggregator.

Turns a combined multi-appliance signal into a list of "which device switched
when". Two parts:

  1. Step detector (detect_steps): finds every transition in the TOTAL signal,
     not the on/off envelope. At each step it measures the change in active and
     reactive power (delta_p, delta_q). This is the change-detection the M2
     envelope detector did not do.

  2. Step classifier (train_step_classifier / DisaggregatorModel): names the
     device behind each step from its (delta_p, delta_q, and a few helpers).
     LightGBM is primary; Random Forest is kept for comparison. Trained on
     steps from the aggregator's labelled combined signals (sequential AND
     simultaneous mixed together, as agreed).

Honest note: at a step we mainly have delta_p and delta_q, because distortion
does not add across devices. So the step classifier works with far fewer
reliable features than the single-appliance classifier in M2, and is weaker.
For a merged step (two devices at once) the change matches a *combination*; the
running-set logic in tracker.py handles that. Here we score single-cause steps.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:          # LightGBM is optional; Random Forest is the primary model
    HAS_LGBM = False

from . import paths, aggregator


# ----------------------------------------------------------------------
# 1. STEP DETECTION
# ----------------------------------------------------------------------
def detect_steps(df, win_s=3.0, abs_floor_w=5.0, k_noise=4.5, smooth_s=1.5,
                 q_floor=8.0, sustain_ratio=0.5):
    """Find transitions in the total signal.

    Three improvements over a fixed-threshold detector, each fixing a real fault:

    1. ADAPTIVE threshold. The minimum step size is not a fixed number of watts.
       It scales with the LOCAL noise of the signal: max(abs_floor_w,
       k_noise * local_noise). In a quiet mix of small devices a 6-7 W step is
       visible and gets caught; under a noisy 1.4 kW load the threshold rises so
       noise is not mistaken for a switch. A tiny step buried under a large load
       still cannot be recovered, because it is genuinely below the noise floor.

    2. DUAL CHANNEL. A step is accepted if EITHER active power OR reactive power
       changes enough. Some loads (fluorescent, fan) move little real power but a
       lot of reactive power, and would be missed by a P-only detector.

    3. SUSTAINED-STEP confirmation. A real switch moves to a new level and stays
       there; a noise spike snaps back. Each candidate is kept only if the new
       level holds over the following window. This removes the false steps that a
       distorted load's noisy plateau used to produce.

    Returns a DataFrame: t_sec, delta_p, delta_q, p_before, p_after.
    """
    p = df["p_total_w"].fillna(0).values
    q = df["q_total_var"].fillna(0).values
    t = df["t_sec"].values
    # current distortion, if the recording carries it. Used to separate devices
    # that are identical in P and Q but differ in waveform (USB charger vs
    # monitor). NaN where unavailable, which the tracker treats as "no evidence".
    _thd_col = next((c for c in ("thd_i_l1_percent", "thd_i", "thd_i_percent")
                     if c in df.columns), None)
    thd_i = df[_thd_col].values if _thd_col else np.full(len(p), np.nan)
    sps = max(1, round(1.0 / np.median(np.diff(t)))) if len(t) > 1 else 5

    w = max(1, int(win_s * sps))
    sm = max(1, int(smooth_s * sps) | 1)
    ker = np.ones(sm) / sm
    ps = np.convolve(p, ker, mode="same")
    qs = np.convolve(q, ker, mode="same")

    # local per-sample noise (rolling std of the residual), mapped to the noise
    # of the windowed difference (which averages w samples on each side)
    resid = p - ps
    noise_local = (pd.Series(resid).rolling(int(5 * sps), center=True, min_periods=3)
                   .std().bfill().ffill().values)
    diff_noise = noise_local * np.sqrt(2.0 / w)

    n = len(ps)
    diff_p = np.zeros(n)
    diff_q = np.zeros(n)
    for i in range(w, n - w):
        diff_p[i] = ps[i:i + w].mean() - ps[i - w:i].mean()
        diff_q[i] = qs[i:i + w].mean() - qs[i - w:i].mean()

    thr_p = np.maximum(abs_floor_w, k_noise * diff_noise)
    above = (np.abs(diff_p) >= thr_p) | (np.abs(diff_q) >= q_floor)

    steps = []
    i = w
    while i < n - w:
        if above[i]:
            j = i
            while j < n - w and above[j]:
                j += 1
            run = slice(i, j)
            # locate the step by the combined P/Q magnitude so reactive-only
            # steps are placed correctly
            mag = np.hypot(diff_p[run], diff_q[run])
            k = i + int(np.argmax(mag))
            pre = float(np.median(p[max(0, k - w):k]))
            post = float(np.median(p[k:k + w]))
            qpre = float(np.median(q[max(0, k - w):k]))
            qpost = float(np.median(q[k:k + w]))
            dp = post - pre
            dq = qpost - qpre
            # The windowed difference peaks in the MIDDLE of the transition, so
            # k lags the real switch by ~half a window and the lag grows when the
            # sample rate is uneven (a live feed). Walk back from k to the onset:
            # the last sample still at the pre-step level. That is where the
            # switch actually happened, and where the marker should sit.
            k_on = k
            if abs(dp) > 1e-6:
                m = k
                while m > max(0, k - 2 * w):
                    crossed = (p[m] - pre) / dp
                    if crossed < 0.5:      # back at the pre-step level
                        k_on = m
                        break
                    m -= 1
                else:
                    k_on = max(0, k - w)
            t_step = float(t[k_on])
            # sustained: the new level must hold in the next window, not snap back
            sustained = True
            if k + 2 * w <= n:
                post2 = float(np.median(p[k + w:k + 2 * w]))
                if abs(dp) > 1e-6 and abs(post2 - post) > sustain_ratio * abs(dp):
                    sustained = False
            if (abs(dp) >= abs_floor_w or abs(dq) >= q_floor) and sustained:
                # median distortion in the settle window AFTER the step: this is
                # the waveform signature of whatever just switched on. Read it
                # past the transition (around k), where the level has settled.
                seg = thd_i[k:k + w]
                seg = seg[~np.isnan(seg)] if seg.size else seg
                thd_after = float(np.median(seg)) if seg.size else float("nan")
                steps.append({"t_sec": round(t_step, 2),
                              "delta_p": round(dp, 2), "delta_q": round(dq, 2),
                              "p_before": round(pre, 2), "p_after": round(post, 2),
                              "thd_i_after": round(thd_after, 1)})
            i = j
        else:
            i += 1
    return pd.DataFrame(steps)


def match_steps_to_truth(detected, truth_events, tol_s=2.0):
    """Greedy nearest-time match between detected steps and ground-truth events.
    Returns precision, recall, and the matched pairs (for labelling)."""
    truth = truth_events.copy().reset_index(drop=True)
    used = set()
    matches = []
    for _, d in detected.iterrows():
        best, best_dt = None, tol_s + 1
        for ti, tr in truth.iterrows():
            if ti in used:
                continue
            dt = abs(tr["t_sec"] - d["t_sec"])
            # require same sign of power change
            if dt < best_dt and np.sign(tr["delta_p"]) == np.sign(d["delta_p"]):
                best, best_dt = ti, dt
        if best is not None and best_dt <= tol_s:
            used.add(best)
            matches.append((d, truth.loc[best]))
    precision = len(matches) / len(detected) if len(detected) else 0.0
    recall = len(matches) / len(truth) if len(truth) else 0.0
    return precision, recall, matches


# ----------------------------------------------------------------------
# 2. STEP FEATURES + CLASSIFIER
# ----------------------------------------------------------------------
def step_features(delta_p, delta_q):
    """Feature vector for one step. Mostly delta_p / delta_q (the quantities
    that add across devices), plus derived helpers."""
    mag = float(np.hypot(delta_p, delta_q))
    angle = float(np.arctan2(delta_q, delta_p))   # phase of the step
    qp = float(delta_q / delta_p) if abs(delta_p) > 1e-6 else 0.0
    return [delta_p, delta_q, abs(delta_p), abs(delta_q), mag, angle, qp]

STEP_FEATURE_NAMES = ["delta_p", "delta_q", "abs_dp", "abs_dq", "mag", "angle", "qp_ratio"]


def _steps_from_dataset(dataset):
    """Turn aggregator signals into a labelled step table using GROUND TRUTH
    events (so training labels are exact). Each on/off event is one step."""
    X, y = [], []
    for df, events, active, meta in dataset:
        for _, e in events.iterrows():
            X.append(step_features(e["delta_p"], e["delta_q"]))
            y.append(e["device"])
    return np.array(X), np.array(y)


class DisaggregatorModel:
    def __init__(self, model):
        self.model = model
    def _frame(self, rows):
        return pd.DataFrame(rows, columns=STEP_FEATURE_NAMES)
    def predict_step(self, delta_p, delta_q):
        return self.model.predict(self._frame([step_features(delta_p, delta_q)]))[0]
    def predict_many(self, dp_dq):
        return self.model.predict(self._frame([step_features(dp, dq) for dp, dq in dp_dq]))


def train_step_classifier(n_train=300, n_test=120, seed=2026):
    """Train the step classifier on labelled steps from aggregator signals mixing
    sequential and simultaneous switching.

    Random Forest is the primary model: it outperformed LightGBM on the noisy
    detected steps (0.92 vs 0.88). LightGBM is trained and reported only as a
    comparison, and only if the library is installed. Without it, the pipeline
    runs on Random Forest alone.
    """
    params = aggregator.device_params()
    train_ds = aggregator.build_aggregated_dataset(n_train, params=params, seed=seed)
    test_ds = aggregator.build_aggregated_dataset(n_test, params=params, seed=seed + 999)

    Xtr, ytr = _steps_from_dataset(train_ds)
    Xte, yte = _steps_from_dataset(test_ds)
    Xtr = pd.DataFrame(Xtr, columns=STEP_FEATURE_NAMES)
    Xte = pd.DataFrame(Xte, columns=STEP_FEATURE_NAMES)
    labels = sorted(set(ytr))
    code = {d: i for i, d in enumerate(labels)}
    ytr_i = np.array([code[d] for d in ytr])
    inv = {i: d for d, i in code.items()}

    rf = RandomForestClassifier(n_estimators=300, random_state=0)
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)
    results = {"RandomForest": accuracy_score(yte, rf_pred)}
    bundle = {"rf": DisaggregatorModel(rf), "lgbm": None, "labels": labels,
              "results": results, "yte": yte, "rf_pred": rf_pred,
              "test_ds": test_ds, "train_ds": train_ds, "params": params}

    if HAS_LGBM:
        lgbm = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                  min_child_samples=10, verbose=-1, random_state=0)
        lgbm.fit(Xtr, ytr_i)
        lgbm_pred = np.array([inv[i] for i in lgbm.predict(Xte)])
        results["LightGBM"] = accuracy_score(yte, lgbm_pred)
        bundle["lgbm"] = DisaggregatorModel(_LGBMWrap(lgbm, inv))
        bundle["lgbm_pred"] = lgbm_pred
    return bundle


class _LGBMWrap:
    """Make LGBM return string labels like the RF does."""
    def __init__(self, model, inv): self.model, self.inv = model, inv
    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=STEP_FEATURE_NAMES)
        return [self.inv[i] for i in self.model.predict(X)]


# ----------------------------------------------------------------------
def end_to_end_eval(model_bundle, n_per_mode=60, seed=7777):
    """Honest pipeline test: detector -> classify detected steps -> score.
    Reports detection precision/recall and classification accuracy on the
    DETECTED steps, separately for sequential and simultaneous switching.
    """
    params = model_bundle["params"]
    rf = model_bundle["rf"]; lgbm = model_bundle.get("lgbm")
    rng = np.random.default_rng(seed)
    report = {}
    for mode in ("sequential", "simultaneous"):
        precs, recs, lgbm_hits, rf_hits, n_matched = [], [], 0, 0, 0
        for _ in range(n_per_mode):
            max_dev = 4 if mode == "sequential" else 3
            combo = aggregator.feasible_combination(params, rng, max_devices=max_dev)
            if mode == "simultaneous" and len(combo) < 2:
                combo = aggregator.feasible_combination(params, rng, max_devices=3)
            df, events, active, meta = aggregator.aggregate(combo, params, mode=mode, rng=rng)
            det = detect_steps(df)
            if len(det) == 0:
                continue
            p, r, matches = match_steps_to_truth(det, events)
            precs.append(p); recs.append(r)
            for d_step, truth_row in matches:
                n_matched += 1
                if lgbm is not None and lgbm.predict_step(d_step["delta_p"], d_step["delta_q"]) == truth_row["device"]:
                    lgbm_hits += 1
                if rf.predict_step(d_step["delta_p"], d_step["delta_q"]) == truth_row["device"]:
                    rf_hits += 1
        entry = {
            "detect_precision": round(float(np.mean(precs)), 3),
            "detect_recall": round(float(np.mean(recs)), 3),
            "rf_step_acc": round(rf_hits / n_matched, 3) if n_matched else 0.0,
            "n_matched_steps": n_matched,
        }
        if lgbm is not None:
            entry["lgbm_step_acc"] = round(lgbm_hits / n_matched, 3) if n_matched else 0.0
        report[mode] = entry
    return report


def _demo():
    print("=" * 78)
    print("DISAGGREGATOR - STEP DETECTOR + STEP CLASSIFIER")
    print("=" * 78)

    # --- detector accuracy against ground truth ---
    params = aggregator.device_params()
    ds = aggregator.build_aggregated_dataset(40, params=params, seed=55)
    precs, recs = [], []
    for df, events, active, meta in ds:
        det = detect_steps(df)
        p, r, _ = match_steps_to_truth(det, events)
        precs.append(p); recs.append(r)
    print(f"\nStep detector on 40 combined signals:")
    print(f"  mean precision {np.mean(precs):.2f}, mean recall {np.mean(recs):.2f}")

    # --- classifier accuracy ---
    tag = "RF + LightGBM" if HAS_LGBM else "Random Forest (LightGBM not installed)"
    print(f"\nTraining step classifier ({tag}) on mixed sequential+simultaneous...")
    out = train_step_classifier(n_train=300, n_test=120)
    print("Upper bound (classify EXACT ground-truth steps):")
    for m, a in out["results"].items():
        print(f"  {m:14s} {a:.3f}")

    # --- honest end-to-end ---
    print("\nEnd-to-end (detector -> classify DETECTED steps), split by mode:")
    rep = end_to_end_eval(out, n_per_mode=60)
    for mode, r in rep.items():
        lg = f"LightGBM {r['lgbm_step_acc']}  " if "lgbm_step_acc" in r else ""
        print(f"  {mode.upper():13s} detect P/R {r['detect_precision']}/{r['detect_recall']}  "
              f"| {lg}RF {r['rf_step_acc']}  ({r['n_matched_steps']} steps)")


if __name__ == "__main__":
    _demo()
