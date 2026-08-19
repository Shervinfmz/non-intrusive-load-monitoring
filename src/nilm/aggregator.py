"""Milestone 3 - Stage 1: Aggregator.

Builds combined multi-appliance signals from the measured single-device
fingerprints, with full ground truth, so the disaggregator has labelled data
to train and test against. (A partner aggregator was not available, so we
build our own.)

What it produces, per combined signal:
  - a PAC4200-schema time series of the TOTAL (so the existing event detector
    and feature extractor run on it unchanged): summed P and Q, with apparent
    power and current derived from them
  - a ground-truth event list: (t_sec, device, kind on/off, delta_p, delta_q)
  - a per-sample active-device table (which devices are on at each instant)

Two switching modes:
  - "sequential"   : devices switch on one at a time, staggered -> clean steps
  - "simultaneous" : two or more switch on at the same instant -> one merged step

Honest limits:
  - Active power P and reactive power Q add up correctly. Apparent power and
    current are computed from the summed P and Q (correct for the fundamental).
  - Current THD does NOT add; the combined THD here is a current-weighted
    approximation, not a physical measurement. This is exactly why the
    disaggregator relies on delta_P and delta_Q, not on distortion.
  - Combinations are kept within the PAC4200 current limit (default 6 A).
"""

import math
import numpy as np
import pandas as pd

from . import paths, registry, acquisition

SAMPLES_PER_SEC = 5
NOMINAL_V = 230.0
CURRENT_LIMIT_A = 6.0


def device_params(fingerprints=None):
    """Per-device steady-state parameters needed to synthesise a contribution."""
    if fingerprints is None:
        _, fingerprints = acquisition.build_stage1()
    params = {}
    for dev, fp in fingerprints.items():
        P = fp["on_p_w_mean"]; Q = fp["on_q_var_mean"]
        V = fp.get("voltage_v_mean", NOMINAL_V) or NOMINAL_V
        S = math.sqrt(P * P + Q * Q)
        params[dev] = {
            "P": P, "Q": Q, "P_std": max(fp["on_p_w_std"], 0.01),
            "Q_std": max(fp["on_q_var_std"], 0.01),
            "thd_i": fp["on_thd_i_mean"], "V": V,
            "I": S / V if V else 0.0,
        }
    return params


def feasible_combination(params, rng, max_devices=4, max_current=CURRENT_LIMIT_A):
    """Pick a random subset of devices whose total current stays under the limit."""
    devices = list(params.keys())
    rng.shuffle(devices)
    chosen, total_I = [], 0.0
    target_n = rng.integers(1, max_devices + 1)
    for d in devices:
        if len(chosen) >= target_n:
            break
        if total_I + params[d]["I"] <= max_current:
            chosen.append(d); total_I += params[d]["I"]
    return chosen if chosen else [min(params, key=lambda d: params[d]["I"])]


def aggregate(combo, params, mode="sequential", duration_s=None, rng=None):
    """Build one combined signal + ground truth for a given device combination."""
    if rng is None:
        rng = np.random.default_rng()
    n_dev = len(combo)
    if duration_s is None:
        duration_s = 20 + 18 * n_dev
    total = int(duration_s * SAMPLES_PER_SEC)

    # choose switch-on / switch-off sample indices per device
    on_idx, off_idx = {}, {}
    if mode == "sequential":
        # staggered: device k on at (k+1)*gap, all off near the end (staggered)
        gap = total // (n_dev + 2)
        for k, d in enumerate(combo):
            on_idx[d] = gap * (k + 1) + rng.integers(-3, 4)
            off_idx[d] = total - gap // 2 - (n_dev - k) * (gap // 3) + rng.integers(-3, 4)
    else:  # simultaneous
        t_on = total // 3 + rng.integers(-3, 4)
        t_off = total - total // 4 + rng.integers(-3, 4)
        for d in combo:
            on_idx[d] = int(t_on); off_idx[d] = int(t_off)
    for d in combo:
        on_idx[d] = int(np.clip(on_idx[d], 2, total - 6))
        off_idx[d] = int(np.clip(off_idx[d], on_idx[d] + 4, total - 2))

    # build each device contribution and sum
    P_tot = np.zeros(total); Q_tot = np.zeros(total)
    thd_num = np.zeros(total); I_sum = np.zeros(total)   # for current-weighted THD
    active = {d: np.zeros(total, dtype=int) for d in combo}
    for d in combo:
        pr = params[d]
        sl = slice(on_idx[d], off_idx[d])
        n_on = off_idx[d] - on_idx[d]
        p = rng.normal(pr["P"], pr["P_std"], n_on)
        q = rng.normal(pr["Q"], pr["Q_std"], n_on)
        P_tot[sl] += p; Q_tot[sl] += q
        active[d][sl] = 1
        dev_I = pr["I"]
        I_sum[sl] += dev_I
        thd_num[sl] += pr["thd_i"] * dev_I     # weight THD by current share

    # derived totals
    V = rng.normal(NOMINAL_V, 0.4, total)
    S_tot = np.sqrt(P_tot ** 2 + Q_tot ** 2)
    I_tot = S_tot / V
    with np.errstate(divide="ignore", invalid="ignore"):
        pf = np.where(S_tot > 1e-6, P_tot / S_tot, np.nan)
        thd_i = np.where(I_sum > 1e-6, thd_num / I_sum, np.nan)   # approx
    thd_u = rng.normal(1.8, 0.05, total)
    freq = rng.normal(50.0, 0.01, total)

    # timestamps
    t0 = pd.Timestamp("2026-06-01T09:00:00Z")
    ts = t0 + pd.to_timedelta(np.arange(total) / SAMPLES_PER_SEC, unit="s")
    df = pd.DataFrame({
        "timestamp_iso": ts.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z",
        "device_name": "AGGREGATE", "run_id": "agg",
        "sample_interval_ms": 200,
        "u_l1_n_v": V.round(4), "i_l1_a": I_tot.round(6),
        "p_total_w": P_tot.round(4), "s_total_va": S_tot.round(4),
        "s_calc_va": (V * I_tot).round(4), "q_total_var": Q_tot.round(4),
        "pf_total": pf, "frequency_hz": freq.round(4),
        "thd_u_l1_percent": thd_u.round(4), "thd_i_l1_percent": thd_i,
        "block_time_difference_ms": 5,
    })
    df["t_sec"] = np.arange(total) / SAMPLES_PER_SEC

    # ground-truth events
    events = []
    for d in combo:
        pr = params[d]
        events.append({"t_sec": round(on_idx[d] / SAMPLES_PER_SEC, 2), "device": d,
                       "kind": "on", "delta_p": round(pr["P"], 1), "delta_q": round(pr["Q"], 1)})
        events.append({"t_sec": round(off_idx[d] / SAMPLES_PER_SEC, 2), "device": d,
                       "kind": "off", "delta_p": round(-pr["P"], 1), "delta_q": round(-pr["Q"], 1)})
    events = sorted(events, key=lambda e: e["t_sec"])
    active_df = pd.DataFrame(active); active_df["t_sec"] = df["t_sec"]

    meta = {"combo": combo, "mode": mode,
            "peak_current_a": round(float(I_tot.max()), 2),
            "peak_power_w": round(float(P_tot.max()), 1)}
    return df, pd.DataFrame(events), active_df, meta


def build_aggregated_dataset(n_signals=40, params=None, modes=("sequential", "simultaneous"),
                             seed=12345):
    """Generate many combined signals; return list of (df, events, active, meta)."""
    if params is None:
        params = device_params()
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_signals):
        mode = modes[i % len(modes)]
        max_dev = 4 if mode == "sequential" else 3
        combo = feasible_combination(params, rng, max_devices=max_dev)
        if mode == "simultaneous" and len(combo) < 2:
            combo = feasible_combination(params, rng, max_devices=3)
        out.append(aggregate(combo, params, mode=mode, rng=rng))
    return out


if __name__ == "__main__":
    params = device_params()
    rng = np.random.default_rng(7)
    print("=" * 78)
    print("AGGREGATOR DEMO")
    print("=" * 78)
    for mode in ("sequential", "simultaneous"):
        combo = feasible_combination(params, rng, max_devices=4 if mode == "sequential" else 3)
        df, ev, act, meta = aggregate(combo, params, mode=mode, rng=rng)
        print(f"\n[{mode}] devices: {meta['combo']}")
        print(f"  peak power {meta['peak_power_w']} W, peak current {meta['peak_current_a']} A "
              f"(limit {CURRENT_LIMIT_A} A) -> {'OK' if meta['peak_current_a'] <= CURRENT_LIMIT_A else 'OVER'}")
        print("  ground-truth events:")
        print(ev.to_string(index=False))
