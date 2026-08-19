"""M1 Stage 1: Acquisition, Preprocessing & Understanding (data-driven).

Discovers every appliance in data/raw via the registry, loads it robustly
(blank rows dropped, device names cleaned), finds the main ON region, and
extracts a measured fingerprint per appliance. PV (generation) is excluded.

The fingerprint file (outputs/appliance_fingerprints.json) drives the synthesizer.
Nothing is invented: every parameter is measured.
"""

import json
import numpy as np
import pandas as pd

from . import paths, registry, events


def classify_load(pf, q_mean, thd_i):
    tags = []
    if thd_i is not None and thd_i > 25:
        tags.append("non-linear")
    if q_mean > 15:
        tags.append("inductive")
    elif q_mean < -15:
        tags.append("capacitive/leading")
    if pf is not None and pf > 0.97 and (thd_i is None or thd_i < 5):
        tags = ["resistive"]
    if not tags:
        tags.append("mostly resistive")
    return " + ".join(tags)


def extract_fingerprint(df, region, settle_s=1.0):
    """Measure parameters from the steady part of the main ON region."""
    sps = region["sps"]
    on_s, on_e = region["on_idx"], region["off_idx"]
    settle = on_s + int(settle_s * sps)
    off = df.iloc[:on_s] if on_s > 5 else df.iloc[:5]
    on_steady = df.iloc[settle:on_e] if settle < on_e else df.iloc[on_s:on_e]
    on_full = df.iloc[on_s:on_e]

    def stat(s):
        s = s.dropna()
        return (round(float(s.mean()), 3), round(float(s.std()), 3)) if len(s) else (0.0, 0.0)

    p_on_mean, p_on_std = stat(on_steady["p_total_w"])
    q_on_mean, q_on_std = stat(on_steady["q_total_var"])
    pf_on_mean, _ = stat(on_steady["pf_total"])
    thd_i_mean, thd_i_std = stat(on_steady["thd_i_l1_percent"])
    thd_u_mean, _ = stat(on_steady["thd_u_l1_percent"])

    inrush_peak = float(on_full["p_total_w"].iloc[:sps].max()) if len(on_full) else 0.0
    inrush_ratio = round(inrush_peak / p_on_mean, 2) if p_on_mean > 1 else 1.0

    return {
        "samples_per_sec": int(sps),
        "on_start_s": region["on_t"], "on_end_s": region["off_t"],
        "off_baseline_p_w": round(float(off["p_total_w"].mean()), 3),
        "off_baseline_q_var": round(float(off["q_total_var"].mean()), 3),
        "on_p_w_mean": p_on_mean, "on_p_w_std": p_on_std,
        "on_q_var_mean": q_on_mean, "on_q_var_std": q_on_std,
        "on_pf_mean": pf_on_mean,
        "on_thd_i_mean": thd_i_mean, "on_thd_i_std": thd_i_std,
        "on_thd_u_mean": thd_u_mean,
        "voltage_v_mean": round(float(df["u_l1_n_v"].mean()), 2),
        "inrush_ratio": inrush_ratio,
        "load_character": classify_load(pf_on_mean, q_on_mean, thd_i_mean),
    }


def build_stage1():
    paths.ensure_dirs()
    appliances = registry.discover()           # PV excluded
    if not appliances:
        raise FileNotFoundError(f"No PAC4200 CSVs found in {paths.RAW}")

    fingerprints, rows = {}, []
    for device, path in appliances.items():
        df = registry.load_pac(path)
        region = events.detect_active_region(df)
        if region["on_idx"] is None:
            print(f"  [warn] {device}: no ON region detected, skipped")
            continue
        fp = extract_fingerprint(df, region)
        fp["device_name"] = device
        fp["source_file"] = path.name
        fp["n_cycles"] = len(events.detect_all_regions(df))
        fingerprints[device] = fp
        df.to_parquet(paths.CLEAN / f"{registry.slug(device)}.parquet")
        rows.append({
            "Device": device, "Peak_W": round(float(df["p_total_w"].max()), 1),
            "On_P_W": fp["on_p_w_mean"], "On_Q_var": fp["on_q_var_mean"],
            "PF": fp["on_pf_mean"], "THD_i_%": fp["on_thd_i_mean"],
            "Cycles": fp["n_cycles"], "Character": fp["load_character"],
        })

    with open(paths.FINGERPRINTS_JSON, "w") as f:
        json.dump(fingerprints, f, indent=2)
    summary = pd.DataFrame(rows).sort_values("On_P_W", ascending=False).reset_index(drop=True)
    summary.to_csv(paths.SUMMARY_CSV, index=False)
    return summary, fingerprints


if __name__ == "__main__":
    summary, _ = build_stage1()
    pd.set_option("display.width", 200)
    print("=" * 100)
    print(f"MILESTONE 1 - STAGE 1: APPLIANCE UNDERSTANDING ({len(summary)} appliances)")
    print("=" * 100)
    print(summary.to_string(index=False))
