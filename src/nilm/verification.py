"""M1 Stage 3: Verification (data-driven).

Compares synthetic data against real recordings for every discovered appliance.

For each appliance x feature (P, Q, THD_i):
  - mean-band check: real steady mean within synthetic 5-95% band
  - noise-band check: real steady std within synthetic 5-95% band
Power factor is informational only (synthetic PF is displacement-only).

Outputs: per-appliance overlays + summary figure + verification_table.csv.
"""

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import paths, registry, events

# Backwards-compatible helper used elsewhere
def load_csv(path):
    return registry.load_pac(path)

FEATURES = ["p_total_w", "q_total_var", "pf_total", "thd_i_l1_percent"]
FEATURE_LABEL = {
    "p_total_w": "Active Power P (W)", "q_total_var": "Reactive Power Q (var)",
    "pf_total": "Power Factor", "thd_i_l1_percent": "Current THD (%)",
}


def steady_state_slice(df, settle_s=1.0):
    """Steady part of the main ON region (uses the shared event detector)."""
    r = events.detect_active_region(df)
    if r["on_idx"] is None:
        return df.iloc[0:0]
    settle = r["on_idx"] + int(settle_s * r["sps"])
    if settle >= r["off_idx"]:
        return df.iloc[r["on_idx"]:r["off_idx"]]
    return df.iloc[settle:r["off_idx"]]


def _synth_files(slug, split="training"):
    tag = "train" if split == "training" else "test"
    return sorted(glob.glob(str(paths.SYNTH / split / f"pac4200_{slug}_{tag}_*.csv")))


def plot_overlay(device, real_df, synth_dfs, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, feat in zip(axes.flatten(), FEATURES):
        for s in synth_dfs:
            ax.plot(s["t_sec"], s[feat], color="steelblue", alpha=0.35, linewidth=0.9)
        ax.plot(real_df["t_sec"], real_df[feat], color="black", linewidth=1.6)
        ax.set_title(FEATURE_LABEL[feat]); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Verification overlay: {device}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out_path, dpi=120); plt.close(fig)


def verify_appliance(real_df, synth_files):
    rows = []
    real_steady = steady_state_slice(real_df)
    s_means = {f: [] for f in FEATURES}
    s_stds = {f: [] for f in FEATURES}
    for path in synth_files:
        ss = steady_state_slice(registry.load_pac(path))
        if len(ss) == 0:
            continue
        for f in FEATURES:
            v = ss[f].dropna()
            if len(v) >= 5:
                s_means[f].append(float(v.mean())); s_stds[f].append(float(v.std()))
    for f in FEATURES:
        rv = real_steady[f].dropna().values
        if len(rv) < 5:
            continue
        rmean, rstd = float(rv.mean()), float(rv.std())
        sm, sd = np.array(s_means[f]), np.array(s_stds[f])
        if len(sm) >= 20:
            mlo, mhi = np.percentile(sm, [5, 95]); mean_pass = mlo <= rmean <= mhi
        else:
            mlo = mhi = np.nan; mean_pass = False
        if len(sd) >= 20:
            slo, shi = np.percentile(sd, [5, 95]); noise_pass = slo <= rstd <= shi
        else:
            slo = shi = np.nan; noise_pass = False
        rows.append({"feature": f, "real_mean": round(rmean, 3),
                     "synth_mean_p05": round(mlo, 3), "synth_mean_p95": round(mhi, 3),
                     "mean_pass": bool(mean_pass), "real_std": round(rstd, 3),
                     "synth_std_p05": round(slo, 3), "synth_std_p95": round(shi, 3),
                     "noise_pass": bool(noise_pass), "informational": (f == "pf_total")})
    return rows


def build_stage3():
    paths.ensure_dirs()
    appliances = registry.discover()
    print("=" * 80)
    print(f"MILESTONE 1 - STAGE 3: VERIFICATION ({len(appliances)} appliances)")
    print("=" * 80)
    table_rows = []
    for device, path in appliances.items():
        slug = registry.slug(device)
        sfiles = _synth_files(slug)
        if len(sfiles) < 20:
            print(f"  [skip] {device}: {len(sfiles)} synth files")
            continue
        real_df = registry.load_pac(path)
        for r in verify_appliance(real_df, sfiles):
            r["device"] = device
            table_rows.append(r)
        rng = np.random.default_rng(42)
        picks = rng.choice(len(sfiles), min(5, len(sfiles)), replace=False)
        plot_overlay(device, real_df, [registry.load_pac(sfiles[i]) for i in picks],
                     paths.FIGURES / f"verification_{slug}.png")
        print(f"  [ok]   {device}")

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(paths.VERIFICATION_CSV, index=False)

    scoring = table_df[~table_df["informational"]]
    n = len(scoring)
    nm = int(scoring["mean_pass"].sum()); nn = int(scoring["noise_pass"].sum())
    print(f"\nMean-band check  (P, Q, THD_i): {nm}/{n} pass")
    print(f"Noise-band check (P, Q, THD_i): {nn}/{n} pass")
    fails = scoring[~scoring["mean_pass"] | ~scoring["noise_pass"]]
    if len(fails):
        print(f"\n{len(fails)} feature(s) flagged:")
        for _, r in fails.iterrows():
            print(f"  {r['device']}/{r['feature']}: "
                  f"mean_pass={r['mean_pass']} noise_pass={r['noise_pass']}")
    return table_df


if __name__ == "__main__":
    build_stage3()
