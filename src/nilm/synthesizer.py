"""Stage 2: Synthesizer.

Reads outputs/appliance_fingerprints.json (from Stage 1) and generates
N realistic PAC4200-format runs per appliance.

Two splits with different seeds:
  - data/synthetic/training/   (150 runs / appliance, SEED_TRAIN)
  - data/synthetic/test/       (50 runs / appliance,  SEED_TEST)

Real recordings in data/clean/ are NEVER touched. Output schema matches the
real PAC4200 CSV exactly. PF and THD_i are NaN whenever current = 0.

Power factor in synthetic files is the DISPLACEMENT power factor
(P / sqrt(P^2 + Q^2)). The PAC4200 records TRUE power factor which differs
under high THD. Use P, Q, THD_i for NILM features; treat PF as informational.
"""

import json
import numpy as np
import pandas as pd

from . import paths, registry

SAMPLES_PER_SEC = 5
NOMINAL_INTERVAL_MS = 200
SEED_TRAIN = 20260529
SEED_TEST  = 99991111


# ----------------------------------------------------------------------
# Device-specific synthesis
# ----------------------------------------------------------------------
def _step_load(fp, on_idx, off_idx, total, rng):
    p_mean = fp["on_p_w_mean"]
    p_std  = max(fp["on_p_w_std"], 0.01)
    q_mean = fp["on_q_var_mean"]
    q_std  = max(fp["on_q_var_std"], 0.01)
    p_scale = rng.normal(1.0, 0.05)
    q_scale = rng.normal(1.0, 0.05)

    p = np.zeros(total); q = np.zeros(total)
    n_on = off_idx - on_idx
    p[on_idx:off_idx] = rng.normal(p_mean * p_scale, p_std, n_on)
    q[on_idx:off_idx] = rng.normal(q_mean * q_scale, q_std, n_on)

    if fp.get("inrush_ratio", 1.0) > 1.05 and n_on > 1:
        p[on_idx] = p_mean * fp["inrush_ratio"] * rng.uniform(0.9, 1.1)
        if q_mean > 30:
            q[on_idx] = q_mean * 1.4 * rng.uniform(0.9, 1.1)
    return p, q


def _two_speed_motor(fp, on_idx, off_idx, total, rng):
    """Mixer: two stable speed plateaus with transient at each change."""
    p_mean_total = fp["on_p_w_mean"]
    p_low  = p_mean_total * rng.uniform(0.55, 0.75)
    p_high = p_mean_total * rng.uniform(1.25, 1.45)
    p_std  = max(fp["on_p_w_std"], 0.01)
    spike_p = fp.get("inrush_ratio", 4.0) * p_mean_total

    p = np.zeros(total); q = np.zeros(total)
    n_on = off_idx - on_idx
    split = on_idx + n_on // 2 + rng.integers(-SAMPLES_PER_SEC, SAMPLES_PER_SEC + 1)
    split = int(np.clip(split, on_idx + 5, off_idx - 5))

    p[on_idx:split] = rng.normal(p_low,  p_std, split - on_idx)
    p[split:off_idx] = rng.normal(p_high, p_std, off_idx - split)
    p[on_idx] = spike_p * rng.uniform(0.85, 1.15)
    p[split]  = spike_p * rng.uniform(0.85, 1.15)
    q[on_idx:off_idx] = rng.normal(
        fp["on_q_var_mean"], max(fp["on_q_var_std"], 0.01), n_on)
    return p, q


DEVICE_DISPATCH = {
    "Mixer": _two_speed_motor,   # the one appliance with a distinct two-speed model
}
# every other appliance uses the clean-step model by default (see build_stage2)


# ----------------------------------------------------------------------
# Single-run generator
# ----------------------------------------------------------------------
def synth_one_run(fp, device, run_id, t_start, rng):
    duration_s = rng.uniform(70, 110)
    total = int(duration_s * SAMPLES_PER_SEC)
    on_start_s = rng.uniform(15, 30)
    on_dur_s   = rng.uniform(20, 60)
    on_idx  = int(on_start_s * SAMPLES_PER_SEC)
    off_idx = min(int((on_start_s + on_dur_s) * SAMPLES_PER_SEC), total - 5)

    p_device, q_device = DEVICE_DISPATCH.get(device, _step_load)(fp, on_idx, off_idx, total, rng)

    voltage_mean = fp["voltage_v_mean"]
    u_l1 = rng.normal(voltage_mean, 0.4, total)
    freq = rng.normal(50.01, 0.012, total)
    thd_u_base = fp["on_thd_u_mean"] if fp["on_thd_u_mean"] > 0 else 1.7
    thd_u = rng.normal(thd_u_base, 0.05, total)
    block_dt = rng.normal(6.0, 2.0, total).clip(min=1)

    s_total = np.sqrt(p_device**2 + q_device**2)
    on_mask = np.zeros(total, dtype=bool)
    on_mask[on_idx:off_idx] = True

    i_l1 = np.zeros(total)
    i_l1[on_mask] = s_total[on_mask] / u_l1[on_mask]
    s_calc = u_l1 * i_l1

    pf = np.full(total, np.nan)
    thd_i = np.full(total, np.nan)
    pf[on_mask] = p_device[on_mask] / s_total[on_mask]
    thd_i_mean = fp["on_thd_i_mean"]
    thd_i_std  = max(fp["on_thd_i_std"], 0.01)
    thd_i[on_mask] = rng.normal(thd_i_mean, thd_i_std, on_mask.sum()).clip(min=0.1)

    t_offsets_s = np.cumsum(rng.normal(NOMINAL_INTERVAL_MS / 1000, 0.005, total))
    timestamps = pd.to_datetime(t_start, utc=True) + pd.to_timedelta(t_offsets_s, unit="s")
    ts_iso = timestamps.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"

    return pd.DataFrame({
        "timestamp_iso":            ts_iso,
        "device_name":              device,
        "run_id":                   run_id,
        "sample_interval_ms":       NOMINAL_INTERVAL_MS,
        "u_l1_n_v":                 u_l1.round(6),
        "i_l1_a":                   i_l1.round(9),
        "p_total_w":                p_device.round(6),
        "s_total_va":               s_total.round(6),
        "s_calc_va":                s_calc.round(6),
        "q_total_var":              q_device.round(6),
        "pf_total":                 pf,
        "frequency_hz":             freq.round(6),
        "thd_u_l1_percent":         thd_u.round(6),
        "thd_i_l1_percent":         thd_i,
        "block_time_difference_ms": block_dt.round().astype(int),
    })


def safe_filename(device):
    return registry.slug(device)


def generate_split(fingerprints, out_dir, n_runs, seed, split_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(seed)
    manifest = []
    t_start = "2026-06-01T08:00:00Z"
    for device, fp in fingerprints.items():
        slug = safe_filename(device)
        for k in range(n_runs):
            sub_seed = int(master_rng.integers(0, 2**31 - 1))
            rng = np.random.default_rng(sub_seed)
            run_id = f"synth_{slug}_{split_name}_{k:04d}"
            df = synth_one_run(fp, device, run_id, t_start, rng)
            fn = f"pac4200_{slug}_{split_name}_{k:04d}.csv"
            df.to_csv(out_dir / fn, index=False)
            manifest.append({"device": device, "run_id": run_id,
                             "file": fn, "seed": sub_seed,
                             "n_samples": len(df)})
    pd.DataFrame(manifest).to_csv(out_dir / "_manifest.csv", index=False)
    return manifest


def build_stage2(n_train=150, n_test=50):
    paths.ensure_dirs()
    with open(paths.FINGERPRINTS_JSON) as f:
        fingerprints = json.load(f)
    m_train = generate_split(fingerprints, paths.SYNTH_TRAIN, n_train, SEED_TRAIN, "train")
    m_test  = generate_split(fingerprints, paths.SYNTH_TEST,  n_test,  SEED_TEST,  "test")
    return m_train, m_test


if __name__ == "__main__":
    print("Generating training + test synthetic data...")
    mt, ms = build_stage2()
    print(f"  training: {len(mt)} files in {paths.SYNTH_TRAIN}")
    print(f"  test:     {len(ms)} files in {paths.SYNTH_TEST}")
