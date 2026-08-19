"""Milestone 3 - site calibration.

The device fingerprints in Milestone 1 come from solo recordings. In the lab the
same devices draw differently: a fan on another speed setting, a warmed-up
element, a different supply voltage. Measured on site:

    Cooler Fan  solo 30.5 W / 48.0 var  ->  real 34.5 W / 58.1 var
    Table Fan   solo 13.4 W / 18.5 var  ->  real 17.5 W / 24.2 var
    Monitor     solo 17.9 W / -3.0 var  ->  real 15.9 W /  +0.3 var

Those gaps are what make the tracker invent devices: the sum of two real fans no
longer matches the sum of two solo fingerprints, so a single wrong device fits
the merged step better.

The fix is what a real NILM install does at commissioning: measure each device
once on site, through the meter that will be used, and use THOSE numbers.

This module extracts a device's (P, Q) from recordings where it switches on its
own, and writes an updated parameter set. Calibrate on recordings with isolated
switches; keep simultaneous recordings as a held-out test.
"""

import json
import os

import numpy as np
import pandas as pd

from . import aggregator
from .disaggregator import detect_steps

CALIB_FILE = "device_calibration.json"


def load_recording(path):
    raw = pd.read_csv(path)
    raw = raw.dropna(subset=["timestamp_iso", "p_total_w"]).reset_index(drop=True)
    raw["_t"] = pd.to_datetime(raw["timestamp_iso"])
    raw["t_sec"] = (raw["_t"] - raw["_t"].iloc[0]).dt.total_seconds()
    return raw[["t_sec", "p_total_w", "q_total_var"]].copy()


def steps_for(path):
    return detect_steps(load_recording(path))


def calibrate(recordings, verbose=True):
    """recordings: list of (csv_path, [device order for ON steps]).

    Each recording must switch its devices on ONE AT A TIME, in the given order.
    OFF steps are matched to whichever listed device is closest in |dP|, so the
    off order does not need to be known.

    Returns {device: {"P":..,"Q":..}} averaged over every isolated observation.
    """
    obs = {}
    for path, order in recordings:
        if not os.path.exists(path):
            continue
        st = steps_for(path)
        ons = st[st["delta_p"] > 0].sort_values("t_sec")
        offs = st[st["delta_p"] < 0].sort_values("t_sec")
        # ON steps map to the given order
        for dev, (_, s) in zip(order, ons.iterrows()):
            obs.setdefault(dev, []).append((abs(s["delta_p"]), s["delta_q"]))
        # OFF steps map to the nearest ON magnitude among the same devices
        on_mag = {d: m for d, (m, _) in
                  zip(order, [(abs(s["delta_p"]), s["delta_q"]) for _, s in ons.iterrows()])}
        for _, s in offs.iterrows():
            mag = abs(s["delta_p"])
            if not on_mag:
                continue
            dev = min(on_mag, key=lambda d: abs(on_mag[d] - mag))
            obs.setdefault(dev, []).append((mag, -s["delta_q"]))

    calib = {}
    for dev, vals in obs.items():
        P = float(np.mean([v[0] for v in vals]))
        Q = float(np.mean([v[1] for v in vals]))
        calib[dev] = {"P": round(P, 2), "Q": round(Q, 2), "n_obs": len(vals)}
        if verbose:
            print(f"  {dev:20s} P={P:8.2f}W  Q={Q:7.2f}var   ({len(vals)} observations)")
    return calib


def apply_calibration(params, calib):
    """Return a copy of params with calibrated devices overwritten or ADDED.

    New entries are allowed because a real device can turn out to have several
    operating states (a hair dryer with a fan setting, a half-wave heat setting,
    and full heat). Each state is registered separately and tagged with a
    "family", so the tracker knows they belong to one physical appliance and a
    move between them is a state change, not a second device switching on.
    """
    out = {d: dict(v) for d, v in params.items()}
    for dev, v in calib.items():
        entry = out.get(dev, {})
        entry = dict(entry)
        entry["P"] = v["P"]
        entry["Q"] = v["Q"]
        if "family" in v:
            entry["family"] = v["family"]
        if v.get("thd_i") is not None:
            # current distortion, usable only when this device runs alone
            entry["thd_i"] = v["thd_i"]
        if v.get("varies"):
            entry["varies"] = True      # power changes while running (e.g. a charger)
        if v.get("thd_i_range"):
            entry["thd_i_range"] = list(v["thd_i_range"])
        if v.get("p_range"):
            entry["p_range"] = list(v["p_range"])
        entry["calibrated"] = True      # measured through the meter it runs on
        entry.setdefault("P_std", 1.0)
        entry.setdefault("Q_std", 1.0)
        out[dev] = entry
    return out


def save(calib, path=CALIB_FILE):
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)


def load(path=CALIB_FILE):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def calibrated_params(path=CALIB_FILE):
    """device_params() with any saved site calibration applied."""
    return apply_calibration(aggregator.device_params(), load(path))
