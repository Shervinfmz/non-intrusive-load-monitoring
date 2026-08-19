"""Milestone 3 - identifying what is ALREADY running.

The event-based method only ever sees changes. If the monitoring starts while a
device is already switched on, no step is produced, and the device is invisible
to the tracker forever. This is a real limitation of the whole edge-detection
family of NILM methods, and it is exactly what happens if the professor plugs a
device in before we press Start.

The fix is to identify the standing load once, at startup, from the steady signal
itself rather than from a transition. Two things make this possible:

  1. Active and reactive power of the standing load are the SUM of the devices
     that are on. So we can search for the subset of known devices whose powers
     add up to what we measure. This is the same combination search the tracker
     already uses, applied to an absolute level instead of a step.

  2. When exactly one device is running, the total distortion of the current
     IS that device's distortion. Distortion does not add up in a mix, but with
     a single device there is nothing to mix with. That gives us a strong extra
     feature that the step-based path can never use.

What this can and cannot do, stated plainly:

  - A single running device is identified from power, reactive power and current
    distortion together. This separates a heater from a laptop charger easily.
  - Several running devices are explained by whichever combination best matches
    the standing power. The answer is not always unique, so the alternatives are
    returned as well instead of pretending there is one answer.
  - Devices whose power varies while running, like a phone charger whose draw
    falls as the battery fills, cannot be pinned to a single power value. They
    are recognised mainly by their distortion, and the confidence is lower.
"""

from itertools import combinations

import numpy as np

# how much a difference in each feature "costs", in the same units as the feature.
# a 2 W power error is treated as comparable to a 20 percentage-point THD error.
SCALE_P = 2.0
SCALE_Q = 2.0
SCALE_THD = 20.0
# A device whose power changes while it runs (a phone charger falls from 30 W to
# 2 W as the battery fills) cannot be pinned to one power value. For those we
# loosen the power term and lean on distortion, which stays in a characteristic
# band across the whole charging curve.
SCALE_P_VARIES = 15.0

IDLE_W = 3.0          # below this the bus is treated as carrying nothing


def steady_snapshot(p, q, thd_i=None, window=None):
    """Median power, reactive power and distortion over a settled window."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    out = {"P": float(np.median(p)), "Q": float(np.median(q)),
           "P_std": float(np.std(p))}
    if thd_i is not None:
        t = np.asarray(thd_i, dtype=float)
        t = t[np.isfinite(t)]
        out["thd_i"] = float(np.median(t)) if t.size else None
    else:
        out["thd_i"] = None
    return out


def _pq_residual(devices, P, Q, params):
    sp = sum(params[d]["P"] for d in devices)
    sq = sum(params[d]["Q"] for d in devices)
    return float(np.hypot(sp - P, sq - Q))


def _dev_residual(d, P, Q, params):
    """Residual for ONE device, allowing a power RANGE for variable loads.

    A charger legitimately draws anywhere between 2 W and 31 W, so measuring its
    error against a single number would reject it for most of its charging curve.
    """
    v = params[d]
    rng = v.get("p_range")
    if rng and v.get("varies"):
        lo, hi = rng
        dp = 0.0 if lo <= P <= hi else (lo - P if P < lo else P - hi)
    else:
        dp = v["P"] - P
    return float(np.hypot(dp, v["Q"] - Q))


def _thd_term(v, thd_i):
    """How badly the measured distortion disagrees with this device.

    A device with a known distortion RANGE costs nothing while the measurement
    sits inside that range; outside it, the cost grows from the nearest edge.
    """
    if thd_i is None:
        return None
    rng = v.get("thd_i_range")
    if rng:
        lo, hi = rng
        if lo <= thd_i <= hi:
            return 0.0
        return (lo - thd_i if thd_i < lo else thd_i - hi) / SCALE_THD
    if v.get("thd_i") is not None:
        return (v["thd_i"] - thd_i) / SCALE_THD
    return None


def _single_score(d, P, Q, thd_i, params):
    """Normalised distance for ONE running device, using distortion when we have it."""
    v = params[d]
    scale_p = SCALE_P_VARIES if v.get("varies") else SCALE_P
    scale_q = SCALE_P_VARIES if v.get("varies") else SCALE_Q
    terms = [(v["P"] - P) / scale_p, (v["Q"] - Q) / scale_q]
    t = _thd_term(v, thd_i)
    if t is not None:
        terms.append(t)
    return float(np.linalg.norm(terms))


def identify_standing_load(P, Q, params, thd_i=None, max_devices=3,
                           tol_abs=8.0, tol_frac=0.10, min_share=0.10,
                           n_alternatives=3):
    """Explain a standing load of (P, Q) with the known devices.

    Returns a dict with:
        idle        True when nothing meaningful is drawing power
        best        set of device names, or empty when nothing fits
        residual    how many VA the explanation fails to account for
        confidence  0-100, from the residual relative to the tolerance
        single      True when the explanation is one device
        alternatives  other plausible explanations, so ambiguity is visible
        note        a short plain-English caveat when relevant
    """
    mag = float(np.hypot(P, Q))
    if mag < IDLE_W:
        return {"idle": True, "best": set(), "residual": 0.0, "confidence": 100,
                "single": False, "alternatives": [], "note": "Nothing is drawing power."}

    tol = max(tol_abs, tol_frac * mag)

    # --- one device: use distortion as well as power
    singles = sorted(params.keys(), key=lambda d: _single_score(d, P, Q, thd_i, params))
    best_single = singles[0]
    single_res = _dev_residual(best_single, P, Q, params)

    # --- several devices: power only, since distortion does not add up
    combo_best, combo_res = None, np.inf
    for r in range(2, max_devices + 1):
        for combo in combinations(params.keys(), r):
            if any(np.hypot(params[d]["P"], params[d]["Q"]) < min_share * mag
                   for d in combo):
                continue
            res = _pq_residual(combo, P, Q, params) + 4.0 * (r - 1)
            if res < combo_res:
                combo_res, combo_best = res, set(combo)

    # a combination must beat the single device clearly, and by more for big loads
    margin = max(2.0, 0.03 * mag)
    if combo_best is not None and combo_res + margin < single_res:
        best, residual, single = combo_best, combo_res, False
    else:
        best, residual, single = {best_single}, single_res, True

    if residual > tol:
        return {"idle": False, "best": set(), "residual": round(residual, 2),
                "confidence": 0, "single": False, "alternatives": [],
                "note": "The standing load does not match any known device or combination."}

    conf = int(max(5, min(99, round(100.0 * (1.0 - residual / tol)))))

    alts = []
    for d in singles[1:1 + n_alternatives]:
        r = _dev_residual(d, P, Q, params)
        if r <= tol * 1.5:
            alts.append({"devices": [d], "residual": round(r, 2)})

    note = ""
    if single and params[best_single].get("varies"):
        note = ("This device's power changes while it runs, so it is recognised mainly "
                "by its current distortion rather than by how many watts it draws.")
    elif single and thd_i is None:
        note = "Distortion was not available, so this rests on power alone."
    elif not single:
        note = ("Several devices explain this level. Distortion cannot separate them, "
                "because it does not add up when devices run together.")

    return {"idle": False, "best": best, "residual": round(residual, 2),
            "confidence": conf, "single": single, "alternatives": alts, "note": note}


def _demo():
    import warnings; warnings.filterwarnings("ignore")
    from . import aggregator, calibration
    params = calibration.apply_calibration(aggregator.device_params(),
                                           calibration.load("device_calibration.json"))
    print("=" * 74)
    print("IDENTIFYING A DEVICE THAT IS ALREADY RUNNING (no switch observed)")
    print("=" * 74)
    cases = [
        ("nothing plugged in",            0.1,   0.0,  None),
        ("toaster already on",          709.0,   1.0,   2.3),
        ("cooler fan already on",        32.9,  55.9,  None),
        ("monitor already on",           16.3,   0.6, 165.5),
        ("USB charger already on",       30.4,  -3.1, 155.5),
        ("hair dryer on full heat",     991.5,   2.8,   2.4),
        ("two fans already on",          50.1,  79.5,  None),
    ]
    for name, P, Q, thd in cases:
        r = identify_standing_load(P, Q, params, thd_i=thd)
        if r["idle"]:
            print(f"  {name:26s} -> bus idle")
            continue
        who = sorted(r["best"]) or ["UNKNOWN"]
        print(f"  {name:26s} -> {', '.join(who):34s} conf {r['confidence']:3d}%  "
              f"residual {r['residual']}")
        if r["alternatives"]:
            print(f"      also possible: "
                  + "; ".join(f"{a['devices'][0]} ({a['residual']})" for a in r["alternatives"]))


if __name__ == "__main__":
    _demo()
