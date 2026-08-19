"""Milestone 3 - load characterisation.

Flagging a device as merely "unknown" is not useful to an operator. Even when a
step matches no trained device, its change in active and reactive power still
says a lot about what kind of load it is, because the physics does not depend on
knowing the brand.

From (dP, dQ) alone we can state:
  - apparent power of the step and its displacement power factor
  - the load class: resistive, inductive, or electronic/capacitive
  - a size band and a plain description of what such a load usually is

This is what the dashboard shows when nothing in the trained set fits, for
example: "Unknown inductive load, about 40 VA, PF 0.62 - likely a motor,
fan, or a lamp with a magnetic ballast."
"""

import numpy as np

# |Q| below this fraction of |P| counts as "no meaningful reactive power"
RESISTIVE_FRAC = 0.15


def describe_load(dp, dq):
    """Characterise a step from its active and reactive power change.

    Returns a dict: class, label, description, p_w, q_var, s_va, pf, angle_deg.
    Sign of dp is ignored (a switch-off of a motor is still a motor).
    """
    P = abs(float(dp))
    Q = float(dq) if dp >= 0 else -float(dq)
    S = float(np.hypot(P, Q))
    pf = (P / S) if S > 1e-6 else 0.0
    angle = float(np.degrees(np.arctan2(Q, P)))

    if S < 1e-6:
        return {"class": "none", "label": "No load", "description": "",
                "p_w": 0.0, "q_var": 0.0, "s_va": 0.0, "pf": 0.0, "angle_deg": 0.0}

    if abs(Q) <= RESISTIVE_FRAC * max(P, 1e-6):
        cls = "resistive"
        base = "draws current in phase with the voltage, so nearly all of its power does real work"
        if P >= 800:
            what = "a large heating element such as a kettle, heater, toaster, or coffee machine"
        elif P >= 250:
            what = "a heating element or an incandescent lamp"
        elif P >= 40:
            what = "a small heater or a filament lamp"
        else:
            what = "a small resistive load"
    elif Q > 0:
        cls = "inductive"
        base = "draws current that lags the voltage, which means magnetic windings"
        if P >= 500:
            what = "a large motor or a transformer"
        elif P >= 60:
            what = "a motor, a pump, or a compressor"
        else:
            what = "a small motor, a fan, or a lamp with a magnetic ballast"
    else:
        cls = "capacitive"
        base = "draws current that leads the voltage, typical of a switched-mode power supply"
        if P >= 100:
            what = "a larger power supply, an inverter, or a charger"
        elif P >= 20:
            what = "a computer, a monitor, or a television"
        else:
            what = "a small electronic device such as an LED lamp, a charger, or a USB supply"

    label = f"Unknown {cls} load"
    description = (f"About {S:.0f} VA at power factor {pf:.2f}. It {base}. "
                   f"This is most likely {what}.")

    return {"class": cls, "label": label, "description": description,
            "p_w": round(P, 1), "q_var": round(Q, 1), "s_va": round(S, 1),
            "pf": round(pf, 3), "angle_deg": round(angle, 1)}


def nearest_known(dp, dq, params, k=2):
    """The k closest trained devices to this step, with their distances.

    Reported alongside an unknown load so the operator can see what it resembles,
    without the system claiming it IS that device.
    """
    target_p = abs(float(dp))
    target_q = float(dq) if dp >= 0 else -float(dq)
    scored = []
    for d, v in params.items():
        dist = float(np.hypot(v["P"] - target_p, v["Q"] - target_q))
        scored.append((d, round(dist, 2)))
    scored.sort(key=lambda x: x[1])
    return scored[:k]
