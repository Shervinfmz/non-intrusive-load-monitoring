"""Milestone 3 - Stage 3: Running-Set Tracker with combination matching.

The step classifier names ONE device per step. That fails when two devices
switch at the same instant, because the merged step matches a combination, not
a single device. The tracker fixes this and produces the actual deliverable:
the set of devices that are ON at any moment.

How it works:
  - Keep a running set of devices currently on.
  - For each detected step, decide turn-on (delta_p > 0) or turn-off (< 0).
  - Match the step's (delta_p, delta_q) to the best SINGLE device or small
    COMBINATION whose summed (P, Q) explains the step:
        on-step  : choose from devices NOT currently on
        off-step : choose from devices currently on
  - Update the running set, and record the timeline.

Combination matching is what handles simultaneous switching: a merged step is
explained by the device set whose summed signature is closest, preferring the
smallest set. The current set constrains the search, which also keeps it cheap.
"""

from itertools import combinations
import numpy as np
import pandas as pd

from . import aggregator


def match_step(dp, dq, candidates, params, max_combo=3, size_penalty=6.0,
               penalty_frac=0.0):
    """Best single device or combination explaining a step of size (dp, dq).

    dp, dq are the MAGNITUDES to explain (positive). Returns (best_set, error).

    The penalty for each extra device scales with the size of the step
    (penalty_frac) as well as having an absolute floor (size_penalty). Without
    that, a big step can always be fitted slightly better by bolting on a small
    extra device, which is where phantom devices come from.
    """
    mag = float(np.hypot(dp, dq))
    per_extra = max(size_penalty, penalty_frac * mag)
    best, best_err = (), np.inf
    for r in range(1, max_combo + 1):
        for combo in combinations(candidates, r):
            sp = sum(params[d]["P"] for d in combo)
            sq = sum(params[d]["Q"] for d in combo)
            err = np.hypot(sp - dp, sq - dq) + per_extra * (r - 1)
            if err < best_err:
                best_err, best = err, combo
    return set(best), best_err


def explain_step(dp, dq, candidates, params, max_combo=3,
                 tol_abs=10.0, tol_frac=0.18):
    """Like match_step, but says 'I do not know' instead of forcing a name.

    Returns (device_set, error, ok). ok is False when even the best explanation
    leaves a residual larger than tol_abs or tol_frac of the step size, meaning
    the step does not correspond to any known device or combination.
    """
    target_p, target_q = abs(dp), (dq if dp >= 0 else -dq)
    combo, err = match_step(target_p, target_q, candidates, params, max_combo)
    mag = float(np.hypot(target_p, target_q))
    ok = err <= max(tol_abs, tol_frac * mag)
    return combo, err, ok


class RunningSetTracker:
    def __init__(self, params, classifier=None, max_combo=2, tie_tol=1.0,
                 tol_abs=10.0, tol_frac=0.18, on_unknown=None, uncal_penalty=2.0):
        self.params = params
        self.classifier = classifier
        self.max_combo = max_combo
        self.tie_tol = tie_tol        # only break genuinely close ties with the model
        self.tol_abs = tol_abs        # residual beyond this -> unknown load
        self.tol_frac = tol_frac
        self.on_unknown = on_unknown  # callback(dp, dq) when nothing fits
        # A device measured on site, through this meter, is more trustworthy than
        # a fingerprint recorded on different hardware. When two devices sit
        # within noise of each other (the monitor and the mixer are 1.4 apart),
        # this small penalty makes the on-site measurement win.
        self.uncal_penalty = uncal_penalty
        self.on = set()
        self.timeline = []
        self.unknowns = []
        self.last = {}

    def _pen(self, d):
        return 0.0 if self.params[d].get("calibrated") else self.uncal_penalty

    def _best_single(self, tp, tq, candidates):
        best, err = None, np.inf
        for d in candidates:
            e = np.hypot(self.params[d]["P"] - tp, self.params[d]["Q"] - tq) + self._pen(d)
            if e < err:
                best, err = d, e
        return best, err

    def _best_combo(self, tp, tq, candidates, min_share=0.10):
        """Best explanation using two or more devices.

        A device may only join a combination if its own apparent power is at
        least min_share of the step. Without this, a 950 W step can always be
        fitted a little better by adding a 25 VA device to soak up the natural
        variance of the big load, which is where phantom devices come from.
        """
        mag = float(np.hypot(tp, tq))
        allowed = [d for d in candidates
                   if np.hypot(self.params[d]["P"], self.params[d]["Q"])
                   >= min_share * mag]
        best, err = set(), np.inf
        for r in range(2, self.max_combo + 1):
            for combo in combinations(allowed, r):
                sp = sum(self.params[d]["P"] for d in combo)
                sq = sum(self.params[d]["Q"] for d in combo)
                e = (np.hypot(sp - tp, sq - tq) + 6.0 * (r - 1)
                     + sum(self._pen(d) for d in combo))
                if e < err:
                    best, err = set(combo), e
        return best, err

    def _explain(self, dp, dq, candidates, from_empty=False):
        """Decide which device(s) a step belongs to.

        The residual against the calibrated (P, Q) of each device is the primary
        evidence. A combination is only preferred when it explains the step
        clearly better than the best single device. The classifier is used only
        to break near-ties between single devices, which is where it earns its
        keep (look-alike pairs). If nothing fits within tolerance, the step is
        reported as an unknown load rather than forced onto a wrong name.
        """
        tp = abs(dp)
        tq = dq if dp >= 0 else -dq
        if not candidates:
            return set()

        s_dev, s_err = self._best_single(tp, tq, candidates)
        c_set, c_err = self._best_combo(tp, tq, candidates)

        mag = float(np.hypot(tp, tq))
        tol = max(self.tol_abs, self.tol_frac * mag)

        # Deciding whether a step is an UNKNOWN device.
        # From an empty state the step is one device, or a group switched at the
        # same instant. If it sits far from EVERY single known device, it is
        # suspicious, because a genuine simultaneous group is still made of
        # devices that individually resemble something known. Once devices are
        # already running we have to allow combinations, so the test is weaker.
        # Honest limit: an unknown load whose signature happens to equal the sum
        # of two known devices cannot be distinguished from them by dP and dQ.
        judge_err = s_err if from_empty else min(s_err, c_err)

        self.last = {"single": s_dev, "single_err": float(s_err),
                     "combo": sorted(c_set), "combo_err": float(c_err),
                     "tol": float(tol)}

        # nothing explains this step
        if judge_err > tol:
            self.unknowns.append({"delta_p": dp, "delta_q": dq,
                                  "residual": round(float(judge_err), 2)})
            if self.on_unknown:
                self.on_unknown(dp, dq)
            return set()

        # A combination must beat the best single by a margin that scales with
        # the size of the step. Large loads vary by tens of watts on their own,
        # and without this a small device gets bolted on to absorb that variance,
        # which is exactly how phantom devices appear.
        combo_margin = max(2.0, 0.03 * mag)
        if c_err + combo_margin < s_err:
            return set(c_set)

        # single device: let the classifier arbitrate only a genuine tie
        if self.classifier is not None:
            close = [d for d in candidates
                     if np.hypot(self.params[d]["P"] - tp,
                                 self.params[d]["Q"] - tq) + self._pen(d)
                     <= s_err + self.tie_tol]
            if len(close) > 1:
                pick = self.classifier.predict_step(dp, dq)
                if pick in close:
                    return {pick}
        return {s_dev}

    def _enforce_family_exclusive(self, added):
        """A 'family' is the set of states of ONE appliance (a fan's speeds, a
        hair dryer's settings), so at most one member can be on at a time. When
        a member is newly switched on, drop any other member of the same family
        that was still marked on - which happens when a fast speed change is
        detected as a fresh turn-on instead of a transition.
        """
        for d in added:
            fam = self.params.get(d, {}).get("family")
            if not fam:
                continue
            for other in list(self.on):
                if other != d and self.params.get(other, {}).get("family") == fam:
                    self.on.discard(other)

    def _match_off_variable(self, dp, candidates):
        """Turn-off fallback for a variable-power device (a charger).

        A charger draws anywhere in its power range while running, so when it is
        unplugged the size of the OFF step is whatever it happened to be drawing
        at that moment - rarely its calibrated value. Matching that step against
        a single power number fails, and the device is left stuck 'on' or the
        step is reported as a spurious unknown. Here, if a currently-on `varies`
        device's power range contains the size of the OFF step, that device is
        the one switching off. Only called after the normal fixed-power match
        has already failed, so it never steals a genuine fixed-device turn-off.
        """
        tp = abs(dp)
        for d in candidates:
            v = self.params.get(d, {})
            pr = v.get("p_range")
            if v.get("varies") and pr and (pr[0] - 2.0) <= tp <= (pr[1] + 2.0):
                return {d}
        return set()

    def _try_transition(self, dp, dq):
        """Could this step be one device changing state, rather than a device
        switching on or off?

        A hair dryer that moves from its fan setting to full heat produces a
        step equal to the DIFFERENCE between those two states. Without this,
        the tracker sees an unexplained jump and invents a second device.
        Devices that share a "family" tag are states of one appliance.
        """
        best, err = None, np.inf
        for d in list(self.on):
            fam = self.params[d].get("family")
            if not fam:
                continue
            for d2, v2 in self.params.items():
                if d2 == d or v2.get("family") != fam:
                    continue
                tdp = v2["P"] - self.params[d]["P"]
                tdq = v2["Q"] - self.params[d]["Q"]
                e = float(np.hypot(tdp - dp, tdq - dq))
                if e < err:
                    err, best = e, (d, d2)
        return best, err

    def _is_generator(self, d):
        v = self.params.get(d, {})
        return bool(v.get("generator")) or v.get("P", 0.0) < 0.0

    def _try_generator(self, dp, dq):
        """Could this step be a generator (PV) turning on or off?

        A generator has negative real power, so it turns ON with a negative step
        (~ its P, Q) and OFF with the mirror positive step. PV is a VARIES device:
        its watts follow the sun, so its off-step is +(whatever it was generating)
        - not its single calibrated value. For a varies generator the active power
        is matched against its power RANGE, and the match leans on the reactive
        power, which is the generator's stable signature. Returns (action, name,
        err) or None.
        """
        best = None
        best_err = np.inf
        for d in self.params:
            if not self._is_generator(d):
                continue
            v = self.params[d]
            gp, gq = v["P"], v["Q"]
            rng = v.get("p_range") if v.get("varies") else None
            if d not in self.on:                     # candidate: generator ON (neg step)
                if rng:
                    lo, hi = rng
                    ep = 0.0 if lo <= dp <= hi else min(abs(dp - lo), abs(dp - hi))
                else:
                    ep = dp - gp
                eq = dq - gq
                action = "on"
            else:                                    # candidate: generator OFF (pos step)
                if rng:
                    lo, hi = rng
                    plo, phi = -hi, -lo              # mirror the power range to positive
                    ep = 0.0 if plo <= dp <= phi else min(abs(dp - plo), abs(dp - phi))
                else:
                    ep = dp + gp
                eq = dq + gq
                action = "off"
            err = float(np.hypot(ep, eq))
            mag = float(np.hypot(gp, gq))
            tol = max(self.tol_abs, self.tol_frac * mag)
            if err <= tol and err < best_err:
                best_err, best = err, (action, d)
        if best is None:
            return None
        return best[0], best[1], best_err

    def confirm_variable(self, chosen, post_p, post_thd, p_tol=2.0, thd_tol=12.0):
        """Correct a look-alike collision using a short post-step observation.

        The Monitor and the USB charger can produce an identical (P, Q) step,
        and - shown on the real recording - an identical current-distortion at
        that instant too, so no single-sample test separates them. What DOES
        separate them is behaviour over the next few seconds: the Monitor holds
        a fixed level, the USB charger's power and distortion wander.

        Call this AFTER a step has been attributed to a FIXED device, passing a
        few seconds of P and THD_i sampled after the step. If the load is in
        fact wandering and a `varies` device in the calibration fits the
        observed band, return that device's name instead. Otherwise return
        `chosen` unchanged.

        post_p, post_thd: sequences of samples (P in W, THD_i in %) after step.
        Returns the possibly-corrected device name.
        """
        cv = self.params.get(chosen, {})
        if cv.get("varies"):                 # already a variable device; nothing to do
            return chosen
        p = np.asarray([x for x in post_p if x is not None and np.isfinite(x)], float)
        thd = np.asarray([x for x in (post_thd or []) if x is not None and np.isfinite(x)], float)
        if len(p) < 3:
            return chosen                    # not enough evidence to overturn the call
        p_swing = float(p.max() - p.min())
        thd_swing = float(thd.max() - thd.min()) if len(thd) >= 3 else 0.0
        steady = (p_swing <= p_tol) and (thd_swing <= thd_tol)
        if steady:
            return chosen                    # behaves like the fixed device: keep it

        # wandering: pick a `varies` device whose bands contain what we observed
        pmid = float(np.median(p))
        tmid = float(np.median(thd)) if len(thd) else None
        for name, v in self.params.items():
            if not v.get("varies"):
                continue
            pr = v.get("p_range")
            tr = v.get("thd_i_range")
            p_ok = (pr is None) or (pr[0] - p_tol <= pmid <= pr[1] + p_tol)
            t_ok = (tr is None) or (tmid is None) or (tr[0] - thd_tol <= tmid <= tr[1] + thd_tol)
            if p_ok and t_ok:
                return name
        return chosen

    def process(self, steps_df):
        # detect_steps returns an empty, column-less DataFrame when nothing is
        # found (e.g. a steady standing load such as PV). Guard it so a quiet
        # feed does not crash the tracker on .sort_values("t_sec").
        if steps_df is None or len(steps_df) == 0 or "t_sec" not in steps_df.columns:
            return self.timeline
        for _, s in steps_df.sort_values("t_sec").iterrows():
            dp, dq = s["delta_p"], s["delta_q"]
            mag = float(np.hypot(dp, dq))

            # 1. a state change within one multi-state appliance
            trans, terr = self._try_transition(dp, dq)
            if trans is not None and terr <= max(8.0, 0.10 * mag):
                old, new = trans
                self.on.discard(old)
                self.on.add(new)
                self.last = {"single": new, "single_err": terr,
                             "combo": [], "combo_err": np.inf,
                             "tol": max(8.0, 0.10 * mag), "transition": True}
                self.timeline.append((round(float(s["t_sec"]), 2), set(self.on)))
                continue

            # 2. a generator (PV) toggling. A generator has NEGATIVE real power,
            #    so it switches ON with a negative step and OFF with a positive
            #    one - the opposite sign convention to a load. Handle it before
            #    the load logic so a PV turn-on is not misread as a load turn-off.
            #    NOTE: this only fires if PV switches DURING monitoring. PV that
            #    is already generating when monitoring starts produces no step and
            #    is the standing-load stage's job (steadystate.identify_standing_load).
            gen = self._try_generator(dp, dq)
            if gen is not None:
                g_action, g_name, g_err = gen
                # A load switch has the same sign pattern as a generator toggle
                # (a fan off looks like PV on; a fan on looks like PV off). Decide
                # by residual: take the generator only if it explains the step at
                # least as well as the best matching load.
                tp = abs(dp); tq = dq if dp >= 0 else -dq
                if g_action == "on":
                    alt = [d for d in self.on if not self._is_generator(d)]
                else:
                    alt = [d for d in self.params
                           if d not in self.on and not self._is_generator(d)]
                load_err = np.inf
                if alt:
                    _, load_err = self._best_single(tp, tq, alt)
                if g_err <= load_err:
                    if g_action == "on":
                        self.on.add(g_name)
                    else:
                        self.on.discard(g_name)
                    self.timeline.append((round(float(s["t_sec"]), 2), set(self.on)))
                    continue
                # a real load explains it better: fall through to the load logic

            # 3. otherwise a normal load switching on or off
            if dp >= 0:
                was_empty = len(self.on) == 0
                cand = [d for d in self.params if d not in self.on
                        and not self._is_generator(d)]
                added = self._explain(dp, dq, cand, from_empty=was_empty)
                self.on |= added
                self._enforce_family_exclusive(added)
            else:
                cand = [d for d in self.on if not self._is_generator(d)]
                if cand:
                    tp = abs(dp); tq = dq if dp >= 0 else -dq
                    _, s_err = self._best_single(tp, tq, cand)
                    mag = float(np.hypot(tp, tq))
                    tol = max(self.tol_abs, self.tol_frac * mag)
                    off_var = self._match_off_variable(dp, cand)
                    if s_err > tol and off_var:
                        # no fixed on-device explains this OFF step, but an on
                        # charger's power range does: it was unplugged at an
                        # off-nominal draw. Remove it instead of reporting unknown.
                        self.on -= off_var
                        self.last = {"single": next(iter(off_var)),
                                     "single_err": float(s_err), "combo": [],
                                     "combo_err": np.inf, "tol": float(tol),
                                     "variable_off": True}
                    else:
                        self.on -= self._explain(dp, dq, cand)
            self.timeline.append((round(float(s["t_sec"]), 2), set(self.on)))
        return self.timeline

    def current(self):
        return set(self.on)


def active_set_at(active_df, t, device_cols):
    """Ground-truth active set at time t from the aggregator's per-sample table."""
    i = int(np.searchsorted(active_df["t_sec"].values, t))
    i = min(max(i, 0), len(active_df) - 1)
    return {d for d in device_cols if active_df[d].values[i] == 1}


def evaluate(n_per_mode=60, seed=4321, classifier=None):
    """Score the tracker's running set against ground truth, per mode.

    Metric: Jaccard overlap between predicted and true active sets, measured
    just after each ground-truth event, plus exact-set match rate.
    """
    from .disaggregator import detect_steps, train_step_classifier
    params = aggregator.device_params()
    if classifier is None:
        classifier = train_step_classifier(n_train=300, n_test=30)["rf"]
    rng = np.random.default_rng(seed)
    out = {}
    for mode in ("sequential", "simultaneous"):
        jac, exact, n = [], 0, 0
        for _ in range(n_per_mode):
            md = 4 if mode == "sequential" else 3
            combo = aggregator.feasible_combination(params, rng, max_devices=md)
            if mode == "simultaneous" and len(combo) < 2:
                combo = aggregator.feasible_combination(params, rng, max_devices=3)
            df, events, active, meta = aggregator.aggregate(combo, params, mode=mode, rng=rng)
            device_cols = [c for c in active.columns if c != "t_sec"]
            steps = detect_steps(df)
            if len(steps) == 0:
                continue
            tr = RunningSetTracker(params, classifier=classifier)
            tr.process(steps)
            for t in sorted(events["t_sec"].unique()):
                pred = _set_from_timeline(tr.timeline, t)
                true = active_set_at(active, t + 0.5, device_cols)
                u = pred | true
                jac.append(len(pred & true) / len(u) if u else 1.0)
                exact += int(pred == true)
                n += 1
        out[mode] = {"jaccard": round(float(np.mean(jac)), 3),
                     "exact_set_rate": round(exact / n, 3) if n else 0.0,
                     "n_checks": n}
    return out


def _set_from_timeline(timeline, t):
    """Predicted active set just after time t."""
    cur = set()
    for tt, s in timeline:
        if tt <= t + 0.5:
            cur = s
        else:
            break
    return cur


def plot_gantt(combo, params, classifier, save_to, seed=3, mode="sequential"):
    """Draw the aggregated signal and a device on/off timeline (predicted vs
    true) for one combination, and save it. Returns the tracker used."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from .disaggregator import detect_steps

    df, ev, active, meta = aggregator.aggregate(combo, params, mode=mode,
                                                rng=np.random.default_rng(seed))
    cols = [c for c in active.columns if c != "t_sec"]
    tmax = active["t_sec"].max()
    steps = detect_steps(df)
    tr = RunningSetTracker(params, classifier=classifier)
    tr.process(steps)

    def iv_active(a, d):
        t = a["t_sec"].values; on = a[d].values; out = []; s = None
        for i in range(len(t)):
            if on[i] == 1 and s is None: s = t[i]
            elif on[i] == 0 and s is not None: out.append((s, t[i])); s = None
        if s is not None: out.append((s, t[-1]))
        return out

    def iv_time(tl, d, tm):
        out = []; s = None
        for tt, st in tl:
            ins = d in st
            if ins and s is None: s = tt
            elif not ins and s is not None: out.append((s, tt)); s = None
        if s is not None: out.append((s, tm))
        return out

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2]})
    axes[0].plot(df["t_sec"], df["p_total_w"], color="black", lw=1.1)
    axes[0].set_ylabel("Total P (W)"); axes[0].grid(alpha=0.3)
    axes[0].set_title("Aggregated signal and device on/off timeline (predicted vs true)")
    ax = axes[1]; yp = {d: i for i, d in enumerate(combo)}
    for d in combo:
        y = yp[d]
        for s, e in iv_active(active, d):
            ax.barh(y + 0.18, e - s, left=s, height=0.30, color="#9ec5e8")
        for s, e in iv_time(tr.timeline, d, tmax):
            ax.barh(y - 0.18, e - s, left=s, height=0.30, color="#2e6da4")
    ax.set_yticks(list(yp.values())); ax.set_yticklabels([d[:16] for d in combo])
    ax.set_xlabel("Time (s)"); ax.set_ylim(-0.6, len(combo) - 0.4)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[Patch(color="#9ec5e8", label="True ON"),
                       Patch(color="#2e6da4", label="Predicted ON")],
              loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(save_to, dpi=130); plt.close()
    return tr


def _demo():
    print("=" * 78)
    print("RUNNING-SET TRACKER (combination matching)")
    print("=" * 78)
    params = aggregator.device_params()

    # show one simultaneous example end to end
    from .disaggregator import detect_steps, train_step_classifier
    clf = train_step_classifier(n_train=300, n_test=30)["rf"]
    rng = np.random.default_rng(8)
    combo = ["WATER_HEATER", "Foen Stufe 1", "LED Lampe 6.3 W"]
    df, events, active, meta = aggregator.aggregate(combo, params, mode="simultaneous", rng=rng)
    steps = detect_steps(df)
    tr = RunningSetTracker(params, classifier=clf)
    tr.process(steps)
    print(f"\nTrue devices: {combo}")
    print("Tracker timeline (time -> devices believed ON):")
    for t, s in tr.timeline:
        print(f"  t={t:5.1f}s  {sorted(s)}")

    print("\nScoring tracker vs ground truth (Jaccard overlap + exact-set rate):")
    rep = evaluate(n_per_mode=60, classifier=clf)
    for mode, r in rep.items():
        print(f"  {mode.upper():13s} Jaccard {r['jaccard']}  exact-set {r['exact_set_rate']}  ({r['n_checks']} checks)")


if __name__ == "__main__":
    _demo()
