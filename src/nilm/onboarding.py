"""Milestone 3 - Stage 4b: Auto-onboarding loop.

When the novelty layer flags "New Device Found", this closes the loop with the
one manual step we agreed is unavoidable: a human types the device's name. From
that single label, everything else is automatic.

Steps:
  1. Capture the new device's fingerprint (its P, Q and noise) from the step or
     the isolated recording.
  2. Take ONE human label (the device name).
  3. Synthesize labelled step samples from that fingerprint, the same idea as the
     Milestone 1 synthesizer.
  4. Add the new class and retrain the step classifier.

After this the model recognises the device on every future switch.

Hard requirement, stated plainly: capturing a clean fingerprint needs the new
device seen ALONE (at least its clean switch-on). A new device buried inside a
running mix cannot be fingerprinted this way.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from . import aggregator
from . import disaggregator as D


def train_rf_on_params(params, n_train=300, seed=2026):
    """Train the step Random Forest on aggregated signals built from `params`."""
    train_ds = aggregator.build_aggregated_dataset(n_train, params=params, seed=seed)
    Xtr, ytr = D._steps_from_dataset(train_ds)
    Xtr = pd.DataFrame(Xtr, columns=D.STEP_FEATURE_NAMES)
    rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)
    return D.DisaggregatorModel(rf)


def capture_params(device_name, full_params=None):
    """Capture a device's (P, Q, noise) fingerprint. In deployment this comes
    from the meter reading of the device seen alone; here we read it from the
    measured fingerprints."""
    full_params = full_params or aggregator.device_params()
    if device_name not in full_params:
        raise KeyError(device_name)
    return dict(full_params[device_name])


def onboard(known_params, new_name, new_params, n_train=300, seed=2026):
    """Add the new device and retrain. Returns (updated_params, new_model)."""
    updated = dict(known_params)
    updated[new_name] = new_params
    model = train_rf_on_params(updated, n_train=n_train, seed=seed)
    return updated, model


def _demo():
    print("=" * 78)
    print("AUTO-ONBOARDING LOOP")
    print("=" * 78)
    from .novelty import known_fingerprints, NoveltyDetector

    full = aggregator.device_params()
    unknown = "Mixer"                       # pretend the model never saw this
    known = {k: v for k, v in full.items() if k != unknown}
    print(f"\nStarting model knows {len(known)} devices. '{unknown}' is unseen.")

    # 1. model trained WITHOUT the unknown device
    old_model = train_rf_on_params(known, n_train=300)

    # 2. the unknown device is plugged in alone -> novelty check on its fingerprint
    fp, feats = known_fingerprints()
    det = NoveltyDetector(dist_k=2.0).fit(fp.drop(index=unknown), feats)
    novel, dmin, thr = det.is_novel_fingerprint(fp.loc[unknown])
    print(f"\nStep 1 - novelty check on the plugged-in device:")
    print(f"  distance to known set {dmin} vs threshold {thr}  -> "
          f"{'NEW DEVICE FOUND' if novel else 'looks known'}")

    # 3. what the OLD model says about the unknown device's switch
    p, q = full[unknown]["P"], full[unknown]["Q"]
    old_pred = old_model.predict_step(p, q)
    print(f"\nStep 2 - old model is asked to name the switch (dP={p:.0f}, dQ={q:.0f}):")
    print(f"  old model says: '{old_pred}'  (wrong - it has no '{unknown}' class)")

    # 4. one human label, then automatic capture + synthesise + retrain
    print(f"\nStep 3 - human types one label: '{unknown}'")
    new_params = capture_params(unknown, full)
    updated, new_model = onboard(known, unknown, new_params, n_train=300)
    print(f"  captured fingerprint, synthesised training steps, retrained on "
          f"{len(updated)} devices.")

    # 5. the retrained model now names it correctly
    new_pred = new_model.predict_step(p, q)
    print(f"\nStep 4 - retrained model is asked the same switch:")
    print(f"  retrained model says: '{new_pred}'  "
          f"({'correct' if new_pred == unknown else 'still wrong'})")
    print("\nOne human label in. New device known on every future switch.")


if __name__ == "__main__":
    _demo()
