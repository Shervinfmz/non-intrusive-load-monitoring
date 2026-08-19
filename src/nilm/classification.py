"""Milestone 2 - Stage 4: Classification / Identification.

Train classifiers on SYNTHETIC training features and evaluate on:
  (a) synthetic TEST  (300 samples) - fine-grained model check
  (b) REAL recordings (6 samples)   - the honest transfer metric

The real files are the held-out test set; they are never used for training.
A high synthetic-test score with a low real score means the model works but the
synthesizer does not transfer - exactly what we want to measure, not hide.

Models:
  - Random Forest (primary): small data, interpretable importances, no scaling.
  - Logistic Regression + k-NN (baselines): sanity references. Scaled.

Honest caveats:
  - 6 real test points => accuracy moves in 1/6 = 17% steps. Coarse by necessity.
  - From clustering we expect USB / LED / fluorescent to be the hard cases.

Run from project root:
    python -m nilm.classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from . import paths
from .features import FEATURE_COLS, build_feature_table, real_feature_table, synth_pairs


def _xy(df):
    return df[FEATURE_COLS].values.astype(float), df["label"].values


def plot_confusion(cm, labels, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_importances(rf, out_path):
    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot.barh(ax=ax, color="steelblue")
    ax.set_title("Random Forest feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return imp.sort_values(ascending=False)


def build_classification(per_device_train=150, per_device_test=50):
    paths.ensure_dirs()
    print("=" * 78)
    print("MILESTONE 2 - CLASSIFICATION / IDENTIFICATION")
    print("=" * 78)

    train = build_feature_table(synth_pairs("training", per_device=per_device_train))
    stest = build_feature_table(synth_pairs("test", per_device=per_device_test))
    real = real_feature_table()

    Xtr, ytr = _xy(train)
    Xte, yte = _xy(stest)
    Xre, yre = _xy(real)
    labels = sorted(set(ytr))

    # Scaler for the linear / distance baselines (RF does not need it)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s, Xre_s = scaler.transform(Xtr), scaler.transform(Xte), scaler.transform(Xre)

    models = {
        "RandomForest": (RandomForestClassifier(n_estimators=300, random_state=0), False),
        "LogisticReg":  (LogisticRegression(max_iter=2000), True),
        "kNN(k=5)":     (KNeighborsClassifier(n_neighbors=5), True),
    }

    results = []
    rf_model = None
    for name, (clf, scaled) in models.items():
        Xtr_use = Xtr_s if scaled else Xtr
        Xte_use = Xte_s if scaled else Xte
        Xre_use = Xre_s if scaled else Xre
        clf.fit(Xtr_use, ytr)
        acc_te = accuracy_score(yte, clf.predict(Xte_use))
        acc_re = accuracy_score(yre, clf.predict(Xre_use))
        results.append({"model": name,
                        "synthetic_test_acc": round(acc_te, 3),
                        "real_acc": round(acc_re, 3),
                        "real_correct": f"{int(acc_re*len(yre))}/{len(yre)}"})
        if name == "RandomForest":
            rf_model = clf

    res_df = pd.DataFrame(results)
    print("\nAccuracy (trained on synthetic):")
    print(res_df.to_string(index=False))

    # Confusion matrices for the Random Forest
    cm_te = confusion_matrix(yte, rf_model.predict(Xte), labels=labels)
    cm_re = confusion_matrix(yre, rf_model.predict(Xre), labels=labels)
    plot_confusion(cm_te, labels, paths.FIGURES / "clf_confusion_synth_test.png",
                   f"RF confusion - synthetic test ({len(yte)})")
    plot_confusion(cm_re, labels, paths.FIGURES / "clf_confusion_real.png",
                   f"RF confusion - REAL ({len(yre)} samples, multi-cycle)")

    # Real per-appliance predictions (the interesting detail)
    real_pred = rf_model.predict(Xre)
    detail = pd.DataFrame({"true": yre, "predicted": real_pred,
                           "correct": yre == real_pred})
    print("\nReal-file predictions (Random Forest):")
    print(detail.to_string(index=False))

    importances = plot_importances(rf_model, paths.FIGURES / "clf_feature_importance.png")
    print("\nTop feature importances (RF):")
    print(importances.head(8).to_string())

    res_df.to_csv(paths.REPORTS / "classification_results.csv", index=False)
    detail.to_csv(paths.REPORTS / "classification_real_predictions.csv", index=False)
    importances.to_frame("importance").to_csv(paths.REPORTS / "classification_feature_importance.csv")

    print("\nReminder: real samples include correlated repeats (multiple cycles per recording),")
    print("A gap between synthetic-test and real accuracy = synthesizer transfer gap,")
    print("signals that the problem is hard and only a robust model separates the lookalikes.")
    return res_df, detail, importances


if __name__ == "__main__":
    build_classification()
