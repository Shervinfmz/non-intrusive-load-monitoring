"""Milestone 2 - Stage 3: Clustering.

Group recordings by their features WITHOUT using labels, then check whether the
groups match the true appliances.

Honest framing (important):
  - Synthetic-only clustering is expected near-perfect, because synthetic data
    is built from 6 fixed fingerprints => 6 separable blobs. SANITY CHECK ONLY.
  - The informative tests are real-only and real+synthetic: does each real
    recording land with its synthetic siblings? A real appliance in the wrong
    cluster is a genuine finding.

Method:
  - Standardize features (z-score) - mandatory; P ~0-1400 would otherwise
    dominate distance and drown out THD / crest factor.
  - K-means (k=6) primary; Ward hierarchical + dendrogram secondary.
  - Silhouette curve over k=2..10 (does the data itself suggest 6?).
  - PCA to 2D for visualization only (clustering uses all 12 features).

Metrics: Adjusted Rand Index (chance-corrected) and cluster purity.

Run from project root:
    python -m nilm.clustering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

from . import paths
from .features import FEATURE_COLS, build_feature_table, real_feature_table, synth_pairs


# ----------------------------------------------------------------------
def purity(true_labels, cluster_labels):
    """Fraction of points in the majority class of their assigned cluster."""
    df = pd.DataFrame({"t": true_labels, "c": cluster_labels})
    return sum(g["t"].value_counts().iloc[0] for _, g in df.groupby("c")) / len(df)


def _standardize(feat_df, scaler=None):
    X = feat_df[FEATURE_COLS].values.astype(float)
    if scaler is None:
        scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def silhouette_curve(X, out_path, true_k=6, k_range=range(2, 16)):
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        scores.append(silhouette_score(X, km.labels_))
    best_k = list(k_range)[int(np.argmax(scores))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), scores, "o-", color="steelblue")
    ax.axvline(best_k, color="red", ls="--", alpha=0.6, label=f"best k = {best_k}")
    ax.axvline(true_k, color="green", ls=":", alpha=0.6, label=f"true k = {true_k}")
    ax.set_xlabel("k (clusters)"); ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette analysis (synthetic features)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return dict(zip(list(k_range), [round(s, 3) for s in scores])), best_k


def pca_scatter(X, true_labels, cluster_labels, out_path, title):
    pca = PCA(n_components=2).fit(X)
    XY = pca.transform(X); var = pca.explained_variance_ratio_
    appliances = sorted(set(true_labels))
    cmap = plt.get_cmap("tab10")
    color_for = {a: cmap(i) for i, a in enumerate(appliances)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    fig, ax = plt.subplots(figsize=(9, 7))
    for (x, y), t, c in zip(XY, true_labels, cluster_labels):
        ax.scatter(x, y, color=color_for[t], marker=markers[int(c) % len(markers)],
                   s=60, alpha=0.7, edgecolors="black", linewidths=0.3)
    for a in appliances:
        ax.scatter([], [], color=color_for[a], marker="o", s=60, label=a)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% var)")
    ax.set_title(title); ax.legend(title="True appliance (color)", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def dendrogram_plot(X, labels, out_path, title):
    Z = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title(title); ax.set_ylabel("Ward distance")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------
def build_clustering(per_device=80):
    paths.ensure_dirs()
    print("=" * 78)
    print("MILESTONE 2 - CLUSTERING")
    print("=" * 78)

    synth = build_feature_table(synth_pairs("training", per_device=per_device))
    real = real_feature_table()

    Xs, scaler = _standardize(synth)
    Xr, _ = _standardize(real, scaler)
    ys, yr = synth["label"].values, real["label"].values
    N_APP = synth["label"].nunique()

    # (1) synthetic only - sanity check
    km_s = KMeans(n_clusters=N_APP, n_init=10, random_state=0).fit(Xs)
    ari_s, pur_s = adjusted_rand_score(ys, km_s.labels_), purity(ys, km_s.labels_)

    # (2) real only - coarse (6 points)
    km_r = KMeans(n_clusters=N_APP, n_init=10, random_state=0).fit(Xr)
    ari_r, pur_r = adjusted_rand_score(yr, km_r.labels_), purity(yr, km_r.labels_)

    # (3) real + synthetic together
    X_all = np.vstack([Xs, Xr]); y_all = np.concatenate([ys, yr])
    km_all = KMeans(n_clusters=N_APP, n_init=10, random_state=0).fit(X_all)
    ari_all, pur_all = adjusted_rand_score(y_all, km_all.labels_), purity(y_all, km_all.labels_)

    synth_clusters = km_all.labels_[:len(ys)]
    real_clusters = km_all.labels_[len(ys):]

    print(f"\n(1) Synthetic only : ARI={ari_s:.3f}  purity={pur_s:.3f}  (n={len(ys)})")
    print( "    -> near-perfect BY CONSTRUCTION (6 fixed fingerprints). Sanity check only.")
    print(f"(2) Real only      : ARI={ari_r:.3f}  purity={pur_r:.3f}  (n={len(yr)})")
    print( "    -> coarse: only 6 points. Indicative, not robust.")
    print(f"(3) Real+synthetic : ARI={ari_all:.3f}  purity={pur_all:.3f}  (n={len(y_all)})")

    # Real-point placement (the meaningful result)
    rows, misplaced = [], 0
    for dev, rc in zip(yr, real_clusters):
        sib = ys[synth_clusters == rc]
        majority = pd.Series(sib).value_counts().idxmax() if len(sib) else "(none)"
        match = (majority == dev)
        if not match:
            misplaced += 1
        rows.append({"appliance": dev, "cluster": int(rc),
                     "cluster_majority": majority, "match": match})
    placement = pd.DataFrame(rows)
    print("\nReal recordings - did each land with its synthetic siblings?")
    print(placement.to_string(index=False))
    print(f"\n{len(yr) - misplaced}/{len(yr)} real recordings clustered with their own kind.")

    # Figures
    sil_scores, best_k = silhouette_curve(Xs, paths.FIGURES / "cluster_silhouette.png", true_k=N_APP)
    pca_scatter(X_all, y_all, km_all.labels_,
                paths.FIGURES / "cluster_pca_real_plus_synth.png",
                "PCA: real + synthetic (color = appliance, marker = cluster)")
    exemplar_idx = [np.where(ys == a)[0][0] for a in sorted(set(ys))]
    X_dendro = np.vstack([Xr, Xs[exemplar_idx]])
    lbl_dendro = [f"REAL:{d}" for d in yr] + [f"synth:{a}" for a in sorted(set(ys))]
    dendrogram_plot(X_dendro, lbl_dendro,
                    paths.FIGURES / "cluster_dendrogram.png",
                    "Ward dendrogram: real recordings + synthetic exemplars")

    print(f"\nSilhouette best k = {best_k} (true k = {N_APP}). Scores: {sil_scores}")

    summary = pd.DataFrame([
        {"scenario": "synthetic_only", "n": len(ys), "ARI": round(ari_s, 3), "purity": round(pur_s, 3)},
        {"scenario": "real_only",      "n": len(yr), "ARI": round(ari_r, 3), "purity": round(pur_r, 3)},
        {"scenario": "real_plus_synth","n": len(y_all), "ARI": round(ari_all, 3), "purity": round(pur_all, 3)},
    ])
    summary.to_csv(paths.REPORTS / "clustering_summary.csv", index=False)
    placement.to_csv(paths.REPORTS / "clustering_real_placement.csv", index=False)
    return summary, placement


if __name__ == "__main__":
    build_clustering()
