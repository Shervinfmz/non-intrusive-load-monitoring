"""Stage 4: Visualization.

Quick-look plotter for any PAC4200 CSV (real or synthetic — same schema).
Used from the walkthrough notebook for "show me this specific file" moments,
and runnable as a module:

    python -m nilm.visualization data/raw/pac4200_toaster_200ms__1_.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import registry


PANELS = [
    ("p_total_w",        "Active Power P (W)",      "tab:blue"),
    ("q_total_var",      "Reactive Power Q (var)",  "tab:orange"),
    ("pf_total",         "Power Factor",            "tab:green"),
    ("thd_i_l1_percent", "Current THD (%)",         "tab:red"),
    ("i_l1_a",           "Current I (A)",           "tab:purple"),
    ("u_l1_n_v",         "Voltage U (V)",           "tab:gray"),
]


def plot_run(csv_path, save_to=None, show=False):
    """Plot one PAC4200 CSV across the 6 most informative channels.

    Args:
        csv_path:  path to a PAC4200-format CSV (real or synthetic)
        save_to:   if given, write a PNG there
        show:      if True and not in notebook, plt.show() blocks
    Returns:
        the matplotlib Figure (so notebook cells can re-use it)
    """
    csv_path = Path(csv_path)
    df = registry.load_pac(csv_path)

    device = df["device_name"].iloc[0]
    rid = df["run_id"].iloc[0] if "run_id" in df.columns else None
    run_id = rid if pd.notna(rid) else csv_path.name

    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    axes = axes.flatten()
    for ax, (col, title, color) in zip(axes, PANELS):
        ax.plot(df["t_sec"], df[col], color=color, linewidth=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{device}    (run: {run_id})", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_to:
        fig.savefig(save_to, dpi=130)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m nilm.visualization <csv_path> [output_png]")
        sys.exit(1)
    csv_path = sys.argv[1]
    save_to = sys.argv[2] if len(sys.argv) > 2 else None
    plot_run(csv_path, save_to=save_to, show=(save_to is None))
    if save_to:
        print(f"Saved figure to {save_to}")
