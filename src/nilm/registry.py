"""Appliance registry + robust PAC4200 loader (single source of truth).

The pipeline is DATA-DRIVEN: instead of hardcoding appliance names, it scans
data/raw/ and reads each file's device_name column. Add a new measurement file
and it appears automatically.

Handles the realities of the lab data:
  - interleaved blank rows (every other row NaN in some Node-RED logs) -> dropped
  - device_name only on some rows -> first valid value used
  - messy device names (e.g. "Laptop_Ravi") -> cleaned via DEVICE_RENAME
  - stale / wrong-format CSVs (no device_name column) -> skipped
  - generation devices (PV, negative power) -> excluded from the classifier set
    but still discoverable for Milestone 3
"""

import glob
from pathlib import Path
import pandas as pd

from . import paths

NUMERIC_COLS = [
    "u_l1_n_v", "i_l1_a", "p_total_w", "s_total_va", "s_calc_va",
    "q_total_var", "pf_total", "frequency_hz",
    "thd_u_l1_percent", "thd_i_l1_percent", "block_time_difference_ms",
]

# Clean up messy raw device names -> canonical class labels
DEVICE_RENAME = {
    "Laptop_Ravi": "Laptop",
}

# Generation / non-switching loads: excluded from the on/off classifier,
# reserved for Milestone 3 (they reduce net load rather than switching on/off).
EXCLUDE_FROM_CLASSIFICATION = {"PV"}


def clean_name(raw):
    return DEVICE_RENAME.get(str(raw).strip(), str(raw).strip())


def slug(device):
    """Filesystem-safe slug for a device name."""
    s = clean_name(device).lower().replace(" ", "_").replace(".", "p")
    return (s.replace("\u00e4", "ae").replace("\u00f6", "oe")
              .replace("\u00fc", "ue").replace("\u00df", "ss"))


def load_pac(path):
    """Load one PAC4200 CSV robustly. Drops blank rows, parses timestamps,
    coerces numerics, keeps off-state NaNs in pf/thd, cleans device_name.

    Returns a DataFrame with a 't_sec' elapsed-seconds column.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "device_name" not in df.columns or "p_total_w" not in df.columns:
        raise ValueError(f"{path} is not in PAC4200 15-column format")

    # 1) drop interleaved blank rows (need a timestamp and a power value)
    df = df.dropna(subset=["timestamp_iso", "p_total_w"]).reset_index(drop=True)

    # 2) timestamps
    df["timestamp_iso"] = pd.to_datetime(df["timestamp_iso"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_iso"]).reset_index(drop=True)

    # 3) numerics
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4) order + elapsed seconds
    df = df.sort_values("timestamp_iso").reset_index(drop=True)
    df["t_sec"] = (df["timestamp_iso"] - df["timestamp_iso"].iloc[0]).dt.total_seconds()

    # 5) clean device name (propagate the first valid one to all rows)
    valid = df["device_name"].dropna()
    if len(valid):
        df["device_name"] = clean_name(valid.iloc[0])
    return df


def discover(include_excluded=False):
    """Scan data/raw and return {clean_device_name: Path}.

    Skips files that are not PAC4200 15-column format (e.g. stale exports).
    By default omits EXCLUDE_FROM_CLASSIFICATION (PV).
    """
    mapping = {}
    for f in sorted(glob.glob(str(paths.RAW / "*.csv"))):
        try:
            head = pd.read_csv(f, nrows=1)
        except Exception:
            continue
        head.columns = [c.strip() for c in head.columns]
        if "device_name" not in head.columns:
            continue  # stale / wrong-format file -> skip silently
        col = pd.read_csv(f, usecols=lambda c: c.strip() == "device_name")
        col.columns = [c.strip() for c in col.columns]
        names = col["device_name"].dropna()
        if len(names) == 0:
            continue
        dev = clean_name(names.iloc[0])
        if not include_excluded and dev in EXCLUDE_FROM_CLASSIFICATION:
            continue
        mapping[dev] = Path(f)
    return mapping


if __name__ == "__main__":
    print("Classification appliances (PV excluded):")
    for dev, p in discover().items():
        print(f"  {dev:24s} <- {p.name}")
    print("\nAll devices (incl. excluded):")
    for dev, p in discover(include_excluded=True).items():
        tag = "  [excluded]" if dev in EXCLUDE_FROM_CLASSIFICATION else ""
        print(f"  {dev:24s} <- {p.name}{tag}")
