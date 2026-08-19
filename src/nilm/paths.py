"""Central path definitions for the NILM project.

Every module imports paths from here rather than hardcoding strings. If the
folder layout changes, only this file needs updating.
"""

from pathlib import Path

# Project root = parent of src/ regardless of where Python is started from
ROOT = Path(__file__).resolve().parents[2]

DATA       = ROOT / "data"
RAW        = DATA / "raw"
CLEAN      = DATA / "clean"
SYNTH      = DATA / "synthetic"
SYNTH_TRAIN = SYNTH / "training"
SYNTH_TEST  = SYNTH / "test"

OUTPUTS    = ROOT / "outputs"
FIGURES    = OUTPUTS / "figures"
REPORTS    = OUTPUTS / "reports"

FINGERPRINTS_JSON = OUTPUTS / "appliance_fingerprints.json"
SUMMARY_CSV       = OUTPUTS / "understanding_summary.csv"
VERIFICATION_CSV  = REPORTS / "verification_table.csv"


def ensure_dirs():
    """Create every output / data folder if missing. Safe to call repeatedly."""
    for d in [DATA, RAW, CLEAN, SYNTH, SYNTH_TRAIN, SYNTH_TEST,
              OUTPUTS, FIGURES, REPORTS]:
        d.mkdir(parents=True, exist_ok=True)
