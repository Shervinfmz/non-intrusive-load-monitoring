"""NILM — Non-Intrusive Load Monitoring (TH Köln, ddmo-sose-26)

Pipeline:
    nilm.acquisition       Stage 1: read real PAC4200, preprocess, fingerprint
    nilm.synthesizer       Stage 2: generate synthetic train + test runs
    nilm.verification      Stage 3: verify synthetic vs real
    nilm.visualization     Stage 4: quick-look plotter for any PAC4200 CSV
"""

from .paths import ensure_dirs

__all__ = ["ensure_dirs"]
__version__ = "0.1.0"   # Milestone 1 complete
