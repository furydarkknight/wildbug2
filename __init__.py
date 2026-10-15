"""
machina_sense/__init__.py
"""
__version__ = "1.0.0"
__author__  = "MachinaSense Team"

from machina_sense.features  import extract_features, FEATURE_NAMES
from machina_sense.detector  import EnsembleDetector, train_ensemble
from machina_sense.alerts    import AlertEngine, MaintenanceAlert
from machina_sense.evaluator import run_full_evaluation

__all__ = [
    "extract_features",
    "FEATURE_NAMES",
    "EnsembleDetector",
    "train_ensemble",
    "AlertEngine",
    "MaintenanceAlert",
    "run_full_evaluation",
]
