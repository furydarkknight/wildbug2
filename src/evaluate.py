"""
evaluate.py
===========
Evaluation pipeline for MachinaSense.

Computes
--------
  - Classification metrics  : precision, recall, F1, accuracy, confusion matrix
  - Early-warning lead time : how many hours before a defined fault event
                              the first alert fires (IMS timeline)
  - Robustness delta        : metric degradation when noise/missing data is added
  - Per-load breakdown      : performance at each operating condition
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score,
)

log = logging.getLogger(__name__)

# Threshold for converting anomaly score to binary prediction
DEFAULT_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Binary classification metrics
# ---------------------------------------------------------------------------

def classification_report(y_true: np.ndarray, scores: np.ndarray,
                           threshold: float = DEFAULT_THRESHOLD,
                           pos_label: int = 1) -> dict:
    """
    Compute precision, recall, F1, accuracy, AUC from anomaly scores.

    Parameters
    ----------
    y_true    : Binary ground-truth labels (0 = normal, 1 = fault).
    scores    : Anomaly scores in [0, 1].
    threshold : Score cut-off for positive prediction.
    pos_label : Label treated as positive (default 1 = fault).

    Returns
    -------
    dict with keys: precision, recall, f1, accuracy, auc, threshold,
                    n_positives, n_negatives, confusion_matrix.
    """
    y_pred = (scores >= threshold).astype(int)

    # Handle edge cases (all one class in test set)
    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc = float("nan")

    try:
        prec  = float(precision_score(y_true, y_pred, pos_label=pos_label,
                                       zero_division=0))
        rec   = float(recall_score(y_true, y_pred, pos_label=pos_label,
                                    zero_division=0))
        f1    = float(f1_score(y_true, y_pred, pos_label=pos_label,
                                zero_division=0))
        acc   = float(accuracy_score(y_true, y_pred))
        cm    = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    except Exception as exc:
        log.warning("Metrics computation error: %s", exc)
        prec = rec = f1 = acc = 0.0
        cm = [[0, 0], [0, 0]]

    return {
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "accuracy":         round(acc,  4),
        "auc":              round(auc,  4) if not np.isnan(auc) else None,
        "threshold":        threshold,
        "n_positives":      int(np.sum(y_true == 1)),
        "n_negatives":      int(np.sum(y_true == 0)),
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Per-load breakdown
# ---------------------------------------------------------------------------

def per_load_metrics(y_true: np.ndarray, scores: np.ndarray,
                     loads: np.ndarray,
                     threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Compute classification metrics separately for each load condition.

    Returns dict mapping load_hp → metric dict.
    """
    result = {}
    for load in np.unique(loads):
        mask = loads == load
        if mask.sum() == 0:
            continue
        result[int(load)] = classification_report(
            y_true[mask], scores[mask], threshold
        )
    return result


# ---------------------------------------------------------------------------
# Early-warning lead time (IMS-style timeline)
# ---------------------------------------------------------------------------

def compute_lead_time(timestamps: np.ndarray, scores: np.ndarray,
                      fault_start_time: float,
                      threshold: float = DEFAULT_THRESHOLD) -> Optional[float]:
    """
    Measure how early the detector fires relative to a known fault onset.

    Parameters
    ----------
    timestamps       : Unix-style timestamps for each reading (seconds).
    scores           : Anomaly scores, same length as timestamps.
    fault_start_time : Timestamp of the confirmed fault onset.
    threshold        : Score threshold for declaring an alert.

    Returns
    -------
    Lead time in hours (positive = alert before fault), or None if no
    alert fired before the fault onset.
    """
    pre_fault_mask = timestamps < fault_start_time
    pre_fault_scores = scores[pre_fault_mask]
    pre_fault_times  = timestamps[pre_fault_mask]

    fired_idx = np.where(pre_fault_scores >= threshold)[0]
    if len(fired_idx) == 0:
        return None

    first_alert_time = pre_fault_times[fired_idx[0]]
    lead_seconds     = fault_start_time - first_alert_time
    return round(float(lead_seconds) / 3600.0, 2)


def ims_early_warning(ims_df: pd.DataFrame, scores: np.ndarray,
                      threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Compute early-warning statistics on the IMS timeline DataFrame.

    IMS convention: fault_code == 1 marks the confirmed fault period.
    """
    if "fault_code" not in ims_df.columns:
        return {}

    fault_rows  = ims_df[ims_df["fault_code"] == 1]
    normal_rows = ims_df[ims_df["fault_code"] == 0]

    if fault_rows.empty:
        return {}

    fault_start_time = float(fault_rows["timestamp"].min())
    timestamps       = ims_df["timestamp"].values

    lead_hours = compute_lead_time(timestamps, scores, fault_start_time, threshold)

    # Precision / recall on the IMS binary labels
    y_true = (ims_df["fault_code"].values > 0).astype(int)
    met    = classification_report(y_true, scores, threshold)

    return {
        "fault_onset_time_h": round(fault_start_time / 3600.0, 1),
        "lead_time_hours":    lead_hours,
        "detection_metrics":  met,
    }


# ---------------------------------------------------------------------------
# Robustness delta evaluation
# ---------------------------------------------------------------------------

def robustness_evaluation(y_true: np.ndarray,
                           clean_scores: np.ndarray,
                           noisy_scores: np.ndarray,
                           missing_scores: np.ndarray,
                           threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Compare metric degradation when noise or missing data is introduced.

    Parameters
    ----------
    y_true          : Ground-truth binary labels.
    clean_scores    : Scores on original clean signals.
    noisy_scores    : Scores after noise injection.
    missing_scores  : Scores after missing-sample imputation.

    Returns
    -------
    dict with baseline metrics and delta metrics.
    """
    baseline = classification_report(y_true, clean_scores, threshold)
    noisy    = classification_report(y_true, noisy_scores, threshold)
    missing  = classification_report(y_true, missing_scores, threshold)

    def _delta(a: dict, b: dict) -> dict:
        return {
            k: round(b[k] - a[k], 4)
            for k in ("precision", "recall", "f1", "accuracy")
            if isinstance(a.get(k), float) and isinstance(b.get(k), float)
        }

    return {
        "baseline":        baseline,
        "noisy":           noisy,
        "missing":         missing,
        "delta_noise":     _delta(baseline, noisy),
        "delta_missing":   _delta(baseline, missing),
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: dict, title: str = "") -> None:
    """Print a formatted metrics table to stdout."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    for key, val in metrics.items():
        if isinstance(val, dict):
            continue
        if isinstance(val, list):
            continue
        print(f"  {key:<28} {val}")
    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        print(f"\n  Confusion matrix (rows=true, cols=pred):")
        print(f"    [Normal  ] pred0={cm[0][0]:>6}  pred1={cm[0][1]:>6}")
        print(f"    [Fault   ] pred0={cm[1][0]:>6}  pred1={cm[1][1]:>6}")
    print()
