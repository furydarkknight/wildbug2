"""
machina_sense/evaluator.py
──────────────────────────
Evaluation pipeline for MachinaSense.

Covers three rubric dimensions:
  1. Detection performance  – precision / recall / F1 on held-out CWRU data
  2. Robustness             – repeat evaluation under noise, missing samples,
                              and simulated load shifts
  3. Early-warning lead time – for IMS-style run-to-failure data, measure how
                               many minutes before the last snapshot the first
                               alert fires

All results are returned as plain dicts / DataFrames so they can be printed
or fed to the visualisation layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from machina_sense.features import (
    extract_features,
    extract_features_df,
    inject_noise,
    inject_missing_samples,
    apply_load_shift,
    N_FEATURES,
)
from machina_sense.detector import EnsembleDetector


# ── 1. Classification evaluation (CWRU held-out set) ─────────────────────────

def evaluate_classification(
    detector: EnsembleDetector,
    X_test: np.ndarray,
    y_true: np.ndarray,
    loads_test: np.ndarray,
    anomaly_labels: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate fault classification on a labelled test set.

    Returns
    -------
    dict with keys: f1_macro, f1_weighted, precision_macro, recall_macro,
                    classification_report_str, confusion_matrix_df,
                    anomaly_detection_f1 (binary: normal vs. any-fault)
    """
    if anomaly_labels is None:
        anomaly_labels = ["inner", "outer", "ball"]

    result_df = detector.score_batch(X_test, loads_test)

    # ── Multi-class fault classification ──────────────────────────────────────
    y_pred_label = result_df["fault_label"].values

    # Only evaluate on rows where classifier gives a known label
    known_mask = np.isin(y_pred_label, list(detector.fault_classifier.classes_)) \
        if detector.fault_classifier else np.ones(len(y_pred_label), dtype=bool)

    report_str = classification_report(
        y_true[known_mask], y_pred_label[known_mask], zero_division=0
    )
    cm = confusion_matrix(
        y_true[known_mask], y_pred_label[known_mask],
        labels=detector.fault_classifier.classes_ if detector.fault_classifier else None,
    )
    labels = detector.fault_classifier.classes_ if detector.fault_classifier else sorted(np.unique(y_true))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    f1_macro    = f1_score(y_true[known_mask], y_pred_label[known_mask], average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true[known_mask], y_pred_label[known_mask], average="weighted", zero_division=0)
    prec_macro  = precision_score(y_true[known_mask], y_pred_label[known_mask], average="macro", zero_division=0)
    rec_macro   = recall_score(y_true[known_mask], y_pred_label[known_mask], average="macro",   zero_division=0)

    # ── Binary anomaly detection ───────────────────────────────────────────────
    y_true_binary = (y_true != "normal").astype(int)
    y_pred_binary = result_df["is_anomaly"].astype(int).values
    binary_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    binary_prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    binary_rec  = recall_score(y_true_binary, y_pred_binary, zero_division=0)

    return {
        "f1_macro":                f1_macro,
        "f1_weighted":             f1_weighted,
        "precision_macro":         prec_macro,
        "recall_macro":            rec_macro,
        "anomaly_f1":              binary_f1,
        "anomaly_precision":       binary_prec,
        "anomaly_recall":          binary_rec,
        "classification_report":   report_str,
        "confusion_matrix":        cm_df,
        "n_test_samples":          int(len(X_test)),
    }


# ── 2. Robustness evaluation ───────────────────────────────────────────────────

def evaluate_robustness(
    detector: EnsembleDetector,
    df_test: pd.DataFrame,
    X_test: np.ndarray,
    y_true: np.ndarray,
    loads_test: np.ndarray,
    fs: float = 12_000.0,
) -> Dict:
    """
    Evaluate model performance under three corruption scenarios and compare
    against the clean baseline:

      1. Gaussian noise  (SNR = 20 dB, 15 dB, 10 dB)
      2. Missing samples (5 %, 15 %, 30 %)
      3. Load shift      (amplitude × 1.3 + offset)

    Returns a dict of scenario → metrics dict.
    """

    def _score(X_aug: np.ndarray) -> float:
        result = detector.score_batch(X_aug, loads_test)
        y_pred_binary = result["is_anomaly"].astype(int).values
        y_true_binary = (y_true != "normal").astype(int)
        return float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

    def _augment_and_extract(aug_fn) -> np.ndarray:
        """Apply aug_fn to raw signal of each row, re-extract features."""
        augmented = []
        for _, row in df_test.iterrows():
            sig = np.asarray(row["signal"])
            aug_sig = aug_fn(sig)
            augmented.append(extract_features(aug_sig, fs=fs))
        return np.vstack(augmented).astype(np.float32)

    results = {}

    # Clean baseline
    results["clean"] = {
        "binary_f1": _score(X_test),
        "description": "No corruption applied",
    }

    # Noise scenarios
    for snr in [20, 15, 10]:
        key = f"noise_snr_{snr}dB"
        X_aug = _augment_and_extract(lambda s, _snr=snr: inject_noise(s, snr_db=_snr))
        results[key] = {
            "binary_f1": _score(X_aug),
            "description": f"Gaussian noise at {snr} dB SNR",
        }

    # Missing-sample scenarios
    for frac in [0.05, 0.15, 0.30]:
        key = f"missing_{int(frac*100)}pct"
        X_aug = _augment_and_extract(lambda s, _f=frac: inject_missing_samples(s, missing_fraction=_f))
        results[key] = {
            "binary_f1": _score(X_aug),
            "description": f"{int(frac*100)}% missing samples (interpolated)",
        }

    # Load shift
    X_aug = _augment_and_extract(apply_load_shift)
    results["load_shift"] = {
        "binary_f1": _score(X_aug),
        "description": "Amplitude × 1.3 + DC offset (simulated load change)",
    }

    return results


# ── 3. Early-warning lead time (IMS / run-to-failure) ─────────────────────────

def evaluate_lead_time(
    detector: EnsembleDetector,
    df_ims: pd.DataFrame,
    X_ims: np.ndarray,
    failure_snapshot_idx: int,
    alert_threshold: float = 0.55,
    fs: float = 20_000.0,
) -> Dict:
    """
    Measure how many snapshots (and minutes) before the labelled failure the
    first alert fires.

    Parameters
    ----------
    df_ims               : IMS-style DataFrame with 'elapsed_min' column
    X_ims                : extracted features for each snapshot
    failure_snapshot_idx : index of the snapshot at which failure is declared
                           (last snapshot for IMS test set 1)
    alert_threshold      : anomaly score above which an alert fires

    Returns
    -------
    dict with: first_alert_idx, first_alert_min, failure_min,
               lead_time_min, lead_time_snapshots, alert_timeline_df
    """
    loads = np.zeros(len(X_ims), dtype=int)
    scores = detector.anomaly_detector.score(X_ims, loads)

    # Use a rolling mean (window=3) to suppress single-snapshot spikes
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(scores.astype(float), size=3)

    alert_mask = smoothed >= alert_threshold
    alert_indices = np.where(alert_mask)[0]

    if len(alert_indices) == 0:
        return {
            "first_alert_idx": None,
            "first_alert_min": None,
            "failure_min":     int(df_ims["elapsed_min"].iloc[failure_snapshot_idx]),
            "lead_time_min":   None,
            "lead_time_snapshots": None,
            "alert_timeline":  _build_timeline(df_ims, scores, smoothed),
            "note": "No alert fired — consider lowering threshold.",
        }

    first_alert_idx = int(alert_indices[0])
    first_alert_min = int(df_ims["elapsed_min"].iloc[first_alert_idx])
    failure_min     = int(df_ims["elapsed_min"].iloc[failure_snapshot_idx])
    lead_time_min   = failure_min - first_alert_min

    return {
        "first_alert_idx":     first_alert_idx,
        "first_alert_min":     first_alert_min,
        "failure_min":         failure_min,
        "lead_time_min":       lead_time_min,
        "lead_time_snapshots": failure_snapshot_idx - first_alert_idx,
        "alert_timeline":      _build_timeline(df_ims, scores, smoothed),
        "note": f"Alert fired {lead_time_min} min ({lead_time_min/60:.1f} h) before failure.",
    }


def _build_timeline(
    df: pd.DataFrame,
    raw_scores: np.ndarray,
    smoothed_scores: np.ndarray,
) -> pd.DataFrame:
    elapsed = df["elapsed_min"].values if "elapsed_min" in df.columns else np.arange(len(raw_scores)) * 10
    return pd.DataFrame({
        "elapsed_min":   elapsed[:len(raw_scores)],
        "anomaly_score": raw_scores,
        "smoothed_score": smoothed_scores,
    })


# ── Convenience: run full evaluation suite ────────────────────────────────────

def run_full_evaluation(
    detector: EnsembleDetector,
    df_test: pd.DataFrame,
    X_test: np.ndarray,
    y_true: np.ndarray,
    loads_test: np.ndarray,
    df_ims: Optional[pd.DataFrame] = None,
    X_ims: Optional[np.ndarray] = None,
    failure_snapshot_idx: Optional[int] = None,
) -> Dict:
    """Run classification + robustness + (optionally) lead-time evaluation."""
    results = {}
    print("── Evaluating classification performance …")
    results["classification"] = evaluate_classification(
        detector, X_test, y_true, loads_test
    )
    print("── Evaluating robustness …")
    results["robustness"] = evaluate_robustness(
        detector, df_test, X_test, y_true, loads_test
    )
    if df_ims is not None and X_ims is not None and failure_snapshot_idx is not None:
        print("── Evaluating early-warning lead time …")
        results["lead_time"] = evaluate_lead_time(
            detector, df_ims, X_ims, failure_snapshot_idx
        )
    return results
