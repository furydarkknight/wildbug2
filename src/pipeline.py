"""
pipeline.py
===========
End-to-end MachinaSense pipeline.

Orchestrates: data loading → robustness preprocessing → feature extraction
→ train/test split → detector training → scoring → alert generation
→ evaluation.

Can be imported as a library or run via the CLI (demo.py).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data_loader import load_dataset
from .features import extract_feature_matrix, feature_names
from .detector import MachinaSenseDetector
from .alerts import AlertGenerator, Alert
from .robustness import RobustnessPipeline
from .evaluate import (
    classification_report, per_load_metrics,
    ims_early_warning, robustness_evaluation,
    print_metrics_table,
)

log = logging.getLogger(__name__)

SAMPLE_RATE_CWRU = 12_000
SAMPLE_RATE_IMS  = 20_000
CWRU_SHAFT_FREQ  = 60.0    # Hz at 0 hp; increases slightly with load


def _shaft_freq(load_hp: int) -> float:
    """Approximate shaft frequency from load (simple linear model)."""
    return CWRU_SHAFT_FREQ + load_hp * 5.0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class MachinaSensePipeline:
    """
    Configurable end-to-end pipeline.

    Parameters
    ----------
    cwru_dir        : Path to CWRU .mat files (optional; uses synthetic if absent).
    ims_dir         : Path to IMS snapshot files (optional).
    train_ratio     : Fraction of normal data used for training (rest = test).
    noise_test      : If True, evaluate robustness under noise injection.
    missing_test    : If True, evaluate robustness under dropped samples.
    model_path      : Save/load trained detector here (None = don't persist).
    score_threshold : Anomaly score above which an alert is raised.
    """

    def __init__(self,
                 cwru_dir:        Optional[str] = None,
                 ims_dir:         Optional[str] = None,
                 train_ratio:     float = 0.70,
                 noise_test:      bool  = True,
                 missing_test:    bool  = True,
                 model_path:      Optional[str] = None,
                 score_threshold: float = 0.45) -> None:
        self.cwru_dir        = cwru_dir
        self.ims_dir         = ims_dir
        self.train_ratio     = train_ratio
        self.noise_test      = noise_test
        self.missing_test    = missing_test
        self.model_path      = model_path
        self.score_threshold = score_threshold

        self.detector  = MachinaSenseDetector()
        self.alert_gen = AlertGenerator(score_threshold=score_threshold)
        self.results_  = {}

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.info("STEP 1: Loading datasets")
        cwru_df, ims_df = load_dataset(self.cwru_dir, self.ims_dir)
        log.info("  CWRU: %d segments | IMS: %d segments", len(cwru_df), len(ims_df))
        return cwru_df, ims_df

    # ------------------------------------------------------------------
    # Step 2: Preprocess + extract features
    # ------------------------------------------------------------------

    def _extract(self, df: pd.DataFrame,
                 sample_rate: int,
                 pipeline: Optional[RobustnessPipeline] = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features from a DataFrame.

        Returns (X, y, loads, timestamps).
        """
        signals   = list(df["signal"])
        loads     = df["load_hp"].values.astype(int)
        y         = (df["fault_code"].values > 0).astype(int)
        ts        = df["timestamp"].values

        if pipeline is not None:
            processed, _ = pipeline.process_dataframe(df)
            signals = list(processed["signal"])

        shaft_freqs = [_shaft_freq(int(l)) for l in loads]
        X = extract_feature_matrix(signals, sample_rate, shaft_freqs)

        # Replace NaN/Inf that can arise from degenerate segments
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, y, loads, ts

    # ------------------------------------------------------------------
    # Step 3: Train/test split
    # ------------------------------------------------------------------

    @staticmethod
    def _train_test_split(cwru_df: pd.DataFrame,
                          train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Stratified split:
          - Train set: only normal data (for unsupervised training)
          - Test  set: all fault types + remaining normal (for evaluation)
        """
        normal_df = cwru_df[cwru_df["fault_code"] == 0].copy()
        fault_df  = cwru_df[cwru_df["fault_code"]  > 0].copy()

        # Shuffle normal data
        normal_df = normal_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        n_train = int(len(normal_df) * train_ratio)
        train_df = normal_df.iloc[:n_train]
        test_normal = normal_df.iloc[n_train:]

        test_df = pd.concat([test_normal, fault_df], ignore_index=True)
        test_df = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        log.info("  Train: %d normal samples", len(train_df))
        log.info("  Test:  %d samples (%d normal, %d fault)",
                 len(test_df), len(test_normal), len(fault_df))
        return train_df, test_df

    # ------------------------------------------------------------------
    # Step 4: Full run
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute the full pipeline.  Returns a results dict.
        """
        # ── 1. Load ─────────────────────────────────────────────────
        cwru_df, ims_df = self.load()

        # ── 2. Split ─────────────────────────────────────────────────
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.info("STEP 2: Train/test split")
        train_df, test_df = self._train_test_split(cwru_df, self.train_ratio)

        # ── 3. Feature extraction ─────────────────────────────────────
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.info("STEP 3: Extracting features")
        X_train, _, loads_train, _ = self._extract(train_df, SAMPLE_RATE_CWRU)
        X_test,  y_test,  loads_test,  ts_test  = self._extract(test_df, SAMPLE_RATE_CWRU)
        feat_names = feature_names(SAMPLE_RATE_CWRU, CWRU_SHAFT_FREQ)
        log.info("  Feature vector size: %d", X_train.shape[1])

        # ── 4. Train detector ─────────────────────────────────────────
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.info("STEP 4: Training load-aware detector (normal data only)")
        if self.model_path and Path(self.model_path).exists():
            self.detector = MachinaSenseDetector.load(self.model_path)
        else:
            self.detector.fit(X_train, loads_train)
            if self.model_path:
                self.detector.save(self.model_path)

        # ── 5. Score test set ─────────────────────────────────────────
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.info("STEP 5: Scoring test set")
        scores_clean = self.detector.score(X_test, loads_test)

        # ── 6. Alert generation ───────────────────────────────────────
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.info("STEP 6: Generating alerts")
        self.alert_gen.compute_feature_stats(X_train, feat_names)

        alerts: list[Alert] = []
        window_scores: list[float] = []
        for i, (score, load, ts) in enumerate(
                zip(scores_clean, loads_test, ts_test)):
            window_scores.append(float(score))
            if len(window_scores) > 10:
                window_scores.pop(0)
            sev  = self.detector.severity(float(score))
            conf = self.detector.confidence(np.array(window_scores))
            feats = dict(zip(feat_names,
                             [float(v) for v in X_test[i]]))
            alert = self.alert_gen.generate(float(ts), float(score),
                                             conf, feats, int(load), sev)
            if alert:
                alerts.append(alert)

        log.info("  Generated %d alerts from %d test samples.", len(alerts), len(X_test))

        # ── 7. Evaluation ─────────────────────────────────────────────
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.info("STEP 7: Evaluating performance")

        main_metrics = classification_report(y_test, scores_clean, self.score_threshold)
        load_metrics = per_load_metrics(y_test, scores_clean, loads_test,
                                         self.score_threshold)
        print_metrics_table(main_metrics, "CWRU Classification Metrics (held-out test)")
        for load_hp, m in load_metrics.items():
            print_metrics_table(m, f"  → Load {load_hp} hp")

        # ── 8. Robustness evaluation ──────────────────────────────────
        rob_results = {}
        if self.noise_test or self.missing_test:
            log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            log.info("STEP 8: Robustness evaluation (noise + missing samples)")

            noise_pipeline   = RobustnessPipeline(add_noise=True, snr_db=15.0)
            missing_pipeline = RobustnessPipeline(drop_fraction=0.10)

            X_noisy,   _, _, _ = self._extract(test_df, SAMPLE_RATE_CWRU, noise_pipeline)
            X_missing, _, _, _ = self._extract(test_df, SAMPLE_RATE_CWRU, missing_pipeline)

            X_noisy   = np.nan_to_num(X_noisy,   nan=0.0, posinf=0.0, neginf=0.0)
            X_missing = np.nan_to_num(X_missing, nan=0.0, posinf=0.0, neginf=0.0)

            scores_noisy   = self.detector.score(X_noisy,   loads_test)
            scores_missing = self.detector.score(X_missing, loads_test)

            rob_results = robustness_evaluation(
                y_test, scores_clean, scores_noisy, scores_missing,
                self.score_threshold
            )
            print_metrics_table(rob_results["baseline"],  "Robustness – Baseline")
            print_metrics_table(rob_results["noisy"],     "Robustness – Noisy (15 dB SNR)")
            print_metrics_table(rob_results["missing"],   "Robustness – 10 % Missing Samples")

        # ── 9. IMS early-warning ──────────────────────────────────────
        ims_results = {}
        if not ims_df.empty:
            log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            log.info("STEP 9: IMS early-warning lead time")
            X_ims, _, loads_ims, _ = self._extract(ims_df, SAMPLE_RATE_IMS)
            X_ims   = np.nan_to_num(X_ims, nan=0.0, posinf=0.0, neginf=0.0)
            scores_ims = self.detector.score(X_ims, loads_ims)
            ims_results = ims_early_warning(ims_df, scores_ims, self.score_threshold)
            if ims_results.get("lead_time_hours") is not None:
                log.info("  ✔ Early warning fired %.1f h before fault onset.",
                         ims_results["lead_time_hours"])
            else:
                log.info("  ✖ No pre-fault alert fired.")

        # ── Collect results ───────────────────────────────────────────
        self.results_ = {
            "main_metrics":  main_metrics,
            "load_metrics":  load_metrics,
            "robustness":    rob_results,
            "ims_results":   ims_results,
            "alerts":        alerts,
            "n_alerts":      len(alerts),
        }
        return self.results_
