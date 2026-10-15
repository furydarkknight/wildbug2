"""
detector.py
===========
Load-aware anomaly detector for industrial bearing vibration.

Design
------
* Training is done ONLY on normal operating data (unsupervised).
  No fault labels are needed to train — a machine can be monitored
  from day one.
* The detector fits one model per load condition (0 hp, 1 hp, 2 hp, 3 hp)
  so that a heavy-load reading is never flagged just because the model was
  trained at light load.
* Two complementary detectors are combined:
    1. Isolation Forest  – fast, robust to high-dimensional features.
    2. Autoencoder       – deep reconstruction error, especially good at
                           detecting gradual drift and compound faults.
* Scores from both are Z-normalised and averaged into a single anomaly
  score in [0, 1].

Confidence
----------
Confidence is reported as  1 - (score std / score mean)  over a sliding
window of the last 10 readings, capped to [0.05, 0.99].  It expresses how
consistently anomalous (or normal) the recent readings are.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Autoencoder (lightweight, CPU-friendly, no GPU required)
# ---------------------------------------------------------------------------

class _Autoencoder:
    """
    Minimal feed-forward autoencoder implemented in plain NumPy
    (no PyTorch dependency required).

    Architecture: input → 64 → 32 → 16 → 32 → 64 → input
    Activation:   ReLU (encoder), linear (decoder output).
    Training:     mini-batch SGD with L2 reconstruction loss.
    """

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (64, 32, 16),
                 lr: float = 1e-3, epochs: int = 50, batch_size: int = 64,
                 seed: int = 42) -> None:
        self.input_dim  = input_dim
        self.hidden     = hidden
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.rng        = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases:  list[np.ndarray] = []
        self._init_weights()

    def _init_weights(self) -> None:
        dims = [self.input_dim, *self.hidden, *reversed(self.hidden), self.input_dim]
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.weights.append(self.rng.normal(0, scale, (dims[i], dims[i + 1])))
            self.biases.append(np.zeros(dims[i + 1]))

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)

    def _forward(self, x: np.ndarray) -> list[np.ndarray]:
        activations = [x]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ W + b
            # Linear activation at output layer
            a = z if i == len(self.weights) - 1 else self._relu(z)
            activations.append(a)
        return activations

    def _backward(self, activations: list[np.ndarray],
                  x: np.ndarray) -> tuple[list, list]:
        n = x.shape[0]
        delta = (activations[-1] - x) / n           # MSE gradient
        dW_list, db_list = [], []
        for i in reversed(range(len(self.weights))):
            dW = activations[i].T @ delta
            db = delta.sum(axis=0)
            dW_list.insert(0, dW)
            db_list.insert(0, db)
            if i > 0:
                delta = delta @ self.weights[i].T * self._relu_grad(
                    activations[i]
                )
        return dW_list, db_list

    def fit(self, X: np.ndarray) -> "_Autoencoder":
        n = len(X)
        for epoch in range(self.epochs):
            idx = self.rng.permutation(n)
            X_shuffled = X[idx]
            total_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = X_shuffled[start: start + self.batch_size]
                acts = self._forward(batch)
                loss = float(np.mean((acts[-1] - batch) ** 2))
                total_loss += loss
                dWs, dbs = self._backward(acts, batch)
                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * dWs[i]
                    self.biases[i]  -= self.lr * dbs[i]
            if (epoch + 1) % 10 == 0:
                log.debug("  AE epoch %d/%d  loss=%.6f", epoch + 1, self.epochs,
                          total_loss / max(1, n // self.batch_size))
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        acts = self._forward(X)
        return np.mean((acts[-1] - X) ** 2, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-load sub-detector
# ---------------------------------------------------------------------------

class _LoadDetector:
    """Isolation Forest + Autoencoder ensemble for one load condition."""

    def __init__(self, n_estimators: int = 100, contamination: float = 0.05,
                 seed: int = 42) -> None:
        self.scaler = StandardScaler()
        self.iforest = IsolationForest(n_estimators=n_estimators,
                                       contamination=contamination,
                                       random_state=seed, n_jobs=-1)
        self.ae: Optional[_Autoencoder] = None
        self._if_mean  = 0.0
        self._if_std   = 1.0
        self._ae_mean  = 0.0
        self._ae_std   = 1.0
        self.fitted    = False

    def fit(self, X_raw: np.ndarray) -> "_LoadDetector":
        X = self.scaler.fit_transform(X_raw)

        # Isolation Forest
        self.iforest.fit(X)
        if_scores = -self.iforest.score_samples(X)   # higher = more anomalous
        self._if_mean = float(np.mean(if_scores))
        self._if_std  = max(float(np.std(if_scores)), 1e-6)

        # Autoencoder
        self.ae = _Autoencoder(input_dim=X.shape[1])
        self.ae.fit(X)
        ae_scores = self.ae.reconstruction_error(X)
        self._ae_mean = float(np.mean(ae_scores))
        self._ae_std  = max(float(np.std(ae_scores)), 1e-6)

        self.fitted = True
        return self

    def score(self, X_raw: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 1] for each sample."""
        if not self.fitted:
            raise RuntimeError("Detector not fitted.")
        X = self.scaler.transform(X_raw)

        if_raw  = -self.iforest.score_samples(X)
        ae_raw  = self.ae.reconstruction_error(X)

        # Z-normalise relative to training distribution
        if_z = (if_raw - self._if_mean) / self._if_std
        ae_z = (ae_raw - self._ae_mean) / self._ae_std

        # Combine and squash into [0, 1] via sigmoid
        combined = (if_z + ae_z) / 2.0
        scores   = 1.0 / (1.0 + np.exp(-combined))
        return scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Public detector
# ---------------------------------------------------------------------------

class MachinaSenseDetector:
    """
    Load-aware anomaly detector.

    Usage
    -----
    detector = MachinaSenseDetector()
    detector.fit(X_normal, loads_normal)   # train on normal data only
    scores = detector.score(X_new, loads)  # inference
    alerts = detector.alerts(X_new, loads, timestamps, feature_names)
    """

    SEVERITY_THRESHOLDS = {
        "critical": 0.80,
        "high":     0.65,
        "medium":   0.50,
        "low":      0.35,
    }

    def __init__(self, contamination: float = 0.05) -> None:
        self.contamination = contamination
        self._detectors: dict[int, _LoadDetector] = {}
        self._score_history: list[float] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, loads: np.ndarray) -> "MachinaSenseDetector":
        """
        Train on normal operating data.

        Parameters
        ----------
        X     : Feature matrix (n_samples, n_features), normal data only.
        loads : Integer load condition per sample (0, 1, 2, 3 for CWRU hp).
        """
        unique_loads = np.unique(loads)
        log.info("Training load-aware detector on %d samples, loads=%s",
                 len(X), unique_loads.tolist())
        for load in unique_loads:
            mask = loads == load
            X_load = X[mask]
            if len(X_load) < 10:
                log.warning("  Load %d: only %d samples – skipping.", load, len(X_load))
                continue
            det = _LoadDetector(contamination=self.contamination)
            det.fit(X_load)
            self._detectors[int(load)] = det
            log.info("  Load %d: fitted on %d samples.", int(load), len(X_load))

        # Fallback: global detector for unseen load values
        global_det = _LoadDetector(contamination=self.contamination)
        global_det.fit(X)
        self._detectors[-1] = global_det
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, X: np.ndarray, loads: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores in [0, 1].  Higher = more anomalous.
        Dispatches samples batched by load condition to its load-specific sub-detector.
        """
        scores = np.zeros(len(X), dtype=np.float32)
        loads_arr = np.asarray(loads)
        for ld in np.unique(loads_arr):
            mask = loads_arr == ld
            det = self._detectors.get(int(ld), self._detectors.get(-1))
            if det is not None:
                scores[mask] = det.score(X[mask])
        return scores

    def severity(self, score: float) -> str:
        for sev, thresh in self.SEVERITY_THRESHOLDS.items():
            if score >= thresh:
                return sev
        return "normal"

    def confidence(self, recent_scores: np.ndarray) -> float:
        """Confidence in [0.05, 0.99]: consistency of recent readings."""
        if len(recent_scores) < 2:
            return 0.5
        mu  = float(np.mean(recent_scores))
        std = float(np.std(recent_scores))
        c   = 1.0 - _safe_div(std, mu + 1e-6)
        return float(np.clip(c, 0.05, 0.99))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "MachinaSenseDetector":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info("Model loaded from %s", path)
        return obj


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0
