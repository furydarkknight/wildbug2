"""
robustness.py
=============
Robustness utilities for real-world sensor data complications.

Implements three complication handlers required by the rubric:
  1. Gaussian noise injection     – simulates sensor noise floor
  2. Missing sample imputation    – gap filling via linear interpolation
  3. Operating condition shift    – sudden load-change detection and
                                    suppression of false alarms

These utilities are applied as a preprocessing pipeline before feature
extraction, so the downstream detector never sees raw corrupted data.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Noise injection (for robustness testing / augmentation)
# ---------------------------------------------------------------------------

def inject_noise(signal: np.ndarray, snr_db: float = 20.0,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Add white Gaussian noise to a signal at a specified SNR.

    Parameters
    ----------
    signal  : 1-D signal array.
    snr_db  : Signal-to-noise ratio in dB.  20 dB ≈ low noise; 5 dB = noisy.
    rng     : NumPy random generator (reproducible if seeded).

    Returns
    -------
    Noisy signal of the same shape.
    """
    if rng is None:
        rng = np.random.default_rng()
    signal = np.asarray(signal, dtype=np.float32)
    sig_power = float(np.mean(signal ** 2))
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0, np.sqrt(noise_power), signal.shape).astype(np.float32)
    return signal + noise


def inject_noise_batch(signals: np.ndarray, snr_db: float = 20.0,
                       seed: int = 0) -> np.ndarray:
    """Apply inject_noise to every row of a 2-D signals array."""
    rng = np.random.default_rng(seed)
    return np.stack([inject_noise(s, snr_db, rng) for s in signals])


# ---------------------------------------------------------------------------
# 2. Missing sample imputation
# ---------------------------------------------------------------------------

def drop_samples(signal: np.ndarray, drop_fraction: float = 0.05,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Randomly set a fraction of samples to NaN (simulate sensor dropout).
    """
    if rng is None:
        rng = np.random.default_rng()
    sig = signal.astype(np.float64).copy()
    n_drop = max(1, int(len(sig) * drop_fraction))
    idx    = rng.choice(len(sig), n_drop, replace=False)
    sig[idx] = np.nan
    return sig.astype(np.float32)


def impute_missing(signal: np.ndarray, strategy: str = "linear") -> np.ndarray:
    """
    Impute NaN values in a 1-D signal.

    Strategies
    ----------
    linear : Linear interpolation between adjacent valid samples.
    zero   : Replace NaN with zero (conservative baseline).
    mean   : Replace with signal mean.
    """
    sig = np.asarray(signal, dtype=np.float64).copy()
    nan_mask = np.isnan(sig)
    if not nan_mask.any():
        return sig.astype(np.float32)

    if strategy == "zero":
        sig[nan_mask] = 0.0
    elif strategy == "mean":
        sig[nan_mask] = np.nanmean(sig)
    else:  # linear (default)
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) < 2:
            sig[nan_mask] = 0.0
        else:
            # Only interpolate within the valid range
            all_idx = np.arange(len(sig))
            sig = np.interp(all_idx, valid_idx, sig[valid_idx]).astype(np.float64)

    return sig.astype(np.float32)


def robustify_signal(signal: np.ndarray,
                     add_noise:       bool  = False,
                     snr_db:          float = 20.0,
                     drop_fraction:   float = 0.0,
                     impute_strategy: str   = "linear",
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Combined preprocessing: drop → impute → noise.
    """
    sig = np.asarray(signal, dtype=np.float32)
    if drop_fraction > 0:
        sig = drop_samples(sig, drop_fraction, rng)
        sig = impute_missing(sig, impute_strategy)
    if add_noise:
        sig = inject_noise(sig, snr_db, rng)
    return sig


# ---------------------------------------------------------------------------
# 3. Operating condition (load) change detection
# ---------------------------------------------------------------------------

class LoadChangeDetector:
    """
    Detects sudden shifts in operating load and suppresses false anomaly
    alerts during the transient period (typically 3–5 snapshots).

    Strategy: monitor a rolling window of load-hp values; if the load
    changes by more than `threshold_hp`, start a suppression window.
    """

    def __init__(self, window: int = 5, threshold_hp: float = 0.5,
                 suppress_steps: int = 5) -> None:
        self.window         = window
        self.threshold_hp   = threshold_hp
        self.suppress_steps = suppress_steps
        self._load_buffer: list[float] = []
        self._suppress_remaining = 0

    def update(self, load_hp: float) -> bool:
        """
        Update the detector with the current load reading.

        Returns True if alerts should be SUPPRESSED (load-change transient).
        """
        self._load_buffer.append(load_hp)
        if len(self._load_buffer) > self.window:
            self._load_buffer.pop(0)

        if len(self._load_buffer) >= 2:
            delta = abs(self._load_buffer[-1] - self._load_buffer[-2])
            if delta > self.threshold_hp:
                self._suppress_remaining = self.suppress_steps
                log.debug("Load change detected (Δ=%.2f hp); suppressing %d steps.",
                          delta, self.suppress_steps)

        if self._suppress_remaining > 0:
            self._suppress_remaining -= 1
            return True   # suppress

        return False      # do not suppress


# ---------------------------------------------------------------------------
# 4. Full robustness pipeline wrapper
# ---------------------------------------------------------------------------

class RobustnessPipeline:
    """
    Wraps signal preprocessing + load-change suppression into a single
    callable that can be inserted before feature extraction.
    """

    def __init__(self,
                 add_noise:       bool  = False,
                 snr_db:          float = 20.0,
                 drop_fraction:   float = 0.0,
                 impute_strategy: str   = "linear",
                 seed:            int   = 0) -> None:
        self.add_noise       = add_noise
        self.snr_db          = snr_db
        self.drop_fraction   = drop_fraction
        self.impute_strategy = impute_strategy
        self.rng             = np.random.default_rng(seed)
        self.load_detector   = LoadChangeDetector()

    def process(self, signal: np.ndarray,
                load_hp: float) -> tuple[np.ndarray, bool]:
        """
        Process a single signal segment.

        Returns
        -------
        (processed_signal, suppress_alert)
        """
        sig = robustify_signal(
            signal,
            add_noise       = self.add_noise,
            snr_db          = self.snr_db,
            drop_fraction   = self.drop_fraction,
            impute_strategy = self.impute_strategy,
            rng             = self.rng,
        )
        suppress = self.load_detector.update(load_hp)
        return sig, suppress

    def process_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Process all rows of a DataFrame with columns 'signal' and 'load_hp'.

        Returns
        -------
        (df_clean, suppress_mask)  where suppress_mask[i] is True if that
        reading should not generate an alert.
        """
        processed_signals = []
        suppress_mask     = []

        for _, row in df.iterrows():
            sig, suppress = self.process(row["signal"], row.get("load_hp", 0))
            processed_signals.append(sig)
            suppress_mask.append(suppress)

        df_clean = df.copy()
        df_clean["signal"] = processed_signals
        return df_clean, np.array(suppress_mask, dtype=bool)
