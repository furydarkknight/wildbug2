"""
features.py
===========
Signal feature extraction for rotating-machinery vibration data.

Extracts time-domain, frequency-domain, and envelope features from
raw 1-D vibration segments.  All features are deterministic and
interpretable — each one maps directly to a physical phenomenon.

Feature groups
--------------
Time domain  : RMS, peak, crest factor, kurtosis, skewness, shape factor,
               impulse factor, peak-to-peak, variance, mean absolute deviation
Frequency    : dominant frequency, spectral centroid, spectral rolloff,
               spectral flatness, spectral entropy, band energy ratios
               (1–5× shaft frequency), total harmonic distortion
Envelope     : envelope RMS, envelope kurtosis (sensitive to impulse faults),
               squared envelope spectral kurtosis (SEK)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew
from typing import Sequence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def _band_energy(psd: np.ndarray, freqs: np.ndarray,
                 f_low: float, f_high: float) -> float:
    mask = (freqs >= f_low) & (freqs < f_high)
    return float(np.sum(psd[mask]))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(signal: np.ndarray,
                     sample_rate: int = 12_000,
                     shaft_freq: float = 60.0) -> dict[str, float]:
    """
    Extract a flat feature vector from a single vibration segment.

    Parameters
    ----------
    signal      : 1-D numpy array (raw acceleration or velocity).
    sample_rate : Sampling frequency in Hz.
    shaft_freq  : Running shaft frequency in Hz (used for harmonic bands).

    Returns
    -------
    dict mapping feature name → float value.
    """
    sig = np.asarray(signal, dtype=np.float64)
    n = len(sig)

    # ------------------------------------------------------------------
    # Time-domain features
    # ------------------------------------------------------------------
    rms          = float(np.sqrt(np.mean(sig ** 2)))
    peak         = float(np.max(np.abs(sig)))
    peak_to_peak = float(np.ptp(sig))
    variance     = float(np.var(sig))
    mean_abs_dev = float(np.mean(np.abs(sig - np.mean(sig))))
    crest_factor = _safe_div(peak, rms)
    kurt         = float(kurtosis(sig, fisher=True))   # excess kurtosis
    skewness     = float(skew(sig))
    # Shape factor: RMS / mean absolute value
    mean_abs     = float(np.mean(np.abs(sig)))
    shape_factor = _safe_div(rms, mean_abs)
    # Impulse factor: peak / mean absolute value
    impulse_factor = _safe_div(peak, mean_abs)

    time_feats = {
        "rms":             rms,
        "peak":            peak,
        "peak_to_peak":    peak_to_peak,
        "variance":        variance,
        "mean_abs_dev":    mean_abs_dev,
        "crest_factor":    crest_factor,
        "kurtosis":        kurt,
        "skewness":        skewness,
        "shape_factor":    shape_factor,
        "impulse_factor":  impulse_factor,
    }

    # ------------------------------------------------------------------
    # Frequency-domain features
    # ------------------------------------------------------------------
    fft_vals = np.fft.rfft(sig * np.hanning(n))
    freqs    = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    psd      = (np.abs(fft_vals) ** 2) / n        # power spectral density

    total_power     = float(np.sum(psd)) + 1e-12
    dom_freq_idx    = int(np.argmax(psd))
    dominant_freq   = float(freqs[dom_freq_idx])
    spectral_centroid = float(np.sum(freqs * psd) / total_power)

    # Spectral rolloff (95 % energy)
    cumulative = np.cumsum(psd)
    rolloff_idx = int(np.searchsorted(cumulative, 0.95 * total_power))
    spectral_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # Spectral flatness (geometric mean / arithmetic mean of PSD)
    psd_pos = np.maximum(psd, 1e-12)
    spectral_flatness = float(
        np.exp(np.mean(np.log(psd_pos))) / np.mean(psd_pos)
    )

    # Spectral entropy
    psd_norm       = psd_pos / psd_pos.sum()
    spectral_entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))

    # Harmonic band energies (1×–5× shaft frequency)
    bw = shaft_freq * 0.5          # half-bandwidth around each harmonic
    band_feats = {}
    for h in range(1, 6):
        centre = h * shaft_freq
        band_feats[f"band_{h}x"] = _band_energy(psd, freqs,
                                                  centre - bw, centre + bw)

    # Total harmonic distortion (energy in 2–5× vs 1×)
    fund = band_feats["band_1x"] + 1e-12
    thd  = sum(band_feats[f"band_{h}x"] for h in range(2, 6)) / fund

    freq_feats = {
        "dominant_freq":    dominant_freq,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff,
        "spectral_flatness": spectral_flatness,
        "spectral_entropy":  spectral_entropy,
        "thd":               thd,
        **band_feats,
    }

    # ------------------------------------------------------------------
    # Envelope (demodulation) features
    #  – highly sensitive to impulse-type faults (inner/outer race, ball)
    # ------------------------------------------------------------------
    analytic  = hilbert(sig)
    envelope  = np.abs(analytic)

    env_rms  = float(np.sqrt(np.mean(envelope ** 2)))
    env_kurt = float(kurtosis(envelope, fisher=True))

    # Squared envelope spectrum kurtosis (spectral kurtosis proxy)
    env2_fft = np.fft.rfft(envelope ** 2)
    env2_psd = (np.abs(env2_fft) ** 2) / len(env2_fft)
    env2_psd_pos = np.maximum(env2_psd, 1e-12)
    env2_norm = env2_psd_pos / env2_psd_pos.sum()
    sek = float(-np.sum(env2_norm * np.log(env2_norm)))

    envelope_feats = {
        "env_rms":  env_rms,
        "env_kurt": env_kurt,
        "sek":      sek,
    }

    return {**time_feats, **freq_feats, **envelope_feats}


def extract_feature_matrix(signals: Sequence[np.ndarray],
                            sample_rate: int = 12_000,
                            shaft_freqs: Sequence[float] | None = None
                            ) -> np.ndarray:
    """
    Batch-extract features from a list of signal segments.

    Parameters
    ----------
    signals     : List / array of 1-D segments.
    sample_rate : Sampling frequency.
    shaft_freqs : Per-segment shaft frequency (defaults to 60 Hz for all).

    Returns
    -------
    2-D numpy array of shape (n_segments, n_features).
    """
    if shaft_freqs is None:
        shaft_freqs = [60.0] * len(signals)

    rows = []
    for sig, sf in zip(signals, shaft_freqs):
        rows.append(list(extract_features(sig, sample_rate, sf).values()))
    return np.array(rows, dtype=np.float32)


def feature_names(sample_rate: int = 12_000,
                  shaft_freq: float = 60.0) -> list[str]:
    """Return the ordered feature name list (matches extract_feature_matrix columns)."""
    dummy = np.random.randn(1024).astype(np.float32)
    return list(extract_features(dummy, sample_rate, shaft_freq).keys())
