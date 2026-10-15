"""
machina_sense/features.py
─────────────────────────
Signal-processing feature extraction for vibration data.

Every segment (1-D numpy array) is converted into a fixed-length feature vector
that combines time-domain statistics, envelope statistics, and frequency-domain
indicators.  The vector is intentionally small (≈ 20 features) so an Isolation
Forest or lightweight classifier can run in milliseconds on edge hardware.

Feature inventory
─────────────────
Time-domain (7)
  rms          – root-mean-square of raw signal
  peak         – absolute peak value
  crest_factor – peak / rms (early indicator of impulsiveness)
  kurtosis     – statistical kurtosis (sensitive to sharp impulses)
  skewness     – asymmetry of distribution
  shape_factor – rms / mean(|x|)
  impulse_factor – peak / mean(|x|)

Envelope / demodulation (3)
  env_rms      – RMS of Hilbert envelope
  env_kurtosis – kurtosis of Hilbert envelope (very sensitive to bearing faults)
  env_peak     – peak of Hilbert envelope

Frequency-domain (5)
  spectral_centroid – amplitude-weighted mean frequency
  spectral_spread   – amplitude-weighted std of frequency
  spectral_skewness – spectral shape asymmetry
  spectral_kurtosis_freq – kurtosis of spectral shape
  high_freq_energy_ratio – energy above fs/4 relative to total (wear indicator)

FFT band energies (5)
  band_0, band_1, band_2, band_3, band_4
  Equally spaced log-frequency bands from 0 to fs/2.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew


FEATURE_NAMES: List[str] = [
    "rms", "peak", "crest_factor", "kurtosis", "skewness",
    "shape_factor", "impulse_factor",
    "env_rms", "env_kurtosis", "env_peak",
    "spectral_centroid", "spectral_spread",
    "spectral_skewness", "spectral_kurtosis_freq",
    "high_freq_energy_ratio",
    "band_0", "band_1", "band_2", "band_3", "band_4",
]

N_FEATURES = len(FEATURE_NAMES)   # 20


# ── Core extractor ─────────────────────────────────────────────────────────────

def extract_features(segment: np.ndarray, fs: float = 12_000.0) -> np.ndarray:
    """
    Extract the 20-feature vector from a 1-D vibration segment.

    Parameters
    ----------
    segment : 1-D float array of raw vibration samples
    fs      : sampling rate in Hz (12 kHz for CWRU, 20 kHz for IMS)

    Returns
    -------
    1-D float32 array of shape (N_FEATURES,)
    """
    x = np.asarray(segment, dtype=np.float64)
    if x.size < 16:
        return np.zeros(N_FEATURES, dtype=np.float32)

    # ── time domain ───────────────────────────────────────────────────────────
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x))) or 1e-12
    crest_factor = peak / (rms or 1e-12)
    kurt = float(sp_kurtosis(x, fisher=True))   # excess kurtosis; normal ≈ 0
    skewness = float(sp_skew(x))
    shape_factor = (rms or 1e-12) / mean_abs
    impulse_factor = peak / mean_abs

    # ── envelope (Hilbert) ────────────────────────────────────────────────────
    analytic = hilbert(x)
    envelope = np.abs(analytic)
    env_rms = float(np.sqrt(np.mean(envelope ** 2)))
    env_kurt = float(sp_kurtosis(envelope, fisher=True))
    env_peak = float(np.max(envelope))

    # ── frequency domain ──────────────────────────────────────────────────────
    n = len(x)
    fft_amp = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # guard against all-zero spectrum
    amp_sum = fft_amp.sum() or 1e-12
    weights = fft_amp / amp_sum

    centroid = float(np.dot(freqs, weights))
    spread = float(np.sqrt(np.dot((freqs - centroid) ** 2, weights)))
    # Spectral skewness / kurtosis (shape of spectrum)
    spread_safe = spread or 1e-12
    spec_skew = float(np.dot(((freqs - centroid) / spread_safe) ** 3, weights))
    spec_kurt = float(np.dot(((freqs - centroid) / spread_safe) ** 4, weights))

    # High-frequency energy ratio (above fs/4)
    hf_mask = freqs >= fs / 4
    hf_ratio = float(
        (fft_amp[hf_mask] ** 2).sum() / ((fft_amp ** 2).sum() or 1e-12)
    )

    # ── 5 log-spaced frequency band energies ──────────────────────────────────
    band_edges = np.logspace(np.log10(max(freqs[1], 1.0)), np.log10(fs / 2), 6)
    band_energies = []
    for i in range(5):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        energy = float((fft_amp[mask] ** 2).sum())
        band_energies.append(energy)
    # Normalise band energies so they are scale-invariant
    total_energy = sum(band_energies) or 1e-12
    band_energies = [e / total_energy for e in band_energies]

    feat = np.array([
        rms, peak, crest_factor, kurt, skewness,
        shape_factor, impulse_factor,
        env_rms, env_kurt, env_peak,
        centroid, spread,
        spec_skew, spec_kurt,
        hf_ratio,
        *band_energies,
    ], dtype=np.float32)

    # Replace any NaN/Inf that slipped through
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


# ── Batch extractor ────────────────────────────────────────────────────────────

def extract_features_df(df, fs_map: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Extract features for every row in a DataFrame that has a 'signal' column
    (list or 1-D array of samples) and a 'source' column.

    Parameters
    ----------
    df      : DataFrame with columns ['signal', 'source']
    fs_map  : optional mapping {source_name → fs_hz}; defaults built in

    Returns
    -------
    2-D float32 array of shape (n_rows, N_FEATURES)
    """
    default_fs = {
        "cwru":          12_000.0,
        "synthetic":     12_000.0,
        "ims":           20_000.0,
        "synthetic_ims": 20_000.0,
    }
    if fs_map:
        default_fs.update(fs_map)

    features = []
    for _, row in df.iterrows():
        src = row.get("source", "cwru")
        fs = default_fs.get(src, 12_000.0)
        feat = extract_features(np.asarray(row["signal"]), fs=fs)
        features.append(feat)

    return np.vstack(features).astype(np.float32)


# ── Noise & missing-sample augmentation (robustness testing) ──────────────────

def inject_noise(segment: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """
    Add Gaussian white noise at a given SNR (dB) to a signal segment.

    SNR_db = 10 * log10(P_signal / P_noise)
    Lower SNR → more noise.  Typical factory floor: 15-30 dB.
    """
    x = np.asarray(segment, dtype=np.float64)
    signal_power = np.mean(x ** 2) or 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.default_rng().normal(0, np.sqrt(noise_power), len(x))
    return (x + noise).astype(np.float32)


def inject_missing_samples(
    segment: np.ndarray,
    missing_fraction: float = 0.05,
    method: str = "interpolate",
    random_seed: int = 0,
) -> np.ndarray:
    """
    Simulate missing samples by zeroing out a fraction of the segment and then
    optionally interpolating them.

    Parameters
    ----------
    segment          : 1-D array
    missing_fraction : fraction of samples to drop (0-1)
    method           : 'zero' | 'interpolate' | 'hold'
    random_seed      : reproducibility
    """
    rng = np.random.default_rng(random_seed)
    x = np.asarray(segment, dtype=np.float64).copy()
    n = len(x)
    n_missing = max(1, int(n * missing_fraction))
    missing_idx = rng.choice(n, size=n_missing, replace=False)

    if method == "zero":
        x[missing_idx] = 0.0
    elif method == "hold":
        sorted_idx = np.sort(missing_idx)
        for idx in sorted_idx:
            x[idx] = x[idx - 1] if idx > 0 else 0.0
    else:   # interpolate
        x[missing_idx] = np.nan
        valid = np.where(~np.isnan(x))[0]
        if len(valid) > 1:
            x = np.interp(np.arange(n), valid, x[valid])
        else:
            x = np.nan_to_num(x)

    return x.astype(np.float32)


def apply_load_shift(
    segment: np.ndarray,
    scale_factor: float = 1.3,
    offset: float = 0.02,
) -> np.ndarray:
    """
    Simulate a change in operating condition (load shift) by scaling amplitude
    and adding a small DC offset — a common source of false alarms that load-
    aware models should handle gracefully.
    """
    x = np.asarray(segment, dtype=np.float32)
    return (x * scale_factor + offset).astype(np.float32)
