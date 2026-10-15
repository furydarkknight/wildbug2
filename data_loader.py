"""
machina_sense/data_loader.py
────────────────────────────
Loads CWRU bearing data (via the `cwru` pip package) and NASA IMS run-to-failure
data (raw ASCII files).  Both loaders return a unified DataFrame format so the
rest of the pipeline is dataset-agnostic.

Unified DataFrame schema
────────────────────────
  signal      float64  – raw vibration sample
  load        int      – nominal motor load in HP (0-3 for CWRU; 0 for IMS)
  label       str      – 'normal', 'inner', 'outer', 'ball', or 'unknown'
  fault_size  float    – fault diameter in inches (CWRU); NaN for IMS
  source      str      – 'cwru' | 'ims'
  file_id     str      – original file or snapshot identifier
  sample_idx  int      – sample position within the file/snapshot
"""

from __future__ import annotations

import os
import glob
import warnings
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

# ── CWRU ──────────────────────────────────────────────────────────────────────

# Each entry: (cwru_config_string, speed_rpm, segment_length, load_hp, label, fault_size)
CWRU_CONFIGS: List[Tuple[str, str, int, int, str, float]] = [
    # Normal baseline – all four loads
    ("Normal",        "1797", 384, 0, "normal", 0.0),
    ("Normal",        "1772", 384, 1, "normal", 0.0),
    ("Normal",        "1750", 384, 2, "normal", 0.0),
    ("Normal",        "1730", 384, 3, "normal", 0.0),
    # Inner-race faults – 0.007" at each load
    ("12DriveEndFault", "1797", 384, 0, "inner", 0.007),
    ("12DriveEndFault", "1772", 384, 1, "inner", 0.007),
    ("12DriveEndFault", "1750", 384, 2, "inner", 0.007),
    ("12DriveEndFault", "1730", 384, 3, "inner", 0.007),
    # Outer-race faults – 0.007"
    ("12FanEndFault",   "1797", 384, 0, "outer", 0.007),
    ("12FanEndFault",   "1772", 384, 1, "outer", 0.007),
    ("12FanEndFault",   "1750", 384, 2, "outer", 0.007),
    ("12FanEndFault",   "1730", 384, 3, "outer", 0.007),
    # Ball faults – 0.007"
    ("12DriveEndFault", "1797", 384, 0, "ball",  0.021),
    ("12DriveEndFault", "1772", 384, 1, "ball",  0.021),
    ("12DriveEndFault", "1750", 384, 2, "ball",  0.021),
    ("12DriveEndFault", "1730", 384, 3, "ball",  0.021),
]


def load_cwru(
    configs: Optional[List[Tuple]] = None,
    max_segments_per_config: int = 200,
    segment_length: int = 1024,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Download (or reuse cached) CWRU .mat files via the `cwru` package and return
    a DataFrame of fixed-length segments.

    Parameters
    ----------
    configs               : subset of CWRU_CONFIGS to load; defaults to all
    max_segments_per_config : cap to keep memory manageable
    segment_length        : samples per segment (must be ≥ 256)
    random_seed           : reproducible shuffling
    """
    try:
        import cwru as cwru_pkg  # noqa: F401
    except ImportError:
        raise ImportError("Run:  pip install cwru")

    rng = np.random.default_rng(random_seed)
    if configs is None:
        configs = CWRU_CONFIGS

    rows = []
    for cfg_str, rpm, seg_len_default, load_hp, label, fault_size in configs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import cwru as cwru_pkg
                data = cwru_pkg.CWRU(cfg_str, rpm, segment_length)
            # data.X shape: (n_segments, segment_length, n_channels)
            # channel 0 is drive-end accelerometer
            X = data.X[:, :, 0]
            n_segs = min(len(X), max_segments_per_config)
            idx = rng.choice(len(X), size=n_segs, replace=False)
            for seg_i, seg in enumerate(X[idx]):
                rows.append({
                    "signal":     seg.tolist(),   # store full segment as list
                    "load":       load_hp,
                    "label":      label,
                    "fault_size": fault_size,
                    "source":     "cwru",
                    "file_id":    f"{cfg_str}_{rpm}hp{load_hp}",
                    "sample_idx": int(idx[seg_i]),
                })
        except Exception as exc:
            warnings.warn(f"Skipped CWRU config {cfg_str}/{rpm}/load{load_hp}: {exc}")

    if not rows:
        raise RuntimeError("No CWRU data could be loaded. Check your internet connection.")

    return pd.DataFrame(rows)


# ── NASA IMS ──────────────────────────────────────────────────────────────────

IMS_CHANNEL_NAMES = ["ch1", "ch2", "ch3", "ch4"]   # bearings 1-4
IMS_FS = 20_000   # Hz
IMS_SNAPSHOT_DURATION_S = 1.0
IMS_INTERVAL_MIN = 10   # snapshots every 10 minutes


def load_ims(
    data_dir: str | Path,
    test_set: int = 1,
    bearing_channel: int = 0,
    max_snapshots: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load NASA IMS run-to-failure data from the extracted ZIP directory.

    Directory layout expected
    ─────────────────────────
      data_dir/
        1st_test/   (or 2nd_test, 3rd_test)
          2003.10.22.12.06.24   ← snapshot files (no extension)
          ...

    Parameters
    ----------
    data_dir         : path to the directory that contains `1st_test`, `2nd_test`, `3rd_test`
    test_set         : 1, 2, or 3
    bearing_channel  : which of the 4 channels (0-3) to use
    max_snapshots    : optional cap for quick testing
    """
    subfolder_map = {1: "1st_test", 2: "2nd_test", 3: "3rd_test"}
    subfolder = subfolder_map.get(test_set)
    if subfolder is None:
        raise ValueError("test_set must be 1, 2, or 3")

    folder = Path(data_dir) / subfolder
    if not folder.exists():
        raise FileNotFoundError(
            f"IMS folder not found: {folder}\n"
            "Download from https://data.nasa.gov/docs/legacy/IMS.zip and extract here."
        )

    snapshot_files = sorted(folder.iterdir())
    if max_snapshots:
        snapshot_files = snapshot_files[:max_snapshots]

    rows = []
    for snap_idx, fpath in enumerate(snapshot_files):
        if not fpath.is_file():
            continue
        try:
            arr = np.loadtxt(fpath, delimiter="\t")
            if arr.ndim == 1:
                arr = arr.reshape(-1, 4)
            col = min(bearing_channel, arr.shape[1] - 1)
            signal = arr[:, col].tolist()
            # Elapsed time in minutes
            elapsed_min = snap_idx * IMS_INTERVAL_MIN
            rows.append({
                "signal":        signal,
                "load":          0,
                "label":         "unknown",
                "fault_size":    float("nan"),
                "source":        "ims",
                "file_id":       fpath.name,
                "sample_idx":    snap_idx,
                "elapsed_min":   elapsed_min,
            })
        except Exception as exc:
            warnings.warn(f"Skipped IMS snapshot {fpath.name}: {exc}")

    if not rows:
        raise RuntimeError(f"No IMS snapshots loaded from {folder}")

    df = pd.DataFrame(rows)
    return df


# ── Synthetic fallback (for unit tests / CI without internet) ─────────────────

def make_synthetic_cwru(
    n_normal: int = 400,
    n_fault: int = 300,
    segment_length: int = 1024,
    fs: int = 12_000,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Generate labelled synthetic vibration data that mimics CWRU statistics.
    Used when the real dataset is unavailable (e.g. offline CI).

    Normal signal : broadband noise
    Inner fault   : noise + impulse at ball pass inner frequency (~162 Hz @ 1797 rpm)
    Outer fault   : noise + impulse at ball pass outer frequency (~107 Hz @ 1797 rpm)
    Ball fault    : noise + impulse at ball spin frequency (~68 Hz)
    """
    rng = np.random.default_rng(random_seed)
    t = np.linspace(0, segment_length / fs, segment_length, endpoint=False)

    FAULT_FREQS = {"inner": 162.0, "outer": 107.0, "ball": 68.0}
    LOADS = [0, 1, 2, 3]
    rows = []

    def _normal_segment(load: int) -> np.ndarray:
        amp = 0.05 * (1 + 0.1 * load)
        return (rng.normal(0, amp, segment_length)).astype(np.float32)

    def _fault_segment(label: str, load: int) -> np.ndarray:
        base = _normal_segment(load)
        freq = FAULT_FREQS[label]
        impulse = 0.4 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 50)
        # repeat impulse pattern
        period = int(fs / freq)
        full = base.copy()
        for start in range(0, segment_length - len(impulse), period):
            end = min(start + len(impulse), segment_length)
            full[start:end] += impulse[: end - start]
        return full.astype(np.float32)

    per_label_normal = n_normal // len(LOADS)
    for load in LOADS:
        for i in range(per_label_normal):
            rows.append({
                "signal":     _normal_segment(load).tolist(),
                "load":       load,
                "label":      "normal",
                "fault_size": 0.0,
                "source":     "synthetic",
                "file_id":    f"synthetic_normal_load{load}",
                "sample_idx": i,
            })

    per_label_fault = n_fault // (len(FAULT_FREQS) * len(LOADS))
    for fault_label in FAULT_FREQS:
        for load in LOADS:
            for i in range(per_label_fault):
                rows.append({
                    "signal":     _fault_segment(fault_label, load).tolist(),
                    "load":       load,
                    "label":      fault_label,
                    "fault_size": 0.007,
                    "source":     "synthetic",
                    "file_id":    f"synthetic_{fault_label}_load{load}",
                    "sample_idx": i,
                })

    return pd.DataFrame(rows)


def make_synthetic_ims(
    n_snapshots: int = 300,
    segment_length: int = 20_000,
    degradation_start: float = 0.6,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Generate a synthetic IMS-style run-to-failure sequence.
    Vibration amplitude increases smoothly from degradation_start onwards.
    """
    rng = np.random.default_rng(random_seed)
    rows = []
    for i in range(n_snapshots):
        progress = i / n_snapshots
        if progress < degradation_start:
            amp = 0.05 + 0.02 * rng.standard_normal()
        else:
            # Exponential rise toward failure
            rise = (progress - degradation_start) / (1.0 - degradation_start)
            amp = 0.05 + 0.5 * rise ** 2 + 0.03 * abs(rng.standard_normal())
        signal = (rng.normal(0, amp, segment_length)).astype(np.float32).tolist()
        rows.append({
            "signal":      signal,
            "load":        0,
            "label":       "unknown",
            "fault_size":  float("nan"),
            "source":      "synthetic_ims",
            "file_id":     f"snap_{i:05d}",
            "sample_idx":  i,
            "elapsed_min": i * 10,
        })
    return pd.DataFrame(rows)
