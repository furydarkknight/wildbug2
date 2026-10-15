"""
data_loader.py
==============
Loads bearing vibration data from:
  - CWRU (Case Western Reserve University) bearing dataset
  - NASA IMS run-to-failure bearing dataset
  - Synthetic fallback (when neither dataset is present)

Both loaders return a uniform DataFrame with columns:
  timestamp, signal, load_hp, fault_label, dataset
"""

from __future__ import annotations

import os
import struct
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CWRU Loader
# ---------------------------------------------------------------------------

# Mapping: (fault_type, fault_size_inches, load_hp) -> .mat variable prefix
CWRU_FAULT_MAP = {
    "normal":     ("normal",   0.000, [0, 1, 2, 3]),
    "inner_race": ("ir",       0.007, [0, 1, 2, 3]),
    "outer_race": ("or",       0.007, [0, 1, 2, 3]),
    "ball":       ("ba",       0.007, [0, 1, 2, 3]),
}

CWRU_LABEL_INT = {
    "normal":     0,
    "inner_race": 1,
    "outer_race": 2,
    "ball":       3,
}


def load_cwru_mat(mat_path: str | Path, fault_label: str, load_hp: int,
                  sample_rate: int = 12_000,
                  segment_len: int = 1024) -> pd.DataFrame:
    """
    Load a single CWRU .mat file and segment it into fixed-length windows.

    Parameters
    ----------
    mat_path      : Path to the .mat file.
    fault_label   : One of 'normal', 'inner_race', 'outer_race', 'ball'.
    load_hp       : Motor load in horsepower (0–3).
    sample_rate   : Sampling frequency in Hz (default 12 kHz drive-end).
    segment_len   : Samples per segment window (default 1024 ~= 85 ms).

    Returns
    -------
    DataFrame with columns: timestamp, signal (list), load_hp, fault_label,
                             fault_code, dataset.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"CWRU mat file not found: {mat_path}")

    mat = loadmat(str(mat_path))

    # Find the drive-end vibration key (e.g. X097_DE_time or DE_time)
    de_key = next(
        (k for k in mat if "DE_time" in k and not k.startswith("__")), None
    )
    if de_key is None:
        raise KeyError(f"No DE_time key found in {mat_path}. Keys: {list(mat.keys())}")

    raw: np.ndarray = mat[de_key].flatten().astype(np.float32)
    n_segments = len(raw) // segment_len
    raw = raw[: n_segments * segment_len]
    segments = raw.reshape(n_segments, segment_len)

    dt = 1.0 / sample_rate
    rows = []
    for i, seg in enumerate(segments):
        t_start = i * segment_len * dt
        rows.append({
            "timestamp":   t_start,
            "signal":      seg,
            "load_hp":     load_hp,
            "fault_label": fault_label,
            "fault_code":  CWRU_LABEL_INT[fault_label],
            "dataset":     "cwru",
        })
    return pd.DataFrame(rows)


def load_cwru_directory(data_dir: str | Path) -> pd.DataFrame:
    """
    Scan a directory for CWRU .mat files named with the standard convention,
    e.g. 97.mat (normal @0hp), 105.mat (inner-race @0hp), etc.

    Files are expected to be the standard CWRU release.  Any .mat file found
    is loaded; fault label and load are inferred from the filename.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        log.warning("CWRU data dir not found: %s – using synthetic data.", data_dir)
        return pd.DataFrame()

    # CWRU standard file-number mapping (drive-end, 12 kHz, 0.007" fault)
    cwru_file_map = {
        97: ("normal",     0), 98: ("normal",     1),
        99: ("normal",     2), 100: ("normal",    3),
        105: ("inner_race", 0), 106: ("inner_race", 1),
        107: ("inner_race", 2), 108: ("inner_race", 3),
        118: ("outer_race", 0), 119: ("outer_race", 1),
        130: ("outer_race", 2), 131: ("outer_race", 3),
        122: ("ball",       0), 123: ("ball",       1),
        124: ("ball",       2), 125: ("ball",       3),
    }

    frames = []
    for fnum, (label, load) in cwru_file_map.items():
        path = data_dir / f"{fnum}.mat"
        if path.exists():
            try:
                frames.append(load_cwru_mat(path, label, load))
                log.info("  Loaded %s → %s load=%d", path.name, label, load)
            except Exception as exc:
                log.warning("  Skipping %s: %s", path.name, exc)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# NASA IMS Loader
# ---------------------------------------------------------------------------

def load_ims_directory(data_dir: str | Path,
                       bearing_idx: int = 1,
                       sample_rate: int = 20_000,
                       segment_len: int = 2_048) -> pd.DataFrame:
    """
    Load NASA IMS run-to-failure data (Test 1, 2, or 3 folder).

    Each file is a raw binary or ASCII text snapshot of ~20 480 samples
    per bearing channel.  Files are named as timestamps
    (e.g. 2003.10.22.12.06.24).

    Parameters
    ----------
    data_dir     : Path to the IMS test folder (contains timestamped files).
    bearing_idx  : 1-based index of the bearing channel to extract (1–4).
    sample_rate  : 20 kHz for IMS.
    segment_len  : Samples per returned window.

    Returns
    -------
    DataFrame with columns: timestamp (float seconds from start),
                             signal, load_hp (always 0 for IMS),
                             fault_label, fault_code, dataset.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        log.warning("IMS data dir not found: %s – using synthetic data.", data_dir)
        return pd.DataFrame()

    # Collect snapshot files sorted chronologically
    files = sorted(
        f for f in data_dir.iterdir()
        if f.is_file() and not f.suffix.lower() in {".txt", ".md", ".pdf"}
    )
    if not files:
        return pd.DataFrame()

    rows = []
    t0 = None
    for fpath in files:
        # IMS: 4 bearing channels × N samples, stored as float64 binary
        # or space-separated ASCII (test 1 = binary float64, test 2/3 = ASCII)
        try:
            raw = np.fromfile(str(fpath), dtype=np.float64)
            if raw.size == 0:
                raise ValueError("empty binary")
        except Exception:
            try:
                raw = np.loadtxt(str(fpath))
                if raw.ndim == 2:
                    raw = raw[:, bearing_idx - 1]
                raw = raw.astype(np.float32)
            except Exception as exc2:
                log.debug("  Skip %s: %s", fpath.name, exc2)
                continue

        # Parse timestamp from filename
        try:
            parts = fpath.stem.split(".")
            from datetime import datetime
            dt_obj = datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                              int(parts[3]), int(parts[4]), int(parts[5]))
            ts = dt_obj.timestamp()
        except Exception:
            ts = float(len(rows))

        if t0 is None:
            t0 = ts
        elapsed = ts - t0

        # Reshape raw into segments
        if raw.ndim == 2:
            raw = raw[:, bearing_idx - 1]
        raw = raw.flatten().astype(np.float32)
        n_seg = max(1, len(raw) // segment_len)
        raw = raw[: n_seg * segment_len]
        segs = raw.reshape(n_seg, segment_len)

        for i, seg in enumerate(segs):
            rows.append({
                "timestamp":   elapsed + i * segment_len / sample_rate,
                "signal":      seg,
                "load_hp":     0,
                "fault_label": "unknown",   # labels derived from lead-time analysis
                "fault_code":  -1,
                "dataset":     "ims",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # IMS Test 1: bearing 3 and 4 fail near the end.
    # Label last 10 % of timeline as fault for demonstration.
    if len(df) > 0:
        cutoff = df["timestamp"].quantile(0.90)
        df.loc[df["timestamp"] >= cutoff, "fault_label"] = "bearing_fault"
        df.loc[df["timestamp"] >= cutoff, "fault_code"]  = 1
        df.loc[df["timestamp"] <  cutoff, "fault_label"] = "normal"
        df.loc[df["timestamp"] <  cutoff, "fault_code"]  = 0

    log.info("IMS: loaded %d segments from %d files.", len(df), len(files))
    return df


# ---------------------------------------------------------------------------
# Synthetic Data Generator (fallback / testing)
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

def _make_normal(n: int, load: int, sample_rate: int = 12_000,
                 seg_len: int = 1024) -> np.ndarray:
    """Simulate normal bearing vibration at a given motor load."""
    freq_base = 60.0 + load * 5          # shaft frequency shifts with load
    t = np.arange(seg_len) / sample_rate
    sig  = 0.3 * np.sin(2 * np.pi * freq_base * t)
    sig += 0.1 * np.sin(2 * np.pi * 4 * freq_base * t)   # 4th harmonic
    sig += RNG.normal(0, 0.02, seg_len).astype(np.float32)
    return sig.astype(np.float32)


def _make_fault(fault: str, n: int, load: int,
                sample_rate: int = 12_000, seg_len: int = 1024) -> np.ndarray:
    """Simulate a specific bearing fault signature."""
    freq_base = 60.0 + load * 5
    t = np.arange(seg_len) / sample_rate
    base = _make_normal(n, load, sample_rate, seg_len)
    if fault == "inner_race":
        # Inner race: impulse train at ~5.4× shaft freq
        bpfi = 5.4 * freq_base
        impulse = 0.6 * np.sin(2 * np.pi * bpfi * t)
        return (base + impulse).astype(np.float32)
    if fault == "outer_race":
        bpfo = 3.6 * freq_base
        impulse = 0.5 * np.sin(2 * np.pi * bpfo * t)
        return (base + impulse).astype(np.float32)
    if fault == "ball":
        bsf = 2.4 * freq_base
        impulse = 0.4 * np.sin(2 * np.pi * bsf * t)
        return (base + impulse).astype(np.float32)
    return base


def generate_synthetic_cwru(n_per_class: int = 200) -> pd.DataFrame:
    """
    Generate a synthetic CWRU-like dataset for testing when real data
    is unavailable.  Preserves class balance and load variance.
    """
    rows = []
    for load in range(4):
        for label, code in CWRU_LABEL_INT.items():
            for i in range(n_per_class):
                if label == "normal":
                    sig = _make_normal(1, load)
                else:
                    # Gradually degrade: later samples have stronger fault
                    severity = 0.3 + 0.7 * (i / n_per_class)
                    base = _make_normal(1, load)
                    fault = _make_fault(label, 1, load)
                    sig = (base * (1 - severity) + fault * severity).astype(np.float32)
                rows.append({
                    "timestamp":   float(i),
                    "signal":      sig,
                    "load_hp":     load,
                    "fault_label": label,
                    "fault_code":  code,
                    "dataset":     "synthetic",
                })
    df = pd.DataFrame(rows)
    log.info("Generated synthetic dataset: %d segments, 4 classes × 4 loads.",
             len(df))
    return df


def generate_synthetic_ims(n_snapshots: int = 400) -> pd.DataFrame:
    """
    Simulate an IMS-style run-to-failure timeline for early-warning testing.
    """
    rows = []
    SR = 20_000
    SEG = 2048
    for i in range(n_snapshots):
        progress = i / n_snapshots          # 0 → 1 as bearing degrades
        load = 0
        if progress < 0.7:
            sig = _make_normal(1, load, SR, SEG)
        else:
            # Progressive degradation
            severity = (progress - 0.7) / 0.3
            base  = _make_normal(1, load, SR, SEG)
            fault = _make_fault("inner_race", 1, load, SR, SEG)
            sig   = (base * (1 - severity) + fault * severity).astype(np.float32)
        fault_label = "normal" if progress < 0.90 else "bearing_fault"
        fault_code  = 0 if progress < 0.90 else 1
        rows.append({
            "timestamp":   float(i * 600),   # 10-min intervals (seconds)
            "signal":      sig,
            "load_hp":     load,
            "fault_label": fault_label,
            "fault_code":  fault_code,
            "dataset":     "synthetic_ims",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_dataset(cwru_dir: Optional[str] = None,
                 ims_dir: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CWRU and IMS datasets.  Falls back to synthetic data if directories
    are not provided or empty.

    Returns
    -------
    (cwru_df, ims_df)  – two DataFrames with the unified schema.
    """
    cwru_df = pd.DataFrame()
    ims_df  = pd.DataFrame()

    if cwru_dir:
        cwru_df = load_cwru_directory(cwru_dir)
    if cwru_df.empty:
        log.info("CWRU real data unavailable – generating synthetic data.")
        cwru_df = generate_synthetic_cwru()

    if ims_dir:
        ims_df = load_ims_directory(ims_dir)
    if ims_df.empty:
        log.info("IMS real data unavailable – generating synthetic timeline.")
        ims_df = generate_synthetic_ims()

    return cwru_df, ims_df


# ---------------------------------------------------------------------------
# University of Ottawa Electric Motor Dataset Loader
# ---------------------------------------------------------------------------
# Column layout (5 cols, no index):
#   Accelerometer 1 (m/s^2) | Microphone (V) | Accelerometer 2 (m/s^2) |
#   Accelerometer 3 (m/s^2) | Temperature (Celsius)
# File naming: {COND}_{RUN}_{LOAD}.csv
#   Load suffix: _0 = Unloaded, _1 = Loaded
#   Condition prefixes: H_H = Healthy, B_R = Broken bearing, F_B = Faulty bearing,
#                       K_A / R_M / R_U / S_W / V_U = other fault modes
# ---------------------------------------------------------------------------

OTTAWA_SAMPLE_RATE = 10_000   # Hz (approximate — original dataset uses 10 kHz)
OTTAWA_SEG_LEN     = 1_024    # samples per window (~0.1 s)
# Temperature thresholds (°C) derived from dataset statistics
OTTAWA_TEMP_NORMAL  = 35.0
OTTAWA_TEMP_WARN    = 50.0
OTTAWA_TEMP_HIGH    = 65.0
# Condition label map (prefix → human label and integer code)
OTTAWA_CONDITION_MAP = {
    "H_H": ("healthy",         0),
    "B_R": ("broken_bearing",  1),
    "F_B": ("faulty_bearing",  2),
    "K_A": ("key_way_fault",   3),
    "R_M": ("rotor_misalign",  4),
    "R_U": ("rotor_unbalance", 5),
    "S_W": ("stator_winding",  6),
    "V_U": ("vibration_unbal", 7),
}


def load_ottawa_zip(
    zip_path: str | Path,
    max_files_per_condition: int = 8,
    segment_len: int = OTTAWA_SEG_LEN,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Load the University of Ottawa Electric Motor Dataset from the raw zip.

    Reads CSV files inside the zip under
      2_CSV_Data_Files/1_Unloaded_Condition/ and
      2_CSV_Data_Files/2_Loaded_Condition/.

    Returns a DataFrame with columns:
      signal          – np.ndarray (segment_len,) of Accelerometer 1 (m/s^2)
      temperature     – mean temperature (°C) for the segment
      temp_max        – max temperature (°C) for the segment
      load_condition  – int  0=unloaded, 1=loaded
      fault_label     – str  e.g. 'healthy', 'broken_bearing', …
      fault_code      – int  0=healthy, 1+=fault
      timestamp       – float  seconds from file start
      source          – 'ottawa'
    """
    import zipfile
    import io

    zip_path = Path(zip_path)
    if not zip_path.exists():
        log.warning("Ottawa zip not found: %s – using synthetic fallback.", zip_path)
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    rows: list[dict] = []
    condition_counts: dict[str, int] = {}

    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            # Filter to CSV files in the 2_CSV_Data_Files folder
            csv_entries = [
                n for n in zf.namelist()
                if "2_CSV_Data_Files" in n and n.endswith(".csv")
            ]
            log.info("Ottawa zip: found %d CSV entries.", len(csv_entries))

            for entry in csv_entries:
                fname = Path(entry).stem        # e.g. "H_H_1_0" or "B_R_2_1"
                parts = fname.split("_")

                # Condition code is first two underscore-separated tokens
                # e.g. H_H, B_R, F_B …
                if len(parts) < 4:
                    continue
                cond_key = parts[0] + "_" + parts[1]   # "H_H", "B_R", …
                load_suffix = parts[-1]                  # last token: "0" or "1"
                load_cond = 1 if load_suffix == "1" else 0

                if cond_key not in OTTAWA_CONDITION_MAP:
                    continue

                fault_label, fault_code = OTTAWA_CONDITION_MAP[cond_key]

                # Cap files per condition to keep training manageable
                key = cond_key + "_" + load_suffix
                if condition_counts.get(key, 0) >= max_files_per_condition:
                    continue
                condition_counts[key] = condition_counts.get(key, 0) + 1

                try:
                    raw_bytes = zf.read(entry)
                    # Files have a UTF-8 BOM header row:
                    # "Accelerometer 1 (m/s^2),Microphone (V),Accelerometer 2 (m/s^2),Accelerometer 3 (m/s^2),Temperature (Celsius)"
                    df_raw = pd.read_csv(
                        io.BytesIO(raw_bytes),
                        encoding="utf-8-sig",   # handles BOM
                        low_memory=False,
                    )
                    # Columns: Acc1, Mic, Acc2, Acc3, Temp  (by position after header)
                    if df_raw.shape[1] < 5:
                        continue

                    # Extract Accelerometer 1 (col 0) and Temperature (col 4)
                    acc1 = df_raw.iloc[:, 0].values.astype(np.float32)
                    temp = df_raw.iloc[:, 4].values.astype(np.float32)

                    # Remove NaN / Inf
                    valid = np.isfinite(acc1) & np.isfinite(temp)
                    acc1 = acc1[valid]
                    temp = temp[valid]

                    n_segs = len(acc1) // segment_len
                    if n_segs == 0:
                        continue

                    acc1 = acc1[: n_segs * segment_len].reshape(n_segs, segment_len)
                    temp = temp[: n_segs * segment_len].reshape(n_segs, segment_len)

                    # Subsample segments randomly (cap at 40 per file)
                    n_pick = min(n_segs, 40)
                    chosen = rng.choice(n_segs, size=n_pick, replace=False)

                    dt = 1.0 / OTTAWA_SAMPLE_RATE
                    for seg_i in chosen:
                        rows.append({
                            "signal":         acc1[seg_i],
                            "temperature":    float(np.nanmean(temp[seg_i])),
                            "temp_max":       float(np.nanmax(temp[seg_i])),
                            "load_condition": load_cond,
                            "fault_label":    fault_label,
                            "fault_code":     fault_code,
                            "timestamp":      float(seg_i * segment_len * dt),
                            "source":         "ottawa",
                        })

                except Exception as exc:
                    log.debug("Ottawa: skipping %s — %s", entry, exc)
                    continue

    except zipfile.BadZipFile as exc:
        log.error("Ottawa zip is corrupt or invalid: %s", exc)
        return pd.DataFrame()

    if not rows:
        log.warning("Ottawa loader: no rows extracted — returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info(
        "Ottawa dataset: %d segments loaded | conditions: %s",
        len(df),
        df["fault_label"].value_counts().to_dict(),
    )
    return df


def generate_synthetic_ottawa(n_per_class: int = 120) -> pd.DataFrame:
    """
    Synthetic fallback for Ottawa dataset when the real zip is unavailable.
    Generates temperature-correlated vibration data per fault class.
    """
    rng_syn = np.random.default_rng(99)
    SEG = OTTAWA_SEG_LEN
    SR  = OTTAWA_SAMPLE_RATE
    t   = np.arange(SEG) / SR

    TEMP_MEANS = {
        "healthy":         27.5,
        "broken_bearing":  31.0,
        "faulty_bearing":  33.5,
        "key_way_fault":   35.0,
        "rotor_misalign":  36.5,
        "rotor_unbalance": 34.0,
        "stator_winding":  40.0,
        "vibration_unbal": 33.0,
    }
    rows: list[dict] = []
    for cond_key, (fault_label, fault_code) in OTTAWA_CONDITION_MAP.items():
        for load_cond in (0, 1):
            temp_mu = TEMP_MEANS.get(fault_label, 30.0) + load_cond * 2.5
            for i in range(n_per_class):
                if fault_code == 0:
                    sig = (0.15 * np.sin(2 * np.pi * 80 * t) +
                           rng_syn.normal(0, 0.05, SEG)).astype(np.float32)
                else:
                    base    = 0.15 * np.sin(2 * np.pi * 80 * t)
                    fault_f = 200.0 + fault_code * 40.0
                    amp     = 0.2 + 0.4 * (i / n_per_class)
                    sig     = (base + amp * np.sin(2 * np.pi * fault_f * t) +
                               rng_syn.normal(0, 0.05, SEG)).astype(np.float32)

                temp_val = float(temp_mu + rng_syn.normal(0, 1.5))
                rows.append({
                    "signal":         sig,
                    "temperature":    temp_val,
                    "temp_max":       temp_val + abs(rng_syn.normal(0, 0.8)),
                    "load_condition": load_cond,
                    "fault_label":    fault_label,
                    "fault_code":     fault_code,
                    "timestamp":      float(i),
                    "source":         "synthetic_ottawa",
                })

    df = pd.DataFrame(rows)
    log.info("Generated synthetic Ottawa dataset: %d segments.", len(df))
    return df
