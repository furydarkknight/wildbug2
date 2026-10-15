"""
machina_sense/visualize.py
──────────────────────────
Matplotlib visualizations for the MachinaSense demo.

Plots produced
──────────────
1. plot_signal_and_spectrum()  – raw waveform + FFT for one segment
2. plot_alert_dashboard()      – anomaly score timeline with alert markers
3. plot_confusion_matrix()     – heatmap of classification results
4. plot_robustness_bars()      – F1 score under different corruptions
5. plot_lead_time_timeline()   – IMS anomaly score + alert onset + failure point
6. save_all()                  – convenience wrapper that writes PNGs to a folder
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; swap to "TkAgg" for live display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from machina_sense.alerts import MaintenanceAlert, SEVERITY_COLORS


# ── 1. Signal + Spectrum ───────────────────────────────────────────────────────

def plot_signal_and_spectrum(
    segment: np.ndarray,
    label: str = "",
    fs: float = 12_000.0,
    title: str = "Vibration Signal",
    save_path: Optional[str] = None,
) -> plt.Figure:
    t = np.linspace(0, len(segment) / fs, len(segment), endpoint=False) * 1000  # ms
    n = len(segment)
    fft_amp = np.abs(np.fft.rfft(segment * np.hanning(n)))
    freqs   = np.fft.rfftfreq(n, d=1.0 / fs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    fig.suptitle(f"{title}  [{label}]", fontsize=11, fontweight="bold")

    axes[0].plot(t, segment, lw=0.7, color="#1976D2")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude (g)")
    axes[0].set_title("Time-domain waveform")
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(freqs / 1000, fft_amp + 1e-9, lw=0.8, color="#D32F2F")
    axes[1].set_xlabel("Frequency (kHz)")
    axes[1].set_ylabel("Amplitude (log)")
    axes[1].set_title("FFT spectrum")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ── 2. Alert dashboard ─────────────────────────────────────────────────────────

def plot_alert_dashboard(
    timeline_df: pd.DataFrame,
    alerts: List[MaintenanceAlert],
    title: str = "Anomaly Score Timeline",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Parameters
    ----------
    timeline_df  : DataFrame with 'elapsed_min' and 'anomaly_score'
    alerts       : list of MaintenanceAlert objects
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    x = timeline_df["elapsed_min"].values

    if "smoothed_score" in timeline_df.columns:
        ax.plot(x, timeline_df["smoothed_score"].values, lw=1.2,
                color="#1976D2", label="Smoothed score", zorder=3)
    ax.plot(x, timeline_df["anomaly_score"].values, lw=0.6,
            color="#90CAF9", alpha=0.6, label="Raw score", zorder=2)

    ax.axhline(0.55, color="#F57C00", ls="--", lw=1, label="MEDIUM threshold")
    ax.axhline(0.70, color="#D32F2F", ls="--", lw=1, label="HIGH threshold")

    # Mark alert onset
    for alert in alerts:
        if alert.elapsed_min is not None:
            color = SEVERITY_COLORS.get(alert.severity, "#888")
            ax.axvline(alert.elapsed_min, color=color, lw=1.5, alpha=0.7)

    ax.set_xlabel("Elapsed time (min)")
    ax.set_ylabel("Anomaly score")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ── 3. Confusion matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    title: str = "Fault Classification – Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    cm = cm_df.values.astype(float)
    # Row-normalise
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = LinearSegmentedColormap.from_list("blues", ["#EEF2FF", "#1565C0"])
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = list(cm_df.index)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=11)

    for r in range(len(labels)):
        for c in range(len(labels)):
            val = cm_norm[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if val > 0.55 else "black")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ── 4. Robustness bar chart ────────────────────────────────────────────────────

def plot_robustness_bars(
    robustness_results: Dict,
    title: str = "Detection F1 Under Corruption Scenarios",
    save_path: Optional[str] = None,
) -> plt.Figure:
    scenarios = list(robustness_results.keys())
    f1_scores  = [robustness_results[s]["binary_f1"] for s in scenarios]
    descriptions = [robustness_results[s]["description"] for s in scenarios]

    # Color-code by F1 level
    colors = ["#D32F2F" if f < 0.6 else "#F57C00" if f < 0.8 else "#388E3C"
              for f in f1_scores]

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.barh(scenarios, f1_scores, color=colors, height=0.55, edgecolor="white")
    ax.set_xlim(0, 1.0)
    ax.axvline(0.8, color="#555", ls="--", lw=1, label="Target F1 = 0.80")
    ax.set_xlabel("Binary F1 (fault vs. normal)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    for bar, f1 in zip(bars, f1_scores):
        ax.text(f1 + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{f1:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ── 5. Lead-time timeline ──────────────────────────────────────────────────────

def plot_lead_time_timeline(
    lead_time_results: Dict,
    title: str = "Early-Warning Lead Time (IMS Run-to-Failure)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    tl = lead_time_results.get("alert_timeline")
    if tl is None:
        return None

    first_alert_min = lead_time_results.get("first_alert_min")
    failure_min     = lead_time_results.get("failure_min")

    fig, ax = plt.subplots(figsize=(13, 4))
    x = tl["elapsed_min"].values

    if "smoothed_score" in tl.columns:
        ax.fill_between(x, tl["smoothed_score"].values, alpha=0.3, color="#1976D2")
        ax.plot(x, tl["smoothed_score"].values, lw=1.2, color="#1976D2", label="Anomaly score")
    else:
        ax.fill_between(x, tl["anomaly_score"].values, alpha=0.3, color="#1976D2")
        ax.plot(x, tl["anomaly_score"].values, lw=1.2, color="#1976D2", label="Anomaly score")

    ax.axhline(0.55, color="#F57C00", ls="--", lw=1, label="Alert threshold")

    if first_alert_min is not None:
        ax.axvline(first_alert_min, color="#F57C00", lw=2,
                   label=f"First alert @ {first_alert_min} min")
        ax.annotate("⚠ ALERT", xy=(first_alert_min, 0.55),
                    xytext=(first_alert_min + max(x)*0.02, 0.65),
                    arrowprops=dict(arrowstyle="->", color="#F57C00"),
                    color="#F57C00", fontsize=9)

    if failure_min is not None:
        ax.axvline(failure_min, color="#D32F2F", lw=2,
                   label=f"Failure @ {failure_min} min")
        ax.annotate("✖ FAILURE", xy=(failure_min, 0.9),
                    xytext=(failure_min - max(x)*0.15, 0.95),
                    arrowprops=dict(arrowstyle="->", color="#D32F2F"),
                    color="#D32F2F", fontsize=9)

    if first_alert_min and failure_min:
        lead = failure_min - first_alert_min
        ax.annotate(
            "", xy=(failure_min, 0.45), xytext=(first_alert_min, 0.45),
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5)
        )
        ax.text((first_alert_min + failure_min) / 2, 0.37,
                f"Lead time: {lead} min ({lead/60:.1f} h)",
                ha="center", fontsize=9, color="#333")

    ax.set_xlabel("Elapsed time (min)")
    ax.set_ylabel("Anomaly score")
    ax.set_ylim(-0.05, 1.10)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ── 6. Convenience wrapper ────────────────────────────────────────────────────

def save_all(
    results: Dict,
    df_test: pd.DataFrame,
    timeline_df: Optional[pd.DataFrame],
    alerts: List[MaintenanceAlert],
    output_dir: str = "output",
) -> None:
    """Save all standard plots to output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if "classification" in results:
        plot_confusion_matrix(
            results["classification"]["confusion_matrix"],
            save_path=f"{output_dir}/confusion_matrix.png",
        )
        print(f"  ✓ {output_dir}/confusion_matrix.png")

    if "robustness" in results:
        plot_robustness_bars(
            results["robustness"],
            save_path=f"{output_dir}/robustness.png",
        )
        print(f"  ✓ {output_dir}/robustness.png")

    if "lead_time" in results and timeline_df is not None:
        plot_lead_time_timeline(
            results["lead_time"],
            save_path=f"{output_dir}/lead_time.png",
        )
        print(f"  ✓ {output_dir}/lead_time.png")

    if timeline_df is not None:
        plot_alert_dashboard(
            timeline_df, alerts,
            save_path=f"{output_dir}/alert_dashboard.png",
        )
        print(f"  ✓ {output_dir}/alert_dashboard.png")

    plt.close("all")
