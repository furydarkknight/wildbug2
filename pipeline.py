"""
machina_sense/pipeline.py
─────────────────────────
End-to-end MachinaSense pipeline.

Usage
─────
  # Full run with CWRU (real data via cwru package) + synthetic IMS
  python -m machina_sense.pipeline

  # Use only synthetic data (no internet required)
  python -m machina_sense.pipeline --synthetic

  # With real IMS data
  python -m machina_sense.pipeline --ims-dir /path/to/IMS_extracted

  # Save plots
  python -m machina_sense.pipeline --synthetic --save-plots

Steps performed
───────────────
1. Load data   (CWRU real or synthetic fallback)
2. Extract 20 signal features per segment
3. Train/test split (80/20, stratified by label+load)
4. Train EnsembleDetector (Isolation Forest per load + Random Forest classifier)
5. Evaluate: classification metrics, robustness under noise/missing/load-shift
6. (Optional) IMS lead-time evaluation
7. Generate plain-language alerts for test anomalies
8. Print summary report; optionally save plots to ./output/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from machina_sense.data_loader import (
    load_cwru,
    load_ims,
    make_synthetic_cwru,
    make_synthetic_ims,
)
from machina_sense.features import extract_features_df
from machina_sense.detector import train_ensemble
from machina_sense.evaluator import run_full_evaluation
from machina_sense.alerts import AlertEngine
from machina_sense.visualize import save_all, plot_signal_and_spectrum


# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    use_synthetic: bool = False,
    ims_dir: Optional[str] = None,
    save_plots: bool = False,
    output_dir: str = "output",
    random_seed: int = 42,
) -> dict:

    t0 = time.time()
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║            MachinaSense — Intelligent Fault Detection        ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    print("▶  Step 1 / 7 — Loading data …")
    if use_synthetic:
        print("   (Using synthetic CWRU-like data — no internet required)")
        df_cwru = make_synthetic_cwru(n_normal=400, n_fault=360, random_seed=random_seed)
    else:
        print("   Loading CWRU bearing dataset via pip package …")
        try:
            df_cwru = load_cwru(max_segments_per_config=150)
        except Exception as exc:
            print(f"   ⚠  CWRU load failed ({exc}); falling back to synthetic data.")
            df_cwru = make_synthetic_cwru(n_normal=400, n_fault=360, random_seed=random_seed)

    # IMS / run-to-failure data
    df_ims = None
    if ims_dir:
        try:
            df_ims = load_ims(ims_dir, test_set=1)
            print(f"   Loaded {len(df_ims)} IMS snapshots from {ims_dir}")
        except Exception as exc:
            print(f"   ⚠  IMS load failed ({exc}); using synthetic run-to-failure sequence.")
            df_ims = make_synthetic_ims(n_snapshots=300)
    else:
        print("   No --ims-dir supplied; generating synthetic run-to-failure sequence.")
        df_ims = make_synthetic_ims(n_snapshots=300)

    print(f"   CWRU segments : {len(df_cwru)}  (labels: {dict(df_cwru['label'].value_counts())})")
    print(f"   IMS snapshots : {len(df_ims)}")

    # ── Step 2: Feature extraction ─────────────────────────────────────────────
    print("\n▶  Step 2 / 7 — Extracting signal features …")
    X_cwru = extract_features_df(df_cwru)
    y_cwru = df_cwru["label"].values
    loads_cwru = df_cwru["load"].values.astype(int)
    print(f"   Feature matrix shape: {X_cwru.shape}  (20 features per segment)")

    X_ims = extract_features_df(df_ims)
    print(f"   IMS feature matrix shape: {X_ims.shape}")

    # ── Step 3: Train / test split ─────────────────────────────────────────────
    print("\n▶  Step 3 / 7 — Train / test split (80 / 20, stratified) …")
    strat_key = [f"{l}_{ld}" for l, ld in zip(y_cwru, loads_cwru)]
    (X_train, X_test,
     y_train, y_test,
     loads_train, loads_test,
     idx_train, idx_test) = train_test_split(
        X_cwru, y_cwru, loads_cwru, np.arange(len(X_cwru)),
        test_size=0.20, stratify=strat_key, random_state=random_seed,
    )
    df_test_meta = df_cwru.iloc[idx_test].reset_index(drop=True)
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"   Test label distribution: {dict(pd.Series(y_test).value_counts())}")

    # ── Step 4: Train models ───────────────────────────────────────────────────
    print("\n▶  Step 4 / 7 — Training detector models …")
    t_train = time.time()
    detector = train_ensemble(X_train, y_train, loads_train, random_state=random_seed)
    print(f"   Training complete in {time.time() - t_train:.1f}s")
    print(f"   Load-aware models trained for loads: {detector.anomaly_detector._load_levels}")
    print(f"   Fault classifier classes: {detector.fault_classifier.classes_}")

    # ── Step 5: Evaluate ───────────────────────────────────────────────────────
    print("\n▶  Step 5 / 7 — Running evaluation suite …")
    failure_idx = len(df_ims) - 1   # last snapshot = failure for synthetic IMS
    results = run_full_evaluation(
        detector,
        df_test=df_test_meta,
        X_test=X_test,
        y_true=y_test,
        loads_test=loads_test,
        df_ims=df_ims,
        X_ims=X_ims,
        failure_snapshot_idx=failure_idx,
    )

    # ── Step 6: Generate alerts ────────────────────────────────────────────────
    print("\n▶  Step 6 / 7 — Generating maintenance alerts …")
    # Set baseline from training normal data
    normal_mask = y_train == "normal"
    engine = AlertEngine(machine_id="BEARING-A1")
    engine.set_baseline(X_train[normal_mask])

    result_df = detector.score_batch(X_ims, np.zeros(len(X_ims), dtype=int))
    alerts = engine.process_batch(result_df, df_ims, X_ims)
    print(f"   Generated {len(alerts)} alert(s) on IMS timeline")

    # Also show sample alerts on test set
    result_cwru = detector.score_batch(X_test, loads_test)
    alerts_cwru = engine.process_batch(result_cwru, df_test_meta, X_test)
    print(f"   Generated {len(alerts_cwru)} alert(s) on CWRU test set")

    # ── Step 7: Print report ───────────────────────────────────────────────────
    print("\n▶  Step 7 / 7 — Summary report")
    _print_report(results, alerts, alerts_cwru)

    # ── Optional: save plots ───────────────────────────────────────────────────
    if save_plots:
        print(f"\n▶  Saving plots to ./{output_dir}/ …")
        timeline = results.get("lead_time", {}).get("alert_timeline")
        save_all(results, df_test_meta, timeline, alerts, output_dir=output_dir)

        # Save one example signal plot per fault type
        for label in ["normal", "inner", "outer", "ball"]:
            mask = df_cwru["label"] == label
            if mask.any():
                seg = np.asarray(df_cwru[mask].iloc[0]["signal"])
                plot_signal_and_spectrum(
                    seg, label=label,
                    save_path=f"{output_dir}/signal_{label}.png",
                )
        print(f"  ✓ Signal plots saved")

    print(f"\n✅  Pipeline complete in {time.time() - t0:.1f}s\n")
    return results


# ── Report printer ─────────────────────────────────────────────────────────────

def _print_report(results: dict, ims_alerts: list, cwru_alerts: list) -> None:
    sep = "─" * 64

    # Classification
    clf = results.get("classification", {})
    print(f"\n{sep}")
    print("  CLASSIFICATION PERFORMANCE (held-out CWRU test set)")
    print(sep)
    print(f"  Macro F1          : {clf.get('f1_macro', 0):.4f}")
    print(f"  Weighted F1       : {clf.get('f1_weighted', 0):.4f}")
    print(f"  Macro Precision   : {clf.get('precision_macro', 0):.4f}")
    print(f"  Macro Recall      : {clf.get('recall_macro', 0):.4f}")
    print(f"  Anomaly F1 (binary) : {clf.get('anomaly_f1', 0):.4f}")
    print(f"  Anomaly Precision   : {clf.get('anomaly_precision', 0):.4f}")
    print(f"  Anomaly Recall      : {clf.get('anomaly_recall', 0):.4f}")
    print(f"\n{clf.get('classification_report', '')}")

    # Robustness
    rob = results.get("robustness", {})
    print(f"\n{sep}")
    print("  ROBUSTNESS (binary F1 under corruption)")
    print(sep)
    for scenario, r in rob.items():
        bar = "█" * int(r["binary_f1"] * 20)
        print(f"  {scenario:<30s} F1={r['binary_f1']:.3f}  {bar}")

    # Lead time
    lt = results.get("lead_time", {})
    if lt:
        print(f"\n{sep}")
        print("  EARLY-WARNING LEAD TIME (IMS run-to-failure)")
        print(sep)
        if lt.get("lead_time_min") is not None:
            print(f"  First alert at    : {lt['first_alert_min']} min elapsed")
            print(f"  Failure at        : {lt['failure_min']} min elapsed")
            print(f"  Lead time         : {lt['lead_time_min']} min  ({lt['lead_time_min']/60:.1f} h)")
            print(f"  Snapshots warning : {lt['lead_time_snapshots']} snapshots before failure")
        print(f"  Note: {lt.get('note', '')}")

    # Sample alerts
    print(f"\n{sep}")
    print("  SAMPLE MAINTENANCE ALERTS (CWRU test anomalies, first 3)")
    print(sep)
    for alert in cwru_alerts[:3]:
        print(alert.summary())
        print()


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MachinaSense — Industrial Fault Detection Pipeline"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data instead of downloading CWRU (no internet needed)"
    )
    parser.add_argument(
        "--ims-dir", type=str, default=None,
        help="Path to extracted IMS dataset (1st_test/, 2nd_test/, 3rd_test/)"
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save matplotlib plots to ./output/"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory for saved plots (default: output)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    run_pipeline(
        use_synthetic=args.synthetic,
        ims_dir=args.ims_dir,
        save_plots=args.save_plots,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
