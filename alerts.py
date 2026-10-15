"""
machina_sense/alerts.py
───────────────────────
Plain-language alert generation with severity scoring.

Each alert is a dataclass that captures:
  - machine_id, timestamp, snapshot identifier
  - anomaly score, fault type, confidence
  - severity tier  (LOW / MEDIUM / HIGH / CRITICAL)
  - human-readable message (English + optional Hindi)
  - the top contributing features (explainability)
  - a recommended action and urgency window

Severity mapping (anomaly_score, confidence)
──────────────────────────────────────────────
  CRITICAL : score ≥ 0.85 and confidence ≥ 0.80
  HIGH     : score ≥ 0.70 or confidence ≥ 0.70
  MEDIUM   : score ≥ 0.55
  LOW      : score < 0.55  (no alert generated, information only)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from machina_sense.features import FEATURE_NAMES


# ── Severity ───────────────────────────────────────────────────────────────────

SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_HIGH     = "HIGH"
SEVERITY_MEDIUM   = "MEDIUM"
SEVERITY_LOW      = "LOW"
SEVERITY_OK       = "OK"

SEVERITY_COLORS = {
    SEVERITY_CRITICAL: "#d32f2f",
    SEVERITY_HIGH:     "#f57c00",
    SEVERITY_MEDIUM:   "#fbc02d",
    SEVERITY_LOW:      "#388e3c",
    SEVERITY_OK:       "#1565c0",
}


def compute_severity(anomaly_score: float, confidence: float) -> str:
    if anomaly_score >= 0.85 and confidence >= 0.80:
        return SEVERITY_CRITICAL
    if anomaly_score >= 0.70 or confidence >= 0.70:
        return SEVERITY_HIGH
    if anomaly_score >= 0.55:
        return SEVERITY_MEDIUM
    return SEVERITY_OK


# ── Fault knowledge base ───────────────────────────────────────────────────────

_FAULT_KNOWLEDGE = {
    "inner": {
        "mechanism": "Inner-race spalling",
        "typical_indicator": "impulses at ball-pass inner frequency (≈ 162 Hz @ 1797 rpm)",
        "action_en": "Schedule bearing inspection. Inner-race faults progress rapidly.",
        "action_hi": "बेयरिंग की जांच का समय निर्धारित करें। इनर-रेस दोष तेजी से बढ़ते हैं।",
        "urgency_days_medium": 7,
        "urgency_days_high": 3,
        "urgency_days_critical": 1,
    },
    "outer": {
        "mechanism": "Outer-race spalling",
        "typical_indicator": "impulses at ball-pass outer frequency (≈ 107 Hz @ 1797 rpm)",
        "action_en": "Plan bearing replacement at next maintenance window.",
        "action_hi": "अगले रखरखाव अवसर पर बेयरिंग बदलने की योजना बनाएं।",
        "urgency_days_medium": 14,
        "urgency_days_high": 5,
        "urgency_days_critical": 2,
    },
    "ball": {
        "mechanism": "Rolling-element (ball) defect",
        "typical_indicator": "impulses at ball-spin frequency (≈ 68 Hz @ 1797 rpm) and sidebands",
        "action_en": "Monitor closely; arrange lubrication check and bearing replacement within the week.",
        "action_hi": "ध्यान से निगरानी करें; इस सप्ताह के भीतर स्नेहन जांच और बेयरिंग प्रतिस्थापन की व्यवस्था करें।",
        "urgency_days_medium": 10,
        "urgency_days_high": 4,
        "urgency_days_critical": 1,
    },
    "normal": {
        "mechanism": "None",
        "typical_indicator": "All signals within normal operating range",
        "action_en": "No action required.",
        "action_hi": "कोई कार्रवाई आवश्यक नहीं।",
        "urgency_days_medium": 999,
        "urgency_days_high": 999,
        "urgency_days_critical": 999,
    },
    "unknown": {
        "mechanism": "Unclassified anomaly",
        "typical_indicator": "Statistically significant deviation from normal baseline",
        "action_en": "Investigate equipment; review sensor readings and check for changed operating conditions.",
        "action_hi": "उपकरण की जांच करें; सेंसर रीडिंग की समीक्षा करें और परिचालन स्थितियों में बदलाव की जांच करें।",
        "urgency_days_medium": 7,
        "urgency_days_high": 3,
        "urgency_days_critical": 1,
    },
}


def _urgency_window(fault_label: str, severity: str) -> Optional[int]:
    kb = _FAULT_KNOWLEDGE.get(fault_label, _FAULT_KNOWLEDGE["unknown"])
    if severity == SEVERITY_CRITICAL:
        return kb["urgency_days_critical"]
    if severity == SEVERITY_HIGH:
        return kb["urgency_days_high"]
    if severity == SEVERITY_MEDIUM:
        return kb["urgency_days_medium"]
    return None


# ── Top contributing features (explainability) ────────────────────────────────

def top_features(
    feature_vector: np.ndarray,
    baseline_mean: Optional[np.ndarray] = None,
    baseline_std: Optional[np.ndarray] = None,
    top_n: int = 3,
) -> List[Dict]:
    """
    Return the top-N features that deviate most from the normal baseline.
    If no baseline is provided, returns the top-N by absolute z-score
    assuming N(0,1) for the standardised feature.
    """
    feat = np.asarray(feature_vector, dtype=np.float64)
    if baseline_mean is not None and baseline_std is not None:
        z = np.abs((feat - baseline_mean) / (baseline_std + 1e-9))
    else:
        z = np.abs(feat)

    top_idx = np.argsort(z)[::-1][:top_n]
    return [
        {
            "name":  FEATURE_NAMES[i],
            "value": float(feat[i]),
            "z_score": float(z[i]),
        }
        for i in top_idx
    ]


# ── Alert dataclass ────────────────────────────────────────────────────────────

@dataclass
class MaintenanceAlert:
    machine_id:      str
    timestamp:       str
    snapshot_id:     str
    anomaly_score:   float
    fault_label:     str
    confidence:      float
    severity:        str
    message_en:      str
    message_hi:      str
    action_en:       str
    action_hi:       str
    urgency_days:    Optional[int]
    top_features:    List[Dict] = field(default_factory=list)
    load_hp:         int = 0
    elapsed_min:     Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"[{self.severity}] {self.machine_id} — {self.timestamp}",
            f"  Fault : {self.fault_label} (confidence {self.confidence*100:.0f}%)",
            f"  Score : {self.anomaly_score:.3f}",
            f"  Action: {self.action_en}",
        ]
        if self.urgency_days is not None and self.urgency_days < 30:
            lines.append(f"  Inspect within ~{self.urgency_days} day(s).")
        if self.top_features:
            lines.append("  Key evidence:")
            for f_ in self.top_features:
                lines.append(f"    • {f_['name']} = {f_['value']:.4f}  (z={f_['z_score']:.1f})")
        lines.append(f"  [हिन्दी] {self.message_hi}")
        return "\n".join(lines)


# ── Alert factory ──────────────────────────────────────────────────────────────

class AlertEngine:
    """
    Converts detector output into MaintenanceAlert objects.

    Maintains a running baseline (mean/std) of normal feature vectors so
    that explainability z-scores are meaningful over time.
    """

    def __init__(self, machine_id: str = "MACHINE-001"):
        self.machine_id = machine_id
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_std:  Optional[np.ndarray] = None

    def set_baseline(
        self,
        X_normal: np.ndarray,
    ) -> None:
        """Pre-compute baseline statistics from normal feature vectors."""
        self._baseline_mean = X_normal.mean(axis=0)
        self._baseline_std  = X_normal.std(axis=0) + 1e-9

    def generate_alert(
        self,
        anomaly_score: float,
        fault_label: str,
        confidence: float,
        feature_vector: np.ndarray,
        load_hp: int = 0,
        snapshot_id: str = "",
        elapsed_min: Optional[float] = None,
        timestamp: Optional[str] = None,
    ) -> Optional[MaintenanceAlert]:
        """
        Generate an alert.  Returns None if severity is OK (no alert needed).
        """
        severity = compute_severity(anomaly_score, confidence)
        if severity == SEVERITY_OK:
            return None

        kb = _FAULT_KNOWLEDGE.get(fault_label, _FAULT_KNOWLEDGE["unknown"])
        urgency = _urgency_window(fault_label, severity)

        conf_pct = int(confidence * 100)
        score_pct = int(anomaly_score * 100)

        # English message
        if fault_label in ("inner", "outer", "ball"):
            msg_en = (
                f"{kb['mechanism']} detected on {self.machine_id}. "
                f"Anomaly score {score_pct}/100 · {kb['typical_indicator']} rising. "
                f"Confidence {conf_pct}%."
            )
            if urgency and urgency < 30:
                msg_en += f" Inspect within ~{urgency} day(s)."
        else:
            msg_en = (
                f"Abnormal vibration on {self.machine_id}. "
                f"Anomaly score {score_pct}/100. "
                f"No specific fault identified — further investigation recommended."
            )

        # Hindi message
        if fault_label in ("inner", "outer", "ball"):
            msg_hi = (
                f"{self.machine_id} पर {kb['mechanism']} का पता चला। "
                f"असामान्यता स्कोर {score_pct}/100, विश्वास {conf_pct}%। "
            )
            if urgency and urgency < 30:
                msg_hi += f"~{urgency} दिन के भीतर जांच करें।"
        else:
            msg_hi = (
                f"{self.machine_id} पर असामान्य कंपन। स्कोर {score_pct}/100। "
                f"आगे की जांच की सिफारिश।"
            )

        tf = top_features(
            feature_vector,
            self._baseline_mean,
            self._baseline_std,
        )

        return MaintenanceAlert(
            machine_id=self.machine_id,
            timestamp=timestamp or datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            snapshot_id=snapshot_id,
            anomaly_score=round(anomaly_score, 4),
            fault_label=fault_label,
            confidence=round(confidence, 4),
            severity=severity,
            message_en=msg_en,
            message_hi=msg_hi,
            action_en=kb["action_en"],
            action_hi=kb["action_hi"],
            urgency_days=urgency,
            top_features=tf,
            load_hp=load_hp,
            elapsed_min=elapsed_min,
        )

    def process_batch(
        self,
        result_df: "pd.DataFrame",
        df_meta: "pd.DataFrame",
        feature_matrix: np.ndarray,
    ) -> List[MaintenanceAlert]:
        """
        Process a batch of detector results and return all alerts.

        Parameters
        ----------
        result_df      : output of EnsembleDetector.score_batch()
        df_meta        : original DataFrame with file_id, load, elapsed_min cols
        feature_matrix : (n, N_FEATURES) feature vectors
        """
        import pandas as pd
        alerts = []
        for i, (_, row) in enumerate(result_df.iterrows()):
            if not row["is_anomaly"]:
                continue
            meta = df_meta.iloc[i]
            alert = self.generate_alert(
                anomaly_score=float(row["anomaly_score"]),
                fault_label=str(row["fault_label"]),
                confidence=float(row["confidence"]),
                feature_vector=feature_matrix[i],
                load_hp=int(meta.get("load", 0)),
                snapshot_id=str(meta.get("file_id", i)),
                elapsed_min=meta.get("elapsed_min"),
            )
            if alert:
                alerts.append(alert)
        return alerts
