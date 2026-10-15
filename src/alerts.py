"""
alerts.py
=========
Generates plain-language maintenance alerts from anomaly detector output.

Alert structure
---------------
Each alert contains:
  - timestamp       : when the anomaly was detected
  - severity        : critical / high / medium / low / normal
  - score           : anomaly score 0–1
  - confidence      : how consistently anomalous recent readings are (0–1)
  - fault_hypothesis: likely fault type based on feature fingerprint
  - evidence        : which features most deviate from normal (top 3)
  - message_en      : English plain-language explanation
  - message_hi      : Hindi translation of the alert
  - rul_estimate    : estimated hours to inspection (if RUL model fitted)
  - load_hp         : motor load at time of detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fault hypothesis rules
# ---------------------------------------------------------------------------
# Each rule maps a set of (feature, relative_threshold) conditions to a
# fault type.  The rule fires if > 50 % of its conditions are satisfied.
# This is a transparent, physics-informed rule engine.

_FAULT_RULES: list[dict] = [
    {
        "name":        "inner_race",
        "description": "Inner race defect",
        "conditions":  [
            ("kurtosis",     3.5),   # high kurtosis → periodic impulses
            ("env_kurt",     3.0),
            ("band_5x",      2.0),   # energy at ~5× shaft (BPFI proxy)
            ("crest_factor", 4.0),
        ],
        "inspect_msg_en": (
            "Bearing vibration shows impulse patterns consistent with inner-race "
            "wear.  Inspect the drive-end bearing."
        ),
        "inspect_msg_hi": (
            "बेयरिंग में कंपन के आवेग पैटर्न इनर-रेस घिसाव का संकेत देते हैं। "
            "ड्राइव-एंड बेयरिंग की जाँच करें।"
        ),
    },
    {
        "name":        "outer_race",
        "description": "Outer race defect",
        "conditions":  [
            ("kurtosis",     3.0),
            ("env_kurt",     2.5),
            ("band_4x",      2.0),   # energy at ~3.6× shaft (BPFO proxy)
            ("thd",          1.8),
        ],
        "inspect_msg_en": (
            "Elevated harmonic content and envelope kurtosis suggest outer-race "
            "damage.  Schedule bearing replacement within the next inspection cycle."
        ),
        "inspect_msg_hi": (
            "हार्मोनिक सामग्री और एनवलप कर्टोसिस में वृद्धि आउटर-रेस क्षति का सुझाव देती है। "
            "अगले निरीक्षण चक्र में बेयरिंग बदलने की योजना बनाएं।"
        ),
    },
    {
        "name":        "ball_fault",
        "description": "Rolling element (ball) defect",
        "conditions":  [
            ("kurtosis",         2.5),
            ("crest_factor",     3.5),
            ("spectral_entropy", 1.5),  # increased spectral spread
            ("band_2x",          2.0),
        ],
        "inspect_msg_en": (
            "Broadened spectrum and elevated crest factor indicate rolling-element "
            "fatigue.  Lubrication check recommended; replace if kurtosis persists."
        ),
        "inspect_msg_hi": (
            "चौड़ा स्पेक्ट्रम और बढ़ा हुआ क्रेस्ट फैक्टर रोलिंग-एलिमेंट थकान का संकेत देते हैं। "
            "स्नेहन जाँच अनुशंसित है; यदि कर्टोसिस बना रहे तो बदलें।"
        ),
    },
    {
        "name":        "imbalance",
        "description": "Rotor imbalance or misalignment",
        "conditions":  [
            ("rms",             1.5),   # elevated overall vibration
            ("dominant_freq",   1.0),   # dominated by 1× shaft
            ("thd",             0.5),   # low THD (energy at 1×, not harmonics)
            ("shape_factor",    1.2),
        ],
        "inspect_msg_en": (
            "High 1× vibration without significant harmonics suggests rotor imbalance "
            "or shaft misalignment.  Check coupling and balance."
        ),
        "inspect_msg_hi": (
            "महत्वपूर्ण हार्मोनिक्स के बिना उच्च 1× कंपन रोटर असंतुलन या शाफ्ट गलत संरेखण का "
            "सुझाव देता है।  कपलिंग और बैलेंस की जाँच करें।"
        ),
    },
]

_GENERIC_MSG_EN = (
    "Abnormal operating pattern detected.  Sensor readings deviate "
    "significantly from the baseline learned on this machine."
)
_GENERIC_MSG_HI = (
    "असामान्य ऑपरेटिंग पैटर्न का पता चला।  सेंसर रीडिंग इस मशीन पर "
    "सीखे गए आधार रेखा से काफी भिन्न हैं।"
)


# ---------------------------------------------------------------------------
# Remaining useful life estimate
# ---------------------------------------------------------------------------

def estimate_rul(score_history: list[float],
                 interval_seconds: float = 600.0) -> Optional[float]:
    """
    Simple linear extrapolation of anomaly score trend to estimate
    time until the score reaches the 'critical' threshold (0.80).

    Returns estimated hours until inspection, or None if trend is flat.
    """
    if len(score_history) < 5:
        return None
    x = np.arange(len(score_history), dtype=np.float32)
    y = np.array(score_history, dtype=np.float32)
    slope, intercept = np.polyfit(x, y, 1)
    if slope <= 1e-6:
        return None   # score is flat or improving
    # Steps until score reaches 0.80
    current = y[-1]
    steps_to_critical = (0.80 - current) / slope
    if steps_to_critical <= 0:
        return 0.0
    hours = steps_to_critical * interval_seconds / 3600.0
    return round(float(hours), 1)


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    timestamp:        float
    severity:         str
    score:            float
    confidence:       float
    fault_hypothesis: str
    evidence:         list[str]
    message_en:       str
    message_hi:       str
    rul_hours:        Optional[float]
    load_hp:          int
    features:         dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp":        self.timestamp,
            "severity":         self.severity,
            "score":            round(self.score, 3),
            "confidence_pct":   round(self.confidence * 100, 1),
            "fault_hypothesis": self.fault_hypothesis,
            "evidence":         self.evidence,
            "rul_hours":        self.rul_hours,
            "load_hp":          self.load_hp,
            "message_en":       self.message_en,
            "message_hi":       self.message_hi,
        }

    def format_terminal(self) -> str:
        rul_str = (
            f"~{self.rul_hours:.0f} h to inspection"
            if self.rul_hours is not None
            else "—"
        )
        lines = [
            f"+-- ALERT [{self.severity.upper()}] " + "-" * 48,
            f"|  Time:       {self.timestamp:.1f} s  |  Load: {self.load_hp} hp",
            f"|  Score:      {self.score:.3f}  |  Confidence: {self.confidence*100:.0f}%",
            f"|  Hypothesis: {self.fault_hypothesis}",
            f"|  Evidence:   {', '.join(self.evidence)}",
            f"|  Lead time:  {rul_str}",
            f"|",
            f"|  EN: {self.message_en}",
            f"|",
            f"|  HI: {self.message_hi}",
            f"+" + "-" * 65,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Alert generator
# ---------------------------------------------------------------------------

class AlertGenerator:
    """
    Converts anomaly scores + feature vectors into human-readable alerts.
    """

    def __init__(self, normal_feature_stats: Optional[dict] = None,
                 score_threshold: float = 0.45) -> None:
        """
        Parameters
        ----------
        normal_feature_stats : dict mapping feature_name → (mean, std) from
                               normal training data.  Used to compute
                               per-feature deviation for evidence.
        score_threshold      : Minimum score to generate an alert.
        """
        self.normal_stats   = normal_feature_stats or {}
        self.score_threshold = score_threshold
        self._score_history: list[float] = []
        self._interval_s    = 600.0      # snapshot interval (seconds)

    def set_snapshot_interval(self, seconds: float) -> None:
        self._interval_s = seconds

    def compute_feature_stats(self, X: np.ndarray,
                               feat_names: list[str]) -> None:
        """Compute mean/std from a normal feature matrix for evidence ranking."""
        self.normal_stats = {
            name: (float(X[:, i].mean()), max(float(X[:, i].std()), 1e-6))
            for i, name in enumerate(feat_names)
        }

    def _top_evidence(self, features: dict[str, float], k: int = 3) -> list[str]:
        """Return the k features that deviate most from normal (in σ units)."""
        if not self.normal_stats:
            return list(features.keys())[:k]
        deviations = {}
        for name, val in features.items():
            if name in self.normal_stats:
                mu, std = self.normal_stats[name]
                deviations[name] = abs(val - mu) / std
        top = sorted(deviations, key=deviations.get, reverse=True)[:k]
        return [f"{n} (+{deviations[n]:.1f}σ)" for n in top]

    def _classify_fault(self, features: dict[str, float]) -> tuple[str, str, str, str]:
        """
        Apply rule-based fault classification.
        Returns (name, description, msg_en, msg_hi).
        """
        if not self.normal_stats:
            return ("unknown", "Unknown fault",
                    _GENERIC_MSG_EN, _GENERIC_MSG_HI)

        best_rule = None
        best_hits = 0

        for rule in _FAULT_RULES:
            hits = 0
            for feat_name, threshold in rule["conditions"]:
                if feat_name not in features or feat_name not in self.normal_stats:
                    continue
                mu, std = self.normal_stats[feat_name]
                val     = features[feat_name]
                # Feature z-score must exceed threshold
                if (val - mu) / std > threshold:
                    hits += 1
            if hits > best_hits:
                best_hits = hits
                best_rule = rule

        # Require at least 2 rule conditions to fire
        if best_rule is None or best_hits < 2:
            return ("unknown", "Unclassified anomaly",
                    _GENERIC_MSG_EN, _GENERIC_MSG_HI)

        return (
            best_rule["name"],
            best_rule["description"],
            best_rule["inspect_msg_en"],
            best_rule["inspect_msg_hi"],
        )

    def generate(self, timestamp: float, score: float, confidence: float,
                 features: dict[str, float], load_hp: int,
                 severity: str) -> Optional[Alert]:
        """
        Generate an alert for a single reading.
        Returns None if score is below threshold or severity is 'normal'.
        """
        self._score_history.append(score)
        if len(self._score_history) > 50:
            self._score_history.pop(0)

        if score < self.score_threshold or severity == "normal":
            return None

        fault_name, fault_desc, msg_en, msg_hi = self._classify_fault(features)
        evidence = self._top_evidence(features)
        rul_hours = estimate_rul(self._score_history, self._interval_s)

        # Enrich English message with RUL and confidence
        conf_pct = round(confidence * 100)
        if rul_hours is not None:
            msg_en = (
                f"{msg_en}  "
                f"[Estimated time to inspection: ~{rul_hours:.0f} h | "
                f"Confidence: {conf_pct}%]"
            )
        else:
            msg_en = f"{msg_en}  [Confidence: {conf_pct}%]"

        return Alert(
            timestamp=timestamp,
            severity=severity,
            score=score,
            confidence=confidence,
            fault_hypothesis=fault_desc,
            evidence=evidence,
            message_en=msg_en,
            message_hi=msg_hi,
            rul_hours=rul_hours,
            load_hp=load_hp,
            features=features,
        )
