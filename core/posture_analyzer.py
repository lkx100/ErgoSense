"""
Posture Analyzer Module for ErgoSense.

This module compares live posture metrics against a previously established
baseline and configurable thresholds to determine whether the user’s posture
requires attention.

Expected Inputs:
    - metrics: A dict generated per frame by the core processing pipeline
        {
            "ear_shoulder_distance": float,
            "shoulder_nose_distance": float,
            "face_size": float,
            ... (others ignored)
        }
    - baseline: A dict produced by the calibration module with the same keys.
    - thresholds: A dict (typically from config.defaults.POSTURE_THRESHOLDS)
        {
            "neck_forward_threshold": 1.3,          # ratio > threshold => neck forward
            "shoulder_raised_threshold": 0.8,       # ratio < threshold => shoulders raised
            "face_too_close_threshold": 1.2,        # ratio > threshold => face too close
        }

Computation:
    We only operate on *ratios* (current / baseline). If baseline is missing or
    a division is not possible, the condition is skipped.

Result:
    A PostureResult dataclass encapsulating:
        - individual boolean flags
        - ratios used
        - overall status ("OK", "NeedsAttention", or "Calibrating")

Usage:
    from config.defaults import POSTURE_THRESHOLDS
    analyzer = PostureAnalyzer(POSTURE_THRESHOLDS)
    result = analyzer.analyze(metrics, baseline)

Extensibility:
    - Add more metric comparisons by extending _METRIC_RULES.
    - Each rule defines how to derive a ratio and what comparison operator and threshold to apply.

Author: ErgoSense Engineering
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any


@dataclass
class PostureResult:
    """Structured outcome of posture analysis."""
    neck_forward: bool
    shoulder_raised: bool
    face_too_close: bool
    ratios: Dict[str, float] = field(default_factory=dict)
    overall: str = "Calibrating"  # "Calibrating" | "OK" | "NeedsAttention"
    # Raw thresholds used (helpful for UI display or debugging)
    thresholds: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Flatten the result for UI or logging."""
        return {
            "neck_forward": self.neck_forward,
            "shoulder_raised": self.shoulder_raised,
            "face_too_close": self.face_too_close,
            "overall": self.overall,
            "thresholds": self.thresholds,
            **{f"ratio_{k}": v for k, v in self.ratios.items()},
        }


class PostureAnalyzer:
    """
    Core analyzer comparing current metrics vs baseline with configurable thresholds.

    Threshold Semantics (given ratio = current / baseline):
        - neck_forward: ratio > neck_forward_threshold  => flag True
        - shoulder_raised: ratio < shoulder_raised_threshold => flag True
        - face_too_close: ratio > face_too_close_threshold => flag True
    """

    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = dict(thresholds) if thresholds else {}

        # Internal rule specification:
        # Each entry: (
        #   metric_key,                    # key in both metrics & baseline
        #   result_flag_name,              # attribute in PostureResult
        #   threshold_config_key,          # key in thresholds dict
        #   comparison_fn,                 # fn(ratio, threshold) -> bool
        #   direction                      # ">" or "<" for debug hints
        # )
        self._rules = [
            (
                "ear_shoulder_distance",
                "neck_forward",
                "neck_forward_threshold",
                lambda r, th: r is not None and r > th,
                ">"
            ),
            (
                "shoulder_nose_distance",
                "shoulder_raised",
                "shoulder_raised_threshold",
                lambda r, th: r is not None and r < th,
                "<"
            ),
            (
                "face_size",
                "face_too_close",
                "face_too_close_threshold",
                lambda r, th: r is not None and r > th,
                ">"
            ),
        ]

    def analyze(
        self,
        metrics: Dict[str, float],
        baseline: Optional[Dict[str, float]],
    ) -> PostureResult:
        """
        Perform posture analysis. If baseline is missing or incomplete, returns
        a "Calibrating" result with all flags False.

        Args:
            metrics: Current frame metrics.
            baseline: Baseline metrics (must contain keys for ratios).
        """
        if not baseline or not isinstance(baseline, dict):
            return PostureResult(
                neck_forward=False,
                shoulder_raised=False,
                face_too_close=False,
                ratios={},
                overall="Calibrating",
                thresholds=self.thresholds.copy(),
            )

        ratios: Dict[str, float] = {}
        flags: Dict[str, bool] = {
            "neck_forward": False,
            "shoulder_raised": False,
            "face_too_close": False,
        }

        # Compute ratios & evaluate rules
        for metric_key, flag_name, th_key, cmp_fn, _direction in self._rules:
            current_val = metrics.get(metric_key)
            base_val = baseline.get(metric_key)
            threshold_val = self.thresholds.get(th_key)

            ratio_val = self._safe_ratio(current_val, base_val)
            if ratio_val is not None:
                ratios[metric_key] = ratio_val

            if threshold_val is None:
                # If threshold missing, skip rule (do not raise errors)
                continue

            flag_state = cmp_fn(ratio_val, threshold_val)
            flags[flag_name] = bool(flag_state)

        # Determine overall
        if any(flags.values()):
            overall = "NeedsAttention"
        else:
            overall = "OK"

        return PostureResult(
            neck_forward=flags["neck_forward"],
            shoulder_raised=flags["shoulder_raised"],
            face_too_close=flags["face_too_close"],
            ratios=ratios,
            overall=overall,
            thresholds=self.thresholds.copy(),
        )

    # --------------- Helpers ---------------

    @staticmethod
    def _safe_ratio(current: Any, baseline: Any) -> Optional[float]:
        """Compute a safe ratio current / baseline if both numeric and baseline > 0."""
        if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
            return None
        if baseline == 0:
            return None
        return float(current) / float(baseline)
