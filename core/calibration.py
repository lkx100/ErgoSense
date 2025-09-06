"""
Calibration module for ErgoSense.

Purpose:
    Establish a stable baseline of posture-related metrics over a short,
    guided period (e.g., the user sits upright for ~5 seconds). The baseline
    values become reference points for relative posture deviation detection
    (e.g., neck forward ratio, shoulder raised, face too close).

Design Goals:
    - Robust against noisy first frames.
    - Outlier rejection (sigma-based) once a provisional distribution exists.
    - Streaming-friendly: update() is cheap (O(number_of_metrics)).
    - Provide progress feedback based on both elapsed time and sample count.
    - Passive: caller decides when to start() / reset(), and polls state.

Typical Usage (in the Streamlit WebRTC loop or similar):
    cal = CalibrationSession(CalibrationConfig(duration_sec=5.0, min_samples=90))
    cal.start(time.time())   # when user triggers or automatically on first frames
    ...
    for each frame metrics:
        now = time.time()
        cal.update(metrics, now)
        progress = cal.get_progress(now)
        if cal.is_complete(now):
            baseline = cal.get_baseline()

Key Concepts:
    - Welford's algorithm for numerically stable running mean & variance.
    - Early phase: accept all values for a metric until it has >= warmup_min samples.
    - After warmup: reject values beyond (reject_sigma * std) distance from mean.
    - Completion: either time threshold AND min_samples satisfied, or all metrics
      have reached min_samples and completion forced early by caller logic.

Extensibility:
    - Additional metrics can be added by updating metrics_of_interest in the config.
    - Could add median or robust estimators if distribution is strongly skewed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import math


# ----------------------------
# Configuration Data Classes
# ----------------------------

@dataclass
class CalibrationConfig:
    """
    Parameters controlling calibration behavior.

    Attributes:
        duration_sec: Target calibration duration in seconds (time-based progress).
        min_samples: Minimum number of accepted samples required for each metric
                     (sample-based progress).
        warmup_min: Number of initial samples for a metric before outlier rejection begins.
        reject_sigma: Z-score threshold (in standard deviations) for outlier rejection
                      once warmup_min samples have been collected.
        metrics_of_interest: Tuple of metric keys expected in the per-frame metrics dict.
    """
    duration_sec: float = 2.0
    min_samples: int = 40
    warmup_min: int = 5
    reject_sigma: float = 2.5
    metrics_of_interest: tuple[str, ...] = (
        "ear_shoulder_distance",
        "shoulder_nose_distance",
        "face_size",
    )

    def validate(self):
        if self.duration_sec <= 0:
            raise ValueError("duration_sec must be > 0")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be > 0")
        if self.warmup_min < 0:
            raise ValueError("warmup_min must be >= 0")
        if self.reject_sigma <= 0:
            raise ValueError("reject_sigma must be > 0")
        if not self.metrics_of_interest:
            raise ValueError("metrics_of_interest cannot be empty")


@dataclass
class CalibrationState:
    """
    Mutable state tracking calibration progress and statistics.
    Fields:
        started_at: Timestamp when calibration started.
        samples_total: Total frames where at least one metric was accepted.
        accepted_per_metric: Number of accepted samples per metric.
        means: Running means per metric.
        m2: Sum of squares of differences from the running mean (Welford) per metric.
        complete: Flag indicating calibration completion.
    """
    started_at: Optional[float] = None
    samples_total: int = 0
    accepted_per_metric: Dict[str, int] = field(default_factory=dict)
    means: Dict[str, float] = field(default_factory=dict)
    m2: Dict[str, float] = field(default_factory=dict)
    complete: bool = False


# ----------------------------
# Calibration Session
# ----------------------------

class CalibrationSession:
    """
    Handles accumulation of metric samples and computes baseline statistics.

    Public Methods:
        start(now: float) -> None
        reset() -> None
        is_started() -> bool
        update(metrics: Dict[str, float], now: float) -> bool
        is_complete(now: float) -> bool
        force_complete() -> None
        get_progress(now: float) -> float
        get_baseline() -> Optional[Dict[str, float]]
        get_stats() -> Dict[str, Dict[str, float]]

    Implementation Details:
        - update() returns True if at least one metric value was accepted.
        - Uses a combination of time-based and sample-based progress.
        - get_stats() can be used for debugging (means, stddev, counts).
    """

    def __init__(self, config: CalibrationConfig | None = None):
        self.config = config or CalibrationConfig()
        self.config.validate()
        self.state = CalibrationState()

    # ---- Lifecycle Control ----

    def start(self, now: float) -> None:
        """Initialize calibration state starting at timestamp 'now'."""
        self.state = CalibrationState(started_at=now)

    def reset(self) -> None:
        """Reset calibration state; a subsequent start() is required."""
        self.state = CalibrationState()

    def is_started(self) -> bool:
        return self.state.started_at is not None

    def force_complete(self) -> None:
        """Force flag as complete (e.g., user manually ends early)."""
        self.state.complete = True

    # ---- Statistical Updates ----

    def _welford_update(self, key: str, value: float) -> None:
        """
        Update running mean & variance using Welford's algorithm for a given metric.
        """
        s = self.state
        count = s.accepted_per_metric.get(key, 0)
        if count == 0:
            s.means[key] = value
            s.m2[key] = 0.0
            s.accepted_per_metric[key] = 1
            return

        new_count = count + 1
        delta = value - s.means[key]
        new_mean = s.means[key] + delta / new_count
        delta2 = value - new_mean
        s.means[key] = new_mean
        s.m2[key] += delta * delta2
        s.accepted_per_metric[key] = new_count

    def _std(self, key: str) -> float:
        """
        Compute standard deviation for a metric. Returns NaN if insufficient samples.
        """
        n = self.state.accepted_per_metric.get(key, 0)
        if n < 2:
            return float("nan")
        return math.sqrt(self.state.m2[key] / (n - 1))

    # ---- Core Update Logic ----

    def update(self, metrics: Dict[str, float], now: float) -> bool:
        """
        Attempt to incorporate current frame's metrics into calibration statistics.

        Returns:
            True if at least one metric was accepted, False otherwise.

        Outlier Rejection:
            - If a metric has at least warmup_min samples, a new value whose
              absolute z-score exceeds reject_sigma is discarded.

        Completion:
            - Marked complete once duration_sec elapsed AND min_samples reached,
              or if forced externally.
        """
        if not self.is_started() or self.state.complete:
            return False

        accepted_any = False
        for key in self.config.metrics_of_interest:
            value = metrics.get(key)
            if not self._is_usable_number(value):
                continue
            # Static type narrowing for tools
            assert isinstance(value, (int, float))
            value_f = float(value)

            current_n = self.state.accepted_per_metric.get(key, 0)
            if current_n >= self.config.warmup_min:
                mean = self.state.means[key]
                std = self._std(key)
                if std > 1e-12:
                    z = abs((value_f - mean) / std)
                    if z > self.config.reject_sigma:
                        continue  # reject outlier

            self._welford_update(key, value_f)
            accepted_any = True

        if accepted_any:
            self.state.samples_total += 1

        if self._time_elapsed(now) >= self.config.duration_sec and self._all_metrics_satisfied():
            self.state.complete = True

        return accepted_any

    # ---- Completion & Progress ----

    def _all_metrics_satisfied(self) -> bool:
        s = self.state
        for key in self.config.metrics_of_interest:
            if s.accepted_per_metric.get(key, 0) < self.config.min_samples:
                return False
        return True

    def _time_elapsed(self, now: float) -> float:
        start = self.state.started_at
        if start is None:
            return 0.0
        return now - start

    def is_complete(self, now: float) -> bool:
        if self.state.complete:
            return True
        if self._all_metrics_satisfied():
            return True
        return False

    def get_progress(self, now: float) -> float:
        """
        Returns a float in [0, 1] approximating calibration progress.
        Combines time fraction and average sample fraction across metrics.
        """
        if not self.is_started():
            return 0.0
        if self.is_complete(now):
            return 1.0

        t_fraction = min(1.0, self._time_elapsed(now) / self.config.duration_sec)

        sample_fracs = []
        for key in self.config.metrics_of_interest:
            c = self.state.accepted_per_metric.get(key, 0)
            sample_fracs.append(min(1.0, c / self.config.min_samples))
        sample_fraction = sum(sample_fracs) / len(sample_fracs) if sample_fracs else 0.0

        return max(0.0, min(1.0, 0.5 * t_fraction + 0.5 * sample_fraction))

    # ---- Retrieval ----

    def get_baseline(self) -> Optional[Dict[str, float]]:
        """
        Return baseline means if calibration is complete (time+samples) OR
        all metrics have satisfied minimum sample counts (early completion condition).
        """
        if not (self.state.complete or self._all_metrics_satisfied()):
            return None
        return {k: self.state.means[k] for k in self.config.metrics_of_interest if k in self.state.means}

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Diagnostic stats for display or debugging:
        {
            metric: {
                "count": int,
                "mean": float,
                "std": float
            },
            ...
        }
        """
        stats: Dict[str, Dict[str, float]] = {}
        for key in self.config.metrics_of_interest:
            if key in self.state.means:
                stats[key] = {
                    "count": float(self.state.accepted_per_metric.get(key, 0)),
                    "mean": self.state.means[key],
                    "std": self._std(key),
                }
        return stats

    # ---- Helpers ----

    @staticmethod
    def _is_usable_number(x) -> bool:
        # Accept only int/float; guards before math operations for static type checkers
        return isinstance(x, (int, float)) and not math.isnan(x) and math.isfinite(x)


# ----------------------------
# Utility Function (Optional)
# ----------------------------

def summarize_baseline(baseline: Dict[str, float] | None) -> str:
    """
    Human-friendly single-line summary for logging/diagnostics.
    """
    if not baseline:
        return "Baseline: <not ready>"
    parts = [f"{k}={baseline[k]:.4f}" for k in sorted(baseline.keys())]
    return "Baseline: " + ", ".join(parts)
