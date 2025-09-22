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
from typing import Dict, Optional, Any
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
        motion_threshold: Maximum allowed change in metrics between frames to consider
                         the user stable (prevents sample rejection during movement).
        motion_pause_frames: Number of consecutive frames with motion before pausing calibration.
        motion_resume_frames: Number of consecutive stable frames before resuming calibration.
    """
    duration_sec: float = 1.5
    min_samples: int = 15    # Reduced from 30
    warmup_min: int = 2      # Reduced from 3
    reject_sigma: float = 3.0  # Slightly more permissive
    metrics_of_interest: tuple[str, ...] = (
        "ear_shoulder_distance",
        "shoulder_nose_distance",
        "face_size",
    )
    motion_threshold: float = 0.05  # 5% change threshold for motion detection
    motion_pause_frames: int = 5    # Frames of motion before pause
    motion_resume_frames: int = 10  # Frames of stability before resume

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
        paused: Whether calibration is currently paused due to motion.
        last_metrics: Previous frame's metrics for motion detection.
        motion_frames: Consecutive frames with detected motion.
        stable_frames: Consecutive frames without detected motion.
    """
    started_at: Optional[float] = None
    samples_total: int = 0
    accepted_per_metric: Dict[str, int] = field(default_factory=dict)
    means: Dict[str, float] = field(default_factory=dict)
    m2: Dict[str, float] = field(default_factory=dict)
    complete: bool = False
    paused: bool = False
    last_metrics: Optional[Dict[str, float]] = None
    motion_frames: int = 0
    stable_frames: int = 0


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

    def _detect_motion(self, current_metrics: Dict[str, float]) -> bool:
        """
        Detect if user is moving by comparing current metrics to previous frame.
        Returns True if motion detected (user is moving).
        """
        if self.state.last_metrics is None:
            self.state.last_metrics = dict(current_metrics)
            return False

        # Calculate relative change for each metric
        max_change = 0.0
        for key in self.config.metrics_of_interest:
            if key in current_metrics and key in self.state.last_metrics:
                prev_val = self.state.last_metrics[key]
                curr_val = current_metrics[key]
                if prev_val > 0:  # Avoid division by zero
                    change = abs(curr_val - prev_val) / prev_val
                    max_change = max(max_change, change)

        # Update last metrics for next comparison
        self.state.last_metrics = dict(current_metrics)

        # Return True if motion exceeds threshold
        return max_change > self.config.motion_threshold

    def _is_user_present(self, metrics: Dict[str, float]) -> bool:
        """
        Check if user is present by verifying we have valid pose metrics.
        Returns True if user is detected and metrics are available.
        """
        # Check if status indicates successful detection
        status = metrics.get("status", "")
        if status in ["no_landmarks", "pose_detector_unavailable", "processing_error"]:
            return False

        # Check if we have at least one valid metric
        for key in self.config.metrics_of_interest:
            if key in metrics and self._is_usable_number(metrics[key]):
                return True

        return False

    def _update_motion_state(self, motion_detected: bool) -> None:
        """
        Update motion tracking state and pause/resume calibration accordingly.
        """
        if motion_detected:
            self.state.motion_frames += 1
            self.state.stable_frames = 0

            # Pause calibration if motion persists
            if self.state.motion_frames >= self.config.motion_pause_frames and not self.state.paused:
                self.state.paused = True
        else:
            self.state.stable_frames += 1
            self.state.motion_frames = 0

            # Resume calibration if stable for enough frames
            if self.state.stable_frames >= self.config.motion_resume_frames and self.state.paused:
                self.state.paused = False

    # ---- Core Update Logic ----

    def update(self, metrics: Dict[str, float], now: float) -> bool:
        """
        Attempt to incorporate current frame's metrics into calibration statistics.

        Returns:
            True if at least one metric was accepted, False otherwise.

        Motion Detection:
            - Detects user movement and pauses calibration during motion to prevent
              sample rejection and speed up calibration when user is stable.

        User Presence:
            - Pauses calibration when user is not detected (no landmarks)
            - Resumes when user returns

        Outlier Rejection:
            - If a metric has at least warmup_min samples, a new value whose
              absolute z-score exceeds reject_sigma is discarded.

        Completion:
            - Marked complete once duration_sec elapsed AND min_samples reached,
              or if forced externally.
        """
        if not self.is_started() or self.state.complete:
            return False

        # Check if user is present (has valid landmarks)
        user_present = self._is_user_present(metrics)

        # If user is not present, pause calibration completely
        if not user_present:
            self.state.paused = True
            return False

        # Check for motion and update pause state
        motion_detected = self._detect_motion(metrics)
        self._update_motion_state(motion_detected)

        # If calibration is paused due to motion, don't process samples
        if self.state.paused:
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

        if self._should_complete(now):
            self.state.complete = True

        return accepted_any

    def _should_complete(self, now: float) -> bool:
        """
        Determine if calibration should complete. More flexible than requiring both time AND samples.
        Completes when:
        - Time elapsed >= duration_sec AND at least some samples collected, OR
        - All metrics have sufficient samples (early completion)
        """
        elapsed = self._time_elapsed(now)

        # Require minimum time to avoid premature completion
        if elapsed < max(1.0, self.config.duration_sec * 0.3):  # At least 30% of duration or 1 second
            return False

        # Complete if time is up and we have samples for all metrics (even if below min_samples)
        if elapsed >= self.config.duration_sec:
            return all(self.state.accepted_per_metric.get(key, 0) > 0 for key in self.config.metrics_of_interest)

        # Complete early if all metrics have sufficient samples
        return self._all_metrics_satisfied()

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

    def get_status(self, now: float) -> Dict[str, Any]:
        """
        Get detailed calibration status including pause information for UI feedback.
        """
        if not self.is_started():
            return {"state": "not_started", "progress": 0.0}

        if self.state.complete:
            return {"state": "complete", "progress": 1.0}

        progress = self.get_progress(now)
        elapsed = self._time_elapsed(now)

        if self.state.paused:
            return {
                "state": "paused",
                "progress": progress,
                "reason": "motion_detected" if self.state.motion_frames > 0 else "user_not_present",
                "elapsed_sec": elapsed,
                "stable_frames": self.state.stable_frames,
                "resume_threshold": self.config.motion_resume_frames
            }

        return {
            "state": "active",
            "progress": progress,
            "elapsed_sec": elapsed,
            "samples_total": self.state.samples_total
        }

    # ---- Retrieval ----

    def get_baseline(self) -> Optional[Dict[str, float]]:
        """
        Return baseline means if calibration has collected sufficient data.
        More flexible than requiring perfect completion - allows baseline when
        we have reasonable samples for most metrics.
        """
        if not self.is_started():
            return None

        # Require at least some samples for each metric
        available_metrics = []
        for key in self.config.metrics_of_interest:
            if self.state.accepted_per_metric.get(key, 0) >= max(3, self.config.min_samples // 3):
                available_metrics.append(key)

        # Need at least 2 metrics with sufficient samples
        if len(available_metrics) < 2:
            return None

        return {k: self.state.means[k] for k in available_metrics if k in self.state.means}

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
