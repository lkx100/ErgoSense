"""
Timer Manager Module for ErgoSense.

Purpose:
    Track how long posture issue conditions (e.g., neck_forward, shoulder_raised,
    face_too_close) remain active and determine when they have persisted long
    enough to justify an alert/reminder.

Key Concepts:
    - A "condition" is a boolean flag (True = active, False = inactive) evaluated
      every update cycle (e.g., each processed frame or throttled interval).
    - Each condition has an associated delay threshold (in seconds) specifying
      how long it must remain continuously active before triggering an alert.
    - Once a condition triggers, it will not re-trigger again until it first
      clears (goes False) and then re-accumulates time past the threshold again.

Usage Pattern:
    from monitoring.timer_manager import TimerManager

    delays = {
        "neck_forward": 30.0,
        "shoulder_raised": 30.0,
        "face_too_close": 10.0,
    }
    tm = TimerManager(delays)

    # Inside a loop (now = time.time()):
    triggers = tm.update({
        "neck_forward": posture_result.neck_forward,
        "shoulder_raised": posture_result.shoulder_raised,
        "face_too_close": posture_result.face_too_close,
    }, now)

    if triggers:
        # Pass to alert system
        alerts = alert_system.process(triggers, now)

Design Decisions:
    - The module does NOT enforce cooldowns (that is handled by the alert system).
    - Computation is O(number_of_conditions) each update—intentionally simple.
    - "Active time" is implicit (now - active_since) instead of accumulated time
      so that intermittent frame drops or variable update intervals are handled robustly.
    - If a condition toggles off (False), its state is reset (active_since cleared),
      requiring a full delay duration again before the next trigger.

Extensibility:
    - Additional metadata (e.g., cumulative active duration over a day) could be added
      by extending ConditionTiming.
    - A method could be introduced to return a structured diagnostic snapshot
      (e.g., for UI debug panels).

Author: ErgoSense Engineering
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class ConditionTiming:
    """
    State for a single condition.

    Attributes:
        active_since: Timestamp when the condition most recently became active
                      (None if currently inactive).
        last_triggered: Timestamp when the condition last emitted a trigger
                        (None if never).
    """
    active_since: Optional[float] = None
    last_triggered: Optional[float] = None


class TimerManager:
    """
    Manage persistence-based triggering of posture condition alerts.

    Public API:
        - update(flags: Dict[str, bool], now: float) -> List[str]
            Process the current active/inactive state of each condition and
            return a list of condition names that just crossed their delay
            thresholds *this* update call.

        - get_state_snapshot(now: float) -> Dict[str, Dict[str, float|None]]
            Obtain diagnostic info (active duration, delay threshold, etc.).

    Behavior Rules:
        1. When a condition transitions from False -> True, record active_since.
        2. While True, if (now - active_since) >= delay and no prior trigger
           occurred after this activation window, emit a trigger.
        3. When a condition transitions to False, clear active_since (reset).
        4. A condition must clear before it can trigger again.
    """

    def __init__(self, delay_thresholds: Dict[str, float]):
        """
        Args:
            delay_thresholds: Mapping of condition key -> required continuous
                              active duration (seconds) before triggering.
        """
        self.delay_thresholds: Dict[str, float] = dict(delay_thresholds)
        self._conditions: Dict[str, ConditionTiming] = {
            key: ConditionTiming() for key in self.delay_thresholds.keys()
        }

    # ---------------------------------------------------------------------
    # Core Update
    # ---------------------------------------------------------------------
    def update(self, flags: Dict[str, bool], now: float) -> List[str]:
        """
        Update condition states given the current boolean flags.

        Args:
            flags: Dict of {condition_name: bool_active} for current frame/interval.
                   Conditions not present in delay_thresholds are ignored (safe no-op).
            now:   Current timestamp (float seconds, e.g., time.time()).

        Returns:
            A list of condition names that have JUST triggered on this call.
        """
        triggers: List[str] = []

        for cond, active in flags.items():
            if cond not in self.delay_thresholds:
                # Unknown condition; skip silently to avoid raising errors in
                # dynamic scenarios (forward compatibility).
                continue

            timing = self._conditions.setdefault(cond, ConditionTiming())
            delay = self.delay_thresholds[cond]

            if active:
                # Condition is active; set start if newly active.
                if timing.active_since is None:
                    timing.active_since = now

                # Determine if we have met or exceeded the delay
                active_duration = now - timing.active_since
                has_met_delay = active_duration >= delay

                # Trigger only if:
                # - delay is satisfied
                # - never triggered during this activation window
                #   (we enforce that by checking last_triggered < active_since)
                if has_met_delay:
                    if (
                        timing.last_triggered is None
                        or (timing.active_since is not None and timing.last_triggered < timing.active_since)
                    ):
                        triggers.append(cond)
                        timing.last_triggered = now
            else:
                # Condition inactive; reset activation window
                timing.active_since = None

        return triggers

    # ---------------------------------------------------------------------
    # Introspection / Diagnostics
    # ---------------------------------------------------------------------
    def get_state_snapshot(self, now: float) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Produce a diagnostic snapshot of current state.

        Returns:
            {
              condition_name: {
                 "active_since": float | None,
                 "active_duration": float | None,
                 "delay": float,
                 "last_triggered": float | None,
                 "ready_in": float | None   # seconds until threshold (if active)
              },
              ...
            }
        """
        snapshot: Dict[str, Dict[str, Optional[float]]] = {}
        for cond, timing in self._conditions.items():
            delay = self.delay_thresholds.get(cond, 0.0)
            if timing.active_since is not None:
                active_duration = now - timing.active_since
                remaining = max(0.0, delay - active_duration)
            else:
                active_duration = None
                remaining = None

            snapshot[cond] = {
                "active_since": timing.active_since,
                "active_duration": active_duration,
                "delay": delay,
                "last_triggered": timing.last_triggered,
                "ready_in": remaining,
            }
        return snapshot

    # ---------------------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------------------
    def set_delay(self, condition: str, delay_seconds: float) -> None:
        """
        Dynamically adjust a delay threshold. Will create the condition record
        if it does not already exist.
        """
        self.delay_thresholds[condition] = delay_seconds
        self._conditions.setdefault(condition, ConditionTiming())

    def reset_condition(self, condition: str) -> None:
        """
        Clear activation state for a single condition (does not erase last_triggered).
        """
        if condition in self._conditions:
            self._conditions[condition].active_since = None

    def clear_all(self) -> None:
        """
        Reset all conditions' activation windows (preserving last_triggered).
        Useful when recalibrating posture baseline.
        """
        for timing in self._conditions.values():
            timing.active_since = None

    # ---------------------------------------------------------------------
    # Representation
    # ---------------------------------------------------------------------
    def __repr__(self) -> str:
        parts = []
        for cond, timing in self._conditions.items():
            parts.append(
                f"{cond}(active_since={timing.active_since}, last_triggered={timing.last_triggered}, delay={self.delay_thresholds.get(cond)})"
            )
        return f"TimerManager({', '.join(parts)})"
