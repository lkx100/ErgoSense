"""
Alert System Module for ErgoSense.

Purpose:
    Transform posture condition triggers (e.g., "neck_forward", "face_too_close")
    into user-facing alert events with rate limiting (cooldowns) and optional
    severity classification.

Separation of Concerns:
    - The TimerManager decides WHEN a condition trigger is eligible (based on
      persistence / dwell time).
    - The AlertSystem decides WHETHER to surface a user-visible alert now,
      respecting per-condition cooldowns or global limits.

Core Concepts:
    - Each condition has:
        * Message template (simple string for now).
        * Severity (informational priority; future UI styling or escalation).
        * Cooldown window (seconds) preventing repeated identical alerts too often.
    - Global guardrails:
        * max_alerts_per_window: Optional cap on the number of alerts within a rolling window.
        * window_seconds: Size of the rolling window for the global cap.
        * deduplicate_in_window: If True, identical condition alerts suppressed within window.

Example:
    alert_system = AlertSystem(
        messages={
            "neck_forward": "Check your neck posture.",
            "shoulder_raised": "Relax your shoulders.",
            "face_too_close": "You are too close to the screen.",
        },
        severities={
            "neck_forward": "warning",
            "shoulder_raised": "info",
            "face_too_close": "warning",
        },
        cooldowns={
            "neck_forward": 90.0,
            "shoulder_raised": 90.0,
            "face_too_close": 60.0,
        },
        default_cooldown=120.0,
    )

    triggers = ["neck_forward", "face_too_close"]
    alerts = alert_system.process(triggers, now=time.time())

Returned alert object format:
    {
        "condition": "neck_forward",
        "message": "Check your neck posture.",
        "severity": "warning",
        "timestamp": 1736199999.123,
        "cooldown": 90.0,
        "sequence_id": 7
    }

Extensibility:
    - Swap message strings with templated strings using metrics context.
    - Implement pluggable backends (e.g., system notifications) by subclassing
      or adding a dispatcher layer.
    - Introduce priority queue if future user experience demands escalations.

Thread-Safety:
    - Not thread-safe by default (designed for single-threaded event loop / Streamlit run).
    - Wrap mutable state with locks if adapting to multi-threaded environment.

Author: ErgoSense Engineering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Any
import time
import math
import collections


# ---------------------------------------------------------
# Data Classes
# ---------------------------------------------------------

@dataclass
class AlertConfig:
    """
    Global configuration for alert governance.

    Attributes:
        default_cooldown: Fallback cooldown seconds if a condition lacks a specific cooldown.
        max_alerts_per_window: Optional cap on number of alerts in rolling time window (None disables).
        window_seconds: Size of the rolling time window used for global limiting.
        deduplicate_in_window: If True, the same condition will not appear twice in the window.
    """
    default_cooldown: float = 5.0   # 5 sec default
    max_alerts_per_window: Optional[int] = None
    window_seconds: float = 300.0
    deduplicate_in_window: bool = False


@dataclass
class AlertEvent:
    """
    Structured alert emitted by the system.
    """
    condition: str
    message: str
    severity: str
    timestamp: float
    cooldown: float
    sequence_id: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "cooldown": self.cooldown,
            "sequence_id": self.sequence_id,
        }


# ---------------------------------------------------------
# Core Alert System
# ---------------------------------------------------------

class AlertSystem:
    """
    Alert system applying per-condition cooldowns and optional global governance.

    Parameters:
        messages: Mapping condition -> message string.
        severities: Mapping condition -> severity token (e.g., "info", "warning", "critical").
        cooldowns: Mapping condition -> cooldown seconds (suppresses repeated alerts within this interval).
        config: AlertConfig object controlling global behaviors.

    Public Methods:
        process(triggers, now=None) -> List[dict]:
            Convert triggered conditions into alert events (dicts) subject to cooldown
            and global limitations.

        get_condition_state(condition) -> dict:
            Diagnostic info for a condition (last timestamp, cooldown remaining, etc).

        recent_alerts(now=None) -> List[AlertEvent]:
            Return list of alerts still inside the rolling window.
    """

    def __init__(
        self,
        messages: Dict[str, str],
        severities: Optional[Dict[str, str]] = None,
        cooldowns: Optional[Dict[str, float]] = None,
        config: Optional[AlertConfig] = None,
    ):
        self.messages = dict(messages)
        self.severities = dict(severities) if severities else {}
        self.cooldowns = dict(cooldowns) if cooldowns else {}
        self.config = config or AlertConfig()

        # Internal tracking
        self._last_sent_ts: Dict[str, float] = {}
        self._sequence_counter: int = 0
        self._alert_history: collections.deque[AlertEvent] = collections.deque()  # time-ordered

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def process(self, triggers: Iterable[str], now: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Given an iterable of triggered condition names, return alert event dicts
        that pass cooling and global gating.

        Steps:
            1. Filter out conditions still in cooldown.
            2. Apply global window constraints (max count, dedup).
            3. Emit AlertEvent objects and update histories.

        Returns:
            List of alert event dictionaries (ready for UI or logging).
        """
        now = now or time.time()
        self._prune_window(now)

        emitted: List[AlertEvent] = []
        seen_conditions_in_window = {event.condition for event in self._alert_history}

        for cond in triggers:
            # Check duplicate-in-window rule
            if self.config.deduplicate_in_window and cond in seen_conditions_in_window:
                continue

            # Cooldown check
            cooldown = self._cooldown_for(cond)
            last_ts = self._last_sent_ts.get(cond)
            if last_ts is not None and (now - last_ts) < cooldown:
                continue

            # Global window limit
            if self._would_exceed_global_limit(now):
                break  # Stop emitting further alerts this cycle (strict policy)

            # Construct alert
            message = self.messages.get(cond, cond)
            severity = self.severities.get(cond, "info")
            self._sequence_counter += 1
            event = AlertEvent(
                condition=cond,
                message=message,
                severity=severity,
                timestamp=now,
                cooldown=cooldown,
                sequence_id=self._sequence_counter,
            )

            emitted.append(event)
            # Update state
            self._last_sent_ts[cond] = now
            self._alert_history.append(event)
            seen_conditions_in_window.add(cond)

        return [e.as_dict() for e in emitted]

    def get_condition_state(self, condition: str, now: Optional[float] = None) -> Dict[str, Any]:
        """
        Retrieve diagnostic info for a condition.
        """
        now = now or time.time()
        last_ts = self._last_sent_ts.get(condition)
        cooldown = self._cooldown_for(condition)
        remaining = None
        if last_ts is not None:
            elapsed = now - last_ts
            if elapsed < cooldown:
                remaining = cooldown - elapsed
        return {
            "condition": condition,
            "last_alert_at": last_ts,
            "cooldown": cooldown,
            "cooldown_remaining": remaining,
        }

    def recent_alerts(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Return alert events still inside the rolling window (diagnostics).
        """
        now = now or time.time()
        self._prune_window(now)
        return [e.as_dict() for e in self._alert_history]

    # -----------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------

    def _cooldown_for(self, condition: str) -> float:
        return float(self.cooldowns.get(condition, self.config.default_cooldown))

    def _prune_window(self, now: float) -> None:
        """
        Remove old alerts outside the rolling time window.
        """
        if self.config.max_alerts_per_window is None and not self.config.deduplicate_in_window:
            # No need to maintain a window if no constraints depend on it
            return
        window = self.config.window_seconds
        while self._alert_history and (now - self._alert_history[0].timestamp) > window:
            self._alert_history.popleft()

    def _would_exceed_global_limit(self, now: float) -> bool:
        if self.config.max_alerts_per_window is None:
            return False
        self._prune_window(now)
        return len(self._alert_history) >= self.config.max_alerts_per_window

    # -----------------------------------------------------
    # Administrative / Dynamic Updates
    # -----------------------------------------------------

    def set_cooldown(self, condition: str, cooldown_seconds: float) -> None:
        """
        Dynamically adjust a specific condition's cooldown.
        """
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be > 0")
        self.cooldowns[condition] = float(cooldown_seconds)

    def set_message(self, condition: str, message: str) -> None:
        self.messages[condition] = message

    def set_severity(self, condition: str, severity: str) -> None:
        self.severities[condition] = severity

    def reset_condition(self, condition: str) -> None:
        """
        Forget last alert timestamp for a condition (useful after successful intervention).
        """
        if condition in self._last_sent_ts:
            del self._last_sent_ts[condition]

    def clear_history(self) -> None:
        """
        Clear all alert history and last sent timestamps (global reset).
        """
        self._alert_history.clear()
        self._last_sent_ts.clear()

    # -----------------------------------------------------
    # Representation
    # -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AlertSystem(last_sent={len(self._last_sent_ts)} conds, "
            f"history={len(self._alert_history)} events, config={self.config})"
        )


# ---------------------------------------------------------
# Convenience Factory (Optional)
# ---------------------------------------------------------

def create_default_alert_system() -> AlertSystem:
    """
    Provide a ready-made AlertSystem using typical ergonomic posture messages.
    Adjust cooldowns or severities as needed for experiments.
    """
    from config.defaults import ALERT_MESSAGES  # Local import to avoid circulars at module import time

    default_severities = {
        "neck_forward": "warning",
        "shoulders_raised": "info",
        "too_close_to_screen": "warning",
        "eye_break_reminder": "info",
    }
    # Map older naming to new consistent keys if required
    # (You can customize externally if you prefer direct key parity).
    cooldowns = {
        "neck_forward": 120.0,
        "shoulders_raised": 120.0,
        "too_close_to_screen": 90.0,
        "eye_break_reminder": 1200.0,
    }

    # The ALERT_MESSAGES keys in defaults may differ (e.g., 'shoulders_raised'),
    # but triggers might use 'shoulder_raised'. Normalize externally as needed.

    return AlertSystem(
        messages=ALERT_MESSAGES,
        severities=default_severities,
        cooldowns=cooldowns,
        config=AlertConfig(
            default_cooldown=120.0,
            max_alerts_per_window=10,
            window_seconds=600.0,
            deduplicate_in_window=False,
        ),
    )
