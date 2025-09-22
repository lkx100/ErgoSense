"""
Streamlit UI for ErgoSense (browser-first).

Responsibilities of this module now split for lower cyclomatic complexity:
- Video processing (WebRTC transformer)
- UI state & rendering helpers
- Notification hook updates (native browser notifications)
- Visual metric chips + raw metrics
"""

from utils.logging_config import configure_silent_logging
configure_silent_logging()  # Must be first import

import os
import logging
import streamlit as st
import streamlit.components.v1 as components
import cv2 as cv
import time
from datetime import datetime
import html
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
from config.defaults import POSTURE_THRESHOLDS, TIMING_SETTINGS, ALERT_MESSAGES

# Suppress all C++ level logs and warnings
os.environ['GLOG_minloglevel'] = '3'  # 0=INFO,1=WARNING,2=ERROR,3=FATAL
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=INFO,1=WARNING,2=ERROR,3=FATAL
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['NVIDIA_DRIVER_CAPABILITIES'] = 'all'

# Python logging configuration
logging.getLogger().setLevel(logging.CRITICAL)  # Most aggressive
for logger_name in ['mediapipe', 'mediapipe.python', 'tensorflow', 'absl']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False

# Suppress specific MediaPipe loggers
import absl.logging
absl.logging.set_verbosity(absl.logging.FATAL)
absl.logging.use_absl_handler()

from core.processing import process_frame
from core.calibration import CalibrationSession, CalibrationConfig
from core.posture_analyzer import PostureAnalyzer
from monitoring.timer_manager import TimerManager
from monitoring.alert_system import AlertSystem, AlertConfig
from utils.system_notifier import SystemNotifier

# Initialize system notifier
system_notifier = SystemNotifier()

st.set_page_config(page_title="ErgoSense", layout="wide")
st.title("ErgoSense — Browser Demo v0.4 (live)")
st.caption("Webcam processing runs locally in your browser/container. No cloud upload.")

# Performance and Alert settings in sidebar
st.sidebar.markdown("### ⚙️ Settings")
st.sidebar.markdown("#### Alert Preferences")
enable_system_notifications = st.sidebar.checkbox("Enable System Notifications", value=True,
    help="Show notifications even when browser is minimized")
enable_sound = st.sidebar.checkbox("Enable Sound Alerts", value=True,
    help="Play sound for posture alerts")

st.sidebar.markdown("### 🔔 Alert Settings")
enable_sound = st.sidebar.checkbox("Enable Sound Alerts", value=True,
    help="Play sound when posture needs attention (works across tabs)")

st.sidebar.markdown("### 📊 Performance")
low_cpu_mode = st.sidebar.checkbox("Low CPU Mode", value=False,
    help="Reduce frame processing to improve performance on slower systems")
show_performance = st.sidebar.checkbox("Show Performance Metrics", value=False,
    help="Display frame rate, latency, and processing stats")

# Sound alert component
ALERT_SOUND = """
<script>
const alertSound = new Audio("data:audio/wav;base64,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qJrx4KOQjahZxWbcZgztj2c49nKmkId44S71j0c8eV9yDK6uPRzx5X18eDvjvQ6yKo9ZSS6l//8elePK/Lf//IInrOF/FvDoADYAGBMGb7FtErm5MXMlmPAJQVgWta7Zx2go+8xJ0UiCb8LHHdftWyLJE0QIAIsI+UbXu67dZMjmgDGCGl1H+vpF4NSDckSIkk7Vd+sxEhBQMRU8j/12UIRhzSaUdQ+rQU5kGeFxm+hb1oh6pWWmv3uvmReDl0UnvtapVaIzo1jZbf/pD6ElLqSX+rUmOQNpJFa/r+sa4e/pBlAABoAAAAA3CUgShLdGIxsY7AUABPRrgCABdDuQ5GC7DqPQCgbbJUAoRSUj+NIEig0YfyWUho1VBBBA//uQZB4ABZx5zfMakeAAAAmwAAAAF5F3P0w9GtAAACfAAAAAwLhMDmAYWMgVEG1U0FIGCBgXBXAtfMH10000EEEEEECUBYln03TTTdNBDZopopYvrTTdNa325mImNg3TTPV9q3pmY0xoO6bv3r00y+IDGid/9aaaZTGMuj9mpu9Mpio1dXrr5HERTZSmqU36A3CumzN/9Robv/Xx4v9ijkSRSNLQhAWumap82WRSBUqXStV/YcS+XVLnSS+WLDroqArFkMEsAS+eWmrUzrO0oEmE40RlMZ5+ODIkAyKAGUwZ3mVKmcamcJnMW26MRPgUw6j+LkhyHGVGYjSUUKNpuJUQoOIAyDvEyG8S5yfK6dhZc0Tx1KI/gviKL6qvvFs1+bWtaz58uUNnryq6kt5RzOCkPWlVqVX2a/EEBUdU1KrXLf40GoiiFXK///qpoiDXrOgqDR38JB0bw7SoL+ZB9o1RCkQjQ2CBYZKd/+VJxZRRZlqSkKiws0WFxUyCwsKiMy7hUVFhIaCrNQsKkTIsLivwKKigsj8XYlwt/WKi2N4d//uQRCSAAjURNIHpMZBGYiaQPSYyAAABLAAAAAAAACWAAAAApUF/Mg+0aohSIRobBAsMlO//Kk4soosy1JSFRYWaLC4qZBYWFRGZdwqKiwkNBVmoWFSJkWFxX4FFRQWR+LsS4W/rFRb/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////VEFHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAU291bmRib3kuZGUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMjAwNGh0dHA6Ly93d3cuc291bmRib3kuZGUAAAAAAAAAACU=");

window.playAlertBeep = function() {
    alertSound.play().catch(e => console.log('Sound play failed:', e));
}
</script>
"""
components.html(ALERT_SOUND, height=0)

# ---------------------------
# Browser Notification Scripts
# ---------------------------
SERVICE_WORKER = """
// ergosense-sw.js
self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(clients.claim());
});

self.addEventListener('push', (event) => {
  const data = event.data.json();
  const options = {
    body: data.message,
    icon: '/favicon.ico',
    badge: '/favicon.ico',
    tag: 'ergosense-notification',
    renotify: true,
    requireInteraction: false,
    silent: false
  };
  event.waitUntil(
    self.registration.showNotification('ErgoSense Alert', options)
  );
});
"""

# NOTIF_BOOTSTRAP = """
# <script>
# (function() {
#   const ID = "ergosense-alert-hook";
#   let swRegistration = null;
#   let isNotificationEnabled = false;

#   // Service Worker Registration
#   async function registerServiceWorker() {
#     if ('serviceWorker' in navigator && 'PushManager' in window) {
#       try {
#         // Create a Blob URL for the service worker
#         const swBlob = new Blob([`${SERVICE_WORKER}`], { type: 'text/javascript' });
#         const swUrl = URL.createObjectURL(swBlob);
        
#         swRegistration = await navigator.serviceWorker.register(swUrl, {
#           scope: '.'
#         });
#         console.log('ServiceWorker registered');
        
#         // Cleanup Blob URL
#         URL.revokeObjectURL(swUrl);
        
#         return true;
#       } catch (error) {
#         console.warn('ServiceWorker registration failed:', error);
#         return false;
#       }
#     }
#     return false;
#   }

#   // Initialize notification system
#   async function initNotifications() {
#     if (!("Notification" in window)) {
#       console.warn("This browser does not support notifications");
#       return;
#     }

#     // Request permission
#     if (Notification.permission === "default") {
#       const permission = await Notification.requestPermission();
#       isNotificationEnabled = permission === "granted";
#     } else {
#       isNotificationEnabled = Notification.permission === "granted";
#     }

#     if (isNotificationEnabled) {
#       await registerServiceWorker();
#     }
#   }

#   // Enhanced notification function
#   async function notify(msg, severity = 'info') {
#     if (!isNotificationEnabled) return;

#     try {
#       // Fallback to regular notifications if service worker isn't available
#       if (!swRegistration || !swRegistration.active) {
#         const notification = new Notification("ErgoSense Alert", {
#           body: msg,
#           icon: '/favicon.ico',
#           tag: 'ergosense-notification',
#           renotify: true,
#           requireInteraction: false,
#           silent: false
#         });
        
#         // Auto-close after 5 seconds for non-critical alerts
#         if (severity !== 'critical') {
#           setTimeout(() => notification.close(), 5000);
#         }
        
#         return;
#       }

#       // Use service worker for more reliable notifications
#       const data = { message: msg, severity };
#       await swRegistration.active.postMessage(data);
      
#     } catch (error) {
#       console.warn('Notification failed:', error);
#     }
#   }

#   // Setup DOM hook
#   function ensureHook() {
#     let el = document.getElementById(ID);
#     if (!el) {
#       el = document.createElement("div");
#       el.id = ID;
#       el.setAttribute("data-alert-seq", "0");
#       el.style.display = "none";
#       document.body.appendChild(el);
#     }
#     return el;
#   }

#   // Initialize
#   const hook = ensureHook();
#   initNotifications();
  
#   // Observe for new alerts
#   let lastSeq = "0";
#   const obs = new MutationObserver(() => {
#     const seq = hook.getAttribute("data-alert-seq");
#     if (seq && seq !== lastSeq) {
#       lastSeq = seq;
#       const payload = hook.getAttribute("data-alert-msg");
#       const severity = hook.getAttribute("data-alert-severity") || 'info';
#       if (payload) {
#         notify(payload, severity);
#       }
#     }
#   });
  
#   obs.observe(hook, {
#     attributes: true,
#     attributeFilter: ["data-alert-seq", "data-alert-msg", "data-alert-severity"]
#   });
# })();
# </script>
# """
# components.html(NOTIF_BOOTSTRAP, height=0)

# ---------------------------
# Layout Placeholders
# ---------------------------
main_col, side_metrics_col = st.columns([3, 2])
visual_metrics_container = side_metrics_col.container()
_metric_cols = visual_metrics_container.columns(3)
_ear_chip_pl = _metric_cols[0].empty()
_shoulder_chip_pl = _metric_cols[1].empty()
_face_chip_pl = _metric_cols[2].empty()
raw_metrics_container = st.container()
alert_hook_placeholder = st.empty()  # dynamic HTML hook updates for notifications

BUILD_STAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("### Status")
    st.write("Refactored UI with browser notifications.")
    st.markdown(f"Build: {BUILD_STAMP}")
    st.markdown("---")
    overlay_flag = st.checkbox("Show landmarks overlay", value=True)
    resolution_option = st.selectbox(
        "Camera resolution",
        ["640x480", "1280x720", "1920x1080"],
        index=0,
        help="Ideal resolution request (browser may negotiate down)."
    )
    res_w, res_h = (int(x) for x in resolution_option.split("x"))
    st.markdown("---")
    st.markdown("### Calibration")
    baseline_status_placeholder = st.empty()
    last_reset_placeholder = st.empty()
    reset_clicked = st.button("Reset calibration")
    st.markdown("---")
    st.markdown("### Posture")
    posture_status_placeholder = st.empty()
    alerts_placeholder = st.empty()
    # New: history of posture issues (persistent counters)
    st.markdown("### Posture Issue History")
    posture_history_placeholder = st.empty()
    st.markdown("---")
    st.markdown("### Metrics (raw)")
    metrics_placeholder = st.empty()
    st.markdown("### Performance")
    perf_placeholder = st.empty()


# ---------------------------
# Utility / Formatting
# ---------------------------
def _format_metrics(m: dict) -> dict:
    if not m:
        return {}
    mapping = {
        "ear_shoulder_distance": "Ear-Shoulder Distance",
        "shoulder_nose_distance": "Shoulder-Nose Distance",
        "face_size": "Face Size (proxy)",
        "latency_ms": "Latency (ms)",
        "status": "Status",
        "fps": "FPS",
        "posture_status": "Posture Status",
    }
    ordered = [
        "ear_shoulder_distance",
        "shoulder_nose_distance",
        "face_size",
        "latency_ms",
        "fps",
        "status",
    ]
    out = {}
    for k in ordered:
        if k in m and m[k] is not None:
            v = m[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                try:
                    v = round(float(v), 3)
                except Exception:
                    pass
            out[mapping.get(k, k)] = v
    for k, v in m.items():
        if k not in ordered and k != "timestamp" and v is not None:
            out[mapping.get(k, k)] = v
    return out


# ---------------------------
# Video Processor
# ---------------------------
class _VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.metrics = {}
        self._frame_count = 0
        self._last_time = cv.getTickCount()
        self.show_overlay = True

        # Performance monitoring for adaptive frame skipping
        self._latency_history = []
        self._skip_count = 0
        self._process_count = 0
        self._target_latency_ms = 50.0  # Target max latency per frame
        self._adaptive_skip_rate = 0.0  # 0.0 = no skip, 1.0 = skip all

        # User presence tracking for post-calibration
        self._user_absent_frames = 0
        self._user_absent_threshold = 30  # Reset calibration after 30 frames (~1 second at 30fps) without user

        self.calibration = CalibrationSession(
            CalibrationConfig(
                duration_sec=TIMING_SETTINGS.get("calibration_duration", 5),
                min_samples=30,  # Reduced from 75 for faster calibration
            )
        )
        self.analyzer = PostureAnalyzer(POSTURE_THRESHOLDS)
        self.timer_manager = TimerManager({
            "neck_forward": TIMING_SETTINGS.get("neck_posture_alert_delay", 30),
            "shoulder_raised": TIMING_SETTINGS.get("shoulder_posture_alert_delay", 30),
            "face_too_close": TIMING_SETTINGS.get("screen_distance_alert_delay", 10),
        })
        self.alert_system = AlertSystem(
            messages=ALERT_MESSAGES,
            cooldowns={
                "neck_forward": 120.0,
                "shoulder_raised": 120.0,
                "face_too_close": 90.0,
            },
            config=AlertConfig(
                default_cooldown=120.0,
                max_alerts_per_window=10,
                window_seconds=600.0,
                deduplicate_in_window=False,
            ),
        )
        self.posture_result = None
        self.low_cpu_mode = False
        self._skip_toggle = False
        self.alerts = []
        self.baseline_progress = 0.0
        # History counters for posture issue occurrences
        self.issue_history = {
            "neck_forward": 0,
            "shoulder_raised": 0,
            "face_too_close": 0,
        }
        # Track last alert timestamp per condition to allow repeated reminders
        self._last_issue_alert_ts = {
            "neck_forward": 0.0,
            "shoulder_raised": 0.0,
            "face_too_close": 0.0,
        }

    def reset_calibration(self):
        self.calibration.reset()
        self.posture_result = None
        self.alerts.clear()
        self.baseline_progress = 0.0
        self.timer_manager.clear_all()

    def _update_performance_metrics(self, latency_ms: float):
        """Update performance tracking with latest frame latency."""
        self._latency_history.append(latency_ms)
        # Keep only last 10 measurements for rolling average
        if len(self._latency_history) > 10:
            self._latency_history.pop(0)

        # Calculate adaptive skip rate based on recent performance
        if len(self._latency_history) >= 3:
            avg_latency = sum(self._latency_history) / len(self._latency_history)
            if avg_latency > self._target_latency_ms:
                # Increase skip rate when latency is high
                self._adaptive_skip_rate = min(0.8, self._adaptive_skip_rate + 0.1)
            elif avg_latency < self._target_latency_ms * 0.7:
                # Decrease skip rate when performance is good
                self._adaptive_skip_rate = max(0.0, self._adaptive_skip_rate - 0.05)

    def _should_skip_frame(self) -> bool:
        """Determine if current frame should be skipped based on performance."""
        if not self.low_cpu_mode:
            return False

        # Use adaptive skipping based on performance
        if self._adaptive_skip_rate > 0:
            # Probabilistic skipping based on adaptive rate
            import random
            return random.random() < self._adaptive_skip_rate

        # Fallback to simple alternating skip
        self._skip_toggle = not self._skip_toggle
        return self._skip_toggle

    def _is_user_present(self, metrics: dict) -> bool:
        """Check if user is present based on processing metrics."""
        status = metrics.get("status", "")
        # User is present if we have successful pose detection
        return status == "ok" and any(
            key in metrics for key in ["ear_shoulder_distance", "shoulder_nose_distance", "face_size"]
        )

    def _update_fps(self, metrics: dict):
        self._frame_count += 1
        if self._frame_count % 10 == 0:
            now_ticks = cv.getTickCount()
            dt = (now_ticks - self._last_time) / cv.getTickFrequency()
            fps = 10.0 / dt if dt > 0 else 0.0
            self._last_time = now_ticks
            metrics["fps"] = round(fps, 2)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")

        # Smart frame skipping based on performance
        if self._should_skip_frame():
            self._skip_count += 1
            return av.VideoFrame.from_ndarray(bgr, format="bgr24")

        self._process_count += 1
        t0 = cv.getTickCount()

        try:
            annotated_bgr, processed = process_frame(bgr)
        except Exception as e:
            # If processing fails completely, return original frame with error metrics
            print(f"[ErgoSense] Frame processing failed: {e}")
            processed = {"status": "frame_processing_error", "error": str(e)}
            annotated_bgr = bgr

        t1 = cv.getTickCount()
        elapsed_ms = (t1 - t0) / cv.getTickFrequency() * 1000.0

        # Update performance tracking
        self._update_performance_metrics(elapsed_ms)

        metrics = {**(processed or {}), "latency_ms": round(elapsed_ms, 2)}
        self._update_fps(metrics)

        # Add performance metrics
        total_frames = self._skip_count + self._process_count
        if total_frames > 0:
            metrics["skip_rate"] = round(self._skip_count / total_frames, 2)
            metrics["adaptive_skip_rate"] = round(self._adaptive_skip_rate, 2)

        now_sec = time.time()

        try:
            if not self.calibration.is_started():
                # Only start calibration if user is present
                if self._is_user_present(metrics):
                    self.calibration.start(now_sec)
                else:
                    # User not present, don't start calibration yet
                    self.baseline_progress = 0.0

            if self.calibration.is_started() and not self.calibration.is_complete(now_sec):
                self.calibration.update(metrics, now_sec)
                self.baseline_progress = self.calibration.get_progress(now_sec)
                # Reset user absent counter when actively calibrating
                self._user_absent_frames = 0
            elif self.calibration.is_complete(now_sec):
                self.baseline_progress = 1.0
                # Check for user presence after calibration is complete
                user_present = self._is_user_present(metrics)
                if user_present:
                    self._user_absent_frames = 0
                else:
                    self._user_absent_frames += 1
                    # Reset calibration if user has been absent too long
                    if self._user_absent_frames >= self._user_absent_threshold:
                        print(f"[ErgoSense] User absent for {self._user_absent_frames} frames, resetting calibration")
                        self.reset_calibration()
                        # Don't start calibration immediately - wait for user to return
                        self.baseline_progress = 0.0
        except Exception as e:
            print(f"[ErgoSense] Calibration error: {e}")
            self.baseline_progress = 0.0

        try:
            baseline = self.calibration.get_baseline()
            # Only analyze posture if we have a baseline and user is present
            if baseline and self._is_user_present(metrics):
                self.posture_result = self.analyzer.analyze(metrics, baseline)
            else:
                self.posture_result = None
        except Exception as e:
            print(f"[ErgoSense] Posture analysis error: {e}")
            self.posture_result = None

        try:
            # Only process alerts if we have posture results (implies user is present and calibrated)
            if baseline and self.posture_result:
                flags = {
                    "neck_forward": self.posture_result.neck_forward,
                    "shoulder_raised": self.posture_result.shoulder_raised,
                    "face_too_close": self.posture_result.face_too_close,
                }
                triggers = self.timer_manager.update(flags, now_sec)
                if triggers:
                    # Update history counters and allow repeated reminders (ignore AlertSystem cooldown for counting)
                    for cond in triggers:
                        if cond in self.issue_history:
                            self.issue_history[cond] += 1
                    self.alerts.extend(self.alert_system.process(triggers, now_sec))
        except Exception as e:
            print(f"[ErgoSense] Alert processing error: {e}")

        # Safe posture result handling
        try:
            user_present = self._is_user_present(metrics)
            calibration_complete = self.calibration.is_complete(time.time())

            if calibration_complete and not user_present:
                # User has left after calibration - show appropriate status
                metrics["posture_status"] = "User not detected"
                # Clear posture flags when user is not present
                metrics.update({
                    "neck_forward": False,
                    "shoulder_raised": False,
                    "face_too_close": False,
                })
            elif self.posture_result:
                metrics["posture_status"] = self.posture_result.overall
                metrics.update({
                    "neck_forward": self.posture_result.neck_forward,
                    "shoulder_raised": self.posture_result.shoulder_raised,
                    "face_too_close": self.posture_result.face_too_close,
                })
                for rk, rv in self.posture_result.ratios.items():
                    metrics[f"ratio_{rk}"] = round(rv, 4) if isinstance(rv, (int, float)) else rv
            else:
                metrics["posture_status"] = "Calibrating"
        except Exception as e:
            print(f"[ErgoSense] Posture result processing error: {e}")
            metrics["posture_status"] = "Error"

        self.metrics = metrics
        return av.VideoFrame.from_ndarray(annotated_bgr if self.show_overlay else bgr, format="bgr24")


# ---------------------------
# Media Constraints
# ---------------------------
video_constraints = {
    "video": {
        "width": {"ideal": res_w},
        "height": {"ideal": res_h},
        "frameRate": {"ideal": 30}
    },
    "audio": False
}

ctx = webrtc_streamer(
    key="ergosense-live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints=video_constraints,
    video_processor_factory=_VideoProcessor,
)

# ---------------------------
# Rendering Helpers
# ---------------------------
def render_visual_chips(metrics: dict):
    def chip(placeholder, label, value, fmt="{:.3f}"):
        if value is None:
            placeholder.markdown(
                f"<div style='padding:6px;border-radius:6px;background:#555;color:#fff;font-size:0.8rem'>{label}<br><b>—</b></div>",
                unsafe_allow_html=True
            )
            return
        if isinstance(value, (int, float)):
            try:
                txt = fmt.format(value)
            except Exception:
                txt = str(value)
        else:
            txt = str(value)
        placeholder.markdown(
            f"<div style='padding:6px;border-radius:6px;background:#424242;color:#fff;font-size:0.75rem;line-height:1.2'>{label}<br><b>{txt}</b></div>",
            unsafe_allow_html=True
        )
    chip(_ear_chip_pl, "Ear-Shoulder", metrics.get("ear_shoulder_distance"))
    chip(_shoulder_chip_pl, "Shoulder-Nose", metrics.get("shoulder_nose_distance"))
    chip(_face_chip_pl, "Face Size", metrics.get("face_size"))


def render_calibration(vp):
    status = vp.calibration.get_status(time.time())
    if status["state"] == "complete":
        baseline_status_placeholder.write("Calibration: Complete")
    elif status["state"] == "paused":
        reason = status.get("reason", "motion_detected")
        if reason == "user_not_present":
            baseline_status_placeholder.write("Calibration: Paused (waiting for user)")
        else:
            stable_needed = status["resume_threshold"] - status["stable_frames"]
            baseline_status_placeholder.write(f"Calibration: Paused (motion detected) - Hold still ({stable_needed} frames)")
    elif status["state"] == "active":
        pct = f"{int(status['progress'] * 100)}%"
        baseline_status_placeholder.write(f"Calibration: In progress ({pct})")
    else:
        # Check if user is present to determine if we should show "waiting" or "not started"
        current_metrics = vp.metrics or {}
        user_present = vp._is_user_present(current_metrics) if hasattr(vp, '_is_user_present') else False
        if user_present:
            baseline_status_placeholder.write("Calibration: Not started")
        else:
            baseline_status_placeholder.write("Calibration: Waiting for user")


def render_posture_status(vp):
    # Check if user is currently detected
    current_metrics = vp.metrics or {}
    user_present = vp._is_user_present(current_metrics) if hasattr(vp, '_is_user_present') else True
    calibration_complete = vp.calibration.is_complete(time.time())

    if calibration_complete and not user_present:
        status = "User not detected"
        bg = "#757575"  # Gray color for user not detected
    elif vp.posture_result:
        status = vp.posture_result.overall
        colors = {"OK": "#2e7d32", "NeedsAttention": "#c62828", "Calibrating": "#f9a825"}
        bg = colors.get(status, "#424242")
    else:
        status = "Calibrating"
        bg = "#f9a825"

    posture_status_placeholder.markdown(
        f"<div style='padding:4px 8px;border-radius:4px;background:{bg};color:#fff;font-weight:600;'>Posture: {status}</div>",
        unsafe_allow_html=True
    )


def _render_posture_history(vp):
    # Build a concise markdown table of issue counts
    rows = []
    mapping = {
        "neck_forward": "Neck Forward",
        "shoulder_raised": "Shoulder Raised",
        "face_too_close": "Face Too Close",
    }
    for key, label in mapping.items():
        count = vp.issue_history.get(key, 0)
        rows.append(f"- **{label}**: {count}")
    posture_history_placeholder.markdown("\n".join(rows) if rows else "No issues yet.")
    return rows


def process_new_alerts(vp):
    if "alerts_shown" not in st.session_state:
        st.session_state["alerts_shown"] = 0
    if "browser_alert_seq" not in st.session_state:
        st.session_state["browser_alert_seq"] = 0
    if not vp.alerts:
        alerts_placeholder.write("No alerts")
        return
    new_alerts = vp.alerts[st.session_state["alerts_shown"]:]
    if new_alerts:
        for a in new_alerts:
            st.toast(a["message"])
            # Update browser notification hook
            st.session_state["browser_alert_seq"] += 1
            seq = st.session_state["browser_alert_seq"]
            safe_msg = html.escape(a["message"])
            severity = "critical" if "neck_forward" in a["condition"] else "warning"
            alert_hook_placeholder.html(
                f"<div id='ergosense-alert-hook' "
                f"data-alert-seq='{seq}' "
                f"data-alert-msg='{safe_msg}' "
                f"data-alert-severity='{severity}' "
                f"style='display:none'></div>"
            )
        st.session_state["alerts_shown"] = len(vp.alerts)
    latest = vp.alerts[-5:]
    lines = []
    for a in latest:
        ts = datetime.fromtimestamp(a["timestamp"]).strftime("%H:%M:%S")
        lines.append(f"[{ts}] {a['message']}")
    alerts_placeholder.write("\n".join(lines))
    # Update posture history each cycle after processing alerts
    _render_posture_history(vp)


def update_metrics_display(vp):
    metrics = {k: v for k, v in (vp.metrics or {}).items() if k != "timestamp"}
    metrics_placeholder.write(_format_metrics(metrics))
    if metrics:
        render_visual_chips(metrics)


def main_loop(vp):
    # Reduced complexity and update frequency
    update_interval = 0.5  # Reduced from 0.25 for better performance
    last_update = 0
    while ctx.state.playing:
        current_time = time.time()
        if current_time - last_update >= update_interval:
            vp.show_overlay = overlay_flag
            update_metrics_display(vp)

            # Show performance metrics if enabled
            if show_performance:
                perf_metrics = {
                    "fps": vp.metrics.get("fps", 0),
                    "latency_ms": vp.metrics.get("latency_ms", 0),
                    "skip_rate": vp.metrics.get("skip_rate", 0),
                    "adaptive_skip": vp.metrics.get("adaptive_skip_rate", 0),
                }
                perf_placeholder.write(f"**Performance:** FPS: {perf_metrics['fps']}, Latency: {perf_metrics['latency_ms']}ms, Skip Rate: {perf_metrics['skip_rate']}, Adaptive: {perf_metrics['adaptive_skip']}")

            # Less frequent updates for non-critical UI
            if current_time - last_update >= 1.0:  # 1-second interval for these
                render_calibration(vp)
                if "calibration_reset_at" in st.session_state and st.session_state["calibration_reset_at"]:
                    last_reset_placeholder.write(f"Last reset at: {st.session_state['calibration_reset_at']}")
                _render_posture_history(vp)

            # Keep posture status and alerts more responsive
            render_posture_status(vp)
            process_new_alerts(vp)

            last_update = current_time
        time.sleep(0.1)  # Shorter sleep but less frequent updates


# ---------------------------
# Top-Level Control
# ---------------------------
if ctx.state.playing and ctx.video_processor:
    vp = ctx.video_processor
    vp.low_cpu_mode = low_cpu_mode
    if "calibration_reset_at" not in st.session_state:
        st.session_state["calibration_reset_at"] = None
    if reset_clicked:
        vp.reset_calibration()
        st.session_state["calibration_reset_at"] = datetime.now().strftime("%H:%M:%S")
        baseline_status_placeholder.write("Calibration reset. Collecting baseline…")
    main_loop(vp)
else:
    st.info("Grant camera access to start the live demo.")
