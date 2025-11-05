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
ALERT_SOUND_FILE = "./utils/sound-alert.mp3"

# Custom CSS for modern, minimal UI
st.set_page_config(page_title="ErgoSense 🧘‍♀️", layout="wide", initial_sidebar_state="expanded")

# Inject custom CSS for better UI
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #667eea30;
        margin-bottom: 0.75rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    /* Alert styling */
    .alert-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: white;
        font-weight: 500;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Sidebar improvements */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
    }
    
    /* Section divider */
    .section-divider {
        border-top: 2px solid #667eea30;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">ErgoSense</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🎯 Real-time posture monitoring — Privacy-first, runs locally in your browser</p>', unsafe_allow_html=True)

# Performance and Alert settings in sidebar with better organization
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Video Settings (Collapsible)
    with st.expander("🎥 Video Settings", expanded=False):
        overlay_flag = st.checkbox("Show landmarks overlay", value=True)
        resolution_option = st.selectbox(
            "Camera resolution",
            ["640x480", "1280x720", "1920x1080"],
            index=0,
            help="Ideal resolution request (browser may negotiate down)."
        )
        res_w, res_h = (int(x) for x in resolution_option.split("x"))
    
    # Alert Preferences (Collapsible)
    with st.expander("🔔 Alert Preferences", expanded=True):
        enable_system_notifications = st.checkbox(
            "System Notifications", value=True,
            help="Use macOS native notifications (Notification Center)"
        )
        enable_overlay_alerts = st.checkbox(
            "Overlay Popups", value=True,
            help="Show small stacked overlay alerts (app-managed)"
        )
        enable_sound = st.checkbox(
            "Sound Alerts", value=True,
            help="Play a short chime for each posture alert (macOS only)"
        )
        test_alert = st.button("🧪 Test Alert", help="Send a sample alert to verify notification + sound")
        
        if test_alert:
            if enable_system_notifications:
                try:
                    system_notifier.notify("ErgoSense Test", "This is a test posture alert")
                except Exception:
                    pass
            if enable_sound:
                system_notifier.play_sound(ALERT_SOUND_FILE)
            st.success("✓ Test alert dispatched")
        
        st.markdown("**Alert Cooldowns**")
        neck_cooldown = st.slider(
            "Neck & Shoulder (s)", 3, 99, 12, 3,
            help="Minimum seconds before repeating the SAME neck/shoulder alert"
        )
        face_cooldown = st.slider(
            "Screen Distance (s)", 3, 99, 12, 3,
            help="Minimum seconds before repeating the SAME screen distance alert"
        )
    
    # Performance Settings (Collapsible)
    with st.expander("⚡ Performance", expanded=False):
        low_cpu_mode = st.checkbox("Low CPU Mode", value=False,
            help="Reduce frame processing to improve performance on slower systems")
        show_performance = st.checkbox("Show Performance Metrics", value=False,
            help="Display frame rate, latency, and processing stats")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Calibration Section
    st.markdown("### 🎯 Calibration")
    baseline_status_placeholder = st.empty()
    last_reset_placeholder = st.empty()
    reset_clicked = st.button("🔄 Reset Calibration", use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Posture Status Section
    st.markdown("### 🧘 Posture Monitor")
    posture_status_placeholder = st.empty()
    alerts_placeholder = st.empty()
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Posture History
    with st.expander("📊 Issue History", expanded=False):
        posture_history_placeholder = st.empty()
    
    # Raw Metrics (Collapsible)
    with st.expander("📈 Raw Metrics", expanded=False):
        metrics_placeholder = st.empty()
    
    # Performance Metrics
    perf_placeholder = st.empty()
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    BUILD_STAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"<small style='color: #888;'>Build: {BUILD_STAMP}</small>", unsafe_allow_html=True)

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
        # Track how many alerts have been dispatched to system notifications
        self._notified_alerts = 0

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
                    # Dispatch any newly added alerts via system notifications / sound
                    if self.alerts:
                        new_alerts = self.alerts[self._notified_alerts:]
                        if new_alerts:
                            for a in new_alerts:
                                raw_msg = a.get("message") if isinstance(a, dict) else str(a)
                                msg = str(raw_msg) if raw_msg is not None else "Posture alert"
                                if enable_system_notifications and msg:
                                    try:
                                        system_notifier.notify("ErgoSense Posture Alert", msg)
                                    except Exception:
                                        pass
                                if enable_sound:
                                    system_notifier.play_sound(ALERT_SOUND_FILE)
                            self._notified_alerts = len(self.alerts)
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
                f"""<div style='padding:12px;border-radius:10px;background:linear-gradient(135deg, #555 0%, #444 100%);
                color:#fff;font-size:0.85rem;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.1)'>
                <div style='opacity:0.7;margin-bottom:4px'>{label}</div><b style='font-size:1.2rem'>—</b></div>""",
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
            f"""<div style='padding:12px;border-radius:10px;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color:#fff;font-size:0.85rem;text-align:center;box-shadow:0 2px 8px rgba(102,126,234,0.3);line-height:1.3'>
            <div style='opacity:0.9;margin-bottom:4px'>{label}</div><b style='font-size:1.3rem'>{txt}</b></div>""",
            unsafe_allow_html=True
        )
    chip(_ear_chip_pl, "Ear-Shoulder", metrics.get("ear_shoulder_distance"))
    chip(_shoulder_chip_pl, "Shoulder-Nose", metrics.get("shoulder_nose_distance"))
    chip(_face_chip_pl, "Face Size", metrics.get("face_size"))


def render_calibration(vp):
    status = vp.calibration.get_status(time.time())
    if status["state"] == "complete":
        baseline_status_placeholder.markdown(
            """<div style='padding:10px;border-radius:8px;background:linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color:#fff;font-weight:600;text-align:center;box-shadow:0 2px 8px rgba(17,153,142,0.3)'>
            ✓ Calibration Complete</div>""",
            unsafe_allow_html=True
        )
    elif status["state"] == "paused":
        reason = status.get("reason", "motion_detected")
        if reason == "user_not_present":
            baseline_status_placeholder.markdown(
                """<div style='padding:10px;border-radius:8px;background:linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color:#fff;font-weight:600;text-align:center;box-shadow:0 2px 8px rgba(240,147,251,0.3)'>
                ⏸ Paused (waiting for user)</div>""",
                unsafe_allow_html=True
            )
        else:
            stable_needed = status["resume_threshold"] - status["stable_frames"]
            baseline_status_placeholder.markdown(
                f"""<div style='padding:10px;border-radius:8px;background:linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                color:#fff;font-weight:600;text-align:center;box-shadow:0 2px 8px rgba(250,112,154,0.3)'>
                ⏸ Paused - Hold still ({stable_needed} frames)</div>""",
                unsafe_allow_html=True
            )
    elif status["state"] == "active":
        pct = int(status['progress'] * 100)
        baseline_status_placeholder.markdown(
            f"""<div style='padding:10px;border-radius:8px;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color:#fff;font-weight:600;text-align:center;box-shadow:0 2px 8px rgba(102,126,234,0.3)'>
            <div>⏳ Calibrating... {pct}%</div>
            <div style='margin-top:6px;height:4px;background:rgba(255,255,255,0.3);border-radius:2px;overflow:hidden'>
                <div style='height:100%;width:{pct}%;background:#fff;transition:width 0.3s'></div>
            </div></div>""",
            unsafe_allow_html=True
        )
    else:
        # Check if user is present to determine if we should show "waiting" or "not started"
        current_metrics = vp.metrics or {}
        user_present = vp._is_user_present(current_metrics) if hasattr(vp, '_is_user_present') else False
        if user_present:
            baseline_status_placeholder.markdown(
                """<div style='padding:10px;border-radius:8px;background:linear-gradient(135deg, #868f96 0%, #596164 100%);
                color:#fff;font-weight:600;text-align:center;box-shadow:0 2px 8px rgba(134,143,150,0.3)'>
                ⚪ Not Started</div>""",
                unsafe_allow_html=True
            )
        else:
            baseline_status_placeholder.markdown(
                """<div style='padding:10px;border-radius:8px;background:linear-gradient(135deg, #868f96 0%, #596164 100%);
                color:#fff;font-weight:600;text-align:center;box-shadow:0 2px 8px rgba(134,143,150,0.3)'>
                👤 Waiting for user</div>""",
                unsafe_allow_html=True
            )


def render_posture_status(vp):
    # Check if user is currently detected
    current_metrics = vp.metrics or {}
    user_present = vp._is_user_present(current_metrics) if hasattr(vp, '_is_user_present') else True
    calibration_complete = vp.calibration.is_complete(time.time())

    if calibration_complete and not user_present:
        status = "User not detected"
        gradient = "linear-gradient(135deg, #868f96 0%, #596164 100%)"
        icon = "👤"
        shadow = "rgba(134,143,150,0.3)"
    elif vp.posture_result:
        status = vp.posture_result.overall
        if status == "OK":
            gradient = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"
            icon = "✓"
            shadow = "rgba(17,153,142,0.3)"
        elif status == "NeedsAttention":
            gradient = "linear-gradient(135deg, #ee0979 0%, #ff6a00 100%)"
            icon = "⚠️"
            shadow = "rgba(238,9,121,0.3)"
        else:
            gradient = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
            icon = "⏳"
            shadow = "rgba(240,147,251,0.3)"
    else:
        status = "Calibrating"
        gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        icon = "⏳"
        shadow = "rgba(102,126,234,0.3)"

    posture_status_placeholder.markdown(
        f"""<div style='padding:12px 16px;border-radius:10px;background:{gradient};
        color:#fff;font-weight:600;text-align:center;font-size:1rem;box-shadow:0 2px 8px {shadow}'>
        {icon} {status}</div>""",
        unsafe_allow_html=True
    )


def _render_posture_history(vp):
    # Build a concise visual list of issue counts with icons
    rows = []
    mapping = {
        "neck_forward": ("🦴 Neck Forward", "#ff6b6b"),
        "shoulder_raised": ("💪 Shoulder Raised", "#4ecdc4"),
        "face_too_close": ("👁️ Face Too Close", "#ffe66d"),
    }
    html_parts = []
    for key, (label, color) in mapping.items():
        count = vp.issue_history.get(key, 0)
        html_parts.append(
            f"""<div style='padding:8px;margin:4px 0;border-radius:6px;background:{color}20;
            border-left:3px solid {color};display:flex;justify-content:space-between;align-items:center'>
            <span style='font-weight:500'>{label}</span>
            <span style='background:{color};color:#fff;padding:2px 8px;border-radius:12px;font-weight:600;font-size:0.85rem'>{count}</span>
            </div>"""
        )
    
    if html_parts:
        posture_history_placeholder.markdown("".join(html_parts), unsafe_allow_html=True)
    else:
        posture_history_placeholder.markdown(
            "<div style='padding:12px;text-align:center;color:#888;font-style:italic'>No issues yet 🎉</div>",
            unsafe_allow_html=True
        )
    return html_parts


def process_new_alerts(vp):
    if "alerts_shown" not in st.session_state:
        st.session_state["alerts_shown"] = 0
    if "browser_alert_seq" not in st.session_state:
        st.session_state["browser_alert_seq"] = 0
    if "active_alerts" not in st.session_state:
        # Each entry: {message, expires_at}
        st.session_state["active_alerts"] = []
    if not vp.alerts:
        # Prune any lingering expired alerts
        now = time.time()
        st.session_state["active_alerts"] = [a for a in st.session_state["active_alerts"] if a["expires_at"] > now]
        if not st.session_state["active_alerts"]:
            alerts_placeholder.write("No alerts")
        else:
            _render_active_alerts()
        return
    new_alerts = vp.alerts[st.session_state["alerts_shown"]:]
    if new_alerts:
        for a in new_alerts:
            # Add to active alert list with 5s lifetime
            st.session_state["active_alerts"].append({
                "message": a["message"],
                "expires_at": time.time() + 5.0
            })
            # Update (legacy hook placeholder) - retained for potential future overlay
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
    _render_active_alerts()
    # Update posture history each cycle after processing alerts
    _render_posture_history(vp)


def _render_active_alerts():
    # Remove expired
    now = time.time()
    st.session_state["active_alerts"] = [a for a in st.session_state["active_alerts"] if a["expires_at"] > now]
    if not st.session_state["active_alerts"]:
        alerts_placeholder.markdown(
            "<div style='padding:10px;text-align:center;color:#888;font-style:italic;border-radius:8px;background:#f8f9fa20'>No active alerts</div>",
            unsafe_allow_html=True
        )
        return
    # Build HTML boxes with modern styling
    html_blocks = []
    for a in st.session_state["active_alerts"]:
        remaining = max(0, a["expires_at"] - now)
        pct = int((remaining / 5.0) * 100)
        bar = f"""<div style='height:3px;background:rgba(255,255,255,0.3);border-radius:2px;margin-top:6px;overflow:hidden'>
                  <div style='height:100%;width:{pct}%;background:#fff;border-radius:2px;transition:width 0.25s'></div></div>"""
        html_blocks.append(
            f"""<div style='background:linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding:10px 12px;margin-bottom:8px;border-radius:8px;font-size:0.85rem;color:#fff;
            box-shadow:0 2px 8px rgba(240,147,251,0.3);animation:slideIn 0.3s ease-out'>
            <div style='display:flex;align-items:center;gap:8px'>
                <span style='font-size:1.2rem'>⚠️</span>
                <span style='font-weight:500'>{html.escape(a['message'])}</span>
            </div>{bar}</div>"""
        )
    alerts_placeholder.markdown("".join(html_blocks), unsafe_allow_html=True)


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
                perf_html = f"""
                <div style='padding:10px;border-radius:8px;background:linear-gradient(135deg, #30cfd0 0%, #330867 100%);
                color:#fff;font-size:0.8rem;margin-top:1rem;box-shadow:0 2px 8px rgba(48,207,208,0.3)'>
                    <div style='font-weight:600;margin-bottom:6px'>⚡ Performance</div>
                    <div style='display:grid;grid-template-columns:1fr 1fr;gap:6px'>
                        <div>FPS: <b>{perf_metrics['fps']}</b></div>
                        <div>Latency: <b>{perf_metrics['latency_ms']}ms</b></div>
                        <div>Skip Rate: <b>{perf_metrics['skip_rate']}</b></div>
                        <div>Adaptive: <b>{perf_metrics['adaptive_skip']}</b></div>
                    </div>
                </div>
                """
                perf_placeholder.markdown(perf_html, unsafe_allow_html=True)
            else:
                perf_placeholder.empty()

            # Less frequent updates for non-critical UI
            if current_time - last_update >= 1.0:  # 1-second interval for these
                render_calibration(vp)
                if "calibration_reset_at" in st.session_state and st.session_state["calibration_reset_at"]:
                    last_reset_placeholder.markdown(
                        f"<small style='color:#888;font-style:italic'>Last reset: {st.session_state['calibration_reset_at']}</small>",
                        unsafe_allow_html=True
                    )
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
    # Dynamically update alert cooldowns from sidebar sliders
    try:
        vp.alert_system.cooldowns["neck_forward"] = float(neck_cooldown)
        vp.alert_system.cooldowns["shoulder_raised"] = float(neck_cooldown)
        vp.alert_system.cooldowns["face_too_close"] = float(face_cooldown)
    except Exception:
        pass
    if "calibration_reset_at" not in st.session_state:
        st.session_state["calibration_reset_at"] = None
    if reset_clicked:
        vp.reset_calibration()
        st.session_state["calibration_reset_at"] = datetime.now().strftime("%H:%M:%S")
        baseline_status_placeholder.write("Calibration reset. Collecting baseline…")
    main_loop(vp)
else:
    st.markdown("""
        <div style='padding:2rem;border-radius:12px;background:linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border:2px dashed #667eea50;text-align:center;margin:2rem 0'>
            <div style='font-size:3rem;margin-bottom:1rem'>📹</div>
            <h3 style='color:#667eea;margin-bottom:0.5rem'>Camera Access Required</h3>
            <p style='color:#888;margin-bottom:1rem'>Grant camera access to start real-time posture monitoring</p>
            <small style='color:#aaa'>Your video is processed locally and never uploaded</small>
        </div>
    """, unsafe_allow_html=True)
