"""
Streamlit UI for ErgoSense (browser-first).

Responsibilities of this module now split for lower cyclomatic complexity:
- Video processing (WebRTC transformer)
- UI state & rendering helpers
- Notification hook updates (native browser notifications)
- Visual metric chips + raw metrics
"""

import streamlit as st
import streamlit.components.v1 as components
import cv2 as cv
import time
from datetime import datetime
import html
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
from config.defaults import POSTURE_THRESHOLDS, TIMING_SETTINGS, ALERT_MESSAGES

from core.processing import process_frame
from core.calibration import CalibrationSession, CalibrationConfig
from core.posture_analyzer import PostureAnalyzer
from monitoring.timer_manager import TimerManager
from monitoring.alert_system import AlertSystem, AlertConfig

st.set_page_config(page_title="ErgoSense", layout="wide")
st.title("ErgoSense — Browser Demo v0.4 (live)")
st.caption("Webcam processing runs locally in your browser/container. No cloud upload.")

# ---------------------------
# Browser Notification Script
# ---------------------------
NOTIF_BOOTSTRAP = """
<script>
(function() {
  const ID = "ergosense-alert-hook";
  function ensureHook(){
     let el = document.getElementById(ID);
     if(!el){
        el = document.createElement("div");
        el.id = ID;
        el.setAttribute("data-alert-seq", "0");
        el.style.display = "none";
        document.body.appendChild(el);
     }
     return el;
  }
  async function askPerm(){
     if(!("Notification" in window)) return;
     if(Notification.permission === "default"){
        try { await Notification.requestPermission(); } catch(e){}
     }
  }
  function notify(msg){
     if(!("Notification" in window)) return;
     if(Notification.permission === "granted"){
        try { new Notification("ErgoSense", { body: msg }); } catch(e){}
     }
  }
  const hook = ensureHook();
  askPerm();
  let lastSeq = "0";
  const obs = new MutationObserver(()=>{
     const seq = hook.getAttribute("data-alert-seq");
     if(seq && seq !== lastSeq){
        lastSeq = seq;
        const payload = hook.getAttribute("data-alert-msg");
        if(payload){
           notify(payload);
        }
     }
  });
  obs.observe(hook, { attributes: true, attributeFilter: ["data-alert-seq","data-alert-msg"]});
})();
</script>
"""
components.html(NOTIF_BOOTSTRAP, height=0)

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
    low_cpu_mode_sidebar = st.checkbox("Low CPU mode (skip alternate frames)", value=False)
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
        self.calibration = CalibrationSession(
            CalibrationConfig(
                duration_sec=TIMING_SETTINGS.get("calibration_duration", 5),
                min_samples=75,
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
        if self.low_cpu_mode:
            self._skip_toggle = not self._skip_toggle
            if self._skip_toggle:
                return av.VideoFrame.from_ndarray(bgr, format="bgr24")
        t0 = cv.getTickCount()
        annotated_bgr, processed = process_frame(bgr)
        t1 = cv.getTickCount()
        elapsed_ms = (t1 - t0) / cv.getTickFrequency() * 1000.0
        metrics = {**(processed or {}), "latency_ms": round(elapsed_ms, 2)}
        self._update_fps(metrics)
        now_sec = time.time()
        if not self.calibration.is_started():
            self.calibration.start(now_sec)
        if not self.calibration.is_complete(now_sec):
            self.calibration.update(metrics, now_sec)
            self.baseline_progress = self.calibration.get_progress(now_sec)
        else:
            self.baseline_progress = 1.0
        baseline = self.calibration.get_baseline()
        self.posture_result = self.analyzer.analyze(metrics, baseline)
        if baseline:
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
        if self.posture_result:
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
    if vp.calibration.is_complete(time.time()):
        baseline_status_placeholder.write("Calibration: Complete")
    else:
        pct = f"{int(vp.baseline_progress * 100)}%"
        baseline_status_placeholder.write(f"Calibration: In progress ({pct})")


def render_posture_status(vp):
    status = vp.posture_result.overall if vp.posture_result else "Calibrating"
    colors = {"OK": "#2e7d32", "NeedsAttention": "#c62828", "Calibrating": "#f9a825"}
    bg = colors.get(status, "#424242")
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
            alert_hook_placeholder.html(
                f"<div id='ergosense-alert-hook' data-alert-seq='{seq}' data-alert-msg='{safe_msg}' style='display:none'></div>"
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
    # Reduced complexity: each concern delegated to helper
    while ctx.state.playing:
        vp.show_overlay = overlay_flag
        update_metrics_display(vp)
        render_calibration(vp)
        if "calibration_reset_at" in st.session_state and st.session_state["calibration_reset_at"]:
            last_reset_placeholder.write(f"Last reset at: {st.session_state['calibration_reset_at']}")
        render_posture_status(vp)
        process_new_alerts(vp)
        # Refresh posture history even if no new alerts
        _render_posture_history(vp)
        time.sleep(0.25)


# ---------------------------
# Top-Level Control
# ---------------------------
if ctx.state.playing and ctx.video_processor:
    vp = ctx.video_processor
    vp.low_cpu_mode = low_cpu_mode_sidebar
    if "calibration_reset_at" not in st.session_state:
        st.session_state["calibration_reset_at"] = None
    if reset_clicked:
        vp.reset_calibration()
        st.session_state["calibration_reset_at"] = datetime.now().strftime("%H:%M:%S")
        baseline_status_placeholder.write("Calibration reset. Collecting baseline…")
    main_loop(vp)
else:
    st.info("Grant camera access to start the live demo.")
