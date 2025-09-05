"""
Streamlit UI for ErgoSense (browser-first).

First increment:
- Capture webcam frames in the browser
- Decode to an OpenCV image
- Call core.processing.process_frame(image)
- Display annotated frame and metrics

The processing pipeline is a stub for now and returns passthrough output.
"""

import streamlit as st
import numpy as np
import cv2 as cv
import time
from pathlib import Path
from datetime import datetime
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
from config.defaults import POSTURE_THRESHOLDS

# Ensure project root on sys.path for consistent imports regardless of CWD
# ROOT = Path(__file__).resolve().parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

from core.processing import process_frame

st.set_page_config(page_title="ErgoSense", layout="wide")
st.title("ErgoSense — Browser Demo v0.3 (live)")
st.caption("Webcam processing runs locally in your browser and container.")

BUILD_STAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with st.sidebar:
    st.markdown("### Status")
    st.write("UI wired to processing; pose detection minimal integration.")
    st.markdown("Build: " + BUILD_STAMP)
    st.markdown("---")
    overlay_flag = st.checkbox("Show landmarks overlay", value=True)
    st.markdown("### Baseline")
    baseline_status_placeholder = st.empty()
    last_reset_placeholder = st.empty()
    reset_clicked = st.button("Reset baseline")
    st.markdown("---")
    st.markdown("### Live Metrics")
    metrics_placeholder = st.empty()


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
    order = [
        "ear_shoulder_distance",
        "shoulder_nose_distance",
        "face_size",
        "latency_ms",
        "fps",
        "status",
    ]
    out = {}
    for k in order:
        if k in m and m[k] is not None:
            v = m[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                try:
                    v = round(float(v), 3)
                except Exception:
                    pass
            out[mapping.get(k, k)] = v
    # Include any other keys not in the order list
    for k, v in m.items():
        if k not in order and k != "timestamp" and v is not None:
            label = mapping.get(k, k)
            out[label] = v
    return out


class _VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.metrics = {}
        self._frame_count = 0
        self._last_time = cv.getTickCount()
        # Live baseline for metrics (simple EMA) to approximate calibration
        self._baseline = {}
        self._baseline_count = 0
        self._baseline_ready = False
        self.show_overlay = True

    def reset_baseline(self):
        self._baseline = {}
        self._baseline_count = 0
        self._baseline_ready = False

    def _update_baseline(self, m: dict):
        # Use a short warm-up to establish a baseline, then EMA
        keys = ("ear_shoulder_distance", "shoulder_nose_distance", "face_size")
        valid = False
        for k in keys:
            v = m.get(k)
            if isinstance(v, (int, float)):
                valid = True
                if k not in self._baseline:
                    self._baseline[k] = float(v)
                else:
                    alpha = 0.05
                    self._baseline[k] = (1 - alpha) * self._baseline[k] + alpha * float(v)
        if valid:
            self._baseline_count += 1
            if self._baseline_count >= 30:  # ~1s at 30fps
                self._baseline_ready = True

    def _posture_status(self, m: dict) -> str:
        if not self._baseline_ready:
            return "Calibrating…"
        try:
            neck_th = POSTURE_THRESHOLDS.get("neck_forward_threshold", 1.3)
            shoulder_th = POSTURE_THRESHOLDS.get("shoulder_raised_threshold", 0.8)
            face_th = POSTURE_THRESHOLDS.get("face_too_close_threshold", 1.2)

            neck_forward = (
                m.get("ear_shoulder_distance") is not None and
                self._baseline.get("ear_shoulder_distance") is not None and
                float(m["ear_shoulder_distance"]) > neck_th * float(self._baseline["ear_shoulder_distance"])
            )
            shoulder_raised = (
                m.get("shoulder_nose_distance") is not None and
                self._baseline.get("shoulder_nose_distance") is not None and
                float(m["shoulder_nose_distance"]) < shoulder_th * float(self._baseline["shoulder_nose_distance"])
            )
            face_close = (
                m.get("face_size") is not None and
                self._baseline.get("face_size") is not None and
                float(m["face_size"]) > face_th * float(self._baseline["face_size"])
            )

            return "Needs attention" if (neck_forward or shoulder_raised or face_close) else "OK"
        except Exception:
            return "Unknown"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        t0 = cv.getTickCount()
        annotated_bgr, metrics = process_frame(bgr)
        t1 = cv.getTickCount()

        elapsed_ms = (t1 - t0) / cv.getTickFrequency() * 1000.0
        metrics = {**(metrics or {}), "latency_ms": round(elapsed_ms, 2)}

        # Simple FPS estimate every 10 frames
        self._frame_count += 1
        if self._frame_count % 10 == 0:
            now = cv.getTickCount()
            dt = (now - self._last_time) / cv.getTickFrequency()
            fps = 10.0 / dt if dt > 0 else 0.0
            self._last_time = now
            metrics["fps"] = round(fps, 2)

        # Update baseline and posture status
        self._update_baseline(metrics)
        metrics["posture_status"] = self._posture_status(metrics)

        self.metrics = metrics
        # Respect overlay setting (pass-through vs annotated)
        out = annotated_bgr if self.show_overlay else bgr
        return av.VideoFrame.from_ndarray(out, format="bgr24")


ctx = webrtc_streamer(
    key="ergosense-live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=_VideoProcessor,
)

if ctx.state.playing and ctx.video_processor:
    vp = ctx.video_processor
    st.markdown("### Live stream running…")
    # Trigger baseline reset once if requested
    if "baseline_reset_at" not in st.session_state:
        st.session_state["baseline_reset_at"] = None
    if reset_clicked:
        vp.reset_baseline()
        st.session_state["baseline_reset_at"] = datetime.now().strftime("%H:%M:%S")
        baseline_status_placeholder.write("Baseline reset. Calibrating…")
    # Poll metrics periodically and update the sidebar placeholder
    # Loop until the stream stops.
    while ctx.state.playing:
        # Keep video processor in sync with overlay toggle
        if ctx.video_processor:
            ctx.video_processor.show_overlay = overlay_flag
        metrics = {k: v for k, v in (vp.metrics or {}).items() if k != "timestamp"}
        metrics_placeholder.write(_format_metrics(metrics))
        # Update baseline status display
        if getattr(vp, "_baseline_ready", False):
            baseline_status_placeholder.write("Baseline: Ready")
        else:
            baseline_status_placeholder.write("Baseline: Calibrating…")
        if st.session_state.get("baseline_reset_at"):
            last_reset_placeholder.write(f"Last reset at: {st.session_state['baseline_reset_at']}")
        time.sleep(0.25)
else:
    st.info("Grant camera access to start the live demo.")
