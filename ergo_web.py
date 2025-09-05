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
import sys
from pathlib import Path
from datetime import datetime
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

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
    st.markdown("---")
    st.markdown("#### Debug")
    # st.write({
    #     "build": BUILD_STAMP,
    #     "script": str(Path(__file__).resolve()),
    # })


class _VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.metrics = {}
        self._frame_count = 0
        self._last_time = cv.getTickCount()

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

        self.metrics = metrics
        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")


ctx = webrtc_streamer(
    key="ergosense-live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=_VideoProcessor,
)

if ctx.state.playing and ctx.video_processor:
    vp = ctx.video_processor
    with st.sidebar:
        st.markdown("### Live Metrics")
        st.write({k: v for k, v in (vp.metrics or {}).items() if k != "timestamp"})
else:
    st.info("Grant camera access to start the live demo.")
