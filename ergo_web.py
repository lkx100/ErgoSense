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

# Ensure project root on sys.path for consistent imports regardless of CWD
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.processing import process_frame

st.set_page_config(page_title="ErgoSense", layout="wide")
st.title("ErgoSense — Browser Demo")
st.caption("Webcam processing runs locally in your browser and container.")

with st.sidebar:
    st.markdown("### Status")
    st.write("UI wired to processing stub. Pose detection will be added next.")

frame = st.camera_input("Enable your webcam to continue")

if frame is not None:
    try:
        # Decode the uploaded frame bytes to an OpenCV BGR image
        bytes_data = frame.getvalue()
        np_arr = np.frombuffer(bytes_data, dtype=np.uint8)
        bgr = cv.imdecode(np_arr, cv.IMREAD_COLOR)

        if bgr is None:
            st.error("Could not decode camera frame.")
        else:
            annotated_bgr, metrics = process_frame(bgr)

            # Convert BGR to RGB for Streamlit display
            rgb = cv.cvtColor(annotated_bgr, cv.COLOR_BGR2RGB)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(rgb, caption="Annotated frame", channels="RGB", use_column_width=True)
            with col2:
                st.markdown("### Metrics")
                st.json(metrics)
    except Exception as e:
        st.error(f"Unexpected error: {e}")
else:
    st.info("Grant camera access to start the demo.")
