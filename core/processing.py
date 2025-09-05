"""
Processing wrapper for ErgoSense.

Minimal wiring:
- Ensure PoseDetector is initialized (MediaPipe LIVE_STREAM mode)
- Convert BGR->RGB, call detect_async, read latest landmarks
- Draw landmarks and compute simple metrics via LandmarkExtractor
"""

from typing import Any, Dict, Tuple, Optional
import time
import math
import cv2 as cv

from core.pose_detector import PoseDetector
from core.landmark_extractor import LandmarkExtractor
from utils.camera import draw_landmarks_on_image
from config.defaults import MODEL_SETTINGS


# Global singletons for lightweight reuse across frames
_detector: Optional[PoseDetector] = None
_extractor: LandmarkExtractor = LandmarkExtractor()
_detector_initialized: bool = False


def _ensure_detector() -> bool:
    global _detector, _detector_initialized
    if _detector is None:
        model_path = MODEL_SETTINGS.get("model_path", "./models/pose_landmarker_full.task")
        print(f"[ErgoSense] Initializing PoseDetector (IMAGE mode) with model: {model_path}")
        # Use IMAGE mode for single still frames from st.camera_input
        _detector = PoseDetector(model_path, running_mode="IMAGE")
    _detector_initialized = _detector.initialize()
    print(f"[ErgoSense] PoseDetector init: {_detector_initialized}")
    return _detector_initialized


def process_frame(image) -> Tuple[Any, Dict[str, Any]]:
    """Process a single BGR image and return (annotated_image, metrics).

    - BGR input (OpenCV)
    - Returns BGR annotated image for display and a metrics dict
    """
    metrics: Dict[str, Any] = {"status": "init"}

    if image is None:
        return image, {"status": "no_input"}

    ok = _ensure_detector()
    if not ok:
        return image, {"status": "pose_detector_init_failed"}
    det = _detector
    if det is None:
        return image, {"status": "pose_detector_unavailable"}

    # Convert to RGB for MediaPipe
    try:
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    except Exception:
        return image, {"status": "bgr_to_rgb_failed"}

    # Use IMAGE mode for a single capture to get immediate results
    landmarks = det.detect_image(rgb)
    if landmarks is None:
        print("[ErgoSense] No landmarks detected in IMAGE mode")

    annotated = image
    if landmarks:
        try:
            annotated = draw_landmarks_on_image(image, landmarks)
        except Exception:
            # If drawing fails, keep the original image
            annotated = image

        # Compute posture metrics (best-effort)
        try:
            raw_metrics = _extractor.get_all_metrics(landmarks) or {}
        except Exception:
            raw_metrics = {}
        # Round numeric metrics for display
        metrics = {}
        for k, v in raw_metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v):
                metrics[k] = round(float(v), 4)
            else:
                metrics[k] = v
        metrics["status"] = "ok"
    else:
        metrics = {"status": "no_landmarks"}

    # Set timestamp in metrics
    metrics["timestamp"] = time.time()
    return annotated, metrics
