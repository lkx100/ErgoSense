"""
Processing wrapper for ErgoSense.

Minimal wiring:
- Ensure PoseDetector is initialized (MediaPipe IMAGE or LIVE_STREAM mode)
- Convert BGR->RGB, perform detection, get landmarks
- Draw landmarks and compute low-level geometric metrics via LandmarkExtractor

NOTE:
The modules core.pose_detector and core.landmark_extractor remain foundational.
Even with the newer calibration, posture analyzer, timer, and alert layers, we still:
  * use PoseDetector for MediaPipe inference (no replacement exists)
  * use LandmarkExtractor to derive primitive metrics (distances) that higher-level
    components (CalibrationSession & PostureAnalyzer) consume.
Do NOT remove these files unless you replace their functionality with an equivalent
detector + metric feature extraction pipeline.
"""

from typing import Any, Dict, Tuple, Optional
import time
import math
import cv2 as cv
import os

# Debug / verbosity toggle:
# Set environment variable ERGOSENSE_DEBUG=1 to enable verbose internal logs.
DEBUG = os.getenv("ERGOSENSE_DEBUG", "0") in ("1", "true", "True")

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
        model_path = MODEL_SETTINGS.get("model_path", "./models/pose_landmarker_lite.task")
        if DEBUG:
            print(f"[ErgoSense] Initializing PoseDetector (IMAGE mode) with model: {model_path}")
        # Use IMAGE mode for single still frames from st.camera_input
        _detector = PoseDetector(model_path, running_mode="IMAGE")
    _detector_initialized = _detector.initialize()
    if DEBUG:
        print(f"[ErgoSense] PoseDetector init: {_detector_initialized}")
    return _detector_initialized


def _convert_bgr_to_rgb(image) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Try converting BGR->RGB. On failure return (None, error_metrics)."""
    try:
        return cv.cvtColor(image, cv.COLOR_BGR2RGB), None
    except Exception:
        return None, {"status": "bgr_to_rgb_failed"}


def _detect_landmarks(det: Optional[PoseDetector], rgb) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Run detection and return landmarks or an appropriate metrics dict if unavailable."""
    if det is None:
        return None, {"status": "pose_detector_unavailable"}
    landmarks = det.detect_image(rgb)
    if landmarks is None:
        if DEBUG:
            print("[ErgoSense] No landmarks detected in IMAGE mode")
        return None, {"status": "no_landmarks"}
    return landmarks, None


def _build_metrics_from_landmarks(landmarks) -> Dict[str, Any]:
    """Extract and format metrics from landmarks (status included)."""
    try:
        raw = _extractor.get_all_metrics(landmarks) or {}
    except Exception:
        raw = {}
    metrics: Dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                if not math.isnan(v):
                    metrics[k] = round(float(v), 4)
                else:
                    metrics[k] = v
            except Exception:
                metrics[k] = v
        else:
            metrics[k] = v
    metrics["status"] = "ok"
    return metrics


def process_frame(image) -> Tuple[Any, Dict[str, Any]]:
    """
    Process a single BGR image and return (annotated_image, metrics).

    Complexity minimized by delegating discrete steps to helpers:
      1. Input validation
      2. Detector readiness
      3. Color conversion
      4. Landmark detection
      5. Metric extraction + annotation
    """
    if image is None:
        return image, {"status": "no_input"}

    if not _ensure_detector():
        return image, {"status": "pose_detector_init_failed"}

    rgb, err = _convert_bgr_to_rgb(image)
    if err:
        return image, err  # color conversion failed

    landmarks, err = _detect_landmarks(_detector, rgb)
    if err:  # includes no_landmarks case
        metrics = err
        metrics["timestamp"] = time.time()
        return image, metrics

    # Landmarks present: annotate + metrics
    try:
        annotated = draw_landmarks_on_image(image, landmarks)
    except Exception:
        annotated = image  # fallback

    metrics = _build_metrics_from_landmarks(landmarks)
    metrics["timestamp"] = time.time()
    return annotated, metrics
