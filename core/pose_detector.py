"""
MediaPipe-based pose detection for ErgoSense
"""
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional, Callable, Any


class PoseDetector:
    """Pose detection using MediaPipe (supports LIVE_STREAM and IMAGE modes)"""
    
    def __init__(self, model_path: str = "./models/pose_landmarker_full.task", running_mode: str = "LIVE_STREAM"):
        self.model_path = model_path
        self.latest_result = None
        self.landmarker = None
        self.running_mode_str = running_mode.upper() if isinstance(running_mode, str) else "LIVE_STREAM"
        
        # MediaPipe classes
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
    def _result_callback(self, result, output_image, timestamp_ms):
        """Callback function to handle pose detection results"""
        self.latest_result = result
        
    def initialize(self) -> bool:
        """Initialize the pose landmarker"""
        try:
            mode = self.VisionRunningMode.LIVE_STREAM if self.running_mode_str == "LIVE_STREAM" else self.VisionRunningMode.IMAGE
            if mode == self.VisionRunningMode.LIVE_STREAM:
                options = self.PoseLandmarkerOptions(
                    base_options=self.BaseOptions(model_asset_path=self.model_path),
                    running_mode=mode,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_segmentation_masks=False,
                    result_callback=self._result_callback,
                )
            else:
                options = self.PoseLandmarkerOptions(
                    base_options=self.BaseOptions(model_asset_path=self.model_path),
                    running_mode=mode,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_segmentation_masks=False,
                )
            
            self.landmarker = self.PoseLandmarker.create_from_options(options)
            return True
        except Exception as e:
            print(f"Failed to initialize pose detector: {e}")
            return False
    
    def detect_async(self, rgb_image: np.ndarray, timestamp_ms: int) -> None:
        """Perform asynchronous pose detection"""
        if self.landmarker is None:
            return
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        self.landmarker.detect_async(mp_image, timestamp_ms)

    def detect_image(self, rgb_image: np.ndarray):
        """Perform synchronous pose detection on a single RGB image (IMAGE mode)."""
        if self.landmarker is None:
            return None
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        try:
            result = self.landmarker.detect(mp_image)
            if result and result.pose_landmarks:
                return result.pose_landmarks[0]
            return None
        except Exception:
            return None
    
    def get_latest_landmarks(self) -> Optional[Any]:
        """Get the latest pose landmarks"""
        if self.latest_result and self.latest_result.pose_landmarks:
            return self.latest_result.pose_landmarks[0]  # Return first pose
        return None
    
    def cleanup(self):
        """Clean up resources"""
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
