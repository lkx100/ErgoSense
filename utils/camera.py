"""
Camera management utilities
"""
import cv2 as cv
import numpy as np
from typing import Optional, Tuple


class CameraManager:
    """Manage camera operations for ErgoSense"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize camera"""
        try:
            self.cap = cv.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                return False
            
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from camera"""
        if not self.is_initialized or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def get_rgb_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get RGB frame (MediaPipe format)"""
        ret, frame = self.read_frame()
        if ret and frame is not None:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            return True, rgb_frame
        return False, None
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_initialized = False


def draw_landmarks_on_image(image: np.ndarray, landmarks) -> np.ndarray:
    """Draw pose landmarks on image"""
    if not landmarks:
        return image
    
    annotated_image = image.copy()
    height, width = annotated_image.shape[:2]
    
    # Draw landmarks as circles
    for i, landmark in enumerate(landmarks):
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv.circle(annotated_image, (x, y), 10, (0, 255, 0), -1)
    
    # Draw key connections
    connections = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        # Arms  
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        # Body
        (11, 23), (12, 24), (23, 24),
        # Legs
        (23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28), (28, 30), (28, 32)
    ]
    
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = (int(landmarks[start_idx].x * width), int(landmarks[start_idx].y * height))
            end_point = (int(landmarks[end_idx].x * width), int(landmarks[end_idx].y * height))
            cv.line(annotated_image, start_point, end_point, (255, 0, 0), 2)
    
    return annotated_image
