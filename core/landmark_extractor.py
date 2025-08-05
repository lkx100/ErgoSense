"""
Extract specific landmarks for posture analysis
"""
import numpy as np
from typing import Optional, Tuple, Any


class LandmarkExtractor:
    """Extract and calculate key posture metrics from pose landmarks"""
    
    # MediaPipe pose landmark indices
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    
    @staticmethod
    def extract_key_points(landmarks) -> Optional[dict]:
        """Extract key landmarks for posture analysis"""
        if not landmarks or len(landmarks) < 33:
            return None
            
        try:
            return {
                'nose': landmarks[LandmarkExtractor.NOSE],
                'left_ear': landmarks[LandmarkExtractor.LEFT_EAR],
                'right_ear': landmarks[LandmarkExtractor.RIGHT_EAR],
                'left_shoulder': landmarks[LandmarkExtractor.LEFT_SHOULDER],
                'right_shoulder': landmarks[LandmarkExtractor.RIGHT_SHOULDER],
                'left_eye': landmarks[LandmarkExtractor.LEFT_EYE],
                'right_eye': landmarks[LandmarkExtractor.RIGHT_EYE],
            }
        except IndexError:
            return None
    
    @staticmethod
    def calculate_ear_shoulder_distance(key_points: dict) -> Optional[float]:
        """Calculate horizontal distance between ear and shoulder (neck posture)"""
        try:
            # Use average of both sides
            left_dist = abs(key_points['left_ear'].x - key_points['left_shoulder'].x)
            right_dist = abs(key_points['right_ear'].x - key_points['right_shoulder'].x)
            return (left_dist + right_dist) / 2
        except (KeyError, AttributeError):
            return None
    
    @staticmethod
    def calculate_shoulder_nose_distance(key_points: dict) -> Optional[float]:
        """Calculate vertical distance between shoulder and nose (shoulder posture)"""
        try:
            # Use average shoulder position
            avg_shoulder_y = (key_points['left_shoulder'].y + key_points['right_shoulder'].y) / 2
            return abs(key_points['nose'].y - avg_shoulder_y)
        except (KeyError, AttributeError):
            return None
    
    @staticmethod
    def calculate_face_size(key_points: dict) -> Optional[float]:
        """Calculate face size using eye distance as proxy for screen distance"""
        try:
            eye_distance = abs(key_points['left_eye'].x - key_points['right_eye'].x)
            return eye_distance
        except (KeyError, AttributeError):
            return None
    
    @staticmethod
    def get_all_metrics(landmarks) -> Optional[dict]:
        """Get all posture metrics from landmarks"""
        key_points = LandmarkExtractor.extract_key_points(landmarks)
        if not key_points:
            return None
            
        return {
            'ear_shoulder_distance': LandmarkExtractor.calculate_ear_shoulder_distance(key_points),
            'shoulder_nose_distance': LandmarkExtractor.calculate_shoulder_nose_distance(key_points),
            'face_size': LandmarkExtractor.calculate_face_size(key_points),
            'timestamp': None  # To be set by caller
        }
