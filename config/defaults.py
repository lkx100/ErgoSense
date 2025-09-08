"""
Default configuration values for ErgoSense
"""

# Posture thresholds (multipliers of baseline values)
POSTURE_THRESHOLDS = {
    'neck_forward_threshold': 1.2,  # 20% increase in ear-shoulder distance
    'shoulder_raised_threshold': 0.85,  # 15% decrease in shoulder-nose distance  
    'face_too_close_threshold': 1.15,  # 15% increase in face size (closer to screen)
}

# Timing settings (seconds)
TIMING_SETTINGS = {
    'calibration_duration': 3,  # default 5 sec
    'neck_posture_alert_delay': 2,  # default 30 sec
    'shoulder_posture_alert_delay': 2,  # default 30 sec
    'screen_distance_alert_delay': 2,  # default 10 sec
    'eye_break_reminder_interval': 300,  # default 20 minutes(1200 sec)
}

# Camera settings
CAMERA_SETTINGS = {
    'camera_id': 0,
    'target_fps': 15,  # Reduced from 30 for better performance
}

# MediaPipe model settings
MODEL_SETTINGS = {
    'model_path': './models/pose_landmarker_lite.task',
    'min_pose_detection_confidence': 0.5,
    'min_pose_presence_confidence': 0.5,
    'min_tracking_confidence': 0.5,
}

# Alert messages
ALERT_MESSAGES = {
    'neck_forward': "Check your neck posture! Sit up straight.",
    'shoulders_raised': "Relax your shoulders!",
    'too_close_to_screen': "You're a bit too close to the screen.",
    'eye_break_reminder': "Time for an eye break!",
    'calibration_instruction': "Sit up straight and look at the screen for calibration.",
}
