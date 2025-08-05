"""
Default configuration values for ErgoSense
"""

# Posture thresholds (multipliers of baseline values)
POSTURE_THRESHOLDS = {
    'neck_forward_threshold': 1.3,  # 30% increase in ear-shoulder distance
    'shoulder_raised_threshold': 0.8,  # 20% decrease in shoulder-nose distance  
    'face_too_close_threshold': 1.2,  # 20% increase in face size (closer to screen)
}

# Timing settings (seconds)
TIMING_SETTINGS = {
    'calibration_duration': 5,
    'neck_posture_alert_delay': 30,
    'shoulder_posture_alert_delay': 30, 
    'screen_distance_alert_delay': 10,
    'eye_break_reminder_interval': 1200,  # 20 minutes
}

# Camera settings
CAMERA_SETTINGS = {
    'camera_id': 0,
    'target_fps': 30,
}

# MediaPipe model settings
MODEL_SETTINGS = {
    'model_path': './models/pose_landmarker_full.task',
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
