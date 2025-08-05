"""
ErgoSense - Real-time posture monitoring application
Entry point for the application
"""
import cv2 as cv
import time

from core.pose_detector import PoseDetector
from core.landmark_extractor import LandmarkExtractor
from utils.camera import CameraManager, draw_landmarks_on_image
from config.defaults import MODEL_SETTINGS


class ErgoSenseApp:
    """Main application class for ErgoSense"""
    
    def __init__(self):
        self.pose_detector = PoseDetector(MODEL_SETTINGS['model_path'])
        self.camera_manager = CameraManager()
        self.landmark_extractor = LandmarkExtractor()
        self.running = False
    
    def initialize(self) -> bool:
        """Initialize all components"""
        print("Initializing ErgoSense...")
        
        # Initialize camera
        if not self.camera_manager.initialize():
            print("Error: Could not initialize camera")
            return False
        
        # Initialize pose detector
        if not self.pose_detector.initialize():
            print("Error: Could not initialize pose detector")
            return False
        
        print("ErgoSense initialized successfully!")
        return True
    
    def run_demo(self):
        """Run pose detection demo"""
        print("Starting pose detection demo... Press 'q' to quit")
        self.running = True
        
        frame_count = 0
        
        while self.running:
            # Read frame from camera
            ret, bgr_frame = self.camera_manager.read_frame()
            if not ret or bgr_frame is None:
                print("Error: Could not read frame")
                break
            
            # Convert to RGB for MediaPipe
            ret_rgb, rgb_frame = self.camera_manager.get_rgb_frame()
            if not ret_rgb or rgb_frame is None:
                continue
            
            # Calculate timestamp
            timestamp_ms = int(cv.getTickCount() / cv.getTickFrequency() * 1000)
            
            # Perform pose detection
            self.pose_detector.detect_async(rgb_frame, timestamp_ms)
            
            # Get landmarks and draw them
            landmarks = self.pose_detector.get_latest_landmarks()
            if landmarks:
                bgr_frame = draw_landmarks_on_image(bgr_frame, landmarks)
                
                # Extract posture metrics (for future use)
                metrics = self.landmark_extractor.get_all_metrics(landmarks)
                if metrics:
                    # Print metrics occasionally
                    if frame_count % 30 == 0:  # Every ~1 second at 30fps
                        print(f"Metrics: {metrics}")
            
            # Display frame
            cv.imshow('ErgoSense - Pose Detection Demo', bgr_frame)
            
            # Check for quit
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        self.camera_manager.release()
        self.pose_detector.cleanup()
        cv.destroyAllWindows()
        print("ErgoSense stopped")


def main():
    """Main function"""
    app = ErgoSenseApp()
    
    if app.initialize():
        try:
            app.run_demo()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            app.cleanup()
    else:
        print("Failed to initialize ErgoSense")


if __name__ == "__main__":
    main()
