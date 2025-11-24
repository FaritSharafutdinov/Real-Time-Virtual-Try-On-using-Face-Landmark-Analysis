"""
Real-Time Virtual Try-On Application
Main application class that integrates all modules
"""

import cv2
import numpy as np
import os
from face_landmark_detector import FaceLandmarkDetector
from geometric_transformer import GeometricTransformer
from accessory_overlay import AccessoryOverlay


class VirtualTryOnApp:
    """Main application class for real-time virtual try-on"""
    
    def __init__(self, camera_index=0, accessory_dir='accessories'):
        """
        Initialize the application
        
        Args:
            camera_index: Webcam index (default 0)
            accessory_dir: Directory containing accessory images
        """
        self.camera_index = camera_index
        self.accessory_dir = accessory_dir
        self.cap = None
        
        # Initialize modules
        self.landmark_detector = FaceLandmarkDetector()
        self.transformer = GeometricTransformer()
        self.overlay = AccessoryOverlay()
        
        # State variables
        self.current_accessory = None
        self.current_accessory_type = None
        self.accessories = []
        self.accessory_index = 0
        self.previous_anchor_points = None
        self.smoothing_alpha = 0.7
        
        # Load accessories
        self.load_accessories()
    
    def load_accessories(self):
        """Load all accessory images from the accessories directory"""
        if not os.path.exists(self.accessory_dir):
            os.makedirs(self.accessory_dir)
            print(f"Created {self.accessory_dir} directory. Please add accessory images (PNG with transparency).")
            return
        
        # Find all PNG files
        png_files = [f for f in os.listdir(self.accessory_dir) if f.lower().endswith('.png')]
        
        if not png_files:
            print(f"No PNG files found in {self.accessory_dir}. Please add accessory images.")
            return
        
        # Load accessories
        for png_file in png_files:
            accessory_path = os.path.join(self.accessory_dir, png_file)
            accessory_img = self.overlay.load_accessory(accessory_path)
            
            if accessory_img is not None:
                # Determine accessory type from filename
                filename_lower = png_file.lower()
                if 'glasses' in filename_lower or 'glasse' in filename_lower:
                    acc_type = 'glasses'
                elif 'hat' in filename_lower or 'cap' in filename_lower:
                    acc_type = 'hat'
                else:
                    acc_type = 'glasses'  # Default
                
                self.accessories.append({
                    'name': png_file,
                    'image': accessory_img,
                    'type': acc_type,
                    'path': accessory_path
                })
        
        if self.accessories:
            self.current_accessory = self.accessories[0]
            self.current_accessory_type = self.accessories[0]['type']
            print(f"Loaded {len(self.accessories)} accessories")
        else:
            print("No valid accessories loaded")
    
    def get_anchor_points(self, landmarks, accessory_type):
        """
        Get anchor points based on accessory type
        
        Args:
            landmarks: Detected landmarks
            accessory_type: Type of accessory
            
        Returns:
            anchor_points: Anchor points for transformation
        """
        if landmarks is None:
            return None
        
        if accessory_type == 'glasses':
            anchor_points = self.landmark_detector.get_glasses_anchor_points(landmarks)
        elif accessory_type == 'hat':
            anchor_points = self.landmark_detector.get_hat_anchor_points(landmarks)
        else:
            # Default to glasses
            anchor_points = self.landmark_detector.get_glasses_anchor_points(landmarks)
        
        # Apply smoothing
        if anchor_points is not None and self.previous_anchor_points is not None:
            anchor_points = self.transformer.compute_smooth_transformation(
                anchor_points,
                self.previous_anchor_points,
                self.smoothing_alpha
            )
        
        self.previous_anchor_points = anchor_points.copy() if anchor_points is not None else None
        
        return anchor_points
    
    def process_frame(self, frame):
        """
        Process a single frame: detect face, transform accessory, and overlay
        
        Args:
            frame: Input video frame
            
        Returns:
            result_frame: Frame with accessory overlaid
        """
        if self.current_accessory is None:
            return frame
        
        # Detect landmarks
        landmarks = self.landmark_detector.detect_landmarks(frame)
        
        if landmarks is None:
            # No face detected, return original frame
            return frame
        
        # Get anchor points
        anchor_points = self.get_anchor_points(landmarks, self.current_accessory_type)
        
        if anchor_points is None:
            return frame
        
        # Get source points from accessory
        accessory_img = self.current_accessory['image']
        source_points = self.transformer.get_accessory_source_points(
            accessory_img,
            self.current_accessory_type
        )
        
        # Compute transformation matrix
        transformation_matrix = self.transformer.get_transformation_matrix(
            source_points,
            anchor_points,
            method='perspective'
        )
        
        if transformation_matrix is None:
            return frame
        
        # Warp accessory
        h, w = frame.shape[:2]
        warped_accessory = self.transformer.warp_accessory(
            accessory_img,
            transformation_matrix,
            (w, h),
            method='perspective'
        )
        
        # Overlay accessory
        result_frame = self.overlay.overlay_accessory(frame, warped_accessory)
        
        return result_frame
    
    def switch_accessory(self, direction=1):
        """
        Switch to next/previous accessory
        
        Args:
            direction: 1 for next, -1 for previous
        """
        if not self.accessories:
            return
        
        self.accessory_index = (self.accessory_index + direction) % len(self.accessories)
        self.current_accessory = self.accessories[self.accessory_index]
        self.current_accessory_type = self.current_accessory['type']
        self.previous_anchor_points = None  # Reset smoothing
        print(f"Switched to: {self.current_accessory['name']}")
    
    def run(self):
        """Main application loop"""
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n=== Virtual Try-On Application ===")
        print("Controls:")
        print("  'n' or 'N' - Next accessory")
        print("  'p' or 'P' - Previous accessory")
        print("  'q' or ESC - Quit")
        print("  's' or 'S' - Toggle smoothing")
        print("===============================\n")
        
        frame_count = 0
        fps_start_time = cv2.getTickCount()
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            result_frame = self.process_frame(frame)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps_end_time = cv2.getTickCount()
                fps = 30.0 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
                fps_start_time = fps_end_time
                print(f"FPS: {fps:.2f}")
            
            # Display current accessory name
            if self.current_accessory:
                cv2.putText(
                    result_frame,
                    f"Accessory: {self.current_accessory['name']}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Display instructions
            cv2.putText(
                result_frame,
                "Press 'n' for next, 'p' for prev, 'q' to quit",
                (10, result_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Show frame
            cv2.imshow('Virtual Try-On', result_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('n') or key == ord('N'):
                self.switch_accessory(1)
            elif key == ord('p') or key == ord('P'):
                self.switch_accessory(-1)
            elif key == ord('s') or key == ord('S'):
                self.smoothing_alpha = 0.3 if self.smoothing_alpha > 0.5 else 0.7
                print(f"Smoothing alpha: {self.smoothing_alpha}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed")


def main():
    """Main entry point"""
    app = VirtualTryOnApp(camera_index=0, accessory_dir='accessories')
    app.run()


if __name__ == '__main__':
    main()

