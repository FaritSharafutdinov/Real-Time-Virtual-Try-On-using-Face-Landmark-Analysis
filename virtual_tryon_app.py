"""
Real-Time Virtual Try-On Application
"""

import cv2
import numpy as np
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from face_landmark_detector import FaceLandmarkDetector
from geometric_transformer import GeometricTransformer
from accessory_overlay import AccessoryOverlay


class VirtualTryOnApp:
    """Main application class for real-time virtual try-on"""
    
    def __init__(self, camera_index=0, accessory_dir='accessories'):
        self.camera_index = camera_index
        self.accessory_dir = accessory_dir
        self.cap = None
        
        self.landmark_detector = FaceLandmarkDetector()
        self.transformer = GeometricTransformer()
        self.overlay = AccessoryOverlay()
        
        self.current_accessory = None
        self.current_accessory_type = None
        self.accessories = []
        self.accessory_index = 0
        self.previous_anchor_points = None
        
        self.glasses_list = []
        self.hats_list = []
        self.current_glasses = None
        self.current_hat = None
        self.glasses_index = 0
        self.hats_index = 0
        self.previous_glasses_points = None
        self.previous_hat_points = None
        self.smoothing_alpha = 0.7
        
        self.load_accessories()
    
    def load_accessories(self):
        """Load all accessory images from the accessories directory"""
        if not os.path.exists(self.accessory_dir):
            os.makedirs(self.accessory_dir)
            print(f"Created {self.accessory_dir} directory. Please add accessory images (PNG with transparency).")
            return
        
        png_files = [f for f in os.listdir(self.accessory_dir) if f.lower().endswith('.png')]
        
        if not png_files:
            print(f"No PNG files found in {self.accessory_dir}. Please add accessory images.")
            return
        
        for png_file in png_files:
            accessory_path = os.path.join(self.accessory_dir, png_file)
            accessory_img = self.overlay.load_accessory(accessory_path)
            
            if accessory_img is not None:
                filename_lower = png_file.lower()
                if 'glasses' in filename_lower or 'glasse' in filename_lower:
                    acc_type = 'glasses'
                    self.glasses_list.append({
                        'name': png_file,
                        'image': accessory_img,
                        'type': acc_type,
                        'path': accessory_path
                    })
                elif 'hat' in filename_lower or 'cap' in filename_lower:
                    acc_type = 'hat'
                    self.hats_list.append({
                        'name': png_file,
                        'image': accessory_img,
                        'type': acc_type,
                        'path': accessory_path
                    })
                else:
                    acc_type = 'glasses'
                    self.glasses_list.append({
                        'name': png_file,
                        'image': accessory_img,
                        'type': acc_type,
                        'path': accessory_path
                    })
                
                self.accessories.append({
                    'name': png_file,
                    'image': accessory_img,
                    'type': acc_type,
                    'path': accessory_path
                })
        
        if self.glasses_list:
            self.current_glasses = self.glasses_list[0]
            self.glasses_index = 0
        if self.hats_list:
            self.current_hat = self.hats_list[0]
            self.hats_index = 0
        if self.accessories:
            self.current_accessory = self.accessories[0]
            self.current_accessory_type = self.accessories[0]['type']
        
        print(f"Loaded {len(self.accessories)} accessories")
        print(f"  - Glasses: {len(self.glasses_list)}")
        print(f"  - Hats: {len(self.hats_list)}")
    
    def get_anchor_points(self, landmarks, accessory_type, previous_points=None):
        """Get anchor points based on accessory type"""
        if landmarks is None:
            return None
        
        if accessory_type == 'glasses':
            anchor_points = self.landmark_detector.get_glasses_anchor_points(landmarks)
        elif accessory_type == 'hat':
            anchor_points = self.landmark_detector.get_hat_anchor_points(landmarks)
        else:
            anchor_points = self.landmark_detector.get_glasses_anchor_points(landmarks)
        
        if anchor_points is not None and previous_points is not None:
            anchor_points = self.transformer.compute_smooth_transformation(
                anchor_points, previous_points, self.smoothing_alpha
            )
        elif anchor_points is not None and self.previous_anchor_points is not None:
            anchor_points = self.transformer.compute_smooth_transformation(
                anchor_points, self.previous_anchor_points, self.smoothing_alpha
            )
            self.previous_anchor_points = anchor_points.copy()
        
        return anchor_points
    
    def process_frame(self, frame):
        """Process a single frame: detect face, transform accessories, and overlay"""
        result_frame = frame.copy()
        
        landmarks = self.landmark_detector.detect_landmarks(frame)
        if landmarks is None:
            return frame
        
        if self.current_hat is not None:
            hat_frame = self._process_single_accessory(
                frame, self.current_hat, 'hat', landmarks, self.previous_hat_points
            )
            if hat_frame is not None:
                result_frame = hat_frame
                hat_anchor = self.get_anchor_points(landmarks, 'hat', self.previous_hat_points)
                if hat_anchor is not None:
                    self.previous_hat_points = hat_anchor.copy()
        
        if self.current_glasses is not None:
            glasses_frame = self._process_single_accessory(
                result_frame, self.current_glasses, 'glasses', landmarks, self.previous_glasses_points
            )
            if glasses_frame is not None:
                result_frame = glasses_frame
                glasses_anchor = self.get_anchor_points(landmarks, 'glasses', self.previous_glasses_points)
                if glasses_anchor is not None:
                    self.previous_glasses_points = glasses_anchor.copy()
        
        if self.current_hat is None and self.current_glasses is None and self.current_accessory is not None:
            return self._process_single_accessory(
                frame, self.current_accessory, self.current_accessory_type, landmarks, self.previous_anchor_points
            ) or frame
        
        return result_frame
    
    def _process_single_accessory(self, frame, accessory, accessory_type, landmarks, previous_points=None):
        """Process a single accessory and overlay it on frame"""
        if accessory_type == 'glasses':
            anchor_points = self.landmark_detector.get_glasses_anchor_points(landmarks)
        elif accessory_type == 'hat':
            anchor_points = self.landmark_detector.get_hat_anchor_points(landmarks)
        else:
            return None
        
        if anchor_points is None:
            return None
        
        if previous_points is not None:
            anchor_points = self.transformer.compute_smooth_transformation(
                anchor_points, previous_points, self.smoothing_alpha
            )
        
        accessory_img = accessory['image']
        source_points = self.transformer.get_accessory_source_points(accessory_img, accessory_type)
        transformation_matrix = self.transformer.get_transformation_matrix(
            source_points, anchor_points, method='perspective'
        )
        
        if transformation_matrix is None:
            return None
        
        h, w = frame.shape[:2]
        warped_accessory = self.transformer.warp_accessory(
            accessory_img, transformation_matrix, (w, h), method='perspective'
        )
        
        return self.overlay.overlay_accessory(frame, warped_accessory)
    
    def switch_glasses(self, direction=1):
        """Switch to next/previous glasses"""
        if not self.glasses_list:
            return
        
        self.glasses_index = (self.glasses_index + direction) % len(self.glasses_list)
        self.current_glasses = self.glasses_list[self.glasses_index]
        self.previous_glasses_points = None
        print(f"Switched glasses to: {self.current_glasses['name']}")
    
    def switch_hat(self, direction=1):
        """Switch to next/previous hat"""
        if not self.hats_list:
            return
        
        self.hats_index = (self.hats_index + direction) % len(self.hats_list)
        self.current_hat = self.hats_list[self.hats_index]
        self.previous_hat_points = None
        print(f"Switched hat to: {self.current_hat['name']}")
    
    def random_combination(self):
        """Select random glasses and hat combination"""
        import random
        
        if len(self.glasses_list) > 0:
            self.glasses_index = random.randint(0, len(self.glasses_list) - 1)
            self.current_glasses = self.glasses_list[self.glasses_index]
            self.previous_glasses_points = None
            print(f"Random glasses: {self.current_glasses['name']}")
        
        if len(self.hats_list) > 0:
            self.hats_index = random.randint(0, len(self.hats_list) - 1)
            self.current_hat = self.hats_list[self.hats_index]
            self.previous_hat_points = None
            print(f"Random hat: {self.current_hat['name']}")
    
    def run(self):
        """Main application loop"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n=== Virtual Try-On Application ===")
        print("Controls:")
        print("  'g' or 'G' - Switch glasses")
        print("  'h' or 'H' - Switch hat")
        print("  'r' or 'R' - Random combination (glasses + hat)")
        print("  'q' or ESC - Quit")
        print("===============================\n")
        
        frame_count = 0
        fps_start_time = cv2.getTickCount()
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            result_frame = self.process_frame(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                fps_end_time = cv2.getTickCount()
                fps = 30.0 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
                fps_start_time = fps_end_time
                print(f"FPS: {fps:.2f}")
            
            y_offset = 30
            if self.current_glasses:
                cv2.putText(
                    result_frame, f"Glasses: {self.current_glasses['name']}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                y_offset += 25
            if self.current_hat:
                cv2.putText(
                    result_frame, f"Hat: {self.current_hat['name']}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                y_offset += 25
            elif self.current_accessory:
                cv2.putText(
                    result_frame, f"Accessory: {self.current_accessory['name']}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            
            cv2.putText(
                result_frame, "Press 'g' glasses, 'h' hat, 'r' random, 'q' quit",
                (10, result_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            
            cv2.imshow('Virtual Try-On', result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('g') or key == ord('G'):
                self.switch_glasses(1)
            elif key == ord('h') or key == ord('H'):
                self.switch_hat(1)
            elif key == ord('r') or key == ord('R'):
                self.random_combination()
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Virtual Try-On Application')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--accessories', type=str, default='accessories',
                       help='Directory with accessories (default: accessories)')
    
    args = parser.parse_args()
    
    app = VirtualTryOnApp(camera_index=args.camera, accessory_dir=args.accessories)
    app.run()


if __name__ == '__main__':
    main()
