"""
Face Landmark Detector Module
Detects faces and extracts facial landmarks using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np


class FaceLandmarkDetector:
    """Detects faces and extracts facial landmarks using MediaPipe Face Mesh"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key landmark indices for different accessories
        # MediaPipe Face Mesh has 468 landmarks
        # Indices for face outline, eyes, nose, etc.
        self.LANDMARK_INDICES = {
            'face_outline': [10, 151, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                            21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 
                            148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 363, 360],
            'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            # Key points for glasses positioning
            'glasses_left': [33, 7, 163, 144, 145, 153, 154, 155, 133],
            'glasses_right': [362, 382, 381, 380, 374, 373, 390, 249, 263],
            'glasses_bridge': [168, 8, 6, 197, 195, 5, 4],
            # Key points for hat positioning
            'forehead_center': [10, 151, 9],
            'forehead_left': [21, 162, 127],
            'forehead_right': [234, 93, 132]
        }
    
    def detect_landmarks(self, frame):
        """
        Detect face and extract landmarks from a frame
        
        Args:
            frame: BGR image frame from webcam
            
        Returns:
            landmarks: Dictionary with landmark coordinates or None if no face detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first face (assuming single face)
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract landmark coordinates
        h, w = frame.shape[:2]
        landmarks = {}
        
        for key, indices in self.LANDMARK_INDICES.items():
            points = []
            for idx in indices:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append([x, y])
            landmarks[key] = np.array(points, dtype=np.float32)
        
        # Also store all landmarks for general use
        all_landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            all_landmarks.append([x, y])
        landmarks['all'] = np.array(all_landmarks, dtype=np.float32)
        
        return landmarks
    
    def get_glasses_anchor_points(self, landmarks):
        """
        Get anchor points for positioning glasses
        
        Args:
            landmarks: Dictionary of landmarks
            
        Returns:
            anchor_points: List of 4 points [top-left, top-right, bottom-right, bottom-left]
        """
        if landmarks is None:
            return None
        
        # Use eye landmarks to determine glasses position
        left_eye = landmarks.get('left_eye', [])
        right_eye = landmarks.get('right_eye', [])
        
        if len(left_eye) == 0 or len(right_eye) == 0:
            return None
        
        # Calculate eye centers
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        
        # Calculate the center point between eyes
        eye_center = (left_eye_center + right_eye_center) / 2
        
        # Calculate eye distance and frame dimensions
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        frame_width = eye_distance * 1.8  # Wider than eye distance
        frame_height = eye_distance * 0.6  # Height of glasses frame
        
        # Calculate vertical position (slightly above eye center)
        y_offset = -eye_distance * 0.1
        
        # Create 4 corner points for glasses rectangle
        # Order: top-left, top-right, bottom-right, bottom-left
        anchor_points = np.array([
            [eye_center[0] - frame_width/2, eye_center[1] + y_offset - frame_height/2],  # Top left
            [eye_center[0] + frame_width/2, eye_center[1] + y_offset - frame_height/2],  # Top right
            [eye_center[0] + frame_width/2, eye_center[1] + y_offset + frame_height/2],  # Bottom right
            [eye_center[0] - frame_width/2, eye_center[1] + y_offset + frame_height/2]   # Bottom left
        ], dtype=np.float32)
        
        return anchor_points
    
    def get_hat_anchor_points(self, landmarks):
        """
        Get anchor points for positioning hat
        
        Args:
            landmarks: Dictionary of landmarks
            
        Returns:
            anchor_points: List of 4 points [top-left, top-right, bottom-right, bottom-left]
        """
        if landmarks is None:
            return None
        
        forehead_center = landmarks.get('forehead_center', [])
        forehead_left = landmarks.get('forehead_left', [])
        forehead_right = landmarks.get('forehead_right', [])
        left_eye = landmarks.get('left_eye', [])
        right_eye = landmarks.get('right_eye', [])
        
        if len(forehead_center) == 0:
            return None
        
        # Calculate forehead center
        forehead_center_point = np.mean(forehead_center, axis=0)
        
        # Calculate eye center for reference
        eye_center_y = None
        if len(left_eye) > 0 and len(right_eye) > 0:
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            eye_center = (left_eye_center + right_eye_center) / 2
            eye_center_y = eye_center[1]
        
        # Calculate head width more accurately using face outline
        # Use the widest points of the face outline for better head width estimation
        face_outline = landmarks.get('face_outline', [])
        head_width = None
        
        if len(face_outline) > 0:
            # Find the leftmost and rightmost points of the face outline
            face_outline_array = np.array(face_outline)
            leftmost_idx = np.argmin(face_outline_array[:, 0])
            rightmost_idx = np.argmax(face_outline_array[:, 0])
            leftmost_point = face_outline_array[leftmost_idx]
            rightmost_point = face_outline_array[rightmost_idx]
            head_width = np.linalg.norm(rightmost_point - leftmost_point)
        
        # Fallback methods if face outline is not available
        if head_width is None or head_width < 50:  # Sanity check
            if len(forehead_left) > 0 and len(forehead_right) > 0:
                left_point = np.mean(forehead_left, axis=0)
                right_point = np.mean(forehead_right, axis=0)
                head_width = np.linalg.norm(right_point - left_point) * 1.4
            elif len(left_eye) > 0 and len(right_eye) > 0:
                left_center = np.mean(left_eye, axis=0)
                right_center = np.mean(right_eye, axis=0)
                head_width = np.linalg.norm(right_center - left_center) * 2.2
            else:
                head_width = 200  # Default
        
        # Hat width should match head width (maybe slightly wider for better coverage)
        hat_width = head_width * 1.1  # Slightly wider than head for natural look
        hat_height = head_width * 0.5  # Proportional height
        
        # Calculate vertical offset to position hat on top of head (at hair level)
        # Move hat much higher up from forehead
        vertical_offset = 0
        if eye_center_y is not None:
            # Distance from eyes to forehead
            eye_to_forehead_dist = abs(forehead_center_point[1] - eye_center_y)
            # Move hat up significantly - 2.5x the distance from eyes to forehead
            # This puts it at the top of the head / hair level
            vertical_offset = -eye_to_forehead_dist * 2.5  # Negative Y means up
        
        # Center point for hat (moved way up from forehead to top of head)
        hat_center = forehead_center_point.copy()
        hat_center[1] += vertical_offset
        
        # Position hat on top of head (at hair level)
        # Order: top-left, top-right, bottom-right, bottom-left
        # Bottom of hat should be well above forehead, top should be at hair level
        hat_bottom_y = forehead_center_point[1] + vertical_offset * 0.3  # Bottom is also raised
        anchor_points = np.array([
            [hat_center[0] - hat_width/2, hat_center[1] - hat_height],  # Top left (at hair level)
            [hat_center[0] + hat_width/2, hat_center[1] - hat_height],  # Top right (at hair level)
            [hat_center[0] + hat_width/2, hat_bottom_y],                # Bottom right (above forehead)
            [hat_center[0] - hat_width/2, hat_bottom_y]                 # Bottom left (above forehead)
        ], dtype=np.float32)
        
        return anchor_points

