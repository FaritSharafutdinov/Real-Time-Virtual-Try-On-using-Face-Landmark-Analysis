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
            anchor_points: List of 4 points [left_outer, left_inner, right_inner, right_outer]
        """
        if landmarks is None:
            return None
        
        # Use eye landmarks to determine glasses position
        left_eye = landmarks.get('left_eye', [])
        right_eye = landmarks.get('right_eye', [])
        
        if len(left_eye) == 0 or len(right_eye) == 0:
            return None
        
        # Calculate eye centers and corners
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        
        # Find outer and inner corners
        left_outer = left_eye[np.argmin(left_eye[:, 0])]  # Leftmost point
        left_inner = left_eye[np.argmax(left_eye[:, 0])]  # Rightmost point
        right_inner = right_eye[np.argmin(right_eye[:, 0])]  # Leftmost point
        right_outer = right_eye[np.argmax(right_eye[:, 0])]  # Rightmost point
        
        # Adjust for glasses frame width
        eye_distance = np.linalg.norm(right_inner - left_inner)
        frame_width = eye_distance * 0.3
        
        anchor_points = np.array([
            left_outer - [frame_width, 0],      # Left outer
            left_inner + [frame_width * 0.5, 0],  # Left inner
            right_inner - [frame_width * 0.5, 0], # Right inner
            right_outer + [frame_width, 0]       # Right outer
        ], dtype=np.float32)
        
        return anchor_points
    
    def get_hat_anchor_points(self, landmarks):
        """
        Get anchor points for positioning hat
        
        Args:
            landmarks: Dictionary of landmarks
            
        Returns:
            anchor_points: List of 4 points for hat positioning
        """
        if landmarks is None:
            return None
        
        forehead_center = landmarks.get('forehead_center', [])
        forehead_left = landmarks.get('forehead_left', [])
        forehead_right = landmarks.get('forehead_right', [])
        
        if len(forehead_center) == 0:
            return None
        
        # Calculate forehead center
        center = np.mean(forehead_center, axis=0)
        
        # Estimate hat width based on face width
        if len(forehead_left) > 0 and len(forehead_right) > 0:
            left_point = np.mean(forehead_left, axis=0)
            right_point = np.mean(forehead_right, axis=0)
            face_width = np.linalg.norm(right_point - left_point)
        else:
            # Fallback: use eye distance
            left_eye = landmarks.get('left_eye', [])
            right_eye = landmarks.get('right_eye', [])
            if len(left_eye) > 0 and len(right_eye) > 0:
                left_center = np.mean(left_eye, axis=0)
                right_center = np.mean(right_eye, axis=0)
                face_width = np.linalg.norm(right_center - left_center) * 1.5
            else:
                face_width = 200  # Default
        
        hat_width = face_width * 1.2
        hat_height = face_width * 0.4
        
        anchor_points = np.array([
            center + [-hat_width/2, -hat_height],  # Top left
            center + [hat_width/2, -hat_height],     # Top right
            center + [hat_width/2, 0],               # Bottom right
            center + [-hat_width/2, 0]               # Bottom left
        ], dtype=np.float32)
        
        return anchor_points

