"""
Face Landmark Detector Module
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
        
        self.LANDMARK_INDICES = {
            'face_outline': [10, 151, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                            21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 
                            148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 363, 360],
            'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'glasses_left': [33, 7, 163, 144, 145, 153, 154, 155, 133],
            'glasses_right': [362, 382, 381, 380, 374, 373, 390, 249, 263],
            'glasses_bridge': [168, 8, 6, 197, 195, 5, 4],
            'forehead_center': [10, 151, 9],
            'forehead_left': [21, 162, 127],
            'forehead_right': [234, 93, 132]
        }
    
    def detect_landmarks(self, frame):
        """Detect face and extract landmarks from a frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
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
        
        all_landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            all_landmarks.append([x, y])
        landmarks['all'] = np.array(all_landmarks, dtype=np.float32)
        
        return landmarks
    
    def get_glasses_anchor_points(self, landmarks):
        """Get anchor points for positioning glasses"""
        if landmarks is None:
            return None
        
        left_eye = landmarks.get('left_eye', [])
        right_eye = landmarks.get('right_eye', [])
        
        if len(left_eye) == 0 or len(right_eye) == 0:
            return None
        
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        eye_center = (left_eye_center + right_eye_center) / 2
        
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        frame_width = eye_distance * 1.8
        frame_height = eye_distance * 0.9
        
        y_offset = -eye_distance * 0.1
        
        anchor_points = np.array([
            [eye_center[0] - frame_width/2, eye_center[1] + y_offset - frame_height/2],
            [eye_center[0] + frame_width/2, eye_center[1] + y_offset - frame_height/2],
            [eye_center[0] + frame_width/2, eye_center[1] + y_offset + frame_height/2],
            [eye_center[0] - frame_width/2, eye_center[1] + y_offset + frame_height/2]
        ], dtype=np.float32)
        
        return anchor_points
    
    def get_hat_anchor_points(self, landmarks):
        """Get anchor points for positioning hat"""
        if landmarks is None:
            return None
        
        forehead_center = landmarks.get('forehead_center', [])
        forehead_left = landmarks.get('forehead_left', [])
        forehead_right = landmarks.get('forehead_right', [])
        left_eye = landmarks.get('left_eye', [])
        right_eye = landmarks.get('right_eye', [])
        
        if len(forehead_center) == 0:
            return None
        
        forehead_center_point = np.mean(forehead_center, axis=0)
        
        eye_center_y = None
        if len(left_eye) > 0 and len(right_eye) > 0:
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            eye_center = (left_eye_center + right_eye_center) / 2
            eye_center_y = eye_center[1]
        
        face_outline = landmarks.get('face_outline', [])
        head_width = None
        
        if len(face_outline) > 0:
            face_outline_array = np.array(face_outline)
            leftmost_idx = np.argmin(face_outline_array[:, 0])
            rightmost_idx = np.argmax(face_outline_array[:, 0])
            leftmost_point = face_outline_array[leftmost_idx]
            rightmost_point = face_outline_array[rightmost_idx]
            head_width = np.linalg.norm(rightmost_point - leftmost_point)
        
        if head_width is None or head_width < 50:
            if len(forehead_left) > 0 and len(forehead_right) > 0:
                left_point = np.mean(forehead_left, axis=0)
                right_point = np.mean(forehead_right, axis=0)
                head_width = np.linalg.norm(right_point - left_point) * 1.4
            elif len(left_eye) > 0 and len(right_eye) > 0:
                left_center = np.mean(left_eye, axis=0)
                right_center = np.mean(right_eye, axis=0)
                head_width = np.linalg.norm(right_center - left_center) * 2.2
            else:
                head_width = 200
        
        hat_width = head_width * 1.1
        hat_height = head_width * 0.5
        
        vertical_offset = 0
        if eye_center_y is not None:
            eye_to_forehead_dist = abs(forehead_center_point[1] - eye_center_y)
            vertical_offset = eye_to_forehead_dist * 0.2
        
        hat_center = forehead_center_point.copy()
        hat_center[1] += vertical_offset
        
        hat_top_y = hat_center[1] - hat_height * 0.6
        hat_bottom_y = forehead_center_point[1] + vertical_offset
        
        anchor_points = np.array([
            [hat_center[0] - hat_width/2, hat_top_y],
            [hat_center[0] + hat_width/2, hat_top_y],
            [hat_center[0] + hat_width/2, hat_bottom_y],
            [hat_center[0] - hat_width/2, hat_bottom_y]
        ], dtype=np.float32)
        
        return anchor_points
