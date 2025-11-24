"""
Geometric Transformation Module
"""

import cv2
import numpy as np


class GeometricTransformer:
    """Handles geometric transformations for accessory overlay"""
    
    def __init__(self):
        pass
    
    def get_transformation_matrix(self, source_points, destination_points, method='perspective'):
        """Compute transformation matrix from source to destination points"""
        if source_points is None or destination_points is None:
            return None
        
        if len(source_points) != len(destination_points):
            return None
        
        source_points = np.array(source_points, dtype=np.float32)
        destination_points = np.array(destination_points, dtype=np.float32)
        
        if method == 'perspective' and len(source_points) >= 4:
            return cv2.getPerspectiveTransform(source_points, destination_points)
        elif method == 'affine' and len(source_points) >= 3:
            return cv2.getAffineTransform(source_points[:3], destination_points[:3])
        else:
            if len(source_points) >= 3:
                return cv2.getAffineTransform(source_points[:3], destination_points[:3])
            return None
    
    def warp_accessory(self, accessory_img, transformation_matrix, output_size, method='perspective'):
        """Warp accessory image using transformation matrix"""
        if transformation_matrix is None:
            return None
        
        if method == 'perspective':
            return cv2.warpPerspective(
                accessory_img, transformation_matrix, output_size,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
            )
        else:
            if transformation_matrix.shape == (2, 3):
                affine_matrix = np.vstack([transformation_matrix, [0, 0, 1]])
            else:
                affine_matrix = transformation_matrix
            
            return cv2.warpPerspective(
                accessory_img, affine_matrix, output_size,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
            )
    
    def get_accessory_source_points(self, accessory_img, accessory_type='glasses'):
        """Define source reference points for accessory based on its type"""
        h, w = accessory_img.shape[:2]
        
        if accessory_type == 'glasses':
            margin_x = w * 0.1
            margin_y = h * 0.05
            source_points = np.array([
                [margin_x, margin_y],
                [w - margin_x, margin_y],
                [w - margin_x, h - margin_y],
                [margin_x, h - margin_y]
            ], dtype=np.float32)
        elif accessory_type == 'hat':
            margin_x = w * 0.1
            margin_y = h * 0.1
            source_points = np.array([
                [margin_x, margin_y],
                [w - margin_x, margin_y],
                [w - margin_x, h - margin_y],
                [margin_x, h - margin_y]
            ], dtype=np.float32)
        else:
            margin = min(w, h) * 0.05
            source_points = np.array([
                [margin, margin],
                [w - margin, margin],
                [w - margin, h - margin],
                [margin, h - margin]
            ], dtype=np.float32)
        
        return source_points
    
    def compute_smooth_transformation(self, current_points, previous_points=None, alpha=0.7):
        """Smooth transformation to reduce jitter"""
        if previous_points is None:
            return current_points
        
        smoothed = alpha * previous_points + (1 - alpha) * current_points
        return smoothed.astype(np.float32)
