"""
Geometric Transformation Module
Handles perspective and affine transformations for accessory positioning
"""

import cv2
import numpy as np


class GeometricTransformer:
    """Handles geometric transformations for accessory overlay"""
    
    def __init__(self):
        pass
    
    def get_transformation_matrix(self, source_points, destination_points, method='perspective'):
        """
        Compute transformation matrix from source to destination points
        
        Args:
            source_points: Source points (Nx2 numpy array)
            destination_points: Destination points (Nx2 numpy array)
            method: 'perspective' or 'affine'
            
        Returns:
            transformation_matrix: 3x3 (perspective) or 2x3 (affine) matrix
        """
        if source_points is None or destination_points is None:
            return None
        
        if len(source_points) != len(destination_points):
            return None
        
        source_points = np.array(source_points, dtype=np.float32)
        destination_points = np.array(destination_points, dtype=np.float32)
        
        if method == 'perspective' and len(source_points) >= 4:
            # Perspective transformation requires at least 4 points
            matrix = cv2.getPerspectiveTransform(source_points, destination_points)
            return matrix
        elif method == 'affine' and len(source_points) >= 3:
            # Affine transformation requires at least 3 points
            matrix = cv2.getAffineTransform(source_points[:3], destination_points[:3])
            return matrix
        else:
            # Fallback to affine with available points
            if len(source_points) >= 3:
                matrix = cv2.getAffineTransform(source_points[:3], destination_points[:3])
                return matrix
            return None
    
    def warp_accessory(self, accessory_img, transformation_matrix, output_size, method='perspective'):
        """
        Warp accessory image using transformation matrix
        
        Args:
            accessory_img: Source accessory image (with alpha channel)
            transformation_matrix: Transformation matrix
            output_size: (width, height) of output image
            method: 'perspective' or 'affine'
            
        Returns:
            warped_img: Warped accessory image
        """
        if transformation_matrix is None:
            return None
        
        if method == 'perspective':
            warped = cv2.warpPerspective(
                accessory_img,
                transformation_matrix,
                output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
        else:  # affine
            # For affine, we need to pad the matrix to 3x3
            if transformation_matrix.shape == (2, 3):
                affine_matrix = np.vstack([transformation_matrix, [0, 0, 1]])
            else:
                affine_matrix = transformation_matrix
            
            warped = cv2.warpPerspective(
                accessory_img,
                affine_matrix,
                output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
        
        return warped
    
    def get_accessory_source_points(self, accessory_img, accessory_type='glasses'):
        """
        Define source reference points for accessory based on its type
        
        Args:
            accessory_img: Accessory image
            accessory_type: Type of accessory ('glasses', 'hat', etc.)
            
        Returns:
            source_points: Reference points in source image coordinates
        """
        h, w = accessory_img.shape[:2]
        
        if accessory_type == 'glasses':
            # Define 4 corner points for glasses
            # Assuming glasses are centered horizontally
            margin_x = w * 0.1
            margin_y = h * 0.2
            
            source_points = np.array([
                [margin_x, margin_y],                    # Top left
                [w - margin_x, margin_y],               # Top right
                [w - margin_x, h - margin_y],           # Bottom right
                [margin_x, h - margin_y]                 # Bottom left
            ], dtype=np.float32)
            
        elif accessory_type == 'hat':
            # Define 4 corner points for hat
            margin_x = w * 0.1
            margin_y = h * 0.1
            
            source_points = np.array([
                [margin_x, margin_y],                    # Top left
                [w - margin_x, margin_y],               # Top right
                [w - margin_x, h - margin_y],           # Bottom right
                [margin_x, h - margin_y]                 # Bottom left
            ], dtype=np.float32)
            
        else:
            # Default: use corners with small margin
            margin = min(w, h) * 0.05
            source_points = np.array([
                [margin, margin],                        # Top left
                [w - margin, margin],                    # Top right
                [w - margin, h - margin],                # Bottom right
                [margin, h - margin]                     # Bottom left
            ], dtype=np.float32)
        
        return source_points
    
    def compute_smooth_transformation(self, current_points, previous_points=None, alpha=0.7):
        """
        Smooth transformation to reduce jitter
        
        Args:
            current_points: Current landmark points
            previous_points: Previous frame's points (for smoothing)
            alpha: Smoothing factor (0-1), higher = more smoothing
            
        Returns:
            smoothed_points: Smoothed points
        """
        if previous_points is None:
            return current_points
        
        # Exponential moving average
        smoothed = alpha * previous_points + (1 - alpha) * current_points
        return smoothed.astype(np.float32)

