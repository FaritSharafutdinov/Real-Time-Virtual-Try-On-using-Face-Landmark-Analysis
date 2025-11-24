"""
Accessory Overlay Module
"""

import cv2
import numpy as np


class AccessoryOverlay:
    """Handles overlay and blending of accessories onto video frames"""
    
    def __init__(self):
        pass
    
    def load_accessory(self, image_path):
        """Load accessory image with alpha channel"""
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return None
        
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        elif img.shape[2] == 4:
            pass
        else:
            return None
        
        return img
    
    def overlay_accessory(self, frame, warped_accessory, method='alpha_blend'):
        """Overlay warped accessory onto frame"""
        if warped_accessory is None:
            return frame
        
        if frame.shape[:2] != warped_accessory.shape[:2]:
            h, w = frame.shape[:2]
            warped_accessory = cv2.resize(warped_accessory, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if warped_accessory.shape[2] == 4:
            alpha = warped_accessory[:, :, 3] / 255.0
            overlay_rgb = warped_accessory[:, :, :3]
        else:
            alpha = np.ones((warped_accessory.shape[0], warped_accessory.shape[1]))
            overlay_rgb = warped_accessory[:, :, :3]
        
        mask = alpha > 0.01
        
        if method == 'alpha_blend':
            result = frame.copy()
            alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
            mask_3d = np.stack([mask, mask, mask], axis=2)
            
            blended = (alpha_3d * overlay_rgb + (1 - alpha_3d) * frame).astype(np.uint8)
            result = np.where(mask_3d, blended, frame)
        else:
            alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
            mask_3d = np.stack([mask, mask, mask], axis=2)
            result = cv2.addWeighted(frame, 1 - alpha_3d, overlay_rgb, alpha_3d, 0)
            result = np.where(mask_3d, result, frame)
        
        return result
