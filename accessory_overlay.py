"""
Accessory Overlay Module
Handles blending and overlaying accessories onto video frames
"""

import cv2
import numpy as np


class AccessoryOverlay:
    """Handles overlay and blending of accessories onto video frames"""
    
    def __init__(self):
        pass
    
    def load_accessory(self, image_path):
        """
        Load accessory image with alpha channel
        
        Args:
            image_path: Path to PNG image with transparency
            
        Returns:
            accessory_img: BGRA image (4 channels)
        """
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return None
        
        # Convert to BGRA if needed
        if img.shape[2] == 3:
            # Add alpha channel (fully opaque)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        elif img.shape[2] == 4:
            # Already has alpha channel
            pass
        else:
            return None
        
        return img
    
    def alpha_blend(self, background, overlay, alpha_mask=None):
        """
        Blend overlay onto background using alpha channel
        
        Args:
            background: Background image (BGR)
            overlay: Overlay image (BGRA)
            alpha_mask: Optional custom alpha mask
            
        Returns:
            blended: Blended image
        """
        if overlay is None:
            return background
        
        # Extract alpha channel
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3] / 255.0
        elif alpha_mask is not None:
            alpha = alpha_mask / 255.0
        else:
            # No alpha channel, use full opacity
            alpha = np.ones((overlay.shape[0], overlay.shape[1]))
        
        # Convert alpha to 3 channels for broadcasting
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        
        # Extract RGB channels from overlay
        overlay_rgb = overlay[:, :, :3]
        
        # Ensure sizes match
        if background.shape[:2] != overlay_rgb.shape[:2]:
            # Resize overlay to match background
            h, w = background.shape[:2]
            overlay_rgb = cv2.resize(overlay_rgb, (w, h))
            alpha = cv2.resize(alpha, (w, h))
            alpha = np.stack([alpha[:, :, 0], alpha[:, :, 0], alpha[:, :, 0]], axis=2)
        
        # Perform alpha blending
        blended = (alpha * overlay_rgb + (1 - alpha) * background).astype(np.uint8)
        
        return blended
    
    def overlay_accessory(self, frame, warped_accessory, method='alpha_blend'):
        """
        Overlay warped accessory onto frame
        
        Args:
            frame: Background video frame (BGR)
            warped_accessory: Warped accessory image (BGRA)
            method: Blending method ('alpha_blend' or 'weighted')
            
        Returns:
            result: Frame with accessory overlaid
        """
        if warped_accessory is None:
            return frame
        
        # Ensure sizes match
        if frame.shape[:2] != warped_accessory.shape[:2]:
            h, w = frame.shape[:2]
            warped_accessory = cv2.resize(warped_accessory, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Extract alpha channel
        if warped_accessory.shape[2] == 4:
            alpha = warped_accessory[:, :, 3] / 255.0
            overlay_rgb = warped_accessory[:, :, :3]
        else:
            # No alpha channel
            alpha = np.ones((warped_accessory.shape[0], warped_accessory.shape[1]))
            overlay_rgb = warped_accessory[:, :, :3]
        
        # Create mask for non-zero alpha regions (where accessory exists)
        mask = alpha > 0.01
        
        if method == 'alpha_blend':
            # Only blend where mask is True
            result = frame.copy()
            alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
            mask_3d = np.stack([mask, mask, mask], axis=2)
            
            # Blend only in masked regions
            blended = (alpha_3d * overlay_rgb + (1 - alpha_3d) * frame).astype(np.uint8)
            result = np.where(mask_3d, blended, frame)
        else:
            # Weighted blending
            alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
            mask_3d = np.stack([mask, mask, mask], axis=2)
            result = cv2.addWeighted(frame, 1 - alpha_3d, overlay_rgb, alpha_3d, 0)
            # Apply mask
            result = np.where(mask_3d, result, frame)
        
        return result
    
    def enhance_blending(self, frame, overlay, blur_kernel=3):
        """
        Enhance blending with edge smoothing
        
        Args:
            frame: Background frame
            overlay: Overlay image
            blur_kernel: Kernel size for edge blurring
            
        Returns:
            enhanced: Enhanced blended result
        """
        if overlay.shape[2] != 4:
            return self.overlay_accessory(frame, overlay)
        
        # Extract alpha channel
        alpha = overlay[:, :, 3]
        
        # Blur alpha channel edges for smoother blending
        alpha_blurred = cv2.GaussianBlur(alpha, (blur_kernel, blur_kernel), 0)
        
        # Create new overlay with blurred alpha
        overlay_enhanced = overlay.copy()
        overlay_enhanced[:, :, 3] = alpha_blurred
        
        return self.overlay_accessory(frame, overlay_enhanced)

