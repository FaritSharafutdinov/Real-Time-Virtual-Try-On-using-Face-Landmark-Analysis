"""
Script to create sample accessory images for testing
Creates simple PNG images with transparency
"""

import cv2
import numpy as np
import os


def create_glasses_image(width=400, height=150):
    """Create a simple glasses image"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Draw glasses frame - thicker and more visible
    frame_thickness = 12
    frame_color = (80, 80, 200, 255)  # Blue frame (BGR)
    
    # Left lens - filled frame
    cv2.ellipse(img, (width//4, height//2), (65, 55), 0, 0, 360, frame_color, frame_thickness)
    # Right lens
    cv2.ellipse(img, (3*width//4, height//2), (65, 55), 0, 0, 360, frame_color, frame_thickness)
    
    # Bridge - thicker
    bridge_thickness = 10
    cv2.line(img, (width//2 - 25, height//2), (width//2 + 25, height//2), frame_color, bridge_thickness)
    
    # Temples (side pieces) - thicker and longer
    temple_thickness = 8
    cv2.line(img, (width//4 - 70, height//2), (0, height//2), frame_color, temple_thickness)
    cv2.line(img, (3*width//4 + 70, height//2), (width, height//2), frame_color, temple_thickness)
    
    # Fill lenses with semi-transparent blue (more visible)
    lens_color = (180, 180, 255, 200)  # Lighter blue, more opaque
    cv2.ellipse(img, (width//4, height//2), (58, 48), 0, 0, 360, lens_color, -1)
    cv2.ellipse(img, (3*width//4, height//2), (58, 48), 0, 0, 360, lens_color, -1)
    
    # Add frame highlights for 3D effect
    highlight_color = (120, 120, 255, 255)
    cv2.ellipse(img, (width//4, height//2 - 10), (50, 40), 0, 0, 180, highlight_color, 3)
    cv2.ellipse(img, (3*width//4, height//2 - 10), (50, 40), 0, 0, 180, highlight_color, 3)
    
    return img


def create_hat_image(width=300, height=200):
    """Create a simple hat/cap image"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Hat color - darker and more visible
    hat_color = (40, 40, 40, 255)  # Dark gray (BGR)
    brim_color = (20, 20, 20, 255)  # Darker gray
    crown_color = (60, 60, 60, 255)  # Lighter gray for crown
    
    # Hat crown (top part) - filled
    crown_points = np.array([
        [width//2, 15],
        [width//2 - 85, 65],
        [width//2 + 85, 65]
    ], np.int32)
    cv2.fillPoly(img, [crown_points], crown_color)
    
    # Hat body - filled ellipse and rectangle
    cv2.ellipse(img, (width//2, 85), (95, 55), 0, 0, 180, hat_color, -1)
    cv2.rectangle(img, (width//2 - 95, 85), (width//2 + 95, 125), hat_color, -1)
    
    # Brim - filled and wider
    cv2.ellipse(img, (width//2, 125), (110, 25), 0, 0, 180, brim_color, -1)
    cv2.rectangle(img, (width//2 - 110, 125), (width//2 + 110, 140), brim_color, -1)
    
    # Add some details - band around hat
    band_color = (80, 80, 80, 255)
    cv2.rectangle(img, (width//2 - 95, 100), (width//2 + 95, 110), band_color, -1)
    
    return img


def create_sunglasses_image(width=400, height=150):
    """Create sunglasses image"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Dark frame - thicker
    frame_thickness = 14
    frame_color = (15, 15, 15, 255)  # Black frame (BGR)
    
    # Left lens (larger) - filled frame
    cv2.ellipse(img, (width//4, height//2), (75, 60), 0, 0, 360, frame_color, frame_thickness)
    # Right lens
    cv2.ellipse(img, (3*width//4, height//2), (75, 60), 0, 0, 360, frame_color, frame_thickness)
    
    # Bridge - thicker
    bridge_thickness = 12
    cv2.line(img, (width//2 - 30, height//2), (width//2 + 30, height//2), frame_color, bridge_thickness)
    
    # Temples - thicker
    temple_thickness = 10
    cv2.line(img, (width//4 - 75, height//2), (0, height//2), frame_color, temple_thickness)
    cv2.line(img, (3*width//4 + 75, height//2), (width, height//2), frame_color, temple_thickness)
    
    # Dark lenses - more opaque
    lens_color = (5, 5, 5, 240)  # Very dark, more opaque
    cv2.ellipse(img, (width//4, height//2), (68, 53), 0, 0, 360, lens_color, -1)
    cv2.ellipse(img, (3*width//4, height//2), (68, 53), 0, 0, 360, lens_color, -1)
    
    # Add gradient effect on lenses
    gradient_color = (20, 20, 20, 180)
    cv2.ellipse(img, (width//4, height//2 - 15), (50, 35), 0, 0, 180, gradient_color, -1)
    cv2.ellipse(img, (3*width//4, height//2 - 15), (50, 35), 0, 0, 180, gradient_color, -1)
    
    return img


def main():
    """Create sample accessories"""
    accessories_dir = 'accessories'
    
    # Create directory if it doesn't exist
    if not os.path.exists(accessories_dir):
        os.makedirs(accessories_dir)
        print(f"Created {accessories_dir} directory")
    
    # Create sample accessories
    print("Creating sample accessories...")
    
    # Glasses
    glasses = create_glasses_image()
    cv2.imwrite(os.path.join(accessories_dir, 'glasses.png'), glasses)
    print("Created: glasses.png")
    
    # Sunglasses
    sunglasses = create_sunglasses_image()
    cv2.imwrite(os.path.join(accessories_dir, 'sunglasses.png'), sunglasses)
    print("Created: sunglasses.png")
    
    # Hat
    hat = create_hat_image()
    cv2.imwrite(os.path.join(accessories_dir, 'hat.png'), hat)
    print("Created: hat.png")
    
    print(f"\nSample accessories created in '{accessories_dir}' directory")
    print("You can now run the application: python virtual_tryon_app.py")


if __name__ == '__main__':
    main()

