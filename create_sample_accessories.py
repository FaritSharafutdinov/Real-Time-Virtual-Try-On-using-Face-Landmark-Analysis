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
    
    # Draw glasses frame
    frame_thickness = 8
    frame_color = (100, 100, 200, 255)  # Blue glasses
    
    # Left lens
    cv2.ellipse(img, (width//4, height//2), (60, 50), 0, 0, 360, frame_color, frame_thickness)
    # Right lens
    cv2.ellipse(img, (3*width//4, height//2), (60, 50), 0, 0, 360, frame_color, frame_thickness)
    
    # Bridge
    cv2.line(img, (width//2 - 20, height//2), (width//2 + 20, height//2), frame_color, frame_thickness)
    
    # Temples (side pieces)
    cv2.line(img, (width//4 - 60, height//2), (0, height//2), frame_color, frame_thickness)
    cv2.line(img, (3*width//4 + 60, height//2), (width, height//2), frame_color, frame_thickness)
    
    # Fill lenses with semi-transparent blue
    lens_color = (150, 150, 255, 180)
    cv2.ellipse(img, (width//4, height//2), (55, 45), 0, 0, 360, lens_color, -1)
    cv2.ellipse(img, (3*width//4, height//2), (55, 45), 0, 0, 360, lens_color, -1)
    
    return img


def create_hat_image(width=300, height=200):
    """Create a simple hat/cap image"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Hat color
    hat_color = (50, 50, 50, 255)  # Dark gray
    brim_color = (30, 30, 30, 255)  # Darker gray
    
    # Hat crown (top part)
    crown_points = np.array([
        [width//2, 20],
        [width//2 - 80, 60],
        [width//2 + 80, 60]
    ], np.int32)
    cv2.fillPoly(img, [crown_points], hat_color)
    
    # Hat body
    cv2.ellipse(img, (width//2, 80), (90, 50), 0, 0, 180, hat_color, -1)
    cv2.rectangle(img, (width//2 - 90, 80), (width//2 + 90, 120), hat_color, -1)
    
    # Brim
    cv2.ellipse(img, (width//2, 120), (100, 20), 0, 0, 180, brim_color, -1)
    
    return img


def create_sunglasses_image(width=400, height=150):
    """Create sunglasses image"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Dark frame
    frame_thickness = 10
    frame_color = (20, 20, 20, 255)  # Black frame
    
    # Left lens (larger)
    cv2.ellipse(img, (width//4, height//2), (70, 55), 0, 0, 360, frame_color, frame_thickness)
    # Right lens
    cv2.ellipse(img, (3*width//4, height//2), (70, 55), 0, 0, 360, frame_color, frame_thickness)
    
    # Bridge
    cv2.line(img, (width//2 - 25, height//2), (width//2 + 25, height//2), frame_color, frame_thickness)
    
    # Temples
    cv2.line(img, (width//4 - 70, height//2), (0, height//2), frame_color, frame_thickness)
    cv2.line(img, (3*width//4 + 70, height//2), (width, height//2), frame_color, frame_thickness)
    
    # Dark lenses
    lens_color = (10, 10, 10, 220)  # Very dark, semi-transparent
    cv2.ellipse(img, (width//4, height//2), (65, 50), 0, 0, 360, lens_color, -1)
    cv2.ellipse(img, (3*width//4, height//2), (65, 50), 0, 0, 360, lens_color, -1)
    
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

