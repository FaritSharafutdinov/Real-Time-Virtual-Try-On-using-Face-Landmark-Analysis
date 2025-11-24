"""
Camera availability checker utility
"""

import cv2
import sys

print("Checking available cameras...\n")

available_cameras = []

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✓ Camera {i}: available ({width}x{height})")
            available_cameras.append(i)
        else:
            print(f"✗ Camera {i}: opened but cannot capture frame")
        cap.release()
    else:
        print(f"✗ Camera {i}: unavailable")

print(f"\nFound {len(available_cameras)} available camera(s)")
if available_cameras:
    print(f"Recommended camera: {available_cameras[0]}")
    print(f"\nTo run the application with this camera:")
    print(f"  python virtual_tryon_app.py --camera {available_cameras[0]}")
else:
    print("\n⚠ No available cameras!")
    print("\nOn macOS, grant camera permission:")
    print("1. System Settings → Privacy & Security → Camera")
    print("2. Enable access for Terminal or Python")
    print("\nRestart the application after granting permission.")
