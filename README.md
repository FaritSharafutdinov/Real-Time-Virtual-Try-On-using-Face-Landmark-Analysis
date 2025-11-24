# Real-Time Virtual Try-On using Face Landmark Analysis

Real-time AR application for virtual try-on of 2D accessories (glasses, hats) using webcam and face landmark detection.

## Description

The system captures video from a webcam, detects faces, and extracts facial landmarks. Based on these landmarks, accessories are positioned and transformed to follow head movements naturally. Geometric transformations and alpha-blending are used for realistic overlay.

## Features

- Real-time face detection using MediaPipe Face Mesh
- 468 facial landmark points extraction
- Perspective and affine transformations for accessory positioning
- Alpha-blending for realistic overlay
- Smoothing for stable tracking
- Multiple accessories support (glasses and hats simultaneously)
- Random combination generator

## Project Structure

```
├── face_landmark_detector.py    # Face detection and landmark extraction
├── geometric_transformer.py     # Geometric transformations
├── accessory_overlay.py          # Overlay and blending
├── virtual_tryon_app.py         # Main application
├── check_camera.py               # Camera availability checker
├── create_sample_accessories.py # Sample accessory generator
├── requirements.txt              # Dependencies
├── accessories/                  # Accessory images (PNG)
└── README.md
```

## Requirements

- Python 3.8 - 3.12 (MediaPipe doesn't support Python 3.13+)
- Webcam
- Windows/Linux/macOS

Note: If you have Python 3.13+, use Python 3.11 or 3.12. You can install it via Homebrew or use a virtual environment.

## Installation

1. Check Python version:
   ```bash
   python --version
   ```
   Should be 3.8 to 3.12.

2. Create virtual environment (recommended):
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Add accessories:
   Place PNG files with transparent background in `accessories/` directory.
   - Glasses: filename must contain "glasses"
   - Hats: filename must contain "hat" or "cap"
   
   Examples: `glasses_1.png`, `hat_1.png`, `sunglasses.png`

## Usage

Run the application:
```bash
python virtual_tryon_app.py
```

Or with specific camera:
```bash
python virtual_tryon_app.py --camera 0
```

Check available cameras:
```bash
python check_camera.py
```

### Controls

- **'g' or 'G'** - Switch glasses
- **'h' or 'H'** - Switch hat
- **'r' or 'R'** - Random combination (glasses + hat)
- **'q' or ESC** - Quit

### Accessory Requirements

- Format: PNG with alpha channel (transparent background)
- Recommended size: 300-500 pixels width
- Proper orientation (glasses horizontal, hat from top)

## Architecture

The project consists of four main modules:

1. **FaceLandmarkDetector** - Face detection and 468 landmark extraction using MediaPipe
2. **GeometricTransformer** - Computes transformation matrices and warps accessories
3. **AccessoryOverlay** - Handles alpha-blending and overlay operations
4. **VirtualTryOnApp** - Main application integrating all modules

## Methodology

Pipeline for each video frame:

1. Face detection using MediaPipe Face Mesh
2. Landmark extraction (eyes, nose, mouth, forehead)
3. Geometric transformation based on anchor points
4. Overlay and blending using alpha channel

## Technologies

- OpenCV - Image and video processing
- MediaPipe - Face detection and landmarks
- NumPy - Mathematical operations

## Performance

- FPS displayed every 30 frames in console
- Smoothing applied for stable tracking
- Requirements: modern webcam and mid-range CPU

## Authors

- Farit Sharafutdinov (f.sharafutdinov@innopolis.university)
- Grigorii Belyaev (g.belyaev@innopolis.university)

B23 - AI01, Innopolis University

## License

Educational purposes only.

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)
