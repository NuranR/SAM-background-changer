# AI Background Changer

Real-time AI-powered background replacement using YOLO detection and SAM segmentation. Professional-grade virtual background system with GPU acceleration.

## ✨ Features

- **High-Quality Segmentation** - YOLO + SAM 2 pipeline for precise edge detection
- **Three Background Modes**:
  - 🖼️ Replace Background - Swap with custom images
  - 🌫️ Blur Background - Professional blur effect
  - � Green Screen - Chroma key effect
- **Modern GUI** - Clean interface with 720p video feed
- **GPU Accelerated** - CUDA support for real-time performance
- **Easy Background Management** - Add/switch backgrounds on the fly

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```


# AI Background Changer

Real-time background replacement using YOLO + SAM segmentation.

## Features
- Replace, blur, or green screen your webcam background
- GPU acceleration (CUDA recommended)
- Simple, modern GUI

## Quick Setup
1. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
2. **Add backgrounds:**
  - Place images in the `backgrounds/` folder
3. **Run the app:**
  ```bash
  python gui_app.py
  ```

## Notes
- Models (`yolov8n.pt`, `sam2_b.pt`) are auto-downloaded to `models_for_project/` on first run
- Captured photos are saved in `captures/`

## Requirements
- Python 3.9+
- Webcam
- (Optional) NVIDIA GPU for best performance
```

### Background Replacement

The application automatically:

- Resizes background to match webcam resolution
- Creates smooth mask blending
- Handles different aspect ratios