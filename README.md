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

For GPU support (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Download Models

Download the required models and place them in `models_for_project/`:
- **YOLOv8n**: `yolov8n.pt` (auto-downloaded by Ultralytics)
- **SAM 2 Base**: `sam2_b.pt` (auto-downloaded by Ultralytics)

### 3. Add Backgrounds

Place your background images in the `backgrounds/` folder:
- Supported formats: JPG, PNG, BMP
- Recommended resolution: 1280x720 or higher
- Multiple backgrounds supported

### 4. Run the Application

```bash
python gui_app.py
```

## 🎮 Usage

1. **Select Mode** - Choose Replace, Blur, or Green Screen
2. **Change Background** - Use Previous/Next buttons to cycle through backgrounds
3. **Capture** - Click "Save Photo" to save current frame to `captures/` folder

The app automatically detects people in the frame and applies the selected effect in real-time.

## 📋 Requirements

- **Python**: 3.9+
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Webcam**: Any standard webcam
- **OS**: Windows, Linux, macOS

## 📁 Project Structure

```
SAM_Background_Changer/
├── gui_app.py              # Main GUI application
├── backgrounds/            # Background images folder
├── captures/               # Saved photos folder
├── models_for_project/     # AI models folder
│   ├── yolov8n.pt
│   └── sam2_b.pt
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## ⚙️ Configuration

### Model Configuration
The application uses the SAM 2 Small model (`sam2_hiera_s.yaml`) which provides:
- **Good performance** on GTX 1650 MaxQ
- **Real-time inference** capabilities
- **Reasonable accuracy** for person segmentation

### Webcam Settings
Default webcam resolution: 1280x720
- Modify in `run_webcam.py` if needed:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
```

### Background Replacement
The application automatically:
- Resizes background to match webcam resolution
- Creates smooth mask blending
- Handles different aspect ratios

## 🔧 Troubleshooting

### Model Not Found
```
Error: Model file sam2.1_hiera_small.pt not found!
```
**Solution**: Run `python download_model.py`

### CUDA Out of Memory
```
CUDA out of memory
```
**Solutions**:
- Reduce webcam resolution in `run_webcam.py`
- Use CPU inference (will be slower)
- Close other GPU-intensive applications

### Webcam Not Detected
```
Error: Could not open webcam
```
**Solutions**:
- Check webcam connections
- Close other applications using the webcam
- Try different camera index: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`

### Slow Performance
- Ensure GPU drivers are up to date
- Check if CUDA is properly installed
- Monitor GPU usage with Task Manager
- Consider using a lower resolution background image

## 🎨 Background Tips

### Best Background Images
- **Resolution**: Match your webcam (usually 1280x720 or 1920x1080)
- **Aspect Ratio**: 16:9 for most webcams
- **Content**: Avoid very busy or distracting backgrounds
- **Lighting**: Consider lighting conditions to match your environment

### Creating Custom Backgrounds
You can use the sample background generator as a template:
```cmd
python create_sample_background.py
```

## 📊 Performance

### Expected Performance (GTX 1650 MaxQ):
- **FPS**: 15-25 fps at 1280x720
- **Latency**: ~40-60ms
- **Memory Usage**: ~2-4GB VRAM

### Optimization Tips:
- Use smaller background images
- Close unnecessary applications
- Ensure good lighting for better tracking
- Keep the person in the center of the frame

## 🔄 Updates

To update the SAM 2 library:
```cmd
venv\Scripts\activate
pip install --upgrade git+https://github.com/facebookresearch/segment-anything-2.git
```

## 📝 License

This project is for educational and personal use. Please respect the licensing terms of:
- [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)

## 🆘 Support

If you encounter issues:

1. Check that all dependencies are properly installed
2. Verify your GPU drivers are up to date
3. Ensure the model file was downloaded completely
4. Test with different background images
5. Try different lighting conditions

## 🚀 Advanced Usage

### Multiple Person Tracking
The current version tracks one person. For multiple people, you would need to modify the tracking logic in `run_webcam.py`.

### Custom Models
To use different SAM 2 models (Base, Large), modify the model configuration in the `_load_model()` method.

### Video Input
The application can be modified to work with video files instead of webcam by changing the `cv2.VideoCapture()` parameter.

---

**Enjoy your new virtual background! 🎉**