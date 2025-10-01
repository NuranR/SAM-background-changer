# Setup Guide

## Step-by-Step Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd SAM_Background_Changer
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

**For CPU:**
```bash
pip install -r requirements.txt
```

**For GPU (NVIDIA CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy Pillow ultralytics
```

### 4. Download AI Models

The models will be automatically downloaded on first run. Alternatively, manually download:

- **YOLOv8n** (11 MB): [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)
- **SAM 2 Base** (158 MB): Downloaded automatically by Ultralytics

Place them in `models_for_project/` folder.

### 5. Add Background Images

1. Create a `backgrounds/` folder if it doesn't exist
2. Add your background images (.jpg, .png, .bmp)
3. Use high-resolution images (1280x720 or higher)

### 6. Run the Application

```bash
python gui_app.py
```

## Troubleshooting

### Issue: "No module named 'ultralytics'"
**Solution:** Install ultralytics: `pip install ultralytics`

### Issue: Low FPS (< 1 FPS)
**Solution:** Install PyTorch with CUDA support for GPU acceleration

### Issue: "CUDA out of memory"
**Solution:** Close other GPU-intensive applications or use CPU mode

### Issue: Webcam not detected
**Solution:** Check if webcam is connected and not used by another application

## Performance Tips

- **GPU Recommended**: NVIDIA GPU with 4GB+ VRAM
- **Resolution**: 720p (1280x720) is optimal for real-time performance
- **Backgrounds**: Use compressed JPG images for faster loading
- **Close Background Apps**: Free up GPU/CPU resources

## System Requirements

- **Minimum**: 
  - CPU: Intel i5 / AMD Ryzen 5
  - RAM: 8GB
  - Webcam: 720p

- **Recommended**:
  - CPU: Intel i7 / AMD Ryzen 7
  - RAM: 16GB
  - GPU: NVIDIA GTX 1650 or better
  - Webcam: 1080p
