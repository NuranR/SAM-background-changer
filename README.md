# SAM 2 Real-Time Background Changer

A real-time background replacement application using Segment Anything 2 (SAM 2) Small model to identify people in webcam feed and replace backgrounds dynamically.

## 🎯 Features

- **Real-time segmentation** using SAM 2 Small model
- **Webcam integration** with live background replacement
- **Interactive selection** - click and drag to select the person
- **GPU acceleration** support for NVIDIA GTX 1650 MaxQ and higher
- **Custom backgrounds** - use any image as your new background

## 📋 Prerequisites

- **Hardware**: Windows PC with webcam
- **GPU**: NVIDIA GTX 1650 MaxQ or better (recommended for real-time performance)
- **Software**: Python 3.8+ (Python 3.10 recommended)

## 🚀 Quick Setup

### Step 1: Install Python Environment

Run the automated setup script:
```cmd
install_requirements.bat
```

Or manually create a virtual environment:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Step 2: Download SAM 2 Model

Download the SAM 2.1 Hiera Small model from Hugging Face:

**Option 1: Direct Download**
- Visit: https://huggingface.co/facebook/sam2.1-hiera-small
- Download the `sam2.1_hiera_small.pt` file
- Place it in the `SAM_Background_Changer` folder

**Option 2: Using Hugging Face CLI**
```cmd
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam2.1-hiera-small', 'sam2.1_hiera_small.pt', local_dir='.')"
```

**Option 3: Check Model Setup**
```cmd
python download_model.py
```
This will show you exactly where to place the model file.

### Step 3: Add Your Background Image

Replace `background.jpg` with your desired background image:
- **Recommended resolution**: 1920x1080 or 1280x720 (16:9 aspect ratio)
- **Supported formats**: JPG, PNG
- **File name**: Must be named `background.jpg`

Or create a sample background:
```cmd
python create_sample_background.py
```

### Step 4: Run the Application

Activate your environment and run:
```cmd
venv\Scripts\activate
python run_webcam.py
```

## 🎮 How to Use

1. **Launch**: A window titled "SAM 2 Background Changer" will open showing your webcam feed
2. **Select Person**: Click and drag to draw a bounding box around the person you want to track
3. **Real-time Tracking**: The application will track the person and replace the background automatically
4. **Controls**:
   - **Mouse**: Click and drag to select person
   - **'r' key**: Reset tracking (select a new person)
   - **'q' key**: Quit the application

## 📁 Project Structure

```
SAM_Background_Changer/
├── background.jpg              # Your background image
├── sam2.1_hiera_small.pt      # SAM 2 model checkpoint
├── run_webcam.py              # Main application
├── download_model.py          # Model download script
├── create_sample_background.py # Sample background generator
├── install_requirements.bat   # Windows setup script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
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