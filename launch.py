#!/usr/bin/env python3
"""
AI Background Changer - Launcher Script
Checks dependencies and launches the application
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.9 or higher"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required = ['torch', 'cv2', 'PIL', 'ultralytics', 'numpy']
    missing = []
    
    for package in required:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} not found")
    
    if missing:
        print("\n⚠️  Missing dependencies detected!")
        print("Run: pip install -r requirements.txt")
        return False
    return True

def check_models():
    """Check if AI models are present"""
    models_dir = Path('models_for_project')
    
    if not models_dir.exists():
        print("\n⚠️  models_for_project/ folder not found")
        print("Creating folder...")
        models_dir.mkdir()
    
    print("\n📦 Checking AI models...")
    yolo_model = models_dir / 'yolov8n.pt'
    sam_model = models_dir / 'sam2_b.pt'
    
    if not yolo_model.exists():
        print("⚠️  YOLOv8n model not found (will auto-download on first run)")
    else:
        print("✓ YOLOv8n model found")
    
    if not sam_model.exists():
        print("⚠️  SAM 2 model not found (will auto-download on first run)")
    else:
        print("✓ SAM 2 model found")
    
    return True

def check_backgrounds():
    """Check if backgrounds folder exists"""
    bg_dir = Path('backgrounds')
    if not bg_dir.exists():
        print("\n⚠️  backgrounds/ folder not found")
        print("Creating folder...")
        bg_dir.mkdir()
    
    images = list(bg_dir.glob('*.jpg')) + list(bg_dir.glob('*.png'))
    if images:
        print(f"✓ Found {len(images)} background image(s)")
    else:
        print("⚠️  No background images found in backgrounds/")
        print("   Add some .jpg or .png files to use custom backgrounds")

def main():
    """Main launcher function"""
    print("=" * 50)
    print("🎨 AI Background Changer - Launcher")
    print("=" * 50)
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check models
    check_models()
    
    # Check backgrounds
    check_backgrounds()
    
    print("\n" + "=" * 50)
    print("✓ All checks passed!")
    print("=" * 50)
    print("\n🚀 Launching GUI Application...\n")
    
    # Launch the application
    try:
        subprocess.run([sys.executable, 'gui_app.py'])
    except KeyboardInterrupt:
        print("\n\n👋 Application closed by user")
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
