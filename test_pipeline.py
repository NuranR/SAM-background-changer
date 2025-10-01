"""
Test script to verify YOLO and SAM models are working correctly
"""
import cv2
import numpy as np
from ultralytics import YOLO, SAM
import torch
import time

def test_models():
    """Test both YOLO and SAM models"""
    print("🧪 Testing YOLO+SAM Pipeline")
    print("="*40)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test YOLO Model
    print("\n1. Testing YOLO Detection Model...")
    try:
        yolo_model = YOLO('models_for_project/yolov8n.pt')
        print("✓ YOLO model loaded successfully")
        
        # Test with a simple image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = yolo_model(test_image, classes=[0], verbose=False)
        print(f"✓ YOLO inference successful - detected {len(results[0].boxes) if results[0].boxes else 0} objects")
        
    except Exception as e:
        print(f"❌ YOLO model test failed: {e}")
        return False
    
    # Test SAM Model
    print("\n2. Testing SAM Segmentation Model...")
    try:
        sam_model = SAM('models_for_project/sam2_b.pt')
        print("✓ SAM model loaded successfully")
        
        # Test with dummy bounding box
        dummy_box = torch.tensor([[100, 100, 200, 200]], device=device)
        results = sam_model(test_image, bboxes=dummy_box, verbose=False)
        print(f"✓ SAM inference successful")
        
    except Exception as e:
        print(f"❌ SAM model test failed: {e}")
        return False
    
    print("\n3. Testing Pipeline Integration...")
    try:
        # Test the full pipeline
        yolo_results = yolo_model(test_image, classes=[0], verbose=False)
        
        # Create dummy person boxes for SAM
        dummy_boxes = torch.tensor([[150, 150, 300, 400]], device=device)
        sam_results = sam_model(test_image, bboxes=dummy_boxes, verbose=False)
        
        print("✓ Pipeline integration successful")
        
    except Exception as e:
        print(f"❌ Pipeline integration failed: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    print("Models are ready for real-time background replacement!")
    return True

def test_webcam():
    """Test webcam access"""
    print("\n4. Testing Webcam Access...")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read from webcam")
            return False
        
        print(f"✓ Webcam working - Resolution: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Model and System Tests...\n")
    
    models_ok = test_models()
    webcam_ok = test_webcam()
    
    print("\n" + "="*50)
    if models_ok and webcam_ok:
        print("🟢 ALL SYSTEMS GO!")
        print("Ready to run: python yolo_sam_changer.py")
    else:
        print("🔴 Some tests failed. Check the errors above.")
    print("="*50)