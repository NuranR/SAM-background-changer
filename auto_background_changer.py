import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import argparse
import os

class AutoBackgroundChanger:
    def __init__(self, background_path="background.jpg", confidence=0.5):
        """
        Initialize the Automatic Background Changer (like Zoom)
        
        Args:
            background_path (str): Path to the background image
            confidence (float): Confidence threshold for person detection
        """
        self.background_path = background_path
        self.confidence = confidence
        
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load YOLOv8 segmentation model
        print("Loading YOLOv8 segmentation model...")
        self.model = YOLO('yolov8n-seg.pt')  # nano model for speed
        
        # Load background image
        self.background = self._load_background()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
            
        # Get webcam resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Webcam resolution: {self.width}x{self.height}")
        
    def _load_background(self):
        """Load and resize background image"""
        try:
            if os.path.exists(self.background_path):
                bg_image = cv2.imread(self.background_path)
                if bg_image is None:
                    print(f"Warning: Could not load background image from {self.background_path}")
                    return self._create_default_background()
                return bg_image
            else:
                print(f"Background image not found: {self.background_path}")
                return self._create_default_background()
        except Exception as e:
            print(f"Error loading background: {e}")
            return self._create_default_background()
    
    def _create_default_background(self):
        """Create a default gradient background"""
        # Create a nice gradient background
        background = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create gradient from blue to purple
        for i in range(480):
            ratio = i / 480
            # Blue to purple gradient
            background[i, :, 0] = int(255 * ratio)      # Blue component
            background[i, :, 1] = int(100 * (1-ratio))  # Green component  
            background[i, :, 2] = int(200 * ratio)      # Red component
            
        return background
    
    def _resize_background(self, target_height, target_width):
        """Resize background to match frame dimensions"""
        return cv2.resize(self.background, (target_width, target_height))
    
    def _smooth_mask(self, mask):
        """Apply smoothing to the mask for better edge quality"""
        # Apply Gaussian blur to soften edges
        mask_float = mask.astype(np.float32) / 255.0
        mask_blur = cv2.GaussianBlur(mask_float, (5, 5), 2)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_blur = cv2.morphologyEx(mask_blur, cv2.MORPH_CLOSE, kernel)
        mask_blur = cv2.morphologyEx(mask_blur, cv2.MORPH_OPEN, kernel)
        
        return mask_blur
    
    def _create_person_mask(self, frame):
        """
        Create a mask for people in the frame using YOLO segmentation
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            mask: Binary mask where people pixels are white (255)
        """
        # Run YOLO segmentation
        results = self.model(frame, verbose=False)
        
        # Initialize empty mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Process results
        for result in results:
            if result.masks is not None:
                # Get masks and classes
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                # Process each detected object
                for i, (cls, conf) in enumerate(zip(classes, confidences)):
                    # Class 0 is 'person' in COCO dataset
                    if int(cls) == 0 and conf > self.confidence:
                        # Get mask for this person
                        person_mask = masks[i]
                        
                        # Resize mask to frame size
                        person_mask_resized = cv2.resize(person_mask, (frame.shape[1], frame.shape[0]))
                        
                        # Convert to binary mask
                        person_mask_binary = (person_mask_resized > 0.5).astype(np.uint8) * 255
                        
                        # Add to combined mask
                        mask = cv2.bitwise_or(mask, person_mask_binary)
        
        return mask
    
    def _blend_background(self, frame, background, mask):
        """
        Blend the background with the original frame using the person mask
        
        Args:
            frame: Original webcam frame
            background: Background image
            mask: Person mask (0-1 float values)
            
        Returns:
            blended: Final blended frame
        """
        # Ensure mask has 3 channels for blending
        if len(mask.shape) == 2:
            mask_3ch = cv2.merge([mask, mask, mask])
        else:
            mask_3ch = mask
            
        # Blend: person pixels from original frame, background elsewhere
        blended = frame * mask_3ch + background * (1 - mask_3ch)
        
        return blended.astype(np.uint8)
    
    def run(self):
        """Main loop for the background changer"""
        print("Starting automatic background changer...")
        print("Press 'q' to quit, 'b' to change background, 's' to save current frame")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from webcam")
                break
            
            # Flip frame horizontally for mirror effect (like Zoom)
            frame = cv2.flip(frame, 1)
            
            # Resize background to match frame
            frame_height, frame_width = frame.shape[:2]
            background_resized = self._resize_background(frame_height, frame_width)
            
            # Create person mask
            person_mask = self._create_person_mask(frame)
            
            if np.any(person_mask > 0):
                # Person detected - apply background replacement
                # Smooth the mask for better blending
                smooth_mask = self._smooth_mask(person_mask)
                
                # Blend background
                result_frame = self._blend_background(frame, background_resized, smooth_mask)
                
                # Add status text
                status_text = f"Person detected - Background replaced"
                cv2.putText(result_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # No person detected - show background only
                result_frame = background_resized.copy()
                
                # Add status text
                status_text = "No person detected - Showing background"
                cv2.putText(result_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add instructions
            instructions = "Press: 'q' to quit | 'b' to change background | 's' to save"
            cv2.putText(result_frame, instructions, (10, frame_height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display result
            cv2.imshow('Auto Background Changer', result_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                self._change_background()
            elif key == ord('s'):
                self._save_frame(result_frame)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _change_background(self):
        """Change to next background (cycle through available backgrounds)"""
        # List of possible background files
        background_files = [
            'background.jpg', 'background.png', 
            'background1.jpg', 'background1.png',
            'background2.jpg', 'background2.png'
        ]
        
        # Find current background index
        current_idx = 0
        for i, bg_file in enumerate(background_files):
            if bg_file == self.background_path:
                current_idx = i
                break
        
        # Try to load next available background
        for i in range(len(background_files)):
            next_idx = (current_idx + 1 + i) % len(background_files)
            next_bg = background_files[next_idx]
            
            if os.path.exists(next_bg):
                self.background_path = next_bg
                self.background = self._load_background()
                print(f"Background changed to: {next_bg}")
                return
        
        print("No other background files found")
    
    def _save_frame(self, frame):
        """Save current frame"""
        filename = f"captured_frame_{cv2.getTickCount()}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved as: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Automatic Background Changer - Like Zoom')
    parser.add_argument('--background', '-b', default='background.jpg',
                       help='Path to background image (default: background.jpg)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold for person detection (default: 0.5)')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run the background changer
        changer = AutoBackgroundChanger(
            background_path=args.background,
            confidence=args.confidence
        )
        changer.run()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()