import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
import os
from PIL import Image

class SAMBackgroundChanger:
    def __init__(self, model_path, background_path):
        """
        Initialize the SAM 2 Background Changer
        
        Args:
            model_path (str): Path to the SAM 2 model checkpoint
            background_path (str): Path to the background image
        """
        self.model_path = model_path
        self.background_path = background_path
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the SAM 2 model
        print("Loading SAM 2 model...")
        self.predictor = self._load_model()
        
        # Load background image
        self.background = self._load_background()
        
        # Initialize tracking variables
        self.tracking_state = None
        self.frame_idx = 0
        self.is_tracking = False
        
        # Mouse callback variables
        self.drawing = False
        self.start_point = None
        self.end_point = None
        
    def _load_model(self):
        """Load the SAM 2 model"""
        try:
            # SAM 2 model configuration
            sam2_checkpoint = self.model_path
            model_cfg = "sam2_hiera_s.yaml"  # Small model config
            
            predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
            return predictor
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have downloaded the model checkpoint and installed SAM 2")
            return None
    
    def _load_background(self):
        """Load and resize background image"""
        try:
            background = cv2.imread(self.background_path)
            if background is None:
                # Create a default blue gradient background
                print("Background image not found, creating default background...")
                height, width = 720, 1280
                background = np.zeros((height, width, 3), dtype=np.uint8)
                for i in range(height):
                    background[i, :] = [int(255 * (1 - i/height)), int(100 * (1 - i/height)), 255]
            return background
        except Exception as e:
            print(f"Error loading background: {e}")
            # Fallback background
            height, width = 720, 1280
            background = np.zeros((height, width, 3), dtype=np.uint8)
            background[:] = [100, 150, 200]  # Light blue
            return background
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for bounding box selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            # Start tracking with the selected bounding box
            if self.start_point and self.end_point:
                self._start_tracking()
    
    def _start_tracking(self):
        """Start tracking with the selected bounding box"""
        if self.predictor is None:
            return
            
        try:
            # Convert bounding box to the format expected by SAM 2
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            # Ensure proper box format
            bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            
            # Initialize the predictor state
            if self.current_frame is not None:
                inference_state = self.predictor.init_state(video_path=None)
                self.predictor.reset_state(inference_state)
                
                # Add the bounding box as a prompt
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=self.frame_idx,
                    obj_id=1,
                    box=np.array(bbox, dtype=np.float32)
                )
                
                self.tracking_state = inference_state
                self.is_tracking = True
                print(f"Started tracking with bounding box: {bbox}")
                
        except Exception as e:
            print(f"Error starting tracking: {e}")
            self.is_tracking = False
    
    def _apply_background_replacement(self, frame, mask):
        """Apply background replacement using the mask"""
        try:
            # Resize background to match frame size
            h, w = frame.shape[:2]
            background_resized = cv2.resize(self.background, (w, h))
            
            # Convert mask to 3-channel
            if len(mask.shape) == 2:
                mask_3ch = np.stack([mask, mask, mask], axis=-1)
            else:
                mask_3ch = mask
            
            # Normalize mask to 0-1 range
            mask_norm = mask_3ch.astype(np.float32) / 255.0
            
            # Apply background replacement
            result = frame * mask_norm + background_resized * (1 - mask_norm)
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"Error applying background replacement: {e}")
            return frame
    
    def run(self):
        """Main application loop"""
        if self.predictor is None:
            print("Model not loaded. Exiting.")
            return
            
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create window and set mouse callback
        window_name = "SAM 2 Background Changer"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("Instructions:")
        print("1. Click and drag to select the person you want to track")
        print("2. The background will be replaced in real-time")
        print("3. Press 'r' to reset tracking")
        print("4. Press 'q' to quit")
        
        self.current_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            # Create display frame
            display_frame = frame.copy()
            
            # Draw selection rectangle
            if self.drawing and self.start_point and self.end_point:
                cv2.rectangle(display_frame, self.start_point, self.end_point, (0, 255, 0), 2)
            
            # Perform tracking and background replacement
            if self.is_tracking and self.tracking_state is not None:
                try:
                    # Get mask for current frame
                    out_obj_ids, out_mask_logits = self.predictor.propagate_in_video(
                        self.tracking_state, self.frame_idx
                    )
                    
                    if len(out_mask_logits) > 0:
                        # Convert mask to binary
                        mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                        mask = mask.squeeze().astype(np.uint8) * 255
                        
                        # Apply background replacement
                        display_frame = self._apply_background_replacement(frame, mask)
                        
                        # Draw tracking indicator
                        cv2.putText(display_frame, "Tracking Active", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    self.frame_idx += 1
                    
                except Exception as e:
                    print(f"Error during tracking: {e}")
                    self.is_tracking = False
            else:
                # Show instructions
                cv2.putText(display_frame, "Click and drag to select person", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow(window_name, display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset tracking
                self.is_tracking = False
                self.tracking_state = None
                self.frame_idx = 0
                print("Tracking reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    # File paths
    model_path = "sam2.1_hiera_small.pt"
    background_path = "background.jpg"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        print("Please run 'python download_model.py' first to download the model.")
        return
    
    # Create and run the background changer
    app = SAMBackgroundChanger(model_path, background_path)
    app.run()

if __name__ == "__main__":
    main()