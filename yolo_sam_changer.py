import cv2
import numpy as np
from ultralytics import YOLO, SAM
import torch
import time

class YoloSamChanger:
    def __init__(self, background_path="background.jpg", confidence=0.5):
        """
        Initialize the two-stage YOLO+SAM background changer.
        
        Args:
            background_path (str): Path to background image
            confidence (float): Confidence threshold for person detection
        """
        self.background_path = background_path
        self.confidence = confidence
        
        # Check for GPU and set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            # Clear GPU cache for optimal performance
            torch.cuda.empty_cache()
        
        # --- Stage 1 Model: YOLOv8 for Detection ---
        print("Loading YOLOv8 detector model...")
        try:
            self.yolo_detector = YOLO('models_for_project/yolov8n.pt')
            print("✓ YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise

        # --- Stage 2 Model: SAM 2 for Segmentation ---
        print("Loading SAM 2 segmenter model...")
        try:
            self.sam_segmenter = SAM('models_for_project/sam2_b.pt')
            print("✓ SAM 2 model loaded successfully")
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            raise
            
        # Warm up the models with dummy inference to initialize GPU
        if torch.cuda.is_available():
            print("Warming up models on GPU...")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Warm up YOLO
            _ = self.yolo_detector(dummy_frame, device=self.device, verbose=False)
            
            # Warm up SAM with a dummy box
            dummy_box = [[100, 100, 200, 200]]  # x1, y1, x2, y2
            _ = self.sam_segmenter(dummy_frame, bboxes=dummy_box, device=self.device, verbose=False)
            
            print("✓ GPU warm-up complete!")
        
        # Load background and initialize webcam
        self.background = self._load_background()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Webcam resolution: {self.width}x{self.height}")
        print("Pipeline initialized successfully!")

    def _load_background(self):
        """Load background image or create default if not found"""
        try:
            bg = cv2.imread(self.background_path)
            if bg is None: 
                raise FileNotFoundError
            print(f"✓ Background loaded: {self.background_path}")
            return bg
        except (FileNotFoundError, Exception):
            print(f"Warning: Background '{self.background_path}' not found. Creating default.")
            return self._create_default_background()

    def _create_default_background(self):
        """Create a default gradient background"""
        bg = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create a nice gradient
        for i in range(480):
            ratio = i / 480
            bg[i, :, 0] = int(150 * ratio)      # Blue
            bg[i, :, 1] = int(50 * (1-ratio))   # Green  
            bg[i, :, 2] = int(100 * ratio)      # Red
        return bg

    def _smooth_mask(self, mask):
        """Apply smoothing to mask for better blending"""
        # Convert to float for processing
        mask_float = mask.astype(np.float32) / 255.0
        
        # Apply Gaussian blur for smooth edges
        mask_smooth = cv2.GaussianBlur(mask_float, (7, 7), 2)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_CLOSE, kernel)
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
        
        return mask_smooth

    def _blend_images(self, foreground, background, mask):
        """Blend foreground and background using the mask"""
        # Resize background to match foreground
        background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
        
        # Ensure mask has 3 channels for blending
        if len(mask.shape) == 2:
            mask_3ch = cv2.merge([mask, mask, mask])
        else:
            mask_3ch = mask
            
        # Blend: person pixels from original frame, background elsewhere
        blended = foreground * mask_3ch + background * (1 - mask_3ch)
        
        return blended.astype(np.uint8)

    def run(self):
        """
        Main loop to run the two-stage pipeline on webcam feed.
        """
        print("\n" + "="*50)
        print("🚀 Starting YOLO+SAM Background Changer")
        print("="*50)
        print("Controls:")
        print("  'q' - Quit application")
        print("  's' - Save current frame")
        print("  'b' - Change background")
        print("="*50 + "\n")
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            frame_count += 1
            start_time = time.time()

            # --- Stage 1: Run YOLO Detection ---
            # Explicitly specify device and optimize for speed
            yolo_results = self.yolo_detector(
                frame, 
                classes=[0], 
                conf=self.confidence, 
                device=self.device,
                verbose=False,
                imgsz=640,  # Optimize input size
                half=True if self.device == 'cuda' else False  # Use FP16 on GPU for speed
            )
            
            # Extract person bounding boxes
            person_boxes = []
            if yolo_results[0].boxes is not None:
                for box in yolo_results[0].boxes:
                    if box.conf >= self.confidence:
                        person_boxes.append(box.xyxy[0].cpu().numpy())

            # Initialize empty mask
            final_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            
            if len(person_boxes) > 0:
                if frame_count % 10 == 1:  # Reduce print frequency
                    print(f"Frame {frame_count}: YOLO detected {len(person_boxes)} person(s)")
                
                # --- Stage 2: Run SAM Segmentation ---
                try:
                    # Run SAM with bounding box prompts, explicitly on GPU
                    sam_results = self.sam_segmenter(
                        frame, 
                        bboxes=person_boxes, 
                        device=self.device,
                        verbose=False
                    )
                    
                    # Process SAM results
                    if sam_results[0].masks is not None:
                        if frame_count % 10 == 1:  # Reduce print frequency
                            print(f"Frame {frame_count}: SAM generated {len(sam_results[0].masks.data)} mask(s)")
                        
                        # Combine all masks from SAM into one
                        for mask_tensor in sam_results[0].masks.data:
                            # Convert tensor to numpy array
                            mask_np = mask_tensor.cpu().numpy()
                            
                            # Resize mask to frame size if needed
                            if mask_np.shape != (frame.shape[0], frame.shape[1]):
                                mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                            
                            # Convert to binary mask and combine
                            mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                            final_mask = cv2.bitwise_or(final_mask, mask_binary)
                    
                except Exception as e:
                    print(f"SAM processing error: {e}")
                    # Fallback: create a simple mask from YOLO boxes
                    for box in person_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(final_mask, (x1, y1), (x2, y2), 255, -1)

            # Process the final result
            if np.any(final_mask):
                # Apply smoothing to the mask
                smooth_mask = self._smooth_mask(final_mask)
                
                # Apply background replacement
                result_frame = self._blend_images(frame, self.background, smooth_mask)
                status = f"✓ Person Detected (YOLO+SAM Pipeline)"
                status_color = (0, 255, 0)  # Green
                
                # Show detection boxes for debugging
                for box in person_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(result_frame, "YOLO", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                # No person detected - show background only
                result_frame = cv2.resize(self.background, (frame.shape[1], frame.shape[0]))
                status = "No Person Detected"
                status_color = (0, 255, 255)  # Yellow

            # Calculate and display performance metrics
            end_time = time.time()
            fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            
            # Add overlay information
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, status, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(result_frame, f"Frame: {frame_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add model information
            cv2.putText(result_frame, "Pipeline: YOLO+SAM", (10, result_frame.shape[0]-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result_frame, "Press 'q' to quit", (10, result_frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display the result
            cv2.imshow("YOLO+SAM Background Changer", result_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting application...")
                break
            elif key == ord('s'):
                filename = f"yolo_sam_capture_{frame_count}_{int(time.time())}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"Frame saved as: {filename}")
            elif key == ord('b'):
                self._change_background()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully!")

    def _change_background(self):
        """Cycle through available background images"""
        import os
        background_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        background_files = [f for f in background_files if 'background' in f.lower()]
        
        if len(background_files) > 1:
            current_idx = 0
            for i, bg_file in enumerate(background_files):
                if bg_file == self.background_path:
                    current_idx = i
                    break
            
            next_idx = (current_idx + 1) % len(background_files)
            self.background_path = background_files[next_idx]
            self.background = self._load_background()
            print(f"Background changed to: {self.background_path}")
        else:
            print("No additional background files found")

def main():
    """Main function to run the application"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced YOLO+SAM Background Changer')
    parser.add_argument('--background', '-b', default='background.jpg',
                       help='Path to background image (default: background.jpg)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold for person detection (default: 0.5)')
    
    args = parser.parse_args()
    
    try:
        changer = YoloSamChanger(
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