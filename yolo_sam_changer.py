import cv2
import numpy as np
from ultralytics import YOLO, SAM
import torch
import time
import os
from pathlib import Path

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
        
        # Application state variables
        self.mode = 'replace'  # Modes: 'replace', 'blur', 'green_screen'
        self.show_debug = False  # Toggle for debug info (bounding boxes)
        self.current_bg_index = 0
        
        # Scan backgrounds directory for available images
        self.background_files = self._scan_backgrounds_directory()
        
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
        print(f"Found {len(self.background_files)} background images")
        print("Pipeline initialized successfully!")

    def _scan_backgrounds_directory(self):
        """Scan the backgrounds directory for available image files"""
        backgrounds_dir = Path('backgrounds')
        
        # Create directory if it doesn't exist
        if not backgrounds_dir.exists():
            backgrounds_dir.mkdir()
            print("✓ Created 'backgrounds' directory")
        
        # Scan for image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        background_files = []
        
        for ext in image_extensions:
            background_files.extend(backgrounds_dir.glob(f'*{ext}'))
            background_files.extend(backgrounds_dir.glob(f'*{ext.upper()}'))
        
        # Convert to strings and sort
        background_files = sorted([str(f) for f in background_files])
        
        # If no backgrounds found, use default background.jpg if it exists
        if not background_files and os.path.exists('background.jpg'):
            background_files = ['background.jpg']
        
        return background_files if background_files else [None]
    
    def _load_background(self):
        """Load background image or create default if not found"""
        # If using background from list
        if self.background_files[self.current_bg_index] is not None:
            bg_path = self.background_files[self.current_bg_index]
            try:
                bg = cv2.imread(bg_path)
                if bg is None: 
                    raise FileNotFoundError
                print(f"✓ Background loaded: {bg_path}")
                return bg
            except (FileNotFoundError, Exception):
                print(f"Warning: Could not load '{bg_path}'")
        
        # Fallback to default
        print("Creating default gradient background")
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
        print("\n" + "="*60)
        print("🚀 YOLO+SAM Background Changer - Feature Rich Edition")
        print("="*60)
        print("✨ Features:")
        print("  • High-quality YOLO+SAM segmentation pipeline")
        print("  • Multiple background modes (Replace/Blur/Green Screen)")
        print("  • Dynamic background management")
        print("="*60)
        print("🎮 Controls:")
        print("  [1] Background Replace mode")
        print("  [2] Background Blur mode") 
        print("  [3] Green Screen mode")
        print("  [B] Cycle through backgrounds")
        print("  [D] Toggle debug info")
        print("  [S] Save current frame")
        print("  [Q] Quit application")
        print("="*60 + "\n")
        
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

            # Process the final result based on selected mode
            if np.any(final_mask):
                # Apply smoothing to the mask
                smooth_mask = self._smooth_mask(final_mask)
                
                # Process based on current mode
                if self.mode == 'replace':
                    # Mode 1: Background Replacement
                    result_frame = self._blend_images(frame, self.background, smooth_mask)
                    mode_text = "Mode: Background Replace"
                    
                elif self.mode == 'blur':
                    # Mode 2: Background Blur
                    # Create blurred version of the original frame
                    blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0)
                    # Blend sharp person with blurred background
                    result_frame = self._blend_images(frame, blurred_frame, smooth_mask)
                    mode_text = "Mode: Background Blur"
                    
                elif self.mode == 'green_screen':
                    # Mode 3: Green Screen (Chroma Key)
                    green_bg = np.zeros_like(frame)
                    green_bg[:, :] = (0, 255, 0)  # Bright green (BGR format)
                    result_frame = self._blend_images(frame, green_bg, smooth_mask)
                    mode_text = "Mode: Green Screen"
                
                status = f"✓ Person Detected"
                status_color = (0, 255, 0)  # Green
                
                # Show detection boxes for debugging (only if debug mode is on)
                if self.show_debug:
                    for box in person_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(result_frame, "YOLO", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                # No person detected - behavior depends on mode
                if self.mode == 'replace':
                    result_frame = cv2.resize(self.background, (frame.shape[1], frame.shape[0]))
                elif self.mode == 'blur':
                    result_frame = cv2.GaussianBlur(frame, (51, 51), 0)
                elif self.mode == 'green_screen':
                    result_frame = np.zeros_like(frame)
                    result_frame[:, :] = (0, 255, 0)
                
                mode_text = f"Mode: {self.mode.replace('_', ' ').title()}"
                status = "No Person Detected"
                status_color = (0, 255, 255)  # Yellow

            # Calculate and display performance metrics
            end_time = time.time()
            fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            
            # Add overlay information with improved UI
            y_offset = 30
            line_height = 35
            
            # FPS
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += line_height
            
            # Status
            cv2.putText(result_frame, status, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            y_offset += line_height
            
            # Current Mode (highlighted)
            cv2.putText(result_frame, mode_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += line_height
            
            # Current background (if in replace mode)
            if self.mode == 'replace' and self.background_files[self.current_bg_index]:
                bg_name = os.path.basename(self.background_files[self.current_bg_index])
                cv2.putText(result_frame, f"BG: {bg_name}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Debug indicator
            if self.show_debug:
                cv2.putText(result_frame, "[DEBUG ON]", (result_frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Help text at bottom
            help_y = result_frame.shape[0] - 80
            cv2.putText(result_frame, "Controls: [1]Replace [2]Blur [3]GreenScreen [B]BG [D]Debug [S]Save [Q]Quit", 
                       (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            help_y += 20
            cv2.putText(result_frame, "Pipeline: YOLO Detection + SAM Segmentation (High Quality)", 
                       (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

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
                print(f"💾 Frame saved as: {filename}")
            elif key == ord('b'):
                # Cycle through backgrounds and switch to replace mode
                self._cycle_background()
            elif key == ord('1'):
                # Switch to Replace mode
                self.mode = 'replace'
                print("🔄 Switched to: Background Replace mode")
            elif key == ord('2'):
                # Switch to Blur mode
                self.mode = 'blur'
                print("🔄 Switched to: Background Blur mode")
            elif key == ord('3'):
                # Switch to Green Screen mode
                self.mode = 'green_screen'
                print("🔄 Switched to: Green Screen mode")
            elif key == ord('d'):
                # Toggle debug mode
                self.show_debug = not self.show_debug
                status_text = "ON" if self.show_debug else "OFF"
                print(f"🐛 Debug mode: {status_text}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully!")

    def _cycle_background(self):
        """Cycle through available background images from backgrounds directory"""
        if len(self.background_files) > 0 and self.background_files[0] is not None:
            # Move to next background
            self.current_bg_index = (self.current_bg_index + 1) % len(self.background_files)
            
            # Load the new background
            self.background = self._load_background()
            
            # Switch to replace mode if not already
            if self.mode != 'replace':
                self.mode = 'replace'
                print("🔄 Switched to: Background Replace mode")
            
            bg_name = os.path.basename(self.background_files[self.current_bg_index])
            print(f"🖼️  Background changed to: {bg_name} ({self.current_bg_index + 1}/{len(self.background_files)})")
        else:
            print("⚠️  No background images found in 'backgrounds' directory")

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