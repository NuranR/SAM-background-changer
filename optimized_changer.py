import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
import threading
from collections import deque

class OptimizedBackgroundChanger:
    def __init__(self, background_path="background.jpg", confidence=0.5):
        """
        High-performance background changer optimized for real-time use.
        Uses techniques similar to commercial applications like Zoom.
        """
        self.background_path = background_path
        self.confidence = confidence
        
        # GPU setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            torch.cuda.empty_cache()
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        
        # Load lightweight YOLO segmentation model (much faster than SAM)
        print("Loading optimized YOLOv8 segmentation model...")
        self.model = YOLO('yolov8n-seg.pt')  # Use YOLO's built-in segmentation
        
        # Warm up the model
        if torch.cuda.is_available():
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_frame, device=self.device, verbose=False)
            print("✓ GPU warm-up complete!")
        
        # Performance optimizations
        self.mask_cache = deque(maxlen=5)  # Cache recent masks for temporal consistency
        self.processed_frames = 0
        self.skip_frames = 2  # Process every Nth frame for heavy operations
        self.last_good_mask = None
        
        # Background and webcam setup
        self.background = self._load_background()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
            
        # Optimize webcam settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set target FPS
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Webcam: {self.width}x{self.height}")

    def _load_background(self):
        try:
            bg = cv2.imread(self.background_path)
            if bg is None: 
                raise FileNotFoundError
            return bg
        except:
            # Create optimized gradient background
            bg = np.zeros((480, 640, 3), dtype=np.uint8)
            for i in range(480):
                ratio = i / 480
                bg[i, :, 0] = int(100 + 155 * ratio)  # Blue
                bg[i, :, 1] = int(50 * (1-ratio))     # Green  
                bg[i, :, 2] = int(150 + 105 * ratio)  # Red
            return bg

    def _create_optimized_mask(self, frame):
        """
        Create mask using YOLOv8 segmentation (much faster than SAM).
        Applies temporal smoothing for stability.
        """
        # Only run full segmentation every N frames
        if self.processed_frames % self.skip_frames == 0:
            # Run YOLO segmentation (much faster than SAM)
            results = self.model(
                frame,
                classes=[0],  # Person class
                conf=self.confidence,
                device=self.device,
                verbose=False,
                imgsz=416,  # Smaller input size for speed (was 640)
                half=True if self.device == 'cuda' else False
            )
            
            # Create mask from YOLO segmentation
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            if results[0].masks is not None:
                for seg_mask in results[0].masks.data:
                    # Convert to numpy and resize
                    seg_np = seg_mask.cpu().numpy()
                    if seg_np.shape != mask.shape:
                        seg_np = cv2.resize(seg_np, (mask.shape[1], mask.shape[0]))
                    
                    # Add to combined mask
                    binary_mask = (seg_np > 0.5).astype(np.uint8) * 255
                    mask = cv2.bitwise_or(mask, binary_mask)
                
                # Apply morphological operations for cleaner edges
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Smooth edges with Gaussian blur
                mask = cv2.GaussianBlur(mask, (3, 3), 1)
                
                self.last_good_mask = mask
                self.mask_cache.append(mask)
            
            elif self.last_good_mask is not None:
                # Use previous mask if current detection failed
                mask = self.last_good_mask
        else:
            # Use cached mask for non-processing frames
            mask = self.last_good_mask if self.last_good_mask is not None else np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Temporal smoothing: blend with recent masks
        if len(self.mask_cache) > 1:
            weights = [0.5, 0.3, 0.2]  # Current frame gets highest weight
            blended_mask = np.zeros_like(mask, dtype=np.float32)
            
            for i, cached_mask in enumerate(list(self.mask_cache)[-3:]):
                if i < len(weights):
                    blended_mask += cached_mask.astype(np.float32) * weights[i]
            
            mask = np.clip(blended_mask, 0, 255).astype(np.uint8)
        
        return mask

    def _blend_images_fast(self, foreground, background, mask):
        """
        Optimized blending using vectorized operations.
        """
        # Resize background once
        if background.shape[:2] != foreground.shape[:2]:
            background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
        
        # Convert mask to 3-channel for blending
        mask_norm = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_norm, mask_norm, mask_norm], axis=2)
        
        # Vectorized blending (much faster than iterating pixels)
        result = (foreground.astype(np.float32) * mask_3ch + 
                 background.astype(np.float32) * (1 - mask_3ch))
        
        return result.astype(np.uint8)

    def run(self):
        """
        Optimized main loop with performance monitoring.
        """
        print("\n" + "="*60)
        print("🚀 OPTIMIZED Background Changer (Zoom-like Performance)")
        print("="*60)
        print("💡 Using YOLOv8 segmentation instead of SAM for real-time performance")
        print("📊 Temporal smoothing and frame skipping enabled")
        print("🎮 Controls: 'q' quit | 's' save | 'b' background | '+/-' adjust skip")
        print("="*60 + "\n")
        
        fps_history = deque(maxlen=30)
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Mirror effect
            frame = cv2.flip(frame, 1)
            frame_count += 1
            start_time = time.time()
            
            # Create optimized mask
            mask = self._create_optimized_mask(frame)
            
            # Apply background replacement
            if np.any(mask):
                result_frame = self._blend_images_fast(frame, self.background, mask)
                status = f"✓ Person Detected (Optimized Pipeline)"
                status_color = (0, 255, 0)
            else:
                result_frame = cv2.resize(self.background, (frame.shape[1], frame.shape[0]))
                status = "No Person - Background Only"
                status_color = (0, 255, 255)
            
            # Performance metrics
            end_time = time.time()
            current_fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            fps_history.append(current_fps)
            avg_fps = np.mean(fps_history)
            
            # Display information
            cv2.putText(result_frame, f"FPS: {current_fps:.1f} (Avg: {avg_fps:.1f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, status, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(result_frame, f"Skip: {self.skip_frames} | Frame: {frame_count}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result_frame, "OPTIMIZED: YOLOv8-Seg + Temporal Smoothing", 
                       (10, result_frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Optimized Background Changer", result_frame)
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"optimized_capture_{frame_count}_{int(time.time())}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"💾 Saved: {filename}")
            elif key == ord('b'):
                self._cycle_background()
            elif key == ord('+') or key == ord('='):
                self.skip_frames = max(1, self.skip_frames - 1)
                print(f"🔧 Skip frames: {self.skip_frames} (Higher processing load)")
            elif key == ord('-'):
                self.skip_frames = min(10, self.skip_frames + 1)
                print(f"🔧 Skip frames: {self.skip_frames} (Lower processing load)")
            
            self.processed_frames += 1
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 Final Stats: Avg FPS: {np.mean(fps_history):.1f}, Total Frames: {frame_count}")

    def _cycle_background(self):
        """Cycle through available backgrounds"""
        import os
        bg_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.png')) and 'background' in f.lower()]
        if len(bg_files) > 1:
            current_idx = bg_files.index(self.background_path) if self.background_path in bg_files else 0
            next_bg = bg_files[(current_idx + 1) % len(bg_files)]
            self.background_path = next_bg
            self.background = self._load_background()
            print(f"🖼️ Background: {next_bg}")

if __name__ == "__main__":
    try:
        app = OptimizedBackgroundChanger()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: {e}")