import cv2
import numpy as np
from ultralytics import YOLO, SAM
import torch
import time
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

class BackgroundChangerGUI:
    def __init__(self, root):
        """
        Modern GUI application for YOLO+SAM Background Changer
        """
        self.root = root
        self.root.title("AI Background Changer - YOLO+SAM")
        
        # Make window maximized
        self.root.state('zoomed')
        
        # Application state
        self.mode = 'replace'
        self.show_debug = False
        self.current_bg_index = 0
        self.is_running = True
        self.confidence = 0.5
        
        # GPU setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Setup UI first
        self._setup_ui()
        
        # Then initialize models in background
        self.status_label.config(text="Loading models...")
        self.root.after(100, self._initialize_models)
        
    def _setup_ui(self):
        """Create the GUI layout"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Video feed (720p: 1280x720)
        video_frame = tk.Frame(main_frame, bg='#000000', width=1280, height=720)
        video_frame.pack(side=tk.LEFT, padx=(0, 10))
        video_frame.pack_propagate(False)
        
        self.video_label = tk.Label(video_frame, bg='#000000')
        self.video_label.pack(expand=True)
        
        # Right side - Control panel
        control_frame = tk.Frame(main_frame, bg='#1e1e1e', width=350)
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        control_frame.pack_propagate(False)
        
        # Title
        title = tk.Label(control_frame, text="AI Background Changer", 
                        font=('Segoe UI', 18, 'bold'), bg='#1e1e1e', fg='#ffffff')
        title.pack(pady=(20, 10))
        
        subtitle = tk.Label(control_frame, text="YOLO Detection + SAM Segmentation", 
                           font=('Segoe UI', 9), bg='#1e1e1e', fg='#888888')
        subtitle.pack(pady=(0, 20))
        
        # Status section
        status_frame = tk.LabelFrame(control_frame, text="Status", 
                                     bg='#1e1e1e', fg='#ffffff', font=('Segoe UI', 10, 'bold'))
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.status_label = tk.Label(status_frame, text="Initializing...", 
                                     bg='#1e1e1e', fg='#00ff00', font=('Segoe UI', 9))
        self.status_label.pack(pady=10, padx=10)
        
        self.fps_label = tk.Label(status_frame, text="FPS: --", 
                                  bg='#1e1e1e', fg='#ffff00', font=('Segoe UI', 9))
        self.fps_label.pack(pady=(0, 10), padx=10)
        
        # Mode selection section
        mode_frame = tk.LabelFrame(control_frame, text="Background Mode", 
                                   bg='#1e1e1e', fg='#ffffff', font=('Segoe UI', 10, 'bold'))
        mode_frame.pack(fill=tk.X, padx=20, pady=10)
        
        btn_style = {'font': ('Segoe UI', 10), 'width': 25, 'height': 2}
        
        self.btn_replace = tk.Button(mode_frame, text="🖼️ Replace Background", 
                                     command=lambda: self.set_mode('replace'),
                                     bg='#0078d4', fg='white', **btn_style)
        self.btn_replace.pack(pady=5, padx=10)
        
        self.btn_blur = tk.Button(mode_frame, text="🌫️ Blur Background", 
                                 command=lambda: self.set_mode('blur'),
                                 bg='#404040', fg='white', **btn_style)
        self.btn_blur.pack(pady=5, padx=10)
        
        self.btn_green = tk.Button(mode_frame, text="🟢 Green Screen", 
                                  command=lambda: self.set_mode('green_screen'),
                                  bg='#404040', fg='white', **btn_style)
        self.btn_green.pack(pady=5, padx=10)
        
        # Background selection (only visible in replace mode)
        self.bg_frame = tk.LabelFrame(control_frame, text="Background Images", 
                                      bg='#1e1e1e', fg='#ffffff', font=('Segoe UI', 10, 'bold'))
        self.bg_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.bg_label = tk.Label(self.bg_frame, text="No backgrounds found", 
                                bg='#1e1e1e', fg='#888888', font=('Segoe UI', 9))
        self.bg_label.pack(pady=10)
        
        self.btn_prev_bg = tk.Button(self.bg_frame, text="◀ Previous", 
                                     command=self.prev_background,
                                     bg='#404040', fg='white', font=('Segoe UI', 9), width=12)
        self.btn_prev_bg.pack(side=tk.LEFT, padx=(10, 5), pady=5)
        
        self.btn_next_bg = tk.Button(self.bg_frame, text="Next ▶", 
                                     command=self.next_background,
                                     bg='#404040', fg='white', font=('Segoe UI', 9), width=12)
        self.btn_next_bg.pack(side=tk.RIGHT, padx=(5, 10), pady=5)
        
        # Settings section
        settings_frame = tk.LabelFrame(control_frame, text="Settings", 
                                       bg='#1e1e1e', fg='#ffffff', font=('Segoe UI', 10, 'bold'))
        settings_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Debug toggle
        self.debug_var = tk.BooleanVar()
        debug_check = tk.Checkbutton(settings_frame, text="Show Debug Info (Bounding Boxes)", 
                                     variable=self.debug_var, command=self.toggle_debug,
                                     bg='#1e1e1e', fg='#ffffff', selectcolor='#404040',
                                     font=('Segoe UI', 9))
        debug_check.pack(pady=10, padx=10, anchor='w')
        
        # Actions section
        action_frame = tk.LabelFrame(control_frame, text="Actions", 
                                     bg='#1e1e1e', fg='#ffffff', font=('Segoe UI', 10, 'bold'))
        action_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.btn_save = tk.Button(action_frame, text="📷 Save Frame", 
                                 command=self.save_frame,
                                 bg='#107c10', fg='white', **btn_style)
        self.btn_save.pack(pady=5, padx=10)
        
        # Info section at bottom
        info_frame = tk.Frame(control_frame, bg='#1e1e1e')
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)
        
        gpu_text = f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
        gpu_label = tk.Label(info_frame, text=gpu_text, 
                            bg='#1e1e1e', fg='#888888', font=('Segoe UI', 8))
        gpu_label.pack()
        
        version_label = tk.Label(info_frame, text="Version 1.0 - High Quality Mode", 
                                bg='#1e1e1e', fg='#888888', font=('Segoe UI', 8))
        version_label.pack()
        
    def _initialize_models(self):
        """Initialize YOLO and SAM models"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
            
            # Load models
            self.status_label.config(text="Loading YOLO detector...")
            self.yolo_detector = YOLO('models_for_project/yolov8n.pt')
            
            self.status_label.config(text="Loading SAM segmenter...")
            self.sam_segmenter = SAM('models_for_project/sam2_b.pt')
            
            # Warm up GPU
            if torch.cuda.is_available():
                self.status_label.config(text="Warming up GPU...")
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                _ = self.yolo_detector(dummy, device=self.device, verbose=False)
                _ = self.sam_segmenter(dummy, bboxes=[[100, 100, 200, 200]], device=self.device, verbose=False)
            
            # Scan backgrounds
            self.background_files = self._scan_backgrounds()
            self._update_bg_display()
            
            # Load first background
            if self.background_files:
                self.background = self._load_background()
            else:
                self.background = self._create_default_background()
            
            # Initialize webcam
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.status_label.config(text="Ready! ✓", fg='#00ff00')
            
            # Start video processing
            self.process_video()
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg='#ff0000')
            
    def _scan_backgrounds(self):
        """Scan backgrounds directory"""
        bg_dir = Path('backgrounds')
        if not bg_dir.exists():
            bg_dir.mkdir()
        
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        files = []
        for ext in exts:
            files.extend(bg_dir.glob(f'*{ext}'))
            files.extend(bg_dir.glob(f'*{ext.upper()}'))
        
        files = sorted([str(f) for f in files])
        if not files and os.path.exists('background.jpg'):
            files = ['background.jpg']
        
        return files if files else []
    
    def _load_background(self):
        """Load current background image"""
        if self.background_files and self.current_bg_index < len(self.background_files):
            bg = cv2.imread(self.background_files[self.current_bg_index])
            if bg is not None:
                return bg
        return self._create_default_background()
    
    def _create_default_background(self):
        """Create gradient background"""
        bg = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(720):
            ratio = i / 720
            bg[i, :, 0] = int(100 + 155 * ratio)
            bg[i, :, 1] = int(50 * (1-ratio))
            bg[i, :, 2] = int(150 + 105 * ratio)
        return bg
    
    def _update_bg_display(self):
        """Update background selection display"""
        if self.background_files:
            current = os.path.basename(self.background_files[self.current_bg_index])
            text = f"{current}\n({self.current_bg_index + 1} of {len(self.background_files)})"
            self.bg_label.config(text=text, fg='#ffffff')
        else:
            self.bg_label.config(text="No backgrounds found\nAdd images to 'backgrounds' folder", fg='#888888')
    
    def set_mode(self, mode):
        """Change background mode"""
        self.mode = mode
        
        # Update button colors
        all_btns = [self.btn_replace, self.btn_blur, self.btn_green]
        for btn in all_btns:
            btn.config(bg='#404040')
        
        if mode == 'replace':
            self.btn_replace.config(bg='#0078d4')
        elif mode == 'blur':
            self.btn_blur.config(bg='#0078d4')
        elif mode == 'green_screen':
            self.btn_green.config(bg='#0078d4')
    
    def prev_background(self):
        """Previous background"""
        if self.background_files:
            self.current_bg_index = (self.current_bg_index - 1) % len(self.background_files)
            self.background = self._load_background()
            self._update_bg_display()
            if self.mode != 'replace':
                self.set_mode('replace')
    
    def next_background(self):
        """Next background"""
        if self.background_files:
            self.current_bg_index = (self.current_bg_index + 1) % len(self.background_files)
            self.background = self._load_background()
            self._update_bg_display()
            if self.mode != 'replace':
                self.set_mode('replace')
    
    def toggle_debug(self):
        """Toggle debug mode"""
        self.show_debug = self.debug_var.get()
    
    def save_frame(self):
        """Save current frame"""
        if hasattr(self, 'current_frame'):
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.status_label.config(text=f"Saved: {filename}", fg='#00ff00')
            self.root.after(2000, lambda: self.status_label.config(text="Ready! ✓"))
    
    def process_video(self):
        """Main video processing loop"""
        if not self.is_running:
            return
        
        start_time = time.time()
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror
            frame = cv2.resize(frame, (1280, 720))  # Ensure 720p
            
            # Run detection and segmentation
            mask = self._create_mask(frame)
            
            # Apply selected mode
            if np.any(mask):
                if self.mode == 'replace':
                    result = self._blend(frame, self.background, mask)
                elif self.mode == 'blur':
                    blurred = cv2.GaussianBlur(frame, (51, 51), 0)
                    result = self._blend(frame, blurred, mask)
                elif self.mode == 'green_screen':
                    green = np.zeros_like(frame)
                    green[:, :] = (0, 255, 0)
                    result = self._blend(frame, green, mask)
            else:
                if self.mode == 'replace':
                    result = cv2.resize(self.background, (1280, 720))
                elif self.mode == 'blur':
                    result = cv2.GaussianBlur(frame, (51, 51), 0)
                else:
                    result = np.zeros_like(frame)
                    result[:, :] = (0, 255, 0)
            
            self.current_frame = result
            
            # Update FPS
            fps = 1 / (time.time() - start_time)
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            
            # Display
            self._display_frame(result)
        
        self.root.after(1, self.process_video)
    
    def _create_mask(self, frame):
        """Create segmentation mask"""
        # YOLO detection
        results = self.yolo_detector(frame, classes=[0], conf=self.confidence, 
                                     device=self.device, verbose=False, half=True if self.device == 'cuda' else False)
        
        boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if box.conf >= self.confidence:
                    boxes.append(box.xyxy[0].cpu().numpy())
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        if boxes:
            # SAM segmentation
            sam_results = self.sam_segmenter(frame, bboxes=boxes, device=self.device, verbose=False)
            
            if sam_results[0].masks is not None:
                for mask_tensor in sam_results[0].masks.data:
                    m = mask_tensor.cpu().numpy()
                    if m.shape != mask.shape:
                        m = cv2.resize(m, (mask.shape[1], mask.shape[0]))
                    mask = cv2.bitwise_or(mask, (m > 0.5).astype(np.uint8) * 255)
            
            # Show debug boxes
            if self.show_debug:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        return mask
    
    def _blend(self, fg, bg, mask):
        """Blend foreground and background"""
        bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))
        mask = cv2.GaussianBlur(mask, (5, 5), 2)
        mask_norm = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_norm] * 3, axis=2)
        return (fg * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)
    
    def _display_frame(self, frame):
        """Display frame in GUI"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.img_tk = img_tk
        self.video_label.configure(image=img_tk)
    
    def on_closing(self):
        """Cleanup on exit"""
        self.is_running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = BackgroundChangerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()