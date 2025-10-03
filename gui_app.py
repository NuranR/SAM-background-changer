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
        self.root.title("AI Background Changer")
        
        # Make window maximized
        self.root.state('zoomed')
        self.root.configure(bg='#1a1a1a')
        
        # Application state
        self.mode = 'replace'
        self.current_bg_index = 0
        self.is_running = True
        self.confidence = 0.5
        self.last_frame = None
        
        # Scan backgrounds early to prevent duplicate counting
        self.background_files = self._scan_backgrounds()
        self.background = None
        
        # GPU setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create Tkinter variables before UI setup
        self.mode_var = tk.StringVar(value='replace')
        
        # Setup UI first
        self._setup_ui()
        
        # Then initialize models in background
        self.status_label.config(text="Loading models...")
        self.root.after(100, self._initialize_models)
        
    def _setup_ui(self):
        """Create the GUI layout"""
        # Main container with elegant dark theme
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Left side - Video feed (720p: 1280x720) with subtle border
        video_container = tk.Frame(main_frame, bg='#0a0a0a', highlightbackground='#333333', 
                                   highlightthickness=1)
        video_container.pack(side=tk.LEFT, padx=(0, 15))
        
        video_frame = tk.Frame(video_container, bg='#000000', width=1280, height=720)
        video_frame.pack(padx=2, pady=2)
        video_frame.pack_propagate(False)
        
        self.video_label = tk.Label(video_frame, bg='#000000')
        self.video_label.pack(expand=True)
        
        # Right side - Elegant Control panel
        control_frame = tk.Frame(main_frame, bg='#242424', width=380)
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        control_frame.pack_propagate(False)
        
        # Elegant Title with icon
        title_frame = tk.Frame(control_frame, bg='#242424')
        title_frame.pack(pady=(30, 5))
        
        # title = tk.Label(title_frame, text="AI Background Changer", 
        #                 font=('Segoe UI', 20, 'bold'), bg='#242424', fg='#ffffff')
        # title.pack()
        
        # Status section with modern minimal design
        status_label_header = tk.Label(control_frame, text="Status", 
                                      font=('Segoe UI', 12, 'bold'), bg='#242424', fg='#ffffff')
        status_label_header.pack(pady=(30, 10))
        
        status_container = tk.Frame(control_frame, bg='#2d2d2d', highlightbackground='#3d3d3d', highlightthickness=1)
        status_container.pack(fill=tk.X, padx=30, pady=(0, 10))
        
        self.status_label = tk.Label(status_container, text="Initializing...", 
                                     bg='#2d2d2d', fg='#5dc461', font=('Segoe UI', 10))
        self.status_label.pack(pady=12, padx=15)
        
        self.fps_label = tk.Label(status_container, text="FPS: --", 
                                  bg='#2d2d2d', fg='#ffd966', font=('Segoe UI', 10))
        self.fps_label.pack(pady=(0, 12), padx=15)
        
        # Mode selection with elegant radio buttons
        mode_label = tk.Label(control_frame, text="Background Mode", 
                             font=('Segoe UI', 12, 'bold'), bg='#242424', fg='#ffffff')
        mode_label.pack(pady=(30, 15))
        
        mode_container = tk.Frame(control_frame, bg='#242424')
        mode_container.pack(fill=tk.X, padx=30)
        
        modes = [
            ("     Replace", "replace"),
            ("       Blur ", "blur"),
            (" Green Screen", "green_screen")
        ]
        
        for text, mode in modes:
            btn = tk.Radiobutton(mode_container, text=text, variable=self.mode_var, value=mode,
                                font=('Segoe UI', 11), bg='#242424', fg='#e8e8e8',
                                selectcolor='#0078d4', activebackground='#242424',
                                activeforeground='#ffffff', relief=tk.FLAT,
                                indicatoron=True, anchor='w', padx=5, pady=8,
                                command=self.set_mode, cursor='hand2')
            btn.pack(fill=tk.X, pady=2)
        
        # Background selection (only visible in replace mode)
        bg_label_header = tk.Label(control_frame, text="Background Images", 
                                  font=('Segoe UI', 12, 'bold'), bg='#242424', fg='#ffffff')
        bg_label_header.pack(pady=(30, 15))
        
        self.bg_frame = tk.Frame(control_frame, bg='#242424')
        self.bg_frame.pack(fill=tk.X, padx=30)
        
        self.bg_label = tk.Label(self.bg_frame, text="No backgrounds found", 
                                bg='#242424', fg='#888888', font=('Segoe UI', 10))
        self.bg_label.pack(pady=(0, 12))
        
        # # Navigation buttons container
        
        # self.btn_next_bg.pack(side=tk.LEFT)
        nav_frame = tk.Frame(self.bg_frame, bg='#242424')
        nav_frame.pack(pady=(0, 10), fill="x")

        nav_frame.columnconfigure(0, weight=1)
        nav_frame.columnconfigure(1, weight=1)

        self.btn_prev_bg = tk.Button(nav_frame, text="◀ Prev",
                             command=self.prev_background,
                             bg='#3a3a3a', fg='white', font=('Segoe UI', 10),
                             relief=tk.FLAT, cursor='hand2')
        self.btn_prev_bg.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.btn_next_bg = tk.Button(nav_frame, text="Next ▶",
                             command=self.next_background,
                             bg='#3a3a3a', fg='white', font=('Segoe UI', 10),
                             relief=tk.FLAT, cursor='hand2')
        self.btn_next_bg.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        
        # Actions section
        action_label = tk.Label(control_frame, text="Actions", 
                               font=('Segoe UI', 12, 'bold'), bg='#242424', fg='#ffffff')
        action_label.pack(pady=(30, 15))
        
        action_frame = tk.Frame(control_frame, bg='#242424')
        action_frame.pack(fill=tk.X, padx=30)
        
        self.btn_save = tk.Button(action_frame, text="📷  Save Photo", 
                                 command=self.save_frame,
                                 bg='#107c10', fg='white', font=('Segoe UI', 11),
                                 relief=tk.FLAT, padx=20, pady=10, cursor='hand2')
        self.btn_save.pack(pady=5)
        
        # Info section at bottom
        info_frame = tk.Frame(control_frame, bg='#242424')
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=30, pady=25)
        
        gpu_text = f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
        gpu_label = tk.Label(info_frame, text=gpu_text, 
                            bg='#242424', fg='#707070', font=('Segoe UI', 9))
        gpu_label.pack()
        
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
            
            # Update background display (already scanned in __init__)
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
        """Scan backgrounds directory (case-insensitive, no duplicates)"""
        bg_dir = Path('backgrounds')
        if not bg_dir.exists():
            bg_dir.mkdir()
        
        # Get all image files, case-insensitive
        files = []
        for f in bg_dir.iterdir():
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                files.append(str(f))
        
        files = sorted(files)
        if not files and os.path.exists('background.jpg'):
            files = ['background.jpg']
        
        return files
    
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
    
    def set_mode(self):
        """Change background mode based on radio button selection"""
        self.mode = self.mode_var.get()
    
    def prev_background(self):
        """Previous background"""
        if self.background_files:
            self.current_bg_index = (self.current_bg_index - 1) % len(self.background_files)
            self.background = self._load_background()
            self._update_bg_display()
            if self.mode != 'replace':
                self.mode_var.set('replace')
                self.set_mode()
    
    def next_background(self):
        """Next background"""
        if self.background_files:
            self.current_bg_index = (self.current_bg_index + 1) % len(self.background_files)
            self.background = self._load_background()
            self._update_bg_display()
            if self.mode != 'replace':
                self.mode_var.set('replace')
                self.set_mode()
    

    
    def save_frame(self):
        """Save current frame to captures directory"""
        if hasattr(self, 'current_frame'):
            # Create captures directory if it doesn't exist
            captures_dir = Path('captures')
            captures_dir.mkdir(exist_ok=True)
            
            # Save with timestamp
            filename = f"capture_{int(time.time())}.jpg"
            filepath = captures_dir / filename
            cv2.imwrite(str(filepath), self.current_frame)
            
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