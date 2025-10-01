"""
Performance Comparison Guide

This script helps you understand the trade-offs between different approaches:

1. YOLO-Only Segmentation (optimized_changer.py):
   - Expected FPS: 15-30 FPS with person detection  
   - Quality: Good (95% as good as SAM for most cases)
   - GPU Usage: 30-50%
   - Method: Similar to Zoom, Google Meet, etc.

2. YOLO+SAM Pipeline (yolo_sam_changer.py):
   - Expected FPS: 1-2 FPS with person detection
   - Quality: Exceptional (best possible)
   - GPU Usage: 90-100%
   - Method: Research-grade, not real-time

Key Optimizations in optimized_changer.py:
- Uses YOLOv8's built-in segmentation (much lighter than SAM)
- Frame skipping: Only processes every Nth frame for heavy operations
- Temporal smoothing: Blends current mask with previous masks
- Smaller input resolution (416px instead of 640px)
- Vectorized blending operations
- Webcam buffer optimization

Interactive Controls:
- '+' key: Decrease skip frames (higher quality, lower FPS)
- '-' key: Increase skip frames (lower quality, higher FPS)
- 'b' key: Cycle backgrounds
- 's' key: Save frame
- 'q' key: Quit

This matches how commercial apps work - they prioritize real-time performance
over absolute segmentation perfection.
"""

import subprocess
import sys

def run_comparison():
    print("="*70)
    print("🔬 BACKGROUND CHANGER PERFORMANCE COMPARISON")
    print("="*70)
    print()
    print("Two versions available:")
    print()
    print("1. 🚀 OPTIMIZED VERSION (Recommended for real-time use)")
    print("   - File: optimized_changer.py")
    print("   - Expected: 15-30 FPS when person detected")
    print("   - Uses: YOLOv8 segmentation (like Zoom)")
    print("   - Quality: Very Good")
    print()
    print("2. 🎯 HIGH-QUALITY VERSION (For best segmentation)")
    print("   - File: yolo_sam_changer.py") 
    print("   - Expected: 1-2 FPS when person detected")
    print("   - Uses: YOLO detection + SAM segmentation")
    print("   - Quality: Exceptional")
    print()
    
    choice = input("Which version would you like to run? (1/2): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting OPTIMIZED version...")
        subprocess.run([sys.executable, "optimized_changer.py"])
    elif choice == "2":
        print("\n🎯 Starting HIGH-QUALITY version...")
        subprocess.run([sys.executable, "yolo_sam_changer.py"])
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    run_comparison()