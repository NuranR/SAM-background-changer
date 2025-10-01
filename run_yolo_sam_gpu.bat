@echo off
echo =====================================================
echo       YOLO+SAM Background Changer - GPU Optimized
echo =====================================================
echo.
echo This version uses both YOLO and SAM models for
echo superior segmentation quality with GPU acceleration.
echo.
echo Your GPU: NVIDIA GTX 1650 MaxQ
echo Expected Performance: 5-15 FPS (much better than CPU!)
echo.
echo Controls:
echo - Press 'q' to quit
echo - Press 's' to save frame
echo - Press 'b' to change background
echo.
echo Starting application...
echo.

REM Ensure we're using the conda environment with GPU support
call conda activate base

REM Set CUDA environment variables for optimal performance
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM Run the optimized application
python yolo_sam_changer.py

pause