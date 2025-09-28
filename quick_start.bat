@echo off
title SAM 2 Background Changer - Quick Start

echo ========================================
echo   SAM 2 Background Changer Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found!
    echo Please run install_requirements.bat first.
    echo.
    pause
    exit /b 1
)

REM Check if model exists
if not exist "sam2.1_hiera_small.pt" (
    echo Model file not found!
    echo Please check model setup instructions...
    echo.
    call venv\Scripts\activate
    python download_model.py
    echo.
    echo Please download the model from Hugging Face and try again.
    pause
    exit /b 1
)

REM Create sample background if none exists
if not exist "background.jpg" (
    echo Creating sample background...
    call venv\Scripts\activate
    python create_sample_background.py
    ren sample_background.jpg background.jpg
)

REM Activate environment and run application
echo Starting SAM 2 Background Changer...
echo.
call venv\Scripts\activate
python run_webcam.py

echo.
echo Application closed.
pause