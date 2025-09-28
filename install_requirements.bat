@echo off
echo Setting up SAM 2 Background Changer Environment...
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    echo Please ensure Python is installed and added to PATH
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support for GTX 1650
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Install other required packages
echo Installing other required packages...
pip install opencv-python
pip install numpy
pip install Pillow
pip install requests
pip install tqdm

REM Install SAM 2
echo Installing Segment Anything 2...
pip install git+https://github.com/facebookresearch/segment-anything-2.git

echo.
echo Installation completed successfully!
echo To use the environment in the future, run: venv\Scripts\activate
echo.
pause