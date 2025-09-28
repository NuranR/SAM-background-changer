@echo off
echo Starting Auto Background Changer...
echo.
echo This will open your webcam and automatically replace backgrounds when a person is detected
echo Similar to Zoom's background replacement feature
echo.
echo Controls:
echo - Press 'q' to quit
echo - Press 'b' to cycle through background images  
echo - Press 's' to save current frame
echo.
echo Starting in 3 seconds...
timeout /t 3 /nobreak >nul

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate
)

REM Run the auto background changer
python auto_background_changer.py

pause