@echo off
REM ==== Part 1: Set Python Environment Variable (if not set) ====
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found in PATH. Adding Python to PATH...
    setx PATH "%PATH%;C:\Python39;C:\Python39\Scripts" /m
    echo Updated PATH to include Python. Restart CMD if needed.
) else (
    echo Python is already in PATH.
)

REM ==== Part 2: Install Requirements ====
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies. Check requirements.txt.
    pause
    exit /b
)

REM ==== Part 3: Run YOLO Object Detection with Timestamps ====
echo Running object detection on video...
python detect_with_timestamps.py
if %errorlevel% neq 0 (
    echo Script failed. Check the error above.
    pause
    exit /b
)

echo Done! Check output_video.mp4 and timestamps.log.
pause