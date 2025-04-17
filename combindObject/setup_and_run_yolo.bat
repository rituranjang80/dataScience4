@echo off
echo Setting up Python environment...
set PYTHONPATH=%PYTHONPATH%;%cd%
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    pause
    exit /b
)

echo Running detection...
python detect_custom_objects.py
if errorlevel 1 (
    echo Script failed.
    pause
    exit /b
)

echo Done! Check output_video.mp4 and detections.json.
pause