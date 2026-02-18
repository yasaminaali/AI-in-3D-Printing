@echo off
REM Launcher for NN inference - handles all path issues automatically

echo [Neural Network Inference Launcher]
echo.

REM Check if venv exists
if not exist "nn_venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_nn.bat first
    exit /b 1
)

REM Activate venv and run inference with proper Python path
call nn_venv\Scripts\activate.bat

REM Set PYTHONPATH to include current directory
set PYTHONPATH=%CD%\src;%PYTHONPATH%

REM Run inference
echo Starting inference...
echo.
nn_venv\Scripts\python.exe model/inference.py %*

if errorlevel 1 (
    echo.
    echo Inference failed with error code %errorlevel%
    exit /b 1
)

echo.
echo Inference complete!
pause
