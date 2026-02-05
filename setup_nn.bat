@echo off
REM One-click setup for AI-in-3D-Printing NN training
REM Creates isolated venv with all dependencies

echo ==========================================
echo  AI-in-3D-Printing NN Setup
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.8+ and try again
    exit /b 1
)

echo [1/4] Creating virtual environment...
if exist "nn_venv" (
    echo      Virtual environment already exists
) else (
    python -m venv nn_venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        exit /b 1
    )
    echo      Created nn_venv\
)

echo.
echo [2/4] Activating virtual environment...
call nn_venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    exit /b 1
)
echo      Activated

echo.
echo [3/4] Installing dependencies...
echo      This may take a few minutes...

REM Install PyTorch with CUDA support first
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 -q

REM Install other dependencies
pip install rich pyyaml numpy matplotlib tqdm -q

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)
echo      All dependencies installed

echo.
echo [4/4] Verifying installation...
python -c "import torch; print(f'     PyTorch: {torch.__version__}'); print(f'     CUDA available: {torch.cuda.is_available()}'); print(f'     GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ==========================================
echo  Setup Complete!
echo ==========================================
echo.
echo To use the environment:
echo   1. Run: nn_venv\Scripts\activate.bat
echo   2. Or use the launcher scripts:
echo      - run_training.bat
echo      - run_inference.bat
echo.
echo Press any key to exit...
pause >nul
