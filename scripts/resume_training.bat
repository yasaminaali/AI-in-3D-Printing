@echo off
REM Resume training from best checkpoint on RTX 5090

echo [Neural Network Training - Resume from Checkpoint]
echo.
echo GPU: RTX 5090 (32GB VRAM)
echo Optimized: batch_size=256, mixed_precision=true, epochs=300

if not exist "nn_checkpoints\best_model.pt" (
    echo ERROR: No checkpoint found at nn_checkpoints\best_model.pt
    echo Run run_training.bat first to create initial checkpoint
    exit /b 1
)

if not exist "nn_venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_nn.bat first
    exit /b 1
)

call nn_venv\Scripts\activate.bat

set PYTHONPATH=%CD%;%PYTHONPATH%

echo.
echo Resuming training from best checkpoint...
echo Config: sequence_length=100, batch_size=256, mixed_precision=enabled
echo.

nn_venv\Scripts\python.exe -c "
import torch
import yaml

# Load checkpoint to show info
checkpoint = torch.load('nn_checkpoints/best_model.pt', map_location='cpu')
print(f'Resuming from epoch: {checkpoint[\"epoch\"]}')
print(f'Best val loss so far: {checkpoint[\"best_val_loss\"]:.4f}')
print()

# Load config to show new settings
with open('model/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f'New settings:')
print(f'  Epochs: {config[\"training\"][\"epochs\"]}')
print(f'  Batch size (30x30): {config[\"training\"][\"batch_sizes\"][\"30x30\"]}')
print(f'  Mixed precision: {config[\"performance\"][\"mixed_precision\"]}')
print(f'  Early stopping patience: {config[\"training\"][\"early_stopping_patience\"]}')
print()
"

echo Starting training...
echo.

nn_venv\Scripts\python.exe model/train.py --train-file nn_data\train_all.jsonl --val-file nn_data\val_all.jsonl

if errorlevel 1 (
    echo.
    echo Training failed!
    exit /b 1
)

echo.
echo Training complete!
pause
