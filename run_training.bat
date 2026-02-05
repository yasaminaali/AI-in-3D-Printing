@echo off
REM Launcher for NN training with automatic preprocessing

echo [Neural Network Training Launcher]
echo.

REM Check if venv exists
if not exist "nn_venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_nn.bat first
    exit /b 1
)

REM Activate venv
call nn_venv\Scripts\activate.bat

REM Set PYTHONPATH
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Check if preprocessed data exists
if exist "nn_data\train_all.jsonl" (
    echo [OK] Preprocessed data found: nn_data\train_all.jsonl
    echo.
    goto :training
)

REM Check for combined dataset
echo [INFO] Preprocessed data not found. Checking for raw datasets...
if exist "output\datasets\combined_dataset.jsonl" (
    echo [OK] Found combined dataset: output\datasets\combined_dataset.jsonl
    echo.
    echo [PREPROCESS] Running preprocessing on combined dataset...
    nn_venv\Scripts\python.exe model\data\preprocess.py --input output\datasets\combined_dataset.jsonl
    if errorlevel 1 (
        echo.
        echo [ERROR] Preprocessing failed!
        exit /b 1
    )
    echo [OK] Preprocessing complete!
    echo.
    goto :training
)

REM Check for individual datasets
if exist "output\datasets\leftright_stripes.jsonl" (
    if exist "output\datasets\voronoi_island.jsonl" (
        echo [OK] Found individual datasets. Merging first...
        nn_venv\Scripts\python.exe merge_datasets_safe.py
        if errorlevel 1 (
            echo.
            echo [ERROR] Dataset merging failed!
            exit /b 1
        )
        echo.
        echo [PREPROCESS] Running preprocessing on merged dataset...
        nn_venv\Scripts\python.exe model\data\preprocess.py --input output\datasets\combined_dataset.jsonl
        if errorlevel 1 (
            echo.
            echo [ERROR] Preprocessing failed!
            exit /b 1
        )
        echo [OK] Preprocessing complete!
        echo.
        goto :training
    )
)

REM Check for ansh dataset (original)
if exist "output\ansh\Dataset.jsonl" (
    echo [OK] Found original dataset: output\ansh\Dataset.jsonl
    echo.
    echo [PREPROCESS] Running preprocessing...
    nn_venv\Scripts\python.exe model\data\preprocess.py --input output\ansh\Dataset.jsonl
    if errorlevel 1 (
        echo.
        echo [ERROR] Preprocessing failed!
        exit /b 1
    )
    echo [OK] Preprocessing complete!
    echo.
    goto :training
)

echo [ERROR] No dataset found!
echo.
echo Please ensure one of the following exists:
echo   - nn_data\train_all.jsonl (preprocessed)
echo   - output\datasets\combined_dataset.jsonl (merged)
echo   - output\datasets\leftright_stripes.jsonl + voronoi_island.jsonl (raw)
echo   - output\ansh\Dataset.jsonl (original)
exit /b 1

:training
echo [TRAINING] Starting training with Rich UI dashboard...
echo.
nn_venv\Scripts\python.exe model\train.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed with error code %errorlevel%
    exit /b 1
)

echo.
echo [DONE] Training complete!
pause
