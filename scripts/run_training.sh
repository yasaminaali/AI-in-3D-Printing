#!/bin/bash
# RunPod Training Launcher for Linux
# Trains the CNN+RNN model on RTX 5090

echo "=========================================="
echo "Neural Network Training Launcher (Linux)"
echo "=========================================="
echo ""

# Check if venv exists
if [ ! -f "nn_venv/bin/python" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run setup_runpod.sh first"
    exit 1
fi

# Activate venv
source nn_venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Check if preprocessed data exists
if [ -f "nn_data/train_all.jsonl" ]; then
    echo "[OK] Preprocessed data found: nn_data/train_all.jsonl"
    echo ""
else
    echo "[INFO] Preprocessed data not found. Checking for raw datasets..."
    
    if [ -f "output/datasets/combined_dataset.jsonl" ]; then
        echo "[OK] Found combined dataset: output/datasets/combined_dataset.jsonl"
        echo ""
        echo "[PREPROCESS] Running preprocessing on combined dataset..."
        python model/data/preprocess.py --input output/datasets/combined_dataset.jsonl
        if [ $? -ne 0 ]; then
            echo "[ERROR] Preprocessing failed!"
            exit 1
        fi
        echo "[OK] Preprocessing complete!"
        echo ""
    else
        echo "[ERROR] No dataset found!"
        echo ""
        echo "Please ensure one of the following exists:"
        echo "  - nn_data/train_all.jsonl (preprocessed)"
        echo "  - output/datasets/combined_dataset.jsonl (merged)"
        exit 1
    fi
fi

# Check for checkpoint to resume
if [ -f "nn_checkpoints/best_model.pt" ]; then
    echo "[OK] Found checkpoint: nn_checkpoints/best_model.pt"
    python -c "
import torch
checkpoint = torch.load('nn_checkpoints/best_model.pt', map_location='cpu')
print(f'Resuming from epoch: {checkpoint[\"epoch\"]}')
print(f'Best val loss: {checkpoint[\"best_val_loss\"]:.4f}')
"
    echo ""
fi

# Show GPU info
echo "[INFO] GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Show config
echo "[INFO] Training Configuration:"
python -c "
import yaml
with open('model/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f'  Epochs: {config[\"training\"][\"epochs\"]}')
print(f'  Batch size (30x30): {config[\"training\"][\"batch_sizes\"][\"30x30\"]}')
print(f'  Sequence length: {config[\"model\"][\"predictor\"][\"sequence_length\"]}')
print(f'  Mixed precision: {config[\"performance\"][\"mixed_precision\"]}')
print(f'  Early stopping patience: {config[\"training\"][\"early_stopping_patience\"]}')
"
echo ""

echo "[TRAINING] Starting training with Rich UI dashboard..."
echo ""

python model/train.py --train-file nn_data/train_all.jsonl --val-file nn_data/val_all.jsonl

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Training failed!"
    exit 1
fi

echo ""
echo "[DONE] Training complete!"
echo "Model saved to: nn_checkpoints/best_model.pt"
