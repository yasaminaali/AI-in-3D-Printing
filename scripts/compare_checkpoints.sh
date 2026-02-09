#!/bin/bash
# Test all checkpoints and find the best performing one
# Usage: ./compare_checkpoints.sh

echo "=========================================="
echo "Checkpoint Comparison Tool"
echo "=========================================="
echo ""

# Check venv
if [ ! -f "nn_venv/bin/python" ]; then
    echo "ERROR: Virtual environment not found!"
    exit 1
fi

source nn_venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

# Find all checkpoints
echo "Scanning for checkpoints..."
CHECKPOINTS=$(ls nn_checkpoints/*.pt 2>/dev/null)

if [ -z "$CHECKPOINTS" ]; then
    echo "ERROR: No checkpoints found in nn_checkpoints/"
    exit 1
fi

echo "Found checkpoints:"
echo "$CHECKPOINTS"
echo ""

# Results file
RESULTS_FILE="checkpoint_comparison.txt"
echo "Checkpoint Comparison Results" > $RESULTS_FILE
echo "Generated: $(date)" >> $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Test each checkpoint
echo "Testing each checkpoint..."
echo ""

BEST_CHECKPOINT=""
BEST_LOSS=999999

for checkpoint in $CHECKPOINTS; do
    echo "Testing: $(basename $checkpoint)"
    
    # Get info
    python -c "
import torch
import sys
try:
    checkpoint = torch.load('$checkpoint', map_location='cpu')
    epoch = checkpoint['epoch']
    loss = checkpoint['best_val_loss']
    print(f'{epoch},{loss:.4f}')
except Exception as e:
    print(f'ERROR,{str(e)}')
    sys.exit(1)
" > /tmp/checkpoint_info.txt
    
    if [ $? -eq 0 ]; then
        INFO=$(cat /tmp/checkpoint_info.txt)
        EPOCH=$(echo $INFO | cut -d',' -f1)
        LOSS=$(echo $INFO | cut -d',' -f2)
        
        echo "  Epoch: $EPOCH, Loss: $LOSS"
        echo "$checkpoint: Epoch $EPOCH, Loss $LOSS" >> $RESULTS_FILE
        
        # Track best
        if (( $(echo "$LOSS < $BEST_LOSS" | bc -l) )); then
            BEST_LOSS=$LOSS
            BEST_CHECKPOINT=$checkpoint
        fi
    else
        echo "  [ERROR] Failed to load checkpoint"
    fi
    echo ""
done

echo "========================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "BEST CHECKPOINT:" >> $RESULTS_FILE
echo "  File: $BEST_CHECKPOINT" >> $RESULTS_FILE
echo "  Loss: $BEST_LOSS" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

echo "=========================================="
echo "Comparison Complete"
echo "=========================================="
echo ""
echo "Best checkpoint: $(basename $BEST_CHECKPOINT)"
echo "Best loss: $BEST_LOSS"
echo ""
echo "Full results saved to: $RESULTS_FILE"
echo ""
echo "To use the best checkpoint:"
echo "  ./run_inference.sh --checkpoint $BEST_CHECKPOINT"
