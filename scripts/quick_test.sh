#!/bin/bash
# Quick test of model inference speed and basic functionality

echo "=========================================="
echo "Quick Model Test"
echo "=========================================="
echo ""

# Check venv
if [ ! -f "nn_venv/bin/python" ]; then
    echo "ERROR: Virtual environment not found!"
    exit 1
fi

source nn_venv/bin/activate
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Check model
if [ ! -f "checkpoints/best_model.pt" ]; then
    echo "ERROR: No model found at checkpoints/best_model.pt"
    exit 1
fi

echo "Testing model: checkpoints/best_model.pt"
echo ""

# Run quick test
python -c "
import torch
import yaml
import sys
import time
sys.path.insert(0, '.')

from model.models.cnn_rnn import CNNRNNHamiltonian

print('Loading model...')
with open('model/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
model = CNNRNNHamiltonian(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f'✓ Model loaded')
print(f'  Epoch: {checkpoint[\"epoch\"]}')
print(f'  Best val loss: {checkpoint[\"best_val_loss\"]:.4f}')

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f'  Parameters: {total:,}')

# Quick inference test
print('')
print('Testing inference speed...')
seq_len = config['model']['predictor']['sequence_length']
dummy = torch.randn(1, seq_len, 4, 30, 30)
global_feat = torch.randn(1, seq_len, 3)

start = time.time()
with torch.no_grad():
    _ = model(dummy, global_feat)
elapsed = time.time() - start

print(f'✓ Inference test: {elapsed*1000:.1f}ms')
print('')
print('Model is ready for use!')
"

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
