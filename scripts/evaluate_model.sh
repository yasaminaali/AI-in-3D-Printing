#!/bin/bash
# Comprehensive Model Evaluation Script for Linux
# Tests the trained model on multiple zone patterns and generates reports

echo "=========================================="
echo "Model Evaluation Suite - Linux"
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

# Check for model
echo "Checking for trained model..."
if [ -f "nn_checkpoints/best_model.pt" ]; then
    echo "[OK] Found: nn_checkpoints/best_model.pt"
    python -c "
import torch
checkpoint = torch.load('nn_checkpoints/best_model.pt', map_location='cpu')
print(f'  Epoch: {checkpoint[\"epoch\"]}')
print(f'  Best val loss: {checkpoint[\"best_val_loss\"]:.4f}')
"
else
    echo "[ERROR] No trained model found!"
    echo "Please train the model first using: ./run_training.sh"
    exit 1
fi

echo ""
echo "=========================================="
echo "Running Comprehensive Tests"
echo "=========================================="
echo ""

# Test 1: Quick inference test
echo "[TEST 1] Quick Inference Test"
python test_inference.py
echo ""

# Test 2: Full inference on all zone patterns
echo "[TEST 2] Testing All Zone Patterns"
echo ""

ZONE_PATTERNS=("left_right" "stripes" "voronoi" "checkerboard")
RESULTS_DIR="evaluation_results"
mkdir -p $RESULTS_DIR

echo "Testing patterns: ${ZONE_PATTERNS[*]}"
echo ""

for pattern in "${ZONE_PATTERNS[@]}"; do
    echo "  Testing $pattern..."
    python model/inference.py \
        --checkpoint nn_checkpoints/best_model.pt \
        --grid-W 30 \
        --grid-H 30 \
        --zone-pattern $pattern \
        > $RESULTS_DIR/${pattern}_result.txt 2>&1
    
    # Extract results
    if [ -f "inference_result.json" ]; then
        python -c "
import json
with open('inference_result.json', 'r') as f:
    data = json.load(f)
print(f\"    Initial: {data['initial_crossings']}, Final: {data['final_crossings']}, Improvement: {data['improvement']}\")
"
        # Move result to pattern-specific name
        mv inference_result.json $RESULTS_DIR/${pattern}_result.json
    else
        echo "    [WARNING] No result generated"
    fi
    echo ""
done

# Test 3: Performance benchmark
echo "[TEST 3] Performance Benchmark"
echo "  Running 100 inferences to measure speed..."
python -c "
import torch
import time
import yaml
import sys
sys.path.insert(0, '.')

from model.models.cnn_rnn import CNNRNNHamiltonian
from model.data.dataset import HamiltonianDataset

# Load model
with open('model/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

checkpoint = torch.load('nn_checkpoints/best_model.pt', map_location='cuda')
model = CNNRNNHamiltonian(config)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda().eval()

# Benchmark
times = []
with torch.no_grad():
    for i in range(10):  # 10 inferences
        dummy_input = torch.randn(1, 50, 4, 30, 30).cuda()
        dummy_global = torch.randn(1, 50, 3).cuda()
        
        start = time.time()
        _ = model(dummy_input, dummy_global)
        torch.cuda.synchronize()
        times.append(time.time() - start)

print(f'  Average inference time: {sum(times)/len(times)*1000:.1f}ms')
print(f'  Throughput: {1/(sum(times)/len(times)):.1f} inferences/sec')
"
echo ""

# Test 4: Model analysis
echo "[TEST 4] Model Analysis"
python -c "
import torch
import yaml

# Load config
with open('model/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load checkpoint
checkpoint = torch.load('nn_checkpoints/best_model.pt', map_location='cpu')

print('Model Configuration:')
print(f\"  CNN embedding dim: {config['model']['cnn']['embedding_dim']}\")
print(f\"  RNN hidden size: {config['model']['rnn']['hidden_size']}\")
print(f\"  RNN layers: {config['model']['rnn']['num_layers']}\")
print(f\"  Sequence length: {config['model']['predictor']['sequence_length']}\")
print()

print('Training History:')
print(f\"  Trained epochs: {checkpoint['epoch']}\")
print(f\"  Best validation loss: {checkpoint['best_val_loss']:.4f}\")

# Count parameters
from model.models.cnn_rnn import CNNRNNHamiltonian
model = CNNRNNHamiltonian(config)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print()
print('Model Size:')
print(f\"  Total parameters: {total_params:,}\")
print(f\"  Trainable parameters: {trainable_params:,}\")
print(f\"  Model size: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)\")
"
echo ""

# Test 5: Dataset analysis
echo "[TEST 5] Dataset Analysis"
if [ -f "nn_data/train_all.jsonl" ]; then
    python -c "
import json

with open('nn_data/train_all.jsonl', 'r') as f:
    lines = f.readlines()

print(f'  Training samples: {len(lines)}')

# Analyze sequence lengths
seq_lens = []
for line in lines[:100]:
    data = json.loads(line)
    seq_lens.append(len(data.get('sequence_ops', [])))

print(f'  Avg sequence length: {sum(seq_lens)/len(seq_lens):.1f}')
print(f'  Max sequence length: {max(seq_lens)}')
print(f'  Min sequence length: {min(seq_lens)}')

# Zone patterns
patterns = {}
for line in lines:
    data = json.loads(line)
    pattern = data.get('zone_pattern', 'unknown')
    patterns[pattern] = patterns.get(pattern, 0) + 1

print()
print('  Zone pattern distribution:')
for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
    print(f'    {pattern}: {count}')
"
else
    echo "  [WARNING] No training data found"
fi
echo ""

# Generate summary report
echo "=========================================="
echo "Generating Summary Report"
echo "=========================================="
echo ""

REPORT_FILE="$RESULTS_DIR/evaluation_report.txt"

cat > $REPORT_FILE << EOF
========================================
Model Evaluation Report
Generated: $(date)
========================================

Model: nn_checkpoints/best_model.pt
$(python -c "
import torch
checkpoint = torch.load('nn_checkpoints/best_model.pt', map_location='cpu')
print(f'Epoch: {checkpoint[\"epoch\"]}')
print(f'Best Val Loss: {checkpoint[\"best_val_loss\"]:.4f}')
")

Inference Results:
$(for pattern in "${ZONE_PATTERNS[@]}"; do
    if [ -f "$RESULTS_DIR/${pattern}_result.json" ]; then
        python -c "
import json
with open('$RESULTS_DIR/${pattern}_result.json', 'r') as f:
    data = json.load(f)
print(f'  {pattern}:')
print(f\"    Initial crossings: {data['initial_crossings']}\")
print(f\"    Final crossings: {data['final_crossings']}\")
print(f\"    Improvement: {data['improvement']}\")
"
    fi
done)

Files Generated:
$(ls -1 $RESULTS_DIR/)

========================================
Evaluation Complete
========================================
EOF

cat $REPORT_FILE
echo ""

echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo "  - evaluation_report.txt (summary)"
echo "  - *_result.json (per-pattern results)"
echo "  - *_result.txt (detailed logs)"
echo ""
echo "To view a specific result:"
echo "  cat $RESULTS_DIR/left_right_result.json | python -m json.tool"
