#!/bin/bash
# Master script showing all available evaluation commands

cat << 'EOF'
========================================
AI-in-3D-Printing - Linux Evaluation Tools
========================================

Available Scripts:

1. ./quick_test.sh
   Quick model load and inference speed test
   Usage: ./quick_test.sh

2. ./run_inference.sh [options]
   Run inference on a specific problem
   Usage: ./run_inference.sh --zone-pattern left_right
   Options:
     --checkpoint PATH   Model path (default: checkpoints/best_model.pt)
     --grid-W N         Grid width (default: 30)
     --grid-H N         Grid height (default: 30)
     --zone-pattern P   Pattern: left_right, stripes, voronoi, checkerboard
     --help             Show full help

3. ./test_inference.py
   Python test script (comprehensive model testing)
   Usage: python test_inference.py

4. ./evaluate_model.sh
   Comprehensive evaluation suite
   Tests all zone patterns, benchmarks performance, generates report
   Usage: ./evaluate_model.sh
   Output: evaluation_results/ directory with full report

5. ./compare_checkpoints.sh
   Compare all saved checkpoints and find the best one
   Usage: ./compare_checkpoints.sh
   Output: checkpoint_comparison.txt

========================================
Quick Start:
========================================

# Test if model works:
./quick_test.sh

# Run single inference:
./run_inference.sh --zone-pattern left_right

# Full evaluation (all patterns):
./evaluate_model.sh

# Find best checkpoint:
./compare_checkpoints.sh

========================================
Examples:
========================================

# Test different zone patterns:
./run_inference.sh --zone-pattern stripes
./run_inference.sh --zone-pattern voronoi
./run_inference.sh --zone-pattern checkerboard

# Use a specific checkpoint:
./run_inference.sh --checkpoint checkpoints/checkpoint_epoch_40.pt

# Different grid size (if supported):
./run_inference.sh --grid-W 50 --grid-H 50 --zone-pattern left_right

========================================
Output Files:
========================================

After inference, check:
  - inference_result.json     (latest result)
  - evaluation_results/         (full evaluation)
  - checkpoint_comparison.txt   (checkpoint comparison)

EOF
