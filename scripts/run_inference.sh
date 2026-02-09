#!/bin/bash
# Inference Launcher for Linux
# Usage: ./run_inference.sh [options]
#   --checkpoint PATH    Model checkpoint (default: nn_checkpoints/best_model.pt)
#   --grid-W N          Grid width (default: 30)
#   --grid-H N          Grid height (default: 30)
#   --zone-pattern P    Pattern: left_right, stripes, voronoi, checkerboard
#   --initial-path T    Initial path: zigzag (default)
#   --max-ops N         Max operations (default: 100)

# Default values
CHECKPOINT="nn_checkpoints/best_model.pt"
GRID_W=30
GRID_H=30
ZONE_PATTERN="left_right"
INITIAL_PATH="zigzag"
MAX_OPS=100

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --grid-W)
            GRID_W="$2"
            shift 2
            ;;
        --grid-H)
            GRID_H="$2"
            shift 2
            ;;
        --zone-pattern)
            ZONE_PATTERN="$2"
            shift 2
            ;;
        --initial-path)
            INITIAL_PATH="$2"
            shift 2
            ;;
        --max-ops)
            MAX_OPS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run_inference.sh [options]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH   Model checkpoint path"
            echo "  --grid-W N         Grid width (default: 30)"
            echo "  --grid-H N         Grid height (default: 30)"
            echo "  --zone-pattern P   Zone pattern: left_right, stripes, voronoi, checkerboard"
            echo "  --initial-path T   Initial Hamiltonian path: zigzag"
            echo "  --max-ops N        Maximum operations to predict (default: 100)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "[Neural Network Inference]"
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

# Check checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Available checkpoints:"
    ls -1 nn_checkpoints/ 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Grid: ${GRID_W}x${GRID_H}"
echo "  Zone pattern: $ZONE_PATTERN"
echo "  Initial path: $INITIAL_PATH"
echo "  Max operations: $MAX_OPS"
echo ""

echo "Running inference..."
echo ""

python model/inference.py \
    --checkpoint "$CHECKPOINT" \
    --grid-W "$GRID_W" \
    --grid-H "$GRID_H" \
    --zone-pattern "$ZONE_PATTERN"

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Inference failed!"
    exit 1
fi

echo ""
echo "[OK] Inference complete!"

# Display results if file exists
if [ -f "inference_result.json" ]; then
    echo ""
    echo "Results:"
    python -m json.tool inference_result.json
fi
