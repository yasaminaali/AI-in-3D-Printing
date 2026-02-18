#!/bin/bash
#=============================================================
# SLURM Batch Script: Comparison Inference (Model+SA on ALL patterns)
# Forces model+SA on left_right/stripes (normally constructive)
# to demonstrate why the constructive approach is needed.
#=============================================================
#
# Submit with:   sbatch scripts/sbatch_inference_comparison.sh
# Check status:  squeue -u $USER
# View output:   tail -f comparison_inference_%j.out
#
#SBATCH --job-name=fusion_comp
#SBATCH --output=comparison_inference_%j.out
#SBATCH --error=comparison_inference_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --account=aip-rnishat
#SBATCH --qos=normal
#=============================================================

echo "=============================================="
echo "  Comparison: Model+SA on ALL Patterns"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  Date:      $(date)"
echo "=============================================="

# --- Module loading ---
module load python/3.11.5 cuda/12.6

# --- Navigate to repo directory ---
cd "$SLURM_SUBMIT_DIR" || cd ~/links/projects/aip-rnishat/shared/AI-in-3D-Printing

echo "Working directory: $(pwd)"
echo ""

# --- Activate virtual environment ---
if [ ! -d "sa_gpu_env" ]; then
    echo "ERROR: Virtual environment not found."
    exit 1
fi
source sa_gpu_env/bin/activate

echo "Python: $(which python3)"
echo ""

# --- Configuration ---
CHECKPOINT="${CHECKPOINT:-checkpoints/best.pt}"
TEST_JSONL="${TEST_JSONL:-datasets/final_dataset.jsonl}"
N_PER_PATTERN="${N_PER_PATTERN:-25}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/comparison}"

echo "Configuration:"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Test JSONL:  $TEST_JSONL"
echo "  N/pattern:   $N_PER_PATTERN"
echo "  Output dir:  $OUTPUT_DIR"
echo ""

# --- Verify files exist ---
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

if [ ! -f "$TEST_JSONL" ]; then
    echo "ERROR: Test JSONL not found at $TEST_JSONL"
    exit 1
fi

echo "Starting comparison inference (model+SA on ALL patterns)..."
echo ""

# --- Run comparison inference ---
PYTHONPATH="$(pwd)/src:$PYTHONPATH" python3 src/model/inference_comparison.py \
    --checkpoint "$CHECKPOINT" \
    --jsonl "$TEST_JSONL" \
    --n_per_pattern "$N_PER_PATTERN" \
    --visualize \
    --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "  Comparison inference finished at $(date)"
echo "  Exit code: $EXIT_CODE"
echo "  Results: $OUTPUT_DIR/comparison_results.json"
echo "  Visualizations: $OUTPUT_DIR/vis/"
echo "=============================================="

exit $EXIT_CODE
