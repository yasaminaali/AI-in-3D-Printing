#!/bin/bash
#=============================================================
# SLURM Batch Script: FusionNet Inference Evaluation (1 GPU)
# Runs inference on test samples and reports performance
#=============================================================
#
# Prerequisites:
#   1. Trained model checkpoint in FusionModel/nn_checkpoints/fusion/
#   2. Test dataset JSONL file
#
# Submit with:   sbatch sbatch_inference_fusion.sh
# Check status:  squeue -u $USER
# Cancel job:    scancel <job_id>
# View output:   tail -f fusion_inference_%j.out
#
#SBATCH --job-name=fusion_infer
#SBATCH --output=fusion_inference_%j.out
#SBATCH --error=fusion_inference_%j.err
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
echo "  FusionNet v5 — Constructive + Model-Only (No SA)"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  GPUs:      $CUDA_VISIBLE_DEVICES"
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
    echo "ERROR: Virtual environment not found. Run setup_env.sh first on the login node."
    exit 1
fi
source sa_gpu_env/bin/activate

echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo ""

# --- Check CUDA availability ---
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'  cuda:0 - {p.name} ({p.total_memory / 1024**3:.0f} GB)')
"

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch/CUDA check failed."
    exit 1
fi

# --- Configuration ---
# Override via environment variables:
#   CHECKPOINT=/path/to/best.pt  TEST_JSONL=/path/to/test.jsonl  N_PER_PATTERN=25
CHECKPOINT="${CHECKPOINT:-FusionModel/nn_checkpoints/fusion/best.pt}"
TEST_JSONL="${TEST_JSONL:-datasets/final_dataset.jsonl}"
N_PER_PATTERN="${N_PER_PATTERN:-25}"

echo "Configuration:"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Test JSONL:  $TEST_JSONL"
echo "  N/pattern:   $N_PER_PATTERN"
echo ""

# --- Verify files exist ---
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Available checkpoints:"
    ls -lht FusionModel/nn_checkpoints/fusion/*.pt 2>/dev/null | head -5
    exit 1
fi

if [ ! -f "$TEST_JSONL" ]; then
    echo "ERROR: Test JSONL not found at $TEST_JSONL"
    exit 1
fi

echo "Starting inference evaluation..."
echo ""

# --- Run inference ---
PYTHONPATH="$(pwd):$PYTHONPATH" python3 FusionModel/fusion/inference_fusion.py \
    --checkpoint "$CHECKPOINT" \
    --jsonl "$TEST_JSONL" \
    --n_per_pattern "$N_PER_PATTERN" \
    --visualize \
    --vis_dir "FusionModel/nn_checkpoints/fusion/vis"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "  Inference finished at $(date)"
echo "  Exit code: $EXIT_CODE"
echo "  Results: FusionModel/nn_checkpoints/fusion/inference_results.json"
echo "  Visualizations: FusionModel/nn_checkpoints/fusion/vis/"
echo "=============================================="

exit $EXIT_CODE
