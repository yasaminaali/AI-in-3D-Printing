#!/bin/bash
#=============================================================
# SLURM Batch Script: Build FusionNet Training Data
# CPU-only job — replays SA trajectories with safe methods
#=============================================================
#
# Submit with:   sbatch sbatch_build_fusion_data.sh
# Check status:  squeue -u $USER
# Cancel job:    scancel <job_id>
# View output:   tail -f fusion_build_data_%j.out
#
#SBATCH --job-name=fusion_build
#SBATCH --output=fusion_build_data_%j.out
#SBATCH --error=fusion_build_data_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --account=aip-rnishat
#SBATCH --qos=normal
#=============================================================

echo "=============================================="
echo "  FusionNet — Build Training Data"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  CPUs:      $SLURM_CPUS_PER_TASK"
echo "  Date:      $(date)"
echo "=============================================="

# --- Module loading ---
module load python/3.11.5

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

# --- Paths ---
# Default: uses final_dataset.jsonl from the datasets directory
# Override with environment variable: DATASET_JSONL=/path/to/file.jsonl
DATASET_JSONL="${DATASET_JSONL:-datasets/final_dataset.jsonl}"
OUTPUT_PT="FusionModel/fusion/fusion_data.pt"

echo "Input JSONL:  $DATASET_JSONL"
echo "Output .pt:   $OUTPUT_PT"
echo ""

# --- Build training data with safe replay methods ---
PYTHONPATH="$(pwd):$PYTHONPATH" python3 FusionModel/fusion/build_fusion_data.py \
    --input "$DATASET_JSONL" \
    --output "$OUTPUT_PT"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "  Data build finished at $(date)"
echo "  Exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Output: $OUTPUT_PT"
    ls -lh "$OUTPUT_PT"
fi
echo "=============================================="

exit $EXIT_CODE
