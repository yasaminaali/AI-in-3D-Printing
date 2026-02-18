#!/bin/bash
#=============================================================
# SLURM Batch Script for SA Dataset Generation on CCRI TamIA
# 4x NVIDIA H100 SXM 80GB
#=============================================================
#
# First time setup (run ONCE on login node before submitting):
#   bash scripts/setup_env.sh
#
# Submit with:   sbatch scripts/sbatch_kazi_gpu.sh
# Check status:  squeue -u $USER
# Cancel job:    scancel <job_id>
# View output:   tail -f sa_gen_kazi_<job_id>.out
#
#SBATCH --job-name=sa_gen_kazi
#SBATCH --output=sa_gen_kazi_%j.out
#SBATCH --error=sa_gen_kazi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=aip-rnishat
#SBATCH --qos=normal
#=============================================================

echo "=============================================="
echo "  SA Dataset Generation - GPU Pipeline"
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
print(f'CUDA devices: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  cuda:{i} - {p.name} ({p.total_memory / 1024**3:.0f} GB)')
"

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch/CUDA check failed."
    exit 1
fi

# --- Warm up Numba JIT (compile once, not during each SA run) ---
echo ""
echo "Warming up Numba JIT compilation..."
python3 -c "from numba_ops import fast_validate_path; print('Numba JIT ready')"

echo ""
echo "Starting pipeline with 4 GPUs..."
echo ""

# --- Run the GPU pipeline with kazi config ---
python3 src/run_pipeline_gpu.py kazi \
    --gpus 4 \
    --config-dir config

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "  Pipeline finished at $(date)"
echo "  Exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE
