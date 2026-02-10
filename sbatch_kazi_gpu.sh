#!/bin/bash
#=============================================================
# SLURM Batch Script for SA Dataset Generation on CCRI TamIA
# 4x NVIDIA H100 SXM 80GB
#=============================================================
#
# Submit with:   sbatch sbatch_kazi_gpu.sh
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
#SBATCH --partition=gpubase_bynode_b1
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
# Uncomment the appropriate module commands for your CCRI TamIA setup.
# Common options:
# module load cuda/12.1
# module load pytorch/2.1
# module load anaconda3
# module load python/3.10

# If using a conda environment:
# conda activate sa_gpu

# --- Navigate to repo directory ---
# Adjust this path to where the repo is cloned on TamIA
cd "$SLURM_SUBMIT_DIR" || cd ~/links/projects/aip-rnishat/shared/AI-in-3D-Printing

echo "Working directory: $(pwd)"
echo ""

# --- Check CUDA availability ---
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  cuda:{i} - {p.name} ({p.total_mem / 1024**3:.0f} GB)')
"

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch/CUDA check failed. Ensure PyTorch with CUDA is installed."
    echo "Try: pip install torch --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

# --- Install dependencies if needed ---
pip install pyyaml rich tqdm numpy matplotlib numba 2>/dev/null

# --- Warm up Numba JIT (compile once, not during each SA run) ---
echo "Warming up Numba JIT compilation..."
python3 -c "from numba_ops import fast_validate_path; print('Numba JIT ready')"

echo ""
echo "Starting pipeline with 4 GPUs..."
echo ""

# --- Run the GPU pipeline with kazi config ---
python3 run_pipeline_gpu.py kazi \
    --gpus 4 \
    --config-dir config

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "  Pipeline finished at $(date)"
echo "  Exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE
