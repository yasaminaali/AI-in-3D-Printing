#!/bin/bash
#=============================================================
# SLURM Batch Script for GA Dataset Generation on CCRI TamIA
# 4x NVIDIA H100 SXM 80GB
#=============================================================
#
# First time setup (run ONCE on login node before submitting):
#   bash setup_env.sh
#
# Submit with:   sbatch sbatch_kazi_ga_gpu.sh
# Check status:  squeue -u $USER
# Cancel job:    scancel <job_id>
# View output:   tail -f ga_gen_kazi_%j.out
#
#SBATCH --job-name=ga_gen_kazi
#SBATCH --output=ga_gen_kazi_%j.out
#SBATCH --error=ga_gen_kazi_%j.err
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
echo "  GA Dataset Generation - GPU Pipeline"
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

# --- Warm up Numba JIT (compile once, not during each GA run) ---
echo ""
echo "Warming up Numba JIT compilation..."
python3 -c "from numba_ops import fast_validate_path; print('Numba JIT ready')"

echo ""
echo "Backing up existing GA output (60x60 + 100x100) before 80x80 run..."
if [ -f "output/kazi_ga/Dataset.jsonl" ]; then
    cp output/kazi_ga/Dataset.jsonl output/kazi_ga/Dataset_60_100.jsonl
    echo "  Backed up -> output/kazi_ga/Dataset_60_100.jsonl"
fi
# Reset checkpoint so pipeline runs fresh 80x80 tasks
rm -f output/kazi_ga/checkpoint_kazi_ga.json

echo ""
echo "Starting GA pipeline with 4 GPUs (80x80 only)..."
echo ""

# --- Run the GPU pipeline with kazi_ga config ---
python3 run_ga_pipeline_gpu.py kazi_ga \
    --gpus 4 \
    --config-dir config

EXIT_CODE=$?

# --- Merge 80x80 GA output with existing 60x60 + 100x100 GA output ---
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Merging GA outputs into combined file..."
    python3 -c "
import os
combined = 'output/kazi_ga/ga_combined_dataset.jsonl'
sources = ['output/kazi_ga/Dataset.jsonl']
# Also include any previous GA output backup
prev = 'output/kazi_ga/Dataset_60_100.jsonl'
if os.path.exists(prev):
    sources.insert(0, prev)

total = 0
with open(combined, 'w', encoding='utf-8') as out:
    for src in sources:
        if not os.path.exists(src):
            print(f'  SKIP (not found): {src}')
            continue
        count = 0
        with open(src, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    out.write(line)
                    count += 1
        total += count
        print(f'  {src}: {count} records')
print(f'  Combined: {total} records -> {combined}')
"
fi

echo ""
echo "=============================================="
echo "  GA Pipeline finished at $(date)"
echo "  Exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE
