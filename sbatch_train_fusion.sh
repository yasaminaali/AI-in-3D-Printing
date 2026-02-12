#!/bin/bash
#=============================================================
# SLURM Batch Script: Train FusionNet v2 with DDP (4x H100)
# Uses torchrun for DistributedDataParallel across 4 GPUs
#=============================================================
#
# Prerequisites:
#   1. Run sbatch_build_fusion_data.sh first to create fusion_data.pt
#   2. Ensure sa_gpu_env is set up (bash setup_env.sh)
#
# Submit with:   sbatch sbatch_train_fusion.sh
# Check status:  squeue -u $USER
# Cancel job:    scancel <job_id>
# View output:   tail -f fusion_train_%j.out
#
# To resume from a checkpoint:
#   RESUME_CKPT=FusionModel/nn_checkpoints/fusion/checkpoint_epoch_50.pt sbatch sbatch_train_fusion.sh
#
#SBATCH --job-name=fusion_train
#SBATCH --output=fusion_train_%j.out
#SBATCH --error=fusion_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=aip-rnishat
#SBATCH --qos=normal
#=============================================================

echo "=============================================="
echo "  FusionNet v2 — DDP Training (4x H100)"
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

# --- Verify training data exists ---
DATA_PT="FusionModel/fusion/fusion_data.pt"
if [ ! -f "$DATA_PT" ]; then
    echo "ERROR: Training data not found at $DATA_PT"
    echo "Run sbatch_build_fusion_data.sh first."
    exit 1
fi
echo ""
echo "Training data: $DATA_PT ($(ls -lh "$DATA_PT" | awk '{print $5}'))"
echo ""

# --- Training configuration ---
# Override any of these via environment variables before sbatch submission
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-4e-4}"
PATIENCE="${PATIENCE:-40}"
STAGE="${STAGE:-4}"

echo "Configuration:"
echo "  Epochs:     $EPOCHS"
echo "  Batch/GPU:  $BATCH_SIZE"
echo "  LR:         $LR"
echo "  Patience:   $PATIENCE"
echo "  Stage:      $STAGE"
echo ""

# --- Build extra args ---
EXTRA_ARGS=""
if [ -n "$RESUME_CKPT" ]; then
    echo "Resuming from checkpoint: $RESUME_CKPT"
    EXTRA_ARGS="$EXTRA_ARGS --resume $RESUME_CKPT"
fi

echo "Starting DDP training with torchrun (4 GPUs)..."
echo ""

# --- Launch with torchrun for DDP ---
PYTHONPATH="$(pwd):$PYTHONPATH" torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    FusionModel/fusion/train_fusion.py \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --patience "$PATIENCE" \
    $EXTRA_ARGS

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "  Training finished at $(date)"
echo "  Exit code: $EXIT_CODE"
echo ""
echo "  Checkpoints saved to: FusionModel/nn_checkpoints/fusion/"
ls -lht FusionModel/nn_checkpoints/fusion/*.pt 2>/dev/null | head -5
echo "=============================================="

exit $EXIT_CODE
