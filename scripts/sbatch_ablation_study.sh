#!/bin/bash
#=============================================================
# SLURM Batch Script: Ablation Study — Train & Evaluate 4 Models
#
# Submits 4 independent jobs (one per model variant) so each fits
# within the 24-hour TamIA walltime limit. After all 4 finish,
# submit the comparison job manually or re-run this script.
#
# Models:
#   1. CNN-Only         — encoder + simple decoder, no RNN, no attention
#   2. U-Net Spatial    — full U-Net + attention, no RNN
#   3. CNN+RNN          — encoder + GRU, concat fusion, no U-Net skips
#   4. U-Net+RNN Concat — full U-Net + GRU, concat at bottleneck (no FiLM)
#
# Usage:
#   bash scripts/sbatch_ablation_study.sh          # submits all 4
#   bash scripts/sbatch_ablation_study.sh compare   # run comparison only
#
# Prerequisites:
#   checkpoints/fusion_data.pt and datasets/final_dataset.jsonl must exist.
#=============================================================

set -e
cd "$(dirname "$0")/.." || exit 1

if [ "${1}" = "compare" ]; then
    echo "Running comparison summary..."
    module load python/3.11.5 2>/dev/null || true
    source sa_gpu_env/bin/activate
    export PYTHONPATH="$(pwd)/src:$(pwd)/src/model:$(pwd)/src/model/ablations:$PYTHONPATH"
    python3 src/model/ablations/compare_results.py \
        --ablation_dir checkpoints/ablations \
        --fusionnet_dir checkpoints \
        --output checkpoints/ablations/ablation_summary.json
    exit 0
fi

VARIANTS=("cnn_only" "unet_spatial" "cnn_rnn" "unet_rnn_concat")
VARIANT_NAMES=("CNN-Only" "U-Net_Spatial" "CNN+RNN" "U-Net+RNN_Concat")
PORTS=(29500 29501 29502 29503)

echo "Submitting 4 ablation jobs to SLURM..."
echo ""

for i in "${!VARIANTS[@]}"; do
    VARIANT="${VARIANTS[$i]}"
    NAME="${VARIANT_NAMES[$i]}"
    PORT="${PORTS[$i]}"

    JOB_ID=$(sbatch --parsable <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=abl_${VARIANT}
#SBATCH --output=ablation_${VARIANT}_%j.out
#SBATCH --error=ablation_${VARIANT}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --account=aip-rnishat
#SBATCH --qos=normal

echo "=============================================="
echo "  Ablation: ${NAME}"
echo "  Job ID:   \$SLURM_JOB_ID"
echo "  Node:     \$SLURM_NODELIST"
echo "  GPUs:     \$CUDA_VISIBLE_DEVICES"
echo "  Date:     \$(date)"
echo "=============================================="

module load python/3.11.5 cuda/12.6

cd "\$SLURM_SUBMIT_DIR" || cd ~/links/projects/aip-rnishat/shared/AI-in-3D-Printing
echo "Working directory: \$(pwd)"

if [ ! -d "sa_gpu_env" ]; then
    echo "ERROR: Virtual environment not found."
    exit 1
fi
source sa_gpu_env/bin/activate

python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
"
if [ \$? -ne 0 ]; then exit 1; fi

DATA_PT="checkpoints/fusion_data.pt"
TEST_JSONL="datasets/final_dataset.jsonl"
if [ ! -f "\$DATA_PT" ]; then echo "ERROR: \$DATA_PT not found"; exit 1; fi
if [ ! -f "\$TEST_JSONL" ]; then echo "ERROR: \$TEST_JSONL not found"; exit 1; fi

export PYTHONPATH="\$(pwd)/src:\$(pwd)/src/model:\$(pwd)/src/model/ablations:\$PYTHONPATH"

EPOCHS="\${EPOCHS:-200}"
BATCH_SIZE="\${BATCH_SIZE:-64}"
LR="\${LR:-4e-4}"
PATIENCE="\${PATIENCE:-40}"
N_PER_PATTERN="\${N_PER_PATTERN:-25}"
CKPT_DIR="checkpoints/ablations/${VARIANT}"

mkdir -p "\$CKPT_DIR"

echo ""
echo "=== Training ${NAME} ==="
echo "  Epochs: \$EPOCHS | Batch: \$BATCH_SIZE | LR: \$LR | Patience: \$PATIENCE"
echo ""

torchrun \\
    --nproc_per_node=4 \\
    --master_port=${PORT} \\
    src/model/ablations/${VARIANT}/train.py \\
    --data_path "\$DATA_PT" \\
    --checkpoint_dir "\$CKPT_DIR" \\
    --epochs "\$EPOCHS" \\
    --batch_size "\$BATCH_SIZE" \\
    --learning_rate "\$LR" \\
    --patience "\$PATIENCE" \\
    --num_workers 8

TRAIN_EXIT=\$?
echo "Training exit code: \$TRAIN_EXIT"

if [ \$TRAIN_EXIT -ne 0 ] || [ ! -f "\$CKPT_DIR/best.pt" ]; then
    echo "ERROR: Training failed or no best.pt produced."
    exit 1
fi

echo ""
echo "=== Inference ${NAME} ==="
echo ""

python3 src/model/ablations/${VARIANT}/inference.py \\
    --checkpoint "\$CKPT_DIR/best.pt" \\
    --jsonl "\$TEST_JSONL" \\
    --n_per_pattern "\$N_PER_PATTERN" \\
    --output_dir "\$CKPT_DIR" \\
    --visualize

echo ""
echo "${NAME} completed at \$(date)"
SBATCH_EOF
)

    echo "  [$((i+1))/4] ${NAME} -> Job ${JOB_ID}"
done

echo ""
echo "All 4 jobs submitted. Monitor with: squeue -u \$USER"
echo "When all complete, run:  bash scripts/sbatch_ablation_study.sh compare"
