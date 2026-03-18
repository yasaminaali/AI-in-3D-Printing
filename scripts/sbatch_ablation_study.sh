#!/bin/bash
#=============================================================
# SLURM Batch Script: Ablation Study — Train & Evaluate 4 Models
#
# Trains all 4 ablation variants sequentially on the same dataset,
# runs inference on each, then produces a comparison summary.
#
# Models (in order of increasing complexity):
#   1. CNN-Only         — encoder + simple decoder, no RNN, no attention
#   2. U-Net Spatial    — full U-Net + attention, no RNN
#   3. CNN+RNN          — encoder + GRU, concat fusion, no U-Net skips
#   4. U-Net+RNN Concat — full U-Net + GRU, concat at bottleneck (no FiLM)
#
# The 5th model (FusionNet with FiLM) is already trained in checkpoints/.
#
# Prerequisites:
#   1. checkpoints/fusion_data.pt must exist (run sbatch_build_fusion_data.sh)
#   2. datasets/final_dataset.jsonl must exist (for inference)
#   3. sa_gpu_env must be set up (bash scripts/setup_env.sh)
#
# Submit: sbatch scripts/sbatch_ablation_study.sh
#=============================================================
#SBATCH --job-name=ablation_study
#SBATCH --output=ablation_study_%j.out
#SBATCH --error=ablation_study_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --account=aip-rnishat
#SBATCH --qos=normal
#=============================================================

echo "=============================================="
echo "  Ablation Study — 4 Model Variants"
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

# --- Activate virtual environment ---
if [ ! -d "sa_gpu_env" ]; then
    echo "ERROR: Virtual environment not found. Run setup_env.sh first."
    exit 1
fi
source sa_gpu_env/bin/activate

echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"

# --- Check CUDA ---
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  cuda:{i} - {p.name} ({p.total_memory / 1024**3:.0f} GB)')
"
if [ $? -ne 0 ]; then
    echo "ERROR: CUDA check failed."
    exit 1
fi

# --- Verify data exists ---
DATA_PT="checkpoints/fusion_data.pt"
TEST_JSONL="datasets/final_dataset.jsonl"

if [ ! -f "$DATA_PT" ]; then
    echo "ERROR: Training data not found at $DATA_PT"
    exit 1
fi
if [ ! -f "$TEST_JSONL" ]; then
    echo "ERROR: Test data not found at $TEST_JSONL"
    exit 1
fi

echo ""
echo "Training data: $DATA_PT ($(ls -lh "$DATA_PT" | awk '{print $5}'))"
echo ""

# --- Training configuration ---
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-4e-4}"
PATIENCE="${PATIENCE:-40}"
N_PER_PATTERN="${N_PER_PATTERN:-25}"

echo "Configuration:"
echo "  Epochs:        $EPOCHS"
echo "  Batch/GPU:     $BATCH_SIZE"
echo "  LR:            $LR"
echo "  Patience:      $PATIENCE"
echo "  N per pattern: $N_PER_PATTERN"
echo ""

# --- Set PYTHONPATH ---
export PYTHONPATH="$(pwd)/src:$(pwd)/src/model:$(pwd)/src/model/ablations:$PYTHONPATH"

# --- Define variants ---
VARIANTS=("cnn_only" "unet_spatial" "cnn_rnn" "unet_rnn_concat")
VARIANT_NAMES=("CNN-Only" "U-Net Spatial" "CNN+RNN" "U-Net+RNN (Concat)")
PORTS=(29500 29501 29502 29503)

FAILED=0

for i in "${!VARIANTS[@]}"; do
    VARIANT="${VARIANTS[$i]}"
    NAME="${VARIANT_NAMES[$i]}"
    PORT="${PORTS[$i]}"
    CKPT_DIR="checkpoints/ablations/$VARIANT"

    echo ""
    echo "######################################################"
    echo "  [$((i+1))/4] TRAINING: $NAME"
    echo "  Checkpoint dir: $CKPT_DIR"
    echo "  Master port: $PORT"
    echo "  Started at: $(date)"
    echo "######################################################"
    echo ""

    mkdir -p "$CKPT_DIR"

    torchrun \
        --nproc_per_node=4 \
        --master_port=$PORT \
        src/model/ablations/$VARIANT/train.py \
        --data_path "$DATA_PT" \
        --checkpoint_dir "$CKPT_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LR" \
        --patience "$PATIENCE" \
        --num_workers 8

    TRAIN_EXIT=$?
    echo ""
    echo "  $NAME training exit code: $TRAIN_EXIT"

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "  WARNING: $NAME training failed, skipping inference."
        FAILED=$((FAILED + 1))
        continue
    fi

    # --- Inference ---
    if [ ! -f "$CKPT_DIR/best.pt" ]; then
        echo "  WARNING: No best.pt found for $NAME, skipping inference."
        FAILED=$((FAILED + 1))
        continue
    fi

    echo ""
    echo "  --- Running inference for $NAME ---"
    echo ""

    python3 src/model/ablations/$VARIANT/inference.py \
        --checkpoint "$CKPT_DIR/best.pt" \
        --jsonl "$TEST_JSONL" \
        --n_per_pattern "$N_PER_PATTERN" \
        --output_dir "$CKPT_DIR" \
        --visualize

    INF_EXIT=$?
    echo "  $NAME inference exit code: $INF_EXIT"

    if [ $INF_EXIT -ne 0 ]; then
        echo "  WARNING: $NAME inference failed."
        FAILED=$((FAILED + 1))
    fi

    echo ""
    echo "  $NAME completed at $(date)"
    echo ""
done

# --- Comparison summary ---
echo ""
echo "######################################################"
echo "  COMPARISON SUMMARY"
echo "######################################################"
echo ""

python3 src/model/ablations/compare_results.py \
    --ablation_dir checkpoints/ablations \
    --fusionnet_dir checkpoints \
    --output checkpoints/ablations/ablation_summary.json

echo ""
echo "=============================================="
echo "  Ablation Study Complete"
echo "  Finished at: $(date)"
echo "  Failed models: $FAILED / 4"
echo ""
echo "  Results:"
ls -lh checkpoints/ablations/*/best.pt 2>/dev/null
ls -lh checkpoints/ablations/*/training_log.csv 2>/dev/null
ls -lh checkpoints/ablations/*/inference_results.json 2>/dev/null
echo ""
echo "  Summary: checkpoints/ablations/ablation_summary.json"
echo "=============================================="

exit $FAILED
