#!/bin/bash
#=============================================================
# SLURM Batch Script: Comparison Inference for All 5 Models
#
# Runs model+SA comparison inference on voronoi/islands patterns
# for all 4 ablation models + FusionNet, then produces comparison.
#
# Prerequisites: All models must be trained (best.pt must exist)
#
# Submit: sbatch scripts/sbatch_ablation_inference.sh
#=============================================================
#SBATCH --job-name=abl_infer
#SBATCH --output=ablation_inference_%j.out
#SBATCH --error=ablation_inference_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --account=aip-rnishat
#SBATCH --qos=normal
#=============================================================

echo "=============================================="
echo "  Ablation Comparison Inference — All 5 Models"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  Date:      $(date)"
echo "=============================================="

module load python/3.11.5 cuda/12.6

cd "$SLURM_SUBMIT_DIR" || cd ~/links/projects/aip-rnishat/shared/AI-in-3D-Printing
echo "Working directory: $(pwd)"

if [ ! -d "sa_gpu_env" ]; then
    echo "ERROR: Virtual environment not found."
    exit 1
fi
source sa_gpu_env/bin/activate

python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

TEST_JSONL="datasets/final_dataset.jsonl"
N_PER_PATTERN="${N_PER_PATTERN:-25}"

export PYTHONPATH="$(pwd)/src:$(pwd)/src/model:$(pwd)/src/model/ablations:$PYTHONPATH"

FAILED=0

# --- Ablation models: comparison inference ---
VARIANTS=("cnn_only" "unet_spatial" "cnn_rnn" "unet_rnn_concat")
VARIANT_NAMES=("CNN-Only" "U-Net Spatial" "CNN+RNN" "U-Net+RNN (Concat)")

for i in "${!VARIANTS[@]}"; do
    VARIANT="${VARIANTS[$i]}"
    NAME="${VARIANT_NAMES[$i]}"
    CKPT_DIR="checkpoints/ablations/$VARIANT"

    if [ ! -f "$CKPT_DIR/best.pt" ]; then
        echo "WARNING: $CKPT_DIR/best.pt not found, skipping $NAME"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo ""
    echo "######################################################"
    echo "  [$((i+1))/5] Comparison Inference: $NAME"
    echo "  Started at: $(date)"
    echo "######################################################"
    echo ""

    python3 "src/model/ablations/$VARIANT/inference.py" \
        --checkpoint "$CKPT_DIR/best.pt" \
        --jsonl "$TEST_JSONL" \
        --n_per_pattern "$N_PER_PATTERN" \
        --output_dir "$CKPT_DIR" \
        --comparison \
        --visualize

    EXIT_CODE=$?
    echo "  $NAME inference exit code: $EXIT_CODE"
    if [ $EXIT_CODE -ne 0 ]; then
        FAILED=$((FAILED + 1))
    fi
done

# --- FusionNet: comparison inference ---
echo ""
echo "######################################################"
echo "  [5/5] Comparison Inference: FusionNet (FiLM)"
echo "  Started at: $(date)"
echo "######################################################"
echo ""

if [ -f "checkpoints/best.pt" ]; then
    python3 src/model/inference_comparison.py \
        --checkpoint checkpoints/best.pt \
        --jsonl "$TEST_JSONL" \
        --n_per_pattern "$N_PER_PATTERN" \
        --output_dir checkpoints/comparison \
        --visualize

    EXIT_CODE=$?
    echo "  FusionNet inference exit code: $EXIT_CODE"
    if [ $EXIT_CODE -ne 0 ]; then
        FAILED=$((FAILED + 1))
    fi
else
    echo "WARNING: checkpoints/best.pt not found, skipping FusionNet"
    FAILED=$((FAILED + 1))
fi

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
echo "  Inference Complete at $(date)"
echo "  Failed: $FAILED / 5"
echo "=============================================="

exit $FAILED
