#!/bin/bash
#=============================================================
# Submit 100K SA Dataset Generation — CPU Version
#
# Submits 12 independent CPU jobs (64 cores each, no GPUs).
# Uses the standard CPU SA pipeline with multiprocessing.
#
# Total: 100,000 trajectories (1250 seeds × 4 configs × 20 cells)
#
# Usage:
#   bash scripts/sbatch_gen100k_cpu.sh           # submit all 12 jobs
#   bash scripts/sbatch_gen100k_cpu.sh status     # check job status
#   bash scripts/sbatch_gen100k_cpu.sh merge      # merge outputs after completion
#=============================================================

set -e
cd "$(dirname "$0")/.." || exit 1

if [ "${1}" = "status" ]; then
    echo "=== Generation Job Status ==="
    squeue -u "$USER" --name="gen100k%" --format="%.10i %.15j %.2t %.10M %.6D %R"
    echo ""
    echo "=== Output Files ==="
    find output/gen100k -name "Dataset.jsonl" -exec wc -l {} + 2>/dev/null || echo "No output yet"
    echo ""
    total=$(find output/gen100k -name "Dataset.jsonl" -exec cat {} + 2>/dev/null | wc -l)
    echo "Total trajectories so far: $total / 100,000"
    exit 0
fi

if [ "${1}" = "merge" ]; then
    echo "=== Merging all Dataset.jsonl files ==="
    mkdir -p datasets
    cat output/gen100k/*/Dataset.jsonl > datasets/gen100k_dataset.jsonl
    total=$(wc -l < datasets/gen100k_dataset.jsonl)
    echo "Merged: $total trajectories -> datasets/gen100k_dataset.jsonl"
    ls -lh datasets/gen100k_dataset.jsonl
    exit 0
fi

CONFIGS=(
    "gen100k_30x30"
    "gen100k_50x50"
    "gen100k_60x60_a"
    "gen100k_60x60_b"
    "gen100k_80x80_lr"
    "gen100k_80x80_str"
    "gen100k_80x80_vor"
    "gen100k_80x80_isl"
    "gen100k_100x100_lr"
    "gen100k_100x100_str"
    "gen100k_100x100_vor"
    "gen100k_100x100_isl"
)

TIMES=(
    "08:00:00"   # 30x30: ~4 hrs
    "08:00:00"   # 50x50: ~5 hrs
    "08:00:00"   # 60x60_a: ~7 hrs
    "08:00:00"   # 60x60_b: ~7 hrs
    "08:00:00"   # 80x80_lr: ~7 hrs
    "08:00:00"   # 80x80_str: ~7 hrs
    "08:00:00"   # 80x80_vor: ~7 hrs
    "08:00:00"   # 80x80_isl: ~7 hrs
    "08:00:00"   # 100x100_lr: ~6 hrs
    "08:00:00"   # 100x100_str: ~6 hrs
    "08:00:00"   # 100x100_vor: ~6 hrs
    "08:00:00"   # 100x100_isl: ~6 hrs
)

echo "Submitting 12 CPU generation jobs..."
echo ""

for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    TIME="${TIMES[$i]}"

    JOB_ID=$(sbatch --parsable <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=gen100k_${CFG}
#SBATCH --output=gen100k_${CFG}_%j.out
#SBATCH --error=gen100k_${CFG}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=${TIME}
#SBATCH --account=aip-rnishat
#SBATCH --qos=normal

echo "=============================================="
echo "  100K Generation (CPU): ${CFG}"
echo "  Job ID:   \$SLURM_JOB_ID"
echo "  Node:     \$SLURM_NODELIST"
echo "  Cores:    \$SLURM_CPUS_PER_TASK"
echo "  Date:     \$(date)"
echo "=============================================="

module load python/3.11.5

cd "\$SLURM_SUBMIT_DIR" || cd ~/links/projects/aip-rnishat/shared/AI-in-3D-Printing
source sa_gpu_env/bin/activate

export PYTHONPATH="\$(pwd)/src:\$PYTHONPATH"

echo "Starting SA generation: ${CFG}"
echo ""

python3 src/run_pipeline.py ${CFG} --config-dir config --skip-venv-check

echo ""
echo "${CFG} completed at \$(date)"

if [ -f "output/gen100k/${CFG}/Dataset.jsonl" ]; then
    COUNT=\$(wc -l < "output/gen100k/${CFG}/Dataset.jsonl")
    echo "Generated: \$COUNT trajectories"
fi
SBATCH_EOF
)

    echo "  [$((i+1))/12] ${CFG} -> Job ${JOB_ID} (${TIME})"
done

echo ""
echo "All 12 CPU jobs submitted."
echo ""
echo "Monitor:  bash scripts/sbatch_gen100k_cpu.sh status"
echo "Merge:    bash scripts/sbatch_gen100k_cpu.sh merge"
