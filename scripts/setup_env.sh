#!/bin/bash
# One-time setup: creates virtual environment with PyTorch + CUDA on TamIA
# Run this ONCE on the login node before submitting sbatch jobs

set -e

echo "Loading modules..."
module load python/3.11.5 cuda/12.6

cd ~/links/projects/aip-rnishat/shared/AI-in-3D-Printing

echo "Creating virtual environment..."
python3 -m venv sa_gpu_env

echo "Activating environment..."
source sa_gpu_env/bin/activate

echo "Installing PyTorch with CUDA 12.6..."
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

echo "Installing other dependencies..."
pip install --no-cache-dir pyyaml rich tqdm numpy matplotlib numba

echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import numba; print(f'Numba {numba.__version__}')"

echo ""
echo "Setup complete! Now submit your job with:"
echo "  sbatch sbatch_kazi_gpu.sh"
