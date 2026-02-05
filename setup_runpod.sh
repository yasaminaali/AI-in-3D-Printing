#!/bin/bash
# RunPod RTX 5090 Setup Script for AI-in-3D-Printing
# Run this on your RunPod instance

set -e  # Exit on error

echo "=================================="
echo "RunPod RTX 5090 Setup"
echo "=================================="
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi

# Update system
echo ""
echo "[1/8] Updating system..."
apt-get update -qq

# Install git and other essentials
echo ""
echo "[2/8] Installing essentials..."
apt-get install -y -qq git wget curl

# Clone the repository
echo ""
echo "[3/8] Cloning repository..."
cd /workspace
if [ -d "AI-in-3D-Printing" ]; then
    echo "Repository already exists, pulling latest..."
    cd AI-in-3D-Printing
    git pull
else
    git clone https://github.com/anshjaiswal/AI-in-3D-Printing.git
    cd AI-in-3D-Printing
fi

# Install Python and pip if not present
echo ""
echo "[4/8] Setting up Python..."
python3 --version || apt-get install -y python3
pip3 --version || apt-get install -y python3-pip

# Create virtual environment
echo ""
echo "[5/8] Creating virtual environment..."
python3 -m venv nn_venv || python3 -m virtualenv nn_venv
source nn_venv/bin/activate

# Install PyTorch with CUDA 12.1 (for RTX 5090)
echo ""
echo "[6/8] Installing PyTorch with CUDA support..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo ""
echo "[7/8] Installing other dependencies..."
pip install -q rich pyyaml matplotlib numpy tqdm

# Verify installation
echo ""
echo "[8/8] Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
"

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Upload your dataset (nn_data/ folder) to /workspace/AI-in-3D-Printing/"
echo "2. Upload your checkpoint (nn_checkpoints/best_model.pt) if resuming"
echo "3. Run training: ./run_training.sh"
echo ""
echo "Or start fresh training:"
echo "   source nn_venv/bin/activate"
echo "   python model/train.py"
