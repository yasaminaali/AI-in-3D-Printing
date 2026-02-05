#!/bin/bash
# Setup script for GPU support (RTX 4080)
# Run this to install PyTorch with CUDA 12.1 support

echo "Setting up PyTorch with CUDA support for RTX 4080..."
echo ""

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Setup complete! You can now run training on your RTX 4080."
