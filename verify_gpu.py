"""
Verify GPU is being used for training
"""

import sys
sys.path.insert(0, '.')

import torch
import yaml
from model.data.dataset import HamiltonianDataset, collate_fn
from model.models.cnn_rnn import CNNRNNHamiltonian
from torch.utils.data import DataLoader

print("="*60)
print("GPU VERIFICATION TEST")
print("="*60)

# Check GPU availability
print("\n[1/5] Checking GPU availability...")
assert torch.cuda.is_available(), "CUDA not available!"
device = torch.device('cuda')
print(f"  [OK] GPU: {torch.cuda.get_device_name(0)}")
print(f"  [OK] CUDA version: {torch.version.cuda}")

# Check GPU memory
print("\n[2/5] Checking GPU memory...")
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  [OK] Total GPU memory: {gpu_mem:.1f} GB")

# Load model to GPU
print("\n[3/5] Loading model to GPU...")
with open('model/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = CNNRNNHamiltonian(config).to(device)
print(f"  [OK] Model on GPU: {next(model.parameters()).device}")

# Check GPU memory usage
print("\n[4/5] Checking GPU memory usage...")
torch.cuda.reset_peak_memory_stats()
memory_allocated = torch.cuda.memory_allocated() / 1e6
print(f"  [OK] GPU memory allocated: {memory_allocated:.1f} MB")

# Test forward pass on GPU
print("\n[5/5] Testing forward pass on GPU...")
dummy_input = torch.randn(2, 500, 4, 100, 100).to(device)
dummy_global = torch.randn(2, 500, 3).to(device)

with torch.no_grad():
    output = model(dummy_input, dummy_global)

memory_after = torch.cuda.memory_allocated() / 1e6
peak_memory = torch.cuda.max_memory_allocated() / 1e6

print(f"  [OK] Forward pass completed")
print(f"  [OK] Memory after forward: {memory_after:.1f} MB")
print(f"  [OK] Peak memory usage: {peak_memory:.1f} MB")

print("\n" + "="*60)
print("[SUCCESS] GPU IS STRICTLY BEING USED!")
print("="*60)
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {gpu_mem:.1f} GB total, {peak_memory:.1f} MB peak usage")
