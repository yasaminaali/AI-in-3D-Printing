"""
Quick diagnostic script to identify training bottlenecks
"""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.models.cnn_rnn import CNNRNNHamiltonian
from model.data.dataset import HamiltonianDataset, collate_fn
from torch.utils.data import DataLoader
import yaml

print("="*80)
print("TRAINING BOTTLENECK DIAGNOSTICS")
print("="*80)
print()

# Check GPU
print("1. GPU Status:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# Load config
with open('model/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Check sequence length
seq_len = config['model']['predictor']['sequence_length']
print(f"2. Model Configuration:")
print(f"   Sequence Length: {seq_len}")
print(f"   Max Positions: {config['model']['predictor']['max_positions']}")
print(f"   Batch Size (30x30): {config['training']['batch_sizes']['30x30']}")
print()

# Load a sample batch and time it
print("3. Testing Data Loading Speed:")
try:
    dataset = HamiltonianDataset('nn_data/train_all.jsonl', max_seq_len=seq_len, max_grid_size=30)
    print(f"   Dataset size: {len(dataset)} records")
    
    # Time loading first 10 samples
    start = time.time()
    for i in range(min(10, len(dataset))):
        _ = dataset[i]
    load_time = time.time() - start
    print(f"   Time to load 10 samples: {load_time:.2f}s ({load_time/10:.2f}s per sample)")
    
    # Check sequence lengths in dataset
    seq_lengths = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        seq_lengths.append(sample['seq_len'])
    
    print(f"   Average sequence length: {sum(seq_lengths)/len(seq_lengths):.1f}")
    print(f"   Max sequence length: {max(seq_lengths)}")
    print(f"   Min sequence length: {min(seq_lengths)}")
    
except Exception as e:
    print(f"   ERROR: {e}")
print()

# Test model forward pass speed
print("4. Testing Model Forward Pass Speed:")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNRNNHamiltonian(config).to(device)
    model.eval()
    
    batch_size = config['training']['batch_sizes']['30x30']
    
    # Create dummy input
    dummy_grid = torch.randn(batch_size, seq_len, 4, 30, 30).to(device)
    dummy_global = torch.randn(batch_size, seq_len, 3).to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model(dummy_grid, dummy_global)
    
    # Time it
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_grid, dummy_global)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    forward_time = (time.time() - start) / 10
    
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Grid size: 30x30")
    print(f"   Average forward pass: {forward_time:.3f}s")
    print(f"   Estimated batches/epoch: ~45")
    print(f"   Estimated training time/epoch: ~{forward_time * 45:.1f}s ({forward_time * 45 / 60:.1f} min)")
    
except Exception as e:
    print(f"   ERROR: {e}")
print()

# Check for common issues
print("5. Potential Issues:")
issues = []

if seq_len > 200:
    issues.append(f"   [!] Sequence length ({seq_len}) is very long - this slows down training significantly")

if not torch.cuda.is_available():
    issues.append("   [!] CUDA not available - training on CPU will be extremely slow")
else:
    # Check if model is actually using GPU
    test_tensor = torch.randn(100, 100).to('cuda')
    if test_tensor.device.type != 'cuda':
        issues.append("   [!] Tensors not moving to GPU properly")

if len(dataset) > 2000:
    issues.append(f"   [!] Large dataset ({len(dataset)} samples) - consider using subset for testing")

if not issues:
    print("   [OK] No obvious issues detected")
else:
    for issue in issues:
        print(issue)

print()
print("="*80)
print("RECOMMENDATIONS:")
print("="*80)
print("If training is slow:")
print("1. Check that GPU is being used (should show ~80-100% utilization)")
print("2. Reduce sequence_length in config if >200 (try 100)")
print("3. Use a subset of data for testing: head -n 500 nn_data/train_all.jsonl > nn_data/train_subset.jsonl")
print("4. Increase batch_size if you have GPU memory available")
print("5. Disable validation during testing: validate_every_n_epochs: 10")
print("="*80)
