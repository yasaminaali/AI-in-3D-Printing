"""
Main Pipeline for AI-in-3D-Printing
Orchestrates data collection, training, and evaluation.
"""

import os
import sys
import time
import argparse
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed."""
    print("=" * 60)
    print("Checking Dependencies...")
    print("=" * 60)
    
    required = ['torch', 'numpy', 'matplotlib']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg} found")
        except ImportError:
            print(f"✗ {pkg} missing")
            missing.append(pkg)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n✓ GPU detected: {gpu_name}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠ No GPU detected - will use CPU (slower)")
    
    print("=" * 60)
    return True


def collect_data(args):
    """Run data collection phase using Simulated Annealing."""
    print("\n" + "=" * 60)
    print("PHASE 1: Data Collection")
    print("=" * 60)
    
    # Check if Collect_SA.py exists
    if not os.path.exists("Collect_SA_quick.py"):
        print("Error: Collect_SA_quick.py not found!")
        return False
    
    # Create Dataset directory if it doesn't exist
    os.makedirs("Dataset", exist_ok=True)
    os.makedirs("Dataset/features", exist_ok=True)
    
    if args.skip_collection:
        print("\n⏭ Skipping data collection (--skip-collection flag set)")
        
        # Check if dataset exists
        if not os.path.exists("Dataset/states.csv") or not os.path.exists("Dataset/actions.csv"):
            print("\n⚠ Warning: Dataset files not found!")
            print("  Expected: Dataset/states.csv and Dataset/actions.csv")
            print("  Run without --skip-collection to generate data.")
            return False
        else:
            print("✓ Found existing dataset files")
            return True
    
    print("\nStarting QUICK TEST data collection with Simulated Annealing...")
    print(f"Configuration:")
    print(f"  - Grid sizes: 10x10, 15x15")
    print(f"  - Zone patterns: left_right, diagonal")
    print(f"  - Output directory: Dataset/")
    print(f"\nThis should take a few minutes...")
    
    try:
        # Import and run data collection
        import Collect_SA_quick
        print("\n✓ Data collection completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error during data collection: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_model(args):
    """Train the CNN+RNN model on collected data."""
    print("\n" + "=" * 60)
    print("PHASE 2: Model Training")
    print("=" * 60)
    
    # Check if CNN+RNN.py exists
    if not os.path.exists("CNN+RNN.py"):
        print("Error: CNN+RNN.py not found!")
        return False
    
    # Check if dataset exists
    if not os.path.exists("Dataset/states.csv") or not os.path.exists("Dataset/actions.csv"):
        print("\n✗ Error: Training data not found!")
        print("  Run data collection first (remove --skip-collection flag)")
        return False
    
    print("\nStarting model training...")
    print(f"Configuration:")
    print(f"  - Architecture: CNN + RNN (GRU)")
    print(f"  - Dataset: Dataset/")
    print(f"  - Train/Val/Test split: seeds 0-6 / 7-8 / 9")
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  - Device: {device.upper()}")
    
    if device == "cuda":
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nTraining in progress...")
    print("(See detailed output below)\n")
    print("-" * 60)
    
    try:
        # Run training
        os.system(f'"{sys.executable}" CNN+RNN.py')
        print("-" * 60)
        print("\n✓ Training completed!")
        return True
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description="AI-in-3D-Printing: Complete pipeline for Hamiltonian path optimization"
    )
    parser.add_argument(
        '--skip-collection',
        action='store_true',
        help='Skip data collection and use existing dataset'
    )
    parser.add_argument(
        '--collect-only',
        action='store_true',
        help='Only run data collection, skip training'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only run training, skip data collection'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(" AI-in-3D-Printing Pipeline")
    print(" Hamiltonian Path Optimization for 3D Printing")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check dependencies
    if not check_dependencies():
        print("\n✗ Dependency check failed. Please install requirements.")
        sys.exit(1)
    
    success = True
    
    # Phase 1: Data Collection
    if not args.train_only:
        if not collect_data(args):
            print("\n✗ Pipeline failed at data collection phase")
            success = False
    
    # Phase 2: Training
    if success and not args.collect_only:
        if not train_model(args):
            print("\n✗ Pipeline failed at training phase")
            success = False
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    
    if success:
        print(f"✓ All phases completed successfully!")
    else:
        print(f"✗ Pipeline encountered errors")
    
    print(f"\nTotal execution time: {elapsed/60:.2f} minutes")
    print("=" * 60)
    
    if success:
        print("\nNext steps:")
        print("  - Check trained model in current directory")
        print("  - Review training logs above")
        print("  - Explore Dataset/ for collected data")
        print("  - Use trained model for path optimization")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
