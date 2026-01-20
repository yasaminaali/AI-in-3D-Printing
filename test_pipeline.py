#!/usr/bin/env python
"""Quick test to verify the reorganized codebase works."""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

def test_imports():
    """Test all core imports work."""
    print("=" * 60)
    print("Testing Imports")
    print("=" * 60)
    
    try:
        from src.core.hamiltonian import HamiltonianSTL
        print("[OK] Core: hamiltonian")
    except Exception as e:
        print(f"[ERROR] Core hamiltonian: {e}")
        return False
    
    try:
        from src.core.zones import zones_left_right
        print("[OK] Core: zones")
    except Exception as e:
        print(f"[ERROR] Core zones: {e}")
        return False
    
    try:
        from src.optimization.simulated_annealing import HamiltonianZoningSA
        print("[OK] Optimization: simulated_annealing")
    except Exception as e:
        print(f"[ERROR] Optimization SA: {e}")
        return False
    
    try:
        from src.data.collector import ZoningCollector
        print("[OK] Data: collector")
    except Exception as e:
        print(f"[ERROR] Data collector: {e}")
        return False
    
    try:
        from src.ml.cnn_rnn import CNN_RNN, scan_action_metadata
        print("[OK] ML: cnn_rnn")
    except Exception as e:
        print(f"[ERROR] ML cnn_rnn: {e}")
        return False
    
    return True


def test_data_loading():
    """Test dataset loading."""
    print("\n" + "=" * 60)
    print("Testing Data Loading")
    print("=" * 60)
    
    if not os.path.exists("Dataset/states.csv"):
        print("[SKIP] No dataset found")
        return True
    
    try:
        from src.ml.cnn_rnn import scan_action_metadata, get_pattern_list
        
        meta = scan_action_metadata("Dataset/actions.csv")
        print(f"[OK] Actions metadata: {len(meta['orientations'])} orientations, {len(meta['ops'])} ops")
        
        patterns = get_pattern_list("Dataset/states.csv")
        print(f"[OK] Zone patterns: {patterns}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        return False


def test_model_loading():
    """Test trained model can be loaded."""
    print("\n" + "=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    if not os.path.exists("models/global_seq_policy.pt"):
        print("[SKIP] No trained model found")
        return True
    
    try:
        import torch
        from src.ml.cnn_rnn import CNN_RNN, scan_action_metadata, get_pattern_list
        
        # Get metadata for model dimensions
        meta = scan_action_metadata("Dataset/actions.csv")
        patterns = get_pattern_list("Dataset/states.csv")
        
        # Dummy dimensions (will be overwritten by actual)
        in_channels = 10  # This should match dataset
        cfg_dim = 3 + len(patterns)
        
        model = CNN_RNN(
            in_channels=in_channels,
            cfg_dim=cfg_dim,
            n_ops=len(meta['ops']),
            n_sg=len(meta['subgrid_kinds']),
            n_ori=len(meta['orientations']),
        )
        
        state_dict = torch.load("models/global_seq_policy.pt", map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model loaded: {param_count:,} parameters")
        
        return True
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hamiltonian_operations():
    """Test basic Hamiltonian operations."""
    print("\n" + "=" * 60)
    print("Testing Hamiltonian Operations")
    print("=" * 60)
    
    try:
        from src.core.hamiltonian import HamiltonianSTL
        
        h = HamiltonianSTL(10, 10)
        print(f"[OK] Created 10x10 grid")
        
        # Test validation
        is_valid = h.validate_full_path()
        print(f"[OK] Path validation: {is_valid}")
        
        # Test subgrid operation
        sub = h.get_subgrid((0, 0), (2, 2))
        result = h.transpose_subgrid(sub, "sr")
        print(f"[OK] Transpose operation completed")
        
        return True
    except Exception as e:
        print(f"[ERROR] Hamiltonian operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Pipeline Test Suite")
    print("#" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Model Loading", test_model_loading),
        ("Hamiltonian Operations", test_hamiltonian_operations),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "#" * 60)
    print("# Test Summary")
    print("#" * 60)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\n" + "=" * 60)
        print("All tests passed! Pipeline is ready.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Some tests failed. Check errors above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
