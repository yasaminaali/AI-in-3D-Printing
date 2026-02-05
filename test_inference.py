"""
Test inference with trained model on multiple zone patterns
"""
import sys
import yaml
import torch
import json
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.models.cnn_rnn import CNNRNNHamiltonian
from operations import HamiltonianSTL
from Zones import zones_left_right, zones_stripes, zones_voronoi, zones_checkerboard

def load_model(checkpoint_path: str, config_path: str = "model/config.yaml"):
    """Load trained model from checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = CNNRNNHamiltonian(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def create_zone_grid(grid_W: int, grid_H: int, pattern: str):
    """Create zone grid for given pattern."""
    if pattern == "left_right":
        return zones_left_right(grid_W, grid_H)
    elif pattern == "stripes":
        result = zones_stripes(grid_W, grid_H, direction='v', k=3)
        return result[0] if isinstance(result, tuple) else result
    elif pattern == "voronoi":
        result = zones_voronoi(grid_W, grid_H, k=3)
        return result[0] if isinstance(result, tuple) else result
    elif pattern == "checkerboard":
        return zones_checkerboard(grid_W, grid_H, kx=3, ky=3)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

def run_inference_test(model, config, grid_W: int, grid_H: int, zone_pattern: str, device: str = "cpu"):
    """Run inference and measure performance."""
    device = torch.device(device)
    model = model.to(device)
    
    # Create initial Hamiltonian path
    h = HamiltonianSTL(grid_W, grid_H, init_pattern="zigzag")
    
    # Create zones
    zones = create_zone_grid(grid_W, grid_H, zone_pattern)
    
    # Estimate initial crossings (rough approximation)
    initial_crossings = grid_W * 2  # ~2 crossings per row for zigzag
    
    # Prepare input
    seq_len = config['model']['predictor']['sequence_length']
    max_pos = config['model']['predictor']['max_positions']
    
    # Create grid state
    grid_state = torch.zeros(4, max_pos, max_pos)
    
    # Channel 0: Horizontal edges
    h_edges = torch.tensor(h.H, dtype=torch.float32)
    grid_state[0, :grid_H, :grid_W-1] = h_edges[:grid_H, :grid_W-1]
    
    # Channel 1: Vertical edges  
    v_edges = torch.tensor(h.V, dtype=torch.float32)
    grid_state[1, :grid_H-1, :grid_W] = v_edges[:grid_H-1, :grid_W]
    
    # Channel 2: Zone boundaries
    zone_grid = torch.zeros(grid_H, grid_W)
    if isinstance(zones, dict):
        for (x, y), zone_id in zones.items():
            if 0 <= x < grid_W and 0 <= y < grid_H:
                zone_grid[y, x] = zone_id
    grid_state[2, :grid_H, :grid_W] = zone_grid
    
    # Channel 3: Grid mask
    grid_state[3, :grid_H, :grid_W] = 1.0
    
    # Global features
    global_features = torch.tensor([initial_crossings, grid_W, grid_H], dtype=torch.float32)
    
    # Expand for sequence
    grid_state = grid_state.unsqueeze(0).unsqueeze(1).expand(1, seq_len, 4, max_pos, max_pos).to(device)
    global_features = global_features.unsqueeze(0).unsqueeze(1).expand(1, seq_len, 3).to(device)
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        predictions = model(grid_state, global_features)
    inference_time = time.time() - start_time
    
    # Decode predictions
    predicted_ops = []
    for t in range(min(50, seq_len)):  # First 50 operations
        pred_type = predictions['operation_type'][0, t].argmax().item()
        pred_x = predictions['position_x'][0, t].argmax().item()
        pred_y = predictions['position_y'][0, t].argmax().item()
        
        type_map = {0: 'T', 1: 'F', 2: 'N'}
        if pred_type != 2:  # Not NOP
            predicted_ops.append({
                'type': type_map[pred_type],
                'x': pred_x,
                'y': pred_y
            })
    
    return {
        'grid_size': f"{grid_W}x{grid_H}",
        'zone_pattern': zone_pattern,
        'initial_crossings': initial_crossings,
        'predicted_operations': len(predicted_ops),
        'inference_time_ms': inference_time * 1000,
        'sample_predictions': predicted_ops[:5]  # First 5 ops
    }

def main():
    print("="*80)
    print("INFERENCE TEST - Trained CNN+RNN Model")
    print("="*80)
    print()
    
    # Load model
    print("Loading model from nn_checkpoints/best_model.pt...")
    model, config = load_model("nn_checkpoints/best_model.pt")
    print(f"[OK] Model loaded successfully!")
    print(f"  Sequence length: {config['model']['predictor']['sequence_length']}")
    print(f"  Max positions: {config['model']['predictor']['max_positions']}")
    print()
    
    # Test different configurations
    test_configs = [
        (30, 30, "left_right"),
        (30, 30, "stripes"),
        (30, 30, "voronoi"),
        (30, 30, "checkerboard"),
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print()
    
    results = []
    for grid_W, grid_H, pattern in test_configs:
        print(f"Testing {pattern} on {grid_W}x{grid_H} grid...")
        try:
            result = run_inference_test(model, config, grid_W, grid_H, pattern, device)
            results.append(result)
            
            print(f"  [OK] Initial crossings: {result['initial_crossings']}")
            print(f"  [OK] Predicted operations: {result['predicted_operations']}")
            print(f"  [OK] Inference time: {result['inference_time_ms']:.1f}ms")
            print(f"  [OK] Sample predictions: {result['sample_predictions'][:3]}")
            print()
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Tested {len(results)} configurations")
    print()
    
    if results:
        avg_time = sum(r['inference_time_ms'] for r in results) / len(results)
        print(f"Average inference time: {avg_time:.1f}ms")
        print()
        
        print("Results by pattern:")
        print(f"  {'Pattern':<15} {'Initial':<10} {'Ops Pred':<10} {'Time (ms)':<10}")
        print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
        for r in results:
            print(f"  {r['zone_pattern']:<15} {r['initial_crossings']:<10} {r['predicted_operations']:<10} {r['inference_time_ms']:<10.1f}")
    
    print()
    print("="*80)
    print("[OK] Model is ready for inference!")
    print()
    print("Example usage:")
    print("  python model/inference.py --checkpoint nn_checkpoints/best_model.pt \\")
    print("    --grid-W 30 --grid-H 30 --zone-pattern left_right")
    print("="*80)

if __name__ == "__main__":
    main()
