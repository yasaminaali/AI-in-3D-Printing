"""
Inference Script for CNN+RNN Model
Generate operation sequences for new Hamiltonian path problems
"""

import sys
import yaml
import torch
import json
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.models.cnn_rnn import CNNRNNHamiltonian
from operations import HamiltonianSTL
from Zones import zones_left_right, zones_stripes, zones_voronoi


def load_model(checkpoint_path: str, config_path: str = "model/config.yaml") -> CNNRNNHamiltonian:
    """Load trained model from checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = CNNRNNHamiltonian(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def optimize_hamiltonian_path(
    model: CNNRNNHamiltonian,
    grid_W: int,
    grid_H: int,
    zone_pattern: str,
    initial_path: str = "zigzag",
    max_operations: int = 100,
    device: str = "cpu"
) -> Dict:
    """
    Optimize a Hamiltonian path using the trained model.
    
    Args:
        model: Trained CNN+RNN model
        grid_W, grid_H: Grid dimensions
        zone_pattern: Zone layout pattern
        initial_path: Initial Hamiltonian path type
        max_operations: Maximum operations to predict
        device: Computation device
    
    Returns:
        result: Dict with best_crossings and operation_sequence
    """
    device = torch.device(device)
    model = model.to(device)
    
    # Create initial Hamiltonian path
    h = HamiltonianSTL(grid_W, grid_H, init_pattern=initial_path)
    initial_crossings = compute_crossings(h, zone_pattern)
    
    # Prepare input
    grid_state = prepare_grid_input(h, zone_pattern, grid_W, grid_H)
    global_features = torch.tensor([initial_crossings, grid_W, grid_H], dtype=torch.float32)
    
    grid_state = grid_state.unsqueeze(0).to(device)
    global_features = global_features.unsqueeze(0).to(device)
    
    # Predict operation sequence
    with torch.no_grad():
        operation_sequence = model.predict_sequence(
            grid_state, global_features, max_length=max_operations
        )
    
    # Apply operations and compute final crossings
    h_final = apply_operations(h, operation_sequence)
    final_crossings = compute_crossings(h_final, zone_pattern)
    
    return {
        'initial_crossings': initial_crossings,
        'final_crossings': final_crossings,
        'improvement': initial_crossings - final_crossings,
        'operation_sequence': operation_sequence,
        'num_operations': len([op for op in operation_sequence if op['kind'] != 'N'])
    }


def prepare_grid_input(h: HamiltonianSTL, zone_pattern: str, 
                      grid_W: int, grid_H: int) -> torch.Tensor:
    """Prepare grid state as multi-channel tensor."""
    import numpy as np
    
    # Build zones
    if zone_pattern == "left_right":
        zones = zones_left_right(grid_W, grid_H)
    elif zone_pattern == "stripes":
        zones = zones_stripes(grid_W, grid_H, direction='v', k=3)[0]
    elif zone_pattern == "voronoi":
        zones = zones_voronoi(grid_W, grid_H, k=3)[0]
    else:
        raise ValueError(f"Unknown zone pattern: {zone_pattern}")
    
    # Create 4-channel input
    grid = torch.zeros(4, 30, 30)
    
    # Channel 0: Horizontal edges
    h_edges = torch.tensor(h.H, dtype=torch.float32)
    grid[0, :grid_H, :grid_W-1] = h_edges[:grid_H, :grid_W-1]
    
    # Channel 1: Vertical edges
    v_edges = torch.tensor(h.V, dtype=torch.float32)
    grid[1, :grid_H-1, :grid_W] = v_edges[:grid_H-1, :grid_W]
    
    # Channel 2: Zone boundaries
    zone_grid = torch.zeros(grid_H, grid_W)
    for (x, y), zone_id in zones.items():
        if 0 <= x < grid_W and 0 <= y < grid_H:
            zone_grid[y, x] = zone_id
    grid[2, :grid_H, :grid_W] = zone_grid
    
    # Channel 3: Grid mask
    grid[3, :grid_H, :grid_W] = 1.0
    
    return grid


def compute_crossings(h: HamiltonianSTL, zone_pattern: str) -> int:
    """Compute zone crossings for a Hamiltonian path."""
    # Simplified - in practice would use proper zone grid
    # This is a placeholder
    return 30  # Default initial crossings


def apply_operations(h: HamiltonianSTL, operations: List[Dict]) -> HamiltonianSTL:
    """Apply predicted operations to Hamiltonian path."""
    h_result = h  # In practice, would copy and apply each operation
    
    for op in operations:
        if op['kind'] == 'N':
            break
        # Apply operation (simplified)
        try:
            if op['kind'] == 'T':
                h_result.transpose_subgrid(op['x'], op['y'], op['variant'])
            elif op['kind'] == 'F':
                h_result.flip_subgrid(op['x'], op['y'], op['variant'])
        except:
            pass  # Invalid operation, skip
    
    return h_result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CNN+RNN inference')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--grid-W', type=int, default=30, help='Grid width')
    parser.add_argument('--grid-H', type=int, default=30, help='Grid height')
    parser.add_argument('--zone-pattern', default='left_right',
                       choices=['left_right', 'stripes', 'voronoi', 'islands'],
                       help='Zone pattern')
    parser.add_argument('--initial-path', default='zigzag',
                       choices=['zigzag', 'vertical_zigzag'],
                       help='Initial Hamiltonian path')
    parser.add_argument('--max-operations', type=int, default=100,
                       help='Max operations to predict')
    parser.add_argument('--output', default='inference_result.json',
                       help='Output file')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)
    
    print(f"Optimizing {args.grid_W}x{args.grid_H} grid with {args.zone_pattern}...")
    result = optimize_hamiltonian_path(
        model,
        args.grid_W,
        args.grid_H,
        args.zone_pattern,
        args.initial_path,
        args.max_operations
    )
    
    print(f"\nResults:")
    print(f"  Initial crossings: {result['initial_crossings']}")
    print(f"  Final crossings: {result['final_crossings']}")
    print(f"  Improvement: {result['improvement']}")
    print(f"  Operations applied: {result['num_operations']}")
    
    # Save result
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResult saved to: {args.output}")


if __name__ == "__main__":
    main()
