"""
Inference Script for CNN+RNN Model
Generate operation sequences to minimize zone crossings in Hamiltonian paths.

The model:
1. Takes a grid with zones (left_right, stripes, voronoi)
2. Starts with an initial Hamiltonian path (zigzag, vertical_zigzag)
3. Uses reconfiguration operations (Transpose, Flip) to search for 
   the path with minimal zone crossings
4. Outputs initial/final crossings, operation sequence, and visualization
"""

import sys
import yaml
import torch
import json
import copy
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.models.cnn_rnn import CNNRNNHamiltonian
from operations import HamiltonianSTL
from Zones import zones_left_right, zones_stripes, zones_voronoi


def load_model(checkpoint_path: str, config_path: str = "model/config.yaml", device: str = None) -> CNNRNNHamiltonian:
    """Load trained model from checkpoint."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = CNNRNNHamiltonian(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def get_zones(grid_W: int, grid_H: int, zone_pattern: str, seed: int = 42) -> Dict[Tuple[int, int], int]:
    """
    Get zone assignments for the grid.
    
    Args:
        grid_W, grid_H: Grid dimensions
        zone_pattern: 'left_right', 'stripes', 'voronoi', or 'islands'
        seed: Random seed for reproducible zones
    
    Returns:
        Dictionary mapping (x, y) -> zone_id
    """
    if zone_pattern == "left_right":
        return zones_left_right(grid_W, grid_H)
    elif zone_pattern == "stripes":
        return zones_stripes(grid_W, grid_H, direction='v', k=3)
    elif zone_pattern == "voronoi":
        import random
        random.seed(seed)
        return zones_voronoi(grid_W, grid_H, k=4)[0]
    elif zone_pattern == "islands":
        # Islands uses voronoi with more zones
        import random
        random.seed(seed)
        return zones_voronoi(grid_W, grid_H, k=6)[0]
    else:
        raise ValueError(f"Unknown zone pattern: {zone_pattern}")


def compute_crossings(h: HamiltonianSTL, zones: Dict[Tuple[int, int], int]) -> int:
    """
    Count zone boundary crossings in the Hamiltonian path.
    
    A crossing occurs when an edge connects two cells in different zones.
    """
    crossings = 0
    
    # Check horizontal edges
    for y in range(h.height):
        for x in range(h.width - 1):
            if h.H[y][x]:  # Edge exists between (x,y) and (x+1,y)
                z1 = zones.get((x, y), 0)
                z2 = zones.get((x + 1, y), 0)
                if z1 != z2:
                    crossings += 1
    
    # Check vertical edges
    for y in range(h.height - 1):
        for x in range(h.width):
            if h.V[y][x]:  # Edge exists between (x,y) and (x,y+1)
                z1 = zones.get((x, y), 0)
                z2 = zones.get((x, y + 1), 0)
                if z1 != z2:
                    crossings += 1
    
    return crossings


def zones_to_grid(zones: Dict[Tuple[int, int], int], grid_W: int, grid_H: int) -> np.ndarray:
    """Convert zones dict to numpy grid."""
    grid = np.zeros((grid_H, grid_W), dtype=np.int32)
    for (x, y), zone_id in zones.items():
        if 0 <= x < grid_W and 0 <= y < grid_H:
            grid[y, x] = zone_id
    return grid


def prepare_grid_input(h: HamiltonianSTL, zones: Dict[Tuple[int, int], int], 
                       grid_W: int, grid_H: int) -> torch.Tensor:
    """
    Prepare grid state as multi-channel tensor for model input.
    
    Channels:
        0: Horizontal edges
        1: Vertical edges
        2: Zone IDs (normalized)
        3: Grid mask (1 where valid, 0 outside)
    """
    # Use fixed size 30x30 as model expects
    grid = torch.zeros(4, 30, 30)
    
    # Channel 0: Horizontal edges
    for y in range(min(grid_H, 30)):
        for x in range(min(grid_W - 1, 29)):
            if y < len(h.H) and x < len(h.H[y]):
                grid[0, y, x] = float(h.H[y][x])
    
    # Channel 1: Vertical edges
    for y in range(min(grid_H - 1, 29)):
        for x in range(min(grid_W, 30)):
            if y < len(h.V) and x < len(h.V[y]):
                grid[1, y, x] = float(h.V[y][x])
    
    # Channel 2: Zone IDs (normalized)
    zone_grid = zones_to_grid(zones, grid_W, grid_H)
    max_zone = max(zone_grid.max(), 1)
    for y in range(min(grid_H, 30)):
        for x in range(min(grid_W, 30)):
            grid[2, y, x] = zone_grid[y, x] / max_zone
    
    # Channel 3: Grid mask
    grid[3, :grid_H, :grid_W] = 1.0
    
    return grid


def apply_operation(h: HamiltonianSTL, op: Dict) -> Tuple[bool, str]:
    """
    Apply a single operation to the Hamiltonian path.
    
    Args:
        h: HamiltonianSTL instance (modified in place)
        op: Operation dict with 'kind', 'x', 'y', 'variant'
    
    Returns:
        (success, message)
    """
    kind = op['kind']
    x, y = op['x'], op['y']
    variant = op['variant']
    
    if kind == 'N':
        return True, "NOOP"
    
    if kind == 'T':
        # Transpose operation on 3x3 subgrid
        # x, y is the top-left corner
        if x + 2 >= h.width or y + 2 >= h.height:
            return False, f"Transpose out of bounds: ({x}, {y})"
        if x < 0 or y < 0:
            return False, f"Transpose negative coords: ({x}, {y})"
        
        sub = h.get_subgrid((x, y), (x + 2, y + 2))
        result, msg = h.transpose_subgrid(sub, variant)
        
        if 'mismatch' in msg.lower() or 'unknown' in msg.lower():
            return False, f"Transpose failed: {msg}"
        return True, f"Transpose at ({x},{y}) variant {variant}"
    
    elif kind == 'F':
        # Flip operation on 3x2 or 2x3 subgrid
        if variant in ['n', 's']:
            # Vertical flip: 2x3 subgrid
            if x + 1 >= h.width or y + 2 >= h.height:
                return False, f"Flip out of bounds: ({x}, {y})"
            sub = h.get_subgrid((x, y), (x + 1, y + 2))
        else:  # 'e', 'w'
            # Horizontal flip: 3x2 subgrid  
            if x + 2 >= h.width or y + 1 >= h.height:
                return False, f"Flip out of bounds: ({x}, {y})"
            sub = h.get_subgrid((x, y), (x + 2, y + 1))
        
        if x < 0 or y < 0:
            return False, f"Flip negative coords: ({x}, {y})"
            
        result, msg = h.flip_subgrid(sub, variant)
        
        if 'mismatch' in msg.lower() or 'unknown' in msg.lower():
            return False, f"Flip failed: {msg}"
        return True, f"Flip at ({x},{y}) variant {variant}"
    
    return False, f"Unknown operation kind: {kind}"


def find_valid_improving_operation(h: HamiltonianSTL, zones: Dict, current_crossings: int) -> Dict:
    """
    Local search to find any valid operation that improves (or maintains) crossings.
    
    Tries all valid Transpose and Flip operations and returns the best one.
    Returns NOOP if no improving operation found.
    """
    best_op = None
    best_crossings = current_crossings
    
    # Make a copy for testing
    original_H = [row[:] for row in h.H]
    original_V = [row[:] for row in h.V]
    
    # Try Transpose operations
    transpose_variants = ['nl', 'nr', 'sl', 'sr', 'eb']
    for y in range(h.height - 2):
        for x in range(h.width - 2):
            for variant in transpose_variants:
                # Try operation
                h.H = [row[:] for row in original_H]
                h.V = [row[:] for row in original_V]
                
                op = {'kind': 'T', 'x': x, 'y': y, 'variant': variant}
                success, _ = apply_operation(h, op)
                
                if success:
                    new_crossings = compute_crossings(h, zones)
                    if new_crossings < best_crossings:
                        best_crossings = new_crossings
                        best_op = op
    
    # Try Flip operations
    flip_variants = ['n', 's', 'e', 'w']
    for y in range(h.height - 2):
        for x in range(h.width - 1):
            for variant in ['n', 's']:  # Vertical flip: 2x3
                h.H = [row[:] for row in original_H]
                h.V = [row[:] for row in original_V]
                
                op = {'kind': 'F', 'x': x, 'y': y, 'variant': variant}
                success, _ = apply_operation(h, op)
                
                if success:
                    new_crossings = compute_crossings(h, zones)
                    if new_crossings < best_crossings:
                        best_crossings = new_crossings
                        best_op = op
    
    for y in range(h.height - 1):
        for x in range(h.width - 2):
            for variant in ['e', 'w']:  # Horizontal flip: 3x2
                h.H = [row[:] for row in original_H]
                h.V = [row[:] for row in original_V]
                
                op = {'kind': 'F', 'x': x, 'y': y, 'variant': variant}
                success, _ = apply_operation(h, op)
                
                if success:
                    new_crossings = compute_crossings(h, zones)
                    if new_crossings < best_crossings:
                        best_crossings = new_crossings
                        best_op = op
    
    # Restore original state
    h.H = [row[:] for row in original_H]
    h.V = [row[:] for row in original_V]
    
    if best_op:
        return best_op
    return {'kind': 'N', 'x': 0, 'y': 0, 'variant': '-'}


def find_any_valid_operation(h: HamiltonianSTL, zones: Dict) -> Dict:
    """
    Find any valid operation (not necessarily improving).
    Used when completely stuck to make progress.
    """
    import random
    
    valid_ops = []
    original_H = [row[:] for row in h.H]
    original_V = [row[:] for row in h.V]
    
    # Collect all valid operations
    transpose_variants = ['nl', 'nr', 'sl', 'sr', 'eb']
    for y in range(h.height - 2):
        for x in range(h.width - 2):
            for variant in transpose_variants:
                h.H = [row[:] for row in original_H]
                h.V = [row[:] for row in original_V]
                
                op = {'kind': 'T', 'x': x, 'y': y, 'variant': variant}
                success, _ = apply_operation(h, op)
                
                if success:
                    valid_ops.append(op)
    
    flip_variants = ['n', 's', 'e', 'w']
    for y in range(h.height - 2):
        for x in range(h.width - 1):
            for variant in ['n', 's']:
                h.H = [row[:] for row in original_H]
                h.V = [row[:] for row in original_V]
                
                op = {'kind': 'F', 'x': x, 'y': y, 'variant': variant}
                success, _ = apply_operation(h, op)
                
                if success:
                    valid_ops.append(op)
    
    for y in range(h.height - 1):
        for x in range(h.width - 2):
            for variant in ['e', 'w']:
                h.H = [row[:] for row in original_H]
                h.V = [row[:] for row in original_V]
                
                op = {'kind': 'F', 'x': x, 'y': y, 'variant': variant}
                success, _ = apply_operation(h, op)
                
                if success:
                    valid_ops.append(op)
    
    # Restore original state
    h.H = [row[:] for row in original_H]
    h.V = [row[:] for row in original_V]
    
    if valid_ops:
        return random.choice(valid_ops)
    return {'kind': 'N', 'x': 0, 'y': 0, 'variant': '-'}


def optimize_hamiltonian_path(
    model: CNNRNNHamiltonian,
    grid_W: int,
    grid_H: int,
    zone_pattern: str,
    initial_path: str = "zigzag",
    max_operations: int = 100,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
    use_local_search: bool = True
) -> Dict:
    """
    Optimize a Hamiltonian path using the trained model.
    
    The model predicts a sequence of operations (Transpose, Flip) to 
    minimize zone boundary crossings. When operations fail, it falls back
    to local search to find valid operations.
    
    Args:
        model: Trained CNN+RNN model
        grid_W, grid_H: Grid dimensions
        zone_pattern: Zone layout pattern
        initial_path: Initial Hamiltonian path type
        max_operations: Maximum operations to predict
        device: Computation device
        seed: Random seed for zone generation
        verbose: Print progress
        use_local_search: Use local search when model fails
    
    Returns:
        result: Dict with crossings, operations, and visualization data
    """
    device = torch.device(device)
    model = model.to(device)
    
    # Create zones
    zones = get_zones(grid_W, grid_H, zone_pattern, seed)
    zone_grid = zones_to_grid(zones, grid_W, grid_H)
    
    # Create initial Hamiltonian path
    h = HamiltonianSTL(grid_W, grid_H, init_pattern=initial_path)
    initial_crossings = compute_crossings(h, zones)
    
    # Save initial state for visualization
    initial_H = [row[:] for row in h.H]
    initial_V = [row[:] for row in h.V]
    
    if verbose:
        print(f"Initial path: {initial_path}")
        print(f"Initial crossings: {initial_crossings}")
    
    # Tracking
    operation_sequence = []
    applied_ops = []
    best_crossings = initial_crossings
    best_H = initial_H
    best_V = initial_V
    
    # Run inference with grid state updates
    hidden = None
    consecutive_failures = 0
    max_consecutive_failures = 5  # After this many, try sampling
    
    with torch.no_grad():
        for step in range(max_operations):
            # Prepare current grid state
            grid_state = prepare_grid_input(h, zones, grid_W, grid_H)
            current_crossings = compute_crossings(h, zones)
            global_features = torch.tensor(
                [current_crossings, grid_W, grid_H], 
                dtype=torch.float32
            )
            
            grid_state = grid_state.unsqueeze(0).to(device)
            global_features = global_features.unsqueeze(0).to(device)
            
            # Get embedding
            emb = model.cnn(grid_state, global_features)
            emb = emb.unsqueeze(1)
            
            # RNN step
            rnn_out, hidden = model.rnn(emb, hidden)
            preds = model.predictor(rnn_out)
            
            # Decode predictions - use sampling if stuck
            use_sampling = consecutive_failures >= max_consecutive_failures
            temperature = 1.5 if use_sampling else 1.0
            
            if use_sampling:
                # Sample from distribution
                op_probs = torch.softmax(preds['operation_type'][:, 0] / temperature, dim=-1)
                op_type = torch.multinomial(op_probs, 1).item()
                
                x_probs = torch.softmax(preds['position_x'][:, 0] / temperature, dim=-1)
                pos_x = torch.multinomial(x_probs, 1).item()
                
                y_probs = torch.softmax(preds['position_y'][:, 0] / temperature, dim=-1)
                pos_y = torch.multinomial(y_probs, 1).item()
            else:
                op_type = torch.argmax(preds['operation_type'][:, 0], dim=-1).item()
                pos_x = torch.argmax(preds['position_x'][:, 0], dim=-1).item()
                pos_y = torch.argmax(preds['position_y'][:, 0], dim=-1).item()
            
            if op_type == 0:  # Transpose
                variant_map = ['nl', 'nr', 'sl', 'sr', 'eb']
                if use_sampling:
                    var_probs = torch.softmax(preds['transpose_variant'][:, 0] / temperature, dim=-1)
                    variant_idx = torch.multinomial(var_probs, 1).item()
                else:
                    variant_idx = torch.argmax(preds['transpose_variant'][:, 0], dim=-1).item()
                kind, variant = 'T', variant_map[min(variant_idx, len(variant_map)-1)]
            elif op_type == 1:  # Flip
                variant_map = ['n', 's', 'e', 'w']
                if use_sampling:
                    var_probs = torch.softmax(preds['flip_variant'][:, 0] / temperature, dim=-1)
                    variant_idx = torch.multinomial(var_probs, 1).item()
                else:
                    variant_idx = torch.argmax(preds['flip_variant'][:, 0], dim=-1).item()
                kind, variant = 'F', variant_map[min(variant_idx, len(variant_map)-1)]
            else:  # NOOP - stop
                if verbose:
                    print(f"Step {step+1}: Model predicted STOP")
                operation_sequence.append({'kind': 'N', 'x': 0, 'y': 0, 'variant': '-'})
                break
            
            op = {'kind': kind, 'x': pos_x, 'y': pos_y, 'variant': variant}
            operation_sequence.append(op)
            
            # Try to apply operation
            success, msg = apply_operation(h, op)
            
            if success:
                applied_ops.append(op)
                new_crossings = compute_crossings(h, zones)
                
                if verbose and step < 10:
                    print(f"Step {step+1}: {kind} at ({pos_x},{pos_y}) variant={variant} -> {new_crossings} crossings")
                
                # Track best
                if new_crossings < best_crossings:
                    best_crossings = new_crossings
                    best_H = [row[:] for row in h.H]
                    best_V = [row[:] for row in h.V]
                    if verbose:
                        print(f"  *** New best: {best_crossings} crossings!")
                
                consecutive_failures = 0  # Reset on success
            else:
                consecutive_failures += 1
                if verbose and step < 10:
                    print(f"Step {step+1}: {kind} at ({pos_x},{pos_y}) FAILED: {msg}")
                
                # Use local search as fallback when model keeps failing
                if use_local_search and consecutive_failures >= max_consecutive_failures:
                    if verbose:
                        print(f"  Model stuck, using local search...")
                    
                    current_crossings = compute_crossings(h, zones)
                    local_op = find_valid_improving_operation(h, zones, current_crossings)
                    
                    if local_op['kind'] != 'N':
                        success, msg = apply_operation(h, local_op)
                        if success:
                            applied_ops.append(local_op)
                            operation_sequence.append(local_op)
                            new_crossings = compute_crossings(h, zones)
                            
                            if verbose:
                                print(f"  Local search: {local_op['kind']} at ({local_op['x']},{local_op['y']}) -> {new_crossings} crossings")
                            
                            if new_crossings < best_crossings:
                                best_crossings = new_crossings
                                best_H = [row[:] for row in h.H]
                                best_V = [row[:] for row in h.V]
                                if verbose:
                                    print(f"  *** New best: {best_crossings} crossings!")
                            
                            consecutive_failures = 0
                    else:
                        # No improving operation found, try any valid operation
                        any_op = find_any_valid_operation(h, zones)
                        if any_op['kind'] != 'N':
                            success, msg = apply_operation(h, any_op)
                            if success:
                                applied_ops.append(any_op)
                                operation_sequence.append(any_op)
                                consecutive_failures = max(0, consecutive_failures - 2)
                                if verbose:
                                    print(f"  Random valid op: {any_op['kind']} at ({any_op['x']},{any_op['y']})")
    
    # Restore best state
    h.H = best_H
    h.V = best_V
    final_crossings = compute_crossings(h, zones)
    
    if verbose:
        print(f"\nFinal crossings: {final_crossings}")
        print(f"Improvement: {initial_crossings} -> {final_crossings} ({initial_crossings - final_crossings} fewer)")
        print(f"Operations predicted: {len(operation_sequence)}")
        print(f"Operations applied: {len(applied_ops)}")
    
    return {
        'grid_W': grid_W,
        'grid_H': grid_H,
        'zone_pattern': zone_pattern,
        'initial_path': initial_path,
        'initial_crossings': initial_crossings,
        'final_crossings': final_crossings,
        'improvement': initial_crossings - final_crossings,
        'operation_sequence': operation_sequence,
        'applied_operations': applied_ops,
        'num_operations_predicted': len(operation_sequence),
        'num_operations_applied': len(applied_ops),
        'zone_grid': zone_grid.tolist(),
        'initial_H': initial_H,
        'initial_V': initial_V,
        'final_H': best_H,
        'final_V': best_V
    }


def visualize_result(result: Dict, output_dir: str = "."):
    """
    Create visualizations of the optimization result.
    
    Generates:
    1. Initial path visualization
    2. Final (optimized) path visualization  
    3. Side-by-side comparison
    """
    from model.utils.visualization import visualize_solution, create_comparison_visualization
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    grid_W = result['grid_W']
    grid_H = result['grid_H']
    zone_grid = np.array(result['zone_grid'])
    
    # Convert edge lists to numpy arrays
    initial_H = np.array(result['initial_H'], dtype=bool)
    initial_V = np.array(result['initial_V'], dtype=bool)
    final_H = np.array(result['final_H'], dtype=bool)
    final_V = np.array(result['final_V'], dtype=bool)
    
    # Initial path visualization
    visualize_solution(
        grid_W, grid_H, zone_grid,
        initial_H, initial_V,
        result['initial_crossings'], result['initial_crossings'],
        [],
        str(output_dir / "initial_path.png"),
        title=f"Initial {result['initial_path']} Path"
    )
    print(f"Saved: {output_dir / 'initial_path.png'}")
    
    # Final path visualization
    visualize_solution(
        grid_W, grid_H, zone_grid,
        final_H, final_V,
        result['initial_crossings'], result['final_crossings'],
        result['applied_operations'],
        str(output_dir / "final_path.png"),
        title=f"Optimized Path ({result['zone_pattern']} zones)"
    )
    print(f"Saved: {output_dir / 'final_path.png'}")
    
    # Side-by-side comparison
    create_comparison_visualization(
        grid_W, grid_H, zone_grid,
        initial_H, initial_V,
        final_H, final_V,
        result['initial_crossings'], result['final_crossings'],
        result['applied_operations'],
        str(output_dir / "comparison.png")
    )
    print(f"Saved: {output_dir / 'comparison.png'}")


def pure_local_search(
    grid_W: int,
    grid_H: int,
    zone_pattern: str,
    initial_path: str = "zigzag",
    max_operations: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Pure local search optimization (no neural network).
    
    Uses greedy hill climbing to find operations that reduce crossings.
    Good baseline to compare model performance against.
    """
    # Create zones
    zones = get_zones(grid_W, grid_H, zone_pattern, seed)
    zone_grid = zones_to_grid(zones, grid_W, grid_H)
    
    # Create initial Hamiltonian path
    h = HamiltonianSTL(grid_W, grid_H, init_pattern=initial_path)
    initial_crossings = compute_crossings(h, zones)
    
    # Save initial state
    initial_H = [row[:] for row in h.H]
    initial_V = [row[:] for row in h.V]
    
    if verbose:
        print(f"Initial crossings: {initial_crossings}")
    
    # Track best
    best_crossings = initial_crossings
    best_H = initial_H
    best_V = initial_V
    applied_ops = []
    no_improve_count = 0
    
    for step in range(max_operations):
        current_crossings = compute_crossings(h, zones)
        
        # Find improving operation
        improving_op = find_valid_improving_operation(h, zones, current_crossings)
        
        if improving_op['kind'] == 'N':
            no_improve_count += 1
            if no_improve_count >= 10:
                if verbose:
                    print(f"Step {step+1}: No improving operations found, stopping")
                break
            # Try a random valid operation to escape local optimum
            any_op = find_any_valid_operation(h, zones)
            if any_op['kind'] != 'N':
                success, _ = apply_operation(h, any_op)
                if success:
                    applied_ops.append(any_op)
            continue
        
        # Apply improving operation
        success, _ = apply_operation(h, improving_op)
        if success:
            applied_ops.append(improving_op)
            new_crossings = compute_crossings(h, zones)
            
            if verbose and step < 10:
                print(f"Step {step+1}: {improving_op['kind']} at ({improving_op['x']},{improving_op['y']}) -> {new_crossings}")
            
            if new_crossings < best_crossings:
                best_crossings = new_crossings
                best_H = [row[:] for row in h.H]
                best_V = [row[:] for row in h.V]
                no_improve_count = 0
                if verbose:
                    print(f"  *** New best: {best_crossings} crossings!")
    
    # Restore best state
    h.H = best_H
    h.V = best_V
    final_crossings = compute_crossings(h, zones)
    
    if verbose:
        print(f"\nFinal crossings: {final_crossings}")
        print(f"Improvement: {initial_crossings} -> {final_crossings}")
    
    return {
        'grid_W': grid_W,
        'grid_H': grid_H,
        'zone_pattern': zone_pattern,
        'initial_path': initial_path,
        'initial_crossings': initial_crossings,
        'final_crossings': final_crossings,
        'improvement': initial_crossings - final_crossings,
        'applied_operations': applied_ops,
        'num_operations_applied': len(applied_ops),
        'zone_grid': zone_grid.tolist(),
        'initial_H': initial_H,
        'initial_V': initial_V,
        'final_H': best_H,
        'final_V': best_V
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimize Hamiltonian paths using CNN+RNN model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  ./run_inference.sh --checkpoint nn_checkpoints/checkpoint_epoch_100.pt
  
  # Custom grid and zone pattern
  python model/inference.py --checkpoint nn_checkpoints/checkpoint_epoch_100.pt \\
      --grid-W 20 --grid-H 20 --zone-pattern voronoi
  
  # Use vertical zigzag for vertical zone boundaries
  python model/inference.py --checkpoint nn_checkpoints/checkpoint_epoch_100.pt \\
      --initial-path vertical_zigzag --zone-pattern left_right
"""
    )
    parser.add_argument('--checkpoint', required=False, help='Model checkpoint path')
    parser.add_argument('--local-search-only', action='store_true',
                       help='Use pure local search without neural network')
    parser.add_argument('--grid-W', type=int, default=30, help='Grid width')
    parser.add_argument('--grid-H', type=int, default=30, help='Grid height')
    parser.add_argument('--zone-pattern', default='left_right',
                       choices=['left_right', 'stripes', 'voronoi', 'islands'],
                       help='Zone pattern for the grid')
    parser.add_argument('--initial-path', default='zigzag',
                       choices=['zigzag', 'vertical_zigzag'],
                       help='Initial Hamiltonian path pattern')
    parser.add_argument('--max-operations', type=int, default=100,
                       help='Maximum operations to predict')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for zone generation')
    parser.add_argument('--output', default='inference_result.json',
                       help='Output JSON file')
    parser.add_argument('--output-dir', default='.',
                       help='Directory for visualization outputs')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.local_search_only and not args.checkpoint:
        parser.error("--checkpoint is required unless --local-search-only is specified")
    
    # Auto-detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=" * 50)
    if args.local_search_only:
        print("Hamiltonian Path Optimizer (Local Search)")
    else:
        print("Hamiltonian Path Optimizer (CNN+RNN)")
    print(f"=" * 50)
    print(f"Device: {device}")
    if not args.local_search_only:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Grid: {args.grid_W}x{args.grid_H}")
    print(f"Zone pattern: {args.zone_pattern}")
    print(f"Initial path: {args.initial_path}")
    print(f"=" * 50)
    print()
    
    if args.local_search_only:
        # Pure local search
        print(f"Running local search optimization...")
        result = pure_local_search(
            args.grid_W,
            args.grid_H,
            args.zone_pattern,
            args.initial_path,
            args.max_operations,
            seed=args.seed,
            verbose=not args.quiet
        )
    else:
        # Load model
        print(f"Loading model...")
        model = load_model(args.checkpoint, device=device)
        
        # Run optimization
        print(f"\nRunning optimization...")
        result = optimize_hamiltonian_path(
            model,
            args.grid_W,
            args.grid_H,
            args.zone_pattern,
            args.initial_path,
            args.max_operations,
            device=device,
            seed=args.seed,
            verbose=not args.quiet
        )
    
    # Save results
    # Convert numpy arrays to lists for JSON serialization
    result_json = {k: v for k, v in result.items() 
                   if k not in ['initial_H', 'initial_V', 'final_H', 'final_V', 'zone_grid']}
    result_json['zone_grid'] = result['zone_grid']
    
    with open(args.output, 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f"\nResults saved to: {args.output}")
    
    # Generate visualizations
    if not args.no_visualize:
        print(f"\nGenerating visualizations...")
        try:
            visualize_result(result, args.output_dir)
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
    
    # Summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"Initial crossings: {result['initial_crossings']}")
    print(f"Final crossings:   {result['final_crossings']}")
    print(f"Improvement:       {result['improvement']} ({100*result['improvement']/max(result['initial_crossings'],1):.1f}%)")
    if 'num_operations_predicted' in result:
        print(f"Operations:        {result['num_operations_applied']} applied / {result['num_operations_predicted']} predicted")
    else:
        print(f"Operations:        {result['num_operations_applied']} applied")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
