"""
Zone generation functions for grid partitioning.

Zones partition the grid into regions for computing crossing penalties.
Each cell (x, y) maps to a zone ID (1-indexed).
"""

from typing import Dict, Tuple, List
import random

Point = Tuple[int, int]


def zones_left_right(W: int, H: int) -> Dict[Point, int]:
    """
    Split grid into left (zone 1) and right (zone 2) halves.
    
    Args:
        W: Grid width
        H: Grid height
        
    Returns:
        Dictionary mapping (x, y) to zone ID
    """
    return {(x, y): (1 if x < W // 2 else 2) for y in range(H) for x in range(W)}


def zones_top_bottom(W: int, H: int) -> Dict[Point, int]:
    """
    Split grid into top (zone 1) and bottom (zone 2) halves.
    
    Args:
        W: Grid width
        H: Grid height
        
    Returns:
        Dictionary mapping (x, y) to zone ID
    """
    return {(x, y): (1 if y < H // 2 else 2) for y in range(H) for x in range(W)}


def zones_diagonal(W: int, H: int) -> Dict[Point, int]:
    """
    Split grid diagonally into two zones.
    
    Zone 1: Below/on the main diagonal
    Zone 2: Above the main diagonal
    
    Args:
        W: Grid width
        H: Grid height
        
    Returns:
        Dictionary mapping (x, y) to zone ID
    """
    zones = {}
    for y in range(H):
        for x in range(W):
            zones[(x, y)] = 1 if (x + y) < (W + H) // 2 else 2
    return zones


def zones_stripes(W: int, H: int, direction: str = "v", k: int = 3) -> Dict[Point, int]:
    """
    Create striped zones.
    
    Args:
        W: Grid width
        H: Grid height
        direction: 'v' for vertical stripes, 'h' for horizontal stripes
        k: Number of stripes
        
    Returns:
        Dictionary mapping (x, y) to zone ID
        
    Raises:
        ValueError: If direction is not 'v' or 'h'
    """
    zones = {}
    
    if direction == "v":
        # Vertical stripes
        stripe_width = W / k
        for y in range(H):
            for x in range(W):
                zone_id = int(x / stripe_width) + 1
                zones[(x, y)] = min(zone_id, k)
    
    elif direction == "h":
        # Horizontal stripes
        stripe_height = H / k
        for y in range(H):
            for x in range(W):
                zone_id = int(y / stripe_height) + 1
                zones[(x, y)] = min(zone_id, k)
    
    else:
        raise ValueError(f"Unknown direction: {direction}. Use 'v' or 'h'")
    
    return zones


def zones_checkerboard(W: int, H: int, size: int = 2) -> Dict[Point, int]:
    """
    Create checkerboard pattern zones.
    
    Args:
        W: Grid width
        H: Grid height
        size: Size of each checkerboard square
        
    Returns:
        Dictionary mapping (x, y) to zone ID
    """
    zones = {}
    for y in range(H):
        for x in range(W):
            checker_x = x // size
            checker_y = y // size
            zones[(x, y)] = 1 if (checker_x + checker_y) % 2 == 0 else 2
    return zones


def zones_voronoi(W: int, H: int, k: int = 3, seed: int = 42) -> Tuple[Dict[Point, int], Dict]:
    """
    Create Voronoi diagram zones with k random seeds.
    
    Each cell is assigned to the zone of the nearest seed point
    using Euclidean distance.
    
    Args:
        W: Grid width
        H: Grid height
        k: Number of Voronoi seeds (zones)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of:
        - zones: Dictionary mapping (x, y) to zone ID
        - metadata: Dictionary with seed positions
    """
    rng = random.Random(seed)
    
    # Generate k random seed points
    seeds = [(rng.randint(0, W - 1), rng.randint(0, H - 1)) for _ in range(k)]
    
    # Assign each point to nearest seed
    zones = {}
    for y in range(H):
        for x in range(W):
            min_dist = float('inf')
            nearest_zone = 1
            
            for idx, (sx, sy) in enumerate(seeds):
                dist = (x - sx) ** 2 + (y - sy) ** 2
                if dist < min_dist:
                    min_dist = dist
                    nearest_zone = idx + 1  # Zones are 1-indexed
            
            zones[(x, y)] = nearest_zone
    
    metadata = {"seeds": seeds}
    return zones, metadata
