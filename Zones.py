"""
Zone generation functions for grid partitioning.
"""

from typing import Dict, Tuple, List
import random

Point = Tuple[int, int]


def zones_left_right(W: int, H: int) -> Dict[Point, int]:
    """Split grid into left (zone 1) and right (zone 2) halves."""
    return {(x, y): (1 if x < W // 2 else 2) for y in range(H) for x in range(W)}


def zones_top_bottom(W: int, H: int) -> Dict[Point, int]:
    """Split grid into top (zone 1) and bottom (zone 2) halves."""
    return {(x, y): (1 if y < H // 2 else 2) for y in range(H) for x in range(W)}


def zones_diagonal(W: int, H: int) -> Dict[Point, int]:
    """Split grid diagonally into two zones."""
    zones = {}
    for y in range(H):
        for x in range(W):
            # zone 1 if below diagonal, zone 2 if above
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
    """
    zones = {}
    
    if direction == "v":
        # Vertical stripes
        stripe_width = W / k
        for y in range(H):
            for x in range(W):
                zone_id = int(x / stripe_width) + 1
                zones[(x, y)] = min(zone_id, k)  # Ensure we don't exceed k zones
    
    elif direction == "h":
        # Horizontal stripes
        stripe_height = H / k
        for y in range(H):
            for x in range(W):
                zone_id = int(y / stripe_height) + 1
                zones[(x, y)] = min(zone_id, k)  # Ensure we don't exceed k zones
    
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
    """
    zones = {}
    for y in range(H):
        for x in range(W):
            # Determine which checker square we're in
            checker_x = x // size
            checker_y = y // size
            # Alternate between zone 1 and 2
            zones[(x, y)] = 1 if (checker_x + checker_y) % 2 == 0 else 2
    return zones


def zones_voronoi(W: int, H: int, k: int = 3, seed: int = 42) -> Tuple[Dict[Point, int], Dict]:
    """
    Create Voronoi diagram zones with k random seeds.
    
    Args:
        W: Grid width
        H: Grid height
        k: Number of Voronoi seeds (zones)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (zones dict, metadata dict with seed positions)
    """
    rng = random.Random(seed)
    
    # Generate k random seed points
    seeds = [(rng.randint(0, W-1), rng.randint(0, H-1)) for _ in range(k)]
    
    # Assign each point to nearest seed
    zones = {}
    for y in range(H):
        for x in range(W):
            # Find nearest seed
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
