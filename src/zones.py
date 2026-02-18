"""
Zones.py - Zone Pattern Generators for Multi-Material 3D Printing

This module provides functions to generate different zone patterns that represent
material regions in a 3D printing layer. Each pattern divides a WxH grid into
distinct zones, where each zone corresponds to a different material.

Zone crossings (transitions between zones along the toolpath) are what we aim
to minimize through optimization, as each crossing requires a material change
which slows down the printing process.

Available Patterns:
    - zones_left_right: Vertical bands (left to right)
    - zones_top_bottom: Horizontal bands (top to bottom)
    - zones_diagonal: Diagonal split with random offset
    - zones_stripes: Parallel stripes (vertical or horizontal)
    - zones_checkerboard: Alternating checkerboard pattern
    - zones_voronoi: Irregular regions based on random seed points

Usage:
    from zones import zones_stripes, zones_voronoi

    # Create 3 vertical stripes on a 32x32 grid
    zones = zones_stripes(32, 32, direction="v", k=3)

    # Create Voronoi-based zones with 3 regions
    zones, meta = zones_voronoi(32, 32, k=3)
    print(meta['seeds'])  # Random seed point locations

Author: AI-in-3D-Printing Team
"""

import random
from typing import Dict, Tuple, List

# Type alias for grid coordinates
Coord = Tuple[int, int]

# left - right
def zones_left_right(W: int, H: int, k: int = 2) -> Dict[Coord, int]:
    band_w = max(1, W / k)
    z: Dict[Coord, int] = {}
    for y in range(H):
        for x in range(W):
            zone = min(int(x // band_w), k - 1)
            z[(x, y)] = zone
    return z

# Up - bottom
def zones_top_bottom(W: int, H: int, k: int = 2) -> Dict[Coord, int]:
    band_h = max(1, H / k)
    z: Dict[Coord, int] = {}
    for y in range(H):
        zone = min(int(y // band_h), k - 1)
        for x in range(W):
            z[(x, y)] = zone
    return z

# Diognal
def zones_diagonal(W: int, H: int) -> Dict[Coord, int]:
    """
    Always produces a diagonal split (45°). Randomly shifts the diagonal.
    Zone 0: one side of the diagonal, Zone 1: the other side.
    """
    # shift controls where the diagonal passes through the grid
    shift = random.randint(-H // 3, H // 3)

    z: Dict[Coord, int] = {}
    for y in range(H):
        for x in range(W):
            # boundary is roughly y = x + shift
            z[(x, y)] = 0 if (y - x) < shift else 1
    return z

# Stripes
def zones_stripes(W: int, H: int, direction: str = "v", k: int = 3) -> Dict[Coord, int]:
    z: Dict[Coord, int] = {}
    if direction == "v":
        band_w = max(1, W / k)
        for y in range(H):
            for x in range(W):
                z[(x, y)] = min(int(x // band_w), k - 1)
    else:
        band_h = max(1, H / k)
        for y in range(H):
            zone = min(int(y // band_h), k - 1)
            for x in range(W):
                z[(x, y)] = zone
    return z

# Checkerboard
def zones_checkerboard(W: int, H: int, kx: int = 2, ky: int = 2) -> Dict[Coord, int]:
    cell_w = max(1, W // (kx * 2))
    cell_h = max(1, H // (ky * 2))
    z: Dict[Coord, int] = {}
    for y in range(H):
        by = (y // cell_h) % (2 * ky)
        for x in range(W):
            bx = (x // cell_w) % (2 * kx)
            z[(x, y)] = (bx + by) % 2
    return z

# Voronoi Diagram
def zones_voronoi(W: int, H: int, k: int = 3):
    seeds: List[Coord] = [(random.randrange(W), random.randrange(H)) for _ in range(k)]
    def d2(p: Coord, q: Coord) -> int:
        return (p[0] - q[0])**2 + (p[1] - q[1])**2

    z: Dict[Coord, int] = {}
    for y in range(H):
        for x in range(W):
            best = min(range(k), key=lambda i: d2((x, y), seeds[i]))
            z[(x, y)] = best
    return z, {"seeds": seeds}

