"""
Core module containing fundamental data structures and operations.

- hamiltonian: HamiltonianSTL class for grid-based path management
- zones: Zone generation functions for grid partitioning
"""

from .hamiltonian import HamiltonianSTL
from .zones import (
    zones_left_right,
    zones_top_bottom,
    zones_diagonal,
    zones_stripes,
    zones_checkerboard,
    zones_voronoi,
)

__all__ = [
    "HamiltonianSTL",
    "zones_left_right",
    "zones_top_bottom",
    "zones_diagonal",
    "zones_stripes",
    "zones_checkerboard",
    "zones_voronoi",
]
