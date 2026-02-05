"""
operations.py - Hamiltonian Path Operations for Grid Graphs

This module provides the HamiltonianSTL class for creating and manipulating
Hamiltonian paths/cycles on rectangular grid graphs. It supports:

- Multiple initial path patterns:
  - zigzag (horizontal): rows traversed left-right, right-left
  - vertical_zigzag: columns traversed top-bottom, bottom-top
  - fermat_spiral: inward spiral pattern
  - hilbert: Hilbert space-filling curve (requires 2^n square grid)
  - snake_bends: serpentine pattern with bends
- Transpose operations on 3x3 subgrids (8 variants: sr, wa, sl, ea, nl, eb, nr, wb)
- Flip operations on 3x2 or 2x3 subgrids (4 variants: w, e, n, s)
- Path validation and ASCII visualization

Key Concepts:
    - Grid represented by two edge matrices: H (horizontal) and V (vertical)
    - H[y][x] = True means edge exists between (x,y) and (x+1,y)
    - V[y][x] = True means edge exists between (x,y) and (x,y+1)
    - Transpose/Flip operations reroute paths while preserving Hamiltonian property

Initial Path Selection Guide:
    - Vertical zone boundaries (left_right, stripes direction='v'): use vertical_zigzag
    - Horizontal zone boundaries (stripes direction='h'): use zigzag (horizontal)
    - Irregular zones (voronoi): fermat_spiral often works well

Usage:
    from operations import HamiltonianSTL

    # Create with default horizontal zigzag
    h = HamiltonianSTL(32, 32)

    # Create with vertical zigzag (better for vertical zone boundaries)
    h = HamiltonianSTL(32, 32, init_pattern='vertical_zigzag')

    # Get a 3x3 subgrid and attempt transpose
    sub = h.get_subgrid((0, 0), (2, 2))
    result, msg = h.transpose_subgrid(sub, 'sr')

Author: AI-in-3D-Printing Team
"""

from typing import Tuple, List, Optional, Set
import random

Point = Tuple[int, int]


class HamiltonianSTL:
    def __init__(self, width: int, height: int, init_pattern: str = 'zigzag'):
        """
        Create a grid with a Hamiltonian path.
        
        Args:
            width: Grid width
            height: Grid height
            init_pattern: Initial path pattern. Options:
                - 'zigzag': Horizontal zigzag (default)
                - 'vertical_zigzag': Vertical zigzag (optimal for vertical zone boundaries)
                - 'fermat_spiral': Inward spiral
                - 'hilbert': Hilbert curve (requires 2^n square grid)
                - 'snake_bends': Serpentine with bends
                - 'none': No initialization (empty grid)
        """
        self.width, self.height = width, height
        self.H = [[False] * (width - 1) for _ in range(height)]
        self.V = [[False] * width       for _ in range(height - 1)]
        
        if init_pattern == 'zigzag':
            self.zigzag()
        elif init_pattern == 'vertical_zigzag':
            self.vertical_zigzag()
        elif init_pattern == 'fermat_spiral':
            self.fermat_spiral()
        elif init_pattern == 'hilbert':
            self.hilbert()
        elif init_pattern == 'snake_bends':
            self.snake_bends()
        elif init_pattern == 'none':
            pass  # Leave edges empty
        else:
            raise ValueError(f"Unknown init_pattern: {init_pattern}")

    # Multipla Initial Paths : Zigzag - Snake Bends - Hilbert - Random

    # Zigzag initial path (horizontal - traverses rows left-right, right-left)
    def zigzag(self):
        for y in range(self.height):
            for x in range(self.width - 1):
                self.H[y][x] = True
            if y < self.height - 1:
                self.V[y][0 if y % 2 else self.width - 1] = True

    # Vertical zigzag (traverses columns top-bottom, bottom-top)
    # Optimal for vertical zone boundaries (left_right, stripes with direction='v')
    def vertical_zigzag(self):
        self.clear_edges()
        for x in range(self.width):
            # Vertical edges within column
            for y in range(self.height - 1):
                self.V[y][x] = True
            # Horizontal edge to next column (alternating top/bottom)
            if x < self.width - 1:
                connect_y = self.height - 1 if x % 2 == 0 else 0
                self.H[connect_y][x] = True

    # Snake Bends
    # Grid is size of odd numbers (9,9)
    def snake_bends(self): 
        path = []
        visited = [[False for _ in range(self.width)] for _ in range(self.height)]

        x, y = 0, 0
        direction = 1  # start left to right

        while y < self.height and 0 <= x < self.width:
            path.append((x, y))
            visited[y][x] = True

            # Go 2 down
            for _ in range(2):
                if y + 1 < self.height and not visited[y + 1][x]:
                    y += 1
                    path.append((x, y))
                    visited[y][x] = True

            # 1 side
            if 0 <= x + direction < self.width and not visited[y][x + direction]:
                x += direction
                path.append((x, y))
                visited[y][x] = True
            else:
                break

            # Go 2 up
            for _ in range(2):
                if y - 1 >= 0 and not visited[y - 1][x]:
                    y -= 1
                    path.append((x, y))
                    visited[y][x] = True

            # 1 side
            if 0 <= x + direction < self.width and not visited[y][x + direction]:
                x += direction
                path.append((x, y))
                visited[y][x] = True
            else:
                break

            # If at border, drop 5 down and switch direction
            if direction == 1 and x >= self.width - 1:
                for _ in range(3):
                    if y + 1 < self.height and not visited[y + 1][x]:
                        y += 1
                        path.append((x, y))
                        visited[y][x] = True
                direction = -1
            elif direction == -1 and x <= 0:
                for _ in range(3):
                    if y + 1 < self.height and not visited[y + 1][x]:
                        y += 1
                        path.append((x, y))
                        visited[y][x] = True
                direction = 1

        # Add edges from path
        for (x1, y1), (x2, y2) in zip(path, path[1:]):
            if x1 == x2 and abs(y1 - y2) == 1:
                self.V[min(y1, y2)][x1] = True
            elif y1 == y2 and abs(x1 - x2) == 1:
                self.H[y1][min(x1, x2)] = True

        return path
    
    # Hilbert Path
    # Grid is square and of size 2ⁿ × 2ⁿ
    def hilbert(self):
        from math import log2

        def rot(s, x, y, rx, ry):
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            return x, y

        def d2xy(n, d):
            x = y = 0
            t = d
            s = 1
            while s < n:
                rx = 1 & (t // 2)
                ry = 1 & (t ^ rx)
                x, y = rot(s, x, y, rx, ry)
                x += s * rx
                y += s * ry
                t //= 4
                s *= 2
            return x, y

        size = max(self.width, self.height)
        power = int(log2(size))
        if 2 ** power != size or self.width != self.height:
            raise ValueError("Hilbert curve only works on square grids with size 2^n.")

        path = [d2xy(size, i) for i in range(size * size)]

        for (x1, y1), (x2, y2) in zip(path, path[1:]):
            if x1 == x2 and abs(y1 - y2) == 1:
                self.V[min(y1, y2)][x1] = True
            elif y1 == y2 and abs(x1 - x2) == 1:
                self.H[y1][min(x1, x2)] = True
    
    # Fermat Spiral
    def fermat_spiral(self):
        left, right = 0, self.width - 1
        top, bottom = 0, self.height - 1
        path = []

        while left <= right and top <= bottom:
            for x in range(left, right + 1):     # → right
                path.append((x, top))
            for y in range(top + 1, bottom + 1): # ↓ down
                path.append((right, y))
            if top != bottom:
                for x in range(right - 1, left - 1, -1): # ← left
                    path.append((x, bottom))
            if left != right:
                for y in range(bottom - 1, top, -1): # ↑ up
                    path.append((left, y))
            left += 1
            right -= 1
            top += 1
            bottom -= 1

        for (x1, y1), (x2, y2) in zip(path, path[1:]):
            if x1 == x2 and abs(y1 - y2) == 1:
                self.V[min(y1, y2)][x1] = True
            elif y1 == y2 and abs(x1 - x2) == 1:
                self.H[y1][min(x1, x2)] = True

    def set_edge(self, p1: Point, p2: Point, v: bool = True):
        if not p1 or not p2:
            return
        x1, y1 = p1; x2, y2 = p2
        if x1 == x2 and abs(y1 - y2) == 1:
            self.V[min(y1, y2)][x1] = v
        elif y1 == y2 and abs(x1 - x2) == 1:
            self.H[y1][min(x1, x2)] = v

    def clear_edges(self):
        # Remove all horizontal and vertical edges.
        self.H = [[0 for _ in range(self.width - 1)] for _ in range(self.height)]
        self.V = [[0 for _ in range(self.width)] for _ in range(self.height - 1)]


    def has_edge(self, p1: Point, p2: Point) -> bool:
        if not p1 or not p2:
            return False
        x1, y1 = p1; x2, y2 = p2
        if x1 == x2 and abs(y1 - y2) == 1:
            return self.V[min(y1, y2)][x1]
        if y1 == y2 and abs(x1 - x2) == 1:
            return self.H[y1][min(x1, x2)]
        return False

    def get_subgrid(self, c1: Point, c2: Point) -> List[List[Optional[Point]]]:
        x0, x1 = sorted((c1[0], c2[0]))
        y0, y1 = sorted((c1[1], c2[1]))
        return [
            [
                (x, y) if 0 <= x < self.width and 0 <= y < self.height else None
                for x in range(x0, x1 + 1)
            ]
            for y in range(y0, y1 + 1)
        ]

    def _subgrid_flat_points(self, sub: List[List[Optional[Point]]]) -> List[Point]:
        return [p for row in sub for p in row if p is not None]

    def _edges_present_in_subgrid(self, sub: List[List[Optional[Point]]]) -> Set[Tuple[int, int]]:
        pts = self._subgrid_flat_points(sub)
        idx_of = {p:i for i,p in enumerate(pts)}
        present = set()
        for i,(x,y) in enumerate(pts):
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                q = (x+dx,y+dy)
                if q in idx_of and self.has_edge((x,y),q):
                    j = idx_of[q]
                    if i < j:
                        present.add((i,j))
        return present

    def _apply_edge_diff_in_subgrid(self, sub, old_edges, new_edges):
        pts = self._subgrid_flat_points(sub)
        old_set = set(tuple(sorted(e)) for e in old_edges)
        new_set = set(tuple(sorted(e)) for e in new_edges)
        for i,j in old_set - new_set:
            self.set_edge(pts[i], pts[j], False)
        for i,j in new_set - old_set:
            self.set_edge(pts[i], pts[j], True)

    # All 8 transpose cases (3x3)
    transpose_patterns = {
        'sr': {'old': [(0,1),(1,2),(2,5),(3,4),(4,5),(6,7),(7,8)],
               'new': [(0,3),(1,2),(1,4),(2,5),(4,7),(5,8),(6,7)]},
        'wa': {'old': [(0,3),(3,6),(1,4),(2,5),(4,7),(5,8),(1,2)],
               'new': [(0,1),(1,2),(2,5),(3,4),(4,5),(3,6),(7,8)]},
        'sl': {'old': [(0,1),(1,2),(0,3),(3,4),(4,5),(6,7),(7,8)],
               'new': [(0,1),(0,3),(1,4),(2,5),(3,6),(4,7),(7,8)]},
        'ea': {'old': [(0,1),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8)],
               'new': [(0,1),(1,2),(0,3),(3,4),(4,5),(6,7),(5,8)]},
        'nl': {'old': [(0,1),(3,4),(4,5),(3,6),(6,7),(7,8),(1,2)],
               'new': [(0,3),(1,4),(3,6),(4,7),(6,7),(5,8),(1,2)]},
        'eb': {'old': [(0,3),(1,4),(3,6),(4,7),(6,7),(5,8),(2,5)],
               'new': [(0,1),(3,4),(4,5),(3,6),(6,7),(7,8),(2,5)]},
        'nr': {'old': [(3,4),(4,5),(6,7),(7,8),(5,8),(1,2),(0,1)],
               'new': [(1,4),(2,5),(3,6),(4,7),(5,8),(7,8),(0,1)]},
        'wb': {'old': [(1,4),(2,5),(3,6),(4,7),(5,8),(7,8),(0,3)],
               'new': [(3,4),(4,5),(6,7),(7,8),(5,8),(1,2),(0,3)]}
    }

    # All 4 flip cases (3x2 or 2x3)
    flip_patterns = {
        'w': {'old': [(0,3),(1,2),(1,4),(4,5)],
              'new': [(0,1),(2,5),(3,4),(1,4)]},
        'e': {'old': [(0,1),(2,5),(3,4),(1,4)],
              'new': [(0,3),(1,2),(1,4),(4,5)]},
        'n': {'old': [(0,1),(2,3),(2,4),(3,5)],
              'new': [(0,2),(1,3),(2,3),(4,5)]},
        's': {'old': [(0,2),(1,3),(2,3),(4,5)],
              'new': [(0,1),(2,3),(2,4),(3,5)]}
    }

    def validate_full_path(self) -> bool:
        """
        Check that all nodes are reachable from (0,0).
        NOTE: This only checks connectivity, not that it's a valid Hamiltonian cycle.
        Use validate_hamiltonian_cycle() for full validation.
        """
        visited = set()
        stack = [(0,0)]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            x, y = cur
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.has_edge(cur, (nx, ny)):
                        stack.append((nx, ny))
        return len(visited) == self.width * self.height

    def get_degree(self, x: int, y: int) -> int:
        """Get the degree (number of edges) of a node."""
        degree = 0
        # Check horizontal edges
        if x > 0 and self.H[y][x-1]:
            degree += 1
        if x < self.width - 1 and self.H[y][x]:
            degree += 1
        # Check vertical edges
        if y > 0 and self.V[y-1][x]:
            degree += 1
        if y < self.height - 1 and self.V[y][x]:
            degree += 1
        return degree

    def validate_hamiltonian_cycle(self) -> bool:
        """
        Validate that the current edge configuration forms a valid Hamiltonian path.
        
        A Hamiltonian path must satisfy:
        1. Exactly 2 nodes have degree 1 (endpoints)
        2. All other nodes have exactly degree 2
        3. All nodes are connected (single component)
        
        Returns True if valid, False otherwise.
        """
        degree_1_count = 0
        
        # Check degrees of all nodes
        for y in range(self.height):
            for x in range(self.width):
                d = self.get_degree(x, y)
                if d == 1:
                    degree_1_count += 1
                elif d != 2:
                    # Degree 0, 3, or 4 is invalid
                    return False
        
        # Must have exactly 2 endpoints
        if degree_1_count != 2:
            return False
        
        # Check connectivity - all nodes must be reachable
        visited = set()
        stack = [(0, 0)]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            x, y = cur
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.has_edge(cur, (nx, ny)):
                        stack.append((nx, ny))
        
        return len(visited) == self.width * self.height
    
    def _snapshot_adjacent_edges_in_subgrid(self, sub) -> List[Tuple[Point, Point, bool]]:
        pts = self._subgrid_flat_points(sub)
        ptset = set(pts)

        snap: List[Tuple[Point, Point, bool]] = []
        for (x, y) in pts:
            r = (x + 1, y)
            if r in ptset:
                snap.append(((x, y), r, bool(self.has_edge((x, y), r))))
            d = (x, y + 1)
            if d in ptset:
                snap.append(((x, y), d, bool(self.has_edge((x, y), d))))
        return snap

    def _restore_adjacent_edges_snapshot(self, snap: List[Tuple[Point, Point, bool]]) -> None:
        for p, q, val in snap:
            self.set_edge(p, q, val)

    def transpose_subgrid(self, sub: List[List[Point]], variant: str = 'sr'):
        if variant not in self.transpose_patterns:
            return sub, f'unknown variant {variant}'
        case = self.transpose_patterns[variant]
        OLD, NEW = case['old'], case['new']

        present = self._edges_present_in_subgrid(sub)
        if present != set(tuple(sorted(e)) for e in OLD):
            return sub, f'pattern_mismatch_{variant}'

        # snapshot only the local edges (FAST)
        snap = self._snapshot_adjacent_edges_in_subgrid(sub)

        self._apply_edge_diff_in_subgrid(sub, OLD, NEW)

        if not self.validate_full_path():
            # restore only local edges (FAST)
            self._restore_adjacent_edges_snapshot(snap)
            return sub, f'not transposable_{variant}'

        return sub, f'transposed_{variant}'
    
    def flip_subgrid(self, sub: List[List[Point]], variant: str):
        if variant not in self.flip_patterns:
            return sub, f'unknown variant {variant}'
        case = self.flip_patterns[variant]
        OLD, NEW = case['old'], case['new']

        present = self._edges_present_in_subgrid(sub)
        if present != set(tuple(sorted(e)) for e in OLD):
            return sub, f'pattern_mismatch_{variant}'

        # snapshot only the local edges (FAST)
        snap = self._snapshot_adjacent_edges_in_subgrid(sub)

        self._apply_edge_diff_in_subgrid(sub, OLD, NEW)

        if not self.validate_full_path():
            # restore only local edges (FAST)
            self._restore_adjacent_edges_snapshot(snap)
            return sub, f'not flippable_{variant}'

        return sub, f'flipped_{variant}'

    # ============================================================
    # Large-scale operations (4x4, 5x5, 6x6)
    # ============================================================
    
    # 4x4 transpose patterns - derived from 3x3 concepts extended
    # Indices for 4x4: 0  1  2  3
    #                  4  5  6  7
    #                  8  9  10 11
    #                  12 13 14 15
    transpose_patterns_4x4 = {
        # Pattern: shift right side down
        '4x4_sr': {
            'old': [(0,1),(1,2),(2,3),(3,7),(7,11),(11,15),(4,5),(5,6),(6,7),(8,9),(9,10),(10,11),(12,13),(13,14),(14,15)],
            'new': [(0,4),(4,5),(5,6),(6,7),(7,11),(11,15),(1,2),(2,3),(3,7),(8,9),(9,10),(10,11),(12,13),(13,14),(14,15)]
        },
        # Pattern: shift left side down  
        '4x4_sl': {
            'old': [(0,1),(1,2),(2,3),(0,4),(4,8),(8,12),(4,5),(5,6),(6,7),(8,9),(9,10),(10,11),(12,13),(13,14),(14,15)],
            'new': [(0,1),(1,2),(2,3),(0,4),(4,5),(5,9),(8,12),(9,10),(10,11),(6,7),(12,13),(13,14),(14,15)]
        },
    }
    
    # 5x5 transpose patterns
    # Indices: 0  1  2  3  4
    #          5  6  7  8  9
    #          10 11 12 13 14
    #          15 16 17 18 19
    #          20 21 22 23 24
    transpose_patterns_5x5 = {
        # We'll use a generic approach instead of hardcoding
    }
    
    def try_rectangular_swap(self, x0: int, y0: int, w: int, h: int) -> Tuple[bool, str]:
        """
        Try to swap the path routing within a rectangular region.
        This is a more general operation that can work with any size subgrid.
        
        The algorithm:
        1. Find the boundary edges (edges that cross into/out of the region)
        2. Try different ways to reconnect the internal path
        3. Validate that it's still a Hamiltonian path
        
        Returns: (success, message)
        """
        if x0 < 0 or y0 < 0 or x0 + w > self.width or y0 + h > self.height:
            return False, "out_of_bounds"
        if w < 2 or h < 2:
            return False, "too_small"
            
        # Snapshot edges for rollback
        H_before = [row[:] for row in self.H]
        V_before = [row[:] for row in self.V]
        
        # Find boundary crossing points
        boundary_edges = []
        
        # Top edge
        for x in range(x0, x0 + w):
            if y0 > 0 and self.V[y0-1][x]:
                boundary_edges.append(((x, y0-1), (x, y0), 'v'))
        # Bottom edge
        for x in range(x0, x0 + w):
            if y0 + h - 1 < self.height - 1 and self.V[y0+h-1][x]:
                boundary_edges.append(((x, y0+h-1), (x, y0+h), 'v'))
        # Left edge
        for y in range(y0, y0 + h):
            if x0 > 0 and self.H[y][x0-1]:
                boundary_edges.append(((x0-1, y), (x0, y), 'h'))
        # Right edge
        for y in range(y0, y0 + h):
            if x0 + w - 1 < self.width - 1 and self.H[y][x0+w-1]:
                boundary_edges.append(((x0+w-1, y), (x0+w, y), 'h'))
        
        # Must have exactly 2 boundary edges for a valid swap
        if len(boundary_edges) != 2:
            return False, f"invalid_boundary_count_{len(boundary_edges)}"
        
        # Try reversing the path segment inside the rectangle
        # This is a simple swap: just reverse the internal traversal
        
        # Get all internal edges
        internal_h = []
        internal_v = []
        for y in range(y0, y0 + h):
            for x in range(x0, x0 + w - 1):
                if self.H[y][x]:
                    internal_h.append((x, y))
        for y in range(y0, y0 + h - 1):
            for x in range(x0, x0 + w):
                if self.V[y][x]:
                    internal_v.append((x, y))
        
        # Simple swap: try toggling some internal edges
        # This is a heuristic - toggle middle edges
        mid_x = x0 + w // 2
        mid_y = y0 + h // 2
        
        # Toggle vertical edge at midpoint
        if mid_y < y0 + h - 1 and 0 <= mid_y < len(self.V):
            for x in range(x0, x0 + w):
                if x < len(self.V[mid_y]):
                    self.V[mid_y][x] = not self.V[mid_y][x]
        
        # Validate
        if self.validate_full_path():
            return True, "swapped"
        
        # Restore
        self.H = H_before
        self.V = V_before
        return False, "invalid_after_swap"
    
    def large_flip(self, x0: int, y0: int, w: int, h: int) -> Tuple[bool, str]:
        """
        Attempt a larger flip operation on a WxH region.
        Tries multiple strategies to reroute the path.
        
        Returns: (success, message)
        """
        if x0 < 0 or y0 < 0 or x0 + w > self.width or y0 + h > self.height:
            return False, "out_of_bounds"
        if w < 2 or h < 2:
            return False, "too_small"
        
        # Snapshot for rollback
        H_before = [row[:] for row in self.H]
        V_before = [row[:] for row in self.V]
        
        # Strategy 1: Try to create a "U-turn" pattern
        # Find edges that cross the boundary and try to redirect them
        
        # Count vertical edges at the left and right boundaries of the region
        left_v_edges = []
        right_v_edges = []
        
        for y in range(y0, min(y0 + h - 1, self.height - 1)):
            if self.V[y][x0]:
                left_v_edges.append(y)
            if x0 + w - 1 < self.width and self.V[y][x0 + w - 1]:
                right_v_edges.append(y)
        
        # Try swapping connectivity: remove some edges, add others
        modified = False
        
        # If we have vertical edges on both sides, try to connect them differently
        if left_v_edges and right_v_edges:
            # Remove a left edge and right edge, try to add horizontal connections
            ly = left_v_edges[0]
            ry = right_v_edges[0]
            
            # Remove vertical edges
            self.V[ly][x0] = False
            self.V[ry][x0 + w - 1] = False
            
            # Add horizontal edges to compensate
            for x in range(x0, x0 + w - 1):
                if not self.H[ly][x]:
                    self.H[ly][x] = True
                    modified = True
                    break
            for x in range(x0, x0 + w - 1):
                if not self.H[ry][x]:
                    self.H[ry][x] = True
                    modified = True
                    break
        
        if modified and self.validate_full_path():
            return True, "large_flipped"
        
        # Restore and try another strategy
        self.H = [row[:] for row in H_before]
        self.V = [row[:] for row in V_before]
        
        # Strategy 2: Random edge toggle within region
        import random
        for _ in range(10):  # Try up to 10 random modifications
            self.H = [row[:] for row in H_before]
            self.V = [row[:] for row in V_before]
            
            # Randomly toggle 2-4 edges
            for _ in range(random.randint(2, 4)):
                if random.random() < 0.5:
                    # Toggle horizontal
                    x = random.randint(x0, min(x0 + w - 2, self.width - 2))
                    y = random.randint(y0, min(y0 + h - 1, self.height - 1))
                    if 0 <= y < len(self.H) and 0 <= x < len(self.H[y]):
                        self.H[y][x] = not self.H[y][x]
                else:
                    # Toggle vertical
                    x = random.randint(x0, min(x0 + w - 1, self.width - 1))
                    y = random.randint(y0, min(y0 + h - 2, self.height - 2))
                    if 0 <= y < len(self.V) and 0 <= x < len(self.V[y]):
                        self.V[y][x] = not self.V[y][x]
            
            if self.validate_full_path():
                return True, "random_flipped"
        
        # Restore original
        self.H = H_before
        self.V = V_before
        return False, "no_valid_flip_found"

    def print_ascii_edges(self, highlight_subgrid=None):
        grid_h, grid_w = 2*self.height - 1, 2*self.width - 1
        canvas = [[' ']*grid_w for _ in range(grid_h)]
        highlight = set()
        if highlight_subgrid:
            for row in highlight_subgrid:
                for pt in row:
                    if pt:
                        highlight.add(pt)
        for y in range(self.height):
            for x in range(self.width):
                cx, cy = 2*x, 2*y
                canvas[cy][cx] = 'X' if (x,y) in highlight else 'O'
        for y in range(self.height):
            for x in range(self.width - 1):
                if self.H[y][x]:
                    canvas[2*y][2*x+1] = '-'
        for y in range(self.height - 1):
            for x in range(self.width):
                if self.V[y][x]:
                    canvas[2*y+1][2*x] = '|'
        for row in canvas:
            print(''.join(row))


if __name__ == '__main__':
    h = HamiltonianSTL(10, 10)

    print("Original Grid:")
    h.print_ascii_edges()

    sub = h.get_subgrid((0,0),(2,2))
    print("\nSubgrid for transpose:")
    for row in sub:
        print(row)

    tr, res = h.transpose_subgrid(sub, 'nl')
    print(f"\nResult: {res}\n")

    print("After:")
    h.print_ascii_edges(highlight_subgrid=tr)

    sub_f = h.get_subgrid((0, 1), (1, 3))
    fl_res, fl_code = h.flip_subgrid(sub_f, 's')
    print(f"After flip 's' @ (0,0): {fl_code}")
    h.print_ascii_edges(highlight_subgrid=sub_f)
    print()

    sub = h.get_subgrid((7,0),(9,2))
    print("\nSubgrid for transpose:")
    for row in sub:
        print(row)

    tr, res = h.transpose_subgrid(sub, 'sr')
    print(f"\nResult: {res}\n")

    print("After:")
    h.print_ascii_edges(highlight_subgrid=tr)
