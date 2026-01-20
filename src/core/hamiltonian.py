"""
Hamiltonian path operations and edge management for grid graphs.

This module provides the HamiltonianSTL class which manages grid-based
Hamiltonian paths using horizontal (H) and vertical (V) edge matrices.
"""

from typing import Tuple, List, Optional, Set
import random

Point = Tuple[int, int]


class HamiltonianSTL:
    """
    Manages a Hamiltonian path/cycle on a grid graph.
    
    The grid is represented using two boolean matrices:
    - H[y][x]: True if horizontal edge exists connecting (x,y) to (x+1,y)
    - V[y][x]: True if vertical edge exists connecting (x,y) to (x,y+1)
    
    Attributes:
        width: Grid width
        height: Grid height
        H: Horizontal edge matrix
        V: Vertical edge matrix
    """
    
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.H = [[False] * (width - 1) for _ in range(height)]
        self.V = [[False] * width       for _ in range(height - 1)]
        self.zigzag()

    # ---------- Initial Path Patterns ----------

    def zigzag(self):
        """Initialize with a zigzag pattern (default)."""
        for y in range(self.height):
            for x in range(self.width - 1):
                self.H[y][x] = True
            if y < self.height - 1:
                self.V[y][0 if y % 2 else self.width - 1] = True

    def snake_bends(self): 
        """
        Initialize with snake bends pattern.
        Best suited for grids with odd dimensions (e.g., 9x9).
        """
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
    
    def hilbert(self):
        """
        Initialize with Hilbert curve pattern.
        Grid must be square with size 2^n (e.g., 8x8, 16x16).
        """
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
    
    def fermat_spiral(self):
        """Initialize with Fermat spiral (inward spiral) pattern."""
        left, right = 0, self.width - 1
        top, bottom = 0, self.height - 1
        path = []

        while left <= right and top <= bottom:
            for x in range(left, right + 1):     # right
                path.append((x, top))
            for y in range(top + 1, bottom + 1): # down
                path.append((right, y))
            if top != bottom:
                for x in range(right - 1, left - 1, -1): # left
                    path.append((x, bottom))
            if left != right:
                for y in range(bottom - 1, top, -1): # up
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

    # ---------- Edge Operations ----------

    def set_edge(self, p1: Point, p2: Point, v: bool = True):
        """Set or clear an edge between two adjacent points."""
        if not p1 or not p2:
            return
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2 and abs(y1 - y2) == 1:
            self.V[min(y1, y2)][x1] = v
        elif y1 == y2 and abs(x1 - x2) == 1:
            self.H[y1][min(x1, x2)] = v

    def clear_edges(self):
        """Remove all edges from the grid."""
        self.H = [[0 for _ in range(self.width - 1)] for _ in range(self.height)]
        self.V = [[0 for _ in range(self.width)] for _ in range(self.height - 1)]

    def has_edge(self, p1: Point, p2: Point) -> bool:
        """Check if an edge exists between two points."""
        if not p1 or not p2:
            return False
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2 and abs(y1 - y2) == 1:
            return self.V[min(y1, y2)][x1]
        if y1 == y2 and abs(x1 - x2) == 1:
            return self.H[y1][min(x1, x2)]
        return False

    # ---------- Subgrid Operations ----------

    def get_subgrid(self, c1: Point, c2: Point) -> List[List[Optional[Point]]]:
        """Extract a rectangular subgrid between two corner points."""
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
        """Flatten subgrid to list of non-None points."""
        return [p for row in sub for p in row if p is not None]

    def _edges_present_in_subgrid(self, sub: List[List[Optional[Point]]]) -> Set[Tuple[int, int]]:
        """Get set of edge indices present in subgrid."""
        pts = self._subgrid_flat_points(sub)
        idx_of = {p: i for i, p in enumerate(pts)}
        present = set()
        for i, (x, y) in enumerate(pts):
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                q = (x + dx, y + dy)
                if q in idx_of and self.has_edge((x, y), q):
                    j = idx_of[q]
                    if i < j:
                        present.add((i, j))
        return present

    def _apply_edge_diff_in_subgrid(self, sub, old_edges, new_edges):
        """Apply edge changes within a subgrid."""
        pts = self._subgrid_flat_points(sub)
        old_set = set(tuple(sorted(e)) for e in old_edges)
        new_set = set(tuple(sorted(e)) for e in new_edges)
        for i, j in old_set - new_set:
            self.set_edge(pts[i], pts[j], False)
        for i, j in new_set - old_set:
            self.set_edge(pts[i], pts[j], True)

    # ---------- Transformation Patterns ----------

    # All 8 transpose cases (3x3 subgrid)
    # Variants: sr, wa, sl, ea, nl, eb, nr, wb (compass directions)
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

    # All 4 flip cases (3x2 or 2x3 subgrid)
    # Variants: n, s (3x2), e, w (2x3)
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

    # ---------- Validation ----------

    def validate_full_path(self) -> bool:
        """Validate that the grid contains a valid Hamiltonian path."""
        visited = set()
        stack = [(0, 0)]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            x, y = cur
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.has_edge(cur, (nx, ny)):
                        stack.append((nx, ny))
        return len(visited) == self.width * self.height

    # ---------- Transformations ----------

    def transpose_subgrid(self, sub: List[List[Point]], variant: str = 'sr'):
        """
        Apply a transpose operation to a 3x3 subgrid.
        
        Args:
            sub: 3x3 subgrid from get_subgrid()
            variant: One of 'sr', 'wa', 'sl', 'ea', 'nl', 'eb', 'nr', 'wb'
            
        Returns:
            Tuple of (subgrid, result_string)
        """
        if variant not in self.transpose_patterns:
            return sub, f'unknown variant {variant}'
        case = self.transpose_patterns[variant]
        OLD, NEW = case['old'], case['new']

        present = self._edges_present_in_subgrid(sub)
        if present != set(tuple(sorted(e)) for e in OLD):
            return sub, f'pattern_mismatch_{variant}'

        H_snap = [row[:] for row in self.H]
        V_snap = [row[:] for row in self.V]
        self._apply_edge_diff_in_subgrid(sub, OLD, NEW)

        if not self.validate_full_path():
            self.H, self.V = H_snap, V_snap
            return sub, f'not transposable_{variant}'

        return sub, f'transposed_{variant}'

    def flip_subgrid(self, sub: List[List[Point]], variant: str):
        """
        Apply a flip operation to a 3x2 or 2x3 subgrid.
        
        Args:
            sub: Subgrid from get_subgrid()
            variant: One of 'n', 's' (3x2) or 'e', 'w' (2x3)
            
        Returns:
            Tuple of (subgrid, result_string)
        """
        if variant not in self.flip_patterns:
            return sub, f'unknown variant {variant}'
        case = self.flip_patterns[variant]
        OLD, NEW = case['old'], case['new']

        present = self._edges_present_in_subgrid(sub)
        if present != set(tuple(sorted(e)) for e in OLD):
            return sub, f'pattern_mismatch_{variant}'

        H_snap = [row[:] for row in self.H]
        V_snap = [row[:] for row in self.V]
        self._apply_edge_diff_in_subgrid(sub, OLD, NEW)

        if not self.validate_full_path():
            self.H, self.V = H_snap, V_snap
            return sub, f'not flippable_{variant}'

        return sub, f'flipped_{variant}'

    # ---------- Visualization ----------

    def print_ascii_edges(self, highlight_subgrid=None):
        """Print ASCII representation of the grid with edges."""
        grid_h, grid_w = 2 * self.height - 1, 2 * self.width - 1
        canvas = [[' '] * grid_w for _ in range(grid_h)]
        highlight = set()
        if highlight_subgrid:
            for row in highlight_subgrid:
                for pt in row:
                    if pt:
                        highlight.add(pt)
        for y in range(self.height):
            for x in range(self.width):
                cx, cy = 2 * x, 2 * y
                canvas[cy][cx] = 'X' if (x, y) in highlight else 'O'
        for y in range(self.height):
            for x in range(self.width - 1):
                if self.H[y][x]:
                    canvas[2 * y][2 * x + 1] = '-'
        for y in range(self.height - 1):
            for x in range(self.width):
                if self.V[y][x]:
                    canvas[2 * y + 1][2 * x] = '|'
        for row in canvas:
            print(''.join(row))


if __name__ == '__main__':
    h = HamiltonianSTL(10, 10)

    print("Original Grid:")
    h.print_ascii_edges()

    sub = h.get_subgrid((0, 0), (2, 2))
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

    sub = h.get_subgrid((7, 0), (9, 2))
    print("\nSubgrid for transpose:")
    for row in sub:
        print(row)

    tr, res = h.transpose_subgrid(sub, 'sr')
    print(f"\nResult: {res}\n")

    print("After:")
    h.print_ascii_edges(highlight_subgrid=tr)
