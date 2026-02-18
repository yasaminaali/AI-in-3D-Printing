"""
numba_ops.py - Numba JIT-compiled operations for Hamiltonian path validation

Provides ~60-100x speedup over pure Python BFS on grid graphs by compiling
the connectivity check and degree validation to native machine code.

The two key functions:
  - check_degrees: O(N) scan with early exit on first invalid degree (~0.001ms for failures)
  - bfs_connected: Full BFS connectivity check (~0.05ms for 100x100 grid)

Combined, these replace Python's ~3-5ms validate_full_path with ~0.05ms.

Usage:
    from numba_ops import fast_validate_path, FastHamiltonianSTL

    h = FastHamiltonianSTL(100, 100, init_pattern='zigzag')
    assert h.validate_full_path()  # Uses Numba internally
"""

import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:

    @numba.njit(cache=True)
    def check_degrees(H_flat, V_flat, width, height):
        """
        Fast degree check: verify all nodes have degree 1 or 2 (Hamiltonian path).
        Returns False immediately on first invalid node (degree 0 or >=3).
        ~0.001ms on early exit, ~0.02ms full scan for 100x100 grid.
        """
        wm1 = width - 1
        hm1 = height - 1
        N = width * height
        for node in range(N):
            x = node % width
            y = node // width
            d = 0
            if x < wm1 and H_flat[y * wm1 + x]:
                d += 1
            if x > 0 and H_flat[y * wm1 + x - 1]:
                d += 1
            if y < hm1 and V_flat[y * width + x]:
                d += 1
            if y > 0 and V_flat[(y - 1) * width + x]:
                d += 1
            if d == 0 or d >= 3:
                return False
        return True

    @numba.njit(cache=True)
    def bfs_connected(H_flat, V_flat, width, height):
        """
        BFS connectivity check compiled to native code.
        Verifies all W*H nodes are reachable from (0,0).
        ~0.05ms for 100x100 grid (vs ~3-5ms in pure Python).
        """
        N = width * height
        visited = np.zeros(N, dtype=numba.boolean)
        # Stack pre-allocated - max size bounded by edges (~2N for grid)
        stack = np.empty(N, dtype=np.int32)
        sp = 0
        stack[0] = 0
        sp = 1
        count = 0
        wm1 = width - 1
        hm1 = height - 1

        while sp > 0:
            sp -= 1
            node = stack[sp]
            if visited[node]:
                continue
            visited[node] = True
            count += 1

            x = node % width
            y = node // width

            # Right: H[y][x]
            if x < wm1 and H_flat[y * wm1 + x]:
                nb = node + 1
                if not visited[nb]:
                    stack[sp] = nb
                    sp += 1

            # Left: H[y][x-1]
            if x > 0 and H_flat[y * wm1 + x - 1]:
                nb = node - 1
                if not visited[nb]:
                    stack[sp] = nb
                    sp += 1

            # Down: V[y][x]
            if y < hm1 and V_flat[y * width + x]:
                nb = node + width
                if not visited[nb]:
                    stack[sp] = nb
                    sp += 1

            # Up: V[y-1][x]
            if y > 0 and V_flat[(y - 1) * width + x]:
                nb = node - width
                if not visited[nb]:
                    stack[sp] = nb
                    sp += 1

        return count == N

    def fast_validate_path(H, V, width, height):
        """
        Fast path validation: degree check (with early exit) + BFS.
        H, V are numpy 2D bool arrays.
        """
        H_flat = H.ravel()
        V_flat = V.ravel()
        if not check_degrees(H_flat, V_flat, width, height):
            return False
        return bool(bfs_connected(H_flat, V_flat, width, height))

    # Warm up Numba JIT (compile on import, not during first SA run)
    _wH = np.array([[True]], dtype=np.bool_)
    _wV = np.array([[True]], dtype=np.bool_)
    _ = check_degrees(_wH.ravel(), _wV.ravel(), 2, 2)
    _ = bfs_connected(_wH.ravel(), _wV.ravel(), 2, 2)
    del _wH, _wV

else:
    # Fallback: pure Python (still used if numba not installed)
    def fast_validate_path(H, V, width, height):
        """Fallback pure-Python BFS (no numba)."""
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
                if 0 <= nx < width and 0 <= ny < height:
                    if x == nx:  # vertical
                        if V[min(y, ny), x]:
                            stack.append((nx, ny))
                    else:  # horizontal
                        if H[y, min(x, nx)]:
                            stack.append((nx, ny))
        return len(visited) == width * height


# ============================================================
# FastHamiltonianSTL - numpy-backed subclass
# ============================================================
from operations import HamiltonianSTL


class FastHamiltonianSTL(HamiltonianSTL):
    """
    HamiltonianSTL with numpy arrays for H/V and Numba-compiled validation.

    Drop-in replacement: all existing methods (set_edge, has_edge, get_subgrid,
    transpose_subgrid, flip_subgrid) work unchanged because numpy 2D arrays
    support the same [y][x] indexing as lists-of-lists.

    validate_full_path() is overridden to use Numba JIT (~60-100x faster).
    """

    def __init__(self, width: int, height: int, init_pattern: str = 'zigzag'):
        # Parent __init__ creates H/V as lists and calls path builder
        super().__init__(width, height, init_pattern=init_pattern)
        # Convert to numpy 2D bool arrays
        self.H = np.array(self.H, dtype=np.bool_)
        self.V = np.array(self.V, dtype=np.bool_)

    def clear_edges(self):
        """Override: create numpy arrays instead of lists."""
        self.H = np.zeros((self.height, self.width - 1), dtype=np.bool_)
        self.V = np.zeros((self.height - 1, self.width), dtype=np.bool_)

    def validate_full_path(self) -> bool:
        """Override: Numba JIT-compiled BFS. ~0.05ms vs ~3-5ms."""
        return fast_validate_path(self.H, self.V, self.width, self.height)
