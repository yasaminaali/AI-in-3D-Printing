from typing import Tuple, List, Optional, Set
import random

Point = Tuple[int, int]

class HamiltonianSTL:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.H = [[False] * (width - 1) for _ in range(height)]
        self.V = [[False] * width       for _ in range(height - 1)]
        self.zigzag()
        #self.snake_bends()
        #self.hilbert()
        #self.fermat_spiral()

    # Multipla Initial Paths : Zigzag - Snake Bends - Hilbert - Random

    # Zigzag initial path
    def zigzag(self):
        for y in range(self.height):
            for x in range(self.width - 1):
                self.H[y][x] = True
            if y < self.height - 1:
                self.V[y][0 if y % 2 else self.width - 1] = True

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
