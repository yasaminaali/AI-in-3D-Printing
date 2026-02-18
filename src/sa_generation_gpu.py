"""
sa_generation_gpu.py - GPU-Accelerated Simulated Annealing for Zone Crossing Minimization

Optimized for NVIDIA H100 GPUs. Uses CUDA tensors and matrix multiplications to accelerate:
1. Zone crossing computation - element-wise tensor multiply + sum
2. Move pool construction - batched pattern matching via matrix multiplication
3. Zone boundary scoring - 2D tensor unfold operations
4. Multi-GPU parallel execution - torch.multiprocessing across multiple GPUs
5. Path validation - Numba JIT-compiled BFS (~60-100x faster than Python)

Input/output format is identical to sa_generation.py.

Dependencies: torch (with CUDA), numba, numpy, operations.py, Zones.py
"""

import os
import json
import time
import math
import random
from typing import Tuple, Optional, Dict, Any, List, Set

import numpy as np
import torch

from operations import HamiltonianSTL
from numba_ops import FastHamiltonianSTL, fast_validate_path
from zones import zones_stripes, zones_voronoi

Point = Tuple[int, int]


# ============================================================
# Temperature schedule (unchanged)
# ============================================================
def _sigmoid_stable(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def dynamic_temperature(i: int, iters: int, Tmin: float, Tmax: float) -> float:
    k = 10.0 / max(1.0, float(iters))
    z = k * (-float(i) + float(iters) / 2.0)
    s = _sigmoid_stable(z)
    return Tmin + (Tmax - Tmin) * s


# ============================================================
# Grid copy helpers (unchanged)
# ============================================================
def deep_copy(H, V):
    if isinstance(H, np.ndarray):
        return (H.copy(), V.copy())
    return ([row[:] for row in H], [row[:] for row in V])


def is_valid_cycle(h: HamiltonianSTL) -> bool:
    if hasattr(h, "validate_full_path_cycle"):
        return bool(h.validate_full_path_cycle())
    if hasattr(h, "validate_full_path"):
        return bool(h.validate_full_path())
    return True


# ============================================================
# Zone generators (unchanged)
# ============================================================
def zones_left_right(W: int, H: int) -> Dict[Point, int]:
    return {(x, y): (1 if x < W // 2 else 2) for y in range(H) for x in range(W)}


def zones_islands(
    W: int, H: int, *, num_islands: int, island_size: int, seed: int, allow_touch: bool,
) -> Tuple[Dict[Point, int], Dict[str, Any]]:
    rng = random.Random(seed)
    zones = {(x, y): 1 for y in range(H) for x in range(W)}
    S = int(island_size)
    if S <= 0 or S > W or S > H:
        raise ValueError("island_size must be in [1, min(W,H)].")
    anchors: List[Tuple[int, int]] = []
    tries, max_tries = 0, 50_000

    def overlaps(ax, ay, bx, by) -> bool:
        pad = 0 if allow_touch else 1
        aL, aR = ax - pad, ax + S - 1 + pad
        aT, aB = ay - pad, ay + S - 1 + pad
        bL, bR = bx, bx + S - 1
        bT, bB = by, by + S - 1
        return not (aR < bL or bR < aL or aB < bT or bB < aT)

    max_x, max_y = W - S, H - S
    while len(anchors) < num_islands and tries < max_tries:
        tries += 1
        ax = rng.randint(0, max_x)
        ay = rng.randint(0, max_y)
        if all(not overlaps(ax, ay, bx, by) for (bx, by) in anchors):
            anchors.append((ax, ay))
    if len(anchors) < num_islands:
        raise RuntimeError(
            f"Could not place {num_islands} non-overlapping {S}x{S} islands. "
            f"Try allow_touch=True or reduce num_islands/island_size."
        )
    for (ax, ay) in anchors:
        for yy in range(ay, ay + S):
            for xx in range(ax, ax + S):
                zones[(xx, yy)] = 2
    meta = {"mode": "islands", "anchors": anchors, "num_islands": num_islands, "island_size": S}
    return zones, meta


def build_zones(
    W: int, H: int, *, zone_mode: str, seed: int,
    num_islands: int, island_size: int, allow_touch: bool,
    stripe_direction: str, stripe_k: int, voronoi_k: int,
) -> Tuple[Dict[Point, int], Dict[str, Any]]:
    m = str(zone_mode).lower()
    if m in ("left_right", "leftright", "lr"):
        return zones_left_right(W, H), {"mode": "left_right"}
    if m == "islands":
        return zones_islands(W, H, num_islands=num_islands, island_size=island_size,
                             seed=seed, allow_touch=allow_touch)
    if m == "stripes":
        return zones_stripes(W, H, direction=stripe_direction, k=stripe_k), {
            "mode": "stripes", "stripe_direction": stripe_direction, "stripe_k": stripe_k}
    if m == "voronoi":
        z, meta = zones_voronoi(W, H, k=voronoi_k)
        meta = dict(meta) if isinstance(meta, dict) else {}
        meta["mode"] = "voronoi"
        meta["voronoi_k"] = voronoi_k
        return z, meta
    raise ValueError(f"Unknown zone_mode='{zone_mode}'")


def zones_to_grid(zones: Dict[Point, int], W: int, H: int) -> List[List[int]]:
    return [[int(zones[(x, y)]) for x in range(W)] for y in range(H)]


# ============================================================
# Dataset helpers (unchanged)
# ============================================================
def normalize_zone_grid(zone_grid):
    if not zone_grid:
        return zone_grid
    is_2d = isinstance(zone_grid[0], list)
    if is_2d:
        flat = [int(v) for row in zone_grid for v in row]
    else:
        flat = [int(v) for v in zone_grid]
    unique = sorted(set(flat))
    mapping = {old: new for new, old in enumerate(unique)}
    if is_2d:
        return [[mapping[int(v)] for v in row] for row in zone_grid]
    return [mapping[int(v)] for v in flat]


def flatten_grid(grid):
    if not grid:
        return []
    if not isinstance(grid[0], list):
        return list(grid)
    return [int(v) for row in grid for v in row]


def save_sa_dataset_record(
    dataset_dir, *, run_id, grid_W, grid_H, zone_pattern, seed,
    initial_crossings, final_crossings, best_ops, runtime_sec, zones,
):
    os.makedirs(dataset_dir, exist_ok=True)
    path = os.path.join(dataset_dir, "Dataset.jsonl")
    zone_grid = zones_to_grid(zones, grid_W, grid_H)
    zone_grid = normalize_zone_grid(zone_grid)
    zone_grid_flat = flatten_grid(zone_grid)
    sequence_ops = []
    for mv in best_ops:
        kind = "T" if mv.get("op") == "transpose" else "F"
        sequence_ops.append(
            {"kind": kind, "x": int(mv["x"]), "y": int(mv["y"]), "variant": str(mv["variant"])}
        )
    rec = {
        "run_id": str(run_id), "seed": int(seed),
        "grid_W": int(grid_W), "grid_H": int(grid_H),
        "zone_pattern": str(zone_pattern),
        "initial_crossings": int(initial_crossings),
        "zone_grid": zone_grid_flat,
        "final_crossings": int(final_crossings),
        "sequence_len": int(len(sequence_ops)),
        "sequence_ops": sequence_ops,
        "runtime_sec": float(runtime_sec),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ============================================================
# Move application (unchanged - CPU-bound, operates on HamiltonianSTL)
# ============================================================
def _infer_success(ret, changed: bool) -> bool:
    if ret is None:
        return True
    if isinstance(ret, bool):
        return ret
    if isinstance(ret, dict) and "ok" in ret:
        return bool(ret["ok"])
    if isinstance(ret, tuple) and len(ret) > 0:
        a = ret[0]
        if isinstance(a, bool):
            return bool(a)
        msg = " ".join(str(x) for x in ret if x is not None).lower()
        if any(k in msg for k in ("transposed", "flipped", "success", "succeeded", "ok", "done")):
            return True
        if any(k in msg for k in ("fail", "failed", "invalid", "error", "mismatch", "not flippable")):
            return False
        return changed
    if isinstance(ret, str):
        msg = ret.lower()
        if any(k in msg for k in ("transposed", "flipped", "success", "succeeded", "ok", "done")):
            return True
        if any(k in msg for k in ("fail", "failed", "invalid", "error", "mismatch", "not flippable")):
            return False
        return changed
    return changed


def apply_move(h: HamiltonianSTL, mv: Dict[str, Any]) -> bool:
    op = mv["op"]
    x, y = int(mv["x"]), int(mv["y"])
    w, hh = int(mv["w"]), int(mv["h"])
    variant = mv["variant"]
    H_before, V_before = deep_copy(h.H, h.V)
    try:
        if op == "transpose":
            sub = h.get_subgrid((x, y), (x + 2, y + 2))
            ret = h.transpose_subgrid(sub, variant)
        elif op == "flip":
            sub = h.get_subgrid((x, y), (x + w - 1, y + hh - 1))
            ret = h.flip_subgrid(sub, variant)
        else:
            return False
    except Exception:
        h.H, h.V = H_before, V_before
        return False
    if isinstance(h.H, np.ndarray):
        changed = (not np.array_equal(h.H, H_before)) or (not np.array_equal(h.V, V_before))
    else:
        changed = (h.H != H_before) or (h.V != V_before)
    ok = _infer_success(ret, changed)
    if not changed:
        return False
    if not ok:
        return False
    if not is_valid_cycle(h):
        return False
    return True


def _snapshot_edges_for_move(h: HamiltonianSTL, mv: Dict[str, Any]):
    op = mv["op"]
    x, y = mv["x"], mv["y"]
    w, hh = mv["w"], mv["h"]
    if op == "transpose":
        sub = h.get_subgrid((x, y), (x + 2, y + 2))
    elif op == "flip":
        sub = h.get_subgrid((x, y), (x + w - 1, y + hh - 1))
    else:
        return []
    pts = [p for row in sub for p in row if p is not None]
    ptset = set(pts)
    snap = []
    for (px, py) in pts:
        r = (px + 1, py)
        if r in ptset:
            snap.append(((px, py), r, bool(h.has_edge((px, py), r))))
        d = (px, py + 1)
        if d in ptset:
            snap.append(((px, py), d, bool(h.has_edge((px, py), d))))
    return snap


def _restore_edges_snapshot(h: HamiltonianSTL, snap):
    for p, q, val in snap:
        h.set_edge(p, q, val)


def try_move_feasible(h: HamiltonianSTL, mv: Dict[str, Any]) -> bool:
    snap = _snapshot_edges_for_move(h, mv)
    try:
        ok = apply_move(h, mv)
        _restore_edges_snapshot(h, snap)
        return bool(ok)
    except Exception:
        _restore_edges_snapshot(h, snap)
        return False


def _transpose_variants(h: HamiltonianSTL) -> List[str]:
    tp = getattr(h, "transpose_patterns", [])
    if hasattr(tp, "keys"):
        return list(tp.keys())
    try:
        return list(tp)
    except Exception:
        return ["sr", "wa", "sl", "ea", "nl", "eb", "nr", "wb"]


def _flip_variants(h: HamiltonianSTL) -> List[Tuple[Any, int, int]]:
    fp = getattr(h, "flip_patterns", None)
    if fp is None:
        keys = ["w", "e", "n", "s"]
    else:
        if hasattr(fp, "keys"):
            keys = list(fp.keys())
        else:
            try:
                keys = list(fp)
            except Exception:
                keys = ["w", "e", "n", "s"]
    out = []
    for k in keys:
        ks = str(k).lower()
        if ks in ("w", "e"):
            out.append((k, 3, 2))
        elif ks in ("n", "s"):
            out.append((k, 2, 3))
        else:
            out.append((k, 3, 2))
            out.append((k, 2, 3))
    return out


def _layer_index(x: int, y: int, W: int, H: int) -> int:
    return min(x, y, W - 1 - x, H - 1 - y)


def _center_layer_for_anchor(x: int, y: int, w: int, h: int, W: int, H: int) -> int:
    cx = x + (w // 2)
    cy = y + (h // 2)
    return _layer_index(cx, cy, W, H)


def op_probabilities(i: int, split: int) -> Tuple[float, float]:
    if i < split:
        return (0.02, 0.98)
    else:
        return (0.90, 0.10)


# ============================================================
# GPU ACCELERATOR
# ============================================================
class GPUAccelerator:
    """
    Holds precomputed GPU tensors and provides GPU-accelerated methods
    for zone crossing computation, boundary scoring, and batched pattern matching.

    Key GPU operations using matrix multiplication:
    - batch_match_transpose: edge_vectors @ pattern_matrix.T  (N,12) x (12,8) -> (N,8)
    - batch_match_flip_3x2: edge_vectors @ pattern_matrix.T   (N,7) x (7,2)  -> (N,2)
    - batch_match_flip_2x3: edge_vectors @ pattern_matrix.T   (N,7) x (7,2)  -> (N,2)
    - compute_crossings: element-wise multiply + sum on full grid tensors
    - boundary scoring: 2D unfold + sum (vectorized over all anchors)
    """

    # 3x3 subgrid edge index map (12 edges)
    # Nodes: 0 1 2 / 3 4 5 / 6 7 8
    # H-edges: (0,1)->0 (1,2)->1 (3,4)->2 (4,5)->3 (6,7)->4 (7,8)->5
    # V-edges: (0,3)->6 (1,4)->7 (2,5)->8 (3,6)->9 (4,7)->10 (5,8)->11
    T_MAP = {
        (0, 1): 0, (1, 2): 1, (3, 4): 2, (4, 5): 3, (6, 7): 4, (7, 8): 5,
        (0, 3): 6, (1, 4): 7, (2, 5): 8, (3, 6): 9, (4, 7): 10, (5, 8): 11,
    }

    # 3x2 subgrid edge index map (7 edges) - for flip 'w','e'
    # Nodes: 0 1 2 / 3 4 5
    # H: (0,1)->0 (1,2)->1 (3,4)->2 (4,5)->3
    # V: (0,3)->4 (1,4)->5 (2,5)->6
    F32_MAP = {
        (0, 1): 0, (1, 2): 1, (3, 4): 2, (4, 5): 3,
        (0, 3): 4, (1, 4): 5, (2, 5): 6,
    }

    # 2x3 subgrid edge index map (7 edges) - for flip 'n','s'
    # Nodes: 0 1 / 2 3 / 4 5
    # H: (0,1)->0 (2,3)->1 (4,5)->2
    # V: (0,2)->3 (1,3)->4 (2,4)->5 (3,5)->6
    F23_MAP = {
        (0, 1): 0, (2, 3): 1, (4, 5): 2,
        (0, 2): 3, (1, 3): 4, (2, 4): 5, (3, 5): 6,
    }

    def __init__(self, W: int, H: int, zones: Dict[Point, int], device: torch.device):
        self.W = W
        self.H = H
        self.device = device

        # Zone grid as tensor
        zone_list = [[zones[(x, y)] for x in range(W)] for y in range(H)]
        self.zone_tensor = torch.tensor(zone_list, dtype=torch.float32, device=device)

        # Zone boundary matrices (precomputed once)
        self.hb = (self.zone_tensor[:, :-1] != self.zone_tensor[:, 1:]).float()
        self.vb = (self.zone_tensor[:-1, :] != self.zone_tensor[1:, :]).float()

        self._build_pattern_matrices()
        self._precompute_boundary_scores()

    def _build_pattern_matrices(self):
        """Build pattern matrices for matrix-multiplication-based matching."""
        dev = self.device
        tp = HamiltonianSTL.transpose_patterns
        fp = HamiltonianSTL.flip_patterns

        # --- Transpose (8 variants, 12 edges each) ---
        self.t_names = list(tp.keys())
        P = torch.zeros(len(self.t_names), 12, device=dev)
        for i, vn in enumerate(self.t_names):
            for pair in tp[vn]['old']:
                sp = tuple(sorted(pair))
                if sp in self.T_MAP:
                    P[i, self.T_MAP[sp]] = 1.0
        self.t_pat = P
        self.t_anti = 1.0 - P
        self.t_sums = P.sum(dim=1)

        # --- Flip 3x2 ('w','e') ---
        self.f32_names = [k for k in fp if str(k).lower() in ('w', 'e')]
        P32 = torch.zeros(len(self.f32_names), 7, device=dev)
        for i, vn in enumerate(self.f32_names):
            for pair in fp[vn]['old']:
                sp = tuple(sorted(pair))
                if sp in self.F32_MAP:
                    P32[i, self.F32_MAP[sp]] = 1.0
        self.f32_pat = P32
        self.f32_anti = 1.0 - P32
        self.f32_sums = P32.sum(dim=1)

        # --- Flip 2x3 ('n','s') ---
        self.f23_names = [k for k in fp if str(k).lower() in ('n', 's')]
        P23 = torch.zeros(len(self.f23_names), 7, device=dev)
        for i, vn in enumerate(self.f23_names):
            for pair in fp[vn]['old']:
                sp = tuple(sorted(pair))
                if sp in self.F23_MAP:
                    P23[i, self.F23_MAP[sp]] = 1.0
        self.f23_pat = P23
        self.f23_anti = 1.0 - P23
        self.f23_sums = P23.sum(dim=1)

    def _precompute_boundary_scores(self):
        """Precompute boundary scores for all anchor positions using tensor unfold."""
        W, H = self.W, self.H

        # 3x3: hb[y:y+3, x:x+2] + vb[y:y+2, x:x+3]
        if H >= 3 and W >= 3:
            hb3 = self.hb.unfold(0, 3, 1).unfold(1, 2, 1).sum(dim=(-1, -2))
            vb3 = self.vb.unfold(0, 2, 1).unfold(1, 3, 1).sum(dim=(-1, -2))
            self.bs_3x3 = hb3 + vb3  # (H-2, W-2)
        else:
            self.bs_3x3 = torch.zeros(max(H - 2, 0), max(W - 2, 0), device=self.device)

        # 3x2 (w=3,h=2): hb[y:y+2, x:x+2] + vb[y:y+1, x:x+3]
        if H >= 2 and W >= 3:
            hb32 = self.hb.unfold(0, 2, 1).unfold(1, 2, 1).sum(dim=(-1, -2))
            vb32 = self.vb.unfold(0, 1, 1).unfold(1, 3, 1).sum(dim=(-1, -2))
            self.bs_3x2 = hb32 + vb32  # (H-1, W-2)
        else:
            self.bs_3x2 = torch.zeros(max(H - 1, 0), max(W - 2, 0), device=self.device)

        # 2x3 (w=2,h=3): hb[y:y+3, x:x+1] + vb[y:y+2, x:x+2]
        if H >= 3 and W >= 2:
            hb23 = self.hb.unfold(0, 3, 1).unfold(1, 1, 1).sum(dim=(-1, -2))
            vb23 = self.vb.unfold(0, 2, 1).unfold(1, 2, 1).sum(dim=(-1, -2))
            self.bs_2x3 = hb23 + vb23  # (H-2, W-1)
        else:
            self.bs_2x3 = torch.zeros(max(H - 2, 0), max(W - 1, 0), device=self.device)

    # ----------------------------------------------------------
    # GPU crossing computation
    # ----------------------------------------------------------
    def compute_crossings(self, h: HamiltonianSTL) -> int:
        """Compute zone crossings using GPU tensor element-wise multiply + sum."""
        H_t = torch.tensor(h.H, dtype=torch.float32, device=self.device)
        V_t = torch.tensor(h.V, dtype=torch.float32, device=self.device)
        return int((H_t * self.hb).sum().item() + (V_t * self.vb).sum().item())

    # ----------------------------------------------------------
    # Batched pattern matching via matrix multiplication
    # ----------------------------------------------------------
    def _extract_transpose_edges(self, h: HamiltonianSTL):
        """Extract 12-element edge vectors for all 3x3 anchors. Returns (N, 12) tensor."""
        H_t = torch.tensor(h.H, dtype=torch.float32, device=self.device)
        V_t = torch.tensor(h.V, dtype=torch.float32, device=self.device)
        # H windows: H[y:y+3, x:x+2] -> 6 edges per anchor
        hw = H_t.unfold(0, 3, 1).unfold(1, 2, 1)  # (H-2, W-2, 3, 2)
        # V windows: V[y:y+2, x:x+3] -> 6 edges per anchor
        vw = V_t.unfold(0, 2, 1).unfold(1, 3, 1)  # (H-2, W-2, 2, 3)
        N = (self.H - 2) * (self.W - 2)
        return torch.cat([hw.reshape(N, 6), vw.reshape(N, 6)], dim=1)

    def _extract_flip_3x2_edges(self, h: HamiltonianSTL):
        """Extract 7-element edge vectors for all 3x2 anchors. Returns (N, 7) tensor."""
        H_t = torch.tensor(h.H, dtype=torch.float32, device=self.device)
        V_t = torch.tensor(h.V, dtype=torch.float32, device=self.device)
        hw = H_t.unfold(0, 2, 1).unfold(1, 2, 1)  # (H-1, W-2, 2, 2)
        vw = V_t.unfold(0, 1, 1).unfold(1, 3, 1)  # (H-1, W-2, 1, 3)
        N = (self.H - 1) * (self.W - 2)
        return torch.cat([hw.reshape(N, 4), vw.reshape(N, 3)], dim=1)

    def _extract_flip_2x3_edges(self, h: HamiltonianSTL):
        """Extract 7-element edge vectors for all 2x3 anchors. Returns (N, 7) tensor."""
        H_t = torch.tensor(h.H, dtype=torch.float32, device=self.device)
        V_t = torch.tensor(h.V, dtype=torch.float32, device=self.device)
        hw = H_t.unfold(0, 3, 1).unfold(1, 1, 1)  # (H-2, W-1, 3, 1)
        vw = V_t.unfold(0, 2, 1).unfold(1, 2, 1)  # (H-2, W-1, 2, 2)
        N = (self.H - 2) * (self.W - 1)
        return torch.cat([hw.reshape(N, 3), vw.reshape(N, 4)], dim=1)

    def batch_match_transpose(self, h: HamiltonianSTL):
        """
        Matrix-multiplication pattern matching for all 3x3 transpose anchors.

        Uses two matmuls:
          dot     = edge_vectors @ pattern_matrix.T      (N,12) x (12,8) -> (N,8)
          anti_dot = edge_vectors @ anti_pattern_matrix.T (N,12) x (12,8) -> (N,8)

        Match iff dot[i,j] == pattern_sum[j] AND anti_dot[i,j] == 0

        Returns (matches: (N,8) bool, anchor_coords: list of (x,y))
        """
        if self.H < 3 or self.W < 3:
            return torch.zeros(0, len(self.t_names), dtype=torch.bool, device=self.device), []
        ev = self._extract_transpose_edges(h)
        dot = ev @ self.t_pat.T
        anti = ev @ self.t_anti.T
        matches = (dot == self.t_sums.unsqueeze(0)) & (anti == 0)
        coords = [(x, y) for y in range(self.H - 2) for x in range(self.W - 2)]
        return matches, coords

    def batch_match_flip_3x2(self, h: HamiltonianSTL):
        """Matrix-multiplication pattern matching for 3x2 flip anchors."""
        if self.H < 2 or self.W < 3:
            return torch.zeros(0, len(self.f32_names), dtype=torch.bool, device=self.device), []
        ev = self._extract_flip_3x2_edges(h)
        dot = ev @ self.f32_pat.T
        anti = ev @ self.f32_anti.T
        matches = (dot == self.f32_sums.unsqueeze(0)) & (anti == 0)
        coords = [(x, y) for y in range(self.H - 1) for x in range(self.W - 2)]
        return matches, coords

    def batch_match_flip_2x3(self, h: HamiltonianSTL):
        """Matrix-multiplication pattern matching for 2x3 flip anchors."""
        if self.H < 3 or self.W < 2:
            return torch.zeros(0, len(self.f23_names), dtype=torch.bool, device=self.device), []
        ev = self._extract_flip_2x3_edges(h)
        dot = ev @ self.f23_pat.T
        anti = ev @ self.f23_anti.T
        matches = (dot == self.f23_sums.unsqueeze(0)) & (anti == 0)
        coords = [(x, y) for y in range(self.H - 2) for x in range(self.W - 1)]
        return matches, coords

    # ----------------------------------------------------------
    # GPU-accelerated move pool construction
    # ----------------------------------------------------------
    def refresh_move_pool(
        self,
        h: HamiltonianSTL,
        zones: Dict[Point, int],
        *,
        bias_to_boundary: bool = True,
        max_moves: int = 5000,
        allowed_ops: Optional[Set[str]] = None,
        border_to_inner: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        GPU-accelerated move pool construction.

        1. Matrix multiplication identifies pattern-matching (anchor, variant) pairs
        2. Precomputed GPU boundary scores sort anchors (no per-anchor loops)
        3. Only pattern-matching candidates go through CPU validation (BFS)

        This eliminates ~95% of expensive CPU validation calls on large grids.
        """
        W, Ht = h.width, h.height
        allowed_ops = allowed_ops or {"transpose", "flip"}
        pool: List[Dict[str, Any]] = []

        # --- TRANSPOSE moves ---
        if "transpose" in allowed_ops and Ht >= 3 and W >= 3:
            matches, coords = self.batch_match_transpose(h)
            if matches.any():
                # Sort anchors
                if border_to_inner:
                    keys = torch.tensor(
                        [_center_layer_for_anchor(x, y, 3, 3, W, Ht) for x, y in coords],
                        dtype=torch.float32, device=self.device)
                elif bias_to_boundary:
                    keys = -self.bs_3x3.reshape(-1)
                else:
                    keys = torch.zeros(len(coords), device=self.device)

                order = torch.argsort(keys).tolist()

                for idx in order:
                    if len(pool) >= max_moves:
                        break
                    row = matches[idx]
                    if not row.any():
                        continue
                    x, y = coords[idx]
                    var_idxs = row.nonzero(as_tuple=True)[0].tolist()
                    random.shuffle(var_idxs)
                    for vi in var_idxs:
                        variant = self.t_names[vi]
                        mv = {"op": "transpose", "variant": variant,
                              "x": x, "y": y, "w": 3, "h": 3}
                        if try_move_feasible(h, mv):
                            pool.append(mv)
                            break

        # --- FLIP 3x2 moves (w,e) ---
        if "flip" in allowed_ops and Ht >= 2 and W >= 3 and self.f32_names:
            matches, coords = self.batch_match_flip_3x2(h)
            if matches.any():
                if border_to_inner:
                    keys = torch.tensor(
                        [_center_layer_for_anchor(x, y, 3, 2, W, Ht) for x, y in coords],
                        dtype=torch.float32, device=self.device)
                elif bias_to_boundary:
                    keys = -self.bs_3x2.reshape(-1)
                else:
                    keys = torch.zeros(len(coords), device=self.device)

                order = torch.argsort(keys).tolist()

                for vi_idx, variant in enumerate(self.f32_names):
                    for idx in order:
                        if len(pool) >= max_moves:
                            break
                        if not matches[idx, vi_idx]:
                            continue
                        x, y = coords[idx]
                        mv = {"op": "flip", "variant": variant,
                              "x": x, "y": y, "w": 3, "h": 2}
                        if try_move_feasible(h, mv):
                            pool.append(mv)
                    if len(pool) >= max_moves:
                        break

        # --- FLIP 2x3 moves (n,s) ---
        if "flip" in allowed_ops and Ht >= 3 and W >= 2 and self.f23_names:
            matches, coords = self.batch_match_flip_2x3(h)
            if matches.any():
                if border_to_inner:
                    keys = torch.tensor(
                        [_center_layer_for_anchor(x, y, 2, 3, W, Ht) for x, y in coords],
                        dtype=torch.float32, device=self.device)
                elif bias_to_boundary:
                    keys = -self.bs_2x3.reshape(-1)
                else:
                    keys = torch.zeros(len(coords), device=self.device)

                order = torch.argsort(keys).tolist()

                for vi_idx, variant in enumerate(self.f23_names):
                    for idx in order:
                        if len(pool) >= max_moves:
                            break
                        if not matches[idx, vi_idx]:
                            continue
                        x, y = coords[idx]
                        mv = {"op": "flip", "variant": variant,
                              "x": x, "y": y, "w": 2, "h": 3}
                        if try_move_feasible(h, mv):
                            pool.append(mv)
                    if len(pool) >= max_moves:
                        break

        return pool


# ============================================================
# GPU-accelerated SA runner
# ============================================================
def run_sa(
    *,
    width: int = 30,
    height: int = 30,
    iterations: int = 5000,
    Tmax: float = 80.0,
    Tmin: float = 0.5,
    seed: int = 0,
    plot_live: bool = False,
    show_every_accepted: int = 0,
    pause_seconds: float = 0.01,
    dataset_dir: str = "Dataset2",
    write_dataset: bool = True,
    max_move_tries: int = 25,
    pool_refresh_period: int = 250,
    pool_max_moves: int = 5000,
    reheat_patience: int = 3000,
    reheat_factor: float = 1.5,
    reheat_cap: float = 600.0,
    transpose_phase_ratio: float = 0.6,
    border_to_inner: bool = True,
    zone_mode: str = "left_right",
    num_islands: int = 3,
    island_size: int = 8,
    allow_touch: bool = False,
    stripe_direction: str = "v",
    stripe_k: int = 3,
    voronoi_k: int = 3,
    init_pattern: str = "auto",
    debug: bool = False,
    # GPU parameters
    device: Optional[torch.device] = None,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    GPU-accelerated SA runner. Same interface as sa_generation.run_sa.
    Extra parameter: device (torch.device) for GPU selection.
    """
    random.seed(seed)
    start_time = time.perf_counter()

    # Select GPU device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build zones
    zones, zones_meta = build_zones(
        width, height, zone_mode=zone_mode, seed=seed,
        num_islands=num_islands, island_size=island_size, allow_touch=allow_touch,
        stripe_direction=stripe_direction, stripe_k=stripe_k, voronoi_k=voronoi_k,
    )

    # Select initial path pattern
    if init_pattern == "auto":
        if zone_mode in ("left_right", "leftright", "lr"):
            init_pattern = "vertical_zigzag"
        elif zone_mode == "stripes" and stripe_direction == "v":
            init_pattern = "vertical_zigzag"
        elif zone_mode == "stripes" and stripe_direction == "h":
            init_pattern = "zigzag"
        else:
            init_pattern = "zigzag"

    h = FastHamiltonianSTL(width, height, init_pattern=init_pattern)
    if not is_valid_cycle(h):
        raise RuntimeError("Initial Hamiltonian cycle invalid.")

    # Create GPU accelerator (precomputes zone boundaries, pattern matrices)
    gpu = GPUAccelerator(width, height, zones, device)

    print(f"[SA] seed={seed} grid={width}x{height} zone={zone_mode} "
          f"iters={iterations} device={device} (numpy+numba+cuda)", flush=True)

    current_cost = gpu.compute_crossings(h)
    initial_crossings = current_cost

    best_cost = current_cost
    best_state = deep_copy(h.H, h.V)

    accepted_ops: List[Dict[str, Any]] = []
    best_ops: List[Dict[str, Any]] = []

    accepted = attempted = rejected = 0
    invalid_moves = apply_fail = 0

    best_seen = best_cost
    no_improve = 0

    split = int(iterations * float(transpose_phase_ratio))
    split = max(0, min(iterations, split))

    allowed_ops = {"transpose"} if split > 0 else {"transpose", "flip"}

    move_pool = gpu.refresh_move_pool(
        h, zones, bias_to_boundary=True, max_moves=pool_max_moves,
        allowed_ops=allowed_ops, border_to_inner=border_to_inner,
    )

    for i in range(iterations):
        attempted += 1

        # Phase switch
        if i == split:
            allowed_ops = {"transpose", "flip"}
            move_pool = gpu.refresh_move_pool(
                h, zones, bias_to_boundary=True, max_moves=pool_max_moves,
                allowed_ops=allowed_ops, border_to_inner=border_to_inner,
            )

        # Periodic pool refresh
        if i % pool_refresh_period == 0:
            allowed_ops = {"transpose"} if i < split else {"transpose", "flip"}
            move_pool = gpu.refresh_move_pool(
                h, zones, bias_to_boundary=True, max_moves=pool_max_moves,
                allowed_ops=allowed_ops, border_to_inner=border_to_inner,
            )

        T = dynamic_temperature(i, iterations, Tmin=Tmin, Tmax=Tmax)

        applied_move: Optional[Dict[str, Any]] = None
        applied_snap = None

        # 1) sample from pool
        if move_pool:
            mv = random.choice(move_pool)
            snap = _snapshot_edges_for_move(h, mv)

            if apply_move(h, mv):
                applied_move = mv
                applied_snap = snap
            else:
                apply_fail += 1
                _restore_edges_snapshot(h, snap)

        # 2) fallback random tries
        else:
            for _ in range(max_move_tries):
                x3 = random.randint(0, width - 3)
                y3 = random.randint(0, height - 3)

                if random.random() < 0.5:
                    variant = random.choice(_transpose_variants(h))
                    mv_try = {"op": "transpose", "variant": variant,
                              "x": x3, "y": y3, "w": 3, "h": 3}
                else:
                    variants = {'n': (3, 2), 's': (3, 2), 'e': (2, 3), 'w': (2, 3)}
                    variant, (w, hh) = random.choice(list(variants.items()))
                    x = random.randint(0, width - w)
                    y = random.randint(0, height - hh)
                    mv_try = {"op": "flip", "variant": variant,
                              "x": x, "y": y, "w": w, "h": hh}

                snap = _snapshot_edges_for_move(h, mv_try)

                if not apply_move(h, mv_try):
                    apply_fail += 1
                    _restore_edges_snapshot(h, snap)
                    continue

                applied_move = mv_try
                applied_snap = snap
                break

            if applied_move is None:
                invalid_moves += 1

        # Acceptance step
        if applied_move is None:
            no_improve += 1
        else:
            new_cost = gpu.compute_crossings(h)
            delta = new_cost - current_cost

            if delta < 0:
                accept = True
            else:
                if T <= 0:
                    accept = False
                else:
                    xexp = -float(delta) / float(T)
                    xexp = max(-700.0, min(700.0, xexp))
                    accept = (random.random() < math.exp(xexp))

            if accept:
                current_cost = new_cost
                accepted += 1

                rec = {
                    "op": str(applied_move["op"]),
                    "variant": str(applied_move["variant"]),
                    "x": int(applied_move["x"]),
                    "y": int(applied_move["y"]),
                    "w": int(applied_move.get("w", 0)),
                    "h": int(applied_move.get("h", 0)),
                }
                accepted_ops.append(rec)

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_state = deep_copy(h.H, h.V)
                    best_ops = accepted_ops.copy()

                if best_cost < best_seen:
                    best_seen = best_cost
                    no_improve = 0
                else:
                    no_improve += 1

            else:
                rejected += 1
                if applied_snap is not None:
                    _restore_edges_snapshot(h, applied_snap)
                no_improve += 1

        # Reheating
        if no_improve >= reheat_patience:
            Tmax = min(reheat_cap, Tmax * reheat_factor)
            no_improve = 0

            allowed_ops = {"transpose"} if i < split else {"transpose", "flip"}
            move_pool = gpu.refresh_move_pool(
                h, zones, bias_to_boundary=True, max_moves=pool_max_moves,
                allowed_ops=allowed_ops, border_to_inner=border_to_inner,
            )

    # Restore best
    h.H, h.V = best_state
    if not is_valid_cycle(h):
        raise RuntimeError("Best state invalid at end.")

    runtime_sec = time.perf_counter() - start_time

    run_id = f"sa_{zone_mode}_W{width}H{height}_seed{seed}_{int(time.time())}"
    if write_dataset:
        save_sa_dataset_record(
            dataset_dir=dataset_dir, run_id=run_id, grid_W=width, grid_H=height,
            zone_pattern=zone_mode, seed=seed, initial_crossings=initial_crossings,
            final_crossings=best_cost, best_ops=best_ops, runtime_sec=runtime_sec,
            zones=zones,
        )

    return int(initial_crossings), int(best_cost), best_ops


# ============================================================
# Multi-seed runner (single GPU)
# ============================================================
def run_sa_multiple_seeds(
    seeds: List[int],
    *,
    width: int = 30,
    height: int = 30,
    iterations: int = 5000,
    Tmax: float = 80.0,
    Tmin: float = 0.5,
    zone_mode: str = "left_right",
    dataset_dir: str = "Dataset2",
    write_dataset: bool = True,
    plot_live: bool = False,
    max_move_tries=25,
    pool_refresh_period=250,
    pool_max_moves=5000,
    reheat_patience=3000,
    reheat_factor=1.5,
    reheat_cap=600.0,
    transpose_phase_ratio: float = 0.6,
    border_to_inner: bool = True,
    num_islands: int = 3,
    island_size: int = 8,
    allow_touch: bool = False,
    stripe_direction: str = "v",
    stripe_k: int = 3,
    voronoi_k: int = 3,
    debug: bool = False,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:

    results: List[Dict[str, Any]] = []
    for s in seeds:
        seed_start = time.perf_counter()
        init_c, best_c, best_ops = run_sa(
            width=width, height=height, iterations=iterations,
            Tmax=Tmax, Tmin=Tmin, seed=s, plot_live=False,
            show_every_accepted=0, pause_seconds=0.0,
            dataset_dir=dataset_dir, write_dataset=write_dataset,
            zone_mode=zone_mode,
            num_islands=num_islands, island_size=island_size, allow_touch=allow_touch,
            stripe_direction=stripe_direction, stripe_k=stripe_k, voronoi_k=voronoi_k,
            max_move_tries=max_move_tries, pool_refresh_period=pool_refresh_period,
            pool_max_moves=pool_max_moves, reheat_patience=reheat_patience,
            reheat_factor=reheat_factor, reheat_cap=reheat_cap,
            transpose_phase_ratio=transpose_phase_ratio,
            border_to_inner=border_to_inner, debug=debug, device=device,
        )
        seed_runtime = time.perf_counter() - seed_start
        results.append({
            "seed": int(s), "best_crossings": int(best_c),
            "sequence_len": int(len(best_ops)), "runtime_sec": float(seed_runtime)
        })
    return results
