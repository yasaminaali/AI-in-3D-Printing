"""
sa_generation.py - Simulated Annealing for Zone Crossing Minimization

This module implements a Simulated Annealing (SA) algorithm to optimize
Hamiltonian toolpaths by minimizing zone crossings. It generates training
data for deep learning models that predict optimal Flip/Transpose sequences.

The optimization process:
    1. Start with an initial Hamiltonian path (zigzag pattern)
    2. Build a pool of feasible moves (transpose/flip operations)
    3. Apply random moves and accept/reject based on Metropolis criterion
    4. Record successful operation sequences for training data

Output Format (JSONL - one record per SA run):
    {
        "run_id": "sa_left_right_W32H32_seed0_1234567890",
        "seed": 0,
        "grid_W": 32,
        "grid_H": 32,
        "zone_pattern": "left_right",
        "zone_grid": [0, 0, ..., 1, 1],  # Flattened, normalized to 0..K-1
        "initial_crossings": 32,
        "final_crossings": 4,
        "sequence_len": 156,
        "sequence_ops": [{"kind": "T", "x": 5, "y": 3, "variant": "sr"}, ...],
        "runtime_sec": 12.5
    }

SA Configurations:
    - short:      1,000 iterations, Tmax=60,  Tmin=0.5 (high-crossing trajectories)
    - medium:     3,000 iterations, Tmax=80,  Tmin=0.5 (balanced optimization)
    - long:       8,000 iterations, Tmax=100, Tmin=0.3 (near-optimal solutions)
    - extra_long: 15,000 iterations, Tmax=120, Tmin=0.2 (for large grids)

Supported Zone Modes:
    - "left_right": Simple vertical split at x = W/2
    - "islands": Background with square island regions (k=3, 8x8)
    - "stripes": Parallel vertical/horizontal bands (k=3)
    - "voronoi": Irregular Voronoi-based regions (k=3)

Dependencies:
    - operations.py: HamiltonianSTL class
    - Zones.py: zones_stripes, zones_voronoi functions

Usage:
    from sa_generation import run_sa, run_sa_multiple_seeds

    # Single run
    best_crossings, ops = run_sa(width=32, height=32, iterations=3000)

    # Multiple seeds for dataset generation
    results = run_sa_multiple_seeds(
        seeds=list(range(100)),
        width=32, height=32,
        iterations=3000,
        zone_mode="left_right"
    )

Author: AI-in-3D-Printing Team
"""
# ============================================================

import os
import json
import time
import math
import random
from typing import Tuple, Optional, Dict, Any, List, Set

import matplotlib.pyplot as plt

from operations import HamiltonianSTL
from zones import zones_stripes, zones_voronoi

Point = Tuple[int, int]


# ============================================================
# Temperature schedule
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
# Grid copy helpers
# ============================================================
def deep_copy(H, V):
    return ([row[:] for row in H], [row[:] for row in V])


def is_valid_cycle(h: HamiltonianSTL) -> bool:
    if hasattr(h, "validate_full_path_cycle"):
        return bool(h.validate_full_path_cycle())
    if hasattr(h, "validate_full_path"):
        return bool(h.validate_full_path())
    return True


# ============================================================
# Zone generators
# ============================================================
def zones_left_right(W: int, H: int) -> Dict[Point, int]:
    return {(x, y): (1 if x < W // 2 else 2) for y in range(H) for x in range(W)}


def zones_islands(
    W: int,
    H: int,
    *,
    num_islands: int,
    island_size: int,
    seed: int,
    allow_touch: bool,
) -> Tuple[Dict[Point, int], Dict[str, Any]]:
    """
    Background zone=1, islands zone=2 (square islands).
    """
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

    max_x = W - S
    max_y = H - S

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
    W: int,
    H: int,
    *,
    zone_mode: str,
    seed: int,
    # islands
    num_islands: int,
    island_size: int,
    allow_touch: bool,
    # stripes
    stripe_direction: str,
    stripe_k: int,
    # voronoi
    voronoi_k: int,
) -> Tuple[Dict[Point, int], Dict[str, Any]]:
    m = str(zone_mode).lower()

    if m in ("left_right", "leftright", "lr"):
        return zones_left_right(W, H), {"mode": "left_right"}

    if m == "islands":
        z, meta = zones_islands(
            W,
            H,
            num_islands=num_islands,
            island_size=island_size,
            seed=seed,
            allow_touch=allow_touch,
        )
        return z, meta

    if m == "stripes":
        return zones_stripes(W, H, direction=stripe_direction, k=stripe_k), {
            "mode": "stripes",
            "stripe_direction": stripe_direction,
            "stripe_k": stripe_k,
        }

    if m == "voronoi":
        z, meta = zones_voronoi(W, H, k=voronoi_k)
        meta = dict(meta) if isinstance(meta, dict) else {}
        meta["mode"] = "voronoi"
        meta["voronoi_k"] = voronoi_k
        return z, meta

    raise ValueError(f"Unknown zone_mode='{zone_mode}'")

def zones_to_grid(zones: Dict[Point, int], W: int, H: int) -> List[List[int]]:
    """
    Convert zones dict {(x,y)->zone_id} into a dense HxW grid:
      zone_grid[y][x] = zone_id
    """
    return [[int(zones[(x, y)]) for x in range(W)] for y in range(H)]


# ============================================================
# Crossings
# ============================================================
def compute_crossings(h: HamiltonianSTL, zones: Dict[Point, int]) -> int:
    W, H = h.width, h.height
    c = 0

    for y in range(H):
        for x in range(W - 1):
            if h.H[y][x]:
                if zones[(x, y)] != zones[(x + 1, y)]:
                    c += 1

    for y in range(H - 1):
        for x in range(W):
            if h.V[y][x]:
                if zones[(x, y)] != zones[(x, y + 1)]:
                    c += 1

    return c


# ============================================================
# Move application
# ============================================================
def _infer_success(ret, changed: bool) -> bool:
    """
    Infer whether an op succeeded from different return styles.
    """
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
            # transpose 3x3
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

    # snapshot edges inside this subgrid only
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

    out: List[Tuple[Any, int, int]] = []
    for k in keys:
        ks = str(k).lower()
        if ks in ("w", "e"):
            out.append((k, 3, 2)) 
        elif ks in ("n", "s"):
            out.append((k, 2, 3)) 
        else:
            # unknown key: try both
            out.append((k, 3, 2))
            out.append((k, 2, 3))
    return out


# Border->inner ordering helpers
def _layer_index(x: int, y: int, W: int, H: int) -> int:
    return min(x, y, W - 1 - x, H - 1 - y)


def _center_layer_for_anchor(x: int, y: int, w: int, h: int, W: int, H: int) -> int:
    cx = x + (w // 2)
    cy = y + (h // 2)
    return _layer_index(cx, cy, W, H)


# ============================================================
# Move pool (zone-boundary scoring + border->inner)
# ============================================================
def _zone_boundary_count_in_rect(
    W: int,
    H: int,
    zones: Dict[Point, int],
    x0: int,
    y0: int,
    w: int,
    h: int,
) -> int:
    x1 = x0 + w - 1
    y1 = y0 + h - 1
    s = 0

    for y in range(y0, y1 + 1):
        for x in range(x0, x1):
            if zones[(x, y)] != zones[(x + 1, y)]:
                s += 1

    for y in range(y0, y1):
        for x in range(x0, x1 + 1):
            if zones[(x, y)] != zones[(x, y + 1)]:
                s += 1

    return s


def _boundary_score_for_anchor(
    W: int,
    H: int,
    zones: Dict[Point, int],
    x: int,
    y: int,
    w: int,
    h: int,
) -> int:
    return _zone_boundary_count_in_rect(W, H, zones, x, y, w, h)


def refresh_move_pool(
    h: HamiltonianSTL,
    zones: Dict[Point, int],
    *,
    bias_to_boundary: bool = True,
    max_moves: int = 5000,
    allowed_ops: Optional[Set[str]] = None,
    border_to_inner: bool = False,
) -> List[Dict[str, Any]]:
    
    W, Ht = h.width, h.height
    allowed_ops = allowed_ops or {"transpose", "flip"}

    pool: List[Dict[str, Any]] = []

    # transpose anchors
    anchors_3x3 = [(x, y) for y in range(0, Ht - 2) for x in range(0, W - 2)]
    if border_to_inner:
        anchors_3x3.sort(key=lambda p: _center_layer_for_anchor(p[0], p[1], 3, 3, W, Ht))
    elif bias_to_boundary:
        anchors_3x3.sort(key=lambda p: -_boundary_score_for_anchor(W, Ht, zones, p[0], p[1], 3, 3))

    fvars = _flip_variants(h)
    tvars = _transpose_variants(h)

    # build transpose moves
    if "transpose" in allowed_ops:
        for (x, y) in anchors_3x3:
            random.shuffle(tvars)
            for variant in tvars:
                mv = {"op": "transpose", "variant": variant, "x": x, "y": y, "w": 3, "h": 3}
                if try_move_feasible(h, mv):
                    pool.append(mv)
                    break
            if len(pool) >= max_moves:
                return pool

    # build flip moves
    if "flip" in allowed_ops:
        for (variant, w, hh) in fvars:
            anchors = [(x, y) for y in range(0, Ht - hh + 1) for x in range(0, W - w + 1)]
            if border_to_inner:
                anchors.sort(key=lambda p: _center_layer_for_anchor(p[0], p[1], w, hh, W, Ht))
            elif bias_to_boundary:
                anchors.sort(key=lambda p: -_boundary_score_for_anchor(W, Ht, zones, p[0], p[1], w, hh))

            for (x, y) in anchors:
                mv = {"op": "flip", "variant": variant, "x": x, "y": y, "w": w, "h": hh}
                if try_move_feasible(h, mv):
                    pool.append(mv)
                if len(pool) >= max_moves:
                    return pool

    return pool


# ============================================================
# Dataset writer (JSONL)
# ============================================================

def normalize_zone_grid(zone_grid):
    """
    Normalize zone labels to 0..K-1.

    """
    if not zone_grid:
        return zone_grid

    # Detect 2D grid
    is_2d = isinstance(zone_grid[0], list)

    if is_2d:
        flat = [int(v) for row in zone_grid for v in row]
    else:
        flat = [int(v) for v in zone_grid]

    unique = sorted(set(flat))
    mapping = {old: new for new, old in enumerate(unique)}

    if is_2d:
        out = [[mapping[int(v)] for v in row] for row in zone_grid]
    else:
        out = [mapping[int(v)] for v in zone_grid]

    return out

def flatten_grid(grid):
    """
    Flatten a 2D grid into a 1D list.

    """
    if not grid:
        return []

    # If already flat, return as-is
    if not isinstance(grid[0], list):
        return list(grid)

    return [int(v) for row in grid for v in row]


def save_sa_dataset_record(
    dataset_dir: str,
    *,
    run_id: str,
    grid_W: int,
    grid_H: int,
    zone_pattern: str,
    seed: int,
    initial_crossings: int,
    final_crossings: int,
    best_ops: List[Dict[str, Any]],
    runtime_sec: float,
    zones: Dict[Point, int],
):
    os.makedirs(dataset_dir, exist_ok=True)
    path = os.path.join(dataset_dir, "Dataset.jsonl")

    # --- zones → grid ---
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
        "run_id": str(run_id),
        "seed": int(seed),
        "grid_W": int(grid_W),
        "grid_H": int(grid_H),
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

    # print(f"[Dataset] appended 1 record -> {path}")


# ============================================================
# Visualization
# ============================================================
def plot_cycle(
    h: HamiltonianSTL,
    zones: Dict[Point, int],
    title: str,
    save_path: Optional[str] = None,
):
    W, H = h.width, h.height
    plt.figure()

    zone_vals = sorted(set(zones.values()))
    a = zone_vals[0] if zone_vals else 0

    for y in range(H):
        for x in range(W):
            z = zones[(x, y)]
            color = "lightblue" if z == a else "lightgreen"
            plt.gca().add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=color))

    for y in range(H):
        for x in range(W - 1):
            if h.H[y][x]:
                col = "red" if zones[(x, y)] != zones[(x + 1, y)] else "black"
                plt.plot([x, x + 1], [y, y], color=col, linewidth=2)

    for y in range(H - 1):
        for x in range(W):
            if h.V[y][x]:
                col = "red" if zones[(x, y)] != zones[(x, y + 1)] else "black"
                plt.plot([x, x], [y, y + 1], color=col, linewidth=2)

    plt.title(title)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.grid(False)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")


# ============================================================
# Debug helpers
# ============================================================
def debug_print_pattern_keys(h: HamiltonianSTL) -> None:
    fp = getattr(h, "flip_patterns", None)
    tp = getattr(h, "transpose_patterns", None)

    if fp is None:
        fp_keys = None
    elif hasattr(fp, "keys"):
        fp_keys = list(fp.keys())
    else:
        try:
            fp_keys = list(fp)
        except Exception:
            fp_keys = fp

    if tp is None:
        tp_keys = None
    elif hasattr(tp, "keys"):
        tp_keys = list(tp.keys())
    else:
        try:
            tp_keys = list(tp)
        except Exception:
            tp_keys = tp

    #print("[Debug] flip_patterns keys:", fp_keys)
    #print("[Debug] transpose_patterns keys:", tp_keys)


def debug_try_one_flip(h: HamiltonianSTL) -> Dict[str, Any]:
    fv = _flip_variants(h)
    variant, w, hh = random.choice(fv)
    x = random.randint(0, h.width - w)
    y = random.randint(0, h.height - hh)
    sub = h.get_subgrid((x, y), (x + w - 1, y + hh - 1))
    ret = h.flip_subgrid(sub, variant)
    return {"variant": variant, "w": w, "h": hh, "x": x, "y": y, "ret": ret}


def estimate_flip_feasibility(h: HamiltonianSTL, trials: int = 2000) -> int:
    ok = 0
    fv = _flip_variants(h)
    if not fv:
        return 0

    for _ in range(trials):
        variant, w, hh = random.choice(fv)
        x = random.randint(0, h.width - w)
        y = random.randint(0, h.height - hh)
        mv = {"op": "flip", "variant": variant, "x": x, "y": y, "w": w, "h": hh}
        if try_move_feasible(h, mv):
            ok += 1
    return ok


# ============================================================
# Op-choice policy (keeps transpose with low probability)
# ============================================================
def op_probabilities(i: int, split: int) -> Tuple[float, float]:
    """
    Returns (p_flip, p_transpose)
    Phase 1: almost all transpose (flips often don't exist early).
    Phase 2: strongly prefer flip, but keep transpose too.
    """
    if i < split:
        return (0.02, 0.98)
    else:
        return (0.90, 0.10)


def choose_move_from_pool(move_pool: List[Dict[str, Any]], p_flip: float) -> Optional[Dict[str, Any]]:
    if not move_pool:
        return None
    flips = [m for m in move_pool if m.get("op") == "flip"]
    trans = [m for m in move_pool if m.get("op") == "transpose"]

    r = random.random()
    if r < p_flip and flips:
        return random.choice(flips)
    if trans:
        return random.choice(trans)
    if flips:
        return random.choice(flips)
    return random.choice(move_pool)


# ============================================================
# SA runner
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
    # pool / attempts
    max_move_tries: int = 25,
    pool_refresh_period: int = 250,
    pool_max_moves: int = 5000,
    # reheating
    reheat_patience: int = 3000,
    reheat_factor: float = 1.5,
    reheat_cap: float = 600.0,
    # phases
    transpose_phase_ratio: float = 0.6,
    border_to_inner: bool = True,
    # ZONES
    zone_mode: str = "left_right",
    # islands
    num_islands: int = 3,
    island_size: int = 8,
    allow_touch: bool = False,
    # stripes
    stripe_direction: str = "v",
    stripe_k: int = 3,
    # voronoi
    voronoi_k: int = 3,
    # Initial path pattern: 'auto', 'zigzag', 'vertical_zigzag', 'fermat_spiral'
    # 'auto' selects optimal pattern based on zone_mode
    init_pattern: str = "auto",
    # DEBUG
    debug: bool = False,
) -> Tuple[int, int, List[Dict[str, Any]]]:

    random.seed(seed)
    start_time = time.perf_counter()

    if plot_live:
        plt.ion()

    # Build zones first to determine optimal initial path
    zones, zones_meta = build_zones(
        width,
        height,
        zone_mode=zone_mode,
        seed=seed,
        num_islands=num_islands,
        island_size=island_size,
        allow_touch=allow_touch,
        stripe_direction=stripe_direction,
        stripe_k=stripe_k,
        voronoi_k=voronoi_k,
    )

    # Select optimal initial path based on zone pattern
    # Vertical zones -> vertical zigzag, Horizontal zones -> horizontal zigzag
    if init_pattern == "auto":
        if zone_mode in ("left_right", "leftright", "lr"):
            init_pattern = "vertical_zigzag"
        elif zone_mode == "stripes" and stripe_direction == "v":
            init_pattern = "vertical_zigzag"
        elif zone_mode == "stripes" and stripe_direction == "h":
            init_pattern = "zigzag"
        else:
            # For voronoi, islands, etc. - use default horizontal zigzag
            init_pattern = "zigzag"

    h = HamiltonianSTL(width, height, init_pattern=init_pattern)
    if not is_valid_cycle(h):
        raise RuntimeError("Initial Hamiltonian cycle invalid. Check HamiltonianSTL initialization.")

    if debug:
        debug_print_pattern_keys(h)

    current_cost = compute_crossings(h, zones)
    initial_crossings = current_cost

    best_cost = current_cost
    best_state = deep_copy(h.H, h.V)

    accepted_ops: List[Dict[str, Any]] = []
    best_ops: List[Dict[str, Any]] = []

    accepted = attempted = rejected = 0
    invalid_moves = apply_fail = 0

    best_seen = best_cost
    no_improve = 0

    # print(f"Initial crossings: {current_cost}")
    # print(f"Zone mode: {zone_mode} | meta={zones_meta}")
    if debug:
        c0 = estimate_flip_feasibility(h, trials=2000)
        #print(f"[Flip probe @start] feasible flips in 2000 trials = {c0}")

    split = int(iterations * float(transpose_phase_ratio))
    split = max(0, min(iterations, split))

    # Phase 1 pool: transpose-only (fast + flips usually absent)
    allowed_ops = {"transpose"} if split > 0 else {"transpose", "flip"}

    move_pool = refresh_move_pool(
        h,
        zones,
        bias_to_boundary=True,
        max_moves=pool_max_moves,
        allowed_ops=allowed_ops,
        border_to_inner=border_to_inner,
    )
    # [Pool] initial size = {len(move_pool)} | allowed={allowed_ops}

    if plot_live:
        plot_cycle(h, zones, title=f"Initial | crossings={current_cost}")
        plt.pause(pause_seconds)

    for i in range(iterations):
        attempted += 1

        # Phase switch: once we enter phase 2, build BOTH ops in the pool,
        # but choose flips with high probability (transpose still allowed).
        if i == split:
            allowed_ops = {"transpose", "flip"}
            move_pool = refresh_move_pool(
                h,
                zones,
                bias_to_boundary=True,
                max_moves=pool_max_moves,
                allowed_ops=allowed_ops,
                border_to_inner=border_to_inner,
            )
            flips = sum(1 for m in move_pool if m.get("op") == "flip")
            trans = sum(1 for m in move_pool if m.get("op") == "transpose")
            #print(f"[Phase switch] iter={i} pool={len(move_pool)} flips={flips} trans={trans} allowed={allowed_ops}")
            if debug:
                count = estimate_flip_feasibility(h, trials=2000)
                #print(f"[Flip probe @phase2] feasible flips in 2000 random trials = {count}")

        # Periodic pool refresh
        if i % pool_refresh_period == 0:
            # keep transpose-only pool in phase 1, both ops in phase 2
            allowed_ops = {"transpose"} if i < split else {"transpose", "flip"}
            move_pool = refresh_move_pool(
                h,
                zones,
                bias_to_boundary=True,  # biases flips toward zone boundary too
                max_moves=pool_max_moves,
                allowed_ops=allowed_ops,
                border_to_inner=border_to_inner,
            )
            if i % (pool_refresh_period * 10) == 0:
                flips = sum(1 for m in move_pool if m.get("op") == "flip")
                trans = sum(1 for m in move_pool if m.get("op") == "transpose")
                # [Pool] iter={i} size={len(move_pool)} flips={flips} trans={trans} allowed={allowed_ops}

        # Optional diagnostics
        """if debug and i >= split and (not move_pool) and (i % 500 == 0):
            count = estimate_flip_feasibility(h, trials=2000)
            #print(f"[Flip probe @empty] feasible flips found in 2000 random trials = {count}")
            for _ in range(3):
                print("[Flip debug]", debug_try_one_flip(h))"""

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
                    mv_try = {"op": "transpose", "variant": variant, "x": x3, "y": y3, "w": 3, "h": 3}
                else:
                    variants = {'n': (3, 2), 's': (3, 2), 'e': (2, 3), 'w': (2, 3)}
                    variant, (w, hh) = random.choice(list(variants.items()))
                    x = random.randint(0, width - w)
                    y = random.randint(0, height - hh)
                    mv_try = {"op": "flip", "variant": variant, "x": x, "y": y, "w": w, "h": hh}

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

        # acceptance step
        if applied_move is None:
            no_improve += 1
        else:
            new_cost = compute_crossings(h, zones)
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

                if plot_live and show_every_accepted > 0 and (accepted % show_every_accepted == 0):
                    plot_cycle(h, zones, title=f"Iter {i} | T={T:.3f} | cost={current_cost} | best={best_cost}")
                    plt.pause(pause_seconds)

                if best_cost < best_seen:
                    best_seen = best_cost
                    no_improve = 0
                else:
                    no_improve += 1

            else:
                rejected += 1
                # restore only local edges for this move
                if applied_snap is not None:
                    _restore_edges_snapshot(h, applied_snap)
                no_improve += 1

        # reheating
        if no_improve >= reheat_patience:
            Tmax = min(reheat_cap, Tmax * reheat_factor)
            no_improve = 0

            allowed_ops = {"transpose"} if i < split else {"transpose", "flip"}
            move_pool = refresh_move_pool(
                h,
                zones,
                bias_to_boundary=True,
                max_moves=pool_max_moves,
                allowed_ops=allowed_ops,
                border_to_inner=border_to_inner,
            )
            flips = sum(1 for m in move_pool if m.get("op") == "flip")
            trans = sum(1 for m in move_pool if m.get("op") == "transpose")
            # print(f"[Reheat] iter={i} Tmax={Tmax:.2f} pool={len(move_pool)} flips={flips} trans={trans} allowed={allowed_ops}")

        if i % 500 == 0:
            flips = sum(1 for m in move_pool if m.get("op") == "flip")
            trans = sum(1 for m in move_pool if m.get("op") == "transpose")
            # print(
            #     f"Iter {i}: T={T:.3f}, Tmax={Tmax:.2f}, Cost={current_cost}, Best={best_cost}, "
            #     f"Accepted={accepted}/{attempted}, Rejected={rejected}, "
            #     f"Invalid={invalid_moves}, ApplyFail={apply_fail}, "
            #     f"Pool={len(move_pool)} (F={flips},T={trans}), NoImprove={no_improve}"
            # )


    # restore best
    h.H, h.V = best_state
    if not is_valid_cycle(h):
        raise RuntimeError("Best state invalid at end (should not happen).")

    runtime_sec = time.perf_counter() - start_time
    # print(f"Final best crossings: {best_cost}")
    # print(f"[SA] seed={seed} runtime = {runtime_sec:.2f} seconds")

    """print("\nBest operation sequence (best_ops):")
    for k, mv in enumerate(best_ops):
        print(
            f"{k:04d} | {mv.get('op')} | var={mv.get('variant')} | "
            f"x={mv.get('x')} y={mv.get('y')} w={mv.get('w')} h={mv.get('h')}"
        )"""
    # print("")

    run_id = f"sa_{zone_mode}_W{width}H{height}_seed{seed}_{int(time.time())}"
    if write_dataset:
        save_sa_dataset_record(
            dataset_dir=dataset_dir,
            run_id=run_id,
            grid_W=width,
            grid_H=height,
            zone_pattern=zone_mode,
            seed=seed,
            initial_crossings=initial_crossings,
            final_crossings=best_cost,
            best_ops=best_ops,
            runtime_sec=runtime_sec,
            zones=zones,
        )

    if plot_live:
        plot_cycle(h, zones, title=f"FINAL SA BEST | crossings={best_cost}")
        plt.ioff()
        plt.show()

    return int(initial_crossings), int(best_cost), best_ops


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
) -> List[Dict[str, Any]]:

    results: List[Dict[str, Any]] = []
    for s in seeds:
        seed_start = time.perf_counter()
        init_c, best_c, best_ops = run_sa(
            width=width,
            height=height,
            iterations=iterations,
            Tmax=Tmax,
            Tmin=Tmin,
            seed=s,
            plot_live=plot_live,
            show_every_accepted=0,
            pause_seconds=0.0,
            dataset_dir=dataset_dir,
            write_dataset=write_dataset,
            zone_mode=zone_mode,
            num_islands=num_islands,
            island_size=island_size,
            allow_touch=allow_touch,
            stripe_direction=stripe_direction,
            stripe_k=stripe_k,
            voronoi_k=voronoi_k,
            max_move_tries=max_move_tries,
            pool_refresh_period=pool_refresh_period,
            pool_max_moves=pool_max_moves,
            reheat_patience=reheat_patience,
            reheat_factor=reheat_factor,
            reheat_cap=reheat_cap,
            transpose_phase_ratio=transpose_phase_ratio,
            border_to_inner=border_to_inner,
            debug=debug,
        )
        seed_runtime = time.perf_counter() - seed_start
        results.append(
            {"seed": int(s), "best_crossings": int(best_c), "sequence_len": int(len(best_ops)), "runtime_sec": float(seed_runtime)}
        )
        # print(f"[SA multi] seed={s} best_crossings={best_c} sequence_len={len(best_ops)} runtime={seed_runtime:.2f}s")

    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    seeds = list(range(30))
    run_sa_multiple_seeds(
        seeds,

        # grid
        width=32,
        height=32,

        # SA schedule
        iterations=3000,
        Tmax=80.0,
        Tmin=0.5,

        # pool / attempts
        max_move_tries=200,
        pool_refresh_period=250,
        pool_max_moves=5000,

        # reheating
        reheat_patience=1500,
        reheat_factor=1.5,
        reheat_cap=600.0,

        transpose_phase_ratio=0.6,
        border_to_inner=True,

        # zone pattern
        zone_mode="left_right",   # "left_right" | "islands" | "stripes" | "voronoi"

        # dataset
        dataset_dir="Dataset",
        write_dataset=True,

        # visualization
        plot_live=False,

        # islands params (used only if zone_mode="islands")
        num_islands=3,
        island_size=8,
        allow_touch=False,

        # stripes params (used only if zone_mode="stripes")
        stripe_direction="v",
        stripe_k=3,

        # voronoi params (used only if zone_mode="voronoi")
        voronoi_k=3,

        debug=False,
    )