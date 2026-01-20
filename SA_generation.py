# SA_generation.py
# ============================================================
# Simulated Annealing (SA) for ALL zone patterns + Dataset generation.
#
# Per SA run (one JSONL record):
#   grid_W, grid_H, zone_pattern, initial_crossings, final_crossings,
#   sequence_len, applied_count, seed, run_id,
#   sequence_ops = ordered list of applied ops (kind, x, y, variant)
#
# Supports zone_mode:
#   "left_right" | "islands" | "stripes" | "voronoi"
#
# Requires:
#   - operations.py  (HamiltonianSTL with get_subgrid, transpose_subgrid, flip_subgrid)
# If using stripes/voronoi:
#   - Zones.py with: zones_stripes, zones_voronoi
# ============================================================

import os
import json
import time
import math
import random
from typing import Tuple, Optional, Dict, Any, List

import matplotlib.pyplot as plt

from operations import HamiltonianSTL
from Zones import zones_stripes, zones_voronoi

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
            W, H,
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
def _msg_from_result(ret) -> str:
    if isinstance(ret, tuple) and len(ret) >= 2:
        return str(ret[1])
    return str(ret)


def apply_move(h: HamiltonianSTL, mv: Dict[str, Any]) -> bool:
    op = mv["op"]
    x, y, w, hh = mv["x"], mv["y"], mv["w"], mv["h"]
    variant = mv["variant"]

    if op == "transpose":
        sub = h.get_subgrid((x, y), (x + 2, y + 2))
        ret = h.transpose_subgrid(sub, variant)
        ok = _msg_from_result(ret).startswith("transposed")
    elif op == "flip":
        sub = h.get_subgrid((x, y), (x + w - 1, y + hh - 1))
        ret = h.flip_subgrid(sub, variant)
        ok = _msg_from_result(ret).startswith("flipped")
    else:
        return False

    if not ok:
        return False
    if not is_valid_cycle(h):
        return False
    return True


def try_move_feasible(h: HamiltonianSTL, mv: Dict[str, Any]) -> bool:
    H0, V0 = deep_copy(h.H, h.V)
    try:
        ok = apply_move(h, mv)
        h.H, h.V = H0, V0
        return bool(ok)
    except Exception:
        h.H, h.V = H0, V0
        return False


def _transpose_variants(h: HamiltonianSTL) -> List[str]:
    tp = getattr(h, "transpose_patterns", [])
    if hasattr(tp, "keys"):
        return list(tp.keys())
    try:
        return list(tp)
    except Exception:
        return ["a", "b", "c", "d", "e", "f", "g", "h"]


# ============================================================
# Move pool (boundary scoring)
# ============================================================
def _boundary_score_from_zones(W: int, H: int, zones: Dict[Point, int], x: int) -> int:
    if x < 0 or x >= W - 1:
        return 10**9
    s = 0
    for y in range(H):
        if zones[(x, y)] != zones[(x + 1, y)]:
            s += 1
    return s


def refresh_move_pool(
    h: HamiltonianSTL,
    zones: Dict[Point, int],
    *,
    bias_to_boundary: bool = True,
    max_moves: int = 5000,
) -> List[Dict[str, Any]]:
    
    W, Ht = h.width, h.height
    pool: List[Dict[str, Any]] = []

    xs = list(range(W))
    if bias_to_boundary:
        xs.sort(key=lambda xx: -_boundary_score_from_zones(W, Ht, zones, min(xx, W - 2)))

    tvars = _transpose_variants(h)
    fvars = [('n', 3, 2), ('s', 3, 2), ('e', 2, 3), ('w', 2, 3)]

    # Transpose scan
    for y in range(0, Ht - 2):
        for x in xs:
            if x > W - 3:
                continue

            random.shuffle(tvars)
            for variant in tvars:
                mv = {"op": "transpose", "variant": variant, "x": x, "y": y, "w": 3, "h": 3}
                if try_move_feasible(h, mv):
                    pool.append(mv)
                    break

            if len(pool) >= max_moves:
                return pool

    # Flip scan
    for (variant, w, hh) in fvars:
        for y in range(0, Ht - hh + 1):
            for x in xs:
                if x > W - w:
                    continue

                mv = {"op": "flip", "variant": variant, "x": x, "y": y, "w": w, "h": hh}
                if try_move_feasible(h, mv):
                    pool.append(mv)

                if len(pool) >= max_moves:
                    return pool

    return pool


# ============================================================
# Dataset writer (JSONL)
# ============================================================
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
):
    
    os.makedirs(dataset_dir, exist_ok=True)
    path = os.path.join(dataset_dir, "Dataset.jsonl")

    sequence_ops = []
    for mv in best_ops:
        kind = "T" if mv.get("op") == "transpose" else "F"
        sequence_ops.append({
            "kind": kind,
            "x": int(mv["x"]),
            "y": int(mv["y"]),
            "variant": str(mv["variant"]),
        })

    rec = {
        "run_id": str(run_id),
        "seed": int(seed),
        "grid_W": int(grid_W),
        "grid_H": int(grid_H),
        "zone_pattern": str(zone_pattern),
        "initial_crossings": int(initial_crossings),
        "final_crossings": int(final_crossings),
        "sequence_len": int(len(sequence_ops)),
        "applied_count": int(len(sequence_ops)),
        "sequence_ops": sequence_ops,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    print(f"[Dataset] appended 1 record -> {path}")


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
    dataset_dir: str = "Dataset",
    write_dataset: bool = True,
    # pool / attempts
    max_move_tries: int = 25,
    pool_refresh_period: int = 250,
    pool_max_moves: int = 5000,
    # reheating
    reheat_patience: int = 3000,
    reheat_factor: float = 1.5,
    reheat_cap: float = 600.0,
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
) -> Tuple[int, List[Dict[str, Any]]]:
    
    random.seed(seed)

    if plot_live:
        plt.ion()

    h = HamiltonianSTL(width, height)
    if not is_valid_cycle(h):
        raise RuntimeError("Initial Hamiltonian cycle invalid. Check HamiltonianSTL initialization.")

    zones, zones_meta = build_zones(
        width, height,
        zone_mode=zone_mode,
        seed=seed,
        num_islands=num_islands,
        island_size=island_size,
        allow_touch=allow_touch,
        stripe_direction=stripe_direction,
        stripe_k=stripe_k,
        voronoi_k=voronoi_k,
    )

    current_cost = compute_crossings(h, zones)
    initial_crossings = current_cost

    best_cost = current_cost
    best_state = deep_copy(h.H, h.V)

    accepted_ops: List[Dict[str, Any]] = []
    best_ops: List[Dict[str, Any]] = []

    accepted = attempted = rejected = 0
    invalid_moves = apply_fail = 0

    # Reheating state
    best_seen = best_cost
    no_improve = 0

    print(f"Initial crossings: {current_cost}")
    print(f"Zone mode: {zone_mode} | meta={zones_meta}")

    # Always allow boundary bias
    move_pool = refresh_move_pool(h, zones, bias_to_boundary=True, max_moves=pool_max_moves)
    print(f"[Pool] initial size = {len(move_pool)}")

    if plot_live:
        plot_cycle(h, zones, title=f"Initial | crossings={current_cost}")
        plt.pause(pause_seconds)

    for i in range(iterations):
        attempted += 1

        # periodic pool refresh
        if i % pool_refresh_period == 0:
            move_pool = refresh_move_pool(h, zones, bias_to_boundary=True, max_moves=pool_max_moves)
            if i % (pool_refresh_period * 10) == 0:
                print(f"[Pool] iter={i} size={len(move_pool)}")

        T = dynamic_temperature(i, iterations, Tmin=Tmin, Tmax=Tmax)

        prev_H, prev_V = deep_copy(h.H, h.V)
        applied_move: Optional[Dict[str, Any]] = None

        # 1) sample from pool
        if move_pool:
            mv = random.choice(move_pool)
            if apply_move(h, mv):
                applied_move = mv
            else:
                apply_fail += 1
                h.H, h.V = prev_H, prev_V

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

                if not apply_move(h, mv_try):
                    apply_fail += 1
                    h.H, h.V = prev_H, prev_V
                    continue

                applied_move = mv_try
                break

            if applied_move is None:
                invalid_moves += 1
                h.H, h.V = prev_H, prev_V

        # acceptance step
        if applied_move is None:
            no_improve += 1
        else:
            new_cost = compute_crossings(h, zones)
            delta = new_cost - current_cost

            # Metropolis acceptance
            if delta < 0:
                accept = True
            else:
                if T <= 0:
                    accept = False
                else:
                    x = -float(delta) / float(T)
                    x = max(-700.0, min(700.0, x))
                    accept = (random.random() < math.exp(x))

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

                # stall tracking
                if best_cost < best_seen:
                    best_seen = best_cost
                    no_improve = 0
                else:
                    no_improve += 1

            else:
                rejected += 1
                h.H, h.V = prev_H, prev_V
                no_improve += 1

        # reheating
        if no_improve >= reheat_patience:
            Tmax = min(reheat_cap, Tmax * reheat_factor)
            no_improve = 0
            move_pool = refresh_move_pool(h, zones, bias_to_boundary=True, max_moves=pool_max_moves)
            print(f"[Reheat] iter={i} Tmax={Tmax:.2f} pool={len(move_pool)}")

        if i % 500 == 0:
            print(
                f"Iter {i}: T={T:.3f}, Tmax={Tmax:.2f}, Cost={current_cost}, Best={best_cost}, "
                f"Accepted={accepted}/{attempted}, Rejected={rejected}, "
                f"Invalid={invalid_moves}, ApplyFail={apply_fail}, "
                f"Pool={len(move_pool)}, NoImprove={no_improve}"
            )

    # restore best
    h.H, h.V = best_state
    if not is_valid_cycle(h):
        raise RuntimeError("Best state invalid at end (should not happen).")

    print(f"Final best crossings: {best_cost}")

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
        )

    if plot_live:
        plot_cycle(h, zones, title=f"FINAL SA BEST | crossings={best_cost}")
        plt.ioff()
        plt.show()

    return int(best_cost), best_ops


def run_sa_multiple_seeds(
    seeds: List[int],
    *,
    width: int = 30,
    height: int = 30,
    iterations: int = 5000,
    Tmax: float = 80.0,
    Tmin: float = 0.5,
    zone_mode: str = "left_right",
    dataset_dir: str = "Dataset",
    write_dataset: bool = True,
    plot_live: bool = False,
    max_move_tries=25,
    pool_refresh_period=250,
    pool_max_moves=5000,
    reheat_patience=3000,
    reheat_factor=1.5,
    reheat_cap=600.0,
    # islands
    num_islands: int = 3,
    island_size: int = 8,
    allow_touch: bool = False,
    # stripes
    stripe_direction: str = "v",
    stripe_k: int = 3,
    # voronoi
    voronoi_k: int = 3,
) -> List[Dict[str, Any]]:
    
    results: List[Dict[str, Any]] = []
    for s in seeds:
        best_c, best_ops = run_sa(
            width=width,
            height=height,
            iterations=iterations,
            Tmax=Tmax,
            Tmin=Tmin,
            seed=s,
            plot_live=True,
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
        )
        results.append({
            "seed": int(s),
            "best_crossings": int(best_c),
            "sequence_len": int(len(best_ops)),
        })
        print(f"[SA multi] seed={s} best_crossings={best_c} sequence_len={len(best_ops)}")

    return results


# ============================================================
# MAin
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # Single-run
    # ============================================================
    """run_sa(
        # grid
        width=32,
        height=32,

        # SA schedule
        iterations=5000,
        Tmax=80.0,
        Tmin=0.5,
        seed=0,

        # visualization
        plot_live=True,
        show_every_accepted=200,
        pause_seconds=0.01,

        # dataset
        dataset_dir="Dataset",
        write_dataset=True,

        # pool / attempts
        max_move_tries=25,
        pool_refresh_period=250,
        pool_max_moves=5000,

        # reheating
        reheat_patience=3000,
        reheat_factor=1.5,
        reheat_cap=600.0,

        # zone pattern
        zone_mode="left_right",   # "left_right" | "islands" | "stripes" | "voronoi"

        # islands params (used only if zone_mode="islands")
        num_islands=3,
        island_size=8,
        allow_touch=False,

        # stripes params (used only if zone_mode="stripes")
        stripe_direction="v",
        stripe_k=3,

        # voronoi params (used only if zone_mode="voronoi")
        voronoi_k=3,
    )"""

    # ============================================================
    # Multi-run
    # ============================================================
    seeds = list(range(10))
    run_sa_multiple_seeds(
        seeds,

        # grid
        width=32,
        height=32,

        # SA schedule
        iterations=5000,
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

        # zone pattern
        zone_mode="left_right",   # "left_right" | "islands" | "stripes" | "voronoi"

        # dataset
        dataset_dir="Dataset",
        write_dataset=True,

        # visualization
        plot_live=True,

        # islands params (used only if zone_mode="islands")
        num_islands=3,
        island_size=8,
        allow_touch=False,

        # stripes params (used only if zone_mode="stripes")
        stripe_direction="v",
        stripe_k=3,

        # voronoi params (used only if zone_mode="voronoi")
        voronoi_k=3,
    )
