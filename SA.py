# Left and right zones
import random
import math
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any, List

from Flip_Transpose2 import HamiltonianSTL
from Collector import ZoningCollector, RunMeta
from Collector_helper import mutate_layer_logged

Point = Tuple[int, int]


def _sigmoid_stable(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def dynamic_temperature(l: int, L: int, Tmin: float, Tmax: float) -> float:
    k = 10.0 / max(1.0, float(L))
    z = k * (-float(l) + float(L) / 2.0)
    s = _sigmoid_stable(z)
    return Tmin + (Tmax - Tmin) * s


def deep_copy(H, V):
    return ([row[:] for row in H], [row[:] for row in V])


def is_valid_cycle(h: HamiltonianSTL) -> bool:
    if hasattr(h, "validate_full_path_cycle"):
        return bool(h.validate_full_path_cycle())
    if hasattr(h, "validate_full_path"):
        return bool(h.validate_full_path())
    return True


class HamiltonianZoningSA:
    def __init__(self, h: HamiltonianSTL):
        self.h = h
        self.W, self.Ht = h.width, h.height

        # Left/right zones
        self.zones = {
            (x, y): 1 if x < self.W // 2 else 2
            for y in range(self.Ht)
            for x in range(self.W)
        }

    def compute_crossings(self) -> int:
        count = 0

        # Horizontal edges
        for y in range(self.Ht):
            for x in range(self.W - 1):
                if self.h.H[y][x]:
                    a, b = (x, y), (x + 1, y)
                    if self.zones[a] != self.zones[b]:
                        count += 1

        # Vertical edges
        for y in range(self.Ht - 1):
            for x in range(self.W):
                if self.h.V[y][x]:
                    a, b = (x, y), (x, y + 1)
                    if self.zones[a] != self.zones[b]:
                        count += 1

        return count

    def apply_move(self, mv: Dict[str, Any]) -> bool:
        op = mv["op"]
        variant = mv["variant"]
        x, y, w, h = mv["x"], mv["y"], mv["w"], mv["h"]

        if op == "transpose":
            sub3 = self.h.get_subgrid((x, y), (x + 2, y + 2))
            _, result = self.h.transpose_subgrid(sub3, variant)
            return isinstance(result, str) and result.startswith("transposed")

        if op == "flip":
            sub = self.h.get_subgrid((x, y), (x + w - 1, y + h - 1))
            _, result = self.h.flip_subgrid(sub, variant)
            return isinstance(result, str) and result.startswith("flipped")

        return False

    def plot(self, title="Path"):
        plt.clf()

        # Zone background
        for y in range(self.Ht):
            for x in range(self.W):
                color = "lightblue" if self.zones[(x, y)] == 1 else "lightgreen"
                plt.gca().add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=color))

        # Horizontal edges
        for y in range(self.Ht):
            for x in range(self.W - 1):
                if self.h.H[y][x]:
                    a, b = (x, y), (x + 1, y)
                    color = "red" if self.zones[a] != self.zones[b] else "black"
                    plt.plot([x, x + 1], [y, y], color=color, linewidth=2)

        # Vertical edges
        for y in range(self.Ht - 1):
            for x in range(self.W):
                if self.h.V[y][x]:
                    a, b = (x, y), (x, y + 1)
                    color = "red" if self.zones[a] != self.zones[b] else "black"
                    plt.plot([x, x], [y, y + 1], color=color, linewidth=2)

        # Vertices
        xs, ys = zip(*[(x, y) for y in range(self.Ht) for x in range(self.W)])
        plt.scatter(xs, ys, color="black", s=10, edgecolors="none", zorder=3)

        plt.title(f"{title} (Crossings = {self.compute_crossings()})")
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.grid(False)


def get_layer_positions_generic(W, H, layer):
    offset = 2 * (layer - 1)
    x_min = offset
    x_max = W - 3 - offset
    y_min = 0
    y_max = H - 3
    pos = []
    for x in range(max(0, x_min), max(-1, x_max) + 1):
        if x == x_min or x == x_max:
            for y in range(max(0, y_min), max(-1, y_max) + 1):
                if x + 2 < W and y + 2 < H:
                    pos.append((x, y))
    return pos


class ZoningAdapterForSA:
    def __init__(self, sa_obj: HamiltonianZoningSA):
        self._sa = sa_obj
        self.h = sa_obj.h
        self.zones = sa_obj.zones
        self.W, self.Ht = sa_obj.W, sa_obj.Ht
        self.step_t = 0

    def compute_crossings(self):
        return self._sa.compute_crossings()

    def get_layer_positions(self, layer):
        return get_layer_positions_generic(self.W, self.Ht, layer)


def _try_move_feasible(zoning: HamiltonianZoningSA, mv: Dict[str, Any]) -> bool:
    h = zoning.h
    H0, V0 = deep_copy(h.H, h.V)

    ok_apply = zoning.apply_move(mv)
    if not ok_apply:
        h.H, h.V = H0, V0
        return False

    ok_cycle = is_valid_cycle(h)
    h.H, h.V = H0, V0
    return ok_cycle


def refresh_move_pool(
    zoning: HamiltonianZoningSA,
    *,
    bias_to_boundary: bool = True,
    boundary_band: int = 6,
    max_moves: int = 5000,
) -> List[Dict[str, Any]]:
    W, Ht = zoning.W, zoning.Ht
    pool: List[Dict[str, Any]] = []

    xs = list(range(W))
    if bias_to_boundary:
        mid = W // 2
        xs.sort(key=lambda x: abs(x - mid))

    # Transpose variants
    tvars = list(zoning.h.transpose_patterns)

    # Flip variants and sizes
    fvars = [('n', 3, 2), ('s', 3, 2), ('e', 2, 3), ('w', 2, 3)]

    # Transpose scan
    for y in range(0, Ht - 2):
        for x in xs:
            if x > W - 3:
                continue
            if bias_to_boundary:
                mid = W // 2
                if abs(x - mid) > boundary_band:
                    pass

            random.shuffle(tvars)
            for variant in tvars:
                mv = {"op": "transpose", "variant": variant, "x": x, "y": y, "w": 3, "h": 3}
                if _try_move_feasible(zoning, mv):
                    pool.append(mv)
                    break

            if len(pool) >= max_moves:
                return pool

    # Flip scan
    for (variant, w, h) in fvars:
        for y in range(0, Ht - h + 1):
            for x in xs:
                if x > W - w:
                    continue

                mv = {"op": "flip", "variant": variant, "x": x, "y": y, "w": w, "h": h}
                if _try_move_feasible(zoning, mv):
                    pool.append(mv)

                if len(pool) >= max_moves:
                    return pool

    return pool


# Main SA runner (with reheating + move pool)
def run_sa(
    width: int = 32,
    height: int = 32,
    iterations: int = 2000,
    Tmax: float = 80.0,
    Tmin: float = 0.5,
    seed: int = 42,
    plot_live: bool = True,
    show_every_accepted: int = 200,
    pause_seconds: float = 0.05,
    dataset_dir: str = "Dataset1",
    # SA move mechanics
    max_move_tries: int = 25,           
    pool_refresh_period: int = 250,     
    pool_max_moves: int = 5000,
    # reheating
    reheat_patience: int = 3000,
    reheat_factor: float = 1.5,
    reheat_cap: float = 600.0,
):
    random.seed(seed)

    if plot_live:
        plt.ion()

    h = HamiltonianSTL(width, height)
    zoning = HamiltonianZoningSA(h)

    if not is_valid_cycle(h):
        raise RuntimeError("Initial Hamiltonian cycle invalid. Check HamiltonianSTL initialization.")

    collector = ZoningCollector(out_dir=dataset_dir, alpha=1.0, gamma=10.0)
    run_id = f"sa_{int(time.time())}"
    collector.write_run_meta(RunMeta(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        algorithm="SA",
        init_path="zigzag",
        grid_w=width, grid_h=height,
        random_seed=seed,
        zones_max=6,
    ))

    instance_id = f"grid{width}x{height}_sa_seed{seed}"
    z_adapt = ZoningAdapterForSA(zoning)
    max_layer = max(1, min(width, height) // 4)

    current_cost = zoning.compute_crossings()
    best_cost = current_cost
    best_state = deep_copy(h.H, h.V)

    accepted_ops: List[Dict[str, Any]] = []
    best_ops: List[Dict[str, Any]] = []

    accepted = 0
    attempted = 0
    invalid_moves = 0
    apply_fail = 0
    rejected = 0

    # Reheating state
    best_seen = best_cost
    no_improve = 0

    print(f"Initial crossings: {current_cost}")

    # Build initial pool
    move_pool = refresh_move_pool(zoning, max_moves=pool_max_moves)
    print(f"[Pool] initial size = {len(move_pool)}")

    if plot_live:
        zoning.plot("Initial Path")
        plt.pause(pause_seconds)

    for i in range(iterations):
        attempted += 1

        # refresh pool periodically
        if i % pool_refresh_period == 0:
            move_pool = refresh_move_pool(zoning, max_moves=pool_max_moves)
            # print occasionally (avoid spamming)
            if i % (pool_refresh_period * 10) == 0:
                print(f"[Pool] iter={i} size={len(move_pool)}")

        T = dynamic_temperature(i, iterations, Tmin=Tmin, Tmax=Tmax)

        prev_H, prev_V = deep_copy(h.H, h.V)

        applied_move: Optional[Dict[str, Any]] = None

        # Prefer sampling from pool
        if move_pool:
            mv = random.choice(move_pool)
            if zoning.apply_move(mv) and is_valid_cycle(h):
                applied_move = mv
            else:
                apply_fail += 1
                h.H, h.V = prev_H, prev_V
        else:
            for _ in range(max_move_tries):
                x3 = random.randint(0, zoning.W - 3)
                y3 = random.randint(0, zoning.Ht - 3)

                if random.random() < 0.5:
                    variant = random.choice(list(zoning.h.transpose_patterns))
                    mv_try = {"op": "transpose", "variant": variant, "x": x3, "y": y3, "w": 3, "h": 3}
                else:
                    variants = {'n': (3, 2), 's': (3, 2), 'e': (2, 3), 'w': (2, 3)}
                    variant, (w, hh) = random.choice(list(variants.items()))
                    x = random.randint(0, zoning.W - w)
                    y = random.randint(0, zoning.Ht - hh)
                    mv_try = {"op": "flip", "variant": variant, "x": x, "y": y, "w": w, "h": hh}

                if not zoning.apply_move(mv_try):
                    apply_fail += 1
                    h.H, h.V = prev_H, prev_V
                    continue

                if not is_valid_cycle(h):
                    invalid_moves += 1
                    h.H, h.V = prev_H, prev_V
                    continue

                applied_move = mv_try
                break

        if applied_move is None:
            no_improve += 1
        else:
            new_cost = zoning.compute_crossings()
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

                # Track the accepted move in the SA trajectory (for GA population)
                accepted_ops.append({
                    "op": str(applied_move["op"]),
                    "variant": str(applied_move["variant"]),
                    "x": int(applied_move["x"]),
                    "y": int(applied_move["y"]),
                    "w": int(applied_move.get("w", 0)),
                    "h": int(applied_move.get("h", 0)),
                })

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_state = deep_copy(h.H, h.V)
                    # Snapshot the operation sequence that produced the best state
                    best_ops = accepted_ops.copy()

                # Plot occasionally
                if plot_live and show_every_accepted > 0 and (accepted % show_every_accepted == 0):
                    zoning.plot(f"Iter {i} | T={T:.3f} | Cost={current_cost} | Best={best_cost}")
                    plt.pause(pause_seconds)

                # Log the accepted move
                layer_id = 1 if max_layer <= 1 else random.randint(1, max_layer)
                mutate_layer_logged(
                    zoning=z_adapt,
                    collector=collector,
                    run_id=run_id,
                    instance_id=instance_id,
                    layer_id=layer_id,
                    attempts=1,
                    zone_pattern="sa_run",
                    zone_params={
                        "T": float(T),
                        "iter": int(i),
                        "delta": int(delta),
                        "op": str(applied_move["op"]),
                        "variant": str(applied_move["variant"]),
                        "x": int(applied_move["x"]),
                        "y": int(applied_move["y"]),
                        "w": int(applied_move["w"]),
                        "h": int(applied_move["h"]),
                        "Tmax": float(Tmax),
                        "pool_size": int(len(move_pool)),
                    },
                    num_zones=2,
                    allow_flip=True,
                    Z_MAX=6,
                    add_dist=False
                )
                z_adapt.step_t += 1

            else:
                rejected += 1
                h.H, h.V = prev_H, prev_V

            if best_cost < best_seen:
                best_seen = best_cost
                no_improve = 0
            else:
                no_improve += 1

        # Reheating
        if no_improve >= reheat_patience:
            Tmax = min(reheat_cap, Tmax * reheat_factor)
            no_improve = 0
            # refresh pool right after reheating
            move_pool = refresh_move_pool(zoning, max_moves=pool_max_moves)
            print(f"[Reheat] iter={i} Tmax={Tmax:.2f} pool={len(move_pool)}")

        if i % 500 == 0:
            print(
                f"Iter {i}: T={T:.3f}, Tmax={Tmax:.2f}, Cost={current_cost}, Best={best_cost}, "
                f"Accepted={accepted}/{attempted}, Rejected={rejected}, "
                f"Invalid={invalid_moves}, ApplyFail={apply_fail}, "
                f"Pool={len(move_pool)}, NoImprove={no_improve}"
            )

    # Restore best at end
    h.H, h.V = best_state

    if not is_valid_cycle(h):
        raise RuntimeError("Best state invalid at end (should not happen).")

    print(f"Final best crossings: {best_cost}")

    if plot_live:
        zoning.plot("Final Optimized Path")
        plt.pause(pause_seconds)
        plt.ioff()
        plt.show()
        
    return best_cost, best_ops

"""if __name__ == "__main__":
    run_sa(
        width=30,
        height=30,
        iterations=5000,
        Tmax=80.0,
        Tmin=0.5,
        seed=42,
        plot_live=True,
        show_every_accepted=200,
        pause_seconds=0.05,
        dataset_dir="Dataset1",
        max_move_tries=25,
        pool_refresh_period=250,
        pool_max_moves=5000,
        reheat_patience=3000,
        reheat_factor=1.5,
        reheat_cap=600.0,
    )"""


# ============================================================
# Multi-run experiment 
import io
import re
import contextlib
import statistics


def run_sa_return_best(**kwargs) -> int:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_sa(**kwargs)

    out = buf.getvalue()

    m = re.search(r"Final best crossings:\s*(\d+)", out)
    if not m:
        tail = "\n".join(out.splitlines()[-40:])
        raise RuntimeError("Could not parse 'Final best crossings' from output.\n--- tail ---\n" + tail)

    return int(m.group(1))


def run_sa_multiple_seeds(
    seeds,
    *,
    width=30,
    height=30,
    iterations=5000,
    Tmax=80.0,
    Tmin=0.5,
    plot_live=False,
    show_every_accepted=0,
    pause_seconds=0.0,
    dataset_dir="Dataset1",
    max_move_tries=25,
    pool_refresh_period=250,
    pool_max_moves=5000,
    reheat_patience=3000,
    reheat_factor=1.5,
    reheat_cap=600.0,
):
    total_start = time.perf_counter()

    results = []

    for s in seeds:
        seed_start = time.perf_counter()
        best_cross = run_sa_return_best(
            width=width,
            height=height,
            iterations=iterations,
            Tmax=Tmax,
            Tmin=Tmin,
            seed=s,
            plot_live=plot_live,
            show_every_accepted=show_every_accepted,
            pause_seconds=pause_seconds,
            dataset_dir=dataset_dir,
            max_move_tries=max_move_tries,
            pool_refresh_period=pool_refresh_period,
            pool_max_moves=pool_max_moves,
            reheat_patience=reheat_patience,
            reheat_factor=reheat_factor,
            reheat_cap=reheat_cap,
        )

        seed_end = time.perf_counter()
        print(f"[Timing] seed={s} runtime = {seed_end - seed_start:.2f} seconds")

        results.append(best_cross)
        print(f"seed={s}  best_crossings={best_cross}")

    best_val = min(results)
    mean_val = statistics.mean(results)
    std_val = statistics.pstdev(results)

    print("\n" + "#" * 60)
    print(f"SA summary over {len(results)} runs")
    print(f"best : {best_val}")
    print(f"mean : {mean_val:.2f}")
    print(f"std  : {std_val:.2f}")
    print("#" * 60)

    total_end = time.perf_counter()
    print(f"[Timing] TOTAL runtime for {len(seeds)} seeds = {total_end - total_start:.2f} seconds")

    return results, best_val, mean_val, std_val


if __name__ == "__main__":
    seeds = list(range(10))  
    run_sa_multiple_seeds(
        seeds,
        width=50,
        height=50,
        iterations=5000,
        Tmax=80.0,
        Tmin=0.5,
        plot_live=False,         # batch mode
        show_every_accepted=0,
        pause_seconds=0.0,
        dataset_dir="Dataset1",
        max_move_tries=25,
        pool_refresh_period=250,
        pool_max_moves=5000,
        reheat_patience=3000,
        reheat_factor=1.5,
        reheat_cap=600.0,
    )

