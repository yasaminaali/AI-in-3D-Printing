# Multiple Zones Patterns: Islands - Diagonal - Stripes - Voronoi Diagram
import random
import math
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any, List

from Flip_Transpose2 import HamiltonianSTL
from Collector import ZoningCollector, RunMeta
from Collector_helper import mutate_layer_logged

from Zones import zones_diagonal, zones_stripes, zones_voronoi

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


def is_valid_path(h: HamiltonianSTL) -> bool:
    if hasattr(h, "validate_full_path"):
        return bool(h.validate_full_path())
    if hasattr(h, "validate_full_path"):
        return bool(h.validate_full_path())
    return True


class HamiltonianZoningSA:
    def __init__(
        self,
        h: HamiltonianSTL,
        *,
        zone_mode: str = "islands",
        num_islands: int = 3,
        island_size: int = 8,
        seed: int = 0,
        allow_touch: bool = False,  # if False, islands won't even touch each other
        # stripes
        stripe_direction: str = "v",
        stripe_k: int = 3,
        # voronoi
        voronoi_k: int = 3,
    ):
        self.h = h
        self.W, self.Ht = h.width, h.height

        rng = random.Random(seed)

        self.zone_colors = {
            1: "lightblue",   # big background zone
            0: "lightgreen",
            2: "lightgreen",  # islands
        }

        # Start all background (zone 1)
        if zone_mode == "diagonal":
            self.zones = zones_diagonal(self.W, self.Ht)
            # store pattern name
            self.zone_mode = "diagonal"
            self._ensure_zone_colors()
            return 
        
        elif zone_mode == "stripes":
            self.zones = zones_stripes(
            self.W,
            self.Ht,
            direction=stripe_direction,
            k=stripe_k,
            )
            self.zone_mode = "stripes"
            self._ensure_zone_colors()
            return
        
        elif zone_mode == "voronoi":
            z, meta = zones_voronoi(self.W, self.Ht, k=voronoi_k)
            self.zones = z
            self.zone_mode = "voronoi"
            self.zone_meta = meta  # contains {"seeds": [...]}

            self._ensure_zone_colors() 

            return
        
        elif zone_mode == "islands":
            self.zone_mode = "islands"
            # Start all background (zone 1)
            self.zones = {(x, y): 1 for y in range(self.Ht) for x in range(self.W)}
        else:
            raise ValueError(f"Unknown zone_mode: {zone_mode}")
        
        S = int(island_size)
        if S <= 0 or S > self.W or S > self.Ht:
            raise ValueError("island_size must be in [1, min(width,height)]")

        max_x = self.W - S
        max_y = self.Ht - S

        anchors: List[Tuple[int, int]] = []

        def overlaps(ax, ay, bx, by) -> bool:
            pad = 0 if allow_touch else 1

            aL, aR = ax - pad, ax + S - 1 + pad
            aT, aB = ay - pad, ay + S - 1 + pad

            bL, bR = bx, bx + S - 1
            bT, bB = by, by + S - 1

            return not (aR < bL or bR < aL or aB < bT or bB < aT)

        attempts = 0
        max_attempts = 50_000
        while len(anchors) < num_islands and attempts < max_attempts:
            attempts += 1
            ax = rng.randint(0, max_x)
            ay = rng.randint(0, max_y)
            if all(not overlaps(ax, ay, bx, by) for (bx, by) in anchors):
                anchors.append((ax, ay))

        if len(anchors) < num_islands:
            raise RuntimeError(
                f"Could not place {num_islands} non-overlapping {S}x{S} islands. "
                f"Try allow_touch=True or reduce num_islands/island_size."
            )

        # Color islands (zone 2)
        for (ax, ay) in anchors:
            for y in range(ay, ay + S):
                for x in range(ax, ax + S):
                    self.zones[(x, y)] = 2

        self.island_anchors = anchors
        self.island_size = S

    def _ensure_zone_colors(self):
            palette = ["lightblue", "lightyellow", "plum", "lightpink", "lightgray", "wheat"]
            for zid in set(self.zones.values()):
                if zid not in self.zone_colors:
                    self.zone_colors[zid] = palette[zid % len(palette)]

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
                zid = self.zones[(x, y)]
                color = self.zone_colors.get(zid, "lightgray")
                plt.gca().add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=color))

        # Draw edges, red if they cross a zone boundary
        for y in range(self.Ht):
            for x in range(self.W - 1):
                if self.h.H[y][x]:
                    a, b = (x, y), (x + 1, y)
                    color = "red" if self.zones[a] != self.zones[b] else "black"
                    plt.plot([x, x + 1], [y, y], color=color, linewidth=2)

        for y in range(self.Ht - 1):
            for x in range(self.W):
                if self.h.V[y][x]:
                    a, b = (x, y), (x, y + 1)
                    color = "red" if self.zones[a] != self.zones[b] else "black"
                    plt.plot([x, x], [y, y + 1], color=color, linewidth=2)

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


# Move Pool (state-dependent)
def _try_move_feasible(zoning: HamiltonianZoningSA, mv: Dict[str, Any]) -> bool:
    h = zoning.h
    H0, V0 = deep_copy(h.H, h.V)

    ok_apply = zoning.apply_move(mv)
    if not ok_apply:
        h.H, h.V = H0, V0
        return False

    ok_cycle = is_valid_path(h)
    h.H, h.V = H0, V0
    return ok_cycle

# Zone boundary
def _boundary_score(zoning: HamiltonianZoningSA, x: int) -> int:
    if x < 0 or x >= zoning.W - 1:
        return 10**9
    s = 0
    for y in range(zoning.Ht):
        if zoning.zones[(x, y)] != zoning.zones[(x + 1, y)]:
            s += 1
    return s


def refresh_move_pool(
    zoning: HamiltonianZoningSA,
    *,
    bias_to_boundary: bool = True,
    max_moves: int = 5000,
) -> List[Dict[str, Any]]:
    
    W, Ht = zoning.W, zoning.Ht
    pool: List[Dict[str, Any]] = []

    xs = list(range(W))
    if bias_to_boundary:
        xs.sort(key=lambda xx: -_boundary_score(zoning, min(xx, W - 2)))

    # Transpose variants
    tvars = list(zoning.h.transpose_patterns)

    # Flip variants and sizes
    fvars = [('n', 3, 2), ('s', 3, 2), ('e', 2, 3), ('w', 2, 3)]

    # Transpose scan
    for y in range(0, Ht - 2):
        for x in xs:
            if x > W - 3:
                continue

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
    iterations: int = 100_000,
    Tmax: float = 80.0,
    Tmin: float = 0.5,
    seed: int = 42,
    plot_live: bool = True,
    show_every_accepted: int = 200,
    pause_seconds: float = 0.05,
    dataset_dir: str = "Dataset1",
    max_move_tries: int = 25,           
    pool_refresh_period: int = 250,    
    pool_max_moves: int = 5000,
    # reheating
    reheat_patience: int = 3000,
    reheat_factor: float = 1.5,
    reheat_cap: float = 600.0,
    # islands params
    zone_mode: str = "islands",
    num_islands: int = 3,
    island_size: int = 8,
    allow_touch: bool = False,
    stripe_direction: str = "v",
    stripe_k: int = 3,
    # Voronoi
    voronoi_k: int = 3
):
    random.seed(None)

    if plot_live:
        plt.ion()

    h = HamiltonianSTL(width, height)
    zoning = HamiltonianZoningSA(
        h,
        zone_mode=zone_mode,
        num_islands=num_islands,
        island_size=island_size,
        seed=seed,                
        allow_touch=allow_touch,
        stripe_direction=stripe_direction,
        stripe_k=stripe_k,
        voronoi_k=voronoi_k
    )

    if not is_valid_path(h):
        raise RuntimeError("Initial Hamiltonian cycle invalid. Check HamiltonianSTL initialization.")

    collector = ZoningCollector(out_dir=dataset_dir, alpha=1.0, gamma=10.0)
    run_id = f"sa_{zone_mode}_{int(time.time())}"
    collector.write_run_meta(RunMeta(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        algorithm="SA",
        init_path="zigzag",
        grid_w=width, grid_h=height,
        random_seed=seed,
        zones_max=6,
    ))

    instance_id = f"grid{width}x{height}_sa_{zone_mode}_seed{seed}"
    z_adapt = ZoningAdapterForSA(zoning)
    max_layer = max(1, min(width, height) // 4)

    current_cost = zoning.compute_crossings()
    best_cost = current_cost
    best_state = deep_copy(h.H, h.V)

    accepted = 0
    attempted = 0
    invalid_moves = 0
    apply_fail = 0
    rejected = 0

    best_seen = best_cost
    no_improve = 0

    print(f"Initial crossings: {current_cost}")
    print(f"Zone mode: {zone_mode}")
    if zone_mode == "islands":
        print(f"Islands: {num_islands} islands, size={island_size}, anchors={getattr(zoning,'island_anchors',None)}")

    move_pool = refresh_move_pool(zoning, max_moves=pool_max_moves)
    print(f"[Pool] initial size = {len(move_pool)}")

    if plot_live:
        zoning.plot("Initial Path (Islands)")
        plt.pause(pause_seconds)

    for i in range(iterations):
        attempted += 1

        if i % pool_refresh_period == 0:
            move_pool = refresh_move_pool(zoning, max_moves=pool_max_moves)
            if i % (pool_refresh_period * 10) == 0:
                print(f"[Pool] iter={i} size={len(move_pool)}")

        T = dynamic_temperature(i, iterations, Tmin=Tmin, Tmax=Tmax)

        prev_H, prev_V = deep_copy(h.H, h.V)
        applied_move: Optional[Dict[str, Any]] = None

        if move_pool:
            mv = random.choice(move_pool)
            if zoning.apply_move(mv) and is_valid_path(h):
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

                if not is_valid_path(h):
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

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_state = deep_copy(h.H, h.V)

                if plot_live and show_every_accepted > 0 and (accepted % show_every_accepted == 0):
                    zoning.plot(f"Iter {i} | T={T:.3f} | Cost={current_cost} | Best={best_cost}")
                    plt.pause(pause_seconds)

                layer_id = 1 if max_layer <= 1 else random.randint(1, max_layer)
                mutate_layer_logged(
                    zoning=z_adapt,
                    collector=collector,
                    run_id=run_id,
                    instance_id=instance_id,
                    layer_id=layer_id,
                    attempts=1,
                    zone_pattern=f"sa_{zone_mode}",
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
                        "zone_mode": str(zone_mode),
                        "num_islands": int(num_islands),
                        "island_size": int(island_size),
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

        if no_improve >= reheat_patience:
            Tmax = min(reheat_cap, Tmax * reheat_factor)
            no_improve = 0
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

    if not is_valid_path(h):
        raise RuntimeError("Best state invalid at end (should not happen).")

    print(f"Final best crossings: {best_cost}")

    if plot_live:
        zoning.plot("Final Optimized Path (Islands)")
        plt.pause(pause_seconds)
        plt.ioff()
        plt.show()
    
    # Return best cost and empty ops list (for consistency with SA.py)
    return best_cost, []


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
    m_init = re.search(r"Initial crossings:\s*(\d+)", out)
    m = re.search(r"Final best crossings:\s*(\d+)", out)

    if not m:
        tail = "\n".join(out.splitlines()[-60:])
        raise RuntimeError("Could not parse 'Final best crossings' from output.\n--- tail ---\n" + tail)

    return int(m_init.group(1)), int(m.group(1))


def run_sa_multiple_seeds(
    seeds,
    *,
    width=32,
    height=32,
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
    num_islands=3,
    island_size=8,
    allow_touch=False,
    zone_mode="islands",
    stripe_direction="v",
    stripe_k=3,
    voronoi_k=3,
):
    results = []

    for s in seeds:
        init_cross, best_cross = run_sa_return_best(
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
            num_islands=num_islands,
            island_size=island_size,
            allow_touch=allow_touch,
            stripe_direction=stripe_direction,
            stripe_k=stripe_k,
            voronoi_k=voronoi_k,
            zone_mode=zone_mode,
        )

        results.append(best_cross)
        print(f"seed={s}  init={init_cross} best_crossings={best_cross}")

    best_val = min(results)
    mean_val = statistics.mean(results)
    std_val = statistics.pstdev(results) 

    print("\n" + "#" * 60)
    print(f"SA summary over {len(results)} runs (islands)")
    print(f"best : {best_val}")
    print(f"mean : {mean_val:.2f}")
    print(f"std  : {std_val:.2f}")
    print("#" * 60)

    return results, best_val, mean_val, std_val


if __name__ == "__main__":
    # Single run
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
        stripe_direction="v", # horizontal "h"
        stripe_k=3,
        num_islands=3,
        island_size=8,
        allow_touch=False,
        zone_mode="islands",
        voronoi_k=5,   # number of regions
    )

    """# Multiple run
    seeds = list(range(10))
    run_sa_multiple_seeds(
        seeds,
        width=80,
        height=80,
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
        stripe_direction="v", # horizontal "h"
        stripe_k=3,
        num_islands=3,
        island_size=8,
        allow_touch=False,
        zone_mode="stripes",
        voronoi_k=3,   # number of regions
    )
"""