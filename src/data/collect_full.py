"""
Full-scale data collection from SA Algorithm.

Runs SA optimization across multiple grid sizes, zone patterns, and seeds.
"""

import time
import random
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.hamiltonian import HamiltonianSTL
from src.core.zones import (
    zones_left_right, zones_top_bottom, zones_diagonal,
    zones_stripes, zones_checkerboard, zones_voronoi
)
from src.data.collector import ZoningCollector, RunMeta
from src.data.collector_helper import mutate_layer_logged
from src.optimization import simulated_annealing as SA
from src.optimization import sa_patterns as SA_patterns

OUT_DIR = "Dataset"
SEED = 42

SIZES = [(30, 30), (50, 50), (80, 80), (100, 100), (150, 150), (200, 200)]
# Number of Zones
K_LIST = [2, 3, 4]

INSTANCES_PER_COMBO = 10

SA_ITERS = 5000
TMAX = 80.0
TMIN = 0.5

max_move_tries = 25
pool_refresh_period = 250
pool_max_moves=5000
reheat_patience=3000
reheat_factor=1.5
reheat_cap=600.0

def patterns(patt_name: str, k: int):
    if patt_name == "left_right":
        return SA.run_sa, {}

    if patt_name == "diagonal":
        return SA_patterns.run_sa, {"zone_mode": "diagonal"}

    if patt_name == "islands":
        return SA_patterns.run_sa, {
            "zone_mode": "islands",
            "num_islands": int(k),
            "island_size": 8,
            "allow_touch": False,
        }
    if patt_name == "stripes_v":
        return SA_patterns.run_sa, {
            "zone_mode": "stripes",
            "stripe_direction": "v",
            "stripe_k": int(k),
        }

    if patt_name == "stripes_h":
        return SA_patterns.run_sa, {
            "zone_mode": "stripes",
            "stripe_direction": "h",
            "stripe_k": int(k),
        }

    if patt_name == "voronoi":
        return SA_patterns.run_sa, {
            "zone_mode": "voronoi",
            "voronoi_k": int(k),
        }
    
    raise ValueError(f"Unknown pattern: {patt_name}")


PATTERNS = [
    "left_right",
    "diagonal",
    "islands",
    "stripes_v",
    "voronoi",
]


def collect_one_instance(W: int, H: int, patt_name: str, k: int, seed: int):
    runner, extra_kwargs = patterns(patt_name, k)

    tag_k = k if patt_name != "left_right" else 2

    print(f"[Collect_SA] start: {W}x{H} pattern={patt_name} k={tag_k} seed={seed}")

    best_cost, best_ops = runner(
        width=W,
        height=H,
        iterations=SA_ITERS,
        Tmax=TMAX,
        Tmin=TMIN,
        seed=seed,
        plot_live=False,               
        show_every_accepted=0,
        pause_seconds=0.0,
        dataset_dir=OUT_DIR,           
        max_move_tries=max_move_tries,
        pool_refresh_period=pool_refresh_period,
        pool_max_moves=pool_max_moves,
        reheat_patience=reheat_patience,
        reheat_factor=reheat_factor,
        reheat_cap=reheat_cap,
        **extra_kwargs,
    )

    print(f"[Collect_SA] done : {W}x{H} pattern={patt_name} k={tag_k} seed={seed} best={best_cost}")
    return best_cost, best_ops

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)

    global_run_idx = 0

    for (W, H) in SIZES:
        for patt_name in PATTERNS:
            for k in K_LIST:
                for j in range(INSTANCES_PER_COMBO):
                    seed = SEED + global_run_idx
                    global_run_idx += 1

                    collect_one_instance(W, H, patt_name, k, seed)

    print(f"\n[Collect_SA] Finished. Data saved in: {OUT_DIR}")

if __name__ == "__main__":
    main()
