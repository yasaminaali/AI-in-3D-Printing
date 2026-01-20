#!/usr/bin/env python
"""
Medium-scale data collection for training.
More data than quick test, but practical runtime (~30-60 minutes).
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

# Medium-scale configuration
SIZES = [(20, 20), (30, 30), (40, 40), (50, 50)]
K_LIST = [2, 3]
INSTANCES_PER_COMBO = 3

SA_ITERS = 2000
TMAX = 60.0
TMIN = 0.5

max_move_tries = 25
pool_refresh_period = 200
pool_max_moves = 3000
reheat_patience = 1500
reheat_factor = 1.5
reheat_cap = 400.0

PATTERNS = [
    "left_right",
    "diagonal",
    "stripes_v",
    "voronoi",
]


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


def collect_one_instance(W: int, H: int, patt_name: str, k: int, seed: int):
    runner, extra_kwargs = patterns(patt_name, k)
    tag_k = k if patt_name != "left_right" else 2

    print(f"[Medium] start: {W}x{H} pattern={patt_name} k={tag_k} seed={seed}")

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

    print(f"[Medium] done: {W}x{H} pattern={patt_name} k={tag_k} seed={seed} best={best_cost}")
    return best_cost, best_ops


def main():
    """Run medium-scale data collection."""
    # Clear old dataset for fresh start
    import shutil
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    random.seed(SEED)
    
    total_runs = len(SIZES) * len(PATTERNS) * len(K_LIST) * INSTANCES_PER_COMBO
    
    print("=" * 60)
    print("Medium-Scale Data Collection")
    print("=" * 60)
    print(f"Grid sizes: {SIZES}")
    print(f"Patterns: {PATTERNS}")
    print(f"K values: {K_LIST}")
    print(f"Instances per combo: {INSTANCES_PER_COMBO}")
    print(f"Total runs: {total_runs}")
    print(f"Iterations per run: {SA_ITERS}")
    print(f"Estimated time: ~{total_runs * 0.5:.0f}-{total_runs * 1.5:.0f} minutes")
    print("=" * 60)
    print()

    global_run_idx = 0
    start_time = time.time()
    results = []

    for (W, H) in SIZES:
        for patt_name in PATTERNS:
            for k in K_LIST:
                for j in range(INSTANCES_PER_COMBO):
                    seed = SEED + global_run_idx
                    global_run_idx += 1

                    print(f"\nProgress: {global_run_idx}/{total_runs}")
                    run_start = time.time()
                    
                    best_cost, _ = collect_one_instance(W, H, patt_name, k, seed)
                    
                    run_time = time.time() - run_start
                    results.append({
                        'size': f"{W}x{H}",
                        'pattern': patt_name,
                        'k': k,
                        'seed': seed,
                        'best_cost': best_cost,
                        'time': run_time
                    })

    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("Data Collection Complete!")
    print("=" * 60)
    print(f"Total runs: {global_run_idx}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Average per run: {total_time / global_run_idx:.1f} seconds")
    print(f"\nBest results by pattern:")
    
    for patt in PATTERNS:
        patt_results = [r for r in results if r['pattern'] == patt]
        if patt_results:
            best = min(patt_results, key=lambda x: x['best_cost'])
            avg = sum(r['best_cost'] for r in patt_results) / len(patt_results)
            print(f"  {patt}: best={best['best_cost']}, avg={avg:.1f}")
    
    print(f"\nData saved in: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
