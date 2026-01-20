"""
Quick test data collection - uses smaller grid and fewer iterations.
"""

import time
import random
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.hamiltonian import HamiltonianSTL
from src.core.zones import zones_left_right, zones_diagonal
from src.data.collector import ZoningCollector, RunMeta
from src.data.collector_helper import mutate_layer_logged
from src.optimization import simulated_annealing as SA
from src.optimization import sa_patterns as SA_patterns

OUT_DIR = "Dataset"
SEED = 42

# Smaller test configuration
SIZES = [(10, 10), (15, 15)]
K_LIST = [2]  # Only test with 2 zones
INSTANCES_PER_COMBO = 2  # Only 2 instances per combo

SA_ITERS = 500  # Reduced iterations for faster testing
TMAX = 40.0
TMIN = 0.5

max_move_tries = 15
pool_refresh_period = 100
pool_max_moves = 1000
reheat_patience = 500
reheat_factor=1.5
reheat_cap=300.0

# Only test 2 patterns for speed
PATTERNS = [
    "left_right",
    "diagonal",
]

def patterns(patt_name: str, k: int):
    if patt_name == "left_right":
        return SA.run_sa, {}
    if patt_name == "diagonal":
        return SA_patterns.run_sa, {"zone_mode": "diagonal"}
    raise ValueError(f"Unknown pattern: {patt_name}")


def collect_one_instance(W: int, H: int, patt_name: str, k: int, seed: int):
    runner, extra_kwargs = patterns(patt_name, k)
    tag_k = k if patt_name != "left_right" else 2

    print(f"[Quick Test] start: {W}x{H} pattern={patt_name} k={tag_k} seed={seed}")

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

    print(f"[Quick Test] done: {W}x{H} pattern={patt_name} k={tag_k} seed={seed} best={best_cost}")
    return best_cost, best_ops

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)

    global_run_idx = 0
    total_runs = len(SIZES) * len(PATTERNS) * len(K_LIST) * INSTANCES_PER_COMBO
    
    print(f"\n{'='*60}")
    print(f"Quick Test Data Collection")
    print(f"{'='*60}")
    print(f"Grid sizes: {SIZES}")
    print(f"Patterns: {PATTERNS}")
    print(f"K values: {K_LIST}")
    print(f"Instances per combo: {INSTANCES_PER_COMBO}")
    print(f"Total runs: {total_runs}")
    print(f"Iterations per run: {SA_ITERS}")
    print(f"{'='*60}\n")

    for (W, H) in SIZES:
        for patt_name in PATTERNS:
            for k in K_LIST:
                for j in range(INSTANCES_PER_COMBO):
                    seed = SEED + global_run_idx
                    global_run_idx += 1
                    print(f"\nProgress: {global_run_idx}/{total_runs}")
                    collect_one_instance(W, H, patt_name, k, seed)

    print(f"\n{'='*60}")
    print(f"[Quick Test] Finished! Data saved in: {OUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
