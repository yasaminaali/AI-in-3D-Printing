#!/usr/bin/env python
"""Quick test of SA optimization."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.optimization.simulated_annealing import run_sa

print("=" * 60)
print("Testing SA Optimization (10x10 grid, 100 iterations)")
print("=" * 60)

best_cost, best_ops = run_sa(
    width=10,
    height=10,
    iterations=100,
    Tmax=40.0,
    Tmin=0.5,
    seed=42,
    plot_live=False,
    show_every_accepted=0,
    pause_seconds=0.0,
    dataset_dir="Dataset_test",
    max_move_tries=10,
    pool_refresh_period=50,
    pool_max_moves=100,
)

print("\n" + "=" * 60)
print(f"Test Complete!")
print(f"Final crossings: {best_cost}")
print(f"Operations applied: {len(best_ops)}")
print("=" * 60)
