"""
Run SA and save before/after visualizations.
Usage: python run_sa_visualize.py [--zone-mode MODE] [--width W] [--height H] [--iterations N] [--seed S]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
import os
import time
import random
import copy

from operations import HamiltonianSTL
from SA_generation import (
    build_zones, compute_crossings, plot_cycle, deep_copy,
    refresh_move_pool, apply_move, is_valid_cycle,
    dynamic_temperature, op_probabilities, choose_move_from_pool,
    _snapshot_edges_for_move, _restore_edges_snapshot,
)
import math


def run_sa_with_viz(*, width, height, iterations, seed, zone_mode, init_pattern, outdir):
    """Run SA and return both initial and final HamiltonianSTL + zones for visualization."""
    os.makedirs(outdir, exist_ok=True)
    random.seed(seed)

    zones, zones_meta = build_zones(
        width, height, zone_mode=zone_mode, seed=seed,
        num_islands=3, island_size=8, allow_touch=False,
        stripe_direction="v", stripe_k=3, voronoi_k=3,
    )

    # Resolve init pattern
    actual_pattern = init_pattern
    if actual_pattern == "auto":
        if zone_mode in ("left_right", "leftright", "lr"):
            actual_pattern = "vertical_zigzag"
        elif zone_mode == "stripes":
            actual_pattern = "vertical_zigzag"
        else:
            actual_pattern = "zigzag"

    h = HamiltonianSTL(width, height, init_pattern=actual_pattern)
    initial_crossings = compute_crossings(h, zones)

    # Save initial H, V for later
    H_init, V_init = deep_copy(h.H, h.V)

    # --- SA loop (inlined so we keep the h object) ---
    current_cost = initial_crossings
    best_cost = current_cost
    best_state = deep_copy(h.H, h.V)

    accepted = attempted = rejected = 0
    invalid_moves = apply_fail = 0
    best_seen = best_cost
    no_improve = 0

    Tmax, Tmin = 80.0, 0.5
    reheat_patience, reheat_factor, reheat_cap = 1500, 1.5, 600.0
    pool_refresh_period, pool_max_moves, max_move_tries = 250, 5000, 200
    transpose_phase_ratio = 0.6
    border_to_inner = True

    split = int(iterations * transpose_phase_ratio)
    allowed_ops = {"transpose"} if split > 0 else {"transpose", "flip"}

    move_pool = refresh_move_pool(h, zones, bias_to_boundary=True,
                                  max_moves=pool_max_moves, allowed_ops=allowed_ops,
                                  border_to_inner=border_to_inner)

    # Track crossing history for plotting
    cost_history = [current_cost]

    t0 = time.perf_counter()
    for i in range(iterations):
        attempted += 1

        if i == split:
            allowed_ops = {"transpose", "flip"}
            move_pool = refresh_move_pool(h, zones, bias_to_boundary=True,
                                          max_moves=pool_max_moves, allowed_ops=allowed_ops,
                                          border_to_inner=border_to_inner)

        if i % pool_refresh_period == 0:
            allowed_ops = {"transpose"} if i < split else {"transpose", "flip"}
            move_pool = refresh_move_pool(h, zones, bias_to_boundary=True,
                                          max_moves=pool_max_moves, allowed_ops=allowed_ops,
                                          border_to_inner=border_to_inner)

        T = dynamic_temperature(i, iterations, Tmin=Tmin, Tmax=Tmax)

        applied_move = None
        applied_snap = None

        if move_pool:
            mv = random.choice(move_pool)
            snap = _snapshot_edges_for_move(h, mv)
            if apply_move(h, mv):
                applied_move = mv
                applied_snap = snap
            else:
                apply_fail += 1
                _restore_edges_snapshot(h, snap)
        else:
            for _ in range(max_move_tries):
                x3 = random.randint(0, width - 3)
                y3 = random.randint(0, height - 3)
                if random.random() < 0.5:
                    variant = random.choice(list(h.transpose_patterns.keys()))
                    mv_try = {"op": "transpose", "variant": variant, "x": x3, "y": y3, "w": 3, "h": 3}
                else:
                    variants = {'n': (2, 3), 's': (2, 3), 'e': (3, 2), 'w': (3, 2)}
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

        if applied_move is None:
            no_improve += 1
        else:
            new_cost = compute_crossings(h, zones)
            delta = new_cost - current_cost
            if delta < 0:
                accept = True
            elif T <= 0:
                accept = False
            else:
                xexp = max(-700.0, min(700.0, -float(delta) / float(T)))
                accept = (random.random() < math.exp(xexp))

            if accept:
                current_cost = new_cost
                accepted += 1
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_state = deep_copy(h.H, h.V)
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

        if no_improve >= reheat_patience:
            Tmax = min(reheat_cap, Tmax * reheat_factor)
            no_improve = 0
            allowed_ops = {"transpose"} if i < split else {"transpose", "flip"}
            move_pool = refresh_move_pool(h, zones, bias_to_boundary=True,
                                          max_moves=pool_max_moves, allowed_ops=allowed_ops,
                                          border_to_inner=border_to_inner)

        cost_history.append(best_cost)

    elapsed = time.perf_counter() - t0

    # Restore best state
    h.H, h.V = best_state

    return h, H_init, V_init, zones, initial_crossings, best_cost, cost_history, elapsed, accepted, attempted


def draw_path(ax, H_edges, V_edges, zones, W, H_grid):
    zone_vals = sorted(set(zones.values()))
    colors = ["#a8d8ea", "#a8e6a0", "#f5c6aa", "#d4a8f0", "#f0e68c", "#f0a8a8"]
    for y in range(H_grid):
        for x in range(W):
            z = zones[(x, y)]
            idx = zone_vals.index(z) % len(colors)
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=colors[idx], ec='none'))

    for y in range(H_grid):
        for x in range(W - 1):
            if H_edges[y][x]:
                col = "red" if zones[(x, y)] != zones[(x + 1, y)] else "#222222"
                lw = 2.5 if col == "red" else 1.2
                ax.plot([x, x + 1], [y, y], color=col, linewidth=lw, solid_capstyle='round')
    for y in range(H_grid - 1):
        for x in range(W):
            if V_edges[y][x]:
                col = "red" if zones[(x, y)] != zones[(x, y + 1)] else "#222222"
                lw = 2.5 if col == "red" else 1.2
                ax.plot([x, x], [y, y + 1], color=col, linewidth=lw, solid_capstyle='round')

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlim(-0.6, W - 0.4)
    ax.set_ylim(H_grid - 0.4, -0.6)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser(description="Run SA and visualize results")
    parser.add_argument("--zone-mode", default="left_right",
                        choices=["left_right", "islands", "stripes", "voronoi"])
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-pattern", default="zigzag",
                        choices=["auto", "zigzag", "vertical_zigzag", "fermat_spiral", "hilbert", "snake_bends"])
    parser.add_argument("--outdir", default="sa_viz_output")
    args = parser.parse_args()

    W, H = args.width, args.height

    print(f"SA optimization: {args.zone_mode} {W}x{H}, {args.iterations} iters, seed={args.seed}")
    print(f"Initial path: {args.init_pattern}")
    print()

    h_final, H_init, V_init, zones, init_cross, best_cross, history, elapsed, accepted, attempted = \
        run_sa_with_viz(
            width=W, height=H, iterations=args.iterations,
            seed=args.seed, zone_mode=args.zone_mode,
            init_pattern=args.init_pattern, outdir=args.outdir,
        )

    improvement = (init_cross - best_cross) / init_cross * 100 if init_cross > 0 else 0
    print(f"Initial crossings: {init_cross}")
    print(f"Final crossings:   {best_cross}")
    print(f"Improvement:       {improvement:.1f}%")
    print(f"Acceptance rate:   {accepted}/{attempted} ({accepted/max(1,attempted)*100:.1f}%)")
    print(f"Runtime:           {elapsed:.1f}s")

    # --- Comparison figure ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7),
                             gridspec_kw={'width_ratios': [1, 1, 0.8]})

    draw_path(axes[0], H_init, V_init, zones, W, H)
    axes[0].set_title(f"Before SA\n{init_cross} crossings", fontsize=16, fontweight="bold", color="#d32f2f")

    draw_path(axes[1], h_final.H, h_final.V, zones, W, H)
    axes[1].set_title(f"After SA\n{best_cross} crossings", fontsize=16, fontweight="bold", color="#2e7d32")

    # Convergence curve
    axes[2].plot(history, color="#1565c0", linewidth=1.5)
    axes[2].set_xlabel("Iteration", fontsize=12)
    axes[2].set_ylabel("Best Crossings", fontsize=12)
    axes[2].set_title("Convergence", fontsize=14, fontweight="bold")
    axes[2].axhline(y=best_cross, color="#e53935", linestyle="--", alpha=0.7, label=f"Final: {best_cross}")
    axes[2].axhline(y=init_cross, color="#9e9e9e", linestyle=":", alpha=0.7, label=f"Initial: {init_cross}")
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Simulated Annealing | {args.zone_mode} | {W}x{H} grid | "
        f"{improvement:.0f}% reduction | {elapsed:.1f}s",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(args.outdir, "comparison.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
