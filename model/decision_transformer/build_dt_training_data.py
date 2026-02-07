"""
Build Decision Transformer training data from RTG cache.

Pipeline:
1. Load rtg_cache.pkl (precomputed per-step crossings + RTG)
2. Filter trajectories by minimum crossing reduction
3. Extract effective-only operations (ones that reduce crossings)
4. Precompute grid states at effective steps via multiprocessing
5. Save as effective_dt_data.pkl

Usage:
    python model/decision_transformer/build_dt_training_data.py
    python model/decision_transformer/build_dt_training_data.py --input rtg_cache.pkl --min-reduction 0.2
"""

import pickle
import argparse
import time
import sys
import numpy as np
import torch
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns
from rich import box

console = Console()

# Variant mappings - all 12 variants
VARIANT_MAP = {
    'nl': 0, 'nr': 1, 'sl': 2, 'sr': 3, 'eb': 4, 'ea': 5, 'wa': 6, 'wb': 7,  # Transpose (8)
    'n': 8, 's': 9, 'e': 10, 'w': 11  # Flip (4)
}
OP_TYPE_MAP = {'N': 0, 'T': 1, 'F': 2}

MAX_GRID_SIZE = 32  # Pad all grids to this size


def compute_boundary_mask(zones: np.ndarray) -> torch.Tensor:
    """Compute zone boundary mask from zone grid."""
    h, w = zones.shape
    mask = torch.zeros(MAX_GRID_SIZE, MAX_GRID_SIZE)
    for y in range(h):
        for x in range(w - 1):
            if zones[y, x] != zones[y, x + 1]:
                mask[y, x] = 1.0
                mask[y, x + 1] = 1.0
    for y in range(h - 1):
        for x in range(w):
            if zones[y, x] != zones[y + 1, x]:
                mask[y, x] = 1.0
                mask[y + 1, x] = 1.0
    return mask


def encode_state(zones: np.ndarray, boundary_mask: torch.Tensor,
                 H_edges, V_edges, grid_w: int, grid_h: int) -> torch.Tensor:
    """Encode grid state as 4-channel tensor padded to MAX_GRID_SIZE."""
    state = torch.zeros(4, MAX_GRID_SIZE, MAX_GRID_SIZE)

    # Channel 0: Zone IDs normalized
    max_zone = max(zones.max(), 1)
    state[0, :grid_h, :grid_w] = torch.from_numpy(zones.astype(np.float32)) / max_zone

    # Channel 1: Horizontal edges
    for y in range(grid_h):
        for x in range(grid_w - 1):
            state[1, y, x] = float(H_edges[y][x])

    # Channel 2: Vertical edges
    for y in range(grid_h - 1):
        for x in range(grid_w):
            state[2, y, x] = float(V_edges[y][x])

    # Channel 3: Boundary mask + grid mask
    state[3] = boundary_mask
    # Also mark valid grid area
    state[3, :grid_h, :grid_w] = torch.maximum(
        state[3, :grid_h, :grid_w],
        torch.ones(grid_h, grid_w) * 0.5  # 0.5 for valid, 1.0 for boundary
    )

    return state


def process_trajectory(traj_data):
    """Process a single trajectory: extract effective ops and precompute states.

    Returns dict with effective ops, states, RTG, and metadata.
    """
    from operations import HamiltonianSTL

    traj_idx, traj = traj_data

    grid_w = traj.get('grid_W', 30)
    grid_h = traj.get('grid_H', 30)

    # Get zone grid
    zone_grid = traj['zone_grid']
    if isinstance(zone_grid, list):
        if isinstance(zone_grid[0], int):
            zones = np.array(zone_grid).reshape(grid_h, grid_w)
        else:
            zones = np.array(zone_grid)
    else:
        zones = np.array(zone_grid)

    boundary_mask = compute_boundary_mask(zones)

    # Get crossings sequence
    crossings_seq = traj['crossings_sequence']
    ops = traj['sequence_ops']

    # Find effective step indices (where crossings decrease)
    effective_indices = []
    for i in range(len(crossings_seq) - 1):
        if crossings_seq[i + 1] < crossings_seq[i]:
            effective_indices.append(i)

    if len(effective_indices) == 0:
        return None

    # Replay trajectory and save states at effective steps
    # Also include N_CONTEXT preceding ops for context
    N_CONTEXT = 3

    # Build set of all step indices we need states for
    needed_indices = set()
    context_map = {}  # effective_idx -> list of context indices

    for eff_idx in effective_indices:
        context_start = max(0, eff_idx - N_CONTEXT)
        context_indices = list(range(context_start, eff_idx + 1))
        context_map[eff_idx] = context_indices
        needed_indices.update(context_indices)

    needed_indices = sorted(needed_indices)

    # Replay and collect states
    h = HamiltonianSTL(grid_w, grid_h, init_pattern='zigzag')
    states_at_step = {}

    for step_idx in range(len(ops)):
        if step_idx in needed_indices:
            states_at_step[step_idx] = encode_state(
                zones, boundary_mask, h.H, h.V, grid_w, grid_h
            )

        # Apply operation
        op = ops[step_idx]
        kind = op['kind']
        if kind == 'N':
            continue
        try:
            x, y = op['x'], op['y']
            variant = op['variant']
            if kind == 'T':
                sub = h.get_subgrid((x, y), (x + 2, y + 2))
                h.transpose_subgrid(sub, variant)
            elif kind == 'F':
                if variant in ['n', 's']:
                    sub = h.get_subgrid((x, y), (x + 1, y + 2))
                else:
                    sub = h.get_subgrid((x, y), (x + 2, y + 1))
                h.flip_subgrid(sub, variant)
        except:
            pass

        # Stop early if past all needed indices
        if step_idx > max(needed_indices):
            break

    # Build condensed trajectory with context
    condensed_steps = []  # list of (state, action, rtg, timestep)

    total_remaining_reduction = crossings_seq[0] - crossings_seq[-1]
    cumulative_reduction = 0

    seen_steps = set()
    for eff_idx in effective_indices:
        for ctx_idx in context_map[eff_idx]:
            if ctx_idx in seen_steps:
                continue
            seen_steps.add(ctx_idx)

            if ctx_idx not in states_at_step:
                continue

            op = ops[ctx_idx]
            is_effective = (ctx_idx == eff_idx)

            # RTG at this step = total reduction remaining
            reduction_at_step = max(0, crossings_seq[ctx_idx] - crossings_seq[ctx_idx + 1])
            rtg_value = total_remaining_reduction - cumulative_reduction

            action = [
                OP_TYPE_MAP.get(op['kind'], 0),
                op['x'],
                op['y'],
                VARIANT_MAP.get(op['variant'], 0)
            ]

            condensed_steps.append({
                'state': states_at_step[ctx_idx],
                'action': action,
                'rtg': rtg_value,
                'timestep': ctx_idx,
                'is_effective': is_effective,
                'crossings_before': crossings_seq[ctx_idx],
                'crossings_after': crossings_seq[ctx_idx + 1],
            })

            if is_effective:
                cumulative_reduction += reduction_at_step

    return {
        'traj_idx': traj_idx,
        'grid_w': grid_w,
        'grid_h': grid_h,
        'zone_pattern': traj.get('zone_pattern', 'unknown'),
        'initial_crossings': crossings_seq[0],
        'final_crossings': crossings_seq[-1],
        'total_ops': len(ops),
        'num_effective': len(effective_indices),
        'num_condensed_steps': len(condensed_steps),
        'condensed_steps': condensed_steps,
        'zones': zones,
    }


def main():
    parser = argparse.ArgumentParser(description='Build DT training data from RTG cache')
    parser.add_argument('--input', default='model/decision_transformer/rtg_cache.pkl',
                        help='Path to RTG cache pickle')
    parser.add_argument('--output', default='model/decision_transformer/effective_dt_data.pkl',
                        help='Output path for effective training data')
    parser.add_argument('--min-reduction', type=float, default=0.2,
                        help='Minimum crossing reduction ratio (0-1)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of multiprocessing workers')
    args = parser.parse_args()

    n_workers = args.workers or min(cpu_count(), 8)

    # Header
    console.print(Panel.fit(
        "[bold cyan]Decision Transformer Training Data Builder[/bold cyan]\n"
        f"Input: {args.input}\n"
        f"Output: {args.output}\n"
        f"Min reduction: {args.min_reduction:.0%}\n"
        f"Workers: {n_workers}",
        border_style="cyan"
    ))

    # Step 1: Load RTG cache
    console.print("\n[bold]Step 1:[/bold] Loading RTG cache...")
    start = time.time()

    with open(args.input, 'rb') as f:
        trajectories = pickle.load(f)

    console.print(f"  Loaded [green]{len(trajectories)}[/green] trajectories in {time.time()-start:.1f}s")

    # Step 2: Filter by minimum reduction
    console.print(f"\n[bold]Step 2:[/bold] Filtering (min {args.min_reduction:.0%} reduction)...")

    filtered = []
    stats = {'total': len(trajectories), 'passed': 0, 'no_crossings': 0, 'low_reduction': 0, 'no_ops': 0}

    for traj in trajectories:
        initial = traj.get('initial_crossings', traj['crossings_sequence'][0])
        final = traj.get('final_crossings', traj['crossings_sequence'][-1])

        if initial == 0:
            stats['no_crossings'] += 1
            continue

        if len(traj.get('sequence_ops', [])) == 0:
            stats['no_ops'] += 1
            continue

        reduction_ratio = (initial - final) / initial
        if reduction_ratio < args.min_reduction:
            stats['low_reduction'] += 1
            continue

        stats['passed'] += 1
        filtered.append(traj)

    filter_table = Table(title="Filtering Results", box=box.ROUNDED)
    filter_table.add_column("Category", style="cyan")
    filter_table.add_column("Count", style="green", justify="right")
    filter_table.add_row("Total trajectories", str(stats['total']))
    filter_table.add_row("Passed filter", str(stats['passed']))
    filter_table.add_row("No initial crossings", str(stats['no_crossings']))
    filter_table.add_row("Low reduction (<{:.0%})".format(args.min_reduction), str(stats['low_reduction']))
    filter_table.add_row("No operations", str(stats['no_ops']))
    console.print(filter_table)

    # Step 3: Process trajectories with multiprocessing
    console.print(f"\n[bold]Step 3:[/bold] Extracting effective ops & precomputing states ({n_workers} workers)...")

    traj_data = list(enumerate(filtered))

    results = []
    failed = 0
    total_effective = 0
    total_condensed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing trajectories", total=len(traj_data))

        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(process_trajectory, traj_data, chunksize=10):
                if result is not None:
                    results.append(result)
                    total_effective += result['num_effective']
                    total_condensed += result['num_condensed_steps']
                else:
                    failed += 1
                progress.advance(task)

    # Sort by original index
    results.sort(key=lambda x: x['traj_idx'])

    # Step 4: Display statistics
    console.print(f"\n[bold]Step 4:[/bold] Summary statistics")

    effective_counts = [r['num_effective'] for r in results]
    condensed_counts = [r['num_condensed_steps'] for r in results]
    reductions = [(r['initial_crossings'] - r['final_crossings']) / max(r['initial_crossings'], 1)
                  for r in results]

    stats_table = Table(title="Dataset Statistics", box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green", justify="right")
    stats_table.add_row("Trajectories processed", str(len(results)))
    stats_table.add_row("Failed/skipped", str(failed))
    stats_table.add_row("Total effective ops", str(total_effective))
    stats_table.add_row("Total condensed steps", str(total_condensed))
    stats_table.add_row("Avg effective ops/traj", f"{np.mean(effective_counts):.1f}")
    stats_table.add_row("Avg condensed steps/traj", f"{np.mean(condensed_counts):.1f}")
    stats_table.add_row("Max effective ops", str(max(effective_counts)))
    stats_table.add_row("Avg reduction ratio", f"{np.mean(reductions):.1%}")
    stats_table.add_row("Grid sizes", str(sorted(set(f"{r['grid_w']}x{r['grid_h']}" for r in results))))
    stats_table.add_row("Zone patterns", str(sorted(set(r['zone_pattern'] for r in results))))
    console.print(stats_table)

    # Estimate memory
    state_bytes = total_condensed * 4 * MAX_GRID_SIZE * MAX_GRID_SIZE * 4  # float32
    console.print(f"\n  Estimated data size: [yellow]{state_bytes / 1e9:.2f} GB[/yellow]")

    # Step 5: Save
    console.print(f"\n[bold]Step 5:[/bold] Saving to {args.output}...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'trajectories': results,
            'variant_map': VARIANT_MAP,
            'op_type_map': OP_TYPE_MAP,
            'max_grid_size': MAX_GRID_SIZE,
            'config': {
                'min_reduction': args.min_reduction,
                'n_context': 3,
                'source': str(args.input),
            }
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = output_path.stat().st_size
    console.print(f"  Saved: [green]{file_size / 1e6:.1f} MB[/green]")

    console.print(Panel.fit(
        f"[bold green]Done![/bold green]\n"
        f"Trajectories: {len(results)}\n"
        f"Effective ops: {total_effective}\n"
        f"Condensed steps: {total_condensed}\n"
        f"Output: {args.output}",
        border_style="green"
    ))


if __name__ == '__main__':
    main()
