"""
Build U-Net training data from combined_dataset.jsonl.

Replays each trajectory through operations.py to:
1. Track crossings at each step
2. Identify effective operations (ones that reduce crossings)
3. Capture the grid state BEFORE each effective operation
4. Save as unet_data.pkl for training

Usage:
    PYTHONPATH=$(pwd):$PYTHONPATH python model/unet/build_unet_data.py
    PYTHONPATH=$(pwd):$PYTHONPATH python model/unet/build_unet_data.py --input combined_dataset.jsonl --min-reduction 0.0
"""

import json
import pickle
import argparse
import time
import sys
import numpy as np
import torch
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

MAX_GRID_SIZE = 32

VARIANT_MAP = {
    'nl': 0, 'nr': 1, 'sl': 2, 'sr': 3, 'eb': 4, 'ea': 5, 'wa': 6, 'wb': 7,
    'n': 8, 's': 9, 'e': 10, 'w': 11
}
OP_TYPE_MAP = {'N': 0, 'T': 1, 'F': 2}


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

    max_zone = max(zones.max(), 1)
    state[0, :grid_h, :grid_w] = torch.from_numpy(zones.astype(np.float32)) / max_zone

    for y in range(grid_h):
        for x in range(grid_w - 1):
            state[1, y, x] = float(H_edges[y][x])

    for y in range(grid_h - 1):
        for x in range(grid_w):
            state[2, y, x] = float(V_edges[y][x])

    state[3] = boundary_mask.clone()
    state[3, :grid_h, :grid_w] = torch.maximum(
        state[3, :grid_h, :grid_w],
        torch.ones(grid_h, grid_w) * 0.5
    )

    return state


def compute_crossings(H_edges, V_edges, zones: np.ndarray, grid_w: int, grid_h: int) -> int:
    """Count zone boundary crossings."""
    crossings = 0
    for y in range(grid_h):
        for x in range(grid_w - 1):
            if H_edges[y][x]:
                if zones[y, x] != zones[y, x + 1]:
                    crossings += 1
    for y in range(grid_h - 1):
        for x in range(grid_w):
            if V_edges[y][x]:
                if zones[y, x] != zones[y + 1, x]:
                    crossings += 1
    return crossings


def process_trajectory(args):
    """Process a single trajectory: replay and extract effective ops with states."""
    from operations import HamiltonianSTL

    traj_idx, traj, min_reduction = args

    grid_w = traj.get('grid_W', 30)
    grid_h = traj.get('grid_H', 30)

    zone_grid = traj['zone_grid']
    if isinstance(zone_grid, list):
        zones = np.array(zone_grid).reshape(grid_h, grid_w)
    else:
        zones = np.array(zone_grid)

    boundary_mask = compute_boundary_mask(zones)
    ops = traj['sequence_ops']

    # Filter by minimum reduction
    initial = traj['initial_crossings']
    final = traj['final_crossings']
    if initial > 0 and (initial - final) / initial < min_reduction:
        return None

    # Replay trajectory, tracking crossings at each step
    h = HamiltonianSTL(grid_w, grid_h, init_pattern='zigzag')

    effective_steps = []
    prev_crossings = compute_crossings(h.H, h.V, zones, grid_w, grid_h)

    for step_idx, op in enumerate(ops):
        kind = op['kind']
        if kind == 'N':
            continue

        # Capture state BEFORE applying this operation
        state_before = encode_state(zones, boundary_mask, h.H, h.V, grid_w, grid_h)
        crossings_before = prev_crossings

        # Apply operation
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
        except Exception:
            continue

        crossings_after = compute_crossings(h.H, h.V, zones, grid_w, grid_h)

        # Check if effective (reduces crossings)
        if crossings_after < crossings_before:
            action_class = VARIANT_MAP.get(variant, 0)
            effective_steps.append({
                'state': state_before,
                'action': [OP_TYPE_MAP.get(kind, 0), x, y, action_class],
                'is_effective': True,
                'crossings_before': crossings_before,
                'crossings_after': crossings_after,
            })

        prev_crossings = crossings_after

    if len(effective_steps) == 0:
        return None

    return {
        'traj_idx': traj_idx,
        'grid_w': grid_w,
        'grid_h': grid_h,
        'zone_pattern': traj.get('zone_pattern', 'unknown'),
        'initial_crossings': initial,
        'final_crossings': final,
        'num_effective': len(effective_steps),
        'effective_steps': effective_steps,
        'zones': zones,
    }


def main():
    parser = argparse.ArgumentParser(description='Build U-Net training data from combined_dataset.jsonl')
    parser.add_argument('--input', type=str, default='combined_dataset.jsonl',
                        help='Path to combined_dataset.jsonl')
    parser.add_argument('--output', type=str, default='model/unet/unet_data.pkl',
                        help='Output pickle path')
    parser.add_argument('--min-reduction', type=float, default=0.0,
                        help='Minimum crossing reduction ratio to include (0.0 = include all)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers (0 = auto)')
    args = parser.parse_args()

    # Load JSONL
    console.print(Panel.fit(
        f"[bold cyan]Build U-Net Training Data[/bold cyan]\n"
        f"Input: {args.input}\n"
        f"Output: {args.output}\n"
        f"Min reduction: {args.min_reduction:.0%}",
        border_style="cyan"
    ))

    console.print("\n[bold]Loading trajectories...[/bold]")
    trajectories = []
    with open(args.input) as f:
        for line in f:
            trajectories.append(json.loads(line))
    console.print(f"  Loaded {len(trajectories)} trajectories")

    # Process trajectories
    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    console.print(f"\n[bold]Processing with {n_workers} workers...[/bold]")

    task_args = [(i, t, args.min_reduction) for i, t in enumerate(trajectories)]

    results = []
    total_effective = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.fields[status]}[/cyan]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing", total=len(task_args), status="")

        if n_workers == 1:
            for ta in task_args:
                result = process_trajectory(ta)
                if result is not None:
                    results.append(result)
                    total_effective += result['num_effective']
                progress.update(task, advance=1,
                                status=f"{len(results)} trajs, {total_effective} eff ops")
        else:
            with Pool(n_workers) as pool:
                for result in pool.imap_unordered(process_trajectory, task_args, chunksize=10):
                    if result is not None:
                        results.append(result)
                        total_effective += result['num_effective']
                    progress.update(task, advance=1,
                                    status=f"{len(results)} trajs, {total_effective} eff ops")

    # Sort by trajectory index
    results.sort(key=lambda x: x['traj_idx'])

    # Statistics
    effective_counts = [r['num_effective'] for r in results]
    patterns = {}
    for r in results:
        p = r['zone_pattern']
        patterns[p] = patterns.get(p, 0) + 1

    stats = Table(title="[bold]Dataset Statistics[/bold]", box=box.ROUNDED)
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", justify="right")
    stats.add_row("Input trajectories", str(len(trajectories)))
    stats.add_row("Valid trajectories", str(len(results)))
    stats.add_row("Total effective ops", str(total_effective))
    stats.add_row("Avg effective ops/traj", f"{np.mean(effective_counts):.1f}")
    stats.add_row("Min effective ops", str(min(effective_counts)))
    stats.add_row("Max effective ops", str(max(effective_counts)))
    stats.add_row("Zone patterns", str(patterns))

    console.print(stats)

    # Save
    console.print(f"\n[bold]Saving to {args.output}...[/bold]")
    output_data = {
        'trajectories': results,
        'variant_map': VARIANT_MAP,
        'op_type_map': OP_TYPE_MAP,
        'max_grid_size': MAX_GRID_SIZE,
        'source': str(args.input),
        'min_reduction': args.min_reduction,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)

    file_size = Path(args.output).stat().st_size / 1e9
    console.print(f"  Size: {file_size:.2f} GB")
    console.print(f"\n[bold green]Done! {total_effective} effective operations saved.[/bold green]")


if __name__ == '__main__':
    main()
