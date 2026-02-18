"""
Precompute Fusion training data v2 from combined_dataset.jsonl.

Key fixes from v1:
- Change 16: Infer correct init_pattern from zone_pattern (was hardcoded 'zigzag')
- Change 16: Use safe transpose_subgrid/flip_subgrid with status checking
- Change 1:  MAX_GRID_SIZE=128
- Change 7:  Save initial_crossings per sample for proper normalization
- Change 15: Position balancing (oversample inner layers)
- Change 17: Compact grouped storage by grid size (float16, ~1.7GB)
- Change 5:  4-channel state (zones, H, V, boundary_combined) — derived channels on-the-fly

Output: fusion_data.pt with grouped storage:
  {
    'grid_groups': {
      '30x30': {
        'states': [N, 4, 30, 30] float16,
        'targets': [N, 3] int16,
        'traj_ids': [N] int32,
        'initial_crossings': [N] int16,
        'history_actions': [N, K] int8,
        'history_positions_y': [N, K] int8,
        'history_positions_x': [N, K] int8,
        'history_crossings_before': [N, K] float16,
        'history_crossings_after': [N, K] float16,
        'history_lengths': [N] int8,
        'grid_w': 30, 'grid_h': 30,
      },
      ...
    },
    'n_trajectories': int,
    'max_history': int,
  }

Usage:
    PYTHONPATH=$(pwd)/src:$PYTHONPATH python src/model/build_fusion_data.py
    PYTHONPATH=$(pwd)/src:$PYTHONPATH python src/model/build_fusion_data.py --input combined_dataset.jsonl
"""

import json
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import deque, defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, MofNCompleteColumn,
)

console = Console()

MAX_GRID_SIZE = 128
MAX_HISTORY = 32

VARIANT_MAP = {
    'nl': 0, 'nr': 1, 'sl': 2, 'sr': 3, 'eb': 4, 'ea': 5, 'wa': 6, 'wb': 7,
    'n': 8, 's': 9, 'e': 10, 'w': 11
}


# ---------------------------------------------------------------------------
# Change 16: Infer correct init_pattern from zone_pattern
# ---------------------------------------------------------------------------

def _infer_init_pattern(zone_pattern, zone_grid, grid_w, grid_h):
    """All existing dataset was generated with default zigzag (horizontal).

    The SA code at the time of dataset generation used HamiltonianSTL(w, h)
    without specifying init_pattern, which defaults to 'zigzag'. The 'auto'
    logic was added later but the dataset was never regenerated.
    """
    return 'zigzag'


def _compute_boundary_mask(zones, grid_w, grid_h):
    """Compute zone boundary mask at natural resolution. Returns numpy array."""
    mask = np.zeros((grid_h, grid_w), dtype=np.float32)
    zf = zones.astype(np.float32)
    if grid_w > 1:
        diff_h = (zf[:, :-1] != zf[:, 1:]).astype(np.float32)
        mask[:, :grid_w - 1] = np.maximum(mask[:, :grid_w - 1], diff_h)
        mask[:, 1:grid_w] = np.maximum(mask[:, 1:grid_w], diff_h)
    if grid_h > 1:
        diff_v = (zf[:-1, :] != zf[1:, :]).astype(np.float32)
        mask[:grid_h - 1, :] = np.maximum(mask[:grid_h - 1, :], diff_v)
        mask[1:grid_h, :] = np.maximum(mask[1:grid_h, :], diff_v)
    return mask


def _encode_state(zones, boundary_mask, H_edges, V_edges, grid_w, grid_h):
    """Encode state as 4 channels at natural grid resolution (no padding to 128).
    Returns numpy array (not torch) to avoid IPC issues with multiprocessing.Pool.
    """
    state = np.zeros((4, grid_h, grid_w), dtype=np.float32)
    max_zone = max(zones.max(), 1)
    state[0, :grid_h, :grid_w] = zones.astype(np.float32) / max_zone

    H_arr = np.array(H_edges, dtype=np.float32)
    state[1, :grid_h, :grid_w - 1] = H_arr[:grid_h, :grid_w - 1]

    V_arr = np.array(V_edges, dtype=np.float32)
    state[2, :grid_h - 1, :grid_w] = V_arr[:grid_h - 1, :grid_w]

    # Channel 3: boundary_combined (0.5 inside grid, 1.0 at boundary)
    state[3, :grid_h, :grid_w] = 0.5  # inside grid
    state[3, :grid_h, :grid_w] = np.maximum(
        state[3, :grid_h, :grid_w],
        boundary_mask[:grid_h, :grid_w]
    )
    return state


def _compute_crossings(H_edges, V_edges, zones, grid_w, grid_h):
    H_arr = np.array(H_edges, dtype=np.float32)[:grid_h, :grid_w - 1]
    h_cross = H_arr * (zones[:grid_h, :grid_w - 1] != zones[:grid_h, 1:grid_w]).astype(np.float32)
    V_arr = np.array(V_edges, dtype=np.float32)[:grid_h - 1, :grid_w]
    v_cross = V_arr * (zones[:grid_h - 1, :grid_w] != zones[1:grid_h, :grid_w]).astype(np.float32)
    return int(h_cross.sum() + v_cross.sum())


def process_trajectory(args):
    """Replay one trajectory using safe methods. Return samples or None."""
    from operations import HamiltonianSTL

    traj_idx, traj_json = args
    traj = json.loads(traj_json)

    grid_w = traj.get('grid_W', 30)
    grid_h = traj.get('grid_H', 30)
    zone_grid = traj.get('zone_grid')
    if zone_grid is None or len(zone_grid) != grid_w * grid_h:
        return None  # GA records lack zone_grid; skip
    zone_pattern = traj.get('zone_pattern', 'unknown')
    zones = np.array(zone_grid).reshape(grid_h, grid_w)

    boundary_mask = _compute_boundary_mask(zones, grid_w, grid_h)

    # Change 16: Infer correct init_pattern
    init_pattern = _infer_init_pattern(zone_pattern, zone_grid, grid_w, grid_h)
    h = HamiltonianSTL(grid_w, grid_h, init_pattern=init_pattern)
    initial_crossings = _compute_crossings(h.H, h.V, zones, grid_w, grid_h)
    prev_crossings = initial_crossings

    states = []
    targets = []
    sample_initial_crossings = []

    # History buffer
    hist_actions = []
    hist_positions_y = []
    hist_positions_x = []
    hist_crossings_before = []
    hist_crossings_after = []
    hist_lengths = []

    history_buffer = deque(maxlen=MAX_HISTORY)

    for op in traj['sequence_ops']:
        kind = op['kind']
        if kind == 'N':
            continue

        x, y, variant = op['x'], op['y'], op['variant']

        # Snapshot BEFORE
        H_snap = [row[:] for row in h.H]
        V_snap = [row[:] for row in h.V]

        # Apply using safe methods
        try:
            if kind == 'T':
                sub = h.get_subgrid((x, y), (x + 2, y + 2))
                sub, status = h.transpose_subgrid(sub, variant)
                if 'transposed_' not in status:
                    continue
            elif kind == 'F':
                if variant in ['n', 's']:
                    sub = h.get_subgrid((x, y), (x + 1, y + 2))
                else:
                    sub = h.get_subgrid((x, y), (x + 2, y + 1))
                sub, status = h.flip_subgrid(sub, variant)
                if 'flipped_' not in status:
                    continue
            else:
                continue
        except Exception:
            h.H = H_snap
            h.V = V_snap
            continue

        crossings_after = _compute_crossings(h.H, h.V, zones, grid_w, grid_h)

        if crossings_after < prev_crossings:
            # Encode state from BEFORE this effective op (using snapshots)
            state_before = _encode_state(zones, boundary_mask, H_snap, V_snap, grid_w, grid_h)
            states.append(state_before)
            targets.append(np.array([
                min(y, grid_h - 1),
                min(x, grid_w - 1),
                min(VARIANT_MAP.get(variant, 0), 11),
            ], dtype=np.int64))
            sample_initial_crossings.append(initial_crossings)

            # Save current history snapshot
            cur_len = len(history_buffer)
            h_act = np.zeros(MAX_HISTORY, dtype=np.int64)
            h_py = np.zeros(MAX_HISTORY, dtype=np.int64)
            h_px = np.zeros(MAX_HISTORY, dtype=np.int64)
            h_cb = np.zeros(MAX_HISTORY, dtype=np.float32)
            h_ca = np.zeros(MAX_HISTORY, dtype=np.float32)

            for i, entry in enumerate(history_buffer):
                h_act[i] = entry['action']
                h_py[i] = entry['py']
                h_px[i] = entry['px']
                h_cb[i] = entry['cb']
                h_ca[i] = entry['ca']

            hist_actions.append(h_act)
            hist_positions_y.append(h_py)
            hist_positions_x.append(h_px)
            hist_crossings_before.append(h_cb)
            hist_crossings_after.append(h_ca)
            hist_lengths.append(cur_len)

            # Add to history buffer for future samples
            history_buffer.append({
                'action': VARIANT_MAP.get(variant, 0),
                'py': min(y, grid_h - 1),
                'px': min(x, grid_w - 1),
                'cb': prev_crossings,
                'ca': crossings_after,
            })

        prev_crossings = crossings_after

    if not states:
        return None

    # Return numpy arrays — torch tensors fail with multiprocessing.Pool IPC
    return (
        traj_idx,
        grid_w,
        grid_h,
        np.stack(states),
        np.stack(targets),
        np.array(sample_initial_crossings, dtype=np.int64),
        np.stack(hist_actions),
        np.stack(hist_positions_y),
        np.stack(hist_positions_x),
        np.stack(hist_crossings_before),
        np.stack(hist_crossings_after),
        np.array(hist_lengths, dtype=np.int64),
    )


def _compute_layer_index(y, x, grid_h, grid_w):
    """Compute distance from boundary (layer 0 = outermost)."""
    return min(y, x, grid_h - 1 - y, grid_w - 1 - x)


def main():
    parser = argparse.ArgumentParser(description='Build Fusion training data v2')
    parser.add_argument('--input', default='datasets/final_dataset.jsonl')
    parser.add_argument('--output', default='checkpoints/fusion_data.pt')
    parser.add_argument('--workers', type=int, default=0, help='0 = 30%% of cores')
    parser.add_argument('--limit', type=int, default=0, help='Process only first N trajectories (0 = all)')
    parser.add_argument('--max_oversample', type=int, default=4, help='Max oversampling factor for inner layers')
    args = parser.parse_args()

    console.print(f"[bold cyan]Build Fusion Training Data v2[/bold cyan]")
    console.print(f"  Input:  {args.input}")
    console.print(f"  Output: {args.output}")
    console.print(f"  MAX_GRID_SIZE: {MAX_GRID_SIZE}")
    console.print(f"  MAX_HISTORY: {MAX_HISTORY}")
    console.print(f"  Compact grouped storage: float16 by grid size")

    with open(args.input) as f:
        lines = f.readlines()
    if args.limit > 0:
        lines = lines[:args.limit]
    console.print(f"  Trajectories: {len(lines)}")

    n_workers = args.workers if args.workers > 0 else max(1, int(cpu_count() * 0.3))
    console.print(f"  Workers: {n_workers}")

    task_args = [(i, line) for i, line in enumerate(lines)]

    # Collect results grouped by grid size
    grid_groups = defaultdict(lambda: {
        'states': [], 'targets': [], 'traj_ids': [],
        'initial_crossings': [],
        'hist_actions': [], 'hist_py': [], 'hist_px': [],
        'hist_cb': [], 'hist_ca': [], 'hist_lengths': [],
    })

    n_trajs = 0
    total_effective = 0
    pattern_counts = Counter()
    init_pattern_counts = Counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.fields[status]}[/cyan]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        ptask = progress.add_task(
            "Replaying trajectories", total=len(lines),
            status="0 trajectories, 0 effective ops"
        )

        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(process_trajectory, task_args, chunksize=2):
                if result is not None:
                    (traj_idx, gw, gh, states_np, targets_np, init_cross_np,
                     h_act_np, h_py_np, h_px_np, h_cb_np, h_ca_np, h_len_np) = result

                    # Convert numpy -> torch on main process (avoids IPC fd issues)
                    states = torch.from_numpy(states_np)
                    targets = torch.from_numpy(targets_np)
                    init_cross = torch.from_numpy(init_cross_np)
                    h_act = torch.from_numpy(h_act_np)
                    h_py = torch.from_numpy(h_py_np)
                    h_px = torch.from_numpy(h_px_np)
                    h_cb = torch.from_numpy(h_cb_np)
                    h_ca = torch.from_numpy(h_ca_np)
                    h_len = torch.from_numpy(h_len_np)

                    n_samples = len(states)
                    key = f"{gw}x{gh}"
                    group = grid_groups[key]
                    group['states'].append(states)
                    group['targets'].append(targets)
                    group['traj_ids'].extend([n_trajs] * n_samples)
                    group['initial_crossings'].append(init_cross)
                    group['hist_actions'].append(h_act)
                    group['hist_py'].append(h_py)
                    group['hist_px'].append(h_px)
                    group['hist_cb'].append(h_cb)
                    group['hist_ca'].append(h_ca)
                    group['hist_lengths'].append(h_len)
                    group['grid_w'] = gw
                    group['grid_h'] = gh

                    n_trajs += 1
                    total_effective += n_samples
                    pattern_counts[key] += n_samples

                progress.update(ptask, advance=1,
                                status=f"{n_trajs} trajs, {total_effective} ops")

    console.print(f"\n[bold]Grid size distribution:[/bold]")
    for key, count in sorted(pattern_counts.items()):
        console.print(f"  {key}: {count} samples")

    # --- Change 15: Position balancing (oversample inner layers) ---
    console.print(f"\n[bold]Applying position balancing (Change 15)...[/bold]")

    for key, group in grid_groups.items():
        if not group['states']:
            continue

        all_targets = torch.cat(group['targets'], dim=0)
        gw = group['grid_w']
        gh = group['grid_h']

        # Compute layer index for each sample
        layers = []
        for i in range(len(all_targets)):
            ty, tx = all_targets[i, 0].item(), all_targets[i, 1].item()
            layer = _compute_layer_index(ty, tx, gh, gw)
            layers.append(layer)
        layers = np.array(layers)

        layer_counts = Counter(layers.tolist())
        if len(layer_counts) <= 1:
            continue

        max_count = max(layer_counts.values())
        # Compute oversampling indices
        oversample_indices = list(range(len(all_targets)))  # start with all

        for layer_val, count in layer_counts.items():
            if count < max_count:
                ratio = min(args.max_oversample, max_count / max(count, 1))
                n_extra = int(count * (ratio - 1))
                layer_indices = np.where(layers == layer_val)[0]
                if len(layer_indices) > 0 and n_extra > 0:
                    extra = np.random.choice(layer_indices, size=n_extra, replace=True)
                    oversample_indices.extend(extra.tolist())

        n_before = len(all_targets)
        n_after = len(oversample_indices)
        if n_after > n_before:
            console.print(f"  {key}: {n_before} -> {n_after} samples "
                          f"(+{n_after - n_before} from oversampling inner layers)")

        # Apply oversampling to all tensors
        idx_tensor = torch.tensor(oversample_indices, dtype=torch.long)
        group['_oversample_idx'] = idx_tensor

    # --- Concatenate and save with compact grouped storage ---
    console.print(f"\n[bold]Building compact grouped storage...[/bold]")

    save_groups = {}
    total_samples_final = 0

    for key, group in grid_groups.items():
        if not group['states']:
            continue

        states_cat = torch.cat(group['states'], dim=0)
        targets_cat = torch.cat(group['targets'], dim=0)
        traj_ids_cat = torch.tensor(group['traj_ids'], dtype=torch.int32)
        init_cross_cat = torch.cat(group['initial_crossings'], dim=0)
        h_act_cat = torch.cat(group['hist_actions'], dim=0)
        h_py_cat = torch.cat(group['hist_py'], dim=0)
        h_px_cat = torch.cat(group['hist_px'], dim=0)
        h_cb_cat = torch.cat(group['hist_cb'], dim=0)
        h_ca_cat = torch.cat(group['hist_ca'], dim=0)
        h_len_cat = torch.cat(group['hist_lengths'], dim=0)

        # Apply oversampling if computed
        if '_oversample_idx' in group:
            idx = group['_oversample_idx']
            states_cat = states_cat[idx]
            targets_cat = targets_cat[idx]
            traj_ids_cat = traj_ids_cat[idx]
            init_cross_cat = init_cross_cat[idx]
            h_act_cat = h_act_cat[idx]
            h_py_cat = h_py_cat[idx]
            h_px_cat = h_px_cat[idx]
            h_cb_cat = h_cb_cat[idx]
            h_ca_cat = h_ca_cat[idx]
            h_len_cat = h_len_cat[idx]

        n = len(states_cat)
        total_samples_final += n

        save_groups[key] = {
            'states': states_cat.half(),  # float16
            'targets': targets_cat.short(),  # int16
            'traj_ids': traj_ids_cat,  # int32
            'initial_crossings': init_cross_cat.short(),  # int16
            'history_actions': h_act_cat.byte(),  # uint8
            'history_positions_y': h_py_cat.byte(),
            'history_positions_x': h_px_cat.byte(),
            'history_crossings_before': h_cb_cat.half(),
            'history_crossings_after': h_ca_cat.half(),
            'history_lengths': h_len_cat.byte(),
            'grid_w': group['grid_w'],
            'grid_h': group['grid_h'],
        }

        console.print(f"  {key}: {n} samples, states shape {states_cat.shape}")

    console.print(f"\n  Total samples (after balancing): [green]{total_samples_final}[/green]")
    console.print(f"  Valid trajectories: [green]{n_trajs}[/green]")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'grid_groups': save_groups,
        'n_trajectories': n_trajs,
        'max_history': MAX_HISTORY,
    }, args.output)

    size_mb = Path(args.output).stat().st_size / 1e6
    console.print(f"\n  Saved: [bold]{args.output}[/bold] ({size_mb:.1f} MB)")
    console.print("[bold green]Done.[/bold green]")


if __name__ == '__main__':
    main()
