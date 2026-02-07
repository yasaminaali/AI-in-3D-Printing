"""
Precompute U-Net training data from combined_dataset.jsonl.

Replays all trajectories in parallel, extracts effective operations
with grid states, dumps flat tensors to a .pt file.

Usage:
    PYTHONPATH=$(pwd):$PYTHONPATH python model/unet/precompute_unet_data.py
    PYTHONPATH=$(pwd):$PYTHONPATH python model/unet/precompute_unet_data.py --input combined_dataset.jsonl --workers 0
"""

import json
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MAX_GRID_SIZE = 32

VARIANT_MAP = {
    'nl': 0, 'nr': 1, 'sl': 2, 'sr': 3, 'eb': 4, 'ea': 5, 'wa': 6, 'wb': 7,
    'n': 8, 's': 9, 'e': 10, 'w': 11
}
OP_TYPE_MAP = {'N': 0, 'T': 1, 'F': 2}


def _compute_boundary_mask(zones):
    h, w = zones.shape
    mask = torch.zeros(MAX_GRID_SIZE, MAX_GRID_SIZE)
    zt = torch.from_numpy(zones.astype(np.float32))
    if w > 1:
        diff_h = (zt[:, :-1] != zt[:, 1:]).float()
        mask[:h, :w-1] = torch.maximum(mask[:h, :w-1], diff_h)
        mask[:h, 1:w] = torch.maximum(mask[:h, 1:w], diff_h)
    if h > 1:
        diff_v = (zt[:-1, :] != zt[1:, :]).float()
        mask[:h-1, :w] = torch.maximum(mask[:h-1, :w], diff_v)
        mask[1:h, :w] = torch.maximum(mask[1:h, :w], diff_v)
    return mask


def _encode_state(zones, boundary_mask, H_edges, V_edges, grid_w, grid_h):
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


def _compute_crossings(H_edges, V_edges, zones, grid_w, grid_h):
    crossings = 0
    for y in range(grid_h):
        for x in range(grid_w - 1):
            if H_edges[y][x] and zones[y, x] != zones[y, x + 1]:
                crossings += 1
    for y in range(grid_h - 1):
        for x in range(grid_w):
            if V_edges[y][x] and zones[y, x] != zones[y + 1, x]:
                crossings += 1
    return crossings


def process_trajectory(args):
    """Replay one trajectory, return (traj_idx, states, targets) or None."""
    from operations import HamiltonianSTL

    traj_idx, traj_json = args
    traj = json.loads(traj_json)

    grid_w = traj.get('grid_W', 30)
    grid_h = traj.get('grid_H', 30)
    zone_grid = traj['zone_grid']
    zones = np.array(zone_grid).reshape(grid_h, grid_w)

    boundary_mask = _compute_boundary_mask(zones)
    h = HamiltonianSTL(grid_w, grid_h, init_pattern='zigzag')
    prev_crossings = _compute_crossings(h.H, h.V, zones, grid_w, grid_h)

    states = []
    targets = []

    for op in traj['sequence_ops']:
        kind = op['kind']
        if kind == 'N':
            continue

        x, y, variant = op['x'], op['y'], op['variant']

        # Capture state BEFORE applying (only encoded if effective — checked after)
        # We snapshot H/V cheaply, apply, check crossings, encode only if needed
        H_snap = [row[:] for row in h.H]
        V_snap = [row[:] for row in h.V]

        # Apply edge diff directly — skip validation (ops are known-good from SA)
        try:
            if kind == 'T':
                sub = h.get_subgrid((x, y), (x + 2, y + 2))
                pattern = h.transpose_patterns[variant]
            elif kind == 'F':
                if variant in ['n', 's']:
                    sub = h.get_subgrid((x, y), (x + 1, y + 2))
                else:
                    sub = h.get_subgrid((x, y), (x + 2, y + 1))
                pattern = h.flip_patterns[variant]
            else:
                continue
            h._apply_edge_diff_in_subgrid(sub, pattern['old'], pattern['new'])
        except Exception:
            h.H = H_snap
            h.V = V_snap
            continue

        crossings_after = _compute_crossings(h.H, h.V, zones, grid_w, grid_h)

        if crossings_after < prev_crossings:
            # Now encode the state from BEFORE this op
            state_before = _encode_state(zones, boundary_mask, H_snap, V_snap, grid_w, grid_h)
            states.append(state_before)
            targets.append(torch.tensor([
                min(y, MAX_GRID_SIZE - 1),
                min(x, MAX_GRID_SIZE - 1),
                min(VARIANT_MAP.get(variant, 0), 11),
            ], dtype=torch.long))

        prev_crossings = crossings_after

    if not states:
        return None

    return (traj_idx, torch.stack(states), torch.stack(targets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='combined_dataset.jsonl')
    parser.add_argument('--output', default='model/unet/unet_data.pt')
    parser.add_argument('--workers', type=int, default=0, help='0 = 90%% of cores')
    parser.add_argument('--limit', type=int, default=0, help='Process only first N trajectories (0 = all)')
    args = parser.parse_args()

    print(f"Precompute U-Net data")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")

    with open(args.input) as f:
        lines = f.readlines()
    if args.limit > 0:
        lines = lines[:args.limit]
    print(f"  Trajectories: {len(lines)}")

    n_workers = args.workers if args.workers > 0 else max(1, int(cpu_count() * 0.3))
    print(f"  Workers: {n_workers}")

    task_args = [(i, line) for i, line in enumerate(lines)]

    all_states = []
    all_targets = []
    all_traj_ids = []
    n_trajs = 0

    with Pool(n_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_trajectory, task_args, chunksize=2),
                           total=len(lines), desc="Replaying trajectories", unit="traj"):
            if result is not None:
                traj_idx, states, targets = result
                all_states.append(states)
                all_targets.append(targets)
                all_traj_ids.extend([n_trajs] * len(states))
                n_trajs += 1

    print(f"\nConcatenating tensors...")
    states_tensor = torch.cat(all_states, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    traj_ids_tensor = torch.tensor(all_traj_ids, dtype=torch.long)

    print(f"  Total samples: {states_tensor.size(0)}")
    print(f"  Valid trajectories: {n_trajs}")
    print(f"  States: {states_tensor.shape}  Targets: {targets_tensor.shape}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'states': states_tensor,
        'targets': targets_tensor,
        'traj_ids': traj_ids_tensor,
        'n_trajectories': n_trajs,
    }, args.output)

    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"  Saved: {args.output} ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == '__main__':
    main()
