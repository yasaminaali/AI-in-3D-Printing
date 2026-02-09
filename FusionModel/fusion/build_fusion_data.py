"""
Precompute Fusion training data from combined_dataset.jsonl.

Replays ALL operations in each trajectory (preserving setup context),
but only SAVES state snapshots before effective operations (those that
reduce crossings). Maintains a rolling history buffer (K=8) of recent
effective ops per trajectory.

Output: fusion_data.pt with:
  states [N, 4, 32, 32]
  targets [N, 3]            (position_y, position_x, action_class)
  traj_ids [N]
  history_actions [N, K]
  history_positions_y [N, K]
  history_positions_x [N, K]
  history_crossings_before [N, K]
  history_crossings_after [N, K]
  history_lengths [N]

Usage:
    PYTHONPATH=$(pwd):$PYTHONPATH python model/fusion/build_fusion_data.py
    PYTHONPATH=$(pwd):$PYTHONPATH python model/fusion/build_fusion_data.py --input combined_dataset.jsonl
"""

import json
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, MofNCompleteColumn,
)

console = Console()

MAX_GRID_SIZE = 32
MAX_HISTORY = 8

VARIANT_MAP = {
    'nl': 0, 'nr': 1, 'sl': 2, 'sr': 3, 'eb': 4, 'ea': 5, 'wa': 6, 'wb': 7,
    'n': 8, 's': 9, 'e': 10, 'w': 11
}


def _compute_boundary_mask(zones):
    h, w = zones.shape
    mask = torch.zeros(MAX_GRID_SIZE, MAX_GRID_SIZE)
    zt = torch.from_numpy(zones.astype(np.float32))
    if w > 1:
        diff_h = (zt[:, :-1] != zt[:, 1:]).float()
        mask[:h, :w - 1] = torch.maximum(mask[:h, :w - 1], diff_h)
        mask[:h, 1:w] = torch.maximum(mask[:h, 1:w], diff_h)
    if h > 1:
        diff_v = (zt[:-1, :] != zt[1:, :]).float()
        mask[:h - 1, :w] = torch.maximum(mask[:h - 1, :w], diff_v)
        mask[1:h, :w] = torch.maximum(mask[1:h, :w], diff_v)
    return mask


def _encode_state(zones, boundary_mask, H_edges, V_edges, grid_w, grid_h):
    state = torch.zeros(4, MAX_GRID_SIZE, MAX_GRID_SIZE)
    max_zone = max(zones.max(), 1)
    state[0, :grid_h, :grid_w] = torch.from_numpy(zones.astype(np.float32)) / max_zone
    H_arr = np.array(H_edges, dtype=np.float32)
    state[1, :grid_h, :grid_w - 1] = torch.from_numpy(H_arr[:grid_h, :grid_w - 1])
    V_arr = np.array(V_edges, dtype=np.float32)
    state[2, :grid_h - 1, :grid_w] = torch.from_numpy(V_arr[:grid_h - 1, :grid_w])
    state[3] = boundary_mask.clone()
    state[3, :grid_h, :grid_w] = torch.maximum(
        state[3, :grid_h, :grid_w],
        torch.ones(grid_h, grid_w) * 0.5
    )
    return state


def _compute_crossings(H_edges, V_edges, zones, grid_w, grid_h):
    H_arr = np.array(H_edges, dtype=np.float32)[:grid_h, :grid_w - 1]
    h_cross = H_arr * (zones[:grid_h, :grid_w - 1] != zones[:grid_h, 1:grid_w]).astype(np.float32)
    V_arr = np.array(V_edges, dtype=np.float32)[:grid_h - 1, :grid_w]
    v_cross = V_arr * (zones[:grid_h - 1, :grid_w] != zones[1:grid_h, :grid_w]).astype(np.float32)
    return int(h_cross.sum() + v_cross.sum())


def process_trajectory(args):
    """Replay one trajectory, return (traj_idx, states, targets, history_*) or None."""
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

    # History buffer: rolling window of recent effective ops
    hist_actions = []
    hist_positions_y = []
    hist_positions_x = []
    hist_crossings_before = []
    hist_crossings_after = []
    hist_lengths = []

    # Current rolling history (deque of recent effective ops)
    history_buffer = deque(maxlen=MAX_HISTORY)

    for op in traj['sequence_ops']:
        kind = op['kind']
        if kind == 'N':
            continue

        x, y, variant = op['x'], op['y'], op['variant']

        # Snapshot edges BEFORE applying
        H_snap = [row[:] for row in h.H]
        V_snap = [row[:] for row in h.V]

        # Apply the operation
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
            # Encode state from BEFORE this effective op
            state_before = _encode_state(zones, boundary_mask, H_snap, V_snap, grid_w, grid_h)
            states.append(state_before)
            targets.append(torch.tensor([
                min(y, MAX_GRID_SIZE - 1),
                min(x, MAX_GRID_SIZE - 1),
                min(VARIANT_MAP.get(variant, 0), 11),
            ], dtype=torch.long))

            # Save current history snapshot for this sample
            cur_len = len(history_buffer)
            h_act = torch.zeros(MAX_HISTORY, dtype=torch.long)
            h_py = torch.zeros(MAX_HISTORY, dtype=torch.long)
            h_px = torch.zeros(MAX_HISTORY, dtype=torch.long)
            h_cb = torch.zeros(MAX_HISTORY, dtype=torch.float)
            h_ca = torch.zeros(MAX_HISTORY, dtype=torch.float)

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

            # NOW add this effective op to the history buffer (for future samples)
            history_buffer.append({
                'action': VARIANT_MAP.get(variant, 0),
                'py': min(y, MAX_GRID_SIZE - 1),
                'px': min(x, MAX_GRID_SIZE - 1),
                'cb': prev_crossings,
                'ca': crossings_after,
            })

        prev_crossings = crossings_after

    if not states:
        return None

    return (
        traj_idx,
        torch.stack(states),
        torch.stack(targets),
        torch.stack(hist_actions),
        torch.stack(hist_positions_y),
        torch.stack(hist_positions_x),
        torch.stack(hist_crossings_before),
        torch.stack(hist_crossings_after),
        torch.tensor(hist_lengths, dtype=torch.long),
    )


def main():
    parser = argparse.ArgumentParser(description='Build Fusion training data')
    parser.add_argument('--input', default='combined_dataset.jsonl')
    parser.add_argument('--output', default='model/fusion/fusion_data.pt')
    parser.add_argument('--workers', type=int, default=0, help='0 = 30%% of cores')
    parser.add_argument('--limit', type=int, default=0, help='Process only first N trajectories (0 = all)')
    args = parser.parse_args()

    console.print(f"[bold cyan]Build Fusion Training Data[/bold cyan]")
    console.print(f"  Input:  {args.input}")
    console.print(f"  Output: {args.output}")

    with open(args.input) as f:
        lines = f.readlines()
    if args.limit > 0:
        lines = lines[:args.limit]
    console.print(f"  Trajectories: {len(lines)}")

    n_workers = args.workers if args.workers > 0 else max(1, int(cpu_count() * 0.3))
    console.print(f"  Workers: {n_workers}")
    console.print(f"  History length: {MAX_HISTORY}")

    task_args = [(i, line) for i, line in enumerate(lines)]

    all_states = []
    all_targets = []
    all_traj_ids = []
    all_hist_actions = []
    all_hist_py = []
    all_hist_px = []
    all_hist_cb = []
    all_hist_ca = []
    all_hist_lengths = []
    n_trajs = 0
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
        ptask = progress.add_task(
            "Replaying trajectories", total=len(lines),
            status="0 trajectories, 0 effective ops"
        )

        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(process_trajectory, task_args, chunksize=2):
                if result is not None:
                    (traj_idx, states, targets,
                     h_act, h_py, h_px, h_cb, h_ca, h_len) = result

                    n_samples = len(states)
                    all_states.append(states)
                    all_targets.append(targets)
                    all_traj_ids.extend([n_trajs] * n_samples)
                    all_hist_actions.append(h_act)
                    all_hist_py.append(h_py)
                    all_hist_px.append(h_px)
                    all_hist_cb.append(h_cb)
                    all_hist_ca.append(h_ca)
                    all_hist_lengths.append(h_len)
                    n_trajs += 1
                    total_effective += n_samples

                progress.update(ptask, advance=1,
                                status=f"{n_trajs} trajectories, {total_effective} effective ops")

    console.print(f"\n[bold]Concatenating tensors...[/bold]")
    states_tensor = torch.cat(all_states, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    traj_ids_tensor = torch.tensor(all_traj_ids, dtype=torch.long)
    hist_actions_tensor = torch.cat(all_hist_actions, dim=0)
    hist_py_tensor = torch.cat(all_hist_py, dim=0)
    hist_px_tensor = torch.cat(all_hist_px, dim=0)
    hist_cb_tensor = torch.cat(all_hist_cb, dim=0)
    hist_ca_tensor = torch.cat(all_hist_ca, dim=0)
    hist_lengths_tensor = torch.cat(all_hist_lengths, dim=0)

    console.print(f"  Total samples: [green]{states_tensor.size(0)}[/green]")
    console.print(f"  Valid trajectories: [green]{n_trajs}[/green]")
    console.print(f"  States: {states_tensor.shape}")
    console.print(f"  Targets: {targets_tensor.shape}")
    console.print(f"  History actions: {hist_actions_tensor.shape}")
    console.print(f"  History lengths: min={hist_lengths_tensor.min().item()}, "
                  f"max={hist_lengths_tensor.max().item()}, "
                  f"mean={hist_lengths_tensor.float().mean().item():.1f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'states': states_tensor,
        'targets': targets_tensor,
        'traj_ids': traj_ids_tensor,
        'n_trajectories': n_trajs,
        'history_actions': hist_actions_tensor,
        'history_positions_y': hist_py_tensor,
        'history_positions_x': hist_px_tensor,
        'history_crossings_before': hist_cb_tensor,
        'history_crossings_after': hist_ca_tensor,
        'history_lengths': hist_lengths_tensor,
        'max_history': MAX_HISTORY,
    }, args.output)

    size_mb = Path(args.output).stat().st_size / 1e6
    console.print(f"\n  Saved: [bold]{args.output}[/bold] ({size_mb:.1f} MB)")
    console.print("[bold green]Done.[/bold green]")


if __name__ == '__main__':
    main()
