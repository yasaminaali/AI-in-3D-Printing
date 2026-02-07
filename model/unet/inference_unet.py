"""
OperationNet (U-Net) Inference for Hamiltonian Path Optimization.

Iterative inference:
1. Encode current grid state as 5-channel tensor
2. Forward pass → position scores + action logits
3. Select best valid (position, action) from boundary candidates
4. Apply operation via operations.py (validates Hamiltonicity)
5. Repeat until no improvement

Usage:
    PYTHONPATH=$(pwd):$PYTHONPATH python model/unet/inference_unet.py --checkpoint nn_checkpoints/unet/best.pt
    PYTHONPATH=$(pwd):$PYTHONPATH python model/unet/inference_unet.py --checkpoint best.pt --n_samples 50
"""

import torch
import torch.nn.functional as F
import pickle
import json
import copy
import argparse
import numpy as np
import sys
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operations import HamiltonianSTL
from Zones import (
    zones_left_right, zones_top_bottom, zones_diagonal,
    zones_stripes, zones_checkerboard, zones_voronoi,
)
from unet_model import OperationNet, VARIANT_REV, NUM_ACTIONS

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich import box

console = Console()

MAX_GRID_SIZE = 32
TRANSPOSE_VARIANTS = {'nl', 'nr', 'sl', 'sr', 'eb', 'ea', 'wa', 'wb'}
FLIP_VARIANTS = {'n', 's', 'e', 'w'}


def _h_edges_np(h: HamiltonianSTL) -> np.ndarray:
    """Convert h.H (list of lists) to numpy array [height, width-1]."""
    return np.array(h.H, dtype=np.float32)


def _v_edges_np(h: HamiltonianSTL) -> np.ndarray:
    """Convert h.V (list of lists) to numpy array [height-1, width]."""
    return np.array(h.V, dtype=np.float32)


def compute_crossings(h: HamiltonianSTL, zones_np: np.ndarray) -> int:
    """Count zone boundary crossings in current path (vectorized)."""
    H_arr = _h_edges_np(h)  # [height, width-1]
    V_arr = _v_edges_np(h)  # [height-1, width]
    h_cross = H_arr * (zones_np[:, :-1] != zones_np[:, 1:]).astype(np.float32)
    v_cross = V_arr * (zones_np[:-1, :] != zones_np[1:, :]).astype(np.float32)
    return int(h_cross.sum() + v_cross.sum())


def compute_boundary_mask(zones_np: np.ndarray, grid_h: int, grid_w: int) -> torch.Tensor:
    """Compute zone boundary mask (vectorized)."""
    mask = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.float32)
    h_diff = zones_np[:, :-1] != zones_np[:, 1:]  # [grid_h, grid_w-1]
    mask[:grid_h, :grid_w - 1] = np.maximum(mask[:grid_h, :grid_w - 1], h_diff)
    mask[:grid_h, 1:grid_w] = np.maximum(mask[:grid_h, 1:grid_w], h_diff)
    v_diff = zones_np[:-1, :] != zones_np[1:, :]  # [grid_h-1, grid_w]
    mask[:grid_h - 1, :grid_w] = np.maximum(mask[:grid_h - 1, :grid_w], v_diff)
    mask[1:grid_h, :grid_w] = np.maximum(mask[1:grid_h, :grid_w], v_diff)
    return torch.from_numpy(mask)


def encode_state_5ch(zones_np: np.ndarray, boundary_mask: torch.Tensor,
                     h: HamiltonianSTL, grid_w: int, grid_h: int) -> torch.Tensor:
    """Encode grid state as 5-channel tensor for OperationNet (vectorized)."""
    state = torch.zeros(5, MAX_GRID_SIZE, MAX_GRID_SIZE)

    # Channel 0: Zone IDs (normalized)
    max_zone = max(zones_np.max(), 1)
    state[0, :grid_h, :grid_w] = torch.from_numpy(zones_np.astype(np.float32)) / max_zone

    # Channel 1: Horizontal edges [height, width-1]
    H_arr = _h_edges_np(h)
    state[1, :grid_h, :grid_w - 1] = torch.from_numpy(H_arr)

    # Channel 2: Vertical edges [height-1, width]
    V_arr = _v_edges_np(h)
    state[2, :grid_h - 1, :grid_w] = torch.from_numpy(V_arr)

    # Channel 3: Boundary mask + grid mask
    state[3] = boundary_mask.clone()
    state[3, :grid_h, :grid_w] = torch.maximum(
        state[3, :grid_h, :grid_w],
        torch.ones(grid_h, grid_w) * 0.5
    )

    # Channel 4: Crossing indicator (vectorized)
    zones_t = torch.from_numpy(zones_np.astype(np.float32))
    H_t = torch.from_numpy(H_arr)
    V_t = torch.from_numpy(V_arr)
    h_cross = H_t * (zones_t[:, :-1] != zones_t[:, 1:]).float()
    v_cross = V_t * (zones_t[:-1, :] != zones_t[1:, :]).float()
    state[4, :grid_h, :grid_w - 1] = torch.maximum(state[4, :grid_h, :grid_w - 1], h_cross)
    state[4, :grid_h, 1:grid_w] = torch.maximum(state[4, :grid_h, 1:grid_w], h_cross)
    state[4, :grid_h - 1, :grid_w] = torch.maximum(state[4, :grid_h - 1, :grid_w], v_cross)
    state[4, 1:grid_h, :grid_w] = torch.maximum(state[4, 1:grid_h, :grid_w], v_cross)

    return state


def validate_action(op_type: str, x: int, y: int, variant: str,
                    grid_w: int, grid_h: int) -> bool:
    """Check if an action is valid (within bounds for the operation type)."""
    if x < 0 or y < 0:
        return False

    if op_type == 'T':
        if variant not in TRANSPOSE_VARIANTS:
            return False
        return x + 2 < grid_w and y + 2 < grid_h

    if op_type == 'F':
        if variant not in FLIP_VARIANTS:
            return False
        if variant in ['n', 's']:
            return x + 1 < grid_w and y + 2 < grid_h
        else:
            return x + 2 < grid_w and y + 1 < grid_h

    return False


def apply_op(h: HamiltonianSTL, op_type: str, x: int, y: int, variant: str) -> bool:
    """Apply operation to Hamiltonian path. Returns True if successful."""
    try:
        if op_type == 'T':
            sub = h.get_subgrid((x, y), (x + 2, y + 2))
            result = h.transpose_subgrid(sub, variant)
            return result is not None and result is not False
        elif op_type == 'F':
            if variant in ['n', 's']:
                sub = h.get_subgrid((x, y), (x + 1, y + 2))
            else:
                sub = h.get_subgrid((x, y), (x + 2, y + 1))
            result = h.flip_subgrid(sub, variant)
            return result is not None and result is not False
    except Exception:
        return False
    return False


def save_grid_state(h: HamiltonianSTL):
    """Save H/V edge matrices for rollback."""
    return (copy.deepcopy(h.H), copy.deepcopy(h.V))


def restore_grid_state(h: HamiltonianSTL, state):
    """Restore H/V edge matrices from saved state."""
    h.H, h.V = state


def dilate_mask(mask: torch.Tensor, dilation: int = 2) -> torch.Tensor:
    """Dilate boundary mask to cover operation neighborhoods."""
    if dilation <= 0:
        return mask
    kernel_size = 2 * dilation + 1
    dilated = F.max_pool2d(
        mask.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size, stride=1, padding=dilation,
    ).squeeze(0).squeeze(0)
    return dilated


def plot_cycle_on_ax(ax, h, zones_np, title):
    """Draw Hamiltonian path on a matplotlib axis (SA-style visualization)."""
    W, H = h.width, h.height
    zone_vals = sorted(set(zones_np.flatten().tolist()))
    a = zone_vals[0] if zone_vals else 0

    # Draw zone colors as image
    colors = np.zeros((H, W, 3))
    colors[zones_np[:H, :W] == a] = [0.68, 0.85, 0.90]  # lightblue
    colors[zones_np[:H, :W] != a] = [0.56, 0.93, 0.56]  # lightgreen
    ax.imshow(colors, extent=(-0.5, W - 0.5, H - 0.5, -0.5), origin='upper')

    # Draw edges vectorized via LineCollection
    from matplotlib.collections import LineCollection
    H_arr = _h_edges_np(h)
    V_arr = _v_edges_np(h)

    # Horizontal edges
    hy, hx = np.where(H_arr > 0.5)
    if len(hx) > 0:
        h_cross = zones_np[hy, hx] != zones_np[hy, hx + 1]
        h_colors = np.where(h_cross, 'red', 'black')
        h_segs = [[(hx[i], hy[i]), (hx[i] + 1, hy[i])] for i in range(len(hx))]
        ax.add_collection(LineCollection(h_segs, colors=h_colors, linewidths=2))

    # Vertical edges
    vy, vx = np.where(V_arr > 0.5)
    if len(vx) > 0:
        v_cross = zones_np[vy, vx] != zones_np[vy + 1, vx]
        v_colors = np.where(v_cross, 'red', 'black')
        v_segs = [[(vx[i], vy[i]), (vx[i], vy[i] + 1)] for i in range(len(vx))]
        ax.add_collection(LineCollection(v_segs, colors=v_colors, linewidths=2))

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(False)


def run_inference(
    model: OperationNet,
    zones_np: np.ndarray,
    boundary_mask: torch.Tensor,
    grid_w: int,
    grid_h: int,
    max_steps: int = 50,
    top_k_pos: int = 15,
    top_k_act: int = 3,
    device: torch.device = torch.device('cuda'),
    verbose: bool = False,
):
    """
    Run iterative inference to produce an operation sequence.

    At each step:
    1. Encode current state → 5ch tensor
    2. Forward pass → position scores + action logits
    3. Try top-k (position, action) candidates
    4. Apply first valid, crossing-reducing operation
    5. Repeat until no improvement

    Returns dict with results.
    """
    h = HamiltonianSTL(grid_w, grid_h, init_pattern='zigzag')
    initial_crossings = compute_crossings(h, zones_np)
    current_crossings = initial_crossings

    # Dilated boundary mask for candidate positions
    dilated_mask = dilate_mask(boundary_mask, dilation=2)
    # Also mask to valid grid area
    valid_area = torch.zeros(MAX_GRID_SIZE, MAX_GRID_SIZE)
    valid_area[:grid_h, :grid_w] = 1.0
    dilated_mask = dilated_mask * valid_area

    sequence = []
    crossings_history = [initial_crossings]
    no_improve_count = 0
    total_attempts = 0
    invalid_count = 0

    model.eval()
    with torch.no_grad():
        for step in range(max_steps):
            # Encode current state
            state = encode_state_5ch(zones_np, boundary_mask, h, grid_w, grid_h)
            state_batch = state.unsqueeze(0).to(device)

            # Forward pass
            pos_logits, act_logits = model(state_batch)

            # Get candidate boundary positions
            bpos = dilated_mask.nonzero(as_tuple=False)  # [N, 2] (y, x)
            if len(bpos) == 0:
                break

            # Score positions
            pos_scores = pos_logits[0, 0, bpos[:, 0], bpos[:, 1]]
            k_pos = min(top_k_pos, len(pos_scores))
            top_pos_indices = pos_scores.topk(k_pos).indices

            applied = False
            for pi in top_pos_indices:
                py = bpos[pi, 0].item()
                px = bpos[pi, 1].item()

                # Top-k actions at this position
                act_scores = act_logits[0, :, py, px]
                k_act = min(top_k_act, NUM_ACTIONS)
                top_act_indices = act_scores.topk(k_act).indices

                for ai in top_act_indices:
                    action_idx = ai.item()
                    variant = VARIANT_REV[action_idx]
                    op_type = 'T' if action_idx < 8 else 'F'
                    total_attempts += 1

                    if not validate_action(op_type, px, py, variant, grid_w, grid_h):
                        invalid_count += 1
                        continue

                    # Save state for rollback
                    saved = save_grid_state(h)
                    success = apply_op(h, op_type, px, py, variant)

                    if success:
                        new_crossings = compute_crossings(h, zones_np)
                        if new_crossings < current_crossings:
                            # Keep the operation
                            sequence.append({
                                'kind': op_type,
                                'x': px,
                                'y': py,
                                'variant': variant,
                                'crossings_before': current_crossings,
                                'crossings_after': new_crossings,
                            })
                            current_crossings = new_crossings
                            crossings_history.append(current_crossings)
                            no_improve_count = 0
                            applied = True

                            if verbose:
                                console.print(
                                    f"  Step {step+1}: {op_type}({variant}) at ({px},{py}) "
                                    f"→ crossings {current_crossings}"
                                )
                            break
                        else:
                            # Didn't improve — rollback
                            restore_grid_state(h, saved)
                    else:
                        # Operation failed (invalid subgrid pattern) — rollback
                        restore_grid_state(h, saved)
                        invalid_count += 1

                if applied:
                    break

            if not applied:
                no_improve_count += 1
                if no_improve_count >= 10:
                    if verbose:
                        console.print(f"  Stopping: no improvement for {no_improve_count} steps")
                    break

    reduction = initial_crossings - current_crossings
    reduction_pct = (reduction / initial_crossings * 100) if initial_crossings > 0 else 0

    return {
        'initial_crossings': initial_crossings,
        'final_crossings': current_crossings,
        'reduction': reduction,
        'reduction_pct': reduction_pct,
        'num_operations': len(sequence),
        'sequence': sequence,
        'crossings_history': crossings_history,
        'total_attempts': total_attempts,
        'invalid_ops': invalid_count,
        'final_h': h,
    }


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained OperationNet from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    args = checkpoint.get('args', {})
    model = OperationNet(
        in_channels=5,
        base_features=args.get('base_features', 64),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, args


def evaluate_on_dataset(
    model: OperationNet,
    data_path: str,
    n_samples: int = 50,
    max_steps: int = 50,
    top_k_pos: int = 15,
    top_k_act: int = 3,
    device: torch.device = torch.device('cuda'),
    visualize: bool = False,
    vis_dir: str = 'nn_checkpoints/unet/vis',
):
    """Evaluate model on held-out trajectories from effective_dt_data.pkl."""
    console.print(Panel.fit(
        "[bold cyan]OperationNet (U-Net) - Evaluation[/bold cyan]\n"
        f"Data: {data_path}\n"
        f"Samples: {n_samples}\n"
        f"Max steps: {max_steps}\n"
        f"Top-k: positions={top_k_pos}, actions={top_k_act}",
        border_style="cyan"
    ))

    if data_path.endswith('.pt'):
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        # .pt format: flat tensors, extract one sample per trajectory
        states = data['states']       # [N, 4, 32, 32]
        traj_ids = data['traj_ids']   # [N]
        n_trajs = data['n_trajectories']

        # Pick last n_samples unique trajectories (test set)
        unique_tids = sorted(set(traj_ids.tolist()))
        if n_samples > len(unique_tids):
            n_samples = len(unique_tids)
        test_tids = set(unique_tids[-n_samples:])

        # Get first sample index per test trajectory
        test_indices = []
        seen_tids = set()
        for i, tid in enumerate(traj_ids.tolist()):
            if tid in test_tids and tid not in seen_tids:
                test_indices.append(i)
                seen_tids.add(tid)

        test_trajs = []
        for idx in test_indices:
            state_4ch = states[idx]  # [4, 32, 32]
            # Reconstruct zone info from channel 0 (normalized zones)
            zones_norm = state_4ch[0].numpy()  # [32, 32]
            # Infer grid size from channel 3 (grid mask: >= 0.4 means valid)
            grid_mask = (state_4ch[3] >= 0.4).numpy()
            ys, xs = np.where(grid_mask)
            grid_h = int(ys.max()) + 1 if len(ys) > 0 else 30
            grid_w = int(xs.max()) + 1 if len(xs) > 0 else 30
            # Un-normalize zones: find unique non-zero values, map to integer IDs
            zone_crop = zones_norm[:grid_h, :grid_w]
            unique_vals = sorted(set(zone_crop.flatten().tolist()))
            val_to_id = {v: i for i, v in enumerate(unique_vals)}
            zones_np = np.zeros((grid_h, grid_w), dtype=np.int32)
            for y in range(grid_h):
                for x in range(grid_w):
                    zones_np[y, x] = val_to_id[zone_crop[y, x]]
            test_trajs.append({
                'zones': zones_np,
                'grid_w': grid_w,
                'grid_h': grid_h,
            })
    else:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        trajectories = data['trajectories']
        if n_samples > len(trajectories):
            n_samples = len(trajectories)
        test_trajs = []
        for traj in trajectories[-n_samples:]:
            zones_np = traj['zones']
            if isinstance(zones_np, list):
                zones_np = np.array(zones_np)
            test_trajs.append({
                'zones': zones_np,
                'grid_w': traj.get('grid_w', 30),
                'grid_h': traj.get('grid_h', 30),
                'initial_crossings': traj.get('initial_crossings'),
                'final_crossings': traj.get('final_crossings'),
                'total_ops': traj.get('total_ops', traj.get('num_condensed_steps', 0)),
                'num_effective': traj.get('num_effective', 0),
                'zone_pattern': traj.get('zone_pattern', 'unknown'),
            })

    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.fields[status]}[/cyan]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=len(test_trajs), status="")

        for traj in test_trajs:
            grid_w = traj['grid_w']
            grid_h = traj['grid_h']
            zones_np = traj['zones']

            boundary_mask = compute_boundary_mask(zones_np, grid_h, grid_w)

            # Run UNet inference
            result = run_inference(
                model=model,
                zones_np=zones_np,
                boundary_mask=boundary_mask,
                grid_w=grid_w,
                grid_h=grid_h,
                max_steps=max_steps,
                top_k_pos=top_k_pos,
                top_k_act=top_k_act,
                device=device,
            )

            # SA baseline (if available from pkl data)
            sa_initial = traj.get('initial_crossings', result['initial_crossings'])
            sa_final = traj.get('final_crossings', result['final_crossings'])
            sa_reduction = sa_initial - sa_final if sa_initial else 0
            result['sa_initial'] = sa_initial
            result['sa_final'] = sa_final
            result['sa_reduction'] = sa_reduction
            result['sa_reduction_pct'] = (sa_reduction / sa_initial * 100) if sa_initial and sa_initial > 0 else 0
            result['sa_ops'] = traj.get('total_ops', 0)
            result['sa_effective_ops'] = traj.get('num_effective', 0)
            result['zone_pattern'] = traj.get('zone_pattern', 'unknown')
            result['grid_size'] = f"{grid_w}x{grid_h}"

            all_results.append(result)

            # Visualization: Initial vs UNet Final
            if visualize:
                os.makedirs(vis_dir, exist_ok=True)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

                # Initial zigzag path
                h_init = HamiltonianSTL(grid_w, grid_h, init_pattern='zigzag')
                init_crossings = result['initial_crossings']
                plot_cycle_on_ax(ax1, h_init, zones_np,
                                 f"Initial (crossings={init_crossings})")

                # UNet final path
                h_final = result['final_h']
                plot_cycle_on_ax(ax2, h_final, zones_np,
                                 f"UNet (crossings={result['final_crossings']}, "
                                 f"ops={result['num_operations']})")

                pattern = result.get('zone_pattern', 'unknown')
                fig.suptitle(f"Sample {len(all_results)} — {result['grid_size']} {pattern}",
                             fontsize=14, fontweight='bold')
                fig.tight_layout()
                fig.savefig(os.path.join(vis_dir, f"sample_{len(all_results)}.png"),
                            dpi=200, bbox_inches='tight')
                plt.close(fig)

            status = f"red={result['reduction']}, ops={result['num_operations']}"
            progress.update(task, advance=1, status=status)

    _display_results(all_results)
    return all_results


def _display_results(results: list):
    """Display evaluation results in Rich tables."""
    if not results:
        console.print("[bold red]No results to display[/bold red]")
        return

    reductions = [r['reduction'] for r in results]
    reduction_pcts = [r['reduction_pct'] for r in results]
    ops = [r['num_operations'] for r in results]
    invalids = [r['invalid_ops'] for r in results]
    attempts = [r['total_attempts'] for r in results]

    sa_reductions = [r['sa_reduction'] for r in results]
    sa_reduction_pcts = [r['sa_reduction_pct'] for r in results]
    sa_ops = [r.get('sa_ops', 0) for r in results]
    sa_eff_ops = [r.get('sa_effective_ops', 0) for r in results]

    # Summary table
    summary = Table(title="[bold]Evaluation Summary[/bold]", box=box.ROUNDED)
    summary.add_column("Metric", style="cyan")
    summary.add_column("OperationNet (U-Net)", style="green", justify="right")
    summary.add_column("SA Baseline", style="yellow", justify="right")

    summary.add_row(
        "Avg crossing reduction",
        f"{np.mean(reductions):.1f} ({np.mean(reduction_pcts):.1f}%)",
        f"{np.mean(sa_reductions):.1f} ({np.mean(sa_reduction_pcts):.1f}%)",
    )
    summary.add_row(
        "Median crossing reduction",
        f"{np.median(reductions):.1f} ({np.median(reduction_pcts):.1f}%)",
        f"{np.median(sa_reductions):.1f} ({np.median(sa_reduction_pcts):.1f}%)",
    )
    summary.add_row(
        "Max crossing reduction",
        f"{max(reductions):.0f} ({max(reduction_pcts):.1f}%)",
        f"{max(sa_reductions):.0f} ({max(sa_reduction_pcts):.1f}%)",
    )
    summary.add_row(
        "Avg operations used",
        f"{np.mean(ops):.1f}",
        f"{np.mean(sa_ops):.0f} total / {np.mean(sa_eff_ops):.1f} effective",
    )
    summary.add_row(
        "Avg invalid attempts",
        f"{np.mean(invalids):.1f}",
        "N/A",
    )

    # Win/loss vs SA
    wins = sum(1 for r in results if r['reduction'] >= r['sa_reduction'])
    close = sum(1 for r in results if abs(r['reduction'] - r['sa_reduction']) <= 2)
    nonzero = sum(1 for r in reductions if r > 0)

    summary.add_row(
        "UNet >= SA",
        f"{wins}/{len(results)} ({wins/len(results)*100:.0f}%)",
        "",
    )
    summary.add_row(
        "UNet within 2 of SA",
        f"{close}/{len(results)} ({close/len(results)*100:.0f}%)",
        "",
    )
    summary.add_row(
        "Samples with reduction > 0",
        f"{nonzero}/{len(results)} ({nonzero/len(results)*100:.0f}%)",
        "",
    )

    # Efficiency
    if np.mean(ops) > 0 and np.mean(sa_ops) > 0:
        summary.add_row(
            "Efficiency (reduction/op)",
            f"{np.mean(reductions)/max(np.mean(ops), 0.01):.2f}",
            f"{np.mean(sa_reductions)/max(np.mean(sa_ops), 0.01):.4f} total / "
            f"{np.mean(sa_reductions)/max(np.mean(sa_eff_ops), 0.01):.2f} effective",
        )

    # Hamiltonicity (always preserved by construction)
    summary.add_row(
        "Hamiltonicity preserved",
        "[bold green]100%[/bold green] (by construction)",
        "100%",
    )

    console.print(summary)

    # Per-sample detail (first 20)
    n_show = min(20, len(results))
    detail = Table(title=f"Per-Sample Results (first {n_show})", box=box.SIMPLE)
    detail.add_column("#", style="dim")
    detail.add_column("Grid")
    detail.add_column("Pattern")
    detail.add_column("Initial", justify="right")
    detail.add_column("UNet Final", justify="right")
    detail.add_column("UNet Red%", justify="right")
    detail.add_column("SA Final", justify="right")
    detail.add_column("SA Red%", justify="right")
    detail.add_column("UNet Ops", justify="right")
    detail.add_column("SA Ops", justify="right")

    for i, r in enumerate(results[:n_show]):
        style = None
        if r['reduction'] >= r.get('sa_reduction', float('inf')):
            style = "green"
        elif r['reduction'] > 0:
            style = "yellow"
        else:
            style = "red"

        detail.add_row(
            str(i + 1),
            r.get('grid_size', '?'),
            r.get('zone_pattern', '?'),
            str(r['initial_crossings']),
            str(r['final_crossings']),
            f"{r['reduction_pct']:.1f}%",
            str(r.get('sa_final', '?')),
            f"{r.get('sa_reduction_pct', 0):.1f}%",
            str(r['num_operations']),
            str(r.get('sa_ops', '?')),
            style=style,
        )

    console.print(detail)

    # Output format for paper
    console.print(Panel.fit(
        f"[bold]Final Output[/bold]\n"
        f"Avg final crossings: {np.mean([r['final_crossings'] for r in results]):.1f}\n"
        f"Avg operations to achieve: {np.mean(ops):.1f}\n"
        f"Avg crossing reduction: {np.mean(reduction_pcts):.1f}%\n"
        f"vs SA: {np.mean(sa_reduction_pcts):.1f}% with {np.mean(sa_ops):.0f} ops",
        border_style="cyan"
    ))


def main():
    parser = argparse.ArgumentParser(description='OperationNet (U-Net) Inference & Evaluation')
    parser.add_argument('--checkpoint', default='nn_checkpoints/unet/best.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data', default='model/unet/unet_data.pkl',
                        help='Path to unet_data.pkl for evaluation')
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of test samples to evaluate')
    parser.add_argument('--max_steps', type=int, default=50,
                        help='Max inference steps per sample')
    parser.add_argument('--top_k_pos', type=int, default=15,
                        help='Top-k boundary positions to try')
    parser.add_argument('--top_k_act', type=int, default=3,
                        help='Top-k actions per position to try')
    parser.add_argument('--device', default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-step details for each sample')
    parser.add_argument('--visualize', action='store_true',
                        help='Save initial vs UNet comparison PNGs')
    parser.add_argument('--vis_dir', default='nn_checkpoints/unet/vis',
                        help='Directory for visualization PNGs')

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    console.print(f"\n[bold]Loading model from {args.checkpoint}...[/bold]")
    model, model_args = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Parameters: {n_params:,}")

    # Run evaluation
    results = evaluate_on_dataset(
        model=model,
        data_path=args.data,
        n_samples=args.n_samples,
        max_steps=args.max_steps,
        top_k_pos=args.top_k_pos,
        top_k_act=args.top_k_act,
        device=device,
        visualize=args.visualize,
        vis_dir=args.vis_dir,
    )

    if args.visualize:
        console.print(f"\n[bold]Visualizations saved to {args.vis_dir}/[/bold]")

    # Save results
    output_path = Path(args.checkpoint).parent / 'inference_results.json'
    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items() if k not in ('sequence', 'crossings_history', 'final_h')}
        sr['crossings_history'] = r.get('crossings_history', [])
        sr['sequence'] = [
            {'kind': op['kind'], 'x': op['x'], 'y': op['y'], 'variant': op['variant']}
            for op in r.get('sequence', [])
        ]
        serializable.append(sr)

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


if __name__ == '__main__':
    main()
