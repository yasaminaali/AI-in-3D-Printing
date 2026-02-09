"""
FusionNet (CNN+RNN+FiLM) Inference for Hamiltonian Path Optimization.

Iterative inference with history context:
1. Encode current grid state as 5-channel tensor
2. Build history tensors from recent effective operations
3. Forward pass -> position scores + action logits
4. Select best valid (position, action) from boundary candidates
5. Update history buffer with effective operation
6. Repeat until no improvement

Tests all 4 zone patterns (left_right, voronoi, islands, stripes) with
per-pattern breakdown. Shows grid size, zone pattern, and zone count per sample.

Usage:
    PYTHONPATH=$(pwd):$PYTHONPATH python model/fusion/inference_fusion.py --checkpoint nn_checkpoints/fusion/best.pt
    PYTHONPATH=$(pwd):$PYTHONPATH python model/fusion/inference_fusion.py --checkpoint nn_checkpoints/fusion/best.pt --visualize
    PYTHONPATH=$(pwd):$PYTHONPATH python model/fusion/inference_fusion.py --checkpoint nn_checkpoints/fusion/best.pt --n_per_pattern 25
"""

import torch
import torch.nn.functional as F
import json
import copy
import argparse
import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque, defaultdict

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operations import HamiltonianSTL
from fusion_model import FusionNet, VARIANT_REV, VARIANT_MAP, NUM_ACTIONS

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, MofNCompleteColumn,
)
from rich import box

console = Console()

MAX_GRID_SIZE = 32
MAX_HISTORY = 8
TRANSPOSE_VARIANTS = {'nl', 'nr', 'sl', 'sr', 'eb', 'ea', 'wa', 'wb'}
FLIP_VARIANTS = {'n', 's', 'e', 'w'}
ALL_PATTERNS = ['left_right', 'voronoi', 'islands', 'stripes']


# ---------------------------------------------------------------------------
# Grid helpers (vectorized)
# ---------------------------------------------------------------------------

def _h_edges_np(h: HamiltonianSTL) -> np.ndarray:
    return np.array(h.H, dtype=np.float32)


def _v_edges_np(h: HamiltonianSTL) -> np.ndarray:
    return np.array(h.V, dtype=np.float32)


def compute_crossings(h: HamiltonianSTL, zones_np: np.ndarray) -> int:
    H_arr = _h_edges_np(h)
    V_arr = _v_edges_np(h)
    h_cross = H_arr * (zones_np[:, :-1] != zones_np[:, 1:]).astype(np.float32)
    v_cross = V_arr * (zones_np[:-1, :] != zones_np[1:, :]).astype(np.float32)
    return int(h_cross.sum() + v_cross.sum())


def compute_boundary_mask(zones_np: np.ndarray, grid_h: int, grid_w: int) -> torch.Tensor:
    mask = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.float32)
    h_diff = zones_np[:, :-1] != zones_np[:, 1:]
    mask[:grid_h, :grid_w - 1] = np.maximum(mask[:grid_h, :grid_w - 1], h_diff)
    mask[:grid_h, 1:grid_w] = np.maximum(mask[:grid_h, 1:grid_w], h_diff)
    v_diff = zones_np[:-1, :] != zones_np[1:, :]
    mask[:grid_h - 1, :grid_w] = np.maximum(mask[:grid_h - 1, :grid_w], v_diff)
    mask[1:grid_h, :grid_w] = np.maximum(mask[1:grid_h, :grid_w], v_diff)
    return torch.from_numpy(mask)


def encode_state_5ch(zones_np, boundary_mask, h, grid_w, grid_h):
    state = torch.zeros(5, MAX_GRID_SIZE, MAX_GRID_SIZE)
    max_zone = max(zones_np.max(), 1)
    state[0, :grid_h, :grid_w] = torch.from_numpy(zones_np.astype(np.float32)) / max_zone
    H_arr = _h_edges_np(h)
    state[1, :grid_h, :grid_w - 1] = torch.from_numpy(H_arr)
    V_arr = _v_edges_np(h)
    state[2, :grid_h - 1, :grid_w] = torch.from_numpy(V_arr)
    state[3] = boundary_mask.clone()
    state[3, :grid_h, :grid_w] = torch.maximum(
        state[3, :grid_h, :grid_w],
        torch.ones(grid_h, grid_w) * 0.5
    )
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


def validate_action(op_type, x, y, variant, grid_w, grid_h):
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


def apply_op(h, op_type, x, y, variant):
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


def save_grid_state(h):
    return (copy.deepcopy(h.H), copy.deepcopy(h.V))


def restore_grid_state(h, state):
    h.H, h.V = state


def dilate_mask(mask, dilation=2):
    if dilation <= 0:
        return mask
    kernel_size = 2 * dilation + 1
    dilated = F.max_pool2d(
        mask.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size, stride=1, padding=dilation,
    ).squeeze(0).squeeze(0)
    return dilated


# ---------------------------------------------------------------------------
# History buffer helpers
# ---------------------------------------------------------------------------

def build_history_tensors(history_buffer, max_history, device):
    """Build history tensors from a deque of recent effective ops."""
    hist_act = torch.zeros(1, max_history, dtype=torch.long, device=device)
    hist_py = torch.zeros(1, max_history, dtype=torch.long, device=device)
    hist_px = torch.zeros(1, max_history, dtype=torch.long, device=device)
    hist_cb = torch.zeros(1, max_history, dtype=torch.float, device=device)
    hist_ca = torch.zeros(1, max_history, dtype=torch.float, device=device)
    hist_mask = torch.zeros(1, max_history, dtype=torch.float, device=device)

    for i, entry in enumerate(history_buffer):
        hist_act[0, i] = entry['action']
        hist_py[0, i] = entry['py']
        hist_px[0, i] = entry['px']
        hist_cb[0, i] = entry['cb'] / 60.0
        hist_ca[0, i] = entry['ca'] / 60.0
        hist_mask[0, i] = 1.0

    return hist_act, hist_py, hist_px, hist_cb, hist_ca, hist_mask


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_cycle_on_ax(ax, h, zones_np, title):
    W, H = h.width, h.height
    zone_vals = sorted(set(zones_np.flatten().tolist()))
    a = zone_vals[0] if zone_vals else 0
    colors = np.zeros((H, W, 3))
    colors[zones_np[:H, :W] == a] = [0.68, 0.85, 0.90]
    colors[zones_np[:H, :W] != a] = [0.56, 0.93, 0.56]
    ax.imshow(colors, extent=(-0.5, W - 0.5, H - 0.5, -0.5), origin='upper')

    from matplotlib.collections import LineCollection
    H_arr = _h_edges_np(h)
    V_arr = _v_edges_np(h)
    hy, hx = np.where(H_arr > 0.5)
    if len(hx) > 0:
        h_cross = zones_np[hy, hx] != zones_np[hy, hx + 1]
        h_colors = np.where(h_cross, 'red', 'black')
        h_segs = [[(hx[i], hy[i]), (hx[i] + 1, hy[i])] for i in range(len(hx))]
        ax.add_collection(LineCollection(h_segs, colors=h_colors, linewidths=2))
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


# ---------------------------------------------------------------------------
# Iterative inference with history
# ---------------------------------------------------------------------------

def run_inference(
    model,
    zones_np,
    boundary_mask,
    grid_w, grid_h,
    max_history=8,
    max_steps=50,
    top_k_pos=15,
    top_k_act=3,
    device=torch.device('cuda'),
    verbose=False,
):
    """
    Iterative FusionNet inference with history context.

    Unlike UNet inference, maintains a rolling history buffer of effective
    operations and feeds it to the model at each step.
    """
    h = HamiltonianSTL(grid_w, grid_h, init_pattern='zigzag')
    initial_crossings = compute_crossings(h, zones_np)
    current_crossings = initial_crossings

    dilated_mask = dilate_mask(boundary_mask, dilation=1)
    valid_area = torch.zeros(MAX_GRID_SIZE, MAX_GRID_SIZE)
    valid_area[:grid_h, :grid_w] = 1.0
    dilated_mask = (dilated_mask * valid_area).to(device)

    history_buffer = deque(maxlen=max_history)
    sequence = []
    crossings_history = [initial_crossings]
    no_improve_count = 0
    total_attempts = 0
    invalid_count = 0

    model.eval()
    with torch.no_grad():
        for step in range(max_steps):
            state = encode_state_5ch(zones_np, boundary_mask, h, grid_w, grid_h)
            state_batch = state.unsqueeze(0).to(device)

            # Build history tensors
            hist_act, hist_py, hist_px, hist_cb, hist_ca, hist_mask = \
                build_history_tensors(history_buffer, max_history, device)

            # Forward pass with history
            pos_logits, act_logits = model(
                state_batch, hist_act, hist_py, hist_px, hist_cb, hist_ca, hist_mask
            )

            # Get candidate boundary positions
            bpos = dilated_mask.nonzero(as_tuple=False)
            if len(bpos) == 0:
                break

            # Pool across K hypotheses: mean-softmax over boundary
            K = pos_logits.shape[1]
            pos_flat = pos_logits[0].reshape(K, -1)
            mask_1d = dilated_mask.reshape(-1).bool()
            pos_flat_masked = pos_flat.masked_fill(~mask_1d.unsqueeze(0), float('-inf'))
            probs_k = torch.softmax(pos_flat_masked, dim=-1)
            pooled_flat = probs_k.mean(dim=0)
            pooled_2d = pooled_flat.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
            pos_scores = pooled_2d[bpos[:, 0], bpos[:, 1]]
            k_pos = min(top_k_pos, len(pos_scores))
            top_pos_indices = pos_scores.topk(k_pos).indices

            applied = False
            for pi in top_pos_indices:
                py = bpos[pi, 0].item()
                px = bpos[pi, 1].item()
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

                    saved = save_grid_state(h)
                    success = apply_op(h, op_type, px, py, variant)

                    if success:
                        new_crossings = compute_crossings(h, zones_np)
                        if new_crossings < current_crossings:
                            sequence.append({
                                'kind': op_type,
                                'x': px, 'y': py,
                                'variant': variant,
                                'crossings_before': current_crossings,
                                'crossings_after': new_crossings,
                            })

                            # Update history buffer
                            history_buffer.append({
                                'action': VARIANT_MAP.get(variant, 0),
                                'py': min(py, MAX_GRID_SIZE - 1),
                                'px': min(px, MAX_GRID_SIZE - 1),
                                'cb': current_crossings,
                                'ca': new_crossings,
                            })

                            current_crossings = new_crossings
                            crossings_history.append(current_crossings)
                            no_improve_count = 0
                            applied = True

                            if verbose:
                                console.print(
                                    f"  Step {step+1}: {op_type}({variant}) at ({px},{py}) "
                                    f"-> crossings {current_crossings} "
                                    f"[hist={len(history_buffer)}]"
                                )
                            break
                        else:
                            restore_grid_state(h, saved)
                    else:
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
        'history_length': len(history_buffer),
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get('args', {})

    model = FusionNet(
        in_channels=5,
        base_features=args.get('base_features', 48),
        n_hypotheses=args.get('n_hypotheses', 4),
        max_history=args.get('max_history', MAX_HISTORY),
        rnn_hidden=args.get('rnn_hidden', 192),
        rnn_layers=args.get('rnn_layers', 2),
        rnn_dropout=args.get('rnn_dropout', 0.15),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, args


# ---------------------------------------------------------------------------
# Evaluation: all 4 zone patterns with per-pattern breakdown
# ---------------------------------------------------------------------------

def evaluate_all_patterns(
    model,
    jsonl_path,
    n_per_pattern=25,
    max_steps=50,
    max_history=8,
    top_k_pos=15,
    top_k_act=3,
    device=torch.device('cuda'),
    visualize=False,
    vis_dir='nn_checkpoints/fusion/vis',
):
    """
    Evaluate FusionNet on all 4 zone patterns with per-pattern statistics.

    Samples n_per_pattern trajectories from each zone pattern (left_right,
    voronoi, islands, stripes) and compares against SA baseline.
    Reports per-pattern breakdown and overall summary.
    """
    console.print(Panel.fit(
        "[bold cyan]FusionNet (CNN+RNN+FiLM) - All Zone Patterns Evaluation[/bold cyan]\n"
        f"Data: {jsonl_path}\n"
        f"Patterns: {', '.join(ALL_PATTERNS)}\n"
        f"Samples per pattern: {n_per_pattern}\n"
        f"Max steps: {max_steps} | Max history: {max_history}\n"
        f"Top-k: positions={top_k_pos}, actions={top_k_act}",
        border_style="cyan"
    ))

    # Load and group trajectories by zone pattern
    with open(jsonl_path) as f:
        lines = f.readlines()

    pattern_lines = defaultdict(list)
    for line in lines:
        traj = json.loads(line.strip())
        pattern = traj.get('zone_pattern', 'unknown')
        pattern_lines[pattern].append(line)

    # Report dataset composition
    comp_table = Table(title="[bold]Dataset Composition[/bold]", box=box.SIMPLE)
    comp_table.add_column("Zone Pattern", style="cyan")
    comp_table.add_column("Total Trajectories", justify="right")
    comp_table.add_column("Test Samples", justify="right")
    for p in ALL_PATTERNS:
        available = len(pattern_lines[p])
        n_test = min(n_per_pattern, available)
        comp_table.add_row(p, str(available), str(n_test))
    console.print(comp_table)

    # Select test samples from each pattern (last n_per_pattern from each)
    test_samples = []
    for pattern in ALL_PATTERNS:
        available = pattern_lines[pattern]
        n_test = min(n_per_pattern, len(available))
        if n_test == 0:
            console.print(f"  [yellow]Warning: no samples for pattern '{pattern}'[/yellow]")
            continue
        selected = available[-n_test:]
        for line in selected:
            test_samples.append((pattern, line))

    total_samples = len(test_samples)
    console.print(f"\n  Total test samples: [green]{total_samples}[/green]")

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
        task = progress.add_task("Evaluating all patterns", total=total_samples, status="")

        for pattern, line in test_samples:
            traj = json.loads(line.strip())
            grid_w = traj.get('grid_W', 30)
            grid_h = traj.get('grid_H', 30)
            zones_np = np.array(traj['zone_grid']).reshape(grid_h, grid_w)
            n_zones = len(set(zones_np.flatten().tolist()))
            boundary_mask = compute_boundary_mask(zones_np, grid_h, grid_w)

            result = run_inference(
                model=model,
                zones_np=zones_np,
                boundary_mask=boundary_mask,
                grid_w=grid_w,
                grid_h=grid_h,
                max_history=max_history,
                max_steps=max_steps,
                top_k_pos=top_k_pos,
                top_k_act=top_k_act,
                device=device,
            )

            # SA baseline from trajectory
            sa_initial = traj.get('initial_crossings', result['initial_crossings'])
            sa_final = traj.get('final_crossings', result['final_crossings'])
            sa_reduction = sa_initial - sa_final if sa_initial else 0
            n_sa_ops = len(traj.get('sequence_ops', []))
            n_sa_effective = sum(
                1 for op in traj.get('sequence_ops', []) if op.get('kind') != 'N'
            )

            result['sa_initial'] = sa_initial
            result['sa_final'] = sa_final
            result['sa_reduction'] = sa_reduction
            result['sa_reduction_pct'] = (sa_reduction / sa_initial * 100) if sa_initial and sa_initial > 0 else 0
            result['sa_ops'] = n_sa_ops
            result['sa_effective_ops'] = n_sa_effective
            result['zone_pattern'] = pattern
            result['grid_size'] = f"{grid_w}x{grid_h}"
            result['n_zones'] = n_zones

            all_results.append(result)

            if visualize:
                os.makedirs(vis_dir, exist_ok=True)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                h_init = HamiltonianSTL(grid_w, grid_h, init_pattern='zigzag')
                plot_cycle_on_ax(ax1, h_init, zones_np,
                                 f"Initial (crossings={result['initial_crossings']})")
                plot_cycle_on_ax(ax2, result['final_h'], zones_np,
                                 f"FusionNet (crossings={result['final_crossings']}, "
                                 f"ops={result['num_operations']})")
                fig.suptitle(
                    f"Sample {len(all_results)} | {pattern} | "
                    f"{result['grid_size']} | {n_zones} zones",
                    fontsize=14, fontweight='bold')
                fig.tight_layout()
                fig.savefig(os.path.join(vis_dir, f"{pattern}_{len(all_results)}.png"),
                            dpi=200, bbox_inches='tight')
                plt.close(fig)

            status = (f"{pattern} | red={result['reduction']}/{result['sa_reduction']} | "
                      f"ops={result['num_operations']}")
            progress.update(task, advance=1, status=status)

    _display_all_pattern_results(all_results)
    return all_results


def _display_all_pattern_results(results):
    """Display per-pattern breakdown + overall summary + per-sample detail."""
    if not results:
        console.print("[bold red]No results to display[/bold red]")
        return

    # --- Per-pattern summary table ---
    pattern_results = defaultdict(list)
    for r in results:
        pattern_results[r['zone_pattern']].append(r)

    per_pattern = Table(
        title="[bold]Per-Pattern Results: FusionNet vs SA Baseline[/bold]",
        box=box.ROUNDED
    )
    per_pattern.add_column("Pattern", style="cyan")
    per_pattern.add_column("N", justify="right", style="dim")
    per_pattern.add_column("Zones", justify="right")
    per_pattern.add_column("Grid", justify="center")
    per_pattern.add_column("Fusion Avg Red", justify="right", style="green")
    per_pattern.add_column("SA Avg Red", justify="right", style="yellow")
    per_pattern.add_column("Fusion Ops", justify="right")
    per_pattern.add_column("SA Eff Ops", justify="right")
    per_pattern.add_column(">= SA", justify="right")
    per_pattern.add_column("Red > 0", justify="right")

    for pattern in ALL_PATTERNS:
        pr = pattern_results.get(pattern, [])
        if not pr:
            per_pattern.add_row(pattern, "0", "-", "-", "-", "-", "-", "-", "-", "-")
            continue

        n = len(pr)
        avg_zones = np.mean([r['n_zones'] for r in pr])
        grid_sizes = set(r['grid_size'] for r in pr)
        grid_str = ', '.join(sorted(grid_sizes))

        fusion_reds = [r['reduction'] for r in pr]
        fusion_pcts = [r['reduction_pct'] for r in pr]
        sa_reds = [r['sa_reduction'] for r in pr]
        sa_pcts = [r['sa_reduction_pct'] for r in pr]
        fusion_ops = [r['num_operations'] for r in pr]
        sa_eff = [r['sa_effective_ops'] for r in pr]

        wins = sum(1 for r in pr if r['reduction'] >= r['sa_reduction'])
        nonzero = sum(1 for r in fusion_reds if r > 0)

        per_pattern.add_row(
            pattern,
            str(n),
            f"{avg_zones:.0f}",
            grid_str,
            f"{np.mean(fusion_reds):.1f} ({np.mean(fusion_pcts):.1f}%)",
            f"{np.mean(sa_reds):.1f} ({np.mean(sa_pcts):.1f}%)",
            f"{np.mean(fusion_ops):.1f}",
            f"{np.mean(sa_eff):.1f}",
            f"{wins}/{n} ({wins/n*100:.0f}%)",
            f"{nonzero}/{n} ({nonzero/n*100:.0f}%)",
        )

    console.print(per_pattern)

    # --- Overall summary ---
    reductions = [r['reduction'] for r in results]
    reduction_pcts = [r['reduction_pct'] for r in results]
    ops = [r['num_operations'] for r in results]
    sa_reductions = [r['sa_reduction'] for r in results]
    sa_reduction_pcts = [r['sa_reduction_pct'] for r in results]
    sa_ops = [r.get('sa_ops', 0) for r in results]
    sa_eff_ops = [r.get('sa_effective_ops', 0) for r in results]

    summary = Table(title="[bold]Overall: FusionNet vs SA Baseline[/bold]", box=box.ROUNDED)
    summary.add_column("Metric", style="cyan")
    summary.add_column("FusionNet", style="green", justify="right")
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

    wins = sum(1 for r in results if r['reduction'] >= r['sa_reduction'])
    close = sum(1 for r in results if abs(r['reduction'] - r['sa_reduction']) <= 2)
    nonzero = sum(1 for r in reductions if r > 0)

    summary.add_row(
        "FusionNet >= SA",
        f"{wins}/{len(results)} ({wins/len(results)*100:.0f}%)", "",
    )
    summary.add_row(
        "Within 2 of SA",
        f"{close}/{len(results)} ({close/len(results)*100:.0f}%)", "",
    )
    summary.add_row(
        "Samples with reduction > 0",
        f"{nonzero}/{len(results)} ({nonzero/len(results)*100:.0f}%)", "",
    )

    if np.mean(ops) > 0 and np.mean(sa_ops) > 0:
        summary.add_row(
            "Efficiency (reduction/op)",
            f"{np.mean(reductions)/max(np.mean(ops), 0.01):.2f}",
            f"{np.mean(sa_reductions)/max(np.mean(sa_ops), 0.01):.4f} total / "
            f"{np.mean(sa_reductions)/max(np.mean(sa_eff_ops), 0.01):.2f} effective",
        )

    summary.add_row(
        "Hamiltonicity preserved",
        "[bold green]100%[/bold green] (by construction)", "100%",
    )

    console.print(summary)

    # --- Per-sample detail table (grouped by pattern) ---
    for pattern in ALL_PATTERNS:
        pr = pattern_results.get(pattern, [])
        if not pr:
            continue

        n_show = min(15, len(pr))
        detail = Table(
            title=f"[bold]{pattern}[/bold] — Per-Sample Results (first {n_show})",
            box=box.SIMPLE,
        )
        detail.add_column("#", style="dim")
        detail.add_column("Grid")
        detail.add_column("Zones", justify="right")
        detail.add_column("Initial", justify="right")
        detail.add_column("Fusion Final", justify="right")
        detail.add_column("Fusion Red%", justify="right")
        detail.add_column("SA Final", justify="right")
        detail.add_column("SA Red%", justify="right")
        detail.add_column("Fusion Ops", justify="right")
        detail.add_column("SA Eff Ops", justify="right")
        detail.add_column("Hist", justify="right")

        for i, r in enumerate(pr[:n_show]):
            if r['reduction'] >= r.get('sa_reduction', float('inf')):
                style = "green"
            elif r['reduction'] > 0:
                style = "yellow"
            else:
                style = "red"

            detail.add_row(
                str(i + 1),
                r.get('grid_size', '?'),
                str(r.get('n_zones', '?')),
                str(r['initial_crossings']),
                str(r['final_crossings']),
                f"{r['reduction_pct']:.1f}%",
                str(r.get('sa_final', '?')),
                f"{r.get('sa_reduction_pct', 0):.1f}%",
                str(r['num_operations']),
                str(r.get('sa_effective_ops', '?')),
                str(r['history_length']),
                style=style,
            )

        console.print(detail)

    # --- Final panel ---
    console.print(Panel.fit(
        f"[bold]Final Summary[/bold]\n"
        f"Patterns tested: {', '.join(ALL_PATTERNS)}\n"
        f"Total samples: {len(results)}\n"
        f"Avg initial crossings: {np.mean([r['initial_crossings'] for r in results]):.1f}\n"
        f"Avg final crossings (FusionNet): {np.mean([r['final_crossings'] for r in results]):.1f}\n"
        f"Avg final crossings (SA): {np.mean([r['sa_final'] for r in results]):.1f}\n"
        f"Avg operations (FusionNet): {np.mean(ops):.1f}\n"
        f"Avg crossing reduction: {np.mean(reduction_pcts):.1f}% "
        f"vs SA: {np.mean(sa_reduction_pcts):.1f}%",
        border_style="cyan"
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='FusionNet Inference & Evaluation')
    parser.add_argument('--checkpoint', default='nn_checkpoints/fusion/best.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--jsonl', default='combined_dataset.jsonl',
                        help='Path to combined_dataset.jsonl')
    parser.add_argument('--n_per_pattern', type=int, default=25,
                        help='Number of test samples per zone pattern')
    parser.add_argument('--max_steps', type=int, default=50,
                        help='Max inference steps per sample')
    parser.add_argument('--top_k_pos', type=int, default=15,
                        help='Top-k boundary positions to try')
    parser.add_argument('--top_k_act', type=int, default=3,
                        help='Top-k actions per position to try')
    parser.add_argument('--device', default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-step details')
    parser.add_argument('--visualize', action='store_true',
                        help='Save comparison PNGs')
    parser.add_argument('--vis_dir', default='nn_checkpoints/fusion/vis',
                        help='Directory for visualization PNGs')

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    console.print(f"\n[bold]Loading FusionNet from {args.checkpoint}...[/bold]")
    model, model_args = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    max_history = model_args.get('max_history', MAX_HISTORY)
    console.print(f"  Parameters: {n_params:,}")
    console.print(f"  Max history: {max_history}")
    console.print(f"  Device: {device}")

    # Evaluate all 4 zone patterns
    results = evaluate_all_patterns(
        model=model,
        jsonl_path=args.jsonl,
        n_per_pattern=args.n_per_pattern,
        max_steps=args.max_steps,
        max_history=max_history,
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

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


if __name__ == '__main__':
    main()
