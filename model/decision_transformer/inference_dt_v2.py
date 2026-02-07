"""
Decision Transformer v2 Inference for Hamiltonian Path Optimization.

Autoregressive inference with:
- Top-k sampling with temperature
- Action validation (bounds checking, Hamiltonicity preservation)
- Proper RTG tracking
- Early termination on plateau
- Rich live dashboard
- Batch evaluation on held-out test data

Usage:
    python model/decision_transformer/inference_dt_v2.py --checkpoint nn_checkpoints/dt_v2/best.pt
    python model/decision_transformer/inference_dt_v2.py --checkpoint best.pt --n_samples 50 --zone_pattern left_right
"""

import torch
import torch.nn.functional as F
import pickle
import json
import argparse
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operations import HamiltonianSTL
from Zones import (
    zones_left_right, zones_top_bottom, zones_diagonal,
    zones_stripes, zones_checkerboard, zones_voronoi,
)

from dt_model import DecisionTransformer

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns
from rich import box

console = Console()

# Canonical mappings (must match training)
MAX_GRID_SIZE = 32
OP_TYPE_MAP = {'N': 0, 'T': 1, 'F': 2}
OP_TYPE_REV = {0: 'N', 1: 'T', 2: 'F'}
VARIANT_MAP = {
    'nl': 0, 'nr': 1, 'sl': 2, 'sr': 3, 'eb': 4, 'ea': 5, 'wa': 6, 'wb': 7,
    'n': 8, 's': 9, 'e': 10, 'w': 11
}
VARIANT_REV = {v: k for k, v in VARIANT_MAP.items()}
TRANSPOSE_VARIANTS = {'nl', 'nr', 'sl', 'sr', 'eb', 'ea', 'wa', 'wb'}
FLIP_VARIANTS = {'n', 's', 'e', 'w'}


def get_zone_function(pattern_name: str):
    """Get zone generation function by name."""
    funcs = {
        'left_right': zones_left_right,
        'top_bottom': zones_top_bottom,
        'diagonal': zones_diagonal,
        'stripes': zones_stripes,
        'checkerboard': zones_checkerboard,
        'voronoi': zones_voronoi,
    }
    return funcs.get(pattern_name)


def compute_crossings(h: HamiltonianSTL, zones_dict: dict) -> int:
    """Count zone boundary crossings in current path."""
    crossings = 0
    for y in range(h.height):
        for x in range(h.width - 1):
            if h.H[y][x]:
                if zones_dict.get((x, y), 0) != zones_dict.get((x + 1, y), 0):
                    crossings += 1
    for y in range(h.height - 1):
        for x in range(h.width):
            if h.V[y][x]:
                if zones_dict.get((x, y), 0) != zones_dict.get((x, y + 1), 0):
                    crossings += 1
    return crossings


def compute_boundary_mask(zones_np: np.ndarray, grid_h: int, grid_w: int) -> torch.Tensor:
    """Compute zone boundary mask."""
    mask = torch.zeros(MAX_GRID_SIZE, MAX_GRID_SIZE)
    for y in range(grid_h):
        for x in range(grid_w - 1):
            if zones_np[y, x] != zones_np[y, x + 1]:
                mask[y, x] = 1.0
                mask[y, x + 1] = 1.0
    for y in range(grid_h - 1):
        for x in range(grid_w):
            if zones_np[y, x] != zones_np[y + 1, x]:
                mask[y, x] = 1.0
                mask[y + 1, x] = 1.0
    return mask


def encode_state(zones_np: np.ndarray, boundary_mask: torch.Tensor,
                 h: HamiltonianSTL, grid_w: int, grid_h: int) -> torch.Tensor:
    """Encode grid state as 4-channel tensor padded to MAX_GRID_SIZE."""
    state = torch.zeros(4, MAX_GRID_SIZE, MAX_GRID_SIZE)

    # Channel 0: Zone IDs normalized
    max_zone = max(zones_np.max(), 1)
    state[0, :grid_h, :grid_w] = torch.from_numpy(zones_np.astype(np.float32)) / max_zone

    # Channel 1: Horizontal edges
    for y in range(grid_h):
        for x in range(grid_w - 1):
            state[1, y, x] = float(h.H[y][x])

    # Channel 2: Vertical edges
    for y in range(grid_h - 1):
        for x in range(grid_w):
            state[2, y, x] = float(h.V[y][x])

    # Channel 3: Boundary mask + grid mask
    state[3] = boundary_mask.clone()
    state[3, :grid_h, :grid_w] = torch.maximum(
        state[3, :grid_h, :grid_w],
        torch.ones(grid_h, grid_w) * 0.5
    )

    return state


def validate_action(op_type: str, x: int, y: int, variant: str,
                    grid_w: int, grid_h: int) -> bool:
    """Check if an action is valid (within bounds for the operation type)."""
    if op_type == 'N':
        return False  # No-op

    if op_type == 'T':
        # Transpose needs 3x3 subgrid: (x, y) to (x+2, y+2)
        if variant not in TRANSPOSE_VARIANTS:
            return False
        if x + 2 > grid_w or y + 2 > grid_h:
            return False
        if x < 0 or y < 0:
            return False
        return True

    if op_type == 'F':
        if variant not in FLIP_VARIANTS:
            return False
        if variant in ['n', 's']:
            # Flip n/s needs 2x3: (x, y) to (x+1, y+2)
            if x + 1 > grid_w or y + 2 > grid_h:
                return False
        else:
            # Flip e/w needs 3x2: (x, y) to (x+2, y+1)
            if x + 2 > grid_w or y + 1 > grid_h:
                return False
        if x < 0 or y < 0:
            return False
        return True

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


def sample_action_topk(logits: dict, k: int = 5, temperature: float = 0.7,
                       grid_w: int = 30, grid_h: int = 30):
    """
    Sample action using top-k with temperature, with validity filtering.

    Tries up to k*2 samples to find a valid action. Falls back to greedy if needed.
    """
    # Get probabilities with temperature
    op_probs = F.softmax(logits['op_type'] / temperature, dim=-1)
    x_probs = F.softmax(logits['x'] / temperature, dim=-1)
    y_probs = F.softmax(logits['y'] / temperature, dim=-1)
    var_probs = F.softmax(logits['variant'] / temperature, dim=-1)

    # Top-k for each head
    op_topk = torch.topk(op_probs, min(k, op_probs.size(-1)))
    x_topk = torch.topk(x_probs, min(k, x_probs.size(-1)))
    y_topk = torch.topk(y_probs, min(k, y_probs.size(-1)))
    var_topk = torch.topk(var_probs, min(k, var_probs.size(-1)))

    # Try combinations starting with most likely
    for attempt in range(k * 3):
        if attempt == 0:
            # First try: greedy (most likely)
            op_idx = op_topk.indices[0].item()
            x_idx = x_topk.indices[0].item()
            y_idx = y_topk.indices[0].item()
            var_idx = var_topk.indices[0].item()
        else:
            # Sample from top-k distributions
            op_idx = op_topk.indices[torch.multinomial(op_topk.values, 1).item()].item()
            x_idx = x_topk.indices[torch.multinomial(x_topk.values, 1).item()].item()
            y_idx = y_topk.indices[torch.multinomial(y_topk.values, 1).item()].item()
            var_idx = var_topk.indices[torch.multinomial(var_topk.values, 1).item()].item()

        op_type = OP_TYPE_REV[op_idx]
        variant = VARIANT_REV.get(var_idx, 'nl')

        # Fix variant/op_type mismatch
        if op_type == 'T' and variant in FLIP_VARIANTS:
            # Pick best transpose variant instead
            for v_idx in var_topk.indices.tolist():
                if VARIANT_REV.get(v_idx, '') in TRANSPOSE_VARIANTS:
                    var_idx = v_idx
                    variant = VARIANT_REV[v_idx]
                    break
        elif op_type == 'F' and variant in TRANSPOSE_VARIANTS:
            for v_idx in var_topk.indices.tolist():
                if VARIANT_REV.get(v_idx, '') in FLIP_VARIANTS:
                    var_idx = v_idx
                    variant = VARIANT_REV[v_idx]
                    break

        if validate_action(op_type, x_idx, y_idx, variant, grid_w, grid_h):
            return op_idx, x_idx, y_idx, var_idx, op_type, variant

    # Fallback: greedy with clamped positions
    op_idx = op_topk.indices[0].item()
    op_type = OP_TYPE_REV[op_idx]
    if op_type == 'N':
        op_type = 'T'
        op_idx = 1

    x_idx = min(x_topk.indices[0].item(), grid_w - 3)
    y_idx = min(y_topk.indices[0].item(), grid_h - 3)
    x_idx = max(0, x_idx)
    y_idx = max(0, y_idx)

    if op_type == 'T':
        for v_idx in var_topk.indices.tolist():
            if VARIANT_REV.get(v_idx, '') in TRANSPOSE_VARIANTS:
                return op_idx, x_idx, y_idx, v_idx, op_type, VARIANT_REV[v_idx]
        return op_idx, x_idx, y_idx, 0, 'T', 'nl'
    else:
        for v_idx in var_topk.indices.tolist():
            if VARIANT_REV.get(v_idx, '') in FLIP_VARIANTS:
                return op_idx, x_idx, y_idx, v_idx, op_type, VARIANT_REV[v_idx]
        return op_idx, x_idx, y_idx, 8, 'F', 'n'


def run_inference(
    model: DecisionTransformer,
    zones_dict: dict,
    zones_np: np.ndarray,
    boundary_mask: torch.Tensor,
    grid_w: int = 30,
    grid_h: int = 30,
    target_rtg: float = 15.0,
    max_steps: int = 100,
    context_len: int = 50,
    top_k: int = 5,
    temperature: float = 0.7,
    no_improve_patience: int = 20,
    device: torch.device = torch.device('cuda'),
    verbose: bool = True,
) -> dict:
    """
    Run autoregressive DT inference.

    Returns dict with crossings trajectory, operations applied, and metrics.
    """
    model.eval()

    # Initialize Hamiltonian path
    h = HamiltonianSTL(grid_w, grid_h, init_pattern='zigzag')
    initial_crossings = compute_crossings(h, zones_dict)

    # History buffers
    states_history = []
    actions_history = []
    rtg_history = []
    timesteps_history = []

    current_crossings = initial_crossings
    best_crossings = initial_crossings
    ops_applied = []
    effective_ops = 0
    invalid_ops = 0
    no_improve_count = 0

    with torch.no_grad():
        for step in range(max_steps):
            # Encode current state
            state = encode_state(zones_np, boundary_mask, h, grid_w, grid_h)
            states_history.append(state)

            # RTG = remaining desired reduction
            achieved_so_far = initial_crossings - current_crossings
            remaining_rtg = max(0.0, target_rtg - achieved_so_far)
            rtg_history.append(remaining_rtg)
            timesteps_history.append(step)

            # Build context tensors (last context_len steps)
            ctx_len = min(len(states_history), context_len)
            start_idx = len(states_history) - ctx_len

            # States
            states_ctx = torch.stack(states_history[start_idx:]).unsqueeze(0).to(device)

            # RTG
            rtg_ctx = torch.tensor(
                [[r] for r in rtg_history[start_idx:]],
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            # Timesteps
            timesteps_ctx = torch.tensor(
                timesteps_history[start_idx:], dtype=torch.long
            ).unsqueeze(0).to(device)

            # Actions (previous actions; current step gets zeros)
            actions_ctx = torch.zeros(1, ctx_len, 4, dtype=torch.long).to(device)
            for i, a_idx in enumerate(range(start_idx, start_idx + ctx_len)):
                if a_idx < len(actions_history):
                    actions_ctx[0, i] = actions_history[a_idx]

            # Left-pad to context_len if needed
            if ctx_len < context_len:
                pad = context_len - ctx_len
                states_ctx = torch.cat([
                    torch.zeros(1, pad, 4, MAX_GRID_SIZE, MAX_GRID_SIZE, device=device),
                    states_ctx
                ], dim=1)
                rtg_ctx = torch.cat([
                    torch.zeros(1, pad, 1, device=device),
                    rtg_ctx
                ], dim=1)
                timesteps_ctx = torch.cat([
                    torch.zeros(1, pad, dtype=torch.long, device=device),
                    timesteps_ctx
                ], dim=1)
                actions_ctx = torch.cat([
                    torch.zeros(1, pad, 4, dtype=torch.long, device=device),
                    actions_ctx
                ], dim=1)

            # Forward pass
            logits = model(states_ctx, actions_ctx, rtg_ctx, timesteps_ctx)

            # Get last-timestep logits
            last_logits = {k: v[0, -1] for k, v in logits.items()}

            # Sample action with top-k
            op_idx, x_idx, y_idx, var_idx, op_type, variant = sample_action_topk(
                last_logits, k=top_k, temperature=temperature,
                grid_w=grid_w, grid_h=grid_h
            )

            # Store action
            actions_history.append(
                torch.tensor([op_idx, x_idx, y_idx, var_idx], dtype=torch.long)
            )

            # Validate and apply
            if not validate_action(op_type, x_idx, y_idx, variant, grid_w, grid_h):
                invalid_ops += 1
                no_improve_count += 1
            else:
                success = apply_op(h, op_type, x_idx, y_idx, variant)
                if success:
                    new_crossings = compute_crossings(h, zones_dict)
                    ops_applied.append({
                        'kind': op_type,
                        'x': x_idx,
                        'y': y_idx,
                        'variant': variant,
                        'crossings_before': current_crossings,
                        'crossings_after': new_crossings,
                    })

                    if new_crossings < current_crossings:
                        effective_ops += 1
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    current_crossings = new_crossings
                    if current_crossings < best_crossings:
                        best_crossings = current_crossings
                else:
                    invalid_ops += 1
                    no_improve_count += 1

            # Early stop: achieved target
            if current_crossings <= initial_crossings - target_rtg:
                break

            # Early stop: plateau
            if no_improve_count >= no_improve_patience:
                break

            # Early stop: RTG exhausted
            if remaining_rtg <= 0:
                break

    reduction = initial_crossings - best_crossings
    reduction_pct = (reduction / initial_crossings * 100) if initial_crossings > 0 else 0.0

    return {
        'initial_crossings': initial_crossings,
        'final_crossings': best_crossings,
        'reduction': reduction,
        'reduction_pct': reduction_pct,
        'total_steps': step + 1,
        'total_ops_applied': len(ops_applied),
        'effective_ops': effective_ops,
        'invalid_ops': invalid_ops,
        'ops_sequence': ops_applied,
    }


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained DT v2 model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    model = DecisionTransformer(
        embed_dim=config.get('embed_dim', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        context_len=config.get('context_len', 50),
        max_timestep=config.get('max_timestep', 500),
        max_grid_size=config.get('max_grid_size', 32),
        n_variants=config.get('n_variants', 12),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def prepare_zones(zone_pattern: str, grid_w: int, grid_h: int, seed: int = 42):
    """Create zone pattern and return zones_dict, zones_np, boundary_mask."""
    func = get_zone_function(zone_pattern)
    if func is None:
        raise ValueError(f"Unknown zone pattern: {zone_pattern}")

    # Generate zones
    if zone_pattern == 'voronoi':
        rng_state = random.getstate()
        random.seed(seed)
        zones_dict, _ = func(grid_w, grid_h, k=3)
        random.setstate(rng_state)
    elif zone_pattern == 'diagonal':
        rng_state = random.getstate()
        random.seed(seed)
        zones_dict = func(grid_w, grid_h)
        random.setstate(rng_state)
    elif zone_pattern == 'stripes':
        zones_dict = func(grid_w, grid_h, direction='v', k=3)
    elif zone_pattern == 'checkerboard':
        zones_dict = func(grid_w, grid_h, kx=2, ky=2)
    else:
        zones_dict = func(grid_w, grid_h)

    # Convert to numpy
    zones_np = np.zeros((grid_h, grid_w), dtype=np.int32)
    for (x, y), z in zones_dict.items():
        if 0 <= y < grid_h and 0 <= x < grid_w:
            zones_np[y, x] = z

    boundary_mask = compute_boundary_mask(zones_np, grid_h, grid_w)

    return zones_dict, zones_np, boundary_mask


def evaluate_on_dataset(
    model: DecisionTransformer,
    data_path: str,
    n_samples: int = 50,
    target_rtg: float = 15.0,
    max_steps: int = 100,
    context_len: int = 50,
    top_k: int = 5,
    temperature: float = 0.7,
    device: torch.device = torch.device('cuda'),
):
    """Evaluate model on trajectories from the training data (held-out)."""
    console.print(Panel.fit(
        "[bold cyan]Decision Transformer v2 - Evaluation[/bold cyan]\n"
        f"Data: {data_path}\n"
        f"Samples: {n_samples}\n"
        f"Target RTG: {target_rtg}\n"
        f"Top-k: {top_k}, Temperature: {temperature}",
        border_style="cyan"
    ))

    # Load data for zone patterns and SA baselines
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    trajectories = data['trajectories']

    # Use last n_samples trajectories as test set
    if n_samples > len(trajectories):
        n_samples = len(trajectories)
    test_trajs = trajectories[-n_samples:]

    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Evaluating", total=len(test_trajs))

        for traj in test_trajs:
            grid_w = traj.get('grid_w', 30)
            grid_h = traj.get('grid_h', 30)
            zones_np = traj['zones']
            if isinstance(zones_np, list):
                zones_np = np.array(zones_np)

            # Build zones_dict
            zones_dict = {}
            for y in range(grid_h):
                for x in range(grid_w):
                    zones_dict[(x, y)] = int(zones_np[y, x])

            boundary_mask = compute_boundary_mask(zones_np, grid_h, grid_w)

            # SA baseline from the trajectory
            sa_initial = traj['initial_crossings']
            sa_final = traj['final_crossings']
            sa_reduction = sa_initial - sa_final

            # Run DT inference
            result = run_inference(
                model=model,
                zones_dict=zones_dict,
                zones_np=zones_np,
                boundary_mask=boundary_mask,
                grid_w=grid_w,
                grid_h=grid_h,
                target_rtg=target_rtg,
                max_steps=max_steps,
                context_len=context_len,
                top_k=top_k,
                temperature=temperature,
                device=device,
                verbose=False,
            )

            result['sa_initial'] = sa_initial
            result['sa_final'] = sa_final
            result['sa_reduction'] = sa_reduction
            result['sa_reduction_pct'] = (sa_reduction / sa_initial * 100) if sa_initial > 0 else 0
            result['sa_ops'] = traj.get('total_ops', 0)
            result['zone_pattern'] = traj.get('zone_pattern', 'unknown')
            result['grid_size'] = f"{grid_w}x{grid_h}"

            all_results.append(result)
            progress.advance(task)

    # Display results
    _display_results(all_results)

    return all_results


def evaluate_zone_pattern(
    model: DecisionTransformer,
    zone_pattern: str = 'left_right',
    grid_w: int = 30,
    grid_h: int = 30,
    n_samples: int = 10,
    target_rtg: float = 15.0,
    max_steps: int = 100,
    context_len: int = 50,
    top_k: int = 5,
    temperature: float = 0.7,
    device: torch.device = torch.device('cuda'),
):
    """Evaluate model on freshly generated zone patterns."""
    console.print(Panel.fit(
        "[bold cyan]Decision Transformer v2 - Zone Pattern Evaluation[/bold cyan]\n"
        f"Pattern: {zone_pattern}\n"
        f"Grid: {grid_w}x{grid_h}\n"
        f"Samples: {n_samples}\n"
        f"Target RTG: {target_rtg}",
        border_style="cyan"
    ))

    all_results = []
    import random as _random

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Evaluating", total=n_samples)

        for i in range(n_samples):
            seed = 42 + i
            zones_dict, zones_np, boundary_mask = prepare_zones(
                zone_pattern, grid_w, grid_h, seed=seed
            )

            result = run_inference(
                model=model,
                zones_dict=zones_dict,
                zones_np=zones_np,
                boundary_mask=boundary_mask,
                grid_w=grid_w,
                grid_h=grid_h,
                target_rtg=target_rtg,
                max_steps=max_steps,
                context_len=context_len,
                top_k=top_k,
                temperature=temperature,
                device=device,
                verbose=False,
            )

            result['zone_pattern'] = zone_pattern
            result['grid_size'] = f"{grid_w}x{grid_h}"
            result['seed'] = seed
            all_results.append(result)
            progress.advance(task)

    _display_results(all_results, show_sa=False)

    return all_results


def _display_results(results: list, show_sa: bool = True):
    """Display evaluation results in Rich tables."""
    # Summary statistics
    dt_reductions = [r['reduction'] for r in results]
    dt_reduction_pcts = [r['reduction_pct'] for r in results]
    dt_ops = [r['total_ops_applied'] for r in results]
    dt_effective = [r['effective_ops'] for r in results]
    dt_invalid = [r['invalid_ops'] for r in results]
    dt_steps = [r['total_steps'] for r in results]

    # Summary table
    summary = Table(title="Evaluation Summary", box=box.ROUNDED)
    summary.add_column("Metric", style="cyan")
    summary.add_column("DT Model", style="green", justify="right")
    if show_sa:
        summary.add_column("SA Baseline", style="yellow", justify="right")

    summary.add_row(
        "Avg crossing reduction",
        f"{np.mean(dt_reductions):.1f} ({np.mean(dt_reduction_pcts):.1f}%)",
        f"{np.mean([r['sa_reduction'] for r in results]):.1f} ({np.mean([r['sa_reduction_pct'] for r in results]):.1f}%)" if show_sa else ""
    )
    summary.add_row(
        "Median crossing reduction",
        f"{np.median(dt_reductions):.1f} ({np.median(dt_reduction_pcts):.1f}%)",
        f"{np.median([r['sa_reduction'] for r in results]):.1f}" if show_sa else ""
    )
    summary.add_row(
        "Max crossing reduction",
        f"{max(dt_reductions):.0f} ({max(dt_reduction_pcts):.1f}%)",
        f"{max([r['sa_reduction'] for r in results]):.0f}" if show_sa else ""
    )
    summary.add_row(
        "Avg operations applied",
        f"{np.mean(dt_ops):.1f}",
        f"{np.mean([r.get('sa_ops', 0) for r in results]):.0f}" if show_sa else ""
    )
    summary.add_row(
        "Avg effective operations",
        f"{np.mean(dt_effective):.1f}",
        ""
    )
    summary.add_row(
        "Avg invalid operations",
        f"{np.mean(dt_invalid):.1f} ({np.mean(dt_invalid) / np.mean(dt_steps) * 100:.0f}%)" if np.mean(dt_steps) > 0 else "0",
        ""
    )
    summary.add_row(
        "Avg total steps",
        f"{np.mean(dt_steps):.1f}",
        ""
    )

    if show_sa:
        wins = sum(1 for r in results if r['reduction'] >= r['sa_reduction'])
        ties = sum(1 for r in results if r['reduction'] == r['sa_reduction'])
        summary.add_row(
            "DT >= SA",
            f"{wins}/{len(results)} ({wins/len(results)*100:.0f}%)",
            ""
        )

    # Samples with >0 reduction
    nonzero = sum(1 for r in dt_reductions if r > 0)
    summary.add_row(
        "Samples with reduction > 0",
        f"{nonzero}/{len(results)} ({nonzero/len(results)*100:.0f}%)",
        ""
    )

    console.print(summary)

    # Per-sample detail table (first 20)
    n_show = min(20, len(results))
    detail = Table(title=f"Per-Sample Results (first {n_show})", box=box.SIMPLE)
    detail.add_column("#", style="dim")
    detail.add_column("Grid")
    detail.add_column("Pattern")
    detail.add_column("Initial", justify="right")
    detail.add_column("DT Final", justify="right")
    detail.add_column("DT Red%", justify="right")
    if show_sa:
        detail.add_column("SA Final", justify="right")
        detail.add_column("SA Red%", justify="right")
    detail.add_column("Ops", justify="right")
    detail.add_column("Eff", justify="right")
    detail.add_column("Invalid", justify="right")

    for i, r in enumerate(results[:n_show]):
        row = [
            str(i + 1),
            r.get('grid_size', '?'),
            r.get('zone_pattern', '?'),
            str(r['initial_crossings']),
            str(r['final_crossings']),
            f"{r['reduction_pct']:.1f}%",
        ]
        if show_sa:
            row.extend([
                str(r.get('sa_final', '?')),
                f"{r.get('sa_reduction_pct', 0):.1f}%",
            ])
        row.extend([
            str(r['total_ops_applied']),
            str(r['effective_ops']),
            str(r['invalid_ops']),
        ])

        # Color code: green if DT matches/beats SA, red if not
        style = None
        if show_sa and r['reduction'] >= r.get('sa_reduction', float('inf')):
            style = "green"
        elif r['reduction'] > 0:
            style = "yellow"
        else:
            style = "red"

        detail.add_row(*row, style=style)

    console.print(detail)


def main():
    parser = argparse.ArgumentParser(description='DT v2 Inference & Evaluation')
    parser.add_argument('--checkpoint', default='nn_checkpoints/dt_v2/best.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data', default='model/decision_transformer/effective_dt_data.pkl',
                        help='Path to effective DT training data (for held-out eval)')
    parser.add_argument('--mode', choices=['dataset', 'pattern'], default='dataset',
                        help='Evaluation mode: dataset (held-out) or pattern (generate fresh)')
    parser.add_argument('--zone_pattern', default='left_right',
                        help='Zone pattern for pattern mode')
    parser.add_argument('--grid_w', type=int, default=30)
    parser.add_argument('--grid_h', type=int, default=30)
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--target_rtg', type=float, default=15.0,
                        help='Target crossing reduction')
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    console.print(f"Device: [green]{device}[/green]")

    # Load model
    console.print(f"\nLoading model from [cyan]{args.checkpoint}[/cyan]...")
    model, config = load_model(args.checkpoint, device)
    context_len = config.get('context_len', 50)

    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"Model parameters: [green]{n_params:,}[/green]")
    console.print(f"Context length: {context_len}")

    if args.mode == 'dataset':
        evaluate_on_dataset(
            model=model,
            data_path=args.data,
            n_samples=args.n_samples,
            target_rtg=args.target_rtg,
            max_steps=args.max_steps,
            context_len=context_len,
            top_k=args.top_k,
            temperature=args.temperature,
            device=device,
        )
    else:
        evaluate_zone_pattern(
            model=model,
            zone_pattern=args.zone_pattern,
            grid_w=args.grid_w,
            grid_h=args.grid_h,
            n_samples=args.n_samples,
            target_rtg=args.target_rtg,
            max_steps=args.max_steps,
            context_len=context_len,
            top_k=args.top_k,
            temperature=args.temperature,
            device=device,
        )


if __name__ == '__main__':
    main()
