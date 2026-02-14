"""
FusionNet v3 (CNN+RNN+FiLM) Inference for Hamiltonian Path Optimization.

Proposal-filter inference: model proposes candidate positions, brute-force
tries ALL 12 actions at each, keeps the best crossing-reducing operation.

Key decisions:
- Decision 1: Target crossing optimization (not minimize to zero)
- Decision 2: Pattern-specific target ranges (left_right/stripes [20-40%],
              islands [10-25%], voronoi [5-20%])
- Decision 3: No history seeding (empty = FiLM identity = pure CNN first step)
- Decision 5: Full trajectories in training (no truncation)
- Decision 6: Two-phase inference (reduction + redistribution)
- Decision 8: Init pattern = zigzag (matches training data)
- Decision 9: Density-based uniformity via boundary CV

Usage:
    PYTHONPATH=$(pwd):$PYTHONPATH python FusionModel/fusion/inference_fusion.py \\
        --checkpoint FusionModel/nn_checkpoints/fusion/best.pt
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

MAX_GRID_SIZE = 128
MAX_HISTORY = 32
TRANSPOSE_VARIANTS = {'nl', 'nr', 'sl', 'sr', 'eb', 'ea', 'wa', 'wb'}
FLIP_VARIANTS = {'n', 's', 'e', 'w'}
ALL_PATTERNS = ['left_right', 'voronoi', 'islands', 'stripes']

# Decision 2: Pattern-specific target reduction ranges (inference-only)
TARGET_RANGES = {
    'left_right': (0.20, 0.40),
    'stripes':    (0.20, 0.40),
    'islands':    (0.10, 0.25),
    'voronoi':    (0.05, 0.20),
}

# Decision 9: CV thresholds for acceptable uniformity
CV_THRESHOLDS = {
    'left_right': 0.3,
    'stripes':    0.3,
    'islands':    0.5,
    'voronoi':    0.5,
}


# ---------------------------------------------------------------------------
# Change 16: Infer correct init_pattern
# ---------------------------------------------------------------------------

def _infer_init_pattern(zone_pattern, zones_np, grid_w, grid_h):
    """All existing dataset was generated with default zigzag (horizontal).

    The SA code at the time of dataset generation used HamiltonianSTL(w, h)
    without specifying init_pattern, which defaults to 'zigzag'. The 'auto'
    logic was added later but the dataset was never regenerated.
    """
    return 'zigzag'


# ---------------------------------------------------------------------------
# Grid helpers
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


def compute_boundary_density_cv(h, zones_np, grid_w, grid_h):
    """Compute coefficient of variation of crossing density across zone-pair boundaries.

    Groups all zone-boundary edges by zone pair (min_id, max_id), computes
    density = crossings / boundary_length for each pair, then returns the CV
    (std / mean) of those densities.

    Returns:
        cv: float, 0 = perfect uniformity, <0.3 good, >1 bad
        details: dict mapping (za, zb) -> {length, crossings, density}
    """
    H_arr = np.array(h.H, dtype=np.float32)[:grid_h, :grid_w - 1]
    V_arr = np.array(h.V, dtype=np.float32)[:grid_h - 1, :grid_w]

    zones = zones_np[:grid_h, :grid_w]

    # Horizontal zone boundaries: between (r, c) and (r, c+1)
    h_diff = zones[:, :-1] != zones[:, 1:]
    h_za = zones[:, :-1][h_diff]
    h_zb = zones[:, 1:][h_diff]
    h_crossing = H_arr[h_diff]

    # Vertical zone boundaries: between (r, c) and (r+1, c)
    v_diff = zones[:-1, :] != zones[1:, :]
    v_za = zones[:-1, :][v_diff]
    v_zb = zones[1:, :][v_diff]
    v_crossing = V_arr[v_diff]

    if len(h_za) == 0 and len(v_za) == 0:
        return 0.0, {}

    all_za = np.concatenate([h_za, v_za]) if len(h_za) > 0 and len(v_za) > 0 else (h_za if len(h_za) > 0 else v_za)
    all_zb = np.concatenate([h_zb, v_zb]) if len(h_zb) > 0 and len(v_zb) > 0 else (h_zb if len(h_zb) > 0 else v_zb)
    all_cross = np.concatenate([h_crossing, v_crossing]) if len(h_crossing) > 0 and len(v_crossing) > 0 else (h_crossing if len(h_crossing) > 0 else v_crossing)

    pair_min = np.minimum(all_za, all_zb)
    pair_max = np.maximum(all_za, all_zb)

    unique_pairs = set(zip(pair_min.tolist(), pair_max.tolist()))

    densities = []
    details = {}
    for za, zb in unique_pairs:
        mask = (pair_min == za) & (pair_max == zb)
        length = int(mask.sum())
        crossings = int((all_cross[mask] > 0.5).sum())
        density = crossings / length if length > 0 else 0.0
        densities.append(density)
        details[(za, zb)] = {'length': length, 'crossings': crossings, 'density': density}

    if len(densities) < 2:
        return 0.0, details

    densities = np.array(densities)
    mean_d = densities.mean()
    if mean_d == 0:
        return 0.0, details
    cv = float(densities.std() / mean_d)
    return cv, details


def encode_state_9ch(zones_np, boundary_mask, h, grid_w, grid_h, initial_crossings):
    """Build 9-channel state tensor (Change 5)."""
    state = torch.zeros(9, MAX_GRID_SIZE, MAX_GRID_SIZE)
    max_zone = max(zones_np.max(), 1)
    zones_t = torch.from_numpy(zones_np.astype(np.float32))

    # Ch 0: zones
    state[0, :grid_h, :grid_w] = zones_t / max_zone

    # Ch 1: H edges
    H_arr = _h_edges_np(h)
    state[1, :grid_h, :grid_w - 1] = torch.from_numpy(H_arr)

    # Ch 2: V edges
    V_arr = _v_edges_np(h)
    state[2, :grid_h - 1, :grid_w] = torch.from_numpy(V_arr)

    # Ch 3: grid validity
    state[3, :grid_h, :grid_w] = 1.0

    # Ch 4: zone boundary
    state[4] = (boundary_mask > 0.5).float()

    # Ch 5: crossing count (normalized)
    H_t = torch.from_numpy(H_arr)
    V_t = torch.from_numpy(V_arr)
    crossing_count = torch.zeros(grid_h, grid_w)
    h_cross = H_t * (zones_t[:, :-1] != zones_t[:, 1:]).float()
    v_cross = V_t * (zones_t[:-1, :] != zones_t[1:, :]).float()
    crossing_count[:, :-1] += h_cross
    crossing_count[:, 1:] += h_cross
    crossing_count[:-1, :] += v_cross
    crossing_count[1:, :] += v_cross
    max_cross = crossing_count.max()
    if max_cross > 0:
        state[5, :grid_h, :grid_w] = crossing_count / max_cross

    # Ch 6: progress
    current_c = crossing_count.sum().item() / 2.0
    init_c = max(initial_crossings, 1)
    state[6, :grid_h, :grid_w] = min(current_c / init_c, 1.0)

    # Ch 7: y_coord
    if grid_h > 1:
        y_coords = torch.linspace(0, 1, grid_h).unsqueeze(1).expand(grid_h, grid_w)
        state[7, :grid_h, :grid_w] = y_coords

    # Ch 8: x_coord
    if grid_w > 1:
        x_coords = torch.linspace(0, 1, grid_w).unsqueeze(0).expand(grid_h, grid_w)
        state[8, :grid_h, :grid_w] = x_coords

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
            result, status = h.transpose_subgrid(sub, variant)
            return 'transposed_' in status
        elif op_type == 'F':
            if variant in ['n', 's']:
                sub = h.get_subgrid((x, y), (x + 1, y + 2))
            else:
                sub = h.get_subgrid((x, y), (x + 2, y + 1))
            result, status = h.flip_subgrid(sub, variant)
            return 'flipped_' in status
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
# History buffer helpers (Change 7: normalize by initial_crossings)
# ---------------------------------------------------------------------------

def build_history_tensors(history_buffer, max_history, initial_crossings, device):
    hist_act = torch.zeros(1, max_history, dtype=torch.long, device=device)
    hist_py = torch.zeros(1, max_history, dtype=torch.long, device=device)
    hist_px = torch.zeros(1, max_history, dtype=torch.long, device=device)
    hist_cb = torch.zeros(1, max_history, dtype=torch.float, device=device)
    hist_ca = torch.zeros(1, max_history, dtype=torch.float, device=device)
    hist_mask = torch.zeros(1, max_history, dtype=torch.float, device=device)

    norm = max(initial_crossings, 1)
    for i, entry in enumerate(history_buffer):
        hist_act[0, i] = entry['action']
        hist_py[0, i] = entry['py']
        hist_px[0, i] = entry['px']
        hist_cb[0, i] = entry['cb'] / norm
        hist_ca[0, i] = entry['ca'] / norm
        hist_mask[0, i] = 1.0

    return hist_act, hist_py, hist_px, hist_cb, hist_ca, hist_mask


# ---------------------------------------------------------------------------
# Change 10: History seeding — greedy boundary operations
# ---------------------------------------------------------------------------

def seed_history(h, zones_np, grid_w, grid_h, boundary_mask, max_seed=5):
    """Run greedy boundary search and return first few effective ops for history seeding."""
    seeded = []
    current_crossings = compute_crossings(h, zones_np)

    bpos = boundary_mask.nonzero(as_tuple=False)
    if len(bpos) == 0:
        return seeded, current_crossings

    all_variants = list(VARIANT_MAP.keys())
    for _ in range(max_seed * 10):  # try up to 50 ops
        if len(seeded) >= max_seed:
            break

        # Random boundary position
        idx = torch.randint(0, len(bpos), (1,)).item()
        py, px = bpos[idx, 0].item(), bpos[idx, 1].item()

        # Random variant
        variant = all_variants[torch.randint(0, len(all_variants), (1,)).item()]
        op_type = 'T' if variant in TRANSPOSE_VARIANTS else 'F'

        if not validate_action(op_type, px, py, variant, grid_w, grid_h):
            continue

        saved = save_grid_state(h)
        success = apply_op(h, op_type, px, py, variant)

        if success:
            new_crossings = compute_crossings(h, zones_np)
            if new_crossings < current_crossings:
                seeded.append({
                    'action': VARIANT_MAP[variant],
                    'py': py,
                    'px': px,
                    'cb': current_crossings,
                    'ca': new_crossings,
                    'kind': op_type,
                    'variant': variant,
                    'x': px, 'y': py,
                    'crossings_before': current_crossings,
                    'crossings_after': new_crossings,
                })
                current_crossings = new_crossings
            else:
                restore_grid_state(h, saved)
        else:
            restore_grid_state(h, saved)

    return seeded, current_crossings


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
# Model candidate position extraction (proposal generator)
# ---------------------------------------------------------------------------

def _get_candidate_positions(model, h_obj, zones_np, boundary_mask, grid_w, grid_h,
                             initial_crossings, history_buffer, dilated_mask, device,
                             max_history=32, n_positions=50):
    """Extract top-N unique positions from all K hypothesis heads.

    The model is used purely as a proposal generator — it suggests WHERE to
    operate, but we try ALL 12 actions at each position externally.

    Returns list of (py, px, confidence) tuples, sorted by confidence descending.
    """
    state = encode_state_9ch(zones_np, boundary_mask, h_obj, grid_w, grid_h, initial_crossings)
    state_batch = state.unsqueeze(0).to(device)

    hist_act, hist_py, hist_px, hist_cb, hist_ca, hist_mask = \
        build_history_tensors(history_buffer, max_history, initial_crossings, device)

    pos_logits, act_logits = model(
        state_batch, hist_act, hist_py, hist_px, hist_cb, hist_ca, hist_mask
    )

    mask_1d = dilated_mask.reshape(-1).bool()
    K = pos_logits.shape[1]
    n_valid = int(mask_1d.sum().item())
    if n_valid == 0:
        return []

    # Collect top positions from each hypothesis
    per_hyp_top = max(n_positions // K, 10)

    pos_scores = {}
    for kk in range(K):
        pos_flat = pos_logits[0, kk].reshape(-1)
        pos_masked = pos_flat.masked_fill(~mask_1d, float('-inf'))
        probs = torch.softmax(pos_masked, dim=-1)
        n_top = min(per_hyp_top, n_valid)
        topk_vals, topk_idx = probs.topk(n_top)

        for i in range(n_top):
            flat_idx = topk_idx[i].item()
            score = topk_vals[i].item()
            py = flat_idx // MAX_GRID_SIZE
            px = flat_idx % MAX_GRID_SIZE
            key = (py, px)
            if key not in pos_scores or score > pos_scores[key]:
                pos_scores[key] = score

    # Sort by confidence, return top n_positions
    sorted_positions = sorted(pos_scores.items(), key=lambda x: -x[1])[:n_positions]
    return [(py, px, score) for (py, px), score in sorted_positions]


# ---------------------------------------------------------------------------
# Proposal-filter inference with all fixes + Decisions 1-9
# ---------------------------------------------------------------------------

def run_inference(
    model,
    zones_np,
    boundary_mask,
    grid_w, grid_h,
    zone_pattern='unknown',
    max_history=32,
    max_steps=150,
    n_candidates=50,
    device=torch.device('cuda'),
    verbose=False,
):
    """
    Proposal-filter inference:
    1. Model proposes top-N candidate positions (proposal generator)
    2. Try ALL 12 actions at each position (brute-force filter)
    3. Keep the operation with the best crossing reduction
    4. Two-phase: reduction then redistribution (Decisions 1, 2, 6, 9)
    """
    # Decision 8: Correct init_pattern
    init_pattern = _infer_init_pattern(zone_pattern, zones_np, grid_w, grid_h)
    h = HamiltonianSTL(grid_w, grid_h, init_pattern=init_pattern)
    initial_crossings = compute_crossings(h, zones_np)
    current_crossings = initial_crossings

    dilated_mask = dilate_mask(boundary_mask, dilation=1)
    valid_area = torch.zeros(MAX_GRID_SIZE, MAX_GRID_SIZE)
    valid_area[:grid_h, :grid_w] = 1.0
    dilated_mask = (dilated_mask * valid_area).to(device)

    history_buffer = deque(maxlen=max_history)
    sequence = []
    crossings_history = [initial_crossings]

    # Decision 2: target ranges
    low, high = TARGET_RANGES.get(zone_pattern, (0.10, 0.40))
    target_upper = round(initial_crossings * (1.0 - low))
    target_lower = round(initial_crossings * (1.0 - high))
    cv_threshold = CV_THRESHOLDS.get(zone_pattern, 0.5)

    # Decision 6 + 9: uniformity tracking
    initial_cv, _ = compute_boundary_density_cv(h, zones_np, grid_w, grid_h)
    current_cv = initial_cv
    phase = 'reduction'

    # Adaptive thresholds
    max_failures = max(30, initial_crossings * 2)
    max_steps_adaptive = max(max_steps, initial_crossings * 5)

    no_improve_count = 0
    total_attempts = 0
    invalid_count = 0
    redistribution_steps = 0
    max_redistribution = 30

    all_variants = list(VARIANT_MAP.keys())

    model.eval()
    with torch.no_grad():
        for step in range(max_steps_adaptive):
            # --- Phase transition check ---
            if current_crossings <= target_upper:
                phase = 'redistribution'
                if current_cv < cv_threshold:
                    if verbose:
                        console.print(
                            f"  Step {step}: TARGET REACHED — "
                            f"crossings={current_crossings} (<={target_upper}), "
                            f"CV={current_cv:.3f} (<{cv_threshold})"
                        )
                    break
                if redistribution_steps >= max_redistribution:
                    if verbose:
                        console.print(
                            f"  Step {step}: redistribution limit — "
                            f"CV={current_cv:.3f}"
                        )
                    break

            # --- Get candidate positions from model ---
            positions = _get_candidate_positions(
                model, h, zones_np, boundary_mask, grid_w, grid_h,
                initial_crossings, history_buffer, dilated_mask, device,
                max_history=max_history, n_positions=n_candidates,
            )

            if not positions:
                if verbose:
                    console.print(f"  Step {step}: no candidate positions")
                break

            # --- Save state once, try ALL actions at each position ---
            saved_H = [row[:] for row in h.H]
            saved_V = [row[:] for row in h.V]

            best_result = None   # (new_crossings, new_cv, op_type, px, py, variant)
            best_score = float('-inf')

            for py, px, confidence in positions:
                for variant in all_variants:
                    op_type = 'T' if variant in TRANSPOSE_VARIANTS else 'F'

                    if not validate_action(op_type, px, py, variant, grid_w, grid_h):
                        continue

                    total_attempts += 1

                    # Restore to saved state before each attempt
                    h.H = [row[:] for row in saved_H]
                    h.V = [row[:] for row in saved_V]

                    success = apply_op(h, op_type, px, py, variant)
                    if not success:
                        invalid_count += 1
                        continue

                    new_crossings = compute_crossings(h, zones_np)

                    if phase == 'reduction':
                        if new_crossings < current_crossings:
                            score = current_crossings - new_crossings
                            if score > best_score:
                                best_score = score
                                best_result = (new_crossings, None,
                                               op_type, px, py, variant)
                    else:
                        # Redistribution: accept if crossings don't increase
                        # and either crossings decrease or uniformity improves
                        if new_crossings <= current_crossings:
                            new_cv, _ = compute_boundary_density_cv(
                                h, zones_np, grid_w, grid_h
                            )
                            reduction = current_crossings - new_crossings
                            cv_gain = max(0.0, current_cv - new_cv)
                            score = (reduction
                                     + 0.5 * cv_gain * current_crossings)
                            if (score > best_score
                                    and (new_crossings < current_crossings
                                         or new_cv < current_cv)):
                                best_score = score
                                best_result = (new_crossings, new_cv,
                                               op_type, px, py, variant)

            # --- Restore and apply best result ---
            h.H = [row[:] for row in saved_H]
            h.V = [row[:] for row in saved_V]

            if best_result is not None:
                new_crossings, new_cv, op_type, px, py, variant = best_result
                apply_op(h, op_type, px, py, variant)

                if new_cv is None:
                    new_cv, _ = compute_boundary_density_cv(
                        h, zones_np, grid_w, grid_h
                    )
                current_cv = new_cv

                sequence.append({
                    'kind': op_type, 'x': px, 'y': py,
                    'variant': variant,
                    'crossings_before': current_crossings,
                    'crossings_after': new_crossings,
                })
                history_buffer.append({
                    'action': VARIANT_MAP.get(variant, 0),
                    'py': min(py, grid_h - 1),
                    'px': min(px, grid_w - 1),
                    'cb': current_crossings,
                    'ca': new_crossings,
                })
                current_crossings = new_crossings
                crossings_history.append(current_crossings)
                no_improve_count = 0
                if phase == 'redistribution':
                    redistribution_steps += 1

                if verbose:
                    console.print(
                        f"  Step {step+1} [{phase}]: "
                        f"{op_type}({variant}) at ({px},{py}) "
                        f"-> crossings {current_crossings}, "
                        f"CV={current_cv:.3f}"
                    )
            else:
                no_improve_count += 1
                if no_improve_count >= max_failures:
                    if verbose:
                        console.print(
                            f"  Stopping: no improvement for "
                            f"{no_improve_count} steps"
                        )
                    break

    reduction = initial_crossings - current_crossings
    reduction_pct = (
        (reduction / initial_crossings * 100)
        if initial_crossings > 0 else 0
    )
    final_cv, boundary_details = compute_boundary_density_cv(
        h, zones_np, grid_w, grid_h
    )

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
        # Uniformity + target metrics
        'initial_cv': initial_cv,
        'final_cv': final_cv,
        'target_upper': target_upper,
        'target_lower': target_lower,
        'target_range_str': f"[{low*100:.0f}%, {high*100:.0f}%]",
        'in_target_range': target_lower <= current_crossings <= target_upper,
        'phase_at_stop': phase,
        'redistribution_steps': redistribution_steps,
        'cv_threshold': cv_threshold,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get('args', {})

    model = FusionNet(
        in_channels=args.get('in_channels', 9),
        base_features=args.get('base_features', 48),
        n_hypotheses=args.get('n_hypotheses', 4),
        max_history=args.get('max_history', MAX_HISTORY),
        rnn_hidden=args.get('rnn_hidden', 192),
        rnn_layers=args.get('rnn_layers', 2),
        max_grid_size=args.get('max_grid_size', MAX_GRID_SIZE),
        rnn_dropout=args.get('rnn_dropout', 0.15),
    ).to(device)

    state_dict = checkpoint['model_state_dict']
    # Handle DDP-wrapped state dicts
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    return model, args


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_all_patterns(
    model,
    jsonl_path,
    n_per_pattern=25,
    max_steps=150,
    max_history=32,
    n_candidates=50,
    device=torch.device('cuda'),
    visualize=False,
    vis_dir='nn_checkpoints/fusion/vis',
):
    console.print(Panel.fit(
        "[bold cyan]FusionNet v3 — Proposal-Filter Inference Evaluation[/bold cyan]\n"
        f"Data: {jsonl_path}\n"
        f"Patterns: {', '.join(ALL_PATTERNS)}\n"
        f"Samples per pattern: {n_per_pattern}\n"
        f"Max steps: {max_steps} | Max history: {max_history}\n"
        f"Candidate positions per step: {n_candidates}\n"
        f"Actions tried per position: 12 (all)\n"
        f"Target ranges: {dict(TARGET_RANGES)}",
        border_style="cyan"
    ))

    with open(jsonl_path) as f:
        lines = f.readlines()

    pattern_lines = defaultdict(list)
    for line in lines:
        traj = json.loads(line.strip())
        pattern = traj.get('zone_pattern', 'unknown')
        pattern_lines[pattern].append(line)

    comp_table = Table(title="[bold]Dataset Composition[/bold]", box=box.SIMPLE)
    comp_table.add_column("Zone Pattern", style="cyan")
    comp_table.add_column("Total Trajectories", justify="right")
    comp_table.add_column("Test Samples", justify="right")
    for p in ALL_PATTERNS:
        available = len(pattern_lines[p])
        n_test = min(n_per_pattern, available)
        comp_table.add_row(p, str(available), str(n_test))
    console.print(comp_table)

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
                zone_pattern=pattern,
                max_history=max_history,
                max_steps=max_steps,
                n_candidates=n_candidates,
                device=device,
            )

            # SA baseline
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
                init_pat = _infer_init_pattern(pattern, zones_np, grid_w, grid_h)
                h_init = HamiltonianSTL(grid_w, grid_h, init_pattern=init_pat)
                plot_cycle_on_ax(ax1, h_init, zones_np,
                                 f"Initial (crossings={result['initial_crossings']})")
                plot_cycle_on_ax(ax2, result['final_h'], zones_np,
                                 f"FusionNet v3 (crossings={result['final_crossings']}, "
                                 f"ops={result['num_operations']})")
                fig.suptitle(
                    f"Sample {len(all_results)} | {pattern} | "
                    f"{result['grid_size']} | {n_zones} zones",
                    fontsize=14, fontweight='bold')
                fig.tight_layout()
                fig.savefig(os.path.join(vis_dir, f"{pattern}_{len(all_results)}.png"),
                            dpi=200, bbox_inches='tight')
                plt.close(fig)

            in_target = "Y" if result.get('in_target_range') else "N"
            status = (
                f"{pattern} | red={result['reduction']}/{result['sa_reduction']} "
                f"| CV={result.get('final_cv', 0):.2f} "
                f"| target={in_target} | ops={result['num_operations']}"
            )
            progress.update(task, advance=1, status=status)

    _display_all_pattern_results(all_results)
    return all_results


def _display_all_pattern_results(results):
    if not results:
        console.print("[bold red]No results to display[/bold red]")
        return

    pattern_results = defaultdict(list)
    for r in results:
        pattern_results[r['zone_pattern']].append(r)

    per_pattern = Table(
        title="[bold]Per-Pattern Results: FusionNet v3 (Proposal-Filter) vs SA Baseline[/bold]",
        box=box.ROUNDED
    )
    per_pattern.add_column("Pattern", style="cyan")
    per_pattern.add_column("N", justify="right", style="dim")
    per_pattern.add_column("Grid", justify="center")
    per_pattern.add_column("Fusion Red", justify="right", style="green")
    per_pattern.add_column("SA Red", justify="right", style="yellow")
    per_pattern.add_column("Ops", justify="right")
    per_pattern.add_column("Avg CV", justify="right")
    per_pattern.add_column("In Target", justify="right")
    per_pattern.add_column(">= SA", justify="right")
    per_pattern.add_column("Red > 0", justify="right")

    for pattern in ALL_PATTERNS:
        pr = pattern_results.get(pattern, [])
        if not pr:
            per_pattern.add_row(
                pattern, "0", "-", "-", "-", "-", "-", "-", "-", "-"
            )
            continue

        n = len(pr)
        grid_sizes = set(r['grid_size'] for r in pr)
        grid_str = ', '.join(sorted(grid_sizes))

        fusion_reds = [r['reduction'] for r in pr]
        fusion_pcts = [r['reduction_pct'] for r in pr]
        sa_reds = [r['sa_reduction'] for r in pr]
        fusion_ops = [r['num_operations'] for r in pr]
        final_cvs = [r.get('final_cv', 0) for r in pr]
        in_target = sum(1 for r in pr if r.get('in_target_range', False))

        wins = sum(1 for r in pr if r['reduction'] >= r['sa_reduction'])
        nonzero = sum(1 for r in fusion_reds if r > 0)

        target_range = TARGET_RANGES.get(pattern, (0.1, 0.4))
        cv_thresh = CV_THRESHOLDS.get(pattern, 0.5)
        avg_cv = np.mean(final_cvs)
        cv_style = "green" if avg_cv < cv_thresh else "yellow"

        per_pattern.add_row(
            pattern, str(n), grid_str,
            f"{np.mean(fusion_reds):.1f} ({np.mean(fusion_pcts):.1f}%)",
            f"{np.mean(sa_reds):.1f}",
            f"{np.mean(fusion_ops):.1f}",
            f"[{cv_style}]{avg_cv:.2f}[/{cv_style}]",
            f"{in_target}/{n} ({in_target/n*100:.0f}%)",
            f"{wins}/{n} ({wins/n*100:.0f}%)",
            f"{nonzero}/{n} ({nonzero/n*100:.0f}%)",
        )

    console.print(per_pattern)

    # Overall summary
    reductions = [r['reduction'] for r in results]
    reduction_pcts = [r['reduction_pct'] for r in results]
    ops = [r['num_operations'] for r in results]
    sa_reductions = [r['sa_reduction'] for r in results]
    sa_reduction_pcts = [r['sa_reduction_pct'] for r in results]
    sa_ops = [r.get('sa_ops', 0) for r in results]
    sa_eff_ops = [r.get('sa_effective_ops', 0) for r in results]

    summary = Table(title="[bold]Overall: FusionNet v3 (Proposal-Filter) vs SA Baseline[/bold]", box=box.ROUNDED)
    summary.add_column("Metric", style="cyan")
    summary.add_column("FusionNet v3", style="green", justify="right")
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
        "Avg operations used",
        f"{np.mean(ops):.1f}",
        f"{np.mean(sa_ops):.0f} total / {np.mean(sa_eff_ops):.1f} effective",
    )

    wins = sum(1 for r in results if r['reduction'] >= r['sa_reduction'])
    close = sum(1 for r in results if abs(r['reduction'] - r['sa_reduction']) <= 2)
    nonzero = sum(1 for r in reductions if r > 0)

    summary.add_row("FusionNet >= SA", f"{wins}/{len(results)} ({wins/len(results)*100:.0f}%)", "")
    summary.add_row("Within 2 of SA", f"{close}/{len(results)} ({close/len(results)*100:.0f}%)", "")
    summary.add_row("Reduction > 0", f"{nonzero}/{len(results)} ({nonzero/len(results)*100:.0f}%)", "")

    # Uniformity metrics
    all_cvs = [r.get('final_cv', 0) for r in results]
    in_target_count = sum(1 for r in results if r.get('in_target_range', False))
    summary.add_row("Avg final CV", f"{np.mean(all_cvs):.3f}", "")
    summary.add_row(
        "In target range",
        f"{in_target_count}/{len(results)} ({in_target_count/len(results)*100:.0f}%)",
        "",
    )

    if np.mean(ops) > 0 and np.mean(sa_ops) > 0:
        summary.add_row(
            "Efficiency (red/op)",
            f"{np.mean(reductions)/max(np.mean(ops), 0.01):.2f}",
            f"{np.mean(sa_reductions)/max(np.mean(sa_eff_ops), 0.01):.2f} effective",
        )

    summary.add_row("Hamiltonicity preserved", "[bold green]100%[/bold green] (by construction)", "100%")
    console.print(summary)

    # Per-sample detail
    for pattern in ALL_PATTERNS:
        pr = pattern_results.get(pattern, [])
        if not pr:
            continue

        n_show = min(15, len(pr))
        target_str = f"{TARGET_RANGES.get(pattern, (0.1,0.4))}"
        detail = Table(
            title=f"[bold]{pattern}[/bold] — Per-Sample (first {n_show}) | target={target_str}",
            box=box.SIMPLE
        )
        detail.add_column("#", style="dim")
        detail.add_column("Grid")
        detail.add_column("Init", justify="right")
        detail.add_column("Final", justify="right")
        detail.add_column("Red%", justify="right")
        detail.add_column("SA Red%", justify="right")
        detail.add_column("Ops", justify="right")
        detail.add_column("CV", justify="right")
        detail.add_column("Target", justify="center")
        detail.add_column("Phase", justify="center")

        for i, r in enumerate(pr[:n_show]):
            in_tgt = r.get('in_target_range', False)
            cv_val = r.get('final_cv', 0)
            cv_thresh = r.get('cv_threshold', 0.5)

            if in_tgt and cv_val < cv_thresh:
                style = "bold green"
            elif r['reduction'] >= r.get('sa_reduction', float('inf')):
                style = "green"
            elif r['reduction'] > 0:
                style = "yellow"
            else:
                style = "red"

            detail.add_row(
                str(i + 1), r.get('grid_size', '?'),
                str(r['initial_crossings']), str(r['final_crossings']),
                f"{r['reduction_pct']:.1f}%",
                f"{r.get('sa_reduction_pct', 0):.1f}%",
                str(r['num_operations']),
                f"{cv_val:.2f}",
                "[green]Y[/green]" if in_tgt else "[red]N[/red]",
                r.get('phase_at_stop', '?'),
                style=style,
            )

        console.print(detail)

    console.print(Panel.fit(
        f"[bold]Final Summary — FusionNet v3[/bold]\n"
        f"Patterns tested: {', '.join(ALL_PATTERNS)}\n"
        f"Total samples: {len(results)}\n"
        f"Avg initial crossings: {np.mean([r['initial_crossings'] for r in results]):.1f}\n"
        f"Avg final crossings (Fusion): {np.mean([r['final_crossings'] for r in results]):.1f}\n"
        f"Avg final crossings (SA): {np.mean([r['sa_final'] for r in results]):.1f}\n"
        f"Avg operations: {np.mean(ops):.1f}\n"
        f"Avg reduction: {np.mean(reduction_pcts):.1f}% vs SA: {np.mean(sa_reduction_pcts):.1f}%\n"
        f"Avg final CV: {np.mean(all_cvs):.3f}\n"
        f"In target range: {in_target_count}/{len(results)} ({in_target_count/len(results)*100:.0f}%)",
        border_style="cyan"
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='FusionNet v3 Inference & Evaluation')
    parser.add_argument('--checkpoint', default='FusionModel/nn_checkpoints/fusion/best.pt')
    parser.add_argument('--jsonl', default='datasets/final_dataset.jsonl')
    parser.add_argument('--n_per_pattern', type=int, default=25)
    parser.add_argument('--max_steps', type=int, default=150)
    parser.add_argument('--n_candidates', type=int, default=50,
                        help='Number of candidate positions per step (model proposals)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--vis_dir', default='nn_checkpoints/fusion/vis')

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    console.print(f"\n[bold]Loading FusionNet v3 from {args.checkpoint}...[/bold]")
    model, model_args = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    max_history = model_args.get('max_history', MAX_HISTORY)
    console.print(f"  Parameters: {n_params:,}")
    console.print(f"  Max history: {max_history}")
    console.print(f"  Device: {device}")
    console.print(f"  Candidate positions: {args.n_candidates}")
    console.print(f"  Actions per position: 12 (brute-force)")

    results = evaluate_all_patterns(
        model=model,
        jsonl_path=args.jsonl,
        n_per_pattern=args.n_per_pattern,
        max_steps=args.max_steps,
        max_history=max_history,
        n_candidates=args.n_candidates,
        device=device,
        visualize=args.visualize,
        vis_dir=args.vis_dir,
    )

    if args.visualize:
        console.print(f"\n[bold]Visualizations saved to {args.vis_dir}/[/bold]")

    output_path = Path(args.checkpoint).parent / 'inference_results.json'
    non_serializable = {'sequence', 'crossings_history', 'final_h', 'boundary_details'}
    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items() if k not in non_serializable}
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
