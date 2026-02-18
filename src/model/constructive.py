"""
Constructive crossing optimization for stripe and left_right zone patterns.

Two-phase approach for even distribution:
  Phase 1 (propagate): Start from optimal zigzag (k-1 crossings), greedily
    propagate crossings to cover the FULL boundary length (near-max crossings).
  Phase 2 (trim): Selectively remove crossings using spread ordering (binary
    subdivision) to bring count into target range while keeping crossings
    evenly distributed along all boundaries.

Standalone module — does not depend on the neural network model.
Can be used by any testing environment for deterministic crossing
optimization on stripe-like zone patterns.

Usage:
    from constructive import run_constructive
    result = run_constructive(zones_np, grid_w, grid_h, zone_pattern='left_right')
    print(result['final_crossings'], result['num_operations'])
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from operations import HamiltonianSTL

# ---------------------------------------------------------------------------
# Variant definitions (transpose 8 + flip 4 = 12 total operations)
# ---------------------------------------------------------------------------

TRANSPOSE_VARIANTS = {'nl', 'nr', 'sl', 'sr', 'eb', 'ea', 'wa', 'wb'}
FLIP_VARIANTS = {'n', 's', 'e', 'w'}
ALL_VARIANTS = sorted(TRANSPOSE_VARIANTS | FLIP_VARIANTS)

# ---------------------------------------------------------------------------
# Target ranges: fraction of reference crossings to reduce by
# ---------------------------------------------------------------------------

TARGET_RANGES = {
    'left_right': (0.20, 0.40),
    'stripes':    (0.20, 0.40),
    'islands':    (0.10, 0.25),
    'voronoi':    (0.05, 0.20),
}

CV_THRESHOLDS = {
    'left_right': 0.3,
    'stripes':    0.3,
    'islands':    0.5,
    'voronoi':    0.5,
}


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def compute_crossings(h, zones_np):
    """Count boundary crossings in the Hamiltonian path."""
    H_arr = np.array(h.H, dtype=np.float32)
    V_arr = np.array(h.V, dtype=np.float32)
    h_cross = H_arr * (zones_np[:, :-1] != zones_np[:, 1:]).astype(np.float32)
    v_cross = V_arr * (zones_np[:-1, :] != zones_np[1:, :]).astype(np.float32)
    return int(h_cross.sum() + v_cross.sum())


def validate_action(op_type, x, y, variant, grid_w, grid_h):
    """Check if an operation is valid at the given position."""
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
    """Apply a single operation to the Hamiltonian path. Returns True on success."""
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


def compute_boundary_density_cv(h, zones_np, grid_w, grid_h):
    """Compute coefficient of variation of crossing density across zone-pair boundaries.

    Returns:
        cv: float (0 = perfect uniformity, <0.3 good, >1 bad)
        details: dict mapping (za, zb) -> {length, crossings, density}
    """
    H_arr = np.array(h.H, dtype=np.float32)[:grid_h, :grid_w - 1]
    V_arr = np.array(h.V, dtype=np.float32)[:grid_h - 1, :grid_w]
    zones = zones_np[:grid_h, :grid_w]

    h_diff = zones[:, :-1] != zones[:, 1:]
    h_za = zones[:, :-1][h_diff]
    h_zb = zones[:, 1:][h_diff]
    h_crossing = H_arr[h_diff]

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


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def detect_stripe_params(zones_np, grid_w, grid_h):
    """Detect stripe direction and k from zone grid.

    Vertical stripes: each column has uniform zone values.
    Horizontal stripes: each row has uniform zone values.

    Returns:
        direction: 'v' or 'h'
        k: number of distinct stripes
    """
    col_uniform = True
    for c in range(grid_w):
        if len(set(zones_np[:grid_h, c].tolist())) > 1:
            col_uniform = False
            break

    if col_uniform:
        k = len(set(zones_np[0, :grid_w].tolist()))
        return 'v', k

    row_uniform = True
    for r in range(grid_h):
        if len(set(zones_np[r, :grid_w].tolist())) > 1:
            row_uniform = False
            break

    if row_uniform:
        k = len(set(zones_np[:grid_h, 0].tolist()))
        return 'h', k

    # Fallback: vertical
    k = len(set(zones_np[0, :grid_w].tolist()))
    return 'v', k


def select_init_and_strategy(zone_pattern, zones_np, grid_w, grid_h):
    """Determine init pattern, inference strategy, and stripe params.

    Works for any k and both vertical/horizontal orientations.

    Returns:
        init_pattern: 'vertical_zigzag' or 'zigzag'
        strategy: 'constructive' or 'model'
        stripe_direction: 'v' or 'h' (for constructive) or None
        stripe_k: number of stripes (for constructive) or None
    """
    if zone_pattern in ('left_right', 'leftright', 'lr', 'stripes'):
        direction, k = detect_stripe_params(zones_np, grid_w, grid_h)
        if direction == 'v':
            return 'vertical_zigzag', 'constructive', 'v', k
        else:
            return 'zigzag', 'constructive', 'h', k

    # voronoi, islands, unknown -> model-only
    return 'zigzag', 'model', None, None


# ---------------------------------------------------------------------------
# Spread ordering for even distribution
# ---------------------------------------------------------------------------

def _spread_indices(n):
    """Return indices 0..n-1 reordered so early elements span the full range.

    Uses recursive midpoint insertion: first/last, then midpoint, then
    quarter-points, etc. Ensures that any prefix of the result covers the
    range as evenly as possible.

    Example for n=8: [0, 7, 3, 1, 5, 2, 4, 6]
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]

    result = []
    added = [False] * n

    def _add(idx):
        if 0 <= idx < n and not added[idx]:
            result.append(idx)
            added[idx] = True

    _add(0)
    _add(n - 1)

    intervals = [(0, n - 1)]
    while len(result) < n:
        next_intervals = []
        for lo, hi in intervals:
            if hi - lo <= 1:
                continue
            mid = (lo + hi) // 2
            _add(mid)
            next_intervals.append((lo, mid))
            next_intervals.append((mid, hi))
        if not next_intervals:
            break
        intervals = next_intervals

    return result


# ---------------------------------------------------------------------------
# Constructive crossing addition
# ---------------------------------------------------------------------------

def constructive_add_crossings(h, zones_np, grid_w, grid_h,
                                stripe_direction, stripe_k,
                                target_lower, target_upper):
    """Optimize crossings with even distribution using propagate-then-trim.

    Phase 1 (propagate): Start from optimal zigzag (k-1 crossings), greedily
    add crossings at boundary positions to cover the FULL boundary length.
    Does NOT stop at target — continues until no more crossings can be added.
    This ensures the entire boundary has crossings, not just one end.

    Phase 2 (trim): If above target_upper, selectively remove crossings using
    spread ordering (binary subdivision) to bring count into target range.
    Spread ordering ensures removals are evenly distributed, so the remaining
    crossings are evenly distributed along the boundary.

    Works for any k and grid size.

    Args:
        h: HamiltonianSTL instance (initialized with optimal zigzag)
        zones_np: zone grid array (grid_h x grid_w)
        grid_w, grid_h: grid dimensions
        stripe_direction: 'v' for vertical stripes, 'h' for horizontal
        stripe_k: number of stripes (determines number of boundaries = k-1)
        target_lower: minimum acceptable crossing count (most reduction from max)
        target_upper: maximum acceptable crossing count (least reduction from max)

    Returns:
        final_crossings: int
        n_ops: int (number of operations applied)
        sequence: list of operation dicts
    """
    current = compute_crossings(h, zones_np)
    sequence = []

    # Find zone boundary positions and generate candidates
    candidates = []
    if stripe_direction == 'v':
        boundary_cols = []
        for c in range(grid_w - 1):
            if zones_np[0, c] != zones_np[0, c + 1]:
                boundary_cols.append(c)

        for y in range(0, grid_h - 1, 2):
            for bc in boundary_cols:
                for x in [max(0, bc - 1), bc]:
                    if x + 2 < grid_w:
                        candidates.append((x, y))
    else:
        boundary_rows = []
        for r in range(grid_h - 1):
            if zones_np[r, 0] != zones_np[r + 1, 0]:
                boundary_rows.append(r)

        for x in range(0, grid_w - 1, 2):
            for br in boundary_rows:
                for y in [max(0, br - 1), br]:
                    if y + 2 < grid_h:
                        candidates.append((x, y))

    # ==== Phase 1: Propagate to full boundary coverage ====
    # Greedy top-to-bottom — does NOT stop at target, goes to near-max.
    # This ensures crossings exist along the entire boundary length.
    max_passes = 30
    for _pass in range(max_passes):
        progress = False

        for cx, cy in candidates:
            saved_H = [row[:] for row in h.H]
            saved_V = [row[:] for row in h.V]

            best_delta = 0
            best_variant = None
            best_op_type = None

            for variant in ALL_VARIANTS:
                op_type = 'T' if variant in TRANSPOSE_VARIANTS else 'F'
                if not validate_action(op_type, cx, cy, variant, grid_w, grid_h):
                    continue

                h.H = [row[:] for row in saved_H]
                h.V = [row[:] for row in saved_V]

                success = apply_op(h, op_type, cx, cy, variant)
                if not success:
                    continue

                new_crossings = compute_crossings(h, zones_np)
                delta = new_crossings - current

                if delta > best_delta:
                    best_delta = delta
                    best_variant = variant
                    best_op_type = op_type

            h.H = [row[:] for row in saved_H]
            h.V = [row[:] for row in saved_V]

            if best_delta > 0 and best_variant is not None:
                apply_op(h, best_op_type, cx, cy, best_variant)
                new_c = compute_crossings(h, zones_np)
                sequence.append({
                    'kind': best_op_type, 'x': cx, 'y': cy,
                    'variant': best_variant,
                    'crossings_before': current,
                    'crossings_after': new_c,
                })
                current = new_c
                progress = True

        if not progress:
            break

    # ==== Phase 2: Trim to target range with even distribution ====
    # If above target_upper, remove crossings using spread-ordered positions.
    # Spread ordering ensures removals are evenly distributed along the
    # boundary, so remaining crossings are evenly spaced.
    # Uses step=1 (not step=2) because reducers can be at any y-position
    # (odd or even, depending on boundary column parity).
    if current > target_upper:
        # Generate candidates in spread order (binary subdivision)
        trim_candidates = []
        if stripe_direction == 'v':
            y_positions = list(range(0, grid_h - 1))
            y_spread = [y_positions[i] for i in _spread_indices(len(y_positions))]

            for y in y_spread:
                for bc in boundary_cols:
                    for x in [max(0, bc - 1), bc]:
                        if x + 2 < grid_w:
                            trim_candidates.append((x, y))
        else:
            x_positions = list(range(0, grid_w - 1))
            x_spread = [x_positions[i] for i in _spread_indices(len(x_positions))]

            for x in x_spread:
                for br in boundary_rows:
                    for y in [max(0, br - 1), br]:
                        if y + 2 < grid_h:
                            trim_candidates.append((x, y))

        # Aim for the lower end of target range (more reduction)
        trim_target = target_lower + (target_upper - target_lower) // 4

        for _pass in range(max_passes):
            if current <= trim_target:
                break
            progress = False

            for cx, cy in trim_candidates:
                if current <= target_lower:
                    break

                saved_H = [row[:] for row in h.H]
                saved_V = [row[:] for row in h.V]

                best_delta = 0
                best_variant = None
                best_op_type = None

                for variant in ALL_VARIANTS:
                    op_type = 'T' if variant in TRANSPOSE_VARIANTS else 'F'
                    if not validate_action(op_type, cx, cy, variant, grid_w, grid_h):
                        continue

                    h.H = [row[:] for row in saved_H]
                    h.V = [row[:] for row in saved_V]

                    success = apply_op(h, op_type, cx, cy, variant)
                    if not success:
                        continue

                    new_crossings = compute_crossings(h, zones_np)
                    delta = new_crossings - current

                    if delta < best_delta:
                        if current + delta >= target_lower:
                            best_delta = delta
                            best_variant = variant
                            best_op_type = op_type

                h.H = [row[:] for row in saved_H]
                h.V = [row[:] for row in saved_V]

                if best_delta < 0 and best_variant is not None:
                    apply_op(h, best_op_type, cx, cy, best_variant)
                    new_c = compute_crossings(h, zones_np)
                    sequence.append({
                        'kind': best_op_type, 'x': cx, 'y': cy,
                        'variant': best_variant,
                        'crossings_before': current,
                        'crossings_after': new_c,
                    })
                    current = new_c
                    progress = True

            if not progress:
                break

    return current, len(sequence), sequence


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def run_constructive(zones_np, grid_w, grid_h, zone_pattern='left_right'):
    """Full constructive pipeline: detect params, init path, add crossings.

    Standalone function — handles everything for stripe/left_right patterns
    without a neural network model. Works for any k and grid size, both
    vertical and horizontal orientations.

    Args:
        zones_np: zone grid array (grid_h x grid_w), integer zone IDs
        grid_w, grid_h: grid dimensions
        zone_pattern: 'left_right', 'stripes', or any stripe-like pattern

    Returns:
        dict with keys:
            h: HamiltonianSTL instance with final path
            initial_crossings, final_crossings, max_crossings
            target_lower, target_upper, in_target_range
            num_operations, sequence
            final_cv, cv_details, cv_threshold
            strategy, init_pattern, stripe_direction, stripe_k

    Raises:
        ValueError: if the zone pattern is not constructive (e.g. voronoi)
    """
    init_pattern, strategy, s_dir, s_k = select_init_and_strategy(
        zone_pattern, zones_np, grid_w, grid_h,
    )

    if strategy != 'constructive':
        raise ValueError(
            f"Pattern '{zone_pattern}' is not constructive "
            f"(got strategy='{strategy}'). Use the model-based inference."
        )

    h = HamiltonianSTL(grid_w, grid_h, init_pattern=init_pattern)
    initial_crossings = compute_crossings(h, zones_np)

    max_crossings = (grid_h if s_dir == 'v' else grid_w) * (s_k - 1)
    low, high = TARGET_RANGES.get(zone_pattern, (0.20, 0.40))
    target_upper = round(max_crossings * (1.0 - low))
    target_lower = round(max_crossings * (1.0 - high))
    cv_threshold = CV_THRESHOLDS.get(zone_pattern, 0.3)

    final_crossings, n_ops, sequence = constructive_add_crossings(
        h, zones_np, grid_w, grid_h, s_dir, s_k, target_lower, target_upper,
    )

    final_cv, cv_details = compute_boundary_density_cv(h, zones_np, grid_w, grid_h)

    return {
        'h': h,
        'initial_crossings': initial_crossings,
        'final_crossings': final_crossings,
        'max_crossings': max_crossings,
        'target_lower': target_lower,
        'target_upper': target_upper,
        'in_target_range': target_lower <= final_crossings <= target_upper,
        'num_operations': n_ops,
        'sequence': sequence,
        'crossings_history': [initial_crossings] + [
            op['crossings_after'] for op in sequence
        ],
        'final_cv': final_cv,
        'cv_details': cv_details,
        'cv_threshold': cv_threshold,
        'strategy': 'constructive',
        'init_pattern': init_pattern,
        'stripe_direction': s_dir,
        'stripe_k': s_k,
    }
