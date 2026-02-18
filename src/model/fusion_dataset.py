"""
Fusion Dataset v2: Loads compact grouped fusion_data.pt.

Key changes from v1:
- Change 1:  MAX_GRID_SIZE=128, loads compact grouped storage
- Change 5:  9 input channels computed on-the-fly (incl. CoordConv)
- Change 7:  Per-sample crossing normalization (initial_crossings instead of /60)
- Change 12: State perturbation (DAgger-lite) for distribution shift
- Removed _detect_grid_size (grid_w/grid_h stored per group)
- Memory-efficient: states stored at native resolution, padded on-the-fly

9 channels:
  Ch 0: zones / max_zone
  Ch 1: H edges
  Ch 2: V edges
  Ch 3: grid validity (1.0 inside grid, 0.0 outside)
  Ch 4: zone boundary (1.0 at boundary cells, 0.0 elsewhere)
  Ch 5: crossing count (normalized by max)
  Ch 6: progress (current_crossings / initial_crossings, tiled)
  Ch 7: y_coord (normalized, 0->1 over grid height)
  Ch 8: x_coord (normalized, 0->1 over grid width)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple, List

MAX_GRID_SIZE = 128

VARIANT_MAP = {
    'nl': 0, 'nr': 1, 'sl': 2, 'sr': 3, 'eb': 4, 'ea': 5, 'wa': 6, 'wb': 7,
    'n': 8, 's': 9, 'e': 10, 'w': 11
}

# Action remapping tables for augmentation
HFLIP_ACTION = torch.tensor([1, 0, 3, 2, 7, 6, 5, 4, 8, 9, 11, 10], dtype=torch.long)
VFLIP_ACTION = torch.tensor([2, 3, 0, 1, 5, 4, 7, 6, 9, 8, 10, 11], dtype=torch.long)


def compute_crossing_indicator(state_4ch, grid_h, grid_w):
    """Vectorized crossing count per cell (Change 5: crossing density instead of binary)."""
    zones_norm = state_4ch[0, :grid_h, :grid_w]
    h_edges = state_4ch[1, :grid_h, :grid_w - 1]
    v_edges = state_4ch[2, :grid_h - 1, :grid_w]

    crossing_count = torch.zeros(grid_h, grid_w)

    # Horizontal crossings
    h_zone_diff = (torch.abs(zones_norm[:, :-1] - zones_norm[:, 1:]) > 0.01)
    h_edge_present = (h_edges > 0.5)
    h_crossing = (h_zone_diff & h_edge_present).float()
    crossing_count[:, :-1] += h_crossing
    crossing_count[:, 1:] += h_crossing

    # Vertical crossings
    v_zone_diff = (torch.abs(zones_norm[:-1, :] - zones_norm[1:, :]) > 0.01)
    v_edge_present = (v_edges > 0.5)
    v_crossing = (v_zone_diff & v_edge_present).float()
    crossing_count[:-1, :] += v_crossing
    crossing_count[1:, :] += v_crossing

    return crossing_count


def dilate_boundary_mask(boundary_channel, grid_h, grid_w, dilation=1):
    """Compute dilated boundary mask from boundary_combined channel."""
    boundary = (boundary_channel[:grid_h, :grid_w] > 0.9).float()
    if dilation > 0:
        kernel_size = 2 * dilation + 1
        boundary_4d = boundary.unsqueeze(0).unsqueeze(0)
        dilated = F.max_pool2d(
            boundary_4d, kernel_size=kernel_size, stride=1, padding=dilation
        ).squeeze(0).squeeze(0)
    else:
        dilated = boundary
    # Mask to valid grid area
    valid_grid = (boundary_channel[:grid_h, :grid_w] >= 0.4).float()
    result = torch.zeros(MAX_GRID_SIZE, MAX_GRID_SIZE)
    result[:grid_h, :grid_w] = dilated * valid_grid
    return result


def augment_hflip(state_4ch, ty, tx, action, grid_h, grid_w):
    """Horizontal flip on 4ch state at natural resolution."""
    s = state_4ch.clone()
    s[0, :grid_h, :grid_w] = s[0, :grid_h, :grid_w].flip(-1)
    s[3, :grid_h, :grid_w] = s[3, :grid_h, :grid_w].flip(-1)
    s[1, :grid_h, :grid_w - 1] = s[1, :grid_h, :grid_w - 1].flip(-1)
    s[2, :grid_h - 1, :grid_w] = s[2, :grid_h - 1, :grid_w].flip(-1)
    new_tx = grid_w - 1 - tx
    new_action = HFLIP_ACTION[action].item()
    return s, ty, new_tx, new_action


def augment_vflip(state_4ch, ty, tx, action, grid_h, grid_w):
    """Vertical flip on 4ch state at natural resolution."""
    s = state_4ch.clone()
    s[0, :grid_h, :grid_w] = s[0, :grid_h, :grid_w].flip(-2)
    s[3, :grid_h, :grid_w] = s[3, :grid_h, :grid_w].flip(-2)
    s[1, :grid_h, :grid_w - 1] = s[1, :grid_h, :grid_w - 1].flip(-2)
    s[2, :grid_h - 1, :grid_w] = s[2, :grid_h - 1, :grid_w].flip(-2)
    new_ty = grid_h - 1 - ty
    new_action = VFLIP_ACTION[action].item()
    return s, new_ty, tx, new_action


def build_9ch_state(state_4ch, grid_h, grid_w, initial_crossings):
    """Build 9-channel 128x128 state from 4ch (CPU, single-sample — used by inference)."""
    state = torch.zeros(9, MAX_GRID_SIZE, MAX_GRID_SIZE)

    # Ch 0-2: zones, H edges, V edges (pad from natural to 128)
    state[0, :grid_h, :grid_w] = state_4ch[0, :grid_h, :grid_w]
    state[1, :grid_h, :grid_w - 1] = state_4ch[1, :grid_h, :grid_w - 1]
    state[2, :grid_h - 1, :grid_w] = state_4ch[2, :grid_h - 1, :grid_w]

    # Ch 3: grid validity (1.0 inside, 0.0 outside)
    state[3, :grid_h, :grid_w] = 1.0

    # Ch 4: zone boundary (from boundary_combined channel: 1.0 where >0.9)
    state[4, :grid_h, :grid_w] = (state_4ch[3, :grid_h, :grid_w] > 0.9).float()

    # Ch 5: crossing count (normalized by max)
    crossing_count = compute_crossing_indicator(state_4ch, grid_h, grid_w)
    max_cross = crossing_count.max()
    if max_cross > 0:
        state[5, :grid_h, :grid_w] = crossing_count / max_cross
    else:
        state[5, :grid_h, :grid_w] = 0.0

    # Ch 6: progress (current_crossings / initial_crossings)
    current_crossings = crossing_count.sum().item() / 2.0  # each crossing counted twice
    init_c = max(initial_crossings, 1)
    progress = min(current_crossings / init_c, 1.0)
    state[6, :grid_h, :grid_w] = progress

    # Ch 7: y_coord (normalized 0->1)
    if grid_h > 1:
        y_coords = torch.linspace(0, 1, grid_h).unsqueeze(1).expand(grid_h, grid_w)
        state[7, :grid_h, :grid_w] = y_coords

    # Ch 8: x_coord (normalized 0->1)
    if grid_w > 1:
        x_coords = torch.linspace(0, 1, grid_w).unsqueeze(0).expand(grid_h, grid_w)
        state[8, :grid_h, :grid_w] = x_coords

    return state


@torch.no_grad()
def build_9ch_batch_gpu(state_4ch, grid_h, grid_w, initial_crossings, boundary_dilation=1):
    """
    Build 9-channel states + boundary masks on GPU from batched 4ch states.

    Replaces per-sample CPU build_9ch_state + dilate_boundary_mask with a single
    batched GPU operation for ~4x training throughput.

    Args:
        state_4ch: [B, 4, 128, 128] float, zero-padded 4ch states on GPU
        grid_h: [B] long, grid heights per sample
        grid_w: [B] long, grid widths per sample
        initial_crossings: [B] float, initial crossing counts
        boundary_dilation: int, boundary mask dilation radius

    Returns:
        state_9ch: [B, 9, 128, 128] float on GPU
        boundary_mask: [B, 128, 128] float on GPU
    """
    B = state_4ch.shape[0]
    device = state_4ch.device
    S = MAX_GRID_SIZE

    state_9ch = torch.zeros(B, 9, S, S, device=device)

    # Ch 0-2: zones, H edges, V edges (copy from padded 4ch)
    state_9ch[:, :3] = state_4ch[:, :3]

    # Ch 3: grid validity (boundary_combined >= 0.4 marks valid cells; padding is 0)
    grid_validity = (state_4ch[:, 3] >= 0.4).float()
    state_9ch[:, 3] = grid_validity

    # Ch 4: zone boundary (boundary_combined > 0.9)
    state_9ch[:, 4] = (state_4ch[:, 3] > 0.9).float()

    # Ch 5: crossing count per cell, normalized by per-sample max
    zones = state_4ch[:, 0]
    h_edges = state_4ch[:, 1]
    v_edges = state_4ch[:, 2]

    h_zone_diff = (torch.abs(zones[:, :, :-1] - zones[:, :, 1:]) > 0.01)
    h_edge_present = (h_edges[:, :, :-1] > 0.5)
    h_crossing = (h_zone_diff & h_edge_present).float()

    v_zone_diff = (torch.abs(zones[:, :-1, :] - zones[:, 1:, :]) > 0.01)
    v_edge_present = (v_edges[:, :-1, :] > 0.5)
    v_crossing = (v_zone_diff & v_edge_present).float()

    crossing_count = torch.zeros(B, S, S, device=device)
    crossing_count[:, :, :-1] += h_crossing
    crossing_count[:, :, 1:] += h_crossing
    crossing_count[:, :-1, :] += v_crossing
    crossing_count[:, 1:, :] += v_crossing

    max_cross = crossing_count.reshape(B, -1).max(dim=1).values.clamp(min=1.0)
    state_9ch[:, 5] = crossing_count / max_cross.view(B, 1, 1)

    # Ch 6: progress (current_crossings / initial_crossings), tiled over valid grid
    current_crossings = crossing_count.reshape(B, -1).sum(dim=1) / 2.0
    init_c = initial_crossings.float().clamp(min=1.0)
    progress = (current_crossings / init_c).clamp(max=1.0)
    state_9ch[:, 6] = progress.view(B, 1, 1) * grid_validity

    # Ch 7: y_coord (0->1 over grid height)
    y_base = torch.arange(S, device=device, dtype=torch.float32).view(1, S, 1)
    y_norm = y_base / (grid_h.float().view(B, 1, 1) - 1).clamp(min=1)
    state_9ch[:, 7] = y_norm.clamp(max=1.0) * grid_validity

    # Ch 8: x_coord (0->1 over grid width)
    x_base = torch.arange(S, device=device, dtype=torch.float32).view(1, 1, S)
    x_norm = x_base / (grid_w.float().view(B, 1, 1) - 1).clamp(min=1)
    state_9ch[:, 8] = x_norm.clamp(max=1.0) * grid_validity

    # Boundary mask with dilation
    boundary = state_9ch[:, 4:5]  # [B, 1, S, S]
    if boundary_dilation > 0:
        ks = 2 * boundary_dilation + 1
        dilated = F.max_pool2d(boundary, kernel_size=ks, stride=1, padding=boundary_dilation)
    else:
        dilated = boundary
    boundary_mask = dilated.squeeze(1) * grid_validity

    return state_9ch, boundary_mask


class FusionDataset(Dataset):
    """
    Memory-efficient dataset: stores states at native resolution per group.
    Pads to 128x128 on-the-fly in __getitem__.
    Uses an index to map flat idx -> (group_idx, local_idx).
    """

    def __init__(self, group_states: List[torch.Tensor],
                 group_targets: List[torch.Tensor],
                 group_traj_ids: List[torch.Tensor],
                 group_initial_crossings: List[torch.Tensor],
                 group_grid_ws: List[int],
                 group_grid_hs: List[int],
                 group_hist_actions: List[torch.Tensor],
                 group_hist_py: List[torch.Tensor],
                 group_hist_px: List[torch.Tensor],
                 group_hist_cb: List[torch.Tensor],
                 group_hist_ca: List[torch.Tensor],
                 group_hist_lengths: List[torch.Tensor],
                 max_history: int,
                 augment=False, perturbation_prob=0.0):
        self.group_states = group_states
        self.group_targets = group_targets
        self.group_traj_ids = group_traj_ids
        self.group_initial_crossings = group_initial_crossings
        self.group_grid_ws = group_grid_ws
        self.group_grid_hs = group_grid_hs
        self.group_hist_actions = group_hist_actions
        self.group_hist_py = group_hist_py
        self.group_hist_px = group_hist_px
        self.group_hist_cb = group_hist_cb
        self.group_hist_ca = group_hist_ca
        self.group_hist_lengths = group_hist_lengths
        self.max_history = max_history
        self.augment = augment
        self.perturbation_prob = perturbation_prob

        # Build flat index -> (group_idx, local_idx)
        self._index = []
        for g_idx, states in enumerate(group_states):
            n = states.shape[0]
            for local_idx in range(n):
                self._index.append((g_idx, local_idx))
        self._len = len(self._index)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        g_idx, local_idx = self._index[idx]

        # Load at native resolution (float16) and upcast
        state_4ch = self.group_states[g_idx][local_idx].float()
        grid_w = self.group_grid_ws[g_idx]
        grid_h = self.group_grid_hs[g_idx]

        t = self.group_targets[g_idx][local_idx]
        ty, tx, action = t[0].item(), t[1].item(), t[2].item()
        init_cross = int(self.group_initial_crossings[g_idx][local_idx].item())

        hist_act = self.group_hist_actions[g_idx][local_idx].long()
        hist_py = self.group_hist_py[g_idx][local_idx].long()
        hist_px = self.group_hist_px[g_idx][local_idx].long()
        hist_cb = self.group_hist_cb[g_idx][local_idx].float()
        hist_ca = self.group_hist_ca[g_idx][local_idx].float()
        hist_len = int(self.group_hist_lengths[g_idx][local_idx].item())

        # --- Augmentation on 4ch state first ---
        if self.augment:
            if torch.rand(1).item() < 0.5:
                state_4ch, ty, tx, action = augment_hflip(
                    state_4ch, ty, tx, action, grid_h, grid_w)
                if hist_len > 0:
                    hist_px[:hist_len] = (grid_w - 1 - hist_px[:hist_len]).clamp(0, grid_w - 1)
                    hist_act[:hist_len] = HFLIP_ACTION[hist_act[:hist_len]]
            if torch.rand(1).item() < 0.5:
                state_4ch, ty, tx, action = augment_vflip(
                    state_4ch, ty, tx, action, grid_h, grid_w)
                if hist_len > 0:
                    hist_py[:hist_len] = (grid_h - 1 - hist_py[:hist_len]).clamp(0, grid_h - 1)
                    hist_act[:hist_len] = VFLIP_ACTION[hist_act[:hist_len]]

        # --- Pad 4ch to 128x128 (GPU builds 9ch + boundary mask) ---
        padded = torch.zeros(4, MAX_GRID_SIZE, MAX_GRID_SIZE)
        padded[0, :grid_h, :grid_w] = state_4ch[0, :grid_h, :grid_w]
        padded[1, :grid_h, :grid_w - 1] = state_4ch[1, :grid_h, :grid_w - 1]
        padded[2, :grid_h - 1, :grid_w] = state_4ch[2, :grid_h - 1, :grid_w]
        padded[3, :grid_h, :grid_w] = state_4ch[3, :grid_h, :grid_w]

        # --- History mask ---
        hist_mask = torch.zeros(self.max_history)
        hist_mask[:hist_len] = 1.0

        # --- Normalize crossings by initial_crossings ---
        norm_factor = max(init_cross, 1.0)
        hist_cb_norm = hist_cb / norm_factor
        hist_ca_norm = hist_ca / norm_factor

        return {
            'state_4ch': padded,
            'grid_h': torch.tensor(grid_h, dtype=torch.long),
            'grid_w': torch.tensor(grid_w, dtype=torch.long),
            'initial_crossings': torch.tensor(init_cross, dtype=torch.float),
            'target_y': torch.tensor(ty, dtype=torch.long),
            'target_x': torch.tensor(tx, dtype=torch.long),
            'target_action': torch.tensor(action, dtype=torch.long),
            'history_actions': hist_act.long(),
            'history_positions_y': hist_py.long(),
            'history_positions_x': hist_px.long(),
            'history_crossings_before': hist_cb_norm,
            'history_crossings_after': hist_ca_norm,
            'history_mask': hist_mask,
        }


def create_train_val_split(
    data_path: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    augment: bool = True,
    perturbation_prob: float = 0.0,
) -> Tuple['FusionDataset', 'FusionDataset']:
    """Load compact grouped .pt, keep at native resolution, split at trajectory level."""
    data = torch.load(data_path, weights_only=False, mmap=True)

    grid_groups = data['grid_groups']
    n_trajs_total = data['n_trajectories']
    max_history = data['max_history']

    # Trajectory-level split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_trajs_total)
    n_val = max(1, int(n_trajs_total * val_ratio))
    val_set = set(perm[:n_val].tolist())

    # Build per-group train/val splits at native resolution (no padding!)
    train_groups = {k: [] for k in [
        'states', 'targets', 'traj_ids', 'initial_crossings',
        'hist_actions', 'hist_py', 'hist_px', 'hist_cb', 'hist_ca', 'hist_lengths',
        'grid_ws', 'grid_hs',
    ]}
    val_groups = {k: [] for k in train_groups}

    for key, group in grid_groups.items():
        gw = group['grid_w']
        gh = group['grid_h']
        states = group['states']       # float16 at native resolution [N, 4, gh, gw]
        targets = group['targets']     # int16
        traj_ids = group['traj_ids']   # int32
        init_cross = group['initial_crossings']
        h_act = group['history_actions']
        h_py = group['history_positions_y']
        h_px = group['history_positions_x']
        h_cb = group['history_crossings_before']
        h_ca = group['history_crossings_after']
        h_len = group['history_lengths']

        # Split mask for this group
        val_mask = torch.tensor([tid.item() in val_set for tid in traj_ids], dtype=torch.bool)
        train_mask = ~val_mask

        for mask, dest in [(train_mask, train_groups), (val_mask, val_groups)]:
            if mask.any():
                dest['states'].append(states[mask])
                dest['targets'].append(targets[mask])
                dest['traj_ids'].append(traj_ids[mask])
                dest['initial_crossings'].append(init_cross[mask])
                dest['hist_actions'].append(h_act[mask])
                dest['hist_py'].append(h_py[mask])
                dest['hist_px'].append(h_px[mask])
                dest['hist_cb'].append(h_cb[mask])
                dest['hist_ca'].append(h_ca[mask])
                dest['hist_lengths'].append(h_len[mask])
                dest['grid_ws'].append(gw)
                dest['grid_hs'].append(gh)

    del data, grid_groups  # free early

    def make_ds(groups, aug, perturb_prob):
        return FusionDataset(
            group_states=groups['states'],
            group_targets=groups['targets'],
            group_traj_ids=groups['traj_ids'],
            group_initial_crossings=groups['initial_crossings'],
            group_grid_ws=groups['grid_ws'],
            group_grid_hs=groups['grid_hs'],
            group_hist_actions=groups['hist_actions'],
            group_hist_py=groups['hist_py'],
            group_hist_px=groups['hist_px'],
            group_hist_cb=groups['hist_cb'],
            group_hist_ca=groups['hist_ca'],
            group_hist_lengths=groups['hist_lengths'],
            max_history=max_history,
            augment=aug,
            perturbation_prob=perturb_prob,
        )

    return make_ds(train_groups, augment, perturbation_prob), make_ds(val_groups, False, 0.0)
