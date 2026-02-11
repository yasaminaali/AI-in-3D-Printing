"""
Fusion Dataset v2: Loads compact grouped fusion_data.pt.

Key changes from v1:
- Change 1:  MAX_GRID_SIZE=128, loads compact grouped storage
- Change 5:  9 input channels computed on-the-fly (incl. CoordConv)
- Change 7:  Per-sample crossing normalization (initial_crossings instead of /60)
- Change 12: State perturbation (DAgger-lite) for distribution shift
- Removed _detect_grid_size (grid_w/grid_h stored per group)
- Updated augmentation for 4ch saved + derived channels

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
from typing import Tuple

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
    """Build 9-channel 128x128 state from 4ch natural-resolution state (Change 5)."""
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


class FusionDataset(Dataset):
    """
    Loads compact grouped storage from fusion_data.pt.
    Computes 9 channels on-the-fly from 4ch stored state.
    """

    def __init__(self, states, targets, traj_ids,
                 initial_crossings, grid_ws, grid_hs,
                 history_actions, history_positions_y, history_positions_x,
                 history_crossings_before, history_crossings_after,
                 history_lengths,
                 boundary_dilation=1, augment=False, perturbation_prob=0.0):
        self.states = states                   # [N, 4, grid_h, grid_w] (padded to max in group)
        self.targets = targets                 # [N, 3]
        self.traj_ids = traj_ids               # [N]
        self.initial_crossings = initial_crossings  # [N]
        self.grid_ws = grid_ws                 # [N]
        self.grid_hs = grid_hs                 # [N]
        self.history_actions = history_actions
        self.history_positions_y = history_positions_y
        self.history_positions_x = history_positions_x
        self.history_crossings_before = history_crossings_before
        self.history_crossings_after = history_crossings_after
        self.history_lengths = history_lengths
        self.boundary_dilation = boundary_dilation
        self.augment = augment
        self.perturbation_prob = perturbation_prob
        self.max_history = history_actions.shape[1]

    def __len__(self):
        return self.states.size(0)

    def __getitem__(self, idx):
        state_4ch = self.states[idx].clone().float()  # upcast from float16
        t = self.targets[idx]
        ty, tx, action = t[0].item(), t[1].item(), t[2].item()

        grid_w = self.grid_ws[idx].item()
        grid_h = self.grid_hs[idx].item()
        init_cross = self.initial_crossings[idx].item()

        hist_act = self.history_actions[idx].clone().long()
        hist_py = self.history_positions_y[idx].clone().long()
        hist_px = self.history_positions_x[idx].clone().long()
        hist_cb = self.history_crossings_before[idx].clone().float()
        hist_ca = self.history_crossings_after[idx].clone().float()
        hist_len = self.history_lengths[idx].item()

        # --- Augmentation on 4ch state first ---
        hflipped = False
        vflipped = False
        if self.augment:
            if torch.rand(1).item() < 0.5:
                state_4ch, ty, tx, action = augment_hflip(
                    state_4ch, ty, tx, action, grid_h, grid_w)
                hflipped = True
                valid = hist_len
                if valid > 0:
                    hist_px[:valid] = (grid_w - 1 - hist_px[:valid]).clamp(0, grid_w - 1)
                    hist_act[:valid] = HFLIP_ACTION[hist_act[:valid].long()]
            if torch.rand(1).item() < 0.5:
                state_4ch, ty, tx, action = augment_vflip(
                    state_4ch, ty, tx, action, grid_h, grid_w)
                vflipped = True
                valid = hist_len
                if valid > 0:
                    hist_py[:valid] = (grid_h - 1 - hist_py[:valid]).clamp(0, grid_h - 1)
                    hist_act[:valid] = VFLIP_ACTION[hist_act[:valid].long()]

        # --- Build 9ch state from augmented 4ch (Change 5) ---
        state_9ch = build_9ch_state(state_4ch, grid_h, grid_w, init_cross)

        # CoordConv channels already flip correctly because build_9ch_state
        # generates them from scratch after augmentation is applied.

        # --- Boundary mask ---
        boundary_mask = dilate_boundary_mask(
            state_4ch[3], grid_h, grid_w, dilation=self.boundary_dilation
        )

        # --- History mask ---
        hist_mask = torch.zeros(self.max_history)
        hist_mask[:hist_len] = 1.0

        # --- Change 7: Normalize crossings by initial_crossings ---
        norm_factor = max(init_cross, 1.0)
        hist_cb_norm = hist_cb / norm_factor
        hist_ca_norm = hist_ca / norm_factor

        return {
            'state': state_9ch,
            'target_y': torch.tensor(ty, dtype=torch.long),
            'target_x': torch.tensor(tx, dtype=torch.long),
            'target_action': torch.tensor(action, dtype=torch.long),
            'boundary_mask': boundary_mask,
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
    boundary_dilation: int = 1,
    seed: int = 42,
    augment: bool = True,
    perturbation_prob: float = 0.0,
) -> Tuple['FusionDataset', 'FusionDataset']:
    """Load compact grouped .pt, pad to 128x128, split at trajectory level."""
    data = torch.load(data_path, weights_only=False)

    grid_groups = data['grid_groups']
    n_trajs_total = data['n_trajectories']
    max_history = data['max_history']

    # Concatenate all groups, padding states to MAX_GRID_SIZE
    all_states = []
    all_targets = []
    all_traj_ids = []
    all_initial_crossings = []
    all_grid_ws = []
    all_grid_hs = []
    all_hist_actions = []
    all_hist_py = []
    all_hist_px = []
    all_hist_cb = []
    all_hist_ca = []
    all_hist_lengths = []

    for key, group in grid_groups.items():
        gw = group['grid_w']
        gh = group['grid_h']
        states = group['states'].float()  # [N, 4, gh, gw]
        N = states.shape[0]

        # Pad to 4 x MAX_GRID_SIZE x MAX_GRID_SIZE
        if gh < MAX_GRID_SIZE or gw < MAX_GRID_SIZE:
            padded = torch.zeros(N, 4, MAX_GRID_SIZE, MAX_GRID_SIZE)
            padded[:, :, :gh, :gw] = states
            states = padded

        all_states.append(states)
        all_targets.append(group['targets'].long())
        all_traj_ids.append(group['traj_ids'].long())
        all_initial_crossings.append(group['initial_crossings'].long())
        all_grid_ws.append(torch.full((N,), gw, dtype=torch.long))
        all_grid_hs.append(torch.full((N,), gh, dtype=torch.long))
        all_hist_actions.append(group['history_actions'].long())
        all_hist_py.append(group['history_positions_y'].long())
        all_hist_px.append(group['history_positions_x'].long())
        all_hist_cb.append(group['history_crossings_before'].float())
        all_hist_ca.append(group['history_crossings_after'].float())
        all_hist_lengths.append(group['history_lengths'].long())

    states = torch.cat(all_states, dim=0)
    targets = torch.cat(all_targets, dim=0)
    traj_ids = torch.cat(all_traj_ids, dim=0)
    initial_crossings = torch.cat(all_initial_crossings, dim=0)
    grid_ws = torch.cat(all_grid_ws, dim=0)
    grid_hs = torch.cat(all_grid_hs, dim=0)
    hist_actions = torch.cat(all_hist_actions, dim=0)
    hist_py = torch.cat(all_hist_py, dim=0)
    hist_px = torch.cat(all_hist_px, dim=0)
    hist_cb = torch.cat(all_hist_cb, dim=0)
    hist_ca = torch.cat(all_hist_ca, dim=0)
    hist_lengths = torch.cat(all_hist_lengths, dim=0)

    # Trajectory-level split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_trajs_total)
    n_val = max(1, int(n_trajs_total * val_ratio))
    val_set = set(perm[:n_val].tolist())

    val_mask = torch.tensor([tid.item() in val_set for tid in traj_ids], dtype=torch.bool)
    train_mask = ~val_mask

    def make_ds(mask, aug, perturb_prob):
        return FusionDataset(
            states[mask], targets[mask], traj_ids[mask],
            initial_crossings[mask], grid_ws[mask], grid_hs[mask],
            hist_actions[mask], hist_py[mask], hist_px[mask],
            hist_cb[mask], hist_ca[mask], hist_lengths[mask],
            boundary_dilation=boundary_dilation, augment=aug,
            perturbation_prob=perturb_prob,
        )

    return make_ds(train_mask, augment, perturbation_prob), make_ds(val_mask, False, 0.0)
