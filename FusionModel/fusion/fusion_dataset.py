"""
Fusion Dataset: Loads precomputed fusion_data.pt (flat tensors + history).

Each effective operation is one training sample:
  Input: 5-channel 32x32 tensor (state + crossing indicator)
         + K most recent effective ops as history context
  Target: (position_y, position_x, action_class)

Trajectory-level train/val split prevents data leakage.
Augmentation flips both state AND history positions/actions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple

MAX_GRID_SIZE = 32

VARIANT_MAP = {
    'nl': 0, 'nr': 1, 'sl': 2, 'sr': 3, 'eb': 4, 'ea': 5, 'wa': 6, 'wb': 7,
    'n': 8, 's': 9, 'e': 10, 'w': 11
}

# Action remapping tables for augmentation
HFLIP_ACTION = torch.tensor([1, 0, 3, 2, 7, 6, 5, 4, 8, 9, 11, 10], dtype=torch.long)
VFLIP_ACTION = torch.tensor([2, 3, 0, 1, 5, 4, 7, 6, 9, 8, 10, 11], dtype=torch.long)


def compute_crossing_indicator_fast(state: torch.Tensor) -> torch.Tensor:
    """Vectorized crossing indicator computation."""
    zones_norm = state[0]
    h_edges = state[1]
    v_edges = state[2]

    crossing = torch.zeros(MAX_GRID_SIZE, MAX_GRID_SIZE)

    h_zone_diff = torch.abs(zones_norm[:, :-1] - zones_norm[:, 1:]) > 0.01
    h_edge_present = h_edges[:, :-1] > 0.5
    h_crossing = h_zone_diff & h_edge_present
    crossing[:, :-1] = torch.maximum(crossing[:, :-1], h_crossing.float())
    crossing[:, 1:] = torch.maximum(crossing[:, 1:], h_crossing.float())

    v_zone_diff = torch.abs(zones_norm[:-1, :] - zones_norm[1:, :]) > 0.01
    v_edge_present = v_edges[:-1, :] > 0.5
    v_crossing = v_zone_diff & v_edge_present
    crossing[:-1, :] = torch.maximum(crossing[:-1, :], v_crossing.float())
    crossing[1:, :] = torch.maximum(crossing[1:, :], v_crossing.float())

    return crossing


def dilate_boundary_mask(boundary_channel: torch.Tensor, dilation: int = 1) -> torch.Tensor:
    boundary = (boundary_channel > 0.9).float()
    if dilation > 0:
        kernel_size = 2 * dilation + 1
        boundary_4d = boundary.unsqueeze(0).unsqueeze(0)
        dilated = F.max_pool2d(
            boundary_4d, kernel_size=kernel_size, stride=1, padding=dilation
        ).squeeze(0).squeeze(0)
    else:
        dilated = boundary
    valid_grid = (boundary_channel >= 0.4).float()
    return dilated * valid_grid


def _detect_grid_size(boundary_ch: torch.Tensor) -> Tuple[int, int]:
    valid = (boundary_ch >= 0.4).nonzero(as_tuple=False)
    if len(valid) == 0:
        return MAX_GRID_SIZE, MAX_GRID_SIZE
    return int(valid[:, 0].max()) + 1, int(valid[:, 1].max()) + 1


def augment_hflip(state_4ch, ty, tx, action, grid_h, grid_w):
    s = state_4ch.clone()
    s[0, :grid_h, :grid_w] = s[0, :grid_h, :grid_w].flip(-1)
    s[3, :grid_h, :grid_w] = s[3, :grid_h, :grid_w].flip(-1)
    s[1, :grid_h, :grid_w - 1] = s[1, :grid_h, :grid_w - 1].flip(-1)
    s[2, :grid_h - 1, :grid_w] = s[2, :grid_h - 1, :grid_w].flip(-1)
    new_tx = grid_w - 1 - tx
    new_action = HFLIP_ACTION[action].item()
    return s, ty, new_tx, new_action


def augment_vflip(state_4ch, ty, tx, action, grid_h, grid_w):
    s = state_4ch.clone()
    s[0, :grid_h, :grid_w] = s[0, :grid_h, :grid_w].flip(-2)
    s[3, :grid_h, :grid_w] = s[3, :grid_h, :grid_w].flip(-2)
    s[1, :grid_h, :grid_w - 1] = s[1, :grid_h, :grid_w - 1].flip(-2)
    s[2, :grid_h - 1, :grid_w] = s[2, :grid_h - 1, :grid_w].flip(-2)
    new_ty = grid_h - 1 - ty
    new_action = VFLIP_ACTION[action].item()
    return s, new_ty, tx, new_action


class FusionDataset(Dataset):
    """
    Loads flat tensors from fusion_data.pt.
    Computes 5th channel and boundary mask on the fly.
    Returns history context for RNN branch.
    """

    def __init__(self, states, targets, traj_ids,
                 history_actions, history_positions_y, history_positions_x,
                 history_crossings_before, history_crossings_after,
                 history_lengths,
                 boundary_dilation=1, augment=False):
        self.states = states
        self.targets = targets
        self.traj_ids = traj_ids
        self.history_actions = history_actions
        self.history_positions_y = history_positions_y
        self.history_positions_x = history_positions_x
        self.history_crossings_before = history_crossings_before
        self.history_crossings_after = history_crossings_after
        self.history_lengths = history_lengths
        self.boundary_dilation = boundary_dilation
        self.augment = augment
        self.max_history = history_actions.shape[1]

    def __len__(self):
        return self.states.size(0)

    def __getitem__(self, idx):
        state_4ch = self.states[idx].clone()
        t = self.targets[idx]
        ty, tx, action = t[0].item(), t[1].item(), t[2].item()

        hist_act = self.history_actions[idx].clone()
        hist_py = self.history_positions_y[idx].clone()
        hist_px = self.history_positions_x[idx].clone()
        hist_cb = self.history_crossings_before[idx].clone().float()
        hist_ca = self.history_crossings_after[idx].clone().float()
        hist_len = self.history_lengths[idx].item()

        if self.augment:
            grid_h, grid_w = _detect_grid_size(state_4ch[3])
            if torch.rand(1).item() < 0.5:
                state_4ch, ty, tx, action = augment_hflip(
                    state_4ch, ty, tx, action, grid_h, grid_w)
                valid = hist_len
                if valid > 0:
                    hist_px[:valid] = (grid_w - 1 - hist_px[:valid]).clamp(0, MAX_GRID_SIZE - 1)
                    hist_act[:valid] = HFLIP_ACTION[hist_act[:valid].long()]
            if torch.rand(1).item() < 0.5:
                state_4ch, ty, tx, action = augment_vflip(
                    state_4ch, ty, tx, action, grid_h, grid_w)
                valid = hist_len
                if valid > 0:
                    hist_py[:valid] = (grid_h - 1 - hist_py[:valid]).clamp(0, MAX_GRID_SIZE - 1)
                    hist_act[:valid] = VFLIP_ACTION[hist_act[:valid].long()]

        crossing = compute_crossing_indicator_fast(state_4ch)
        state_5ch = torch.zeros(5, MAX_GRID_SIZE, MAX_GRID_SIZE)
        state_5ch[:4] = state_4ch
        state_5ch[4] = crossing

        boundary_mask = dilate_boundary_mask(
            state_4ch[3], dilation=self.boundary_dilation
        )

        hist_mask = torch.zeros(self.max_history)
        hist_mask[:hist_len] = 1.0

        # Normalize crossings (divide by 60, ~2x max for 30x30)
        hist_cb_norm = hist_cb / 60.0
        hist_ca_norm = hist_ca / 60.0

        return {
            'state': state_5ch,
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
) -> Tuple['FusionDataset', 'FusionDataset']:
    """Load .pt, split at trajectory level. Augment train only."""
    data = torch.load(data_path, weights_only=False)
    states = data['states']
    targets = data['targets']
    traj_ids = data['traj_ids']
    n_trajs = data['n_trajectories']
    history_actions = data['history_actions']
    history_positions_y = data['history_positions_y']
    history_positions_x = data['history_positions_x']
    history_crossings_before = data['history_crossings_before']
    history_crossings_after = data['history_crossings_after']
    history_lengths = data['history_lengths']

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_trajs)
    n_val = max(1, int(n_trajs * val_ratio))
    val_set = set(perm[:n_val].tolist())

    val_mask = torch.tensor([tid.item() in val_set for tid in traj_ids], dtype=torch.bool)
    train_mask = ~val_mask

    def make_ds(mask, aug):
        return FusionDataset(
            states[mask], targets[mask], traj_ids[mask],
            history_actions[mask], history_positions_y[mask],
            history_positions_x[mask],
            history_crossings_before[mask], history_crossings_after[mask],
            history_lengths[mask],
            boundary_dilation=boundary_dilation, augment=aug,
        )

    return make_ds(train_mask, augment), make_ds(val_mask, False)
