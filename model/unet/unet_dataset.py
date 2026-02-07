"""
U-Net Dataset: Loads precomputed unet_data.pt (flat tensors).

Each effective operation is one training sample:
  Input: 5-channel 32x32 tensor (state + crossing indicator)
  Target: (position_y, position_x, action_class)

Trajectory-level train/val split prevents data leakage.
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
# Horizontal flip: left↔right, east↔west
HFLIP_ACTION = torch.tensor([1, 0, 3, 2, 7, 6, 5, 4, 8, 9, 11, 10], dtype=torch.long)
# Vertical flip: north↔south, above↔bottom
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


def dilate_boundary_mask(boundary_channel: torch.Tensor, dilation: int = 2) -> torch.Tensor:
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
    """Infer grid_h, grid_w from channel 3 (valid area >= 0.4)."""
    valid = (boundary_ch >= 0.4).nonzero(as_tuple=False)
    if len(valid) == 0:
        return MAX_GRID_SIZE, MAX_GRID_SIZE
    return int(valid[:, 0].max()) + 1, int(valid[:, 1].max()) + 1


def augment_hflip(state_4ch: torch.Tensor, ty: int, tx: int, action: int,
                  grid_h: int, grid_w: int):
    """Horizontal flip within valid grid region."""
    s = state_4ch.clone()
    # Ch 0 (zones), Ch 3 (boundary): flip columns [0, grid_w)
    s[0, :grid_h, :grid_w] = s[0, :grid_h, :grid_w].flip(-1)
    s[3, :grid_h, :grid_w] = s[3, :grid_h, :grid_w].flip(-1)
    # Ch 1 (H_edges): flip columns [0, grid_w-1)
    s[1, :grid_h, :grid_w - 1] = s[1, :grid_h, :grid_w - 1].flip(-1)
    # Ch 2 (V_edges): flip columns [0, grid_w)
    s[2, :grid_h - 1, :grid_w] = s[2, :grid_h - 1, :grid_w].flip(-1)
    new_tx = grid_w - 1 - tx
    new_action = HFLIP_ACTION[action].item()
    return s, ty, new_tx, new_action


def augment_vflip(state_4ch: torch.Tensor, ty: int, tx: int, action: int,
                  grid_h: int, grid_w: int):
    """Vertical flip within valid grid region."""
    s = state_4ch.clone()
    # Ch 0 (zones), Ch 3 (boundary): flip rows [0, grid_h)
    s[0, :grid_h, :grid_w] = s[0, :grid_h, :grid_w].flip(-2)
    s[3, :grid_h, :grid_w] = s[3, :grid_h, :grid_w].flip(-2)
    # Ch 1 (H_edges): flip rows [0, grid_h)
    s[1, :grid_h, :grid_w - 1] = s[1, :grid_h, :grid_w - 1].flip(-2)
    # Ch 2 (V_edges): flip rows [0, grid_h-1)
    s[2, :grid_h - 1, :grid_w] = s[2, :grid_h - 1, :grid_w].flip(-2)
    new_ty = grid_h - 1 - ty
    new_action = VFLIP_ACTION[action].item()
    return s, new_ty, tx, new_action


class UNetDataset(Dataset):
    """
    Loads flat tensors from unet_data.pt.
    Computes 5th channel and boundary mask on the fly (cheap).
    """

    def __init__(self, states: torch.Tensor, targets: torch.Tensor,
                 boundary_dilation: int = 2, augment: bool = False):
        self.states = states    # [N, 4, 32, 32]
        self.targets = targets  # [N, 3] — (y, x, action)
        self.boundary_dilation = boundary_dilation
        self.augment = augment

    def __len__(self):
        return self.states.size(0)

    def __getitem__(self, idx):
        state_4ch = self.states[idx].clone()  # [4, 32, 32]
        t = self.targets[idx]
        ty, tx, action = t[0].item(), t[1].item(), t[2].item()

        if self.augment:
            grid_h, grid_w = _detect_grid_size(state_4ch[3])
            if torch.rand(1).item() < 0.5:
                state_4ch, ty, tx, action = augment_hflip(
                    state_4ch, ty, tx, action, grid_h, grid_w)
            if torch.rand(1).item() < 0.5:
                state_4ch, ty, tx, action = augment_vflip(
                    state_4ch, ty, tx, action, grid_h, grid_w)

        crossing = compute_crossing_indicator_fast(state_4ch)
        state_5ch = torch.zeros(5, MAX_GRID_SIZE, MAX_GRID_SIZE)
        state_5ch[:4] = state_4ch
        state_5ch[4] = crossing

        boundary_mask = dilate_boundary_mask(
            state_4ch[3], dilation=self.boundary_dilation
        )

        return {
            'state': state_5ch,
            'target_y': torch.tensor(ty, dtype=torch.long),
            'target_x': torch.tensor(tx, dtype=torch.long),
            'target_action': torch.tensor(action, dtype=torch.long),
            'boundary_mask': boundary_mask,
        }


def create_train_val_split(
    data_path: str,
    val_ratio: float = 0.1,
    boundary_dilation: int = 2,
    seed: int = 42,
    augment: bool = True,
) -> Tuple[UNetDataset, UNetDataset]:
    """Load .pt, split at trajectory level. Augment train only."""
    data = torch.load(data_path, weights_only=True)
    states = data['states']       # [N, 4, 32, 32]
    targets = data['targets']     # [N, 3]
    traj_ids = data['traj_ids']   # [N]
    n_trajs = data['n_trajectories']

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_trajs)
    n_val = max(1, int(n_trajs * val_ratio))
    val_set = set(perm[:n_val].tolist())

    val_mask = torch.tensor([tid.item() in val_set for tid in traj_ids], dtype=torch.bool)
    train_mask = ~val_mask

    return (
        UNetDataset(states[train_mask], targets[train_mask], boundary_dilation, augment=augment),
        UNetDataset(states[val_mask], targets[val_mask], boundary_dilation, augment=False),
    )
