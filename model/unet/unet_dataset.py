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


class UNetDataset(Dataset):
    """
    Loads flat tensors from unet_data.pt.
    Computes 5th channel and boundary mask on the fly (cheap).
    """

    def __init__(self, states: torch.Tensor, targets: torch.Tensor,
                 boundary_dilation: int = 2):
        self.states = states    # [N, 4, 32, 32]
        self.targets = targets  # [N, 3] — (y, x, action)
        self.boundary_dilation = boundary_dilation

    def __len__(self):
        return self.states.size(0)

    def __getitem__(self, idx):
        state_4ch = self.states[idx]  # [4, 32, 32]

        crossing = compute_crossing_indicator_fast(state_4ch)
        state_5ch = torch.zeros(5, MAX_GRID_SIZE, MAX_GRID_SIZE)
        state_5ch[:4] = state_4ch
        state_5ch[4] = crossing

        boundary_mask = dilate_boundary_mask(
            state_4ch[3], dilation=self.boundary_dilation
        )

        t = self.targets[idx]
        return {
            'state': state_5ch,
            'target_y': t[0],
            'target_x': t[1],
            'target_action': t[2],
            'boundary_mask': boundary_mask,
        }


def create_train_val_split(
    data_path: str,
    val_ratio: float = 0.1,
    boundary_dilation: int = 2,
    seed: int = 42,
) -> Tuple[UNetDataset, UNetDataset]:
    """Load .pt, split at trajectory level."""
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
        UNetDataset(states[train_mask], targets[train_mask], boundary_dilation),
        UNetDataset(states[val_mask], targets[val_mask], boundary_dilation),
    )
