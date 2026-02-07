"""
Decision Transformer Dataset v2 - Loads precomputed effective training data.

Designed for:
- Effective-only trajectories (condensed from SA data)
- Variable grid sizes (padded to max_grid_size)
- Sliding window sampling with configurable context_len
- Trajectory-level train/val split (no data leakage)
"""

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional


MAX_GRID_SIZE = 32

# Canonical mappings
OP_TYPE_MAP = {'N': 0, 'T': 1, 'F': 2}
VARIANT_MAP = {
    'nl': 0, 'nr': 1, 'sl': 2, 'sr': 3, 'eb': 4, 'ea': 5, 'wa': 6, 'wb': 7,
    'n': 8, 's': 9, 'e': 10, 'w': 11
}


class EffectiveDTDataset(Dataset):
    """
    Dataset for Decision Transformer training on effective-only trajectories.

    Each sample is a context_len window from a condensed trajectory where
    most steps are crossing-reducing operations (with a few context ops).

    All grid states are precomputed and stored in memory.
    """

    def __init__(
        self,
        data_path: str,
        context_len: int = 50,
        max_grid_size: int = MAX_GRID_SIZE,
        trajectory_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            data_path: Path to effective_dt_data.pkl
            context_len: Context window length for transformer
            max_grid_size: Max grid dimension (for padding)
            trajectory_indices: If provided, only use these trajectory indices
                               (for train/val splitting at trajectory level)
        """
        self.context_len = context_len
        self.max_grid_size = max_grid_size

        # Load precomputed data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        all_trajectories = data['trajectories']

        # Filter to requested indices
        if trajectory_indices is not None:
            self.trajectories = [all_trajectories[i] for i in trajectory_indices]
        else:
            self.trajectories = all_trajectories

        # Build sample index: (traj_idx_in_self, window_start)
        self.samples = self._build_sample_index()

    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """Build sliding window samples from all trajectories."""
        samples = []
        stride = max(1, self.context_len // 2)

        for traj_idx, traj in enumerate(self.trajectories):
            n_steps = traj['num_condensed_steps']
            if n_steps == 0:
                continue

            # Sliding windows
            for start in range(0, n_steps, stride):
                if start + 1 <= n_steps:  # Need at least 1 step
                    samples.append((traj_idx, start))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_idx, start = self.samples[idx]
        traj = self.trajectories[traj_idx]
        steps = traj['condensed_steps']

        end = min(start + self.context_len, len(steps))
        seq_len = end - start

        # Allocate padded tensors
        states = torch.zeros(self.context_len, 4, self.max_grid_size, self.max_grid_size)
        actions = torch.zeros(self.context_len, 4, dtype=torch.long)
        rtg = torch.zeros(self.context_len, 1)
        timesteps = torch.zeros(self.context_len, dtype=torch.long)
        mask = torch.zeros(self.context_len, dtype=torch.float)

        # Fill from condensed steps
        for i in range(seq_len):
            step = steps[start + i]

            # State (already a 4×MAX_GRID_SIZE×MAX_GRID_SIZE tensor from precomputation)
            state_tensor = step['state']
            if isinstance(state_tensor, torch.Tensor):
                states[i] = state_tensor
            else:
                states[i] = torch.tensor(state_tensor, dtype=torch.float32)

            # Action
            actions[i, 0] = step['action'][0]  # op_type
            actions[i, 1] = min(step['action'][1], self.max_grid_size - 1)  # x
            actions[i, 2] = min(step['action'][2], self.max_grid_size - 1)  # y
            actions[i, 3] = min(step['action'][3], 11)  # variant

            # RTG
            rtg[i, 0] = step['rtg']

            # Timestep (position in original trajectory)
            timesteps[i] = min(step['timestep'], 499)

            # Mask
            mask[i] = 1.0

        return {
            'states': states,
            'actions': actions,
            'returns_to_go': rtg,
            'timesteps': timesteps,
            'attention_mask': mask,
        }


def create_train_val_split(
    data_path: str,
    val_ratio: float = 0.1,
    context_len: int = 50,
    seed: int = 42,
) -> Tuple[EffectiveDTDataset, EffectiveDTDataset]:
    """
    Create train/val datasets with trajectory-level splitting.

    This ensures no data leakage - different windows from the same
    trajectory never appear in both train and val sets.
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    n_trajs = len(data['trajectories'])

    # Shuffle trajectory indices
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_trajs).tolist()

    n_val = max(1, int(n_trajs * val_ratio))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_ds = EffectiveDTDataset(data_path, context_len=context_len,
                                   trajectory_indices=train_indices)
    val_ds = EffectiveDTDataset(data_path, context_len=context_len,
                                 trajectory_indices=val_indices)

    return train_ds, val_ds


if __name__ == '__main__':
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'model/decision_transformer/effective_dt_data.pkl'

    print(f'Loading from {data_path}...')
    train_ds, val_ds = create_train_val_split(data_path, context_len=50)

    print(f'Train samples: {len(train_ds)}')
    print(f'Val samples: {len(val_ds)}')

    sample = train_ds[0]
    print('\nSample shapes:')
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: {v.shape} ({v.dtype})')
