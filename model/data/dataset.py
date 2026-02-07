"""
PyTorch Dataset for Hamiltonian Path Optimization
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from operations import HamiltonianSTL


class HamiltonianDataset(Dataset):
    """Dataset for CNN+RNN training with actual path edges."""
    
    def __init__(self, data_file: str, max_seq_len: int = 50, max_grid_size: int = 30):
        self.data_file = Path(data_file)
        self.max_seq_len = max_seq_len
        self.max_grid_size = max_grid_size
        self.records = []
        
        self._load_data()
    
    def _load_data(self):
        """Load records from JSONL file."""
        skipped = 0
        with open(self.data_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                # Skip records that exceed max grid size
                if data.get('grid_W', 0) <= self.max_grid_size and data.get('grid_H', 0) <= self.max_grid_size:
                    self.records.append(data)
                else:
                    skipped += 1
        if skipped > 0:
            print(f"  [INFO] Skipped {skipped} records exceeding {self.max_grid_size}x{self.max_grid_size} grid size")
    
    def __len__(self):
        return len(self.records)
    
    def _get_initial_path_edges(self, grid_W: int, grid_H: int) -> tuple:
        """Get initial zigzag path edge matrices."""
        h = HamiltonianSTL(grid_W, grid_H, init_pattern='zigzag')
        H_edges = np.array(h.H, dtype=np.float32)
        V_edges = np.array(h.V, dtype=np.float32)
        return H_edges, V_edges
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example with actual path edges."""
        record = self.records[idx]
        
        grid_W = record['grid_W']
        grid_H = record['grid_H']
        
        # Create multi-channel grid state
        # Channel 0: Horizontal edges (actual path)
        # Channel 1: Vertical edges (actual path)
        # Channel 2: Zone boundaries
        # Channel 3: Grid mask
        grid_state = torch.zeros(4, self.max_grid_size, self.max_grid_size)
        
        # Get initial path edges
        H_edges, V_edges = self._get_initial_path_edges(grid_W, grid_H)
        
        # Channel 0: Horizontal edges
        grid_state[0, :grid_H, :grid_W-1] = torch.from_numpy(H_edges)
        
        # Channel 1: Vertical edges  
        grid_state[1, :grid_H-1, :grid_W] = torch.from_numpy(V_edges)
        
        # Channel 2: Zone grid (normalized)
        if 'zone_grid' in record:
            zone_grid = np.array(record['zone_grid']).reshape(grid_H, grid_W)
            max_zone = max(zone_grid.max(), 1)
            zone_normalized = zone_grid / max_zone
            grid_state[2, :grid_H, :grid_W] = torch.from_numpy(zone_normalized).float()
        
        # Channel 3: Grid mask (1 where grid exists, 0 elsewhere)
        grid_state[3, :grid_H, :grid_W] = 1.0
        
        # Global features
        initial_crossings = record.get('initial_crossings', 0)
        final_crossings = record.get('final_crossings', 0)
        global_features = torch.tensor([
            initial_crossings,
            grid_W,
            grid_H
        ], dtype=torch.float32)
        
        # Encode operation sequence
        sequence_ops = record.get('sequence_ops', [])[:self.max_seq_len]
        
        op_types, op_x, op_y, flip_variants, trans_variants = self._encode_operations(sequence_ops)
        
        target = {
            'operation_type': torch.tensor(op_types, dtype=torch.long),
            'position_x': torch.tensor(op_x, dtype=torch.long),
            'position_y': torch.tensor(op_y, dtype=torch.long),
            'flip_variant': torch.tensor(flip_variants, dtype=torch.long),
            'transpose_variant': torch.tensor(trans_variants, dtype=torch.long),
            'crossing_reduction': torch.tensor(initial_crossings - final_crossings, dtype=torch.float32)
        }
        
        return {
            'grid_state': grid_state,
            'global_features': global_features,
            'target': target,
            'seq_len': len(sequence_ops),
            'grid_W': grid_W,
            'grid_H': grid_H
        }
    
    def _encode_operations(self, sequence_ops: List[Dict]) -> tuple:
        """Encode operation sequence to tensors."""
        op_types = []
        op_x = []
        op_y = []
        flip_variants = []
        trans_variants = []
        
        for op in sequence_ops:
            kind = op.get('kind', 'N')
            x = op.get('x', 0)
            y = op.get('y', 0)
            variant = op.get('variant', '-')
            
            # Encode type: T=0, F=1, N=2
            if kind == 'T':
                op_types.append(0)
            elif kind == 'F':
                op_types.append(1)
            else:
                op_types.append(2)
            
            # Clamp positions to grid bounds
            op_x.append(min(x, self.max_grid_size - 1))
            op_y.append(min(y, self.max_grid_size - 1))
            
            # Encode variants separately for flip and transpose
            if kind == 'T':
                variant_map = {'nl': 0, 'nr': 1, 'sl': 2, 'sr': 3, 'eb': 4}
                trans_variants.append(variant_map.get(variant, 0))
                flip_variants.append(0)  # Dummy value for non-flip operations
            elif kind == 'F':
                variant_map = {'n': 0, 's': 1, 'e': 2, 'w': 3}
                flip_variants.append(variant_map.get(variant, 0))
                trans_variants.append(0)  # Dummy value for non-transpose operations
            else:
                flip_variants.append(0)
                trans_variants.append(0)
        
        # Pad sequences
        while len(op_types) < self.max_seq_len:
            op_types.append(2)  # NOP
            op_x.append(0)
            op_y.append(0)
            flip_variants.append(0)
            trans_variants.append(0)
        
        return op_types, op_x, op_y, flip_variants, trans_variants


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate for batching."""
    return {
        'grid_states': torch.stack([item['grid_state'] for item in batch]),
        'global_features': torch.stack([item['global_features'] for item in batch]),
        'targets': {
            'operation_type': torch.stack([item['target']['operation_type'] for item in batch]),
            'position_x': torch.stack([item['target']['position_x'] for item in batch]),
            'position_y': torch.stack([item['target']['position_y'] for item in batch]),
            'flip_variant': torch.stack([item['target']['flip_variant'] for item in batch]),
            'transpose_variant': torch.stack([item['target']['transpose_variant'] for item in batch]),
            'crossing_reduction': torch.stack([item['target']['crossing_reduction'] for item in batch])
        },
        'seq_lens': [item['seq_len'] for item in batch]
    }
