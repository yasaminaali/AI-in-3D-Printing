"""
Loss Functions for CNN+RNN Training
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class HamiltonianLoss(nn.Module):
    """Multi-task loss for Hamiltonian path optimization."""
    
    def __init__(self, weights: Dict[str, float], max_positions: int = 100):
        super().__init__()
        self.weights = weights
        self.max_positions = max_positions
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                seq_lens: list,
                device: torch.device) -> Tuple[torch.Tensor, Dict]:
        """
        Compute weighted multi-task loss.
        
        Args:
            predictions: Model outputs (B, T, ...)
            targets: Ground truth (B, T, ...)
            seq_lens: Actual sequence lengths
            device: Computation device
        
        Returns:
            total_loss: Weighted sum
            components: Individual loss values
        """
        B = len(seq_lens)
        T = predictions['operation_type'].size(1)  # Model output sequence length
        max_len = max(seq_lens)
        
        # Truncate to actual max length needed
        truncate_len = min(T, max_len)
        
        # Create mask for valid positions (truncate to actual length)
        mask = torch.zeros(B, truncate_len, device=device)
        for i, length in enumerate(seq_lens):
            mask[i, :min(length, truncate_len)] = 1.0
        
        # Truncate predictions and targets
        op_type_pred = predictions['operation_type'][:, :truncate_len, :]
        op_type_target = targets['operation_type'][:, :truncate_len]
        
        pos_x_pred = predictions['position_x'][:, :truncate_len, :]
        pos_x_target = targets['position_x'][:, :truncate_len]
        
        pos_y_pred = predictions['position_y'][:, :truncate_len, :]
        pos_y_target = targets['position_y'][:, :truncate_len]
        
        flip_var_pred = predictions['flip_variant'][:, :truncate_len, :]
        trans_var_pred = predictions['transpose_variant'][:, :truncate_len, :]
        flip_var_target = targets['flip_variant'][:, :truncate_len]
        trans_var_target = targets['transpose_variant'][:, :truncate_len]
        
        # Operation type loss
        type_loss = self.ce_loss(
            op_type_pred.reshape(-1, 3),
            op_type_target.reshape(-1)
        ).view(B, truncate_len)
        type_loss = (type_loss * mask).sum() / (mask.sum() + 1e-8)
        
        # Position X loss
        x_loss = self.ce_loss(
            pos_x_pred.reshape(-1, self.max_positions),
            pos_x_target.reshape(-1)
        ).view(B, truncate_len)
        x_loss = (x_loss * mask).sum() / (mask.sum() + 1e-8)
        
        # Position Y loss
        y_loss = self.ce_loss(
            pos_y_pred.reshape(-1, self.max_positions),
            pos_y_target.reshape(-1)
        ).view(B, truncate_len)
        y_loss = (y_loss * mask).sum() / (mask.sum() + 1e-8)
        
        # Variant losses - only compute for appropriate operation types
        # Create masks for flip and transpose operations
        flip_mask = (op_type_target == 1).float()  # F operations
        trans_mask = (op_type_target == 0).float()  # T operations
        
        # Flip variant loss (only for F operations)
        flip_var_loss_raw = self.ce_loss(
            flip_var_pred.reshape(-1, 4),
            flip_var_target.reshape(-1)
        ).view(B, truncate_len)
        flip_var_loss = (flip_var_loss_raw * flip_mask).sum() / (flip_mask.sum() + 1e-8)
        
        # Transpose variant loss (only for T operations)
        trans_var_loss_raw = self.ce_loss(
            trans_var_pred.reshape(-1, 5),
            trans_var_target.reshape(-1)
        ).view(B, truncate_len)
        trans_var_loss = (trans_var_loss_raw * trans_mask).sum() / (trans_mask.sum() + 1e-8)
        
        # Combine variant losses (average, handling case where one type is missing)
        if flip_mask.sum() > 0 and trans_mask.sum() > 0:
            variant_loss = (flip_var_loss + trans_var_loss) / 2
        elif flip_mask.sum() > 0:
            variant_loss = flip_var_loss
        elif trans_mask.sum() > 0:
            variant_loss = trans_var_loss
        else:
            variant_loss = torch.tensor(0.0, device=device)
        
        # Weighted total
        total_loss = (
            self.weights['operation_type'] * type_loss +
            self.weights['position_x'] * x_loss +
            self.weights['position_y'] * y_loss +
            self.weights['variant'] * variant_loss
        )
        
        components = {
            'type': type_loss.item(),
            'x': x_loss.item(),
            'y': y_loss.item(),
            'variant': variant_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, components
