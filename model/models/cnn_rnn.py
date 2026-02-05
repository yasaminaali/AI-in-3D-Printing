"""
CNN+RNN Model Architecture
Paper-compliant implementation (CCAI Section 3.3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class CNNBackbone(nn.Module):
    """
    CNN extracts 128-dimensional spatial embeddings.
    Paper: 4 layers, 3x3 conv, batch norm, ReLU, padding maintained
    """
    
    def __init__(self, input_channels: int = 4, embedding_dim: int = 128):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Global features projection
        self.global_proj = nn.Linear(3, 32)
        
        # Final embedding
        self.embedding_proj = nn.Linear(128 + 32, embedding_dim)
    
    def forward(self, grid_input: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_input: (B, C, H, W)
            global_features: (B, 3) - crossings, grid_W, grid_H
        
        Returns:
            embedding: (B, 128)
        """
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(grid_input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # (B, 128)
        
        # Process global features
        global_embed = F.relu(self.global_proj(global_features))
        
        # Combine
        combined = torch.cat([x, global_embed], dim=1)
        embedding = F.relu(self.embedding_proj(combined))
        
        return embedding


class GRUSolver(nn.Module):
    """
    GRU for temporal sequence modeling.
    Paper: Maintains hidden state to capture operation dependencies
    """
    
    def __init__(self, input_dim: int = 128, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.hidden_init = nn.Linear(input_dim, hidden_size * num_layers)
    
    def forward(self, embeddings: torch.Tensor, 
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (B, T, 128)
            hidden: (num_layers, B, hidden_size)
        
        Returns:
            outputs: (B, T, hidden_size)
            hidden: (num_layers, B, hidden_size)
        """
        if hidden is None:
            init_hidden = self.hidden_init(embeddings[:, 0, :])
            init_hidden = init_hidden.view(self.num_layers, -1, self.hidden_size)
            hidden = init_hidden
        
        outputs, hidden = self.gru(embeddings, hidden)
        return outputs, hidden


class OperationPredictor(nn.Module):
    """Multi-head predictor for operation sequences."""
    
    def __init__(self, hidden_size: int = 256, max_grid_size: int = 100):
        super().__init__()
        
        self.max_grid_size = max_grid_size
        
        # Operation type: T, F, N
        self.type_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)
        )
        
        # Positions
        self.x_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, max_grid_size)
        )
        
        self.y_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, max_grid_size)
        )
        
        # Variants
        self.flip_variant_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # n, s, e, w
        )
        
        self.transpose_variant_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # nl, nr, sl, sr, eb
        )
        
        # Crossing reduction auxiliary
        self.crossing_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, rnn_outputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            rnn_outputs: (B, T, hidden_size)
        
        Returns:
            predictions: Dict with operation logits
        """
        B, T, _ = rnn_outputs.shape
        flat = rnn_outputs.reshape(B * T, -1)
        
        # Predictions
        type_logits = self.type_head(flat).view(B, T, 3)
        x_logits = self.x_head(flat).view(B, T, self.max_grid_size)
        y_logits = self.y_head(flat).view(B, T, self.max_grid_size)
        flip_var_logits = self.flip_variant_head(flat).view(B, T, 4)
        trans_var_logits = self.transpose_variant_head(flat).view(B, T, 5)
        crossing_pred = self.crossing_head(flat).view(B, T)
        
        return {
            'operation_type': type_logits,
            'position_x': x_logits,
            'position_y': y_logits,
            'flip_variant': flip_var_logits,
            'transpose_variant': trans_var_logits,
            'crossing_reduction': crossing_pred
        }


class CNNRNNHamiltonian(nn.Module):
    """
    Complete CNN+RNN hybrid model.
    Paper architecture: CNN (4-layer) + GRU + Multi-head predictor
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        model_config = config['model']
        
        self.cnn = CNNBackbone(
            input_channels=model_config['cnn']['input_channels'],
            embedding_dim=model_config['cnn']['embedding_dim']
        )
        
        self.rnn = GRUSolver(
            input_dim=model_config['cnn']['embedding_dim'],
            hidden_size=model_config['rnn']['hidden_size'],
            num_layers=model_config['rnn']['num_layers'],
            dropout=model_config['rnn']['dropout']
        )
        
        self.predictor = OperationPredictor(
            hidden_size=model_config['rnn']['hidden_size'],
            max_grid_size=model_config['predictor']['max_positions']
        )
        
        self.max_sequence_length = model_config['predictor']['sequence_length']
    
    def forward(self, grid_states: torch.Tensor, global_features: torch.Tensor,
                target_sequence: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            grid_states: (B, T, C, H, W)
            global_features: (B, T, 3)
        
        Returns:
            predictions: Dict with all logits
        """
        B, T, C, H, W = grid_states.shape
        
        # Extract CNN embeddings
        embeddings = []
        for t in range(T):
            emb = self.cnn(grid_states[:, t], global_features[:, t])
            embeddings.append(emb)
        
        embeddings = torch.stack(embeddings, dim=1)  # (B, T, 128)
        
        # RNN processing
        rnn_outputs, _ = self.rnn(embeddings)
        
        # Predict operations
        predictions = self.predictor(rnn_outputs)
        
        return predictions
    
    def predict_sequence(self, initial_grid: torch.Tensor, global_features: torch.Tensor,
                         max_length: int = 100) -> list:
        """Autoregressive inference."""
        self.eval()
        sequence = []
        current_grid = initial_grid
        
        with torch.no_grad():
            hidden = None
            
            for step in range(max_length):
                emb = self.cnn(current_grid, global_features)
                emb = emb.unsqueeze(1)
                
                rnn_out, hidden = self.rnn(emb, hidden)
                preds = self.predictor(rnn_out)
                
                # Decode predictions
                op_type = torch.argmax(preds['operation_type'][:, 0], dim=-1).item()
                pos_x = torch.argmax(preds['position_x'][:, 0], dim=-1).item()
                pos_y = torch.argmax(preds['position_y'][:, 0], dim=-1).item()
                
                if op_type == 0:
                    variant_map = ['nl', 'nr', 'sl', 'sr', 'eb']
                    variant = torch.argmax(preds['transpose_variant'][:, 0], dim=-1).item()
                    kind, var = 'T', variant_map[variant]
                elif op_type == 1:
                    variant_map = ['n', 's', 'e', 'w']
                    variant = torch.argmax(preds['flip_variant'][:, 0], dim=-1).item()
                    kind, var = 'F', variant_map[variant]
                else:
                    kind, var = 'N', '-'
                
                sequence.append({'kind': kind, 'x': pos_x, 'y': pos_y, 'variant': var})
                
                if kind == 'N':
                    break
        
        return sequence


def create_model(config_path: str = "model/config.yaml") -> CNNRNNHamiltonian:
    """Factory function."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return CNNRNNHamiltonian(config)
