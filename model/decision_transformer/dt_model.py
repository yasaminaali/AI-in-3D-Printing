"""
Decision Transformer for Hamiltonian Path Optimization (v2).

Architecture:
- SpatialStateEncoder: CNN that preserves spatial info (AdaptiveAvgPool2d(4), not 1)
- GPT-style transformer for sequence modeling
- Return-to-go embedding conditions action generation
- Separate prediction heads for op_type, x, y, variant

Key changes from v1:
- SpatialStateEncoder preserves 4x4 spatial grid (16 spatial positions)
- 12 variants (was 9) - includes ea, wa, wb transpose variants
- Variable grid size support via max_grid_size + zero-padding
- Larger capacity: embed_dim=256, n_heads=8
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialStateEncoder(nn.Module):
    """
    CNN encoder that preserves spatial information.

    Unlike the original StateEncoder which used AdaptiveAvgPool2d(1) and
    destroyed all spatial info, this uses AdaptiveAvgPool2d(4) to keep
    a 4x4 spatial feature map. Each of the 16 positions maps to a region
    of the original grid, allowing the model to predict WHERE to apply ops.
    """

    def __init__(self, embed_dim: int = 256, max_grid_size: int = 32):
        super().__init__()

        # 4 channels: zones, H_edges, V_edges, boundary_mask
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16->8
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # any size -> 4x4 (NOT 1x1!)
        )

        # 64 channels * 4*4 spatial = 1024 features with spatial layout
        self.spatial_proj = nn.Linear(64 * 4 * 4, embed_dim)
        self.output_dim = embed_dim

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 4, H, W) or (batch, 4, H, W)
        Returns:
            embeddings: (batch, seq_len, embed_dim) or (batch, embed_dim)
        """
        squeeze_seq = False
        if x.dim() == 5:
            batch, seq_len, c, h, w = x.shape
            x = x.view(batch * seq_len, c, h, w)
            squeeze_seq = True

        x = self.conv(x)                  # (B*S, 64, 4, 4)
        x = x.view(x.size(0), -1)         # (B*S, 1024)
        x = self.spatial_proj(x)           # (B*S, embed_dim)

        if squeeze_seq:
            x = x.view(batch, seq_len, -1)

        return x


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for grid operation prediction (v2).

    Input sequence per timestep: [RTG, state, action]
    Model predicts action given (RTG, state) context.

    Changes from v1:
    - SpatialStateEncoder (preserves spatial info)
    - 12 variants (was 9)
    - Variable grid size via max_grid_size
    - Larger embed_dim=256, n_heads=8
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        context_len: int = 50,
        max_timestep: int = 500,
        dropout: float = 0.1,
        # Action space
        n_op_types: int = 3,   # N, T, F
        max_grid_size: int = 32,
        n_variants: int = 12,  # 8 transpose + 4 flip
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.context_len = context_len
        self.max_grid_size = max_grid_size
        self.n_op_types = n_op_types
        self.n_variants = n_variants

        # State encoder (spatial CNN)
        self.state_encoder = SpatialStateEncoder(embed_dim, max_grid_size)

        # RTG embedding (continuous value -> embedding)
        self.rtg_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Tanh()
        )

        # Action embedding - 4 components concatenated
        # embed_dim must be divisible by 4
        assert embed_dim % 4 == 0, f"embed_dim must be divisible by 4, got {embed_dim}"
        action_part_dim = embed_dim // 4
        self.action_embed_op = nn.Embedding(n_op_types, action_part_dim)
        self.action_embed_x = nn.Embedding(max_grid_size, action_part_dim)
        self.action_embed_y = nn.Embedding(max_grid_size, action_part_dim)
        self.action_embed_var = nn.Embedding(n_variants, action_part_dim)

        # Timestep embedding
        self.timestep_embed = nn.Embedding(max_timestep, embed_dim)

        # Token type embedding (to distinguish RTG, state, action in sequence)
        self.token_type_embed = nn.Embedding(3, embed_dim)  # 0=RTG, 1=state, 2=action

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=n_layers
        )

        # Layer norm
        self.ln = nn.LayerNorm(embed_dim)

        # Action prediction heads
        self.head_op_type = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, n_op_types)
        )
        self.head_x = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, max_grid_size)
        )
        self.head_y = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, max_grid_size)
        )
        self.head_variant = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, n_variants)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed_action(self, action):
        """
        Args:
            action: (batch, seq_len, 4) - [op_type, x, y, variant]
        Returns:
            embedding: (batch, seq_len, embed_dim)
        """
        # Clamp to valid ranges
        op = action[:, :, 0].clamp(0, self.n_op_types - 1)
        x = action[:, :, 1].clamp(0, self.max_grid_size - 1)
        y = action[:, :, 2].clamp(0, self.max_grid_size - 1)
        var = action[:, :, 3].clamp(0, self.n_variants - 1)

        op_emb = self.action_embed_op(op)
        x_emb = self.action_embed_x(x)
        y_emb = self.action_embed_y(y)
        var_emb = self.action_embed_var(var)

        return torch.cat([op_emb, x_emb, y_emb, var_emb], dim=-1)

    def forward(
        self,
        states,
        actions,
        returns_to_go,
        timesteps,
        attention_mask=None
    ):
        """
        Forward pass for training.

        Args:
            states: (batch, seq_len, 4, H, W) - grid states
            actions: (batch, seq_len, 4) - [op_type, x, y, variant]
            returns_to_go: (batch, seq_len, 1) - remaining reward
            timesteps: (batch, seq_len) - position in trajectory
            attention_mask: (batch, seq_len) - valid positions

        Returns:
            logits_dict with op_type, x, y, variant predictions
        """
        batch_size, seq_len = states.shape[:2]
        device = states.device

        # Clamp timesteps to valid range
        timesteps = timesteps.clamp(0, self.timestep_embed.num_embeddings - 1)

        # Encode components
        state_emb = self.state_encoder(states)     # (batch, seq_len, embed_dim)
        action_emb = self._embed_action(actions)   # (batch, seq_len, embed_dim)
        rtg_emb = self.rtg_encoder(returns_to_go)  # (batch, seq_len, embed_dim)

        # Add timestep embedding
        time_emb = self.timestep_embed(timesteps)
        state_emb = state_emb + time_emb
        action_emb = action_emb + time_emb
        rtg_emb = rtg_emb + time_emb

        # Add token type embedding
        rtg_emb = rtg_emb + self.token_type_embed(torch.zeros(1, device=device, dtype=torch.long))
        state_emb = state_emb + self.token_type_embed(torch.ones(1, device=device, dtype=torch.long))
        action_emb = action_emb + self.token_type_embed(torch.full((1,), 2, device=device, dtype=torch.long))

        # Interleave: [RTG_1, s_1, a_1, RTG_2, s_2, a_2, ...]
        stacked = torch.stack([rtg_emb, state_emb, action_emb], dim=2)
        stacked = stacked.view(batch_size, seq_len * 3, self.embed_dim)

        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len * 3, device)

        # Create padding mask if attention_mask provided
        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(-1).repeat(1, 1, 3).view(batch_size, -1)
            padding_mask = padding_mask == 0  # True = masked
        else:
            padding_mask = None

        # Transformer
        hidden = self.transformer(
            stacked,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        hidden = self.ln(hidden)

        # Extract state token outputs (indices 1, 4, 7, ... = 3*i + 1)
        state_indices = torch.arange(seq_len, device=device) * 3 + 1
        state_hidden = hidden[:, state_indices, :]

        # Predict actions
        logits_op = self.head_op_type(state_hidden)
        logits_x = self.head_x(state_hidden)
        logits_y = self.head_y(state_hidden)
        logits_var = self.head_variant(state_hidden)

        return {
            'op_type': logits_op,
            'x': logits_x,
            'y': logits_y,
            'variant': logits_var
        }

    def _create_causal_mask(self, size, device):
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def get_action(self, states, actions, returns_to_go, timesteps):
        """
        Get action prediction for the last timestep (for inference).

        Returns:
            action: (batch, 4) predicted action [op_type, x, y, variant]
            logits: dict of logits for the last timestep
        """
        logits = self.forward(states, actions, returns_to_go, timesteps)

        last_logits = {k: v[:, -1, :] for k, v in logits.items()}

        op_type = last_logits['op_type'].argmax(dim=-1)
        x = last_logits['x'].argmax(dim=-1)
        y = last_logits['y'].argmax(dim=-1)
        variant = last_logits['variant'].argmax(dim=-1)

        return torch.stack([op_type, x, y, variant], dim=-1), last_logits


def compute_loss(logits, targets, attention_mask=None, label_smoothing=0.1):
    """
    Compute variant-aware cross-entropy loss for action prediction.

    Position losses are weighted 1.5x since position prediction is harder
    and more important for valid operations.
    """
    losses = {}

    # Op type loss
    losses['op_type'] = F.cross_entropy(
        logits['op_type'].reshape(-1, logits['op_type'].size(-1)),
        targets[:, :, 0].reshape(-1),
        reduction='none',
        label_smoothing=label_smoothing
    )

    # Position losses (weighted higher)
    losses['x'] = F.cross_entropy(
        logits['x'].reshape(-1, logits['x'].size(-1)),
        targets[:, :, 1].reshape(-1),
        reduction='none',
        label_smoothing=label_smoothing
    )

    losses['y'] = F.cross_entropy(
        logits['y'].reshape(-1, logits['y'].size(-1)),
        targets[:, :, 2].reshape(-1),
        reduction='none',
        label_smoothing=label_smoothing
    )

    # Variant loss
    losses['variant'] = F.cross_entropy(
        logits['variant'].reshape(-1, logits['variant'].size(-1)),
        targets[:, :, 3].reshape(-1),
        reduction='none',
        label_smoothing=label_smoothing
    )

    # Apply mask if provided
    if attention_mask is not None:
        mask = attention_mask.reshape(-1)
        for k in losses:
            losses[k] = (losses[k] * mask).sum() / mask.sum().clamp(min=1)
    else:
        for k in losses:
            losses[k] = losses[k].mean()

    # Weighted total: position heads get 1.5x weight
    total = (losses['op_type'] +
             1.5 * losses['x'] +
             1.5 * losses['y'] +
             losses['variant'])

    return total, losses


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DecisionTransformer(
        embed_dim=256,
        n_heads=8,
        n_layers=6,
        context_len=50,
        max_grid_size=32,
        n_variants=12,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 50

    states = torch.randn(batch_size, seq_len, 4, 32, 32).to(device)
    actions = torch.randint(0, 3, (batch_size, seq_len, 4)).to(device)
    actions[:, :, 1] = torch.randint(0, 32, (batch_size, seq_len))
    actions[:, :, 2] = torch.randint(0, 32, (batch_size, seq_len))
    actions[:, :, 3] = torch.randint(0, 12, (batch_size, seq_len))
    rtg = torch.randn(batch_size, seq_len, 1).to(device)
    timesteps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)

    print("\nForward pass...")
    logits = model(states, actions, rtg, timesteps)

    print("Output shapes:")
    for k, v in logits.items():
        print(f"  {k}: {v.shape}")

    loss, loss_dict = compute_loss(logits, actions)
    print(f"\nTotal loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")

    action, last_logits = model.get_action(states, actions, rtg, timesteps)
    print(f"\nPredicted action shape: {action.shape}")
    print(f"Predicted action sample: {action[0].tolist()}")
