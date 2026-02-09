"""
FusionNet: CNN+RNN Fusion for per-pixel Hamiltonian path operation prediction.

CNN branch: ResU-Net maintaining full 32x32 spatial resolution (same as OperationNet).
RNN branch: GRU over recent effective operation history (action, position, crossings).
Fusion: FiLM conditioning — RNN context generates scale/shift for CNN decoder features.
Output: per-pixel position (WTA K hypotheses) + per-pixel action (12 classes).

Identity-initialized FiLM ensures the model starts as a pure U-Net and gradually
learns to incorporate temporal context — it cannot be worse than U-Net alone.

Input: 5-channel 32x32 tensor (zones, H_edges, V_edges, boundary_mask, crossing_indicator)
       + history of K most recent effective operations
Output: position scores (Kx32x32) + action logits (12x32x32)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


# Canonical mappings (must match unet_model.py / dt_dataset_v2.py / build_dt_training_data.py)
VARIANT_MAP = {
    'nl': 0, 'nr': 1, 'sl': 2, 'sr': 3, 'eb': 4, 'ea': 5, 'wa': 6, 'wb': 7,
    'n': 8, 's': 9, 'e': 10, 'w': 11
}
VARIANT_REV = {v: k for k, v in VARIANT_MAP.items()}
NUM_ACTIONS = 12  # 8 transpose + 4 flip


# ---------------------------------------------------------------------------
# ResU-Net building blocks (identical to model/unet/unet_model.py)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """3x3 conv -> BN -> LeakyReLU -> 3x3 conv -> BN + additive skip."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.leaky_relu(x + self.block(x), 0.01, inplace=True)


class DownBlock(nn.Module):
    """1x1 project -> ResBlock -> MaxPool."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.res = ResBlock(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.res(self.proj(x))
        return feat, self.pool(feat)


class UpBlock(nn.Module):
    """Upsample + concat skip -> 1x1 project -> ResBlock."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.res = ResBlock(out_ch)

    def forward(self, x, skip):
        return self.res(self.proj(torch.cat([self.up(x), skip], dim=1)))


# ---------------------------------------------------------------------------
# FiLM conditioning layer
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: gamma * features + beta."""

    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

    def forward(self, features, gamma, beta):
        """
        features: [B, C, H, W]
        gamma: [B, C]
        beta: [B, C]
        """
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)     # [B, C, 1, 1]
        return gamma * features + beta


# ---------------------------------------------------------------------------
# History encoder (RNN branch)
# ---------------------------------------------------------------------------

class HistoryEncoder(nn.Module):
    """
    GRU encoder for operation history.

    Embeds each history entry (action, pos_y, pos_x, crossings_before, crossings_after)
    and processes with a GRU to produce a context vector.
    """

    def __init__(self, max_grid_size=32, rnn_hidden=192, rnn_layers=2, dropout=0.15):
        super().__init__()
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers

        # Embedding tables
        emb_dim = 16
        self.action_embed = nn.Embedding(NUM_ACTIONS, emb_dim)
        self.pos_y_embed = nn.Embedding(max_grid_size, emb_dim)
        self.pos_x_embed = nn.Embedding(max_grid_size, emb_dim)
        self.cb_proj = nn.Sequential(nn.Linear(1, emb_dim), nn.ReLU())
        self.ca_proj = nn.Sequential(nn.Linear(1, emb_dim), nn.ReLU())

        input_dim = emb_dim * 5  # 80

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0,
        )

    def forward(self, hist_actions, hist_py, hist_px, hist_cb, hist_ca, hist_mask):
        """
        All inputs: [B, K] where K = max_history.
        hist_cb, hist_ca: float tensors (normalized crossings).
        hist_mask: [B, K] float, 1=valid 0=padding.
        Returns: context [B, rnn_hidden].
        """
        B, K = hist_actions.shape
        device = hist_actions.device

        # Embed each component
        act_emb = self.action_embed(hist_actions)          # [B, K, 16]
        py_emb = self.pos_y_embed(hist_py)                 # [B, K, 16]
        px_emb = self.pos_x_embed(hist_px)                 # [B, K, 16]
        cb_emb = self.cb_proj(hist_cb.unsqueeze(-1))       # [B, K, 16]
        ca_emb = self.ca_proj(hist_ca.unsqueeze(-1))       # [B, K, 16]

        history_emb = torch.cat([act_emb, py_emb, px_emb, cb_emb, ca_emb], dim=-1)  # [B, K, 80]

        # Compute valid lengths
        lengths = hist_mask.sum(dim=1).long().clamp(min=1)  # [B], min 1 for packing

        # Pack and run GRU
        packed = pack_padded_sequence(
            history_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)  # h_n: [layers, B, hidden]

        # Use last layer's hidden state
        context = h_n[-1]  # [B, rnn_hidden]

        # Zero out context for samples with no history
        no_history = (hist_mask.sum(dim=1) == 0).unsqueeze(-1)  # [B, 1]
        context = context.masked_fill(no_history, 0.0)

        return context


# ---------------------------------------------------------------------------
# FusionNet
# ---------------------------------------------------------------------------

class FusionNet(nn.Module):
    """
    CNN+RNN Fusion model with FiLM conditioning.

    Architecture:
        Encoder: 3-level ResU-Net [f, 2f, 4f] (identical to OperationNet)
        Bottleneck: ResBlock + Dropout
        Decoder: 3 levels with skip connections + FiLM from RNN context
        Heads: position (K hypotheses) + action (12ch)

    The FiLM projection is identity-initialized so model starts as pure U-Net.
    """

    def __init__(self, in_channels=5, base_features=48, n_hypotheses=4,
                 max_history=8, rnn_hidden=192, rnn_layers=2, max_grid_size=32,
                 rnn_dropout=0.15):
        super().__init__()
        self.n_hypotheses = n_hypotheses
        self.max_history = max_history
        f = base_features

        # --- CNN Encoder (same as OperationNet) ---
        self.enc1 = DownBlock(in_channels, f)
        self.enc2 = DownBlock(f, f * 2)
        self.enc3 = DownBlock(f * 2, f * 4)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(f * 4),
            nn.Dropout2d(0.15),
        )

        # --- CNN Decoder ---
        self.dec3 = UpBlock(f * 4, f * 4, f * 2)
        self.dec2 = UpBlock(f * 2, f * 2, f)
        self.dec1 = UpBlock(f, f, f)

        # --- RNN Branch ---
        self.history_encoder = HistoryEncoder(
            max_grid_size=max_grid_size,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
            dropout=rnn_dropout,
        )

        # --- FiLM Projection ---
        # dec3 has 2f channels, dec2 has f, dec1 has f
        # Total FiLM params: 2*(2f + f + f) = 8f (gamma + beta per level)
        total_film_params = 2 * (2 * f + f + f)  # 8f
        self.film_proj = nn.Sequential(
            nn.Linear(rnn_hidden, total_film_params),
            nn.Dropout(0.1),
        )

        # Identity-initialize: gamma=1, beta=0
        nn.init.zeros_(self.film_proj[0].weight)
        # Bias: first half is gammas (init to 1), second half is betas (init to 0)
        with torch.no_grad():
            bias = self.film_proj[0].bias
            bias.zero_()
            # Gamma biases for each level: 2f + f + f = 4f total gamma params
            gamma_total = 2 * f + f + f  # 4f
            bias[:gamma_total] = 1.0  # gammas init to 1

        # FiLM layers (one per decoder level)
        self.film3 = FiLMLayer(f * 2)
        self.film2 = FiLMLayer(f)
        self.film1 = FiLMLayer(f)

        # --- Output Heads ---
        self.position_head = nn.Conv2d(f, n_hypotheses, 1)
        self.action_head = nn.Conv2d(f, NUM_ACTIONS, 1)

    def forward(self, state, hist_actions, hist_py, hist_px, hist_cb, hist_ca, hist_mask):
        """
        state: [B, 5, 32, 32]
        hist_actions: [B, K] long
        hist_py: [B, K] long
        hist_px: [B, K] long
        hist_cb: [B, K] float (normalized crossings before)
        hist_ca: [B, K] float (normalized crossings after)
        hist_mask: [B, K] float (1=valid, 0=pad)

        Returns: position_logits [B, n_hyp, 32, 32], action_logits [B, 12, 32, 32]
        """
        f = self.enc1.proj[0].out_channels  # base_features

        # --- CNN Encoder ---
        e1, d1 = self.enc1(state)          # e1: [B,f,32,32], d1: [B,f,16,16]
        e2, d2 = self.enc2(d1)             # e2: [B,2f,16,16], d2: [B,2f,8,8]
        e3, d3 = self.enc3(d2)             # e3: [B,4f,8,8], d3: [B,4f,4,4]

        b = self.bottleneck(d3)            # [B, 4f, 4, 4]

        # --- RNN Branch ---
        context = self.history_encoder(
            hist_actions, hist_py, hist_px, hist_cb, hist_ca, hist_mask
        )  # [B, rnn_hidden]

        # --- Generate FiLM parameters ---
        film_params = self.film_proj(context)  # [B, 8f]

        # Split into per-level (gamma, beta) pairs
        # Layout: [gamma3(2f), gamma2(f), gamma1(f), beta3(2f), beta2(f), beta1(f)]
        idx = 0
        gamma3 = film_params[:, idx:idx + 2 * f]; idx += 2 * f
        gamma2 = film_params[:, idx:idx + f]; idx += f
        gamma1 = film_params[:, idx:idx + f]; idx += f
        beta3 = film_params[:, idx:idx + 2 * f]; idx += 2 * f
        beta2 = film_params[:, idx:idx + f]; idx += f
        beta1 = film_params[:, idx:idx + f]

        # --- CNN Decoder with FiLM ---
        u3 = self.dec3(b, e3)              # [B, 2f, 8, 8]
        u3 = self.film3(u3, gamma3, beta3)

        u2 = self.dec2(u3, e2)             # [B, f, 16, 16]
        u2 = self.film2(u2, gamma2, beta2)

        u1 = self.dec1(u2, e1)             # [B, f, 32, 32]
        u1 = self.film1(u1, gamma1, beta1)

        # --- Output Heads ---
        position_logits = self.position_head(u1)   # [B, K, 32, 32]
        action_logits = self.action_head(u1)        # [B, 12, 32, 32]

        return position_logits, action_logits


# ---------------------------------------------------------------------------
# Loss function (WTA + diversity, same as unet_model.py)
# ---------------------------------------------------------------------------

def compute_loss(position_logits, action_logits, target_y, target_x, target_action,
                 boundary_masks, label_smoothing=0.1, pos_weight=10.0,
                 diversity_weight=0.5):
    """
    Winner-Takes-All (WTA) loss with diversity regularizer.

    1. Each of K hypotheses produces a softmax over boundary positions.
    2. Only the closest hypothesis to GT receives gradients (WTA).
    3. Diversity: soft entropy over winner assignments — penalizes mode collapse.
    """
    B, K, H, W = position_logits.shape
    device = position_logits.device
    batch_idx = torch.arange(B, device=device)

    # Flatten spatial dims
    pos_flat = position_logits.reshape(B, K, -1)               # [B, K, H*W]
    mask_flat = boundary_masks.reshape(B, -1).bool()            # [B, H*W]

    target_flat = target_y * W + target_x                       # [B]
    mask_flat[batch_idx, target_flat] = True                     # ensure GT in mask

    # Expand mask for all K hypotheses
    mask_k = mask_flat.unsqueeze(1).expand_as(pos_flat)         # [B, K, H*W]
    pos_masked = pos_flat.masked_fill(~mask_k, float('-inf'))

    # Log-softmax per hypothesis over boundary positions
    log_probs = F.log_softmax(pos_masked, dim=-1)               # [B, K, H*W]

    # CE per hypothesis: -log_prob at GT position
    target_idx = target_flat.view(B, 1, 1).expand(B, K, 1)     # [B, K, 1]
    per_hyp_loss = -log_probs.gather(2, target_idx).squeeze(2)  # [B, K]

    # WTA: min across hypotheses
    pos_loss = per_hyp_loss.min(dim=1).values.mean()

    # Diversity: soft winner assignment entropy
    winner_probs = F.softmax(-per_hyp_loss, dim=1)              # [B, K]
    avg_usage = winner_probs.mean(dim=0)                         # [K]
    diversity_loss = (avg_usage * (avg_usage + 1e-8).log()).sum() + math.log(K)

    # Action loss at GT position
    act_at_gt = action_logits[batch_idx, :, target_y, target_x]
    act_loss = F.cross_entropy(
        act_at_gt, target_action, label_smoothing=label_smoothing
    )

    total_loss = pos_weight * pos_loss + act_loss + diversity_weight * diversity_loss
    return total_loss, pos_loss, act_loss, diversity_loss


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
