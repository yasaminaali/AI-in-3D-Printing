"""
FusionNet v2: CNN+RNN Fusion for per-pixel Hamiltonian path operation prediction.

Architecture changes from v1:
- 4-level encoder/decoder (was 3) for 128x128 input
- GroupNorm instead of BatchNorm (handles variable grid sizes)
- Bottleneck self-attention (global reasoning at 8x8)
- Refinement heads for position and action (was single 1x1 conv)
- Gaussian heatmap position targets with KL divergence
- Neighborhood action supervision
- 9 input channels (was 5): zones, H, V, grid_validity, boundary,
  crossing_count, progress, y_coord, x_coord

Input: 9-channel 128x128 tensor + history of K recent effective operations
Output: position scores (Kx128x128) + action logits (12x128x128)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


# Canonical mappings
VARIANT_MAP = {
    'nl': 0, 'nr': 1, 'sl': 2, 'sr': 3, 'eb': 4, 'ea': 5, 'wa': 6, 'wb': 7,
    'n': 8, 's': 9, 'e': 10, 'w': 11
}
VARIANT_REV = {v: k for k, v in VARIANT_MAP.items()}
NUM_ACTIONS = 12  # 8 transpose + 4 flip


# ---------------------------------------------------------------------------
# ResU-Net building blocks — GroupNorm replaces BatchNorm (Change 3)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """3x3 conv -> GroupNorm -> LeakyReLU -> 3x3 conv -> GroupNorm + additive skip."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, ch), num_channels=ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, ch), num_channels=ch),
        )

    def forward(self, x):
        return F.leaky_relu(x + self.block(x), 0.01, inplace=True)


class DownBlock(nn.Module):
    """1x1 project -> ResBlock -> MaxPool."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
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
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.res = ResBlock(out_ch)

    def forward(self, x, skip):
        return self.res(self.proj(torch.cat([self.up(x), skip], dim=1)))


# ---------------------------------------------------------------------------
# Bottleneck Self-Attention (Change 13)
# ---------------------------------------------------------------------------

class BottleneckAttention(nn.Module):
    """Multi-head self-attention at bottleneck resolution (8x8 = 64 tokens)."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        flat = x.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        attn_out, _ = self.attn(flat, flat, flat)
        flat = self.norm(flat + attn_out)  # residual
        return flat.permute(0, 2, 1).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# FiLM conditioning layer
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: gamma * features + beta."""

    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

    def forward(self, features, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta


# ---------------------------------------------------------------------------
# History encoder (RNN branch)
# ---------------------------------------------------------------------------

class HistoryEncoder(nn.Module):
    """GRU encoder for operation history."""

    def __init__(self, max_grid_size=128, rnn_hidden=192, rnn_layers=2, dropout=0.15):
        super().__init__()
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers

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
        B, K = hist_actions.shape
        device = hist_actions.device

        act_emb = self.action_embed(hist_actions)
        py_emb = self.pos_y_embed(hist_py)
        px_emb = self.pos_x_embed(hist_px)
        cb_emb = self.cb_proj(hist_cb.unsqueeze(-1))
        ca_emb = self.ca_proj(hist_ca.unsqueeze(-1))

        history_emb = torch.cat([act_emb, py_emb, px_emb, cb_emb, ca_emb], dim=-1)

        lengths = hist_mask.sum(dim=1).long().clamp(min=1)

        packed = pack_padded_sequence(
            history_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)

        context = h_n[-1]

        no_history = (hist_mask.sum(dim=1) == 0).unsqueeze(-1)
        context = context.masked_fill(no_history, 0.0)

        return context


# ---------------------------------------------------------------------------
# FusionNet v2
# ---------------------------------------------------------------------------

class FusionNet(nn.Module):
    """
    CNN+RNN Fusion model v2 with FiLM conditioning.

    Changes from v1:
        - 4-level encoder/decoder (Change 2)
        - GroupNorm everywhere (Change 3)
        - Refinement heads (Change 4)
        - Bottleneck self-attention (Change 13)
        - 9 input channels (Change 5)
    """

    def __init__(self, in_channels=9, base_features=48, n_hypotheses=1,
                 max_history=32, rnn_hidden=192, rnn_layers=2, max_grid_size=128,
                 rnn_dropout=0.15):
        super().__init__()
        self.n_hypotheses = n_hypotheses
        self.max_history = max_history
        f = base_features

        # --- CNN Encoder (4 levels, Change 2) ---
        self.enc1 = DownBlock(in_channels, f)          # 128->64
        self.enc2 = DownBlock(f, f * 2)                # 64->32
        self.enc3 = DownBlock(f * 2, f * 4)            # 32->16
        self.enc4 = DownBlock(f * 4, f * 4)            # 16->8

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(f * 4),
            nn.Dropout2d(0.15),
        )

        # Bottleneck self-attention (Change 13)
        self.bottleneck_attn = BottleneckAttention(f * 4, num_heads=4)

        # --- CNN Decoder (4 levels) ---
        self.dec4 = UpBlock(f * 4, f * 4, f * 4)      # 8->16
        self.dec3 = UpBlock(f * 4, f * 4, f * 2)      # 16->32
        self.dec2 = UpBlock(f * 2, f * 2, f)           # 32->64
        self.dec1 = UpBlock(f, f, f)                   # 64->128

        # --- RNN Branch ---
        self.history_encoder = HistoryEncoder(
            max_grid_size=max_grid_size,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
            dropout=rnn_dropout,
        )

        # --- FiLM Projection ---
        # dec4: 4f, dec3: 2f, dec2: f, dec1: f -> total gamma+beta = 2*(4f+2f+f+f) = 16f
        total_film_params = 2 * (4 * f + 2 * f + f + f)  # 16f
        self.film_proj = nn.Sequential(
            nn.Linear(rnn_hidden, total_film_params),
            nn.Dropout(0.1),
        )

        # Identity-initialize: gamma=1, beta=0
        nn.init.zeros_(self.film_proj[0].weight)
        with torch.no_grad():
            bias = self.film_proj[0].bias
            bias.zero_()
            gamma_total = 4 * f + 2 * f + f + f  # 8f
            bias[:gamma_total] = 1.0

        # FiLM layers (one per decoder level)
        self.film4 = FiLMLayer(f * 4)
        self.film3 = FiLMLayer(f * 2)
        self.film2 = FiLMLayer(f)
        self.film1 = FiLMLayer(f)

        # --- Output Heads (Change 4: refinement heads) ---
        self.position_head = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1, bias=False),
            nn.GroupNorm(8, f),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(f, f // 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, f // 2), f // 2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(f // 2, n_hypotheses, 1),
        )

        self.action_head = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1, bias=False),
            nn.GroupNorm(8, f),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(f, NUM_ACTIONS, 1),
        )

    def forward(self, state, hist_actions, hist_py, hist_px, hist_cb, hist_ca, hist_mask):
        """
        state: [B, 9, 128, 128]
        hist_*: [B, K] history tensors
        Returns: position_logits [B, n_hyp, 128, 128], action_logits [B, 12, 128, 128]
        """
        f = self.enc1.proj[0].out_channels

        # --- CNN Encoder (4 levels) ---
        e1, d1 = self.enc1(state)     # e1: [B,f,128,128], d1: [B,f,64,64]
        e2, d2 = self.enc2(d1)        # e2: [B,2f,64,64], d2: [B,2f,32,32]
        e3, d3 = self.enc3(d2)        # e3: [B,4f,32,32], d3: [B,4f,16,16]
        e4, d4 = self.enc4(d3)        # e4: [B,4f,16,16], d4: [B,4f,8,8]

        b = self.bottleneck(d4)       # [B, 4f, 8, 8]
        b = self.bottleneck_attn(b)   # Self-attention at 8x8

        # --- RNN Branch ---
        context = self.history_encoder(
            hist_actions, hist_py, hist_px, hist_cb, hist_ca, hist_mask
        )

        # --- Generate FiLM parameters ---
        film_params = self.film_proj(context)

        # Split: [gamma4(4f), gamma3(2f), gamma2(f), gamma1(f), beta4(4f), beta3(2f), beta2(f), beta1(f)]
        idx = 0
        gamma4 = film_params[:, idx:idx + 4 * f]; idx += 4 * f
        gamma3 = film_params[:, idx:idx + 2 * f]; idx += 2 * f
        gamma2 = film_params[:, idx:idx + f]; idx += f
        gamma1 = film_params[:, idx:idx + f]; idx += f
        beta4 = film_params[:, idx:idx + 4 * f]; idx += 4 * f
        beta3 = film_params[:, idx:idx + 2 * f]; idx += 2 * f
        beta2 = film_params[:, idx:idx + f]; idx += f
        beta1 = film_params[:, idx:idx + f]

        # --- CNN Decoder with FiLM ---
        u4 = self.dec4(b, e4)
        u4 = self.film4(u4, gamma4, beta4)

        u3 = self.dec3(u4, e3)
        u3 = self.film3(u3, gamma3, beta3)

        u2 = self.dec2(u3, e2)
        u2 = self.film2(u2, gamma2, beta2)

        u1 = self.dec1(u2, e1)
        u1 = self.film1(u1, gamma1, beta1)

        # --- Output Heads ---
        position_logits = self.position_head(u1)
        action_logits = self.action_head(u1)

        return position_logits, action_logits


# ---------------------------------------------------------------------------
# Gaussian heatmap helper (Change 6)
# ---------------------------------------------------------------------------

def gaussian_2d(H, W, cy, cx, sigma=2.0):
    """Generate 2D Gaussian heatmap centered at (cy, cx)."""
    yy = torch.arange(H, device=cy.device).float()
    xx = torch.arange(W, device=cx.device).float()
    gy = torch.exp(-((yy - cy.float()) ** 2) / (2 * sigma ** 2))
    gx = torch.exp(-((xx - cx.float()) ** 2) / (2 * sigma ** 2))
    return gy.unsqueeze(1) * gx.unsqueeze(0)  # [H, W]


# ---------------------------------------------------------------------------
# Loss function (Changes 6, 14)
# ---------------------------------------------------------------------------

def compute_loss(position_logits, action_logits, target_y, target_x, target_action,
                 boundary_masks, label_smoothing=0.1, pos_weight=5.0,
                 diversity_weight=0.5, gaussian_sigma=2.0, neighbor_weight=0.3):
    """
    Winner-Takes-All (WTA) loss with:
    - Gaussian heatmap position targets + KL divergence (Change 6)
    - Neighborhood action supervision (Change 14)
    - Diversity regularizer
    """
    B, K, H, W = position_logits.shape
    device = position_logits.device
    batch_idx = torch.arange(B, device=device)

    # Flatten spatial dims — compute in float32 for numerical stability
    pos_flat = position_logits.float().reshape(B, K, -1)
    mask_flat = boundary_masks.reshape(B, -1).bool()

    target_flat = target_y * W + target_x
    mask_flat[batch_idx, target_flat] = True

    # Ensure every sample has at least one valid position
    has_valid = mask_flat.any(dim=-1)
    if not has_valid.all():
        # Fallback: unmask the target position for samples with empty masks
        for i in range(B):
            if not has_valid[i]:
                mask_flat[i, target_flat[i]] = True

    # Expand mask for all K hypotheses
    mask_k = mask_flat.unsqueeze(1).expand_as(pos_flat)
    # Use large negative instead of -inf to avoid NaN in softmax
    pos_masked = pos_flat.masked_fill(~mask_k, -1e9)

    # --- Gaussian heatmap targets (Change 6) ---
    # Build per-sample Gaussian targets, masked to boundary, renormalized
    gaussian_targets = torch.zeros(B, H * W, device=device, dtype=torch.float32)
    for i in range(B):
        g = gaussian_2d(H, W, target_y[i], target_x[i], sigma=gaussian_sigma)
        g_flat = g.reshape(-1)
        # Mask to boundary positions
        g_flat = g_flat * mask_flat[i].float()
        # Renormalize to sum to 1
        g_sum = g_flat.sum()
        if g_sum > 1e-8:
            g_flat = g_flat / g_sum
        else:
            # Fallback to one-hot at target
            g_flat = torch.zeros_like(g_flat)
            g_flat[target_flat[i]] = 1.0
        gaussian_targets[i] = g_flat

    # Log-softmax per hypothesis over boundary positions (float32)
    log_probs = F.log_softmax(pos_masked, dim=-1)  # [B, K, H*W]

    # Cross-entropy: -sum(target * log_pred)
    gt_expanded = gaussian_targets.unsqueeze(1).expand_as(log_probs)  # [B, K, H*W]
    per_hyp_loss = -(gt_expanded * log_probs).sum(dim=-1)  # [B, K]

    # Clamp to avoid NaN from any remaining numerical issues
    per_hyp_loss = per_hyp_loss.clamp(max=100.0)

    # WTA: min across hypotheses
    pos_loss = per_hyp_loss.min(dim=1).values.mean()

    # Diversity: soft winner assignment entropy (float32)
    winner_probs = F.softmax(-per_hyp_loss.float(), dim=1)
    avg_usage = winner_probs.mean(dim=0)
    diversity_loss = (avg_usage * torch.log(avg_usage + 1e-8)).sum() + math.log(K)

    # --- Action loss at GT position ---
    act_at_gt = action_logits[batch_idx, :, target_y, target_x]
    act_loss = F.cross_entropy(
        act_at_gt, target_action, label_smoothing=label_smoothing
    )

    # --- Neighborhood action supervision (Change 14) ---
    neighbor_loss = torch.tensor(0.0, device=device)
    n_neighbors = 0
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny = (target_y + dy).clamp(0, H - 1)
        nx = (target_x + dx).clamp(0, W - 1)
        # Only add loss where neighbor is different from target (avoid double-counting center)
        valid = ((ny != target_y) | (nx != target_x))
        if valid.any():
            act_neighbor = action_logits[batch_idx, :, ny, nx]
            nl = F.cross_entropy(
                act_neighbor, target_action, label_smoothing=0.2, reduction='none'
            )
            neighbor_loss = neighbor_loss + (nl * valid.float()).sum()
            n_neighbors += valid.sum()

    if n_neighbors > 0:
        neighbor_loss = neighbor_loss / n_neighbors.float()
    act_loss = act_loss + neighbor_weight * neighbor_loss

    total_loss = pos_weight * pos_loss + act_loss + diversity_weight * diversity_loss
    return total_loss, pos_loss, act_loss, diversity_loss


def compute_ranking_loss(position_logits, action_logits, target_y, target_x, target_action,
                         boundary_masks, margin=1.0, n_hard_neg=32,
                         label_smoothing=0.1, neighbor_weight=0.3):
    """
    Margin-based ranking loss with hard negative mining.

    Instead of predicting SA's exact position (classification), train the model
    to RANK SA's position higher than other boundary positions (ranking).

    Position loss: hinge loss max(0, margin + s_neg - s_pos), with top-k hard negatives.
    Action loss: cross-entropy at GT position + neighbor supervision (unchanged).
    """
    B, K, H, W = position_logits.shape  # K=1
    device = position_logits.device
    batch_idx = torch.arange(B, device=device)

    # Position scores: [B, H*W]
    scores = position_logits[:, 0].float().reshape(B, -1)
    mask = boundary_masks.reshape(B, -1).bool()
    target_flat = target_y * W + target_x

    # Ensure target is always in the valid mask
    mask[batch_idx, target_flat] = True

    # Positive scores
    s_pos = scores[batch_idx, target_flat]  # [B]

    # Negative mask (boundary minus positive)
    neg_mask = mask.clone()
    neg_mask[batch_idx, target_flat] = False

    # Hinge loss: max(0, margin + s_neg - s_pos) for all negatives
    hinge = F.relu(margin + scores - s_pos.unsqueeze(1))  # [B, H*W]
    hinge = hinge * neg_mask.float()

    # Hard negative mining: top-k hardest per sample
    n_valid_min = int(neg_mask.float().sum(1).min().item())
    top_k = min(n_hard_neg, max(n_valid_min, 1))
    if top_k > 0:
        top_hinge, _ = hinge.topk(top_k, dim=1)
        pos_loss = top_hinge.mean()
    else:
        pos_loss = torch.tensor(0.0, device=device)

    # --- Action loss at GT position (unchanged) ---
    act_at_gt = action_logits[batch_idx, :, target_y, target_x]
    act_loss = F.cross_entropy(act_at_gt, target_action, label_smoothing=label_smoothing)

    # Neighbor action supervision (unchanged)
    neighbor_loss = torch.tensor(0.0, device=device)
    n_neighbors = 0
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny = (target_y + dy).clamp(0, H - 1)
        nx = (target_x + dx).clamp(0, W - 1)
        valid = ((ny != target_y) | (nx != target_x))
        if valid.any():
            act_neighbor = action_logits[batch_idx, :, ny, nx]
            nl = F.cross_entropy(
                act_neighbor, target_action, label_smoothing=0.2, reduction='none'
            )
            neighbor_loss = neighbor_loss + (nl * valid.float()).sum()
            n_neighbors += valid.sum()

    if n_neighbors > 0:
        neighbor_loss = neighbor_loss / n_neighbors.float()
    act_loss = act_loss + neighbor_weight * neighbor_loss

    total_loss = pos_loss + act_loss
    return total_loss, pos_loss, act_loss


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
