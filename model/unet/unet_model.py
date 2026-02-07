"""
OperationNet: U-Net for per-pixel Hamiltonian path operation prediction.

Maintains full spatial resolution — no compression bottleneck.
Predicts WHERE to apply operations (position head) and WHAT to apply (action head).

Uses Winner-Takes-All (WTA) multi-hypothesis position prediction:
K independent position heads each learn to specialize on different modes.
Only the hypothesis closest to GT receives gradients during training.

Input: 5-channel 32x32 tensor (zones, H_edges, V_edges, boundary_mask, crossing_indicator)
Output: position scores (Kx32x32) + action logits (12x32x32)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Canonical mappings (must match dt_dataset_v2.py / build_dt_training_data.py)
VARIANT_MAP = {
    'nl': 0, 'nr': 1, 'sl': 2, 'sr': 3, 'eb': 4, 'ea': 5, 'wa': 6, 'wb': 7,
    'n': 8, 's': 9, 'e': 10, 'w': 11
}
VARIANT_REV = {v: k for k, v in VARIANT_MAP.items()}
NUM_ACTIONS = 12  # 8 transpose + 4 flip


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


class OperationNet(nn.Module):
    """
    Lightweight ResU-Net with WTA multi-hypothesis position prediction.

    Architecture:
        Encoder: 3 levels [f, 2f, 4f] with additive residual blocks
        Bottleneck: 4f -> 4f (ResBlock + Dropout)
        Decoder: 3 levels with skip connections
        Heads: position (K hypotheses) + action (12ch)

    Input: [B, 5, 32, 32]
    Output: position_logits [B, K, 32, 32], action_logits [B, 12, 32, 32]
    """

    def __init__(self, in_channels=5, base_features=24, n_hypotheses=4):
        super().__init__()
        self.n_hypotheses = n_hypotheses

        f = base_features

        # Encoder
        self.enc1 = DownBlock(in_channels, f)      # 5 -> f
        self.enc2 = DownBlock(f, f * 2)             # f -> 2f
        self.enc3 = DownBlock(f * 2, f * 4)         # 2f -> 4f

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(f * 4),
            nn.Dropout2d(0.1),
        )

        # Decoder
        self.dec3 = UpBlock(f * 4, f * 4, f * 2)   # 4f+4f -> 2f
        self.dec2 = UpBlock(f * 2, f * 2, f)        # 2f+2f -> f
        self.dec1 = UpBlock(f, f, f)                 # f+f -> f

        # Output heads
        self.position_head = nn.Conv2d(f, n_hypotheses, 1)   # f -> K hypotheses
        self.action_head = nn.Conv2d(f, NUM_ACTIONS, 1)       # f -> 12

    def forward(self, x):
        # Encoder
        e1, d1 = self.enc1(x)          # e1: [B,f,32,32], d1: [B,f,16,16]
        e2, d2 = self.enc2(d1)         # e2: [B,2f,16,16], d2: [B,2f,8,8]
        e3, d3 = self.enc3(d2)         # e3: [B,4f,8,8],   d3: [B,4f,4,4]

        # Bottleneck
        b = self.bottleneck(d3)        # [B, 4f, 4, 4]

        # Decoder with skip connections
        u3 = self.dec3(b, e3)          # [B, 2f, 8, 8]
        u2 = self.dec2(u3, e2)         # [B, f, 16, 16]
        u1 = self.dec1(u2, e1)         # [B, f, 32, 32]

        # Heads
        position_logits = self.position_head(u1)   # [B, K, 32, 32]
        action_logits = self.action_head(u1)        # [B, 12, 32, 32]

        return position_logits, action_logits



def compute_loss(position_logits, action_logits, target_y, target_x, target_action,
                 boundary_masks, label_smoothing=0.1, pos_weight=3.0,
                 diversity_weight=0.5):
    """
    Winner-Takes-All (WTA) loss with diversity regularizer.

    1. Each of K hypotheses produces a softmax over boundary positions.
    2. Only the closest hypothesis to GT receives gradients (WTA).
    3. Diversity: soft entropy over winner assignments — penalizes mode collapse
       where one hypothesis wins everything.

    Returns: total_loss, pos_loss, act_loss, diversity_loss
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

    # WTA: min across hypotheses — gradients flow only to winner
    pos_loss = per_hyp_loss.min(dim=1).values.mean()

    # Diversity: soft winner assignment entropy
    # softmin gives probability of each hypothesis being the winner
    winner_probs = F.softmax(-per_hyp_loss, dim=1)              # [B, K]
    avg_usage = winner_probs.mean(dim=0)                         # [K]
    # Max entropy = log(K) when uniform; diversity_loss=0 when perfectly balanced
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
