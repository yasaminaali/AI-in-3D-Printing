"""
OperationNet: U-Net for per-pixel Hamiltonian path operation prediction.

Maintains full spatial resolution — no compression bottleneck.
Predicts WHERE to apply operations (position head) and WHAT to apply (action head).

Input: 5-channel 32x32 tensor (zones, H_edges, V_edges, boundary_mask, crossing_indicator)
Output: position scores (1x32x32) + action logits (12x32x32)
"""

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
    Lightweight ResU-Net for per-pixel operation prediction.

    Architecture:
        Encoder: 3 levels [24, 48, 96] with additive residual blocks
        Bottleneck: 96 -> 96 (ResBlock + Dropout)
        Decoder: 3 levels with skip connections
        Heads: position (1ch) + action (12ch)

    Input: [B, 5, 32, 32]
    Output: position_logits [B, 1, 32, 32], action_logits [B, 12, 32, 32]
    """

    def __init__(self, in_channels=5, base_features=24):
        super().__init__()

        f = base_features  # 24

        # Encoder
        self.enc1 = DownBlock(in_channels, f)      # 5 -> 24
        self.enc2 = DownBlock(f, f * 2)             # 24 -> 48
        self.enc3 = DownBlock(f * 2, f * 4)         # 48 -> 96

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(f * 4),                         # 96 -> 96
            nn.Dropout2d(0.1),
        )

        # Decoder
        self.dec3 = UpBlock(f * 4, f * 4, f * 2)   # 96+96 -> 48
        self.dec2 = UpBlock(f * 2, f * 2, f)        # 48+48 -> 24
        self.dec1 = UpBlock(f, f, f)                 # 24+24 -> 24

        # Output heads
        self.position_head = nn.Conv2d(f, 1, 1)              # 24 -> 1
        self.action_head = nn.Conv2d(f, NUM_ACTIONS, 1)       # 24 -> 12

    def forward(self, x):
        # Encoder
        e1, d1 = self.enc1(x)          # e1: [B,24,32,32], d1: [B,24,16,16]
        e2, d2 = self.enc2(d1)         # e2: [B,48,16,16], d2: [B,48,8,8]
        e3, d3 = self.enc3(d2)         # e3: [B,96,8,8],   d3: [B,96,4,4]

        # Bottleneck
        b = self.bottleneck(d3)        # [B, 96, 4, 4]

        # Decoder with skip connections
        u3 = self.dec3(b, e3)          # [B, 48, 8, 8]
        u2 = self.dec2(u3, e2)         # [B, 24, 16, 16]
        u1 = self.dec1(u2, e1)         # [B, 24, 32, 32]

        # Heads
        position_logits = self.position_head(u1)   # [B, 1, 32, 32]
        action_logits = self.action_head(u1)        # [B, 12, 32, 32]

        return position_logits, action_logits


def compute_loss(position_logits, action_logits, target_y, target_x, target_action,
                 boundary_masks, label_smoothing=0.1):
    """
    Factored loss that avoids extreme class imbalance.

    Args:
        position_logits: [B, 1, 32, 32] per-pixel position scores
        action_logits: [B, 12, 32, 32] per-pixel action class logits
        target_y: [B] ground-truth y positions
        target_x: [B] ground-truth x positions
        target_action: [B] ground-truth action class (0-11)
        boundary_masks: [B, 32, 32] binary mask of valid prediction positions
        label_smoothing: smoothing factor for action loss

    Returns:
        total_loss, position_loss, action_loss
    """
    batch_size = position_logits.size(0)
    device = position_logits.device

    pos_loss_total = torch.tensor(0.0, device=device)
    act_loss_total = torch.tensor(0.0, device=device)
    valid_count = 0

    for i in range(batch_size):
        # Get boundary positions for this sample
        mask = boundary_masks[i]  # [32, 32]
        boundary_pos = mask.nonzero(as_tuple=False)  # [N_boundary, 2] (y, x)

        if len(boundary_pos) == 0:
            continue

        # Position loss: softmax over boundary positions only
        pos_scores = position_logits[i, 0, boundary_pos[:, 0], boundary_pos[:, 1]]  # [N_boundary]

        # Find target position index in boundary list
        ty, tx = target_y[i].item(), target_x[i].item()
        target_match = (boundary_pos[:, 0] == ty) & (boundary_pos[:, 1] == tx)

        if target_match.any():
            target_idx = target_match.nonzero(as_tuple=False)[0, 0]
            pos_loss_total += F.cross_entropy(pos_scores.unsqueeze(0), target_idx.unsqueeze(0))
        else:
            # Target not in boundary mask — use nearest boundary position
            dists = (boundary_pos[:, 0].float() - ty) ** 2 + (boundary_pos[:, 1].float() - tx) ** 2
            nearest_idx = dists.argmin()
            pos_loss_total += F.cross_entropy(pos_scores.unsqueeze(0), nearest_idx.unsqueeze(0))

        # Action loss: at ground-truth position, predict correct action
        action_at_target = action_logits[i, :, ty, tx]  # [12]
        act_loss_total += F.cross_entropy(
            action_at_target.unsqueeze(0),
            target_action[i:i+1],
            label_smoothing=label_smoothing,
        )

        valid_count += 1

    if valid_count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), \
               torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    pos_loss = pos_loss_total / valid_count
    act_loss = act_loss_total / valid_count
    total_loss = pos_loss + act_loss

    return total_loss, pos_loss, act_loss


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
