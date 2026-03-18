"""
CNN-Only ablation model.

Architecture: 4-level CNN encoder → simple bilinear upsample decoder (no skip
connections, no RNN, no FiLM, no bottleneck attention).

Tests: baseline CNN encoder capability without any architectural enhancements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

_model_dir = Path(__file__).resolve().parent.parent.parent  # src/model/
if str(_model_dir) not in sys.path:
    sys.path.insert(0, str(_model_dir))

from fusion_model import ResBlock, DownBlock, NUM_ACTIONS


class SimpleUpBlock(nn.Module):
    """Bilinear upsample + 1x1 projection (no skip connections)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.res = ResBlock(out_ch)

    def forward(self, x):
        return self.res(self.proj(self.up(x)))


class CNNOnly(nn.Module):
    """CNN encoder with simple upsampling decoder. No U-Net skips, no RNN, no FiLM."""

    def __init__(self, in_channels=9, base_features=48, n_hypotheses=1,
                 max_grid_size=128, **kwargs):
        super().__init__()
        f = base_features

        # Encoder (identical to FusionNet)
        self.enc1 = DownBlock(in_channels, f)       # 128->64
        self.enc2 = DownBlock(f, f * 2)             # 64->32
        self.enc3 = DownBlock(f * 2, f * 4)         # 32->16
        self.enc4 = DownBlock(f * 4, f * 4)         # 16->8

        # Bottleneck (no attention)
        self.bottleneck = nn.Sequential(
            ResBlock(f * 4),
            nn.Dropout2d(0.15),
        )

        # Simple decoder (no skip connections)
        self.up4 = SimpleUpBlock(f * 4, f * 4)      # 8->16
        self.up3 = SimpleUpBlock(f * 4, f * 2)      # 16->32
        self.up2 = SimpleUpBlock(f * 2, f)           # 32->64
        self.up1 = SimpleUpBlock(f, f)               # 64->128

        # Output heads (identical to FusionNet)
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
        # Ignore all history inputs
        _, d1 = self.enc1(state)
        _, d2 = self.enc2(d1)
        _, d3 = self.enc3(d2)
        _, d4 = self.enc4(d3)

        b = self.bottleneck(d4)

        u4 = self.up4(b)
        u3 = self.up3(u4)
        u2 = self.up2(u3)
        u1 = self.up1(u2)

        position_logits = self.position_head(u1)
        action_logits = self.action_head(u1)
        return position_logits, action_logits
