"""
U-Net Spatial-Only ablation model.

Architecture: Full 4-level ResU-Net with skip connections + bottleneck
self-attention + GroupNorm + refinement heads. No RNN, no FiLM.

Tests: spatial architecture (U-Net + attention) contribution without
temporal history.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

_model_dir = Path(__file__).resolve().parent.parent.parent
if str(_model_dir) not in sys.path:
    sys.path.insert(0, str(_model_dir))

from fusion_model import ResBlock, DownBlock, UpBlock, BottleneckAttention, NUM_ACTIONS


class UNetSpatial(nn.Module):
    """Full U-Net with skip connections and bottleneck attention. No RNN, no FiLM."""

    def __init__(self, in_channels=9, base_features=48, n_hypotheses=1,
                 max_grid_size=128, **kwargs):
        super().__init__()
        f = base_features

        # Encoder (identical to FusionNet)
        self.enc1 = DownBlock(in_channels, f)
        self.enc2 = DownBlock(f, f * 2)
        self.enc3 = DownBlock(f * 2, f * 4)
        self.enc4 = DownBlock(f * 4, f * 4)

        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ResBlock(f * 4),
            nn.Dropout2d(0.15),
        )
        self.bottleneck_attn = BottleneckAttention(f * 4, num_heads=4)

        # Decoder with skip connections (identical to FusionNet, but no FiLM)
        self.dec4 = UpBlock(f * 4, f * 4, f * 4)
        self.dec3 = UpBlock(f * 4, f * 4, f * 2)
        self.dec2 = UpBlock(f * 2, f * 2, f)
        self.dec1 = UpBlock(f, f, f)

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
        e1, d1 = self.enc1(state)
        e2, d2 = self.enc2(d1)
        e3, d3 = self.enc3(d2)
        e4, d4 = self.enc4(d3)

        b = self.bottleneck(d4)
        b = self.bottleneck_attn(b)

        u4 = self.dec4(b, e4)
        u3 = self.dec3(u4, e3)
        u2 = self.dec2(u3, e2)
        u1 = self.dec1(u2, e1)

        position_logits = self.position_head(u1)
        action_logits = self.action_head(u1)
        return position_logits, action_logits
