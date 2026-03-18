"""
CNN+RNN ablation model.

Architecture: 4-level CNN encoder + GRU history encoder, fused via
concatenation at the bottleneck. Simple upsampling decoder (no skip
connections, no FiLM, no bottleneck attention).

Tests: whether adding temporal history (RNN) helps even without spatial
skip connections (U-Net) or sophisticated fusion (FiLM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

_model_dir = Path(__file__).resolve().parent.parent.parent
if str(_model_dir) not in sys.path:
    sys.path.insert(0, str(_model_dir))

from fusion_model import ResBlock, DownBlock, HistoryEncoder, NUM_ACTIONS


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


class CNNRNN(nn.Module):
    """CNN encoder + GRU, concat fusion at bottleneck, simple upsample decoder."""

    def __init__(self, in_channels=9, base_features=48, n_hypotheses=1,
                 max_history=32, rnn_hidden=192, rnn_layers=2,
                 max_grid_size=128, rnn_dropout=0.15, **kwargs):
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = DownBlock(in_channels, f)
        self.enc2 = DownBlock(f, f * 2)
        self.enc3 = DownBlock(f * 2, f * 4)
        self.enc4 = DownBlock(f * 4, f * 4)

        # Bottleneck (no attention)
        self.bottleneck = nn.Sequential(
            ResBlock(f * 4),
            nn.Dropout2d(0.15),
        )

        # RNN branch
        self.history_encoder = HistoryEncoder(
            max_grid_size=max_grid_size,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
            dropout=rnn_dropout,
        )

        # Fusion: concat RNN context with bottleneck, project back
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(f * 4 + rnn_hidden, f * 4, 1, bias=False),
            nn.GroupNorm(min(8, f * 4), f * 4),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # Simple decoder (no skip connections)
        self.up4 = SimpleUpBlock(f * 4, f * 4)
        self.up3 = SimpleUpBlock(f * 4, f * 2)
        self.up2 = SimpleUpBlock(f * 2, f)
        self.up1 = SimpleUpBlock(f, f)

        # Output heads
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
        # CNN encoder
        _, d1 = self.enc1(state)
        _, d2 = self.enc2(d1)
        _, d3 = self.enc3(d2)
        _, d4 = self.enc4(d3)

        b = self.bottleneck(d4)  # [B, 4f, 8, 8]

        # RNN branch
        context = self.history_encoder(
            hist_actions, hist_py, hist_px, hist_cb, hist_ca, hist_mask
        )  # [B, rnn_hidden]

        # Fusion: tile context spatially and concat with bottleneck
        B, C, H, W = b.shape
        ctx_spatial = context.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        fused = torch.cat([b, ctx_spatial], dim=1)  # [B, 4f+rnn_hidden, 8, 8]
        fused = self.fusion_proj(fused)             # [B, 4f, 8, 8]

        # Simple upsample decoder
        u4 = self.up4(fused)
        u3 = self.up3(u4)
        u2 = self.up2(u3)
        u1 = self.up1(u2)

        position_logits = self.position_head(u1)
        action_logits = self.action_head(u1)
        return position_logits, action_logits
