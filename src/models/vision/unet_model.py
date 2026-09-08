"""Mini U-Net Supervised Segmentation Model with Bottleneck Feature Extractor.

Architecture:
  Encoder: Conv Blocks with MaxPool
  Bottleneck: Deep Feature Map -> Image Embedding Extractor
  Decoder: Upconv Blocks with Skip Connections
  Output: 1-channel Mask Probability Map (Sigmoid)
  + Image Embedding extraction for Multimodal Fusion.

Follows ADR Phase 4 specifications.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.config import IMAGE_EMBED_DIM


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2 block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MiniUNet(nn.Module):
    """Mini U-Net for supervised industrial defect segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        embed_dim: int = IMAGE_EMBED_DIM,
        base_filters: int = 32
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        # Encoder (Contracting Path)
        self.enc1 = DoubleConv(in_channels, base_filters)       # 32
        self.pool1 = nn.MaxPool2d(2)                            # 128x128

        self.enc2 = DoubleConv(base_filters, base_filters * 2)  # 64
        self.pool2 = nn.MaxPool2d(2)                            # 64x64

        self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)  # 128
        self.pool3 = nn.MaxPool2d(2)                            # 32x32

        # Bottleneck (32x32x256)
        self.bottleneck = DoubleConv(base_filters * 4, base_filters * 8)  # 256

        # Feature Extractor for Fusion (Global Average Pool + Linear to embed_dim)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_proj = nn.Sequential(
            nn.Linear(base_filters * 8, embed_dim),
            nn.ReLU(inplace=True)
        )

        # Decoder (Expanding Path)
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_filters * 2, base_filters)

        # Output Segmentation Head
        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def extract_image_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck feature embedding for Multimodal Fusion.

        Args:
            x: Image tensor (B, 3, 256, 256)

        Returns:
            embedding: (B, embed_dim)
        """
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        # GAP + Projection
        pooled = self.gap(b).view(b.size(0), -1)
        embedding = self.embed_proj(pooled)

        return embedding

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input image tensor (B, 3, H, W)

        Returns:
            mask_logits: (B, 1, H, W)
            image_embedding: (B, embed_dim)
        """
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Extract Embedding
        pooled = self.gap(b).view(b.size(0), -1)
        embedding = self.embed_proj(pooled)

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output logits
        mask_logits = self.out_conv(d1)

        return mask_logits, embedding
