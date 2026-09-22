"""
Deep Learning CNN Architectures for Low-Light Image Enhancement.

Includes:
1. LowLightResCNN: Full-resolution lightweight residual network (preserves high frequencies)
2. UNetEnhancer: Multi-scale encoder-decoder with residual blocks and skip connections
3. ZeroDCENet: Deep Curve Estimation network for real-time tone-curve adjustment
4. Helper functions for model instantiation and checkpoint loading across CPU/CUDA
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


# =====================================================================
# 1. LIGHTWEIGHT RESIDUAL CNN (ResCNN)
# =====================================================================

class ResidualBlock(nn.Module):
    """
    Residual Block with Conv-ReLU-Conv and identity skip connection.
    No BatchNorm to ensure stability with single-image inference and small batch sizes.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class LowLightResCNN(nn.Module):
    """
    Full-Resolution Residual CNN for Low-Light Enhancement.
    Operates without downsampling to retain all fine-grained spatial high-frequencies.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_features: int = 32, num_blocks: int = 6):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True),
            nn.PReLU()
        )
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        # Initialize final conv layer so untrained model performs smooth brightening curve
        nn.init.xavier_normal_(self.tail[2].weight, gain=0.1)
        nn.init.constant_(self.tail[2].bias, 1.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        res = self.body(feat)
        out = self.tail(feat + res)
        # Input skip to ensure identity baseline
        return torch.clamp(x * 0.4 + out * 0.7, 0.0, 1.0)


# =====================================================================
# 2. MULTI-SCALE U-NET ENHANCER
# =====================================================================

class DoubleConv(nn.Module):
    """(Conv2d -> LeakyReLU) * 2 with residual connection option"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class UNetEnhancer(nn.Module):
    """
    Multi-Scale U-Net with Skip Connections for wide illumination context and edge preservation.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_ch: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.down2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.down3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad image to multiples of 8 if needed
        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_padded = x

        # Encode
        e1 = self.enc1(x_padded)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))

        # Decode with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.out_conv(d1)

        # Crop back to original dimensions
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
        return out


# =====================================================================
# 3. ZERO-DCE CURVE ESTIMATION NETWORK
# =====================================================================

class ZeroDCENet(nn.Module):
    """
    Zero-Reference Deep Curve Estimation Network (DCE-Net).
    Estimates pixel-wise high-order illumination curves iteratively:
    LE_n(x) = LE_{n-1}(x) + A_n(x) * LE_{n-1}(x) * (1 - LE_{n-1}(x))
    """
    def __init__(self, num_iterations: int = 8):
        super().__init__()
        self.num_iterations = num_iterations
        num_ch = 32

        self.conv1 = nn.Conv2d(3, num_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(num_ch, num_ch, 3, padding=1)
        self.conv3 = nn.Conv2d(num_ch, num_ch, 3, padding=1)
        self.conv4 = nn.Conv2d(num_ch, num_ch, 3, padding=1)
        self.conv5 = nn.Conv2d(num_ch * 2, num_ch, 3, padding=1)
        self.conv6 = nn.Conv2d(num_ch * 2, num_ch, 3, padding=1)
        self.conv7 = nn.Conv2d(num_ch * 2, 3 * num_iterations, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))

        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        curves = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))

        # Split curve parameter maps into N iterations
        curve_chunks = torch.chunk(curves, self.num_iterations, dim=1)
        enhanced = x
        for a in curve_chunks:
            enhanced = enhanced + a * enhanced * (1.0 - enhanced)
        return torch.clamp(enhanced, 0.0, 1.0)


# =====================================================================
# 4. MODEL BUILDER & LOADER UTILITY
# =====================================================================

def build_model(model_name: str = "res_cnn", device: Optional[str] = None) -> nn.Module:
    """
    Instantiate model by name on appropriate device.
    Options: 'res_cnn', 'unet', 'zero_dce'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = model_name.lower()
    if model_name in ["res_cnn", "rescnn", "residual"]:
        model = LowLightResCNN(in_channels=3, out_channels=3, num_features=32, num_blocks=6)
    elif model_name in ["unet", "u_net"]:
        model = UNetEnhancer(in_channels=3, out_channels=3, base_ch=32)
    elif model_name in ["zero_dce", "zerodce", "dce"]:
        model = ZeroDCENet(num_iterations=8)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from 'res_cnn', 'unet', 'zero_dce'")

    model = model.to(device)
    return model


def load_checkpoint(model: nn.Module, ckpt_path: str, device: Optional[str] = None) -> bool:
    """Safely loads model state dict with CPU/CUDA device mapping."""
    if not os.path.exists(ckpt_path):
        return False

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return True
