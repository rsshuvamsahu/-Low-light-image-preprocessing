"""
Loss Functions for Supervised and Self-Supervised Low-Light Image Enhancement.

Includes:
1. CharbonnierLoss / L1 Loss (Smooth pixel-wise difference)
2. SSIMLoss (Differentiable Structural Similarity loss)
3. EdgeGradientLoss (Gradient difference loss to enforce sharp edges)
4. PerceptualLoss (VGG feature perceptual similarity with safe fallback)
5. CompositeEnhancementLoss (Weighted combination)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class CharbonnierLoss(nn.Module):
    """Smooth L1 / Charbonnier loss: sqrt((x - y)^2 + eps^2)"""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)


class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity (SSIM) Loss.
    Loss = 1 - SSIM(pred, target)
    """
    def __init__(self, window_size: int = 11, in_channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.register_buffer("window", self._create_window(window_size, in_channels))

    def _gaussian_1d(self, size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, size: int, channels: int) -> torch.Tensor:
        _1d = self._gaussian_1d(size).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2d.expand(channels, 1, size, size).contiguous()
        return window

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # Move window to same device and dtype as input
        window = self.window.to(dtype=img1.dtype, device=img1.device)

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.in_channels)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.in_channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.in_channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.in_channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.in_channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1.0 - ssim_map.mean()


class EdgeGradientLoss(nn.Module):
    """
    Penalizes difference in Sobel edge gradients between prediction and target.
    Prevents over-smoothing and preserves structural details.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _get_gradients(self, img: torch.Tensor) -> torch.Tensor:
        # Convert to grayscale for gradient calculation
        gray = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
        gx = F.conv2d(gray, self.sobel_x.to(device=img.device, dtype=img.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(device=img.device, dtype=img.dtype), padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grad_pred = self._get_gradients(pred)
        grad_target = self._get_gradients(target)
        return F.l1_loss(grad_pred, grad_target)


class CompositeEnhancementLoss(nn.Module):
    """
    Unified multi-task loss combining:
    L_total = w_l1 * L1 + w_ssim * L_SSIM + w_grad * L_Grad
    """
    def __init__(self, w_l1: float = 1.0, w_ssim: float = 0.5, w_grad: float = 0.2):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_grad = w_grad

        self.l1_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.grad_loss = EdgeGradientLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        grad = self.grad_loss(pred, target)

        total = self.w_l1 * l1 + self.w_ssim * ssim + self.w_grad * grad
        return {
            "total": total,
            "l1": l1,
            "ssim": ssim,
            "grad": grad
        }
