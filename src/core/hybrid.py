"""
Hybrid Preprocessing + CNN Enhancement Pipeline.

Combines:
1. Classical Pre-Stage: Edge-preserving noise suppression & dynamic range conditioning
2. Deep Learning Core: Deep feature illumination & color reconstruction via CNN
3. Classical Post-Stage: Guided filter high-frequency detail injection and dynamic range refinement
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple

from .classical import (
    bilateral_denoise,
    guided_filter,
    adaptive_gamma_correction,
    gray_world_white_balance,
    ensure_uint8,
    ensure_float32
)
from .models import build_model, load_checkpoint


class HybridEnhancer:
    """
    Synergistic Hybrid Pipeline combining classical signal pre-conditioning with CNN inference.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_name: str = "res_cnn",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        pre_denoise: bool = True,
        denoise_strength: float = 25.0,
        post_guided_refine: bool = True,
        guided_radius: int = 3,
        guided_eps: float = 0.02
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = build_model(model_name, device=self.device)
            if checkpoint_path and os.path.exists(checkpoint_path):
                load_checkpoint(self.model, checkpoint_path, device=self.device)

        self.model.eval()
        self.pre_denoise = pre_denoise
        self.denoise_strength = denoise_strength
        self.post_guided_refine = post_guided_refine
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps

    def _preprocess(self, img_u8: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Apply classical pre-filtering and convert to PyTorch tensor."""
        if self.pre_denoise:
            # Stage 1: Classical bilateral noise attenuation
            clean_pre = bilateral_denoise(img_u8, d=5, sigma_color=self.denoise_strength, sigma_space=self.denoise_strength)
        else:
            clean_pre = img_u8

        # Stage 2: Convert to float tensor [1, 3, H, W]
        img_f = (clean_pre.astype(np.float32) / 255.0)
        tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return clean_pre, tensor

    @torch.no_grad()
    def enhance_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run deep learning model on input tensor."""
        return self.model(tensor)

    def enhance(self, img: np.ndarray, is_rgb: bool = True) -> np.ndarray:
        """
        Run the full 3-stage Hybrid Pipeline on an input image.

        Args:
            img: Input image (uint8 or float32, RGB or BGR)
            is_rgb: True if RGB channel order, False if BGR
        Returns:
            Enhanced image (uint8, same color format as input)
        """
        img_u8 = ensure_uint8(img)
        if not is_rgb:
            img_u8_rgb = cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB)
        else:
            img_u8_rgb = img_u8

        # Step 1: Classical Pre-processing
        pre_img, input_tensor = self._preprocess(img_u8_rgb)

        # Step 2: CNN Core Enhancement
        enhanced_tensor = self.enhance_tensor(input_tensor)
        cnn_out_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        cnn_out_u8 = np.clip(cnn_out_np * 255.0, 0, 255).astype(np.uint8)

        # Step 3: Classical Post-processing (Guided detail fusion)
        if self.post_guided_refine:
            # Use original pre-filtered luminance as guidance to inject sharp edges
            final_u8 = guided_filter(
                guidance=pre_img,
                src=cnn_out_u8,
                radius=self.guided_radius,
                eps=self.guided_eps
            )
        else:
            final_u8 = cnn_out_u8

        if not is_rgb:
            return cv2.cvtColor(final_u8, cv2.COLOR_RGB2BGR)
        return final_u8
