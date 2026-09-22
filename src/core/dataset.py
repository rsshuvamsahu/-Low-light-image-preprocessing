"""
Dataset Loaders and Synthetic Low-Light Generators for Training & Evaluation.

Includes:
1. LOLDataset: PyTorch Dataset for paired Low-Light / High-Light images
2. SyntheticLowLightGenerator: Creates synthetic paired dark data for offline development
"""

import os
import glob
import cv2
import numpy as np
from typing import Tuple, List, Optional


class LOLDataset:
    """
    LOL (Low-Light) Dataset Loader for paired (low, high) image training & evaluation.
    Supports random cropping, horizontal/vertical flips, and color normalization.
    """

    def __init__(self, root_dir: str, split: str = "train", crop_size: Optional[int] = 256, is_training: bool = True):
        self.root_dir = root_dir
        self.split = split
        self.crop_size = crop_size
        self.is_training = is_training

        # Support both standard directory layouts:
        # 1. root/our485/low and root/our485/high  (or root/eval15/low and root/eval15/high)
        # 2. root/train/low and root/train/high (or root/val/low and root/val/high)
        subfolder = "our485" if split == "train" else "eval15"
        low_dir = os.path.join(root_dir, subfolder, "low")
        high_dir = os.path.join(root_dir, subfolder, "high")

        if not os.path.exists(low_dir):
            low_dir = os.path.join(root_dir, split, "low")
            high_dir = os.path.join(root_dir, split, "high")

        self.low_files = sorted(glob.glob(os.path.join(low_dir, "*.png")) + glob.glob(os.path.join(low_dir, "*.jpg")))
        self.high_files = sorted(glob.glob(os.path.join(high_dir, "*.png")) + glob.glob(os.path.join(high_dir, "*.jpg")))

    def __len__(self) -> int:
        return len(self.low_files)

    def __getitem__(self, idx: int):
        low_path = self.low_files[idx]
        high_path = self.high_files[idx]

        low_img = cv2.imread(low_path)
        high_img = cv2.imread(high_path)

        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)

        # Apply random crop during training
        if self.is_training and self.crop_size is not None:
            h, w, _ = low_img.shape
            if h > self.crop_size and w > self.crop_size:
                top = np.random.randint(0, h - self.crop_size)
                left = np.random.randint(0, w - self.crop_size)
                low_img = low_img[top:top + self.crop_size, left:left + self.crop_size]
                high_img = high_img[top:top + self.crop_size, left:left + self.crop_size]

            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                low_img = np.fliplr(low_img).copy()
                high_img = np.fliplr(high_img).copy()

            # Random Vertical Flip
            if np.random.rand() > 0.5:
                low_img = np.flipud(low_img).copy()
                high_img = np.flipud(high_img).copy()

        # Convert to float [0, 1]
        low_f = low_img.astype(np.float32) / 255.0
        high_f = high_img.astype(np.float32) / 255.0

        try:
            import torch
            t_low = torch.from_numpy(low_f).permute(2, 0, 1)
            t_high = torch.from_numpy(high_f).permute(2, 0, 1)
            return t_low, t_high
        except Exception:
            return low_f, high_f


def generate_synthetic_dataset(output_dir: str, num_samples: int = 20, img_size: Tuple[int, int] = (256, 256)):
    """
    Generates synthetic paired low-light/normal-light dataset for testing training loops offline.
    """
    os.makedirs(os.path.join(output_dir, "train", "low"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "high"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "eval15", "low"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "eval15", "high"), exist_ok=True)

    h, w = img_size
    for i in range(num_samples):
        # Create rich high-light reference image
        high = np.zeros((h, w, 3), dtype=np.float32)
        # Random colorful shapes
        for _ in range(6):
            pt1 = (np.random.randint(10, w - 50), np.random.randint(10, h - 50))
            pt2 = (pt1[0] + np.random.randint(30, 80), pt1[1] + np.random.randint(30, 80))
            color = list(np.random.rand(3))
            cv2.rectangle(high, pt1, pt2, color, -1)

        # Smooth ambient gradient
        x = np.linspace(0, 1, w)[:, None]
        y = np.linspace(0, 1, h)[None, :]
        high = np.clip(high * 0.7 + (x * 0.3 + y * 0.3)[:, :, None], 0.0, 1.0)

        # Simulate low-light degradation: Illumination decay + Non-linear response + Noise
        alpha = np.random.uniform(0.08, 0.15)
        gamma = np.random.uniform(1.8, 2.5)
        dark = (high ** gamma) * alpha
        noise = np.random.normal(0, 0.012, dark.shape).astype(np.float32)
        low = np.clip(dark + noise, 0.0, 1.0)

        high_u8 = (high * 255.0).astype(np.uint8)
        low_u8 = (low * 255.0).astype(np.uint8)

        split = "eval15" if i >= (num_samples - 5) else "train"
        filename = f"sample_{i:04d}.png"
        cv2.imwrite(os.path.join(output_dir, split, "high", filename), cv2.cvtColor(high_u8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, split, "low", filename), cv2.cvtColor(low_u8, cv2.COLOR_RGB2BGR))

    print(f"[SUCCESS] Synthetic paired dataset generated at '{output_dir}' with {num_samples} samples.")
