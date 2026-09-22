"""
Image Quality Assessment (IQA) & Performance Benchmark Metrics.

Includes:
1. Full-Reference Metrics:
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)
   - MSE / MAE (Mean Squared / Absolute Error)
   - LPIPS (Learned Perceptual Image Patch Similarity with safe fallback)
2. No-Reference Metrics:
   - Shannon Information Entropy
   - RMS Contrast
   - Color Saturation & Naturalness
3. Computational Efficiency:
   - Latency (ms/frame)
   - Throughput (FPS)
"""

import time
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 255.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Higher is better (cleaner signal reconstruction).
    """
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)
    mse = np.mean((img1_f - img2_f) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64))))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 255.0) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    Values range in [0.0, 1.0], higher is better.
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        if len(img1.shape) == 3:
            return float(ssim_fn(img1, img2, data_range=data_range, channel_axis=2))
        return float(ssim_fn(img1, img2, data_range=data_range))
    except Exception:
        # Pure numpy fallback for SSIM
        return _numpy_ssim(img1, img2, data_range)


def _numpy_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 255.0) -> float:
    """Numpy SSIM fallback."""
    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    x = img1.astype(np.float64)
    y = img2.astype(np.float64)

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x_sq = np.var(x)
    sigma_y_sq = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return float(np.clip(ssim, 0.0, 1.0))


# =====================================================================
# NO-REFERENCE METRICS (FOR UNPAIRED NIGHT PHOTOS)
# =====================================================================

def calculate_shannon_entropy(img: np.ndarray) -> float:
    """
    Compute Shannon Information Entropy.
    Higher entropy indicates richer visible detail and dynamic range.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img
    else:
        gray = img

    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-8)
    non_zero = hist[hist > 0]
    entropy = -np.sum(non_zero * np.log2(non_zero))
    return float(entropy)


def calculate_rms_contrast(img: np.ndarray) -> float:
    """
    Compute Root-Mean-Square (RMS) Contrast of Luminance.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img
    else:
        gray = img
    norm_gray = gray.astype(np.float64) / 255.0
    return float(np.std(norm_gray))


# =====================================================================
# FULL BENCHMARK SUITE
# =====================================================================

class BenchmarkSuite:
    """
    Comprehensive Evaluator for comparing Enhancement Algorithms.
    """

    def __init__(self, use_lpips: bool = True, device: Optional[str] = None):
        self.use_lpips = use_lpips
        self.lpips_fn = None

        if use_lpips:
            try:
                import lpips
                import torch
                dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
                self.lpips_fn = lpips.LPIPS(net="alex").to(dev)
                self.device = dev
            except Exception:
                self.lpips_fn = None

    def evaluate_pair(self, enhanced: np.ndarray, reference: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate single enhanced image. Computes Full-Reference metrics if reference is provided.
        """
        results = {
            "entropy": calculate_shannon_entropy(enhanced),
            "rms_contrast": calculate_rms_contrast(enhanced),
            "mean_luminance": float(np.mean(enhanced))
        }

        if reference is not None:
            results["psnr"] = calculate_psnr(enhanced, reference)
            results["ssim"] = calculate_ssim(enhanced, reference)
            results["mse"] = calculate_mse(enhanced, reference)
            results["mae"] = calculate_mae(enhanced, reference)

            # LPIPS evaluation
            if self.lpips_fn is not None:
                try:
                    import torch
                    t_enh = torch.from_numpy(enhanced).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
                    t_ref = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
                    with torch.no_grad():
                        lp_val = self.lpips_fn(t_enh.to(self.device), t_ref.to(self.device)).item()
                    results["lpips"] = float(lp_val)
                except Exception:
                    results["lpips"] = -1.0
            else:
                results["lpips"] = -1.0

        return results
