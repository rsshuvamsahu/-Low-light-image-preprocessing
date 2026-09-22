"""
Classical Computer Vision Algorithms for Low-Light Image Preprocessing.

Includes:
- Global and Adaptive Gamma Correction (AGC)
- Multi-Color Space CLAHE (Lab, HSV, RGB)
- Single-Scale & Multi-Scale Retinex with Color Restoration (SSR, MSR, MSRCR)
- Edge-preserving denoising (Bilateral, Guided Filter, Non-Local Means)
- Gray-World Color Constancy / Auto White Balance
- Modular ClassicalPipeline for chained execution
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Union, Dict, Any


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure image array is in uint8 [0, 255] format."""
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0 + 1e-6:
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def ensure_float32(img: np.ndarray) -> np.ndarray:
    """Ensure image array is in float32 [0.0, 1.0] format."""
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() > 1.0 + 1e-6:
            return (img / 255.0).astype(np.float32)
        return img.astype(np.float32)
    return (img.astype(np.float32) / 255.0)


# =====================================================================
# 1. GAMMA CORRECTION
# =====================================================================

def gamma_correction(img: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """
    Standard Power-Law (Gamma) Transformation:
    I_out = 255 * (I_in / 255) ** gamma

    Args:
        img: Input image (uint8 or float32, BGR or RGB)
        gamma: Exponent (< 1.0 brightens shadows, > 1.0 darkens)
    Returns:
        Enhanced image (uint8)
    """
    img_u8 = ensure_uint8(img)
    inv_gamma = gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img_u8, table)


def adaptive_gamma_correction(img: np.ndarray, alpha: float = 0.5, epsilon: float = 1e-6) -> np.ndarray:
    """
    Adaptive Gamma Correction (AGC) based on image illumination statistics.
    Calculates gamma dynamically based on the mean luminance:
    gamma = -log2(mean_luminance + eps) * alpha

    Args:
        img: Input image (uint8, BGR or RGB)
        alpha: Scaling sensitivity factor
        epsilon: Numerical stability term
    Returns:
        Enhanced image (uint8)
    """
    img_u8 = ensure_uint8(img)
    # Compute mean intensity across luminance
    if len(img_u8.shape) == 3:
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY) if img_u8.shape[2] == 3 else img_u8
    else:
        gray = img_u8

    norm_mean = float(np.mean(gray)) / 255.0
    # Dynamic gamma calculation: low mean -> lower gamma (higher brightness boost)
    gamma = np.clip(-np.log2(norm_mean + epsilon) * alpha, 0.2, 1.0)
    return gamma_correction(img_u8, gamma=float(gamma))


# =====================================================================
# 2. CONTRAST-LIMITED ADAPTIVE HISTOGRAM EQUALIZATION (CLAHE)
# =====================================================================

def clahe_lab(img: np.ndarray, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8), is_rgb: bool = True) -> np.ndarray:
    """
    Apply CLAHE in CIE-Lab Color Space.
    CLAHE is applied ONLY to the L (Luminance) channel to avoid color distortion.

    Args:
        img: Input image (uint8)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Grid size for local histogram equalization (default 8x8)
        is_rgb: True if RGB format, False if BGR
    Returns:
        Enhanced image (uint8, same color format as input)
    """
    img_u8 = ensure_uint8(img)
    if is_rgb:
        lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    else:
        lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)

    merged_lab = cv2.merge((cl, a_channel, b_channel))
    if is_rgb:
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def clahe_hsv(img: np.ndarray, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8), is_rgb: bool = True) -> np.ndarray:
    """
    Apply CLAHE in HSV Color Space on the V (Value/Brightness) channel.
    """
    img_u8 = ensure_uint8(img)
    if is_rgb:
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    else:
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cv_channel = clahe.apply(v)

    merged = cv2.merge((h, s, cv_channel))
    if is_rgb:
        return cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


# =====================================================================
# 3. RETINEX ALGORITHMS (SSR, MSR, MSRCR)
# =====================================================================

def single_scale_retinex(img: np.ndarray, sigma: float = 80.0) -> np.ndarray:
    """
    Single-Scale Retinex (SSR):
    R(x, y) = log(I(x, y)) - log(I(x, y) * G(x, y, sigma))

    Args:
        img: Input image (float32 [0, 1] or uint8)
        sigma: Gaussian blur standard deviation
    Returns:
        Retinex reflectance component (float32)
    """
    img_f = ensure_float32(img)
    eps = 1e-6
    blur = cv2.GaussianBlur(img_f, (0, 0), sigma)
    # log(I) - log(blur)
    retinex = np.log10(img_f + eps) - np.log10(blur + eps)
    return retinex


def multi_scale_retinex(img: np.ndarray, sigma_list: List[float] = [15.0, 80.0, 250.0], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Multi-Scale Retinex (MSR):
    Weighted sum of multiple SSR outputs across small, medium, and large scales.

    Args:
        img: Input image
        sigma_list: Gaussian scales (captures fine, medium, and wide contextual illumination)
        weights: Weights for each scale (default equal weights)
    Returns:
        MSR reflectance output (float32)
    """
    if weights is None:
        weights = [1.0 / len(sigma_list)] * len(sigma_list)

    msr = np.zeros_like(ensure_float32(img), dtype=np.float32)
    for sigma, w in zip(sigma_list, weights):
        msr += w * single_scale_retinex(img, sigma=sigma)
    return msr


def msrcr(
    img: np.ndarray,
    sigma_list: List[float] = [15.0, 80.0, 250.0],
    alpha: float = 125.0,
    beta: float = 46.0,
    dynamic_range_factor: float = 2.0
) -> np.ndarray:
    """
    Multi-Scale Retinex with Color Restoration (MSRCR).
    Adds color restoration factor C_i(x, y) to prevent grayscale desaturation:
    C_i = beta * (log(alpha * I_i) - log(sum(I_c)))
    MSRCR_i = C_i * MSR_i

    Args:
        img: Input image (uint8 or float32, RGB or BGR)
        sigma_list: Scale standard deviations
        alpha: Color restoration scale parameter
        beta: Color restoration gain
        dynamic_range_factor: Standard deviation multiplier for dynamic range clipping
    Returns:
        Enhanced image (uint8)
    """
    img_f = ensure_float32(img)
    eps = 1e-6

    # 1. Multi-Scale Retinex
    msr = multi_scale_retinex(img_f, sigma_list=sigma_list)

    # 2. Color Restoration Factor
    if len(img_f.shape) == 3:
        sum_channels = np.sum(img_f, axis=2, keepdims=True) + eps
        crf = beta * (np.log10(alpha * img_f + eps) - np.log10(sum_channels))
        msrcr_val = crf * msr
    else:
        msrcr_val = msr

    # 3. Dynamic Range Normalization (Simplest Color Balance / Stretch)
    out = np.zeros_like(msrcr_val)
    if len(msrcr_val.shape) == 3:
        for c in range(msrcr_val.shape[2]):
            mean = np.mean(msrcr_val[:, :, c])
            std = np.std(msrcr_val[:, :, c])
            min_val = mean - dynamic_range_factor * std
            max_val = mean + dynamic_range_factor * std
            channel = np.clip((msrcr_val[:, :, c] - min_val) / (max_val - min_val + eps) * 255.0, 0, 255)
            out[:, :, c] = channel
    else:
        mean = np.mean(msrcr_val)
        std = np.std(msrcr_val)
        min_val = mean - dynamic_range_factor * std
        max_val = mean + dynamic_range_factor * std
        out = np.clip((msrcr_val - min_val) / (max_val - min_val + eps) * 255.0, 0, 255)

    return out.astype(np.uint8)


# =====================================================================
# 4. EDGE-PRESERVING DENOISING & GUIDED FILTERING
# =====================================================================

def bilateral_denoise(img: np.ndarray, d: int = 7, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """
    Bilateral Filter: Smooths sensor noise while preserving high-contrast edges.
    """
    img_u8 = ensure_uint8(img)
    return cv2.bilateralFilter(img_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def guided_filter(guidance: np.ndarray, src: np.ndarray, radius: int = 4, eps: float = 0.04) -> np.ndarray:
    """
    Guided Filter implementation for edge-preserving smoothing and detail recovery.
    Uses guidance image to steer the filtering of src.
    """
    I = ensure_float32(guidance)
    p = ensure_float32(src)

    # Convert to single-channel guidance luminance if multi-channel
    if len(I.shape) == 3 and len(p.shape) == 3:
        # Filter each channel independently
        out = np.zeros_like(p)
        for c in range(p.shape[2]):
            out[:, :, c] = _guided_filter_2d(I[:, :, c], p[:, :, c], radius, eps)
        return ensure_uint8(out)
    elif len(I.shape) == 3:
        I_gray = cv2.cvtColor(ensure_uint8(guidance), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        out = np.zeros_like(p)
        for c in range(p.shape[2]):
            out[:, :, c] = _guided_filter_2d(I_gray, p[:, :, c], radius, eps)
        return ensure_uint8(out)
    else:
        out = _guided_filter_2d(I, p, radius, eps)
        return ensure_uint8(out)


def _guided_filter_2d(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """2D single channel guided filter."""
    ksize = (2 * r + 1, 2 * r + 1)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    corr_I = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    q = mean_a * I + mean_b
    return np.clip(q, 0.0, 1.0)


# =====================================================================
# 5. WHITE BALANCE / COLOR CONSTANCY
# =====================================================================

def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    """
    Gray World White Balance:
    Assumes average scene reflectance is neutral gray across R, G, B channels.
    """
    img_f = ensure_float32(img)
    if len(img_f.shape) != 3 or img_f.shape[2] != 3:
        return ensure_uint8(img)

    mean_r = np.mean(img_f[:, :, 0]) + 1e-6
    mean_g = np.mean(img_f[:, :, 1]) + 1e-6
    mean_b = np.mean(img_f[:, :, 2]) + 1e-6
    gray_avg = (mean_r + mean_g + mean_b) / 3.0

    out = np.zeros_like(img_f)
    out[:, :, 0] = img_f[:, :, 0] * (gray_avg / mean_r)
    out[:, :, 1] = img_f[:, :, 1] * (gray_avg / mean_g)
    out[:, :, 2] = img_f[:, :, 2] * (gray_avg / mean_b)

    return ensure_uint8(np.clip(out, 0.0, 1.0))


# =====================================================================
# 6. COMPOSITE CLASSICAL PREPROCESSING PIPELINE
# =====================================================================

class ClassicalPipeline:
    """
    Configurable multi-stage classical enhancement and preprocessing pipeline.
    Combines Denoising -> Exposure Normalization -> Color Balance -> CLAHE/Retinex.
    """

    def __init__(
        self,
        denoise_type: str = "bilateral",     # 'none', 'bilateral', 'guided'
        denoise_strength: float = 35.0,
        enhancer_type: str = "clahe_lab",    # 'clahe_lab', 'clahe_hsv', 'msrcr', 'gamma'
        clahe_clip_limit: float = 2.5,
        gamma: float = 0.55,
        apply_white_balance: bool = True
    ):
        self.denoise_type = denoise_type
        self.denoise_strength = denoise_strength
        self.enhancer_type = enhancer_type
        self.clahe_clip_limit = clahe_clip_limit
        self.gamma = gamma
        self.apply_white_balance = apply_white_balance

    def process(self, img: np.ndarray, is_rgb: bool = True) -> np.ndarray:
        """Execute full classical pipeline on input image."""
        img_u8 = ensure_uint8(img)

        # Stage 1: Edge-Preserving Pre-Denoise
        if self.denoise_type == "bilateral":
            d = 5
            stage1 = bilateral_denoise(img_u8, d=d, sigma_color=self.denoise_strength, sigma_space=self.denoise_strength)
        elif self.denoise_type == "guided":
            stage1 = guided_filter(img_u8, img_u8, radius=3, eps=0.02)
        else:
            stage1 = img_u8

        # Stage 2: Illumination Enhancement
        if self.enhancer_type == "clahe_lab":
            stage2 = clahe_lab(stage1, clip_limit=self.clahe_clip_limit, is_rgb=is_rgb)
        elif self.enhancer_type == "clahe_hsv":
            stage2 = clahe_hsv(stage1, clip_limit=self.clahe_clip_limit, is_rgb=is_rgb)
        elif self.enhancer_type == "msrcr":
            stage2 = msrcr(stage1)
        elif self.enhancer_type == "gamma":
            stage2 = gamma_correction(stage1, gamma=self.gamma)
        elif self.enhancer_type == "adaptive_gamma":
            stage2 = adaptive_gamma_correction(stage1)
        else:
            stage2 = stage1

        # Stage 3: Color Constancy / Auto White Balance
        if self.apply_white_balance:
            out = gray_world_white_balance(stage2)
        else:
            out = stage2

        return out
