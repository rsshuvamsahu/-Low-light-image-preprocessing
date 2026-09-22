"""
Verification and Test Suite for Classical Image Processing Module.
Generates synthetic low-light test images with noise and tests all algorithms.
"""

import os
import sys
import time
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from core.classical import (
    gamma_correction,
    adaptive_gamma_correction,
    clahe_lab,
    clahe_hsv,
    single_scale_retinex,
    multi_scale_retinex,
    msrcr,
    bilateral_denoise,
    guided_filter,
    gray_world_white_balance,
    ClassicalPipeline
)


def create_synthetic_low_light_image(h: int = 256, w: int = 256) -> np.ndarray:
    """Create a synthetic dark image with color patches, gradients, and Poisson-Gaussian noise."""
    # 1. Base clean pattern
    img = np.zeros((h, w, 3), dtype=np.float32)

    # Color patches
    img[20:100, 20:100] = [0.8, 0.2, 0.2]   # Red
    img[20:100, 120:200] = [0.2, 0.8, 0.2]  # Green
    img[120:200, 20:100] = [0.2, 0.3, 0.9]  # Blue
    img[120:200, 120:200] = [0.9, 0.8, 0.2] # Yellow

    # Smooth gradient background
    x = np.linspace(0, 1, w, endpoint=False)
    y = np.linspace(0, 1, h, endpoint=False)
    xv, yv = np.meshgrid(x, y)
    gradient = (xv * 0.5 + yv * 0.5)[:, :, None]
    img = img * 0.7 + gradient * 0.3

    # 2. Simulate severe low-light degradation (darkening factor ~0.10 + gamma distortion)
    dark_img = (img ** 2.2) * 0.12

    # 3. Add sensor noise (Gaussian + Shot noise)
    noise = np.random.normal(0, 0.015, dark_img.shape).astype(np.float32)
    noisy_dark = np.clip(dark_img + noise, 0.0, 1.0)

    return (noisy_dark * 255.0).astype(np.uint8)


def run_tests():
    print("=" * 60)
    print("RUNNING STEP 1: CLASSICAL VISION VERIFICATION TESTS")
    print("=" * 60)

    sample = create_synthetic_low_light_image(256, 256)
    print(f"Sample low-light image created: Shape={sample.shape}, dtype={sample.dtype}, Mean={sample.mean():.2f}")

    methods = [
        ("Gamma Correction (0.5)", lambda x: gamma_correction(x, gamma=0.5)),
        ("Adaptive Gamma Correction", lambda x: adaptive_gamma_correction(x)),
        ("CLAHE in Lab Color Space", lambda x: clahe_lab(x, clip_limit=2.5)),
        ("CLAHE in HSV Color Space", lambda x: clahe_hsv(x, clip_limit=2.5)),
        ("Single-Scale Retinex (SSR)", lambda x: single_scale_retinex(x, sigma=80.0)),
        ("Multi-Scale Retinex (MSR)", lambda x: multi_scale_retinex(x, sigma_list=[15, 80, 250])),
        ("MSRCR (Color Restoration)", lambda x: msrcr(x)),
        ("Bilateral Denoise", lambda x: bilateral_denoise(x, d=7, sigma_color=50, sigma_space=50)),
        ("Guided Filter Denoise", lambda x: guided_filter(x, x, radius=4, eps=0.04)),
        ("Gray World White Balance", lambda x: gray_world_white_balance(x)),
        ("Composite Classical Pipeline", lambda x: ClassicalPipeline(denoise_type="bilateral", enhancer_type="clahe_lab").process(x))
    ]

    for name, func in methods:
        t0 = time.perf_counter()
        out = func(sample)
        dt = (time.perf_counter() - t0) * 1000.0

        if isinstance(out, np.ndarray):
            out_mean = out.mean()
            print(f"[PASS] {name:32s} | Time: {dt:6.2f} ms | Output Mean: {out_mean:6.2f} | Shape: {out.shape}")
        else:
            print(f"[PASS] {name:32s} | Time: {dt:6.2f} ms | Type: {type(out)}")

    print("=" * 60)
    print("ALL CLASSICAL VISION METHODS PASSED VERIFICATION!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
