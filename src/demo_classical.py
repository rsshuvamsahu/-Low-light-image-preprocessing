"""
Quick Demonstration Script for Classical Low-Light Preprocessing Pipeline.

Usage:
    python src/demo_classical.py
    python src/demo_classical.py --input path/to/your_dark_image.jpg
"""

import os
import sys
import argparse
import cv2
import numpy as np

# Ensure Python finds the core module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.classical import (
    ClassicalPipeline,
    msrcr,
    clahe_lab,
    gamma_correction,
    adaptive_gamma_correction,
    bilateral_denoise
)


def create_sample_dark_image() -> np.ndarray:
    """Creates a sample low-light test image if no input image is provided."""
    h, w = 400, 600
    img = np.zeros((h, w, 3), dtype=np.float32)

    # Add shapes with vibrant colors
    cv2.circle(img, (150, 150), 80, (0.2, 0.3, 0.9), -1)   # Blue
    cv2.rectangle(img, (280, 80), (420, 220), (0.2, 0.8, 0.3), -1) # Green
    cv2.circle(img, (480, 250), 70, (0.9, 0.2, 0.2), -1)   # Red
    cv2.putText(img, "Night Vision Test", (80, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0.9, 0.9, 0.9), 2)

    # Darken significantly and add sensor noise
    dark = (img ** 2.2) * 0.12
    noise = np.random.normal(0, 0.015, dark.shape).astype(np.float32)
    noisy_dark = np.clip(dark + noise, 0.0, 1.0)
    return (noisy_dark * 255.0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Test Classical Low-Light Enhancement")
    parser.add_argument("--input", type=str, default=None, help="Path to input dark image (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load or generate dark image
    if args.input and os.path.exists(args.input):
        print(f"[*] Loading input image: {args.input}")
        img = cv2.imread(args.input)
    else:
        print("[*] No input image provided. Generating synthetic dark test image...")
        img = create_sample_dark_image()
        cv2.imwrite(os.path.join(args.output_dir, "0_input_dark.png"), img)

    # 2. Method 1: Multi-Scale Retinex with Color Restoration (MSRCR)
    print("[*] Running Multi-Scale Retinex (MSRCR)...")
    enhanced_msrcr = msrcr(img)
    cv2.imwrite(os.path.join(args.output_dir, "1_enhanced_msrcr.png"), enhanced_msrcr)

    # 3. Method 2: CLAHE in Lab Color Space
    print("[*] Running CLAHE in Lab color space...")
    enhanced_clahe = clahe_lab(img, clip_limit=3.0, is_rgb=False)
    cv2.imwrite(os.path.join(args.output_dir, "2_enhanced_clahe.png"), enhanced_clahe)

    # 4. Method 3: Full Composite Preprocessing Pipeline (Denoise + CLAHE + White Balance)
    print("[*] Running Composite Classical Pipeline (Denoise + CLAHE + White Balance)...")
    pipeline = ClassicalPipeline(
        denoise_type="bilateral",
        denoise_strength=35.0,
        enhancer_type="clahe_lab",
        clahe_clip_limit=2.5,
        apply_white_balance=True
    )
    enhanced_pipeline = pipeline.process(img, is_rgb=False)
    cv2.imwrite(os.path.join(args.output_dir, "3_enhanced_pipeline.png"), enhanced_pipeline)

    # 5. Create a side-by-side comparison grid
    grid = np.hstack([img, enhanced_clahe, enhanced_msrcr, enhanced_pipeline])
    grid_path = os.path.join(args.output_dir, "comparison_grid.png")
    cv2.imwrite(grid_path, grid)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] All enhanced images saved to '{args.output_dir}/'")
    print(f"  - Original Dark:     {args.output_dir}/0_input_dark.png")
    print(f"  - MSRCR Retinex:     {args.output_dir}/1_enhanced_msrcr.png")
    print(f"  - Lab CLAHE:         {args.output_dir}/2_enhanced_clahe.png")
    print(f"  - Composite Pipeline:{args.output_dir}/3_enhanced_pipeline.png")
    print(f"  - Side-by-Side Grid: {grid_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
