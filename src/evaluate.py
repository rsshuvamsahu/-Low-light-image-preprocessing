"""
Comprehensive Benchmark & Ablation Study Evaluation Suite.

Evaluates and compares:
1. Baseline (Raw Unprocessed Low-Light)
2. Gamma Correction
3. Adaptive Gamma Correction
4. CLAHE (Lab Space)
5. CLAHE (HSV Space)
6. Multi-Scale Retinex (MSRCR)
7. Composite Classical Pipeline (Denoise + CLAHE + White Balance)
8. Standalone CNN (LowLightResCNN / UNet)
9. Hybrid Pipeline (Classical Pre-Denoise -> CNN -> Guided Detail Post-Filter)

Generates:
- Formatted Markdown Summary Table
- Detailed CSV Benchmark Report
- Visual Side-by-Side Comparison Grids
"""

import os
import sys
import glob
import time
import argparse
from typing import Optional
import pandas as pd
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.classical import (
    gamma_correction,
    adaptive_gamma_correction,
    clahe_lab,
    clahe_hsv,
    msrcr,
    bilateral_denoise,
    ClassicalPipeline
)
from core.metrics import BenchmarkSuite, calculate_psnr, calculate_ssim
from core.dataset import generate_synthetic_dataset


def run_benchmark(data_dir: str, output_dir: str = "outputs/benchmark", model_ckpt: Optional[str] = "checkpoints/best.pth"):
    os.makedirs(output_dir, exist_ok=True)
    eval_low = sorted(glob.glob(os.path.join(data_dir, "eval15", "low", "*.png")) + glob.glob(os.path.join(data_dir, "eval15", "low", "*.jpg")))
    eval_high = sorted(glob.glob(os.path.join(data_dir, "eval15", "high", "*.png")) + glob.glob(os.path.join(data_dir, "eval15", "high", "*.jpg")))

    if not eval_low or not eval_high:
        print(f"[*] No evaluation pairs found in '{data_dir}/eval15/'. Generating sample pairs...")
        generate_synthetic_dataset(data_dir, num_samples=10, img_size=(256, 256))
        eval_low = sorted(glob.glob(os.path.join(data_dir, "eval15", "low", "*.png")))
        eval_high = sorted(glob.glob(os.path.join(data_dir, "eval15", "high", "*.png")))

    print(f"[*] Evaluating {len(eval_low)} test image pairs...")

    # Load Deep Learning model if available
    cnn_model = None
    hybrid_enhancer = None
    try:
        import torch
        from core.models import build_model, load_checkpoint
        from core.hybrid import HybridEnhancer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        cnn_model = build_model("res_cnn", device=dev)
        if model_ckpt and os.path.exists(model_ckpt):
            load_checkpoint(cnn_model, model_ckpt, device=dev)
            print(f"[*] Loaded trained CNN weights from '{model_ckpt}'")
        else:
            print("[*] Using initialized CNN weights for benchmark.")
        hybrid_enhancer = HybridEnhancer(model=cnn_model, device=dev)
    except Exception as e:
        print(f"[!] Deep learning model skipped ({e})")

    # Define Methods to Benchmark
    classical_pipe = ClassicalPipeline(denoise_type="bilateral", enhancer_type="clahe_lab", apply_white_balance=True)

    methods = {
        "Raw Input (Baseline)": lambda x: x,
        "Gamma Correction (0.55)": lambda x: gamma_correction(x, gamma=0.55),
        "Adaptive Gamma": lambda x: adaptive_gamma_correction(x),
        "CLAHE (Lab Space)": lambda x: clahe_lab(x, clip_limit=2.5, is_rgb=False),
        "CLAHE (HSV Space)": lambda x: clahe_hsv(x, clip_limit=2.5, is_rgb=False),
        "Retinex (MSRCR)": lambda x: msrcr(x),
        "Composite Classical": lambda x: classical_pipe.process(x, is_rgb=False),
    }

    if cnn_model is not None:
        try:
            import torch
            def run_cnn(img_bgr):
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(hybrid_enhancer.device)
                with torch.no_grad():
                    out_t = cnn_model(t)
                out_np = (out_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

            methods["Standalone ResCNN"] = run_cnn
            methods["Hybrid (Pre-Denoise + CNN + Guided)"] = lambda x: hybrid_enhancer.enhance(x, is_rgb=False)
        except Exception:
            pass

    evaluator = BenchmarkSuite(use_lpips=False)
    summary_records = []

    for name, func in methods.items():
        psnr_vals, ssim_vals, entropy_vals, time_vals = [], [], [], []

        for low_p, high_p in zip(eval_low, eval_high):
            low = cv2.imread(low_p)
            high = cv2.imread(high_p)

            t0 = time.perf_counter()
            enhanced = func(low)
            dt = (time.perf_counter() - t0) * 1000.0

            metrics = evaluator.evaluate_pair(enhanced, high)
            psnr_vals.append(metrics["psnr"])
            ssim_vals.append(metrics["ssim"])
            entropy_vals.append(metrics["entropy"])
            time_vals.append(dt)

        summary_records.append({
            "Method": name,
            "PSNR (dB)": float(np.mean(psnr_vals)),
            "SSIM": float(np.mean(ssim_vals)),
            "Entropy": float(np.mean(entropy_vals)),
            "Latency (ms)": float(np.mean(time_vals))
        })

    df = pd.DataFrame(summary_records)
    csv_path = os.path.join(output_dir, "ablation_benchmark_results.csv")
    df.to_csv(csv_path, index=False)

    try:
        md_table = df.to_markdown(index=False)
    except Exception:
        md_table = df.to_string(index=False)

    print("\n" + "=" * 70)
    print("LOW-LIGHT ENHANCEMENT ABLATION BENCHMARK RESULTS")
    print("=" * 70)
    print(md_table)
    print("=" * 70)
    print(f"[SUCCESS] Benchmark report saved to '{csv_path}'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/synthetic_data")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmark")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pth")
    args = parser.parse_args()
    run_benchmark(args.data_dir, args.output_dir, args.ckpt)
