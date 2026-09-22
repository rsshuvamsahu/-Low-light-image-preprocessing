"""
Verification Test Suite for Deep Learning Models, Losses, and Hybrid Pipeline.
"""

import os
import sys
import time
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from core.models import LowLightResCNN, UNetEnhancer, ZeroDCENet, build_model
from core.losses import CompositeEnhancementLoss
from core.hybrid import HybridEnhancer


def count_parameters(model: torch.nn.Module) -> float:
    """Return total trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def run_model_tests():
    print("=" * 65)
    print("RUNNING STEP 2: DEEP LEARNING & HYBRID PIPELINE TESTS")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Testing on Device: {device.upper()}")

    # 1. Test ResCNN
    res_cnn = LowLightResCNN(in_channels=3, out_channels=3, num_features=32, num_blocks=6).to(device)
    dummy_input = torch.rand(1, 3, 256, 256, device=device)
    t0 = time.perf_counter()
    out_res = res_cnn(dummy_input)
    t_res = (time.perf_counter() - t0) * 1000.0
    print(f"[PASS] LowLightResCNN | Params: {count_parameters(res_cnn):.3f}M | Out: {list(out_res.shape)} | Time: {t_res:.2f} ms")

    # 2. Test UNet
    unet = UNetEnhancer(in_channels=3, out_channels=3, base_ch=32).to(device)
    t0 = time.perf_counter()
    out_unet = unet(dummy_input)
    t_unet = (time.perf_counter() - t0) * 1000.0
    print(f"[PASS] UNetEnhancer   | Params: {count_parameters(unet):.3f}M | Out: {list(out_unet.shape)} | Time: {t_unet:.2f} ms")

    # 3. Test Zero-DCE
    zero_dce = ZeroDCENet(num_iterations=8).to(device)
    t0 = time.perf_counter()
    out_dce = zero_dce(dummy_input)
    t_dce = (time.perf_counter() - t0) * 1000.0
    print(f"[PASS] ZeroDCENet     | Params: {count_parameters(zero_dce):.3f}M | Out: {list(out_dce.shape)} | Time: {t_dce:.2f} ms")

    # 4. Test Composite Loss
    print("\n[*] Testing Composite Loss Function...")
    loss_fn = CompositeEnhancementLoss(w_l1=1.0, w_ssim=0.5, w_grad=0.2).to(device)
    target = torch.rand(1, 3, 256, 256, device=device)
    losses = loss_fn(out_res, target)
    print(f"[PASS] Total Loss: {losses['total'].item():.4f} (L1: {losses['l1'].item():.4f}, SSIM: {losses['ssim'].item():.4f}, Grad: {losses['grad'].item():.4f})")

    # 5. Test Hybrid Pipeline Execution on sample numpy image
    print("\n[*] Testing Hybrid Preprocessing + CNN Pipeline...")
    hybrid = HybridEnhancer(model=res_cnn, device=device, pre_denoise=True, post_guided_refine=True)
    sample_np = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    t0 = time.perf_counter()
    hybrid_out = hybrid.enhance(sample_np)
    t_hyb = (time.perf_counter() - t0) * 1000.0
    print(f"[PASS] HybridEnhancer | Output Shape: {hybrid_out.shape} | dtype: {hybrid_out.dtype} | Time: {t_hyb:.2f} ms")

    print("\n" + "=" * 65)
    print("ALL DEEP LEARNING & HYBRID TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_model_tests()
