"""
Training Script for Low-Light Image Enhancement CNNs.

Features:
- Automatic Device Detection (CUDA / CPU)
- Mixed Precision Training (torch.amp) with safe CPU fallback
- Validation Tracking (PSNR & SSIM monitoring)
- Checkpoint Saving (best.pth, last.pth)
- CSV Loss and Metric Logging
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.models import build_model
from core.losses import CompositeEnhancementLoss
from core.dataset import LOLDataset, generate_synthetic_dataset
from core.metrics import calculate_psnr, calculate_ssim


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    l1_sum, ssim_sum, grad_sum = 0.0, 0.0, 0.0

    for low, high in loader:
        low, high = low.to(device), high.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.autocast(device_type="cuda", enabled=True):
                pred = model(low)
                losses = criterion(pred, high)
                loss = losses["total"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(low)
            losses = criterion(pred, high)
            loss = losses["total"]
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        l1_sum += losses["l1"].item()
        ssim_sum += losses["ssim"].item()
        grad_sum += losses["grad"].item()

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "l1": l1_sum / n,
        "ssim_loss": ssim_sum / n,
        "grad_loss": grad_sum / n
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    psnr_list, ssim_list = [], []

    for low, high in loader:
        low, high = low.to(device), high.to(device)
        pred = model(low)

        pred_np = (pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        high_np = (high.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        psnr_list.append(calculate_psnr(pred_np, high_np))
        ssim_list.append(calculate_ssim(pred_np, high_np))

    n = max(len(psnr_list), 1)
    return {
        "val_psnr": sum(psnr_list) / n,
        "val_ssim": sum(ssim_list) / n
    }


def main():
    parser = argparse.ArgumentParser(description="Train Low-Light Enhancement CNN")
    parser.add_argument("--data_dir", type=str, default="data/lol_dataset", help="Root dataset directory")
    parser.add_argument("--model", type=str, default="res_cnn", choices=["res_cnn", "unet", "zero_dce"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--crop_size", type=int, default=128, help="Crop size for training patches")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--use_synthetic", action="store_true", help="Generate and use synthetic data if no LOL dataset")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    print("=" * 65)
    print(f"[*] Starting Training: Model={args.model} | Device={device.upper()} | AMP={use_amp}")
    print("=" * 65)

    # 1. Dataset Setup
    if not os.path.exists(args.data_dir) or args.use_synthetic:
        synth_dir = "data/synthetic_data"
        print(f"[*] Dataset not found at '{args.data_dir}'. Generating synthetic dataset at '{synth_dir}'...")
        generate_synthetic_dataset(synth_dir, num_samples=30, img_size=(128, 128))
        args.data_dir = synth_dir

    train_dataset = LOLDataset(args.data_dir, split="train", crop_size=args.crop_size, is_training=True)
    val_dataset = LOLDataset(args.data_dir, split="eval15", crop_size=None, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"[*] Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # 2. Model, Optimizer, Loss, Scaler
    model = build_model(args.model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CompositeEnhancementLoss(w_l1=1.0, w_ssim=0.5, w_grad=0.2).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_psnr = 0.0
    history = []

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_stats = evaluate(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_psnr": val_stats["val_psnr"],
            "val_ssim": val_stats["val_ssim"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": dt
        }
        history.append(record)

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Loss: {train_stats['loss']:.4f} | Val PSNR: {val_stats['val_psnr']:.2f} dB | Val SSIM: {val_stats['val_ssim']:.4f} | Time: {dt:.1f}s")

        # Save best model
        if val_stats["val_psnr"] > best_psnr:
            best_psnr = val_stats["val_psnr"]
            best_path = os.path.join(args.ckpt_dir, "best.pth")
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "psnr": best_psnr}, best_path)

        # Save latest model
        last_path = os.path.join(args.ckpt_dir, "last.pth")
        torch.save({"state_dict": model.state_dict(), "epoch": epoch}, last_path)

    # Save training history to CSV
    log_df = pd.DataFrame(history)
    log_df.to_csv(os.path.join(args.ckpt_dir, "training_log.csv"), index=False)
    print("=" * 65)
    print(f"[SUCCESS] Training Completed! Best Validation PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved to '{args.ckpt_dir}/best.pth'")
    print("=" * 65)


if __name__ == "__main__":
    main()
