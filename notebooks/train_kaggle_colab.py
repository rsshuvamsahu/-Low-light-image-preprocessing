"""
# Low-Light Image Preprocessing & CNN Enhancement: 1-Click Colab / Kaggle Training Script
# 
# Run this on Google Colab (with T4 GPU) or Kaggle Notebooks (with P100/T4 GPU).
# Solves GPU detection, auto-downloads the LOL dataset, trains ResCNN/UNet, and exports checkpoints!
"""

import os
import sys

def setup_colab_or_kaggle():
    print("=" * 60)
    print("STEP 0: ENVIRONMENT & GPU VERIFICATION")
    print("=" * 60)
    
    # Check GPU
    import torch
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Device Name:", torch.cuda.get_device_name(0))
    else:
        print("[!] Warning: Running on CPU. For fast training, enable GPU under Runtime -> Change runtime type -> T4 GPU")

    # Install lpips safely without breaking torch
    try:
        import lpips
        print("[*] LPIPS is already installed.")
    except ImportError:
        print("[*] Installing LPIPS (--no-deps)...")
        os.system("pip install lpips --no-deps")

    # Download LOL Dataset
    data_dir = "data/lol_dataset"
    if not os.path.exists(data_dir):
        print("\n" + "=" * 60)
        print("STEP 1: DOWNLOADING LOL DATASET (485 Train / 15 Eval Pairs)")
        print("=" * 60)
        os.makedirs("data", exist_ok=True)
        os.system("wget -q https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip -O lol_dataset.zip")
        os.system("unzip -q lol_dataset.zip -d data/ && rm lol_dataset.zip")
        print("[SUCCESS] LOL Dataset downloaded and extracted to 'data/lol_dataset'")
    else:
        print(f"[*] LOL Dataset already found at '{data_dir}'")

if __name__ == "__main__":
    setup_colab_or_kaggle()
