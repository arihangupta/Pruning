#!/usr/bin/env python3
"""
Download and save DermaMNIST and BloodMNIST at 224x224 resolution
into the standardized numpy format in /home/arihangupta/Pruning/dinov2/Pruning/datasets_npy.
No trimming is performed.
"""

import medmnist
from medmnist import DermaMNIST, BloodMNIST
import numpy as np
import os

# -------------------------
# Config
# -------------------------
IMG_SIZE = 224
ROOT_OUT = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_npy"
os.makedirs(ROOT_OUT, exist_ok=True)

# -------------------------
# Dataset registry
# -------------------------
datasets = {
    "dermamnist": DermaMNIST,
    "bloodmnist": BloodMNIST
}

# -------------------------
# Save helper
# -------------------------
def save_dataset(name: str, cls):
    print(f"\nDownloading {name} (size={IMG_SIZE})...")

    # Trigger medmnist download and load each split
    splits = {}
    for split in ["train", "val", "test"]:
        ds = cls(split=split, download=True, size=IMG_SIZE, root=ROOT_OUT)
        splits[split] = (ds.imgs, ds.labels)

    # Save to standardized folder
    out_dir = os.path.join(ROOT_OUT, f"{name}_{IMG_SIZE}")
    os.makedirs(out_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        imgs, labels = splits[split]
        np.save(os.path.join(out_dir, f"{split}_images.npy"), imgs)
        np.save(os.path.join(out_dir, f"{split}_labels.npy"), labels)

    print(f"Saved {name} dataset → {out_dir}")
    # Quick check
    print(f"  Train shape: {splits['train'][0].shape}, Labels: {splits['train'][1].shape}")

# -------------------------
# Main loop
# -------------------------
for name, cls in datasets.items():
    save_dataset(name, cls)

print("\nAll datasets downloaded and saved in standardized format.")
