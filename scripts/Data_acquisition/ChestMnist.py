#!/usr/bin/env python3
"""
Download ChestMNIST at 224x224 and save the FULL dataset (no subsampling, no balancing)
as .npy files in datasets_npy/chestmnist_224/.

ChestMNIST is a MULTI-LABEL dataset (14 binary labels per image).
Training scripts use BCEWithLogitsLoss + sigmoid, not CrossEntropyLoss + softmax.

Memory-efficient: processes one split at a time to avoid loading the entire
dataset (~5GB+) into RAM simultaneously.
"""

import medmnist
from medmnist import ChestMNIST
import numpy as np
import os
import gc

# -------------------------
# Config
# -------------------------
IMG_SIZE = 224
ROOT_RAW = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_raw"
ROOT_NPY = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_npy"

os.makedirs(ROOT_RAW, exist_ok=True)
os.makedirs(ROOT_NPY, exist_ok=True)

npy_dir = os.path.join(ROOT_NPY, f"chestmnist_{IMG_SIZE}")
os.makedirs(npy_dir, exist_ok=True)

# -------------------------
# Download and save one split at a time to keep memory low
# -------------------------
print("Downloading and saving ChestMNIST (size=224)...")

for split in ["train", "val", "test"]:
    print(f"\n  Processing {split} split...")
    ds = ChestMNIST(split=split, download=True, size=IMG_SIZE, root=ROOT_RAW)

    print(f"    images={ds.imgs.shape}, labels={ds.labels.shape}")

    np.save(os.path.join(npy_dir, f"{split}_images.npy"), ds.imgs)
    np.save(os.path.join(npy_dir, f"{split}_labels.npy"), ds.labels)
    print(f"    Saved to {npy_dir}")

    # Free memory before loading next split
    del ds
    gc.collect()

# -------------------------
# Verify using memory-mapped reads (no RAM impact)
# -------------------------
print("\nVerification (memory-mapped):")
for split in ["train", "val", "test"]:
    imgs = np.load(os.path.join(npy_dir, f"{split}_images.npy"), mmap_mode="r")
    lbls = np.load(os.path.join(npy_dir, f"{split}_labels.npy"), mmap_mode="r")
    print(f"  {split}: images={imgs.shape}, labels={lbls.shape}")

sample_lbl = np.load(os.path.join(npy_dir, "train_labels.npy"), mmap_mode="r")
print(f"\n  Num labels (multi-label): {sample_lbl.shape[1]}")
print(f"  Label dtype: {sample_lbl.dtype}")
print(f"  Sample label: {sample_lbl[0]}")

print("\nChestMNIST ready for training.")
