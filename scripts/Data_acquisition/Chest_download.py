#!/usr/bin/env python3
"""
Download ChestMNIST at 224x224 and save the FULL dataset (no subsampling, no balancing)
in the standardized format used by the ViT and CNN training pipelines.

NOTE: ChestMNIST is a MULTI-LABEL dataset (14 binary labels per image),
unlike the other MedMNIST datasets which are multi-class (single label).
Training scripts must use BCEWithLogitsLoss + sigmoid, not CrossEntropyLoss + softmax.
"""

import medmnist
from medmnist import ChestMNIST
import numpy as np
import os

# -------------------------
# Config
# -------------------------
IMG_SIZE = 224
ROOT_RAW = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_raw"
ROOT_NPY = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_npy"

os.makedirs(ROOT_RAW, exist_ok=True)
os.makedirs(ROOT_NPY, exist_ok=True)

# -------------------------
# Download and save
# -------------------------
print("Downloading ChestMNIST (size=224)...")

splits = {}
for split in ["train", "val", "test"]:
    ds = ChestMNIST(split=split, download=True, size=IMG_SIZE, root=ROOT_RAW)
    splits[split] = (ds.imgs, ds.labels)
    print(f"  {split}: images={ds.imgs.shape}, labels={ds.labels.shape}")

# -------------------------
# Save as .npy to datasets_npy (same format as other datasets)
# Labels are kept as (N, 14) — NOT flattened — since this is multi-label.
# -------------------------
npy_dir = os.path.join(ROOT_NPY, f"chestmnist_{IMG_SIZE}")
os.makedirs(npy_dir, exist_ok=True)

for split in ["train", "val", "test"]:
    imgs, labels = splits[split]
    np.save(os.path.join(npy_dir, f"{split}_images.npy"), imgs)
    np.save(os.path.join(npy_dir, f"{split}_labels.npy"), labels)

print(f"\nSaved .npy files to {npy_dir}")

# -------------------------
# Verify
# -------------------------
print("\nVerification:")
for split in ["train", "val", "test"]:
    imgs = np.load(os.path.join(npy_dir, f"{split}_images.npy"), mmap_mode="r")
    lbls = np.load(os.path.join(npy_dir, f"{split}_labels.npy"), mmap_mode="r")
    print(f"  {split}: images={imgs.shape}, labels={lbls.shape}")

print(f"\n  Num labels (multi-label): 14")
sample_lbl = np.load(os.path.join(npy_dir, "train_labels.npy"), mmap_mode="r")
print(f"  Label dtype: {sample_lbl.dtype}")
print(f"  Sample label: {sample_lbl[0]}")

print("\nChestMNIST ready for training.")
