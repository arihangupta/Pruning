#!/usr/bin/env python3
"""
Memory-efficient script to create class-balanced, trimmed MedMNIST datasets.
Uses np.memmap streaming to avoid exhausting RAM.
"""

import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import gc
import psutil

# -------------------------
# Config
# -------------------------
IMG_SIZE = 224
ROOT_NPY = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_npy"
ROOT_BAL = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
os.makedirs(ROOT_BAL, exist_ok=True)

TRIM_SPLITS = {"train": 14000, "val": 2000, "test": 4000}
RNG_SEED = 42
CHUNK_SIZE = 100
USE_COMPRESSION = True

datasets = ["pathmnist_224", "octmnist_224", "tissuemnist_224"]

rng = np.random.default_rng(RNG_SEED)

# -------------------------
# Helpers
# -------------------------
def get_memory_usage():
    return psutil.Process().memory_info().rss / 1024 / 1024

def balanced_indices(lbls, n_keep, rng):
    """Return indices balanced across classes, allowing compensation if some classes are small."""
    # flatten labels in case they are (N,1)
    lbls = lbls.reshape(-1)
    classes = np.unique(lbls)
    n_classes = len(classes)

    # initial quota per class
    base_quota = n_keep // n_classes
    remainder = n_keep % n_classes

    per_class_counts = {c: base_quota for c in classes}
    # distribute remainder one by one
    for i, c in enumerate(classes[:remainder]):
        per_class_counts[c] += 1

    # adjust if some classes are too small
    deficit = 0
    for c in classes:
        n_available = np.sum(lbls == c)
        if n_available < per_class_counts[c]:
            deficit += per_class_counts[c] - n_available
            per_class_counts[c] = n_available

    # redistribute deficit evenly to other classes with capacity
    while deficit > 0:
        for c in classes:
            n_available = np.sum(lbls == c)
            if per_class_counts[c] < n_available:
                per_class_counts[c] += 1
                deficit -= 1
                if deficit == 0:
                    break

    # now sample indices
    all_indices = []
    for c in classes:
        cls_indices = np.where(lbls == c)[0]
        chosen = rng.choice(cls_indices, size=per_class_counts[c], replace=False)
        all_indices.extend(chosen)

    rng.shuffle(all_indices)
    return np.array(all_indices, dtype=np.int64)


def subsample_split_npy_balanced(npy_dir, split, n_keep, rng):
    """Subsample split directly from .npy arrays with mmap, class-balanced."""
    img_path = os.path.join(npy_dir, f"{split}_images.npy")
    lbl_path = os.path.join(npy_dir, f"{split}_labels.npy")

    imgs = np.load(img_path, mmap_mode="r")
    lbls = np.load(lbl_path, mmap_mode="r")

    n_keep = min(n_keep, len(imgs))
    idxs = balanced_indices(lbls, n_keep, rng)

    out_imgs = np.empty((n_keep, *imgs.shape[1:]), dtype=imgs.dtype)
    out_lbls = np.empty((n_keep, *lbls.shape[1:]), dtype=lbls.dtype)

    for i in tqdm(range(0, n_keep, CHUNK_SIZE), desc=f"{split} balanced", unit="chunk"):
        j = min(i + CHUNK_SIZE, n_keep)
        out_imgs[i:j] = imgs[idxs[i:j]]
        out_lbls[i:j] = lbls[idxs[i:j]]
        gc.collect()

    return out_imgs, out_lbls

# -------------------------
# Main
# -------------------------
print(f"Starting memory usage: {get_memory_usage():.1f} MB")

for name in datasets:
    print(f"\nProcessing {name}...")
    dataset_dir = os.path.join(ROOT_NPY, name)
    out_path = os.path.join(ROOT_BAL, f"{name}.npz")

    if os.path.exists(out_path):
        print(f"   Skipping, already exists: {out_path}")
        continue

    rng = np.random.default_rng(RNG_SEED)

    train_imgs, train_lbls = subsample_split_npy_balanced(dataset_dir, "train", TRIM_SPLITS["train"], rng)
    val_imgs, val_lbls = subsample_split_npy_balanced(dataset_dir, "val", TRIM_SPLITS["val"], rng)
    test_imgs, test_lbls = subsample_split_npy_balanced(dataset_dir, "test", TRIM_SPLITS["test"], rng)

    save_func = np.savez_compressed if USE_COMPRESSION else np.savez
    save_func(
        out_path,
        train_images=train_imgs, train_labels=train_lbls,
        val_images=val_imgs, val_labels=val_lbls,
        test_images=test_imgs, test_labels=test_lbls
    )
    print(f"   Balanced dataset saved to {out_path}")
    print(f"   Memory after: {get_memory_usage():.1f} MB")

print("\nAll balanced datasets created.")
