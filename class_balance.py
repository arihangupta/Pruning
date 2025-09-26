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
    """Return balanced subset of indices for lbls."""
    class_indices = defaultdict(list)
    for i, y in enumerate(lbls):
        class_indices[int(y)].append(i)

    classes = sorted(class_indices.keys())
    per_class = n_keep // len(classes)
    remainder = n_keep % len(classes)

    selected = []
    for ci, c in enumerate(classes):
        n_c = per_class + (1 if ci < remainder else 0)
        idxs = rng.choice(class_indices[c], size=min(n_c, len(class_indices[c])), replace=False)
        selected.extend(idxs)

    rng.shuffle(selected)
    return np.array(selected, dtype=np.int64)

def subsample_split_npy_balanced(dataset_dir, split, target_per_class, rng, chunk_size=100):
    imgs = np.load(os.path.join(dataset_dir, f"{split}_images.npy"), mmap_mode="r")
    lbls = np.load(os.path.join(dataset_dir, f"{split}_labels.npy"), mmap_mode="r")

    # collect indices per class without materializing all at once
    class_indices = defaultdict(list)
    for i, y in enumerate(lbls):
        class_indices[int(y)].append(i)

    # instead of forcing equality, allow smallest class to define cap
    min_class_size = min(len(idxs) for idxs in class_indices.values())
    n_samples = min(target_per_class, min_class_size)

    total_per_class = {cls: 0 for cls in class_indices}
    out_imgs = np.empty((n_samples * len(class_indices),) + imgs.shape[1:], dtype=imgs.dtype)
    out_lbls = np.empty((n_samples * len(class_indices),), dtype=lbls.dtype)

    pos = 0
    for cls, idxs in class_indices.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        chosen = idxs[:n_samples]

        # write in chunks to avoid blowing RAM
        for i in range(0, n_samples, chunk_size):
            j = min(i + chunk_size, n_samples)
            out_imgs[pos:pos+(j-i)] = imgs[chosen[i:j]]
            out_lbls[pos:pos+(j-i)] = lbls[chosen[i:j]]
            pos += (j - i)

        total_per_class[cls] = n_samples

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
