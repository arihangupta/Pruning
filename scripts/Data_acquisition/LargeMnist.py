#!/usr/bin/env python3
"""
Memory-efficient script to trim large MedMNIST datasets (PathMNIST, OCTMNIST, TissueMNIST)
to 20,000 images. Strategy:
1. Download compressed .npz (MedMNIST default).
2. If dataset too large, convert once to uncompressed .npy arrays (on disk).
3. Subsample directly from .npy arrays with np.memmap, save trimmed .npz.
"""

import medmnist
from medmnist import PathMNIST, OCTMNIST, TissueMNIST
import numpy as np
import os
import shutil
from tqdm import tqdm
import gc
import psutil

# -------------------------
# Config
# -------------------------
IMG_SIZE = 224
ROOT_RAW = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_raw"
ROOT_NPY = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_npy"
ROOT_TRIM = "/home/arihangupta/Pruning/dinov2/Pruning/datasets"
os.makedirs(ROOT_RAW, exist_ok=True)
os.makedirs(ROOT_NPY, exist_ok=True)
os.makedirs(ROOT_TRIM, exist_ok=True)

MAX_TOTAL = 20000
TRIM_SPLITS = {"train": 14000, "val": 2000, "test": 4000}
RNG_SEED = 42
CHUNK_SIZE = 100
USE_COMPRESSION = True  # True: smaller final files, False: faster saving

# -------------------------
# Dataset registry
# -------------------------
datasets = {
    "PathMNIST": PathMNIST,
    "OCTMNIST": OCTMNIST,
    "TissueMNIST": TissueMNIST
}

# -------------------------
# Helper functions
# -------------------------
def get_memory_usage():
    return psutil.Process().memory_info().rss / 1024 / 1024

def efficient_random_indices(n_total, n_keep, rng):
    n_keep = min(n_keep, n_total)
    return rng.choice(n_total, size=n_keep, replace=False).astype(np.int64)

def convert_npz_to_npy(npz_path, out_dir):
    """Convert compressed .npz arrays to separate .npy files once."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"   Converting {npz_path} → {out_dir}")
    with np.load(npz_path) as data:
        for key in data.files:
            out_path = os.path.join(out_dir, f"{key}.npy")
            if not os.path.exists(out_path):
                np.save(out_path, data[key])
    return out_dir

def subsample_split_npy(out_dir, split, n_keep, rng):
    """Subsample split directly from .npy arrays with mmap."""
    img_path = os.path.join(out_dir, f"{split}_images.npy")
    lbl_path = os.path.join(out_dir, f"{split}_labels.npy")

    imgs = np.load(img_path, mmap_mode="r")
    lbls = np.load(lbl_path, mmap_mode="r")

    n_keep = min(n_keep, len(imgs))
    indices = efficient_random_indices(len(imgs), n_keep, rng)

    out_imgs = np.empty((n_keep, *imgs.shape[1:]), dtype=imgs.dtype)
    out_lbls = np.empty((n_keep, *lbls.shape[1:]), dtype=lbls.dtype)

    for i in tqdm(range(0, n_keep, CHUNK_SIZE), desc=f"Subsampling {split}", unit="chunk"):
        j = min(i + CHUNK_SIZE, n_keep)
        out_imgs[i:j] = imgs[indices[i:j]]
        out_lbls[i:j] = lbls[indices[i:j]]
        gc.collect()

    return out_imgs, out_lbls

def maybe_trim_dataset(name: str, npz_path: str):
    """Ensure dataset trimmed to <= MAX_TOTAL."""
    raw_copy = os.path.join(ROOT_RAW, os.path.basename(npz_path))
    if not os.path.exists(raw_copy):
        shutil.copy(npz_path, raw_copy)

    with np.load(npz_path) as data:
        total = len(data["train_images"]) + len(data["val_images"]) + len(data["test_images"])
    print(f"   Original total samples: {total}")

    final_path = os.path.join(ROOT_TRIM, os.path.basename(npz_path))
    if total <= MAX_TOTAL:
        print("   Dataset under limit, copying directly.")
        if not os.path.exists(final_path):
            shutil.copy(npz_path, final_path)
        return final_path

    # Convert to uncompressed .npy if not already done
    npy_dir = os.path.join(ROOT_NPY, os.path.splitext(os.path.basename(npz_path))[0])
    convert_npz_to_npy(npz_path, npy_dir)

    rng = np.random.default_rng(RNG_SEED)
    print("   Trimming splits...")
    train_imgs, train_lbls = subsample_split_npy(npy_dir, "train", TRIM_SPLITS["train"], rng)
    val_imgs, val_lbls = subsample_split_npy(npy_dir, "val", TRIM_SPLITS["val"], rng)
    test_imgs, test_lbls = subsample_split_npy(npy_dir, "test", TRIM_SPLITS["test"], rng)

    save_func = np.savez_compressed if USE_COMPRESSION else np.savez
    save_func(
        final_path,
        train_images=train_imgs, train_labels=train_lbls,
        val_images=val_imgs, val_labels=val_lbls,
        test_images=test_imgs, test_labels=test_lbls
    )
    print(f"   Trimmed dataset saved to {final_path}")
    return final_path

# -------------------------
# Main
# -------------------------
print(f"Starting memory usage: {get_memory_usage():.1f} MB")

for name, cls in datasets.items():
    print(f"\nDownloading {name} (size={IMG_SIZE})")
    for split in ["train", "val", "test"]:
        cls(split=split, download=True, size=IMG_SIZE, root=ROOT_RAW)

    npz_path = os.path.join(ROOT_RAW, f"{name.lower()}_{IMG_SIZE}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Expected {npz_path} not found!")

    final_path = maybe_trim_dataset(name, npz_path)
    print(f"Memory after processing {name}: {get_memory_usage():.1f} MB")

    # Verify sample
    with np.load(final_path) as data:
        img, label = data["train_images"][0], data["train_labels"][0]
    print(f"   Sample shape: {img.shape}, Label: {label}")

print(f"\nAll datasets processed. Final memory usage: {get_memory_usage():.1f} MB")
