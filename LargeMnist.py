#!/usr/bin/env python3
"""
Memory-efficient script to trim large MedMNIST datasets (PathMNIST, OCTMNIST, TissueMNIST)
to 20,000 images using np.memmap for input and preallocated arrays for output.
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
ROOT_TRIM = "/home/arihangupta/Pruning/dinov2/Pruning/datasets"
os.makedirs(ROOT_RAW, exist_ok=True)
os.makedirs(ROOT_TRIM, exist_ok=True)

MAX_TOTAL = 20000
TRIM_SPLITS = {"train": 14000, "val": 2000, "test": 4000}
RNG_SEED = 42
CHUNK_SIZE = 100  # Small chunks for input processing
USE_COMPRESSION = True  # True for np.savez_compressed (smaller files, slower), False for np.savez (faster, larger files)

# -------------------------
# Dataset registry
# -------------------------
datasets = {
    "PathMNIST": PathMNIST,
    "OCTMNIST": OCTMNIST,
    "TissueMNIST": TissueMNIST
}

# -------------------------
# Memory-efficient helper functions
# -------------------------
def efficient_random_indices(n_total, n_keep, rng):
    """Generate random indices efficiently."""
    n_keep = min(n_keep, n_total)
    return rng.choice(n_total, size=n_keep, replace=False).astype(np.int64)

def trim_split(input_images, input_labels, n_keep, desc=""):
    """Trim a split using memmap for input and preallocated array for output."""
    n_keep = min(n_keep, len(input_images))
    rng = np.random.default_rng(RNG_SEED)
    indices = efficient_random_indices(len(input_images), n_keep, rng)

    # Preallocate output arrays (manageable for 20,000 images total)
    img_shape = input_images.shape[1:]  # e.g., (224, 224, 3)
    label_shape = input_labels.shape[1:] if input_labels.ndim > 1 else ()
    output_images = np.empty((n_keep, *img_shape), dtype=input_images.dtype)
    output_labels = np.empty((n_keep, *label_shape), dtype=input_labels.dtype)

    # Process in chunks
    for i in tqdm(range(0, n_keep, CHUNK_SIZE), desc=f"Trimming {desc}", unit="chunk"):
        chunk_end = min(i + CHUNK_SIZE, n_keep)
        chunk_indices = indices[i:chunk_end]

        # Load and copy chunk
        output_images[i:chunk_end] = input_images[chunk_indices]
        output_labels[i:chunk_end] = input_labels[chunk_indices]

        # Clear references and force garbage collection
        gc.collect()

    return output_images, output_labels

def maybe_trim_dataset_efficient(name: str, npz_path: str):
    """Trim dataset to MAX_TOTAL with minimal RAM usage."""
    # Copy original to raw folder
    raw_copy = os.path.join(ROOT_RAW, os.path.basename(npz_path))
    if not os.path.exists(raw_copy):
        shutil.copy(npz_path, raw_copy)

    print(f"Processing {name} at {npz_path} ...")
    print(f"Memory before loading: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
    data = np.load(npz_path, mmap_mode="r")

    train_images, train_labels = data["train_images"], data["train_labels"]
    val_images, val_labels = data["val_images"], data["val_labels"]
    test_images, test_labels = data["test_images"], data["test_labels"]

    total = len(train_images) + len(val_images) + len(test_images)
    print(f"   Original sizes → Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)} (Total={total})")

    final_path = os.path.join(ROOT_TRIM, os.path.basename(npz_path))

    if total <= MAX_TOTAL:
        print("   Dataset under limit, copying to trimmed folder.")
        if not os.path.exists(final_path):
            shutil.copy(npz_path, final_path)
        return final_path

    print(f"   Trimming to {MAX_TOTAL} samples...")

    # Trim splits
    try:
        print(f"   Processing train split... Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        train_images_trim, train_labels_trim = trim_split(train_images, train_labels, TRIM_SPLITS["train"], "train")
        gc.collect()

        print(f"   Processing val split... Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        val_images_trim, val_labels_trim = trim_split(val_images, val_labels, TRIM_SPLITS["val"], "val")
        gc.collect()

        print(f"   Processing test split... Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        test_images_trim, test_labels_trim = trim_split(test_images, test_labels, TRIM_SPLITS["test"], "test")
        gc.collect()

        print(f"   Saving trimmed dataset... Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        save_func = np.savez_compressed if USE_COMPRESSION else np.savez
        save_func(
            final_path,
            train_images=train_images_trim, train_labels=train_labels_trim,
            val_images=val_images_trim, val_labels=val_labels_trim,
            test_images=test_images_trim, test_labels=test_labels_trim
        )

        print(f"   Trimmed dataset saved to {final_path}")

    finally:
        # Clean up
        del train_images_trim, train_labels_trim, val_images_trim, val_labels_trim, test_images_trim, test_labels_trim
        gc.collect()

    return final_path

def get_memory_usage():
    """Get current memory usage in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024

# -------------------------
# Main loop with memory monitoring
# -------------------------
print(f"Starting memory usage: {get_memory_usage():.1f} MB")

for name, cls in datasets.items():
    print(f"\nDownloading {name} (size={IMG_SIZE}) ...")
    print(f"Memory before download: {get_memory_usage():.1f} MB")

    # Trigger medmnist download
    for split in ["train", "val", "test"]:
        cls(split=split, download=True, size=IMG_SIZE, root=ROOT_RAW)

    npz_path = os.path.join(ROOT_RAW, f"{name.lower()}_{IMG_SIZE}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Expected {npz_path} not found!")

    final_path = maybe_trim_dataset_efficient(name, npz_path)
    print(f"Memory after processing {name}: {get_memory_usage():.1f} MB")

    # Quick verification
    data = np.load(final_path, mmap_mode="r")
    img, label = data["train_images"][0], data["train_labels"][0]
    print(f"   Sample image shape: {img.shape}, Label: {label}")
    del data
    gc.collect()

print(f"\nAll datasets processed. Final memory usage: {get_memory_usage():.1f} MB")