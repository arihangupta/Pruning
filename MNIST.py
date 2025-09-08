import medmnist
from medmnist import DermaMNIST, PathMNIST, OCTMNIST, BloodMNIST, TissueMNIST
import numpy as np
import os
import shutil
from tqdm import tqdm

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

# -------------------------
# Dataset registry
# -------------------------
datasets = {
    "DermaMNIST": DermaMNIST,
    "PathMNIST": PathMNIST,
    "OCTMNIST": OCTMNIST,
    "BloodMNIST": BloodMNIST,
    "TissueMNIST": TissueMNIST
}

# -------------------------
# Helper functions
# -------------------------
def random_indices(n_total, n_keep, rng):
    """Select n_keep unique random indices from range(n_total) without large arrays."""
    selected = set()
    while len(selected) < min(n_keep, n_total):
        selected.add(rng.integers(0, n_total))
    return list(selected)

def trim_split(images, labels, n_keep, desc=""):
    """Trim a single split using random indices, with progress bar."""
    idxs = random_indices(len(images), n_keep, np.random.default_rng(RNG_SEED))
    trimmed_images = np.empty((len(idxs), *images.shape[1:]), dtype=images.dtype)
    trimmed_labels = np.empty((len(idxs),), dtype=labels.dtype)
    for i, idx in enumerate(tqdm(idxs, desc=f"Trimming {desc}", unit="img")):
        trimmed_images[i] = images[idx]
        trimmed_labels[i] = labels[idx]
    return trimmed_images, trimmed_labels

def maybe_trim_dataset(name: str, npz_path: str):
    """Trim dataset to MAX_TOTAL if needed without fully loading it."""
    # Always copy original to raw folder
    raw_copy = os.path.join(ROOT_RAW, os.path.basename(npz_path))
    if not os.path.exists(raw_copy):
        shutil.copy(npz_path, raw_copy)

    print(f"Processing {name} at {npz_path} ...")
    data = np.load(npz_path, mmap_mode="r")  # memory-map large arrays

    train_images, train_labels = data["train_images"], data["train_labels"]
    val_images, val_labels     = data["val_images"], data["val_labels"]
    test_images, test_labels   = data["test_images"], data["test_labels"]

    total = len(train_images) + len(val_images) + len(test_images)
    print(f"   Original sizes → Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)} (Total={total})")

    final_path = os.path.join(ROOT_TRIM, os.path.basename(npz_path))

    if total <= MAX_TOTAL:
        print("   Dataset under limit, copying to trimmed folder.")
        if not os.path.exists(final_path):
            shutil.copy(npz_path, final_path)
        return final_path

    print(f"   Dataset too large, trimming to {MAX_TOTAL} samples...")

    train_images_trim, train_labels_trim = trim_split(train_images, train_labels, TRIM_SPLITS["train"], "train")
    val_images_trim, val_labels_trim     = trim_split(val_images, val_labels, TRIM_SPLITS["val"], "val")
    test_images_trim, test_labels_trim   = trim_split(test_images, test_labels, TRIM_SPLITS["test"], "test")

    np.savez_compressed(
        final_path,
        train_images=train_images_trim, train_labels=train_labels_trim,
        val_images=val_images_trim, val_labels=val_labels_trim,
        test_images=test_images_trim, test_labels=test_labels_trim
    )
    print(f"   Trimmed dataset saved to {final_path}")
    return final_path

# -------------------------
# Main loop
# -------------------------
for name, cls in datasets.items():
    print(f"\nDownloading {name} (size={IMG_SIZE}) ...")
    # Trigger medmnist download (ensures file exists)
    for split in ["train", "val", "test"]:
        cls(split=split, download=True, size=IMG_SIZE, root=ROOT_RAW)

    npz_path = os.path.join(ROOT_RAW, f"{name.lower()}_{IMG_SIZE}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Expected {npz_path} not found!")

    # Trim or copy to ROOT_TRIM
    final_path = maybe_trim_dataset(name, npz_path)

    # Quick verification
    data = np.load(final_path, mmap_mode="r")
    img, label = data["train_images"][0], data["train_labels"][0]
    print(f"   Sample image shape: {img.shape}, Label: {label}")

print("\nAll datasets downloaded, trimmed if needed, and verified.")
