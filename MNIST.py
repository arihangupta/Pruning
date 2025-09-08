import medmnist
from medmnist import DermaMNIST, PathMNIST, OCTMNIST, BloodMNIST, TissueMNIST
import numpy as np
import os
import shutil

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
# Helper: trim via memory mapping
# -------------------------
def maybe_trim_dataset(name: str, npz_path: str):
    """
    If the dataset is too large, randomly select TRIM_SPLITS without loading all data.
    Always copy original to ROOT_RAW. Save trimmed (or unchanged) version to ROOT_TRIM.
    """
    # Always copy original to raw folder
    raw_copy = os.path.join(ROOT_RAW, os.path.basename(npz_path))
    if not os.path.exists(raw_copy):
        shutil.copy(npz_path, raw_copy)

    print(f"Processing {name} at {npz_path}")
    data = np.load(npz_path, mmap_mode="r")

    n_train = data['train_images'].shape[0]
    n_val   = data['val_images'].shape[0]
    n_test  = data['test_images'].shape[0]
    total = n_train + n_val + n_test
    print(f"Original sizes → Train: {n_train}, Val: {n_val}, Test: {n_test} (Total={total})")

    if total <= MAX_TOTAL:
        # Dataset under limit, just copy
        final_path = os.path.join(ROOT_TRIM, os.path.basename(npz_path))
        if not os.path.exists(final_path):
            shutil.copy(npz_path, final_path)
        return final_path

    # Dataset too large, select random indices
    print(f"Trimming dataset to {MAX_TOTAL} samples ({TRIM_SPLITS['train']}/{TRIM_SPLITS['val']}/{TRIM_SPLITS['test']})")
    rng = np.random.default_rng(RNG_SEED)

    def select_subset(arr, keep):
        keep = min(keep, arr.shape[0])
        idx = rng.choice(arr.shape[0], size=keep, replace=False)
        return arr[idx]

    train_images_trimmed  = select_subset(data['train_images'], TRIM_SPLITS['train'])
    train_labels_trimmed  = select_subset(data['train_labels'], TRIM_SPLITS['train'])
    val_images_trimmed    = select_subset(data['val_images'], TRIM_SPLITS['val'])
    val_labels_trimmed    = select_subset(data['val_labels'], TRIM_SPLITS['val'])
    test_images_trimmed   = select_subset(data['test_images'], TRIM_SPLITS['test'])
    test_labels_trimmed   = select_subset(data['test_labels'], TRIM_SPLITS['test'])

    final_path = os.path.join(ROOT_TRIM, os.path.basename(npz_path))
    np.savez_compressed(
        final_path,
        train_images=train_images_trimmed,
        train_labels=train_labels_trimmed,
        val_images=val_images_trimmed,
        val_labels=val_labels_trimmed,
        test_images=test_images_trimmed,
        test_labels=test_labels_trimmed
    )
    print(f"Trimmed dataset saved to {final_path}")
    return final_path

# -------------------------
# Main loop
# -------------------------
for name, cls in datasets.items():
    print(f"\nDownloading {name} (size={IMG_SIZE})...")
    # Trigger medmnist download (ensure npz exists in ROOT_RAW)
    for split in ["train", "val", "test"]:
        cls(split=split, download=True, size=IMG_SIZE, root=ROOT_RAW)

    npz_path = os.path.join(ROOT_RAW, f"{name.lower()}_{IMG_SIZE}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Expected {npz_path} not found!")

    # Trim if necessary
    final_path = maybe_trim_dataset(name, npz_path)

    # Quick verification
    data = np.load(final_path, mmap_mode='r')
    img, label = data['train_images'][0], data['train_labels'][0]
    print(f"Sample image shape: {img.shape}, Label: {label}")

print("\nAll datasets downloaded, saved (raw + trimmed), and verified.")
