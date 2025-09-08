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

def maybe_trim_dataset(name: str, npz_path: str):
    """Check size of dataset and trim to 20k if needed.
       Always copy original to ROOT_RAW, and save final usable dataset to ROOT_TRIM.
    """
    # Always copy original to raw folder
    raw_copy = os.path.join(ROOT_RAW, os.path.basename(npz_path))
    if not os.path.exists(raw_copy):
        shutil.copy(npz_path, raw_copy)

    print(f"📂 Processing {name} at {npz_path}")
    data = np.load(npz_path, mmap_mode="r")

    train_images, train_labels = data["train_images"], data["train_labels"]
    val_images, val_labels     = data["val_images"], data["val_labels"]
    test_images, test_labels   = data["test_images"], data["test_labels"]

    total = len(train_images) + len(val_images) + len(test_images)
    print(f"   Original sizes → Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)} (Total={total})")

    if total <= MAX_TOTAL:
        print(" Dataset under limit, copying to trimmed folder.")
        final_path = os.path.join(ROOT_TRIM, os.path.basename(npz_path))
        if not os.path.exists(final_path):
            shutil.copy(npz_path, final_path)
        return final_path

    print("Trimming dataset to 20,000 samples (14k/2k/4k)...")

    rng = np.random.default_rng(42)

    def subsample(images, labels, keep):
        idx = rng.choice(len(images), size=min(keep, len(images)), replace=False)
        return images[idx], labels[idx]

    train_images, train_labels = subsample(train_images, train_labels, TRIM_SPLITS["train"])
    val_images, val_labels     = subsample(val_images, val_labels, TRIM_SPLITS["val"])
    test_images, test_labels   = subsample(test_images, test_labels, TRIM_SPLITS["test"])

    final_path = os.path.join(ROOT_TRIM, os.path.basename(npz_path))
    np.savez_compressed(
        final_path,
        train_images=train_images, train_labels=train_labels,
        val_images=val_images, val_labels=val_labels,
        test_images=test_images, test_labels=test_labels
    )
    print(f"Trimmed dataset saved to {final_path}")
    return final_path

# -------------------------
# Main loop
# -------------------------
for name, cls in datasets.items():
    print(f"Downloading {name} (size={IMG_SIZE})...")
    # Trigger medmnist download (this ensures file exists at medmnist’s default path)
    for split in ["train", "val", "test"]:
        cls(split=split, download=True, size=IMG_SIZE, root=ROOT_RAW)

    npz_path = os.path.join(ROOT_RAW, f"{name.lower()}_{IMG_SIZE}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Expected {npz_path} not found!")

    # Trim or copy to ROOT_TRIM
    final_path = maybe_trim_dataset(name, npz_path)

    # Quick verification
    data = np.load(final_path)
    img, label = data["train_images"][0], data["train_labels"][0]
    print(f"   🔎 {name} sample image shape: {img.shape}, Label: {label}")

print("\n🎉 All datasets downloaded, saved (raw + trimmed), and verified.")
