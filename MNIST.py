import medmnist
from medmnist import DermaMNIST, PathMNIST, OCTMNIST, BloodMNIST, TissueMNIST
import numpy as np
import os

# Choose the highest resolution available (224 for MedMNIST+)
IMG_SIZE = 224

# Use absolute path for root directory
ROOT_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets"
os.makedirs(ROOT_DIR, exist_ok=True)  # Create directory if it doesn't exist

# -------------------------
# Download datasets
# -------------------------
datasets = {
    "DermaMNIST": DermaMNIST,
    "PathMNIST": PathMNIST,
    "OCTMNIST": OCTMNIST,
    "BloodMNIST": BloodMNIST,
    "TissueMNIST": TissueMNIST
}

loaded_datasets = {}

for name, cls in datasets.items():
    print(f"\nDownloading {name} (size={IMG_SIZE})...")
    train_set = cls(split="train", download=True, size=IMG_SIZE, root=ROOT_DIR)
    val_set   = cls(split="val", download=True, size=IMG_SIZE, root=ROOT_DIR)
    test_set  = cls(split="test", download=True, size=IMG_SIZE, root=ROOT_DIR)
    loaded_datasets[name] = (train_set, val_set, test_set)

    print(f"{name} Train: {len(train_set)}")
    print(f"{name} Val:   {len(val_set)}")
    print(f"{name} Test:  {len(test_set)}")

    # Verify one sample
    img, label = train_set[0]
    img_array = np.array(img)
    print(f"{name} sample image shape:", img_array.shape, "Label:", label)

print("\n✅ All datasets downloaded and verified.")
