import numpy as np
import os

ROOT = "/home/arihangupta/Pruning/dinov2/Pruning/datasets"

datasets = [
    "bloodmnist_224.npz",
    "dermamnist_224.npz",
    "octmnist_224.npz",
    "pathmnist_224.npz",
    "tissuemnist_224.npz"
]
splits = ["train", "val", "test"]

for ds_file in datasets:
    ds_path = os.path.join(ROOT, ds_file)
    print(f"\nDataset: {ds_file}")
    with np.load(ds_path, mmap_mode='r') as data:
        for split in splits:
            images = data[f"{split}_images"]
            labels = data[f"{split}_labels"]
            print(f"  {split}: images={images.shape[0]}, labels={labels.shape[0]}")
