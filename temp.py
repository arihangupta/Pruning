import numpy as np
import os

ROOT = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_npy"

datasets = ["bloodmnist_224", "dermamnist_224", "octmnist_224", "pathmnist_224", "tissuemnist_224"]
splits = ["train", "val", "test"]

for ds in datasets:
    ds_path = os.path.join(ROOT, ds)
    print(f"\nDataset: {ds}")
    for split in splits:
        images_file = os.path.join(ds_path, f"{split}_images.npy")
        labels_file = os.path.join(ds_path, f"{split}_labels.npy")

        # Memory-map to avoid loading entire array
        images = np.load(images_file, mmap_mode='r')
        labels = np.load(labels_file, mmap_mode='r')

        print(f"  {split}: images={images.shape[0]}, labels={labels.shape[0]}")
