#!/usr/bin/env python3
import os
import numpy as np

# Path to your balanced datasets
ROOT_BAL = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"

def print_split_counts(npz_path):
    name = os.path.basename(npz_path)
    print(f"\n=== {name} ===")
    with np.load(npz_path) as data:
        for split in ["train_labels", "val_labels", "test_labels"]:
            if split not in data:
                continue
            lbls = data[split].reshape(-1)  # flatten in case shape is (N,1)
            classes, counts = np.unique(lbls, return_counts=True)
            print(f"{split.replace('_labels',''):>5}: ", end="")
            print(", ".join(f"class {int(c)}: {cnt}" for c, cnt in zip(classes, counts)))

def main():
    for fname in sorted(os.listdir(ROOT_BAL)):
        if fname.endswith(".npz"):
            print_split_counts(os.path.join(ROOT_BAL, fname))

if __name__ == "__main__":
    main()
