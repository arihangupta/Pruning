import medmnist
from medmnist import DermaMNIST, PathMNIST
import numpy as np
import os

# Choose the highest resolution available (224 for MedMNIST+)
IMG_SIZE = 224

# Use absolute path for root directory
ROOT_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets"
os.makedirs(ROOT_DIR, exist_ok=True)  # Create directory if it doesn't exist

# Download DermaMNIST (train/val/test) at 224x224
derma_train = DermaMNIST(split="train", download=True, size=IMG_SIZE, root=ROOT_DIR)
derma_val = DermaMNIST(split="val", download=True, size=IMG_SIZE, root=ROOT_DIR)
derma_test = DermaMNIST(split="test", download=True, size=IMG_SIZE, root=ROOT_DIR)

# Download PathMNIST (train/val/test) at 224x224
path_train = PathMNIST(split="train", download=True, size=IMG_SIZE, root=ROOT_DIR)
path_val = PathMNIST(split="val", download=True, size=IMG_SIZE, root=ROOT_DIR)
path_test = PathMNIST(split="test", download=True, size=IMG_SIZE, root=ROOT_DIR)

# Print dataset sizes
print("DermaMNIST Train:", len(derma_train))
print("DermaMNIST Val:", len(derma_val))
print("DermaMNIST Test:", len(derma_test))
print("PathMNIST Train:", len(path_train))
print("PathMNIST Val:", len(path_val))
print("PathMNIST Test:", len(path_test))

# Verify a sample image from each dataset
derma_img, derma_label = derma_train[0]
path_img, path_label = path_train[0]

# Convert PIL images to NumPy arrays for shape verification
derma_img_array = np.array(derma_img)
path_img_array = np.array(path_img)

print("DermaMNIST sample image shape:", derma_img_array.shape, "Label:", derma_label)
print("PathMNIST sample image shape:", path_img_array.shape, "Label:", path_label)