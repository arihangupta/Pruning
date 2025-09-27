#!/usr/bin/env python3
"""
train_prune_blocklevel.py

Runs baseline training (ResNet50) and saves model metrics including AUC, accuracy, and size.
Writes a CSV summary per dataset for dermamnist, bloodmnist, and pathmnist.

Requires: torch, torchvision, numpy, (scikit-learn optional for AUC)
"""
import os
import csv
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
from thop import profile

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN = True
except Exception:
    SKLEARN = False

# -------------------------
# Config
# -------------------------
DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
SAVE_DIR    = "/home/arihangupta/Pruning/dinov2/Pruning/CNN_exp1"

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 20
SEED = 42

os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Dataset utilities
# -------------------------
class NumpyMemmapDataset(Dataset):
    """
    Wraps a numpy array (H,W[,C]) and a label array.
    Auto-detects grayscale vs RGB and normalizes accordingly.
    """
    def __init__(self, imgs_np, labels_np, img_size=224):
        self.imgs = imgs_np
        self.labels = labels_np
        self.img_size = img_size

        self.base_tfms = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),  # float32 [0,1], shape CxHxW
        ])
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = int(self.labels[idx])
        x = self.base_tfms(img)
        if x.shape[0] == 1:  # if grayscale → expand to RGB
            x = x.repeat(3, 1, 1)
        x = self.normalize(x)
        return x, label

def make_loaders(npz_path: str) -> Tuple[DataLoader, DataLoader, DataLoader, int, str]:
    print(f"\nLoading {npz_path} ...")
    data = np.load(npz_path, mmap_mode="r")

    X_train = data["train_images"]
    y_train = data["train_labels"].flatten()
    X_val   = data["val_images"]
    y_val   = data["val_labels"].flatten()
    X_test  = data["test_images"]
    y_test  = data["test_labels"].flatten()

    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    total = n_train + n_val + n_test
    print(f"Dataset sizes: train={n_train}, val={n_val}, test={n_test}, total={total}")

    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE)
    val_ds   = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE)
    test_ds  = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    num_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    ds_name = os.path.splitext(os.path.basename(npz_path))[0]
    return train_loader, val_loader, test_loader, num_classes, ds_name

# -------------------------
# Model / train / eval
# -------------------------
def build_model(num_classes: int) -> nn.Module:
    print("Building ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

def make_optimizer(model: nn.Module):
    return optim.Adam(model.parameters(), lr=LR)

criterion = nn.CrossEntropyLoss()

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
    opt = make_optimizer(model)
    for ep in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for bidx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            if bidx % LOG_INTERVAL == 0 or bidx == len(train_loader):
                print(f"  Epoch {ep+1} Batch {bidx}/{len(train_loader)} - loss {running_loss/total:.4f} acc {correct/total:.4f}")

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader) -> Tuple[float, float, float]:
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    probs_list, labels_list = [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_total += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        probs = torch.softmax(outputs, dim=1)
        probs_list.append(probs.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
    avg_loss = loss_total / max(1, total)
    acc = correct / max(1, total)
    if SKLEARN:
        try:
            probs_all = np.concatenate(probs_list, axis=0)
            labels_all = np.concatenate(labels_list, axis=0)
            auc = roc_auc_score(labels_all, probs_all, multi_class="ovr", average="macro")
        except Exception:
            auc = float('nan')
    else:
        auc = float('nan')
    return avg_loss, acc, auc

def count_params(model: nn.Module) -> float:
    """
    Returns number of parameters in millions.
    """
    return sum(p.numel() for p in model.parameters()) / 1e6

# -------------------------
# Dataset runner
# -------------------------
def run_dataset(npz_path: str):
    train_loader, val_loader, test_loader, num_classes, ds_name = make_loaders(npz_path)
    print(f"\n=== Running dataset: {ds_name} ===")

    model = build_model(num_classes)

    # Baseline Training
    print("\n--- Baseline Training ---")
    train_model(model, train_loader, val_loader, epochs=EPOCHS)
    loss, acc, auc = evaluate_model(model, test_loader)
    params_m = count_params(model)
    print(f"Baseline Test → Loss {loss:.4f} Acc {acc:.4f} AUC {auc:.4f} Params {params_m:.2f}M")

    save_path = os.path.join(SAVE_DIR, f"{ds_name}_baseline.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved baseline model to {save_path}")

    # CSV summary
    csv_path = os.path.join(SAVE_DIR, f"{ds_name}_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "method", "loss", "acc", "auc", "params_m"])
        writer.writeheader()
        writer.writerow({
            "dataset": ds_name,
            "method": "baseline",
            "loss": loss,
            "acc": acc,
            "auc": auc,
            "params_m": params_m
        })

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    set_seed(SEED)
    print("Running on device:", DEVICE)
    print("SKLEARN available for AUC:", SKLEARN)

    target_datasets = ["dermamnist", "bloodmnist", "pathmnist"]
    npz_files = [os.path.join(DATASET_DIR, f"{ds}_224.npz") for ds in target_datasets]
    print("\nFound datasets:", npz_files)

    for npz_path in npz_files:
        run_dataset(npz_path)

    print("\nAll done.")