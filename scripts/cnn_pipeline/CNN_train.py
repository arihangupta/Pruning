#!/usr/bin/env python3
"""
train_prune_blocklevel.py

Runs baseline training (ResNet50) and saves model metrics including AUC, accuracy, and size.
Writes a CSV summary per dataset for dermamnist, bloodmnist, pathmnist, and chestmnist.
ChestMNIST is multi-label (14 binary labels) and uses BCEWithLogitsLoss + sigmoid.

Requires: torch, torchvision, numpy, (scikit-learn optional for AUC)
"""
import os
import csv
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
from thop import profile

try:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    SKLEARN = True
except Exception:
    SKLEARN = False

# -------------------------
# Config
# -------------------------
DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
NPY_DIR     = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_npy"
SAVE_DIR    = "/home/arihangupta/Pruning/dinov2/Pruning/CNN/CNN_exp1"

EPOCHS = 10
CHEST_EPOCHS = 30
CHEST_EARLY_STOP_PATIENCE = 5
BATCH_SIZE = 32
LR = 1e-3
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 20
SEED = 42

# Multi-label datasets
MULTI_LABEL_DATASETS = {'chestmnist'}

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
    Supports multi-class (single int label) and multi-label (float vector).
    """
    def __init__(self, imgs_np, labels_np, img_size=224, multi_label=False):
        self.imgs = imgs_np
        self.labels = labels_np
        self.img_size = img_size
        self.multi_label = multi_label

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
        x = self.base_tfms(img)
        if x.shape[0] == 1:  # if grayscale → expand to RGB
            x = x.repeat(3, 1, 1)
        x = self.normalize(x)

        if self.multi_label:
            label = torch.FloatTensor(np.array(self.labels[idx]))
        else:
            label = int(self.labels[idx])

        return x, label

def make_loaders(dataset_name: str) -> Tuple[DataLoader, DataLoader, DataLoader, int, str, bool]:
    multi_label = dataset_name in MULTI_LABEL_DATASETS

    if multi_label:
        npy_dir = os.path.join(NPY_DIR, f"{dataset_name}_{IMG_SIZE}")
        print(f"\nLoading {npy_dir} (multi-label) ...")
        X_train = np.load(os.path.join(npy_dir, "train_images.npy"), mmap_mode="r")
        y_train = np.load(os.path.join(npy_dir, "train_labels.npy"), mmap_mode="r")
        X_val   = np.load(os.path.join(npy_dir, "val_images.npy"), mmap_mode="r")
        y_val   = np.load(os.path.join(npy_dir, "val_labels.npy"), mmap_mode="r")
        X_test  = np.load(os.path.join(npy_dir, "test_images.npy"), mmap_mode="r")
        y_test  = np.load(os.path.join(npy_dir, "test_labels.npy"), mmap_mode="r")
        num_classes = y_train.shape[1]
    else:
        npz_path = os.path.join(DATASET_DIR, f"{dataset_name}_224.npz")
        print(f"\nLoading {npz_path} ...")
        data = np.load(npz_path, mmap_mode="r")
        X_train = data["train_images"]
        y_train = data["train_labels"].flatten()
        X_val   = data["val_images"]
        y_val   = data["val_labels"].flatten()
        X_test  = data["test_images"]
        y_test  = data["test_labels"].flatten()
        num_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))

    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    total = n_train + n_val + n_test
    print(f"Dataset sizes: train={n_train}, val={n_val}, test={n_test}, total={total}")
    print(f"Num classes: {num_classes}, Multi-label: {multi_label}")

    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE, multi_label=multi_label)
    val_ds   = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE, multi_label=multi_label)
    test_ds  = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, multi_label=multi_label)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    ds_name = f"{dataset_name}_224"
    return train_loader, val_loader, test_loader, num_classes, ds_name, multi_label

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

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int, multi_label: bool = False, early_stop_patience: int = 0):
    opt = make_optimizer(model)
    loss_fn = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()

    best_auc = -1.0
    patience_counter = 0

    for ep in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        label_correct, label_total = 0, 0  # for multi-label per-label accuracy
        all_probs, all_labels = [], []  # for running AUC
        for bidx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            if multi_label:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                # Per-label accuracy (mean across 14 labels)
                label_correct += (preds == labels).sum().item()
                label_total += labels.numel()
                # Exact-match accuracy (all 14 labels correct)
                correct += (preds == labels).all(dim=1).sum().item()
                # Collect for running AUC
                all_probs.append(probs.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
            else:
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()

            if bidx % LOG_INTERVAL == 0 or bidx == len(train_loader):
                if multi_label:
                    per_label_acc = label_correct / max(1, label_total)
                    exact_match = correct / max(1, total)
                    # Compute running AUC
                    running_auc = float('nan')
                    if SKLEARN:
                        try:
                            _probs = np.concatenate(all_probs, axis=0)
                            _labels = np.concatenate(all_labels, axis=0)
                            running_auc = roc_auc_score(_labels, _probs, average='macro')
                        except Exception:
                            pass
                    print(f"  Epoch {ep+1} Batch {bidx}/{len(train_loader)} - loss {running_loss/total:.4f} "
                          f"per-label-acc {per_label_acc:.4f} exact-match {exact_match:.4f} AUC {running_auc:.4f}")
                else:
                    print(f"  Epoch {ep+1} Batch {bidx}/{len(train_loader)} - loss {running_loss/total:.4f} acc {correct/total:.4f}")

        # End-of-epoch validation for early stopping (multi-label only)
        if multi_label and early_stop_patience > 0:
            val_loss, val_acc, val_auc = evaluate_model(model, val_loader, multi_label=multi_label)
            print(f"  Epoch {ep+1} Val → loss {val_loss:.4f} acc {val_acc:.4f} AUC {val_auc:.4f}")
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  Early stopping counter: {patience_counter}/{early_stop_patience}")
                if patience_counter >= early_stop_patience:
                    print(f"  Early stopping triggered at epoch {ep+1}")
                    break

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader,
                   multi_label: bool = False) -> Tuple[float, float, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    loss_total, correct, total = 0.0, 0, 0
    probs_list, labels_list = [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss_total += loss.item() * images.size(0)
        total += labels.size(0)

        if multi_label:
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).all(dim=1).sum().item()
        else:
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()

        probs_list.append(probs.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

    avg_loss = loss_total / max(1, total)
    acc = correct / max(1, total)

    if SKLEARN:
        try:
            probs_all = np.concatenate(probs_list, axis=0)
            labels_all = np.concatenate(labels_list, axis=0)
            if multi_label:
                auc = roc_auc_score(labels_all, probs_all, average="macro")
            else:
                auc = roc_auc_score(labels_all, probs_all, multi_class="ovr", average="macro")
        except Exception:
            auc = float('nan')
    else:
        auc = float('nan')
    return avg_loss, acc, auc

def count_params(model: nn.Module) -> float:
    """Returns number of parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6

# -------------------------
# Dataset runner
# -------------------------
def run_dataset(dataset_name: str):
    train_loader, val_loader, test_loader, num_classes, ds_name, multi_label = make_loaders(dataset_name)
    print(f"\n=== Running dataset: {ds_name} ===")

    model = build_model(num_classes)

    # Baseline Training
    if multi_label:
        epochs = CHEST_EPOCHS
        patience = CHEST_EARLY_STOP_PATIENCE
        print(f"\n--- Baseline Training (multi-label: {epochs} epochs, early stop patience {patience}) ---")
    else:
        epochs = EPOCHS
        patience = 0
        print("\n--- Baseline Training ---")
    train_model(model, train_loader, val_loader, epochs=epochs, multi_label=multi_label,
                early_stop_patience=patience)
    loss, acc, auc = evaluate_model(model, test_loader, multi_label=multi_label)
    params_m = count_params(model)
    print(f"Baseline Test → Loss {loss:.4f} Acc {acc:.4f} AUC {auc:.4f} Params {params_m:.2f}M")

    save_path = os.path.join(SAVE_DIR, f"{ds_name}_baseline.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved baseline model to {save_path}")

    # CSV summary
    csv_path = os.path.join(SAVE_DIR, f"{ds_name}_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "method", "loss", "acc", "auc", "params_m", "multi_label"])
        writer.writeheader()
        writer.writerow({
            "dataset": ds_name,
            "method": "baseline",
            "loss": loss,
            "acc": acc,
            "auc": auc,
            "params_m": params_m,
            "multi_label": multi_label
        })

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    set_seed(SEED)
    print("Running on device:", DEVICE)
    print("SKLEARN available for AUC:", SKLEARN)

    target_datasets = ["dermamnist", "bloodmnist", "pathmnist", "chestmnist"]

    for ds in target_datasets:
        save_path = os.path.join(SAVE_DIR, f"{ds}_224_baseline.pth")
        if os.path.exists(save_path):
            print(f"\nSkipping {ds}: baseline already exists at {save_path}")
            continue
        run_dataset(ds)

    print("\nAll done.")
