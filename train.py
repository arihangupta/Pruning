#!/usr/bin/env python3
"""
train_prune_blocklevel.py

Runs:
 - Baseline training (ResNet50)
 - Manual threshold pruning (0.001,0.01,0.05,0.1)
 - Global 70% unstructured pruning
 - Layerwise 70% unstructured pruning
 - Block-level pruning at stage granularity: layer1, layer2, layer3, layer4, fc

For each prune: evaluate BEFORE finetune, finetune (2 epochs), evaluate AFTER,
and save the finetuned model. Writes a CSV summary per dataset.

Requires: torch, torchvision, numpy, (scikit-learn optional for AUC)
"""
import os
import time
import random
import math
import csv
import numpy as np
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN = True
except Exception:
    SKLEARN = False

# -------------------------
# Config
# -------------------------
DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets"
SAVE_DIR    = "/home/arihangupta/Pruning/dinov2/Pruning/exp1_saved_models"

EPOCHS = 10
FINETUNE_EPOCHS = 2
BATCH_SIZE = 32
LR = 1e-3
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 20
SEED = 42

MANUAL_THRESHOLDS = [0.001, 0.01, 0.05, 0.1]
GLOBAL_SPARSITY = 0.7
LAYERWISE_SPARSITY = 0.7

BLOCKS_TO_PRUNE = ["layer1", "layer2", "layer3", "layer4", "fc"]

os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Data utilities
# -------------------------
def preprocess(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32") / 255.0
    if arr.ndim == 3 or arr.shape[-1] == 1:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    arr = np.transpose(arr, (0, 3, 1, 2))  # NHWC -> NCHW
    return arr

def make_loaders(npz_path: str) -> Tuple[DataLoader, DataLoader, DataLoader, int, str]:
    print(f"\nLoading {npz_path} ...")
    data = np.load(npz_path, mmap_mode="r")

    n_train = data["train_images"].shape[0]
    n_val   = data["val_images"].shape[0]
    n_test  = data["test_images"].shape[0]
    total   = n_train + n_val + n_test
    print(f"Dataset sizes: train={n_train}, val={n_val}, test={n_test}, total={total}")

    if total > 20000:
        print("Downsampling to 20,000 images (14k train, 2k val, 4k test)")
        idx = np.arange(total)
        np.random.shuffle(idx)
        idx = idx[:20000]

        train_idx = idx[:14000]
        val_idx   = idx[14000:16000]
        test_idx  = idx[16000:]

        def gather(indices):
            out_imgs, out_lbls = [], []
            for i in indices:
                if i < n_train:
                    out_imgs.append(data["train_images"][i])
                    out_lbls.append(data["train_labels"][i])
                elif i < n_train + n_val:
                    j = i - n_train
                    out_imgs.append(data["val_images"][j])
                    out_lbls.append(data["val_labels"][j])
                else:
                    j = i - n_train - n_val
                    out_imgs.append(data["test_images"][j])
                    out_lbls.append(data["test_labels"][j])
            return np.stack(out_imgs), np.array(out_lbls)

        X_train, y_train = gather(train_idx)
        X_val, y_val     = gather(val_idx)
        X_test, y_test   = gather(test_idx)

    else:
        X_train, y_train = data["train_images"], data["train_labels"].flatten()
        X_val, y_val     = data["val_images"], data["val_labels"].flatten()
        X_test, y_test   = data["test_images"], data["test_labels"].flatten()

    X_train = preprocess(X_train)
    X_val   = preprocess(X_val)
    X_test  = preprocess(X_test)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                             torch.tensor(y_val, dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = int(len(np.unique(y_train)))
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
        start = time.time()
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
        vloss, vacc, vauc = evaluate_model(model, val_loader)
        print(f"Epoch {ep+1} done in {time.time()-start:.1f}s - TrainLoss {running_loss/total:.4f} TrainAcc {correct/total:.4f} | ValLoss {vloss:.4f} ValAcc {vacc:.4f} ValAUC {vauc:.4f}")

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

# -------------------------
# Model stats helpers, pruning fns, run_dataset, etc.
# -------------------------
# [UNCHANGED from your original script — keep everything below as-is]
# -------------------------

if __name__ == "__main__":
    set_seed(SEED)
    print("Running on device:", DEVICE)
    print("SKLEARN available for AUC:", SKLEARN)

    npz_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith("_224.npz")]
    print("\nFound datasets:", npz_files)

    for npz_path in npz_files:
        run_dataset(npz_path)

    print("\nAll done.")
