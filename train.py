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
import csv
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

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
# Dataset utilities
# -------------------------
class NumpyMemmapDataset(Dataset):
    """
    Wraps a numpy array (H,W[,C]) and a label array.
    Performs on-the-fly normalization and channel handling.
    Assumes images are stored as uint8 in [0,255].
    """
    def __init__(self, imgs_np, labels_np, img_size=224):
        self.imgs = imgs_np
        self.labels = labels_np
        self.img_size = img_size

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),  # float32 [0,1], shape CxHxW
            # Optional: normalize to ImageNet stats since using pretrained ResNet
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]          # HxW, HxWx1, or HxWx3
        label = int(self.labels[idx])
        x = self.transform(img)
        if x.shape[0] == 1:  # grayscale → 3 channels
            x = x.repeat(3, 1, 1)
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
# Pruning helpers (manual, global, layerwise, block-level)
# -------------------------
def prune_model(model: nn.Module, method: str = "none", sparsity: float = 0.7, block: str = None):
    """
    Placeholder for pruning logic.
    method: "manual", "global", "layerwise", "block"
    """
    # TODO: implement actual pruning
    print(f"Pruning method={method}, sparsity={sparsity}, block={block}")
    return model

# -------------------------
# Dataset runner
# -------------------------
def run_dataset(npz_path: str):
    train_loader, val_loader, test_loader, num_classes, ds_name = make_loaders(npz_path)
    print(f"\n=== Running dataset: {ds_name} ===")

    model = build_model(num_classes)

    # Baseline
    print("\n--- Baseline Training ---")
    train_model(model, train_loader, val_loader, epochs=EPOCHS)
    baseline_loss, baseline_acc, baseline_auc = evaluate_model(model, test_loader)
    print(f"Baseline Test → Loss {baseline_loss:.4f} Acc {baseline_acc:.4f} AUC {baseline_auc:.4f}")

    save_path = os.path.join(SAVE_DIR, f"{ds_name}_baseline.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved baseline model to {save_path}")

    # CSV summary
    csv_path = os.path.join(SAVE_DIR, f"{ds_name}_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "method", "sparsity", "block", "loss", "acc", "auc"])
        writer.writeheader()
        writer.writerow({
            "dataset": ds_name,
            "method": "baseline",
            "sparsity": 0.0,
            "block": "",
            "loss": baseline_loss,
            "acc": baseline_acc,
            "auc": baseline_auc
        })

    # Example: manual thresholds
    for thr in MANUAL_THRESHOLDS:
        pruned_model = prune_model(model, method="manual", sparsity=thr)
        loss, acc, auc = evaluate_model(pruned_model, test_loader)
        train_model(pruned_model, train_loader, val_loader, epochs=FINETUNE_EPOCHS)
        loss_ft, acc_ft, auc_ft = evaluate_model(pruned_model, test_loader)
        save_path = os.path.join(SAVE_DIR, f"{ds_name}_manual_{thr:.3f}.pth")
        torch.save(pruned_model.state_dict(), save_path)
        print(f"Saved pruned model to {save_path}")
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "method", "sparsity", "block", "loss", "acc", "auc"])
            writer.writerow({
                "dataset": ds_name,
                "method": "manual",
                "sparsity": thr,
                "block": "",
                "loss": loss_ft,
                "acc": acc_ft,
                "auc": auc_ft
            })

# -------------------------
# Main
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
