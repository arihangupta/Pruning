#!/usr/bin/env python3
"""
train_vit_baseline.py

Runs DINO pretraining followed by fine-tuning with DINOv2 ViT-S/14 backbone + linear head on MedMNIST datasets.
Mimics the provided CNN script: trains, evaluates on test, saves model, writes CSV summary.
No pruning.

Requires: torch, torchvision, numpy, (scikit-learn optional for AUC)
DINOv2 loaded via torch.hub (requires internet on first run).
"""
import os
import time
import random
import csv
import numpy as np
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN = True
except Exception:
    SKLEARN = False

# -------------------------
# Config
# -------------------------
DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets"
SAVE_DIR    = "/home/arihangupta/Pruning/dinov2/Pruning/exp2_saved_models"

PRETRAIN_EPOCHS = 50  # Reduced for practicality
FINETUNE_EPOCHS = 50
BATCH_SIZE = 32
LR = 5e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 20
SEED = 42
TEMPERATURE = 0.1  # DINO temperature
CENTER_MOMENTUM = 0.9
PROJ_OUT_DIM = 65536  # projection head output dim (large; reduce if memory is tight)

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
    def __init__(self, imgs_np, labels_np, img_size=224, train=True, multi_crop=False):
        self.imgs = imgs_np
        self.labels = labels_np if labels_np is not None else None
        self.img_size = img_size
        self.train = train
        self.multi_crop = multi_crop

        self.base_tfms = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.multi_crop_tfms = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]) if multi_crop else None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        x = self.base_tfms(img)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        if self.multi_crop:
            # 2 global + 6 local (as earlier). You can change counts to reduce memory.
            crops = [self.multi_crop_tfms(x) for _ in range(2)] + [
                self.multi_crop_tfms(T.Resize((self.img_size, self.img_size))(x)) for _ in range(6)
            ]
            crops = [self.normalize(c) for c in crops]
            return crops, (self.labels[idx] if self.labels is not None else None)
        x = self.normalize(x)
        return x, (self.labels[idx] if self.labels is not None else None)

def collate_fn(batch):
    imgs, labels = zip(*batch)
    if isinstance(imgs[0], list):  # Multi-crop case: imgs is list-of-lists
        n_crops = len(imgs[0])
        # Stack per-crop across batch producing list of length n_crops each [B,3,H,W]
        images_stacked = [torch.stack([img[i] for img in imgs]) for i in range(n_crops)]
        images_tensor = torch.stack(images_stacked)  # [n_crops, B, 3, H, W]
    else:
        images_tensor = torch.stack(imgs)  # [B, 3, H, W]
    if labels[0] is None:
        labels_tensor = None
    else:
        labels_tensor = torch.tensor(labels)
    return images_tensor, labels_tensor

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

    pretrain_ds = NumpyMemmapDataset(X_train, None, img_size=IMG_SIZE, train=True, multi_crop=True)
    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE, train=True)
    val_ds   = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE, train=False)
    test_ds  = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, train=False)

    pretrain_loader = DataLoader(pretrain_ds, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=2, pin_memory=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    ds_name = os.path.splitext(os.path.basename(npz_path))[0]
    return pretrain_loader, train_loader, val_loader, test_loader, num_classes, ds_name

# -------------------------
# Model / train / eval
# -------------------------
class DINOModel(nn.Module):
    def __init__(self, backbone, out_dim=PROJ_OUT_DIM):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(384, 2048), nn.GELU(), nn.Linear(2048, out_dim))

        # Probe backbone once (no grads) to detect pooled vs sequence output
        self._pooled = None
        self.cls_index = None
        with torch.no_grad():
            sample = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            sample_out = self.backbone(sample)
            if sample_out.dim() == 2:
                self._pooled = True
            else:
                self._pooled = False
                self.cls_index = 0

    def forward(self, x):
        # x: [B, 3, H, W]
        feat = self.backbone(x)
        if not self._pooled:
            feat = feat[:, self.cls_index]
        out = self.head(feat)
        return out  # [B, out_dim]


class ViTClassifier(nn.Module):
    def __init__(self, backbone, num_classes, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(384, num_classes)

        # Probe backbone once
        self._pooled = None
        self.cls_index = None
        with torch.no_grad():
            sample = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            sample_out = self.backbone(sample)
            if sample_out.dim() == 2:
                self._pooled = True
            else:
                self._pooled = False
                self.cls_index = 0

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        feat = self.backbone(x)
        if not self._pooled:
            feat = feat[:, self.cls_index]
        out = self.head(feat)
        return out  # [B, num_classes]


def build_model(num_classes_or_outdim: int, pretrain=False, freeze_backbone=False):
    """
    If pretrain=True, num_classes_or_outdim is treated as projection out_dim for DINOModel.
    Otherwise it's treated as num_classes for ViTClassifier.
    """
    print("Building DINOv2 ViT-S/14 backbone (torch.hub)...")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    # ensure backbone on device
    backbone = backbone.to(DEVICE)
    if pretrain:
        return DINOModel(backbone, out_dim=num_classes_or_outdim)
    return ViTClassifier(backbone, num_classes_or_outdim, freeze_backbone)


def make_optimizer(model: nn.Module):
    return optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

criterion = nn.CrossEntropyLoss()

# -------------------------
# DINO pretraining
# -------------------------
def train_dino(model: nn.Module, pretrain_loader: DataLoader, epochs: int, out_dim=PROJ_OUT_DIM):
    model = model.to(DEVICE)   ### FIX ensure student on device
    opt = make_optimizer(model)
    n_batches = len(pretrain_loader)
    scheduler = CosineAnnealingLR(opt, T_max=epochs * n_batches)

    # create teacher initialized as copy of student
    teacher = build_model(out_dim, pretrain=True).to(DEVICE)   ### FIX ensure teacher on device
    teacher.load_state_dict(model.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    center = torch.zeros(out_dim, device=DEVICE)
    ema_m = 0.996  # momentum for teacher EMA

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        start = time.time()
        for bidx, (images, _) in enumerate(pretrain_loader, 1):
            # images can be [n_crops, B, 3, H, W]
            images = images.to(DEVICE)  # keep on device
            if images.dim() == 5 and images.size(0) > 1:
                n_crops = images.size(0)
                B = images.size(1)
                C = images.size(2)
                H = images.size(3)
                W = images.size(4)
                # Flatten crops+batch into a single batch: order is crop0 batch0..B-1, crop1 batch0..B-1, ...
                imgs_reshaped = images.reshape(n_crops * B, C, H, W)
                # student uses first 2 global views (all batch images of crops 0 and 1)
                student_in = imgs_reshaped[:2 * B]
                # teacher uses remaining local views
                teacher_in = imgs_reshaped[2 * B:]
                # forward
                student_feats = model(student_in)           # [2*B, D]
                teacher_feats = teacher(teacher_in).detach()  # [(n_crops-2)*B, D], detached
                # reshape
                student_out = student_feats.reshape(2, B, -1)   # [2, B, D]
                teacher_out = teacher_feats.reshape(n_crops - 2, B, -1)  # [n_local, B, D]
            else:
                # No multi-crop case (rare here): handle single view
                imgs = images if images.dim() == 4 else images.squeeze(0)
                B = imgs.size(0)
                student_out = model(imgs).unsqueeze(0)  # [1, B, D]
                teacher_out = student_out.detach()      # [1, B, D]
                n_crops = 1

            # Centering and softmax/temperature
            # teacher_out: [n_local, B, D]
            # student_out: [n_global, B, D]
            # teacher logits: subtract center then /T, produce soft targets
            teacher_logits = (teacher_out - center.unsqueeze(0).unsqueeze(0)) / TEMPERATURE
            teacher_probs = F.softmax(teacher_logits, dim=2).detach()  # [n_local, B, D], no grad

            student_logits = (student_out - center.unsqueeze(0).unsqueeze(0)) / TEMPERATURE
            student_log_probs = F.log_softmax(student_logits, dim=2)  # [n_global, B, D]

            # loss: average cross-entropy across all global x local pairs
            loss = 0.0
            n_pairs = 0
            for i in range(student_log_probs.size(0)):      # global views
                for j in range(teacher_probs.size(0)):     # local views
                    # (teacher_probs[j] * student_log_probs[i]).sum(dim=1) is per-sample
                    loss += - (teacher_probs[j] * student_log_probs[i]).sum(dim=1).mean()
                    n_pairs += 1
            loss = loss / max(1, n_pairs)

            # backward + step
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            # EMA update of teacher params
            with torch.no_grad():
                for ps, pt in zip(model.parameters(), teacher.parameters()):
                    pt.data = pt.data * ema_m + ps.data * (1.0 - ema_m)

            # update running center (mean over teacher outputs before softmax)
            with torch.no_grad():
                mean_teacher = teacher_out.mean(dim=(0, 1))  # [D]
                center = center * CENTER_MOMENTUM + mean_teacher * (1 - CENTER_MOMENTUM)

            running_loss += loss.item() * B
            total += B
            if bidx % LOG_INTERVAL == 0 or bidx == n_batches:
                print(f"  DINO Epoch {ep+1} Batch {bidx}/{n_batches} - loss {running_loss/total:.4f}")
        print(f"DINO Epoch {ep+1} done in {time.time()-start:.1f}s - Loss {running_loss/total:.4f}")

# -------------------------
# Fine-tune training
# -------------------------
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
    opt = make_optimizer(model)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)  # step once per epoch
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
        scheduler.step()
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
# Dataset runner
# -------------------------
def run_dataset(npz_path: str, freeze_backbone=False):
    pretrain_loader, train_loader, val_loader, test_loader, num_classes, ds_name = make_loaders(npz_path)
    print(f"\n=== Running dataset: {ds_name} ===")

    # DINO Pretraining
    print("\n--- DINO Pretraining ---")
    dino_model = build_model(PROJ_OUT_DIM, pretrain=True).to(DEVICE)   ### FIX
    train_dino(dino_model, pretrain_loader, PRETRAIN_EPOCHS, out_dim=PROJ_OUT_DIM)
    dino_pre_path = os.path.join(SAVE_DIR, f"{ds_name}_dino_pretrained.pth")
    torch.save(dino_model.state_dict(), dino_pre_path)

    # Fine-tuning with pretrained weights
    print("\n--- Baseline Fine-tuning ---")
    model = build_model(num_classes, pretrain=False, freeze_backbone=freeze_backbone).to(DEVICE)   ### FIX
    model.backbone.load_state_dict(dino_model.backbone.state_dict())
    train_model(model, train_loader, val_loader, FINETUNE_EPOCHS)
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

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    set_seed(SEED)
    print("Running on device:", DEVICE)
    print("SKLEARN available for AUC:", SKLEARN)

    npz_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith("_224.npz")]
    print("\nFound datasets:", npz_files)

    # Run with full fine-tuning (default)
    for npz_path in npz_files:
        run_dataset(npz_path, freeze_backbone=False)
    # Optional: Run with frozen backbone
    # for npz_path in npz_files:
    #     run_dataset(npz_path, freeze_backbone=True)

    print("\nAll done.")
