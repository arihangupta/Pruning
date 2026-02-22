#!/usr/bin/env python3
"""
cnn_fix_chestmnist_acc.py

Recalculates accuracy for existing chestMNIST CNN model checkpoints using
mean per-label accuracy (instead of the incorrect exact-match accuracy that
was previously recorded), then updates the CSV files in-place.

Run this once on the server where the model checkpoints and CSVs live.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision.models.resnet import Bottleneck
from torchvision import models, transforms as T
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

# -------------------------
# Config — edit if paths differ
# -------------------------
DATASET_NAME  = "chestmnist"
IMG_SIZE      = 224
NPY_DIR       = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_npy"
SAVE_DIR      = f"/home/arihangupta/Pruning/dinov2/Pruning/PruneAndTrain/CNN/CNN_pruned_models/{DATASET_NAME}"
BATCH_SIZE    = 64
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Dataset (mirrors CNN_pruning.py exactly)
# -------------------------
class NumpyMemmapDataset(Dataset):
    def __init__(self, imgs_np, labels_np, img_size=224):
        self.imgs   = imgs_np
        self.labels = labels_np
        self.base_tfms = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = torch.FloatTensor(np.array(self.labels[idx]))
        x = self.base_tfms(img)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        x = self.normalize(x)
        return x, label


def load_test_loader():
    npy_dir = os.path.join(NPY_DIR, f"{DATASET_NAME}_{IMG_SIZE}")
    X_test  = np.load(os.path.join(npy_dir, "test_images.npy"), mmap_mode="r")
    y_test  = np.load(os.path.join(npy_dir, "test_labels.npy"), mmap_mode="r")
    num_classes = y_test.shape[1]
    ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)
    print(f"Test set: {len(ds)} samples, {num_classes} labels")
    return loader, num_classes


# -------------------------
# Architecture (mirrors CNN_pruning.py exactly)
# -------------------------
class CustomResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=(3, 4, 6, 3),
                 stage_planes=(64, 128, 256, 512), num_classes=1000):
        super().__init__()
        self.inplanes   = 64
        self.conv1      = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1        = nn.BatchNorm2d(64)
        self.relu       = nn.ReLU(inplace=True)
        self.maxpool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage_planes = list(stage_planes)
        self.layers_cfg   = list(layers)
        self.layer1 = self._make_layer(block, stage_planes[0], layers[0])
        self.layer2 = self._make_layer(block, stage_planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, stage_planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, stage_planes[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_planes[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x


def _infer_stage_planes(state_dict):
    """Read stage_planes from the first conv1 weight of each layer block."""
    planes = []
    for layer in ('layer1', 'layer2', 'layer3', 'layer4'):
        key = f'{layer}.0.conv1.weight'
        if key not in state_dict:
            return None
        planes.append(state_dict[key].shape[0])
    return planes


def load_model(path, num_classes):
    """Auto-detect architecture (ResNet50 baseline vs. pruned CustomResNet) and load."""
    state = torch.load(path, map_location='cpu', weights_only=True)

    # Try vanilla ResNet50 first (baseline checkpoint)
    try:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(state, strict=True)
        return model.to(DEVICE).eval()
    except Exception:
        pass

    # Fall back to CustomResNet with inferred stage_planes (pruned checkpoints)
    stage_planes = _infer_stage_planes(state)
    if stage_planes is None:
        print(f"    Could not infer architecture from {os.path.basename(path)}")
        return None
    try:
        model = CustomResNet(stage_planes=stage_planes, num_classes=num_classes)
        model.load_state_dict(state, strict=True)
        return model.to(DEVICE).eval()
    except Exception as e:
        print(f"    Load failed for {os.path.basename(path)}: {e}")
        return None


# -------------------------
# Evaluation
# -------------------------
def evaluate(model, loader):
    """Return (mean_per_label_acc, macro_auc, per_label_acc_list, per_label_auc_list)."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            probs  = torch.sigmoid(model(images)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    preds      = (all_probs > 0.5).astype(float)

    num_labels     = all_labels.shape[1]
    per_label_acc  = []
    per_label_auc  = []

    for i in range(num_labels):
        per_label_acc.append(float((preds[:, i] == all_labels[:, i]).mean()))
        try:
            per_label_auc.append(float(roc_auc_score(all_labels[:, i], all_probs[:, i])))
        except Exception:
            per_label_auc.append(float('nan'))

    mean_acc = float(np.mean(per_label_acc))
    try:
        macro_auc = float(roc_auc_score(all_labels, all_probs, average='macro'))
    except Exception:
        macro_auc = float('nan')

    return mean_acc, macro_auc, per_label_acc, per_label_auc


# -------------------------
# Main
# -------------------------
def main():
    print(f"Device: {DEVICE}")
    print(f"Loading test data...")
    test_loader, num_classes = load_test_loader()

    csv_path = os.path.join(SAVE_DIR, f"{DATASET_NAME}_combined_pruning_kd_metrics_with_energy.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded CSV: {len(df)} rows")

    per_label_rows = []
    updated = 0

    for idx, row in df.iterrows():
        model_path = str(row.get("ModelPath", ""))
        variant    = str(row.get("Variant", f"row_{idx}"))

        if not model_path or not os.path.isfile(model_path):
            print(f"  [{idx}] '{variant}' — no valid model path, skipping")
            continue

        print(f"  [{idx}] Evaluating '{variant}' ({os.path.basename(model_path)})...")
        model = load_model(model_path, num_classes)
        if model is None:
            continue

        mean_acc, macro_auc, per_label_acc, per_label_auc = evaluate(model, test_loader)

        old_acc = row.get("Acc", float('nan'))
        df.at[idx, "Acc"] = mean_acc
        print(f"         Acc: {old_acc:.4f} → {mean_acc:.4f} (mean per-label)  |  AUC: {macro_auc:.4f}")

        for i, (acc_i, auc_i) in enumerate(zip(per_label_acc, per_label_auc)):
            per_label_rows.append({
                "dataset":   DATASET_NAME,
                "variant":   variant,
                "stage":     row.get("Stage", ""),
                "label_idx": i,
                "auc":       auc_i,
                "acc":       acc_i,
            })

        del model
        torch.cuda.empty_cache()
        updated += 1

    # Write updated main CSV
    df.to_csv(csv_path, index=False)
    print(f"\nDone. Updated {updated}/{len(df)} rows in:\n  {csv_path}")

    # Write per-label CSV
    if per_label_rows:
        per_label_csv = os.path.join(SAVE_DIR, f"{DATASET_NAME}_per_label_metrics.csv")
        pd.DataFrame(per_label_rows).to_csv(per_label_csv, index=False)
        print(f"Per-label metrics written to:\n  {per_label_csv}")


if __name__ == "__main__":
    main()
