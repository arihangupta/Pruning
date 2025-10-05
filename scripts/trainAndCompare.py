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
# Config (editable)
# -------------------------
DERMA_PATH = "/home/arihangupta/Pruning/dinov2/Pruning/datasets/dermamnist_224.npz"
PATH_PATH  = "/home/arihangupta/Pruning/dinov2/Pruning/datasets/pathmnist_224.npz"
SAVE_DIR   = "/home/arihangupta/Pruning/dinov2/Pruning/saved_models"

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
    # Make deterministic (may reduce performance)
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
    data = np.load(npz_path)
    X_train, y_train = data["train_images"], data["train_labels"].flatten()
    X_val, y_val     = data["val_images"], data["val_labels"].flatten()
    X_test, y_test   = data["test_images"], data["test_labels"].flatten()
    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

    X_train = preprocess(X_train)
    X_val = preprocess(X_val)
    X_test = preprocess(X_test)

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
    ds_name = "dermamnist" if "derma" in os.path.basename(npz_path).lower() else "pathmnist"
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
        running_loss = 0.0
        correct = 0
        total = 0
        start = time.time()
        for bidx, (images, labels) in enumerate(train_loader, 1):
            images = images.to(DEVICE); labels = labels.to(DEVICE)
            opt.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            if bidx % LOG_INTERVAL == 0 or bidx==len(train_loader):
                print(f"  Epoch {ep+1} Batch {bidx}/{len(train_loader)} - loss {running_loss/total:.4f} acc {correct/total:.4f}")
        vloss, vacc, vauc = evaluate_model(model, val_loader)
        print(f"Epoch {ep+1} done in {time.time()-start:.1f}s - TrainLoss {running_loss/total:.4f} TrainAcc {correct/total:.4f} | ValLoss {vloss:.4f} ValAcc {vacc:.4f} ValAUC {vauc:.4f}")

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader) -> Tuple[float, float, float]:
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    probs_list = []
    labels_list = []
    for images, labels in loader:
        images = images.to(DEVICE); labels = labels.to(DEVICE)
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
# Model stats helpers
# -------------------------
def count_zeros_and_total(model: nn.Module) -> Tuple[int, int]:
    total = 0
    zeros = 0
    for p in model.parameters():
        num = p.numel()
        total += num
        zeros += int((p == 0).sum().item())
    return zeros, total

def compute_memory_mb(model: nn.Module) -> float:
    params = sum(p.numel() for p in model.parameters())
    return params * 4.0 / (1024**2)

def compute_flops(model: nn.Module, input_size=(1,3,IMG_SIZE,IMG_SIZE)) -> int:
    """
    Approximate FLOPs by registering hooks that record conv/dense operations.
    Only counts conv and linear multiplies-adds in a forward pass.
    """
    flops = 0
    hooks = []

    def conv_hook(self, inp, out):
        # inp[0] shape: N, C_in, H_in, W_in
        # out shape: N, C_out, H_out, W_out
        in_t = inp[0]
        out_t = out
        batch = in_t.shape[0]
        C_in = in_t.shape[1]
        C_out = out_t.shape[1]
        H_out = out_t.shape[2]
        W_out = out_t.shape[3]
        kh, kw = self.kernel_size
        # conv FLOPs: kh*kw*C_in*C_out*H_out*W_out (per output) * 2 for MAC? we'll count MACs
        nonlocal flops
        flops += kh * kw * C_in * C_out * H_out * W_out

    def linear_hook(self, inp, out):
        in_t = inp[0]
        N = in_t.shape[0]
        in_features = in_t.shape[1]
        out_features = out.shape[1]
        nonlocal flops
        flops += in_features * out_features

    # register hooks
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    # run a dummy forward
    model.eval()
    device = next(model.parameters()).device if any(p.numel()>0 for p in model.parameters()) else DEVICE
    dummy = torch.randn(*input_size).to(device)
    with torch.no_grad():
        _ = model(dummy)

    # remove hooks
    for h in hooks:
        h.remove()
    return int(flops)

def measure_inference_time(model: nn.Module, loader: DataLoader, warmup=1) -> float:
    # measure full pass over test set
    model.eval()
    start = time.time()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            _ = model(images)
    return time.time() - start

# -------------------------
# Pruning implementations
# -------------------------
def manual_prune_threshold(model: nn.Module, threshold: float):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None:
                    w = m.weight.data
                    mask = w.abs() < threshold
                    w[mask] = 0.0
    return model

def global_prune_percentile(model: nn.Module, sparsity: float):
    # gather all weights
    all_w = torch.cat([p.detach().abs().view(-1) for p in model.parameters() if p.requires_grad])
    k = int(math.floor(sparsity * all_w.numel()))
    if k <= 0:
        return model
    thr = torch.kthvalue(all_w, k).values.item()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                mask = p.abs() < thr
                p[mask] = 0.0
    return model

def layerwise_prune_percentile(model: nn.Module, sparsity: float):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                flat = p.detach().abs().view(-1)
                if flat.numel() == 0:
                    continue
                k = int(math.floor(sparsity * flat.numel()))
                if k <= 0:
                    continue
                thr = torch.kthvalue(flat, k).values.item()
                mask = p.abs() < thr
                p[mask] = 0.0
    return model

def block_prune_stage(model: nn.Module, stage_name: str):
    """
    Zero all parameters inside the stage 'layer1','layer2','layer3','layer4' or fc.
    """
    with torch.no_grad():
        if stage_name == "fc":
            m = model.fc
            if hasattr(m, "weight"):
                m.weight.data.zero_()
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.data.zero_()
            return model
        stage = getattr(model, stage_name, None)
        if stage is None:
            return model
        # stage is nn.Sequential
        for child in stage.children():
            for p in child.parameters():
                p.data.zero_()
    return model

# -------------------------
# Running experiments + result recording
# -------------------------
def evaluate_and_record(model: nn.Module, test_loader: DataLoader) -> Dict:
    zeros, total = count_zeros_and_total(model)
    nonzero = total - zeros
    flops = compute_flops(model)
    mem = compute_memory_mb(model)
    nonzero_frac = nonzero / total if total > 0 else 1.0
    power_proxy = flops * nonzero_frac
    inf_time = measure_inference_time(model, test_loader)
    loss, acc, auc = evaluate_model(model, test_loader)
    return {
        "Zeros": int(zeros),
        "Total": int(total),
        "Nonzero": int(nonzero),
        "FLOPs": float(flops),
        "MemoryMB": float(mem),
        "PowerProxy": float(power_proxy),
        "InferenceTime": float(inf_time),
        "Loss": float(loss),
        "Accuracy": float(acc),
        "AUC": float(auc) if not math.isnan(auc) else float('nan')
    }

def save_model(path: str, model: nn.Module):
    torch.save(model.state_dict(), path)

def run_dataset(npz_path: str):
    set_seed(SEED)
    train_loader, val_loader, test_loader, num_classes, ds_name = make_loaders(npz_path)
    results = []

    # Baseline
    print(f"\n=== {ds_name} : Baseline training ===")
    base = build_model(num_classes)
    base = base.to(DEVICE)
    train_model(base, train_loader, val_loader, EPOCHS)
    base_ckpt = os.path.join(SAVE_DIR, f"{ds_name}_resnet50_BASELINE.pth")
    save_model(base_ckpt, base)
    print("Saved baseline:", base_ckpt)
    base_stats = evaluate_and_record(base, test_loader)
    base_row = {"Method":"BASELINE", **base_stats, "ModelPath": base_ckpt}
    results.append(base_row)
    print("Baseline stats:", base_row)

    # Manual thresholds
    for thr in MANUAL_THRESHOLDS:
        print(f"\n--- {ds_name} Manual pruning thr={thr} ---")
        m = build_model(num_classes); m.load_state_dict(torch.load(base_ckpt, map_location=DEVICE)); m.to(DEVICE)
        m = manual_prune_threshold(m, thr)
        before = evaluate_and_record(m, test_loader)
        print(f"Before fine-tune (manual {thr}): Acc {before['Accuracy']:.4f} AUC {before['AUC']:.4f}")
        # finetune
        train_model(m, train_loader, val_loader, FINETUNE_EPOCHS)
        after = evaluate_and_record(m, test_loader)
        fname = os.path.join(SAVE_DIR, f"{ds_name}_resnet50_manual_thr{thr}_finetuned.pth")
        save_model(fname, m)
        print(f"After fine-tune (manual {thr}): Acc {after['Accuracy']:.4f} AUC {after['AUC']:.4f}")
        results.append({"Method":f"manual_thr{thr}_before", **before, "ModelPath": ""})
        results.append({"Method":f"manual_thr{thr}_after", **after, "ModelPath": fname})

    # Global 70%
    print(f"\n--- {ds_name} Global {GLOBAL_SPARSITY*100:.0f}% prune ---")
    g = build_model(num_classes); g.load_state_dict(torch.load(base_ckpt, map_location=DEVICE)); g.to(DEVICE)
    g = global_prune_percentile(g, GLOBAL_SPARSITY)
    before = evaluate_and_record(g, test_loader)
    print(f"Global before finetune: Acc {before['Accuracy']:.4f} AUC {before['AUC']:.4f}")
    train_model(g, train_loader, val_loader, FINETUNE_EPOCHS)
    after = evaluate_and_record(g, test_loader)
    fname = os.path.join(SAVE_DIR, f"{ds_name}_resnet50_global_{int(GLOBAL_SPARSITY*100)}_finetuned.pth")
    save_model(fname, g)
    results.append({"Method":"global_before", **before, "ModelPath": ""})
    results.append({"Method":"global_after", **after, "ModelPath": fname})

    # Layerwise 70%
    print(f"\n--- {ds_name} Layerwise {LAYERWISE_SPARSITY*100:.0f}% prune ---")
    l = build_model(num_classes); l.load_state_dict(torch.load(base_ckpt, map_location=DEVICE)); l.to(DEVICE)
    l = layerwise_prune_percentile(l, LAYERWISE_SPARSITY)
    before = evaluate_and_record(l, test_loader)
    print(f"Layerwise before finetune: Acc {before['Accuracy']:.4f} AUC {before['AUC']:.4f}")
    train_model(l, train_loader, val_loader, FINETUNE_EPOCHS)
    after = evaluate_and_record(l, test_loader)
    fname = os.path.join(SAVE_DIR, f"{ds_name}_resnet50_layerwise_{int(LAYERWISE_SPARSITY*100)}_finetuned.pth")
    save_model(fname, l)
    results.append({"Method":"layerwise_before", **before, "ModelPath": ""})
    results.append({"Method":"layerwise_after", **after, "ModelPath": fname})

    # Block-level (stage) pruning
    for stage in BLOCKS_TO_PRUNE:
        print(f"\n--- {ds_name} Block-level prune stage={stage} ---")
        b = build_model(num_classes); b.load_state_dict(torch.load(base_ckpt, map_location=DEVICE)); b.to(DEVICE)
        b = block_prune_stage(b, stage)
        before = evaluate_and_record(b, test_loader)
        print(f"Stage {stage} before finetune: Acc {before['Accuracy']:.4f} AUC {before['AUC']:.4f}")
        train_model(b, train_loader, val_loader, FINETUNE_EPOCHS)
        after = evaluate_and_record(b, test_loader)
        fname = os.path.join(SAVE_DIR, f"{ds_name}_resnet50_block_{stage}_finetuned.pth")
        save_model(fname, b)
        results.append({"Method":f"block_{stage}_before", **before, "ModelPath": ""})
        results.append({"Method":f"block_{stage}_after", **after, "ModelPath": fname})

    # Write CSV
    csv_path = os.path.join(SAVE_DIR, f"{ds_name}_pruning_summary.csv")
    keys = ["Method","Zeros","Total","Nonzero","FLOPs","MemoryMB","PowerProxy","InferenceTime","Loss","Accuracy","AUC","ModelPath"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in keys})
    print("Wrote summary:", csv_path)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    set_seed(SEED)
    print("Running on device:", DEVICE)
    print("SKLEARN available for AUC:", SKLEARN)

    print("\n=== DermamNIST ===")
    run_dataset(DERMA_PATH)

    print("\n=== PathMNIST ===")
    run_dataset(PATH_PATH)

    print("\nAll done.")
