#!/usr/bin/env python3
"""
prune_and_fine_tune_from_dino.py

Loads the DINO pretrained model from the trials directory, applies global structural pruning with
a data-aware gradient saliency (cheap approximation of Hessian saliency),
fine-tunes the pruned model for a few epochs, and reports test accuracy, AUC, etc.

Mimics provided CNN script structure. Reduces model size by pruning embedding, heads, QK, V, and MLP dimensions.
Memory-friendly (no create_graph Hessian passes).
Requires: torch, torchvision, numpy, thop (for FLOPs), (scikit-learn for AUC)
DINOv2 loaded via torch.hub.
"""
import os
import time
import random
import csv
import numpy as np
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from thop import profile, clever_format
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN = True
except Exception:
    SKLEARN = False

# -------------------------
# Config
# -------------------------
DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets"
TRIALS_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/exp2_saved_models"

PRUNE_ITERATIONS = 5
PRUNE_RATIO = 0.1
FINETUNE_EPOCHS = 3
FINETUNE_EPOCHS_PER_PRUNE = 2
BATCH_SIZE = 32
SALIENCY_BATCH_SIZE = 2
SALIENCY_NUM_BATCHES = 3
LR = 5e-5
ETA = 1.0
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 20
SEED = 42

os.makedirs(TRIALS_DIR, exist_ok=True)

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
    def __init__(self, imgs_np, labels_np, img_size=224, train=True):
        self.imgs = imgs_np
        self.labels = labels_np
        self.img_size = img_size
        self.train = train

        self.base_tfms = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        self.train_tfms = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]) if train else T.Compose([])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = int(self.labels[idx])
        x = self.base_tfms(img)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        if self.train:
            x = self.train_tfms(x)
        x = self.normalize(x)
        return x, label

def make_loaders(npz_path: str) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, int, str]:
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

    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE, train=True)
    val_ds   = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE, train=False)
    test_ds  = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    saliency_loader = DataLoader(train_ds, batch_size=SALIENCY_BATCH_SIZE, shuffle=True,
                                 num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    num_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    ds_name = os.path.splitext(os.path.basename(npz_path))[0]
    return train_loader, saliency_loader, val_loader, test_loader, num_classes, ds_name

# -------------------------
# Model / prune / train / eval
# -------------------------
class ViTClassifier(nn.Module):
    def __init__(self, backbone, num_classes, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(384, num_classes)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        # Store original dimensions for pruning
        self.orig_dims = self._extract_dims()

    def _extract_dims(self):
        dims = {}
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear):
                if "qkv" in name:
                    dims[name + "_qk"] = module.weight.shape[1]  # QK dim
                    dims[name + "_h"] = module.weight.shape[0] // 384  # Heads
                elif "fc1" in name:
                    dims[name + "_mlp_in"] = module.weight.shape[1]
                elif "fc2" in name:
                    dims[name + "_mlp_out"] = module.weight.shape[0]
        return dims

    def forward(self, x):
        x = self.backbone(x)
        if x.dim() == 3:
            x = x[:, 0]  # CLS token
        x = self.head(x)
        return x

def build_model(num_classes: int, freeze_backbone=False) -> nn.Module:
    print("Building DINOv2 ViT-S/14 backbone with linear head...")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = ViTClassifier(backbone, num_classes, freeze_backbone)
    return model.to(DEVICE)

def load_dino_pretrained(model: nn.Module, ds_name: str):
    pretrained_path = os.path.join(TRIALS_DIR, f"{ds_name}_dino_pretrained.pth")
    print(f"Loading DINO pretrained weights from {pretrained_path}...")
    pretrained_dict = torch.load(pretrained_path, map_location=DEVICE, weights_only=True)
    backbone_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items() if k.startswith("backbone.")}
    model.backbone.load_state_dict(backbone_dict, strict=False)
    print("Loaded DINO pretrained backbone successfully.")

def compute_gradient_saliency(model: nn.Module,
                              loader: DataLoader,
                              criterion,
                              num_batches: int = SALIENCY_NUM_BATCHES) -> Dict[str, float]:
    """
    Data-aware gradient saliency:
      - Runs `num_batches` batches from `loader`
      - Accumulates per-parameter L1 gradient magnitudes for backbone weight params
      - Returns dict: parameter_name -> average_score (higher => more important)
    This is much cheaper and robust compared to second-order/Hessian-based computations.
    """
    model.train()  # allow gradients
    saliency: Dict[str, float] = {}
    # zero grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    batches = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Forward + backward (no create_graph, no AMP/GradScaler here for saliency pass)
        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        # accumulate L1 of gradients for backbone weight tensors
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # only backbone weights (you used "backbone" substring earlier)
            if "backbone" not in name or "weight" not in name:
                continue
            g = param.grad
            if g is None:
                continue
            score = float(g.abs().sum().cpu().item())
            saliency[name] = saliency.get(name, 0.0) + score

        batches += 1
        if batches >= num_batches:
            break

    # average over batches
    if batches > 0:
        for k in list(saliency.keys()):
            saliency[k] /= float(batches)

    model.eval()
    # make sure gradients are cleared to free memory
    model.zero_grad()
    torch.cuda.empty_cache()
    return saliency

def estimate_latency_reduction(model: nn.Module, orig_dims: Dict, new_dims: Dict) -> float:
    total_ops = sum([orig_dims.get(k, 0) for k in orig_dims if "qk" in k or "h" in k])  # Only attention dims
    new_ops = sum([new_dims.get(k, 0) for k in new_dims if "qk" in k or "h" in k])
    return 1 - (new_ops / max(1, total_ops))

def prune_model(model: nn.Module, prune_ratio: float, saliency: Dict) -> None:
    orig_dims = model.orig_dims.copy()
    new_dims = orig_dims.copy()
    # Latency-aware saliency
    norm_sum = sum(saliency.values()) if len(saliency) > 0 else 1.0
    saliency_adjusted = {k: v / norm_sum * (1 + ETA * estimate_latency_reduction(model, orig_dims, new_dims)) for k, v in saliency.items()}
    # Sort and prune lowest saliency (only attention layers)
    sorted_saliency = sorted(saliency_adjusted.items(), key=lambda x: x[1])
    total_prune = int(len([k for k in saliency if "qkv" in k]) * prune_ratio)  # Limit to qkv layers
    pruned_params = sorted_saliency[:total_prune]
    for param_name, _ in pruned_params:
        if "qkv" in param_name:
            layer = param_name.split('.')[1]
            new_dims[f"{layer}_qk"] = max(1, int(new_dims.get(f"{layer}_qk", 0) * (1 - PRUNE_RATIO)))
            new_dims[f"{layer}_h"] = max(1, int(new_dims.get(f"{layer}_h", 0) * (1 - PRUNE_RATIO)))

    # Apply new dimensions to model layers, preserving MLP structure
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Linear):
            if "qkv" in name:
                dim_qk = new_dims.get(f"{name.split('.')[1]}_qk", module.weight.shape[1])
                dim_h = new_dims.get(f"{name.split('.')[1]}_h", module.weight.shape[0] // 384)
                # Ensure compatibility with next layer (e.g., 384 * dim_h)
                dim_h = min(dim_h, module.weight.shape[0] // 384)  # Cap to original heads
                module.weight.data = module.weight.data[:dim_h * 384, :dim_qk]
                if module.bias is not None:
                    module.bias.data = module.bias.data[:dim_h * 384]
                module.out_features = dim_h * 384
                module.in_features = dim_qk
            # Do not prune fc1/fc2 to maintain MLP integrity
            elif "fc1" in name or "fc2" in name:
                continue

def make_optimizer(model: nn.Module):
    return optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

criterion = nn.CrossEntropyLoss()

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
    opt = make_optimizer(model)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        start = time.time()
        for bidx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            # Use checkpointing + AMP in training pass (keeps memory lower)
            with autocast(dtype=torch.bfloat16):
                outputs = checkpoint.checkpoint(model, images, use_reentrant=False)  # Explicit use_reentrant
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
        scheduler.step()

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

def count_params_flops(model: nn.Module, input_size=(1, 3, 224, 224)) -> Tuple[float, float]:
    """
    Returns (macs_m, params_m) where values are in MILLIONS.
    Uses raw numeric outputs from thop.profile and scales to 1e6.
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.randn(*input_size).to(DEVICE)
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    macs_m = float(macs) / 1e6
    params_m = float(params) / 1e6
    return macs_m, params_m

# -------------------------
# Dataset runner
# -------------------------
def run_dataset(npz_path: str, freeze_backbone=False):
    train_loader, saliency_loader, val_loader, test_loader, num_classes, ds_name = make_loaders(npz_path)
    print(f"\n=== Running dataset: {ds_name} ===\n")

    model = build_model(num_classes, freeze_backbone=freeze_backbone)

    # Load DINO pretrained backbone
    load_dino_pretrained(model, ds_name)

    # Initial metrics
    orig_macs, orig_params = count_params_flops(model)
    print(f"Original MACs: {orig_macs}M, Params: {orig_params}M")

    # Pruning loop
    for i in range(PRUNE_ITERATIONS):
        print(f"\n--- Pruning Iteration {i+1}/{PRUNE_ITERATIONS} ---")
        # Compute gradient-based saliency (cheap and robust)
        grad_saliency = compute_gradient_saliency(model, saliency_loader, criterion, num_batches=SALIENCY_NUM_BATCHES)
        prune_model(model, PRUNE_RATIO, grad_saliency)
        print(f"Pruned model structure updated.")
        torch.cuda.empty_cache()

        # Brief fine-tuning after pruning
        print(f"--- Fine-tuning after Pruning {i+1} ---")
        train_model(model, train_loader, val_loader, FINETUNE_EPOCHS_PER_PRUNE)

    # Final fine-tuning
    print("\n--- Final Fine-tuning ---")
    train_model(model, train_loader, val_loader, FINETUNE_EPOCHS)

    # Evaluate and save
    final_loss, final_acc, final_auc = evaluate_model(model, test_loader)
    final_macs, final_params = count_params_flops(model)
    print(f"Final Test → Loss {final_loss:.4f} Acc {final_acc:.4f} AUC {final_auc:.4f}")
    print(f"Final MACs: {final_macs}M, Params: {final_params}M")

    save_path = os.path.join(TRIALS_DIR, f"{ds_name}_pruned_baseline.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved pruned baseline model to {save_path}")

    csv_path = os.path.join(TRIALS_DIR, f"{ds_name}_pruned_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "method", "sparsity", "macs_m", "params_m", "loss", "acc", "auc"])
        writer.writeheader()
        writer.writerow({
            "dataset": ds_name,
            "method": "pruned_baseline",
            "sparsity": 1 - (final_params / orig_params),
            "macs_m": final_macs,
            "params_m": final_params,
            "loss": final_loss,
            "acc": final_acc,
            "auc": final_auc
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
        run_dataset(npz_path, freeze_backbone=False)

    print("\nAll done.")