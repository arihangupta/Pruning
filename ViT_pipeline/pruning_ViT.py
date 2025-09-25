#!/usr/bin/env python3
"""
prune_and_fine_tune_from_dino.py

Loads the DINO pretrained model from the trials directory, applies global structural pruning with Hessian-aware saliency,
fine-tunes the pruned model for 10 epochs, and reports test accuracy, AUC, etc.
Mimics the provided CNN script structure.
Reduces model size by pruning embedding, heads, QK, V, and MLP dimensions.
Memory optimizations for Hessian calculation.

Requires: torch, torchvision, numpy, thop (for FLOPs), (scikit-learn for AUC)
Install thop: pip install thop
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
TRIALS_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/trials"

PRUNE_ITERATIONS = 5
PRUNE_RATIO = 0.1
FINETUNE_EPOCHS = 10
FINETUNE_EPOCHS_PER_PRUNE = 2
BATCH_SIZE = 32
SALIENCY_BATCH_SIZE = 4  # Reduced for memory
SALIENCY_NUM_BATCHES = 5  # Reduced for memory
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

    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE, train=True)
    val_ds   = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE, train=False)
    test_ds  = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, train=False)

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
                    dims[name + "_mlp_out"] = module.weight.shape[0]
                elif "fc2" in name:
                    dims[name + "_mlp_out"] = module.weight.shape[1]
        return dims

    def forward(self, x):
        x = self.backbone(x)[:, 0]  # CLS token
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
    pretrained_dict = torch.load(pretrained_path, map_location=DEVICE)
    backbone_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items() if k.startswith("backbone.")}
    model.backbone.load_state_dict(backbone_dict, strict=False)
    print("Loaded DINO pretrained backbone successfully.")

def compute_hessian_saliency(model: nn.Module, loader: DataLoader, criterion) -> Dict[str, float]:
    model.eval()
    hessian_norms = {}
    scaler = GradScaler()
    for name, param in model.named_parameters():
        if param.requires_grad and "backbone" in name and "weight" in name:
            param.grad = None
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with autocast(dtype=torch.bfloat16):
            outputs = checkpoint.checkpoint(model, images)  # Checkpoint for memory
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward(create_graph=True)
        for name, param in model.named_parameters():
            if param.requires_grad and "backbone" in name and "weight" in name:
                grad = param.grad
                hessian_diag = autograd.grad(loss, param, grad_outputs=grad, retain_graph=True)[0]
                norm = hessian_diag.norm().item()
                hessian_norms[name] = hessian_norms.get(name, 0) + norm
        model.zero_grad()
        torch.cuda.empty_cache()  # Clear cache per batch
        if i >= SALIENCY_NUM_BATCHES - 1:
            break
    for name in hessian_norms:
        hessian_norms[name] /= SALIENCY_NUM_BATCHES
    return hessian_norms

def estimate_latency_reduction(model: nn.Module, orig_dims: Dict, new_dims: Dict) -> float:
    total_ops = sum([orig_dims.get(k, 0) for k in orig_dims if "mlp" in k or "qk" in k or "h" in k])
    new_ops = sum([new_dims.get(k, 0) for k in new_dims if "mlp" in k or "qk" in k or "h" in k])
    return 1 - (new_ops / max(1, total_ops))

def prune_model(model: nn.Module, prune_ratio: float, hessian_norms: Dict) -> None:
    orig_dims = model.orig_dims.copy()
    new_dims = orig_dims.copy()
    # Latency-aware saliency
    norm_sum = sum(hessian_norms.values())
    saliency = {k: v / norm_sum * (1 + ETA * estimate_latency_reduction(model, orig_dims, new_dims)) for k, v in hessian_norms.items()}
    # Sort and prune lowest saliency
    sorted_saliency = sorted(saliency.items(), key=lambda x: x[1])
    total_prune = int(len(sorted_saliency) * prune_ratio)
    pruned_params = sorted_saliency[:total_prune]
    for param_name, _ in pruned_params:
        # Prune dimensions based on param type
        layer = param_name.split('.')[1]
        if "qkv" in param_name:
            new_dims[f"{layer}_qk"] = max(1, int(new_dims.get(f"{layer}_qk", 0) * 0.9))
            new_dims[f"{layer}_h"] = max(1, int(new_dims.get(f"{layer}_h", 0) * 0.9))
        elif "fc1" in param_name:
            new_dims[f"{layer}_mlp_in"] = max(1, int(new_dims.get(f"{layer}_mlp_in", 0) * 0.9))
        elif "fc2" in param_name:
            new_dims[f"{layer}_mlp_out"] = max(1, int(new_dims.get(f"{layer}_mlp_out", 0) * 0.9))
    # Apply new dimensions to model layers
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Linear):
            if "qkv" in name:
                dim_qk = new_dims.get(f"{name.split('.')[1]}_qk", module.weight.shape[1])
                dim_h = new_dims.get(f"{name.split('.')[1]}_h", module.weight.shape[0] // 384)
                module.weight.data = module.weight.data[:dim_h * 384, :dim_qk]
                module.bias.data = module.bias.data[:dim_h * 384] if module.bias is not None else None
                module.out_features = dim_h * 384
                module.in_features = dim_qk
            elif "fc1" in name:
                dim_in = new_dims.get(f"{name.split('.')[1]}_mlp_in", module.weight.shape[1])
                module.weight.data = module.weight.data[:, :dim_in]
                module.bias.data = module.bias.data[:dim_in] if module.bias is not None else None
                module.in_features = dim_in
            elif "fc2" in name:
                dim_out = new_dims.get(f"{name.split('.')[1]}_mlp_out", module.weight.shape[0])
                module.weight.data = module.weight.data[:dim_out, :]
                module.bias.data = module.bias.data[:dim_out] if module.bias is not None else None
                module.out_features = dim_out

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
            with autocast(dtype=torch.bfloat16):
                outputs = checkpoint.checkpoint(model, images)  # Checkpoint for memory
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
    input_tensor = torch.randn(*input_size).to(DEVICE)
    macs, params = profile(model, inputs=(input_tensor,))
    macs, params = clever_format([macs, params], "%.3f")
    return float(macs.split()[0]), float(params.split()[0])

# -------------------------
# Dataset runner
# -------------------------
def run_dataset(npz_path: str, freeze_backbone=False):
    train_loader, val_loader, test_loader, num_classes, ds_name = make_loaders(npz_path)
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
        hessian_norms = compute_hessian_saliency(model, train_loader, criterion)
        prune_model(model, PRUNE_RATIO, hessian_norms)
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