#!/usr/bin/env python3
"""
hessian_prune_vit.py

Loads a DINOv2 pre-trained and fine-tuned ViT model from baseline_models directory,
applies Hessian-Aware Saliency-based pruning as per CVPR 2023 paper,
fine-tunes the pruned model for 3 epochs, and reports test accuracy, AUC, FLOPs, etc.
Processes models based on _baseline.pth files in baseline_models folder.

Requires: torch, torchvision, numpy, thop (for FLOPs), scikit-learn (for AUC)
Install thop: pip install thop
DINOv2 loaded via torch.hub.
"""
import os
import time
import random
import csv
import numpy as np
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from thop import profile, clever_format

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN = True
except ImportError:
    SKLEARN = False

# -------------------------
# Config
# -------------------------
DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/pruned_models_hessian"
TRIALS_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/baseline_models"

PRUNING_RATIO = 0.5  # Target 50% parameter reduction
FINETUNE_EPOCHS = 3
BATCH_SIZE = 32
LR = 5e-5
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 20
SEED = 42
HESSIAN_H = 1e-4  # Finite difference step for Hessian approximation
LATENCY_ETA = 0.1  # Weight for latency-aware regularization

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
# Model
# -------------------------
class ViTClassifier(nn.Module):
    def __init__(self, backbone, num_classes, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone.to(DEVICE)  # Move backbone to DEVICE
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
            test_output = self.backbone(test_input)[:, 0]
            feature_dim = test_output.shape[-1]
        self.head = nn.Linear(feature_dim, num_classes)
        print(f"Linear head: in_features={self.head.in_features}, out_features={self.head.out_features}")
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)[:, 0]  # CLS token
        x = self.head(x)
        return x

def build_model(num_classes: int, freeze_backbone=False) -> nn.Module:
    print("Building DINOv2 ViT-S/14 backbone with linear head...")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = ViTClassifier(backbone, num_classes, freeze_backbone)
    return model.to(DEVICE)

def load_baseline_model(model: nn.Module, ds_name: str):
    pretrained_path = os.path.join(TRIALS_DIR, f"{ds_name}_baseline.pth")
    print(f"Loading baseline weights from {pretrained_path}...")
    pretrained_dict = torch.load(pretrained_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(pretrained_dict, strict=False)
    print("Loaded baseline model successfully.")

# -------------------------
# Hessian-based Pruning
# -------------------------
def compute_hessian_importance(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> List[Dict]:
    """
    Compute Hessian-based importance scores for prunable structures (Q, K, V, Proj, FC1, FC2).
    Returns a list of dictionaries with layer name, parameter index, and importance score.
    """
    model.eval()
    importance_scores = []
    
    # Collect gradients for a single batch
    images, labels = next(iter(loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    
    # Original loss and gradients
    model.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    grad_dict = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    
    # Hessian approximation via finite difference
    h = HESSIAN_H
    for name, param in model.named_parameters():
        if 'weight' in name and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'out_proj' in name or 'fc1' in name or 'fc2' in name):
            importance = 0.0
            param_grad = grad_dict.get(name, torch.zeros_like(param))
            for idx in range(param.size(0)):  # Iterate over output dimension
                # Perturb parameter
                perturbation = torch.zeros_like(param)
                perturbation[idx] = h
                param.data.add_(perturbation)
                
                # Compute loss with perturbation
                model.zero_grad()
                outputs_perturbed = model(images)
                loss_perturbed = criterion(outputs_perturbed, labels)
                loss_perturbed.backward()
                
                # Compute importance score
                grad_perturbed = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
                grad_diff = (grad_perturbed.get(name, torch.zeros_like(param)) - param_grad) / h
                importance = (grad_diff[idx] * param[idx]).pow(2).sum()
                
                importance_scores.append({
                    'layer': name,
                    'index': idx,
                    'score': importance.item()
                })
                
                # Restore parameter
                param.data.sub_(perturbation)
    
    return importance_scores

def estimate_latency(model: nn.Module) -> float:
    """Dummy latency estimation based on parameter count (simplified)."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 1e-6  # Scale to simulate latency in seconds

def prune_structure(model: nn.Module, importance_scores: List[Dict], pruning_ratio: float):
    """Prune structures based on sorted importance scores with latency-aware regularization."""
    total_params = sum(p.numel() for p in model.backbone.parameters())
    target_params = total_params * (1 - pruning_ratio)
    current_params = total_params
    
    # Sort scores by importance
    sorted_scores = sorted(importance_scores, key=lambda x: x['score'] - LATENCY_ETA * estimate_latency(model))
    
    for score in sorted_scores:
        if current_params <= target_params:
            break
        layer_name = score['layer']
        idx = score['index']
        module = dict(model.backbone.named_modules())[layer_name.replace('.weight', '')]
        
        # Prune based on layer type
        if 'q_proj' in layer_name or 'k_proj' in layer_name or 'v_proj' in layer_name or 'out_proj' in layer_name:
            # Prune attention projections
            module.weight.data[idx] = 0
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data[idx] = 0
            current_params -= module.weight.size(1)
        elif 'fc1' in layer_name:
            # Prune FC1 and adjust FC2
            module.weight.data[idx] = 0
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data[idx] = 0
            next_fc = layer_name.replace('fc1', 'fc2')
            next_module = dict(model.backbone.named_modules())[next_fc.replace('.weight', '')]
            next_module.weight.data[:, idx] = 0
            current_params -= (module.weight.size(1) + next_module.weight.size(0))
        elif 'fc2' in layer_name:
            # Prune FC2
            module.weight.data[:, idx] = 0
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data[idx] = 0
            current_params -= module.weight.size(0)
    
    print(f"Pruned to {current_params/total_params*100:.2f}% of original parameters.")

# -------------------------
# Train / Eval
# -------------------------
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
def run_dataset(npz_path: str, ds_name: str, freeze_backbone=False):
    train_loader, val_loader, test_loader, num_classes, _ = make_loaders(npz_path)
    print(f"\n=== Running dataset: {ds_name} ===\n")

    model = build_model(num_classes, freeze_backbone=freeze_backbone)
    load_baseline_model(model, ds_name)

    # Initial metrics
    orig_macs, orig_params = count_params_flops(model)
    print(f"Original MACs: {orig_macs}M, Params: {orig_params}M")

    # Compute Hessian-based importance scores
    print("\n--- Computing Hessian-based importance scores ---")
    importance_scores = compute_hessian_importance(model, train_loader, criterion)

    # Apply pruning
    print("\n--- Applying Hessian-based pruning ---")
    prune_structure(model, importance_scores, PRUNING_RATIO)

    # Fine-tuning
    print("\n--- Fine-tuning ---")
    train_model(model, train_loader, val_loader, FINETUNE_EPOCHS)

    # Evaluate and save
    final_loss, final_acc, final_auc = evaluate_model(model, test_loader)
    final_macs, final_params = count_params_flops(model)
    print(f"Final Test → Loss {final_loss:.4f} Acc {final_acc:.4f} AUC {final_auc:.4f}")
    print(f"Final MACs: {final_macs}M, Params: {final_params}M")

    save_path = os.path.join(SAVE_DIR, f"{ds_name}_hessian_pruned.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved Hessian pruned model to {save_path}")

    csv_path = os.path.join(TRIALS_DIR, f"{ds_name}_hessian_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "method", "pruning_ratio", "macs_m", "params_m", "loss", "acc", "auc"])
        writer.writeheader()
        writer.writerow({
            "dataset": ds_name,
            "method": "hessian_pruned",
            "pruning_ratio": PRUNING_RATIO,
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

    # Collect model files from baseline_models directory
    model_files = [f for f in os.listdir(TRIALS_DIR) if f.endswith("_baseline.pth")]
    print("\nFound models:", model_files)

    for model_file in model_files:
        # Extract dataset name from model file (e.g., 'pathmnist_224' from 'pathmnist_224_baseline.pth')
        ds_name = model_file.replace("_baseline.pth", "")
        npz_path = os.path.join(DATASET_DIR, f"{ds_name}.npz")
        if not os.path.exists(npz_path):
            print(f"Warning: Dataset file {npz_path} not found, skipping {ds_name}")
            continue
        run_dataset(npz_path, ds_name, freeze_backbone=False)

    print("\nAll done.")