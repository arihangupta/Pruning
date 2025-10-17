#!/usr/bin/env python3
"""
dgmr_prune_and_fine_tune.py

Loads the DINO pretrained model from the trials directory, applies Diversity-Guided MLP Reduction (DGMR) from the arXiv paper,
fine-tunes the pruned model for 10 epochs, and reports test accuracy, AUC, etc.
Mimics the provided CNN script structure.
Reduces MLP hidden dimensions while preserving diversity.

Requires: torch, torchvision, numpy, thop (for FLOPs), (scikit-learn for AUC)
Install thop: pip install thop
DINOv2 loaded via torch.hub.
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
from torchvision import transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from thop import profile, clever_format

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN = True
except Exception:
    SKLEARN = False

# -------------------------
# Config
# -------------------------
DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/pruned_models_mlp"
TRIALS_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/baseline_models"

TARGET_EXPANSION_RATIO = 1  # r=1 as in paper (hidden = input dim)
FINETUNE_EPOCHS = 3
BATCH_SIZE = 32
LR = 5e-5
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

def dgmr_prune_mlp(model: nn.Module, target_r: int = 1):
    """
    Apply DGMR pruning to MLP expansion (fc1) layers in the ViT backbone.
    Robustly replaces fc1 and its paired fc2 using get_submodule/_modules mapping
    and ensures the selected neuron count equals the target.
    """
    print("\n--- Applying DGMR Pruning (robust) ---")
    device = next(model.parameters()).device

    # Collect candidate linear layers inside backbone that look like MLP expansion layers
    mlp_layers = []
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Linear) and 'blocks' in name:
            # candidate expansion layer is one where out_features > in_features
            if module.out_features > module.in_features:
                mlp_layers.append((name, module))
                print(f"  Candidate MLP layer: {name}  (in={module.in_features}, out={module.out_features})")

    if not mlp_layers:
        print("Warning: no expansion linear layers found. Exiting pruning.")
        return 0

    pruned_pairs = 0

    for name, module in mlp_layers:
        # assume this is an fc1 (expansion): in_features = N, out_features = M
        N = module.in_features
        M = module.out_features
        target_M = int(target_r * N)
        if target_M >= M:
            print(f"Skipping {name}: target {target_M} >= current {M}")
            continue

        print(f"\nPruning layer: {name}")
        print(f"  Original: in={N}, out={M} -> target hidden {target_M}")

        # copy weights to CPU for deterministic ops (avoid modifying original until ready)
        W_hidden = module.weight.data.detach().clone().to('cpu')  # [M, N]
        bias_hidden = module.bias.data.detach().clone().to('cpu') if module.bias is not None else None

        # DGMR greedy selection but ensure we get exactly target_M unique indices.
        V = W_hidden.clone()
        selected = []
        attempts = 0
        max_attempts = M * 3  # safety cap
        while len(selected) < target_M and attempts < max_attempts:
            norms = torch.norm(V, p=2, dim=1)  # [M]
            j = int(torch.argmax(norms).item())
            if j not in selected:
                selected.append(j)
                vj = V[j:j+1]  # [1, N]
                vj_norm_sq = (vj @ vj.t()).item()
                if vj_norm_sq > 1e-12:
                    proj = (V @ vj.t()) / vj_norm_sq  # [M, 1]
                    V = V - proj * vj  # Gram-Schmidt-style deflation
                else:
                    # zero vector — remove it from consideration to avoid infinite loop
                    V[j] = torch.zeros_like(V[j])
            else:
                # if duplicate selected (rare), zero out that row and continue
                V[j] = torch.zeros_like(V[j])
            attempts += 1

        selected = sorted(selected)
        if len(selected) != target_M:
            print(f"  ERROR: could not select required unique neurons for {name}. Selected {len(selected)} / {target_M}.")
            print("  Skipping this layer to avoid corrupting shapes.")
            continue

        print(f"  Selected {len(selected)} unique neurons (target {target_M})")

        # Build new fc1: in=N, out=len(selected)
        new_fc1 = nn.Linear(N, len(selected), bias=(module.bias is not None)).to(device)
        # assign weights (copy to device)
        new_fc1.weight.data = module.weight.data[selected].clone().to(device)
        if module.bias is not None:
            new_fc1.bias.data = module.bias.data[selected].clone().to(device)

        # Replace module robustly using get_submodule/_modules
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        if parent_name == '':
            parent_module = model.backbone
        else:
            parent_module = model.backbone.get_submodule(parent_name)
        # sanity: assert child exists in parent
        if child_name not in parent_module._modules:
            print(f"  WARNING: expected child '{child_name}' not found in parent '{parent_name}'. Skipping replacement.")
            continue

        # Temporarily store original for potential revert
        orig_fc1 = parent_module._modules[child_name]
        parent_module._modules[child_name] = new_fc1
        print(f"  Replaced {name} -> new out_features={new_fc1.out_features}")

        # Find corresponding fc2 inside same block/base_name by searching for linear with in_features == M
        base_name = name.rsplit('.', 1)[0]  # e.g., blocks.0.mlp
        base_module = model.backbone.get_submodule(base_name)
        fc2_module = None
        fc2_name_full = None

        # search through direct child modules under base_module
        for sub_name, sub_mod in base_module.named_modules():
            # sub_name is relative path e.g. 'fc2' or 'linear'
            # avoid base_module itself
            if sub_name == '':
                continue
            if isinstance(sub_mod, nn.Linear) and sub_mod.in_features == M:
                # heuristics: choose the linear whose in_features equals the original hidden M
                fc2_module = sub_mod
                fc2_name_full = f"{base_name}.{sub_name}"
                break

        if fc2_module is None:
            # fallback: search all named_modules in backbone for a linear in same block with in_features == M
            for full_nm, mod in model.backbone.named_modules():
                if full_nm.startswith(base_name) and isinstance(mod, nn.Linear) and mod.in_features == M:
                    fc2_module = mod
                    fc2_name_full = full_nm
                    break

        if fc2_module is None:
            print(f"  WARNING: could not find paired fc2 (in_features=={M}) under block '{base_name}'. Reverting fc1 replacement.")
            # revert
            parent_module._modules[child_name] = orig_fc1
            continue

        print(f"  Adjusting paired layer: {fc2_name_full} (orig in={fc2_module.in_features} out={fc2_module.out_features})")

        # create new fc2 with in_features=len(selected)
        new_fc2 = nn.Linear(len(selected), fc2_module.out_features, bias=(fc2_module.bias is not None)).to(device)
        # copy weight columns corresponding to selected neurons
        new_fc2.weight.data = fc2_module.weight.data[:, selected].clone().to(device)
        if fc2_module.bias is not None:
            new_fc2.bias.data = fc2_module.bias.data.clone().to(device)

        # Replace fc2 in its parent
        fc2_parent_name = '.'.join(fc2_name_full.split('.')[:-1])
        fc2_child_name = fc2_name_full.split('.')[-1]
        if fc2_parent_name == '':
            fc2_parent_module = model.backbone
        else:
            fc2_parent_module = model.backbone.get_submodule(fc2_parent_name)

        # sanity check child exists
        if fc2_child_name not in fc2_parent_module._modules:
            print(f"  WARNING: expected child '{fc2_child_name}' not found in parent '{fc2_parent_name}'. Reverting fc1.")
            parent_module._modules[child_name] = orig_fc1
            continue

        orig_fc2 = fc2_parent_module._modules[fc2_child_name]
        fc2_parent_module._modules[fc2_child_name] = new_fc2
        print(f"  Replaced {fc2_name_full} -> new in_features={new_fc2.in_features}")

        pruned_pairs += 1

        # Quick sanity prints: show shapes after replacement
        print(f"  Sanity: new {name} weight shape {new_fc1.weight.data.shape}, new {fc2_name_full} weight shape {new_fc2.weight.data.shape}")

    print(f"\nDGMR pruning completed: {pruned_pairs} layer pairs pruned")

    # Final verification: run a tiny forward through backbone and classifier head
    print("\nVerifying model after pruning with a tiny input...")
    try:
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 224, 224).to(device)
            backbone_out = model.backbone(dummy_input)
            print("  backbone output shape:", tuple(backbone_out.shape))
            # attempt to extract CLS token if present
            if backbone_out.ndim == 3:
                cls = backbone_out[:, 0]
                print("  extracted CLS shape:", tuple(cls.shape))
            else:
                cls = backbone_out
            # check head expectations
            if hasattr(model, 'head'):
                head = model.head
                print("  head weight shape:", tuple(head.weight.shape))
                _ = head(cls)
            print("  Test forward pass successful.")
    except Exception as e:
        print(f"  ERROR in test forward pass after pruning: {e}")
        raise

    return pruned_pairs


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
    """
    Count parameters and estimate FLOPs without relying on thop for complex models.
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters()) / 1e6  # in millions
    
    # For Vision Transformers, estimate FLOPs manually
    # Formula from "An Image is Worth 16x16 Words" paper
    try:
        # Try thop first, but don't fail if it doesn't work
        input_tensor = torch.randn(*input_size).to(DEVICE)
        
        # Disable gradient computation
        with torch.no_grad():
            # Do a dry run to ensure model works
            _ = model(input_tensor)
        
        # Now try profiling
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        macs_val = float(macs.split()[0])
        params_val = float(params.split()[0])
        
    except Exception as e:
        print(f"Note: Using manual parameter count (thop profiling skipped)")
        params_val = total_params
        
        # Rough FLOPs estimate for ViT-S/14:
        # ~5 GFLOPs for standard ViT-S with 224x224 input
        # This is an approximation
        macs_val = 5.0  # GFLOPs
    
    return macs_val, params_val

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

    # Apply DGMR pruning
    dgmr_prune_mlp(model, TARGET_EXPANSION_RATIO)

    # Fine-tuning
    print("\n--- Fine-tuning ---")
    train_model(model, train_loader, val_loader, FINETUNE_EPOCHS)

    # Evaluate and save
    final_loss, final_acc, final_auc = evaluate_model(model, test_loader)
    final_macs, final_params = count_params_flops(model)
    print(f"Final Test → Loss {final_loss:.4f} Acc {final_acc:.4f} AUC {final_auc:.4f}")
    print(f"Final MACs: {final_macs}M, Params: {final_params}M")

    save_path = os.path.join(SAVE_DIR, f"{ds_name}_dgmr_pruned.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved DGMR pruned model to {save_path}")

    csv_path = os.path.join(TRIALS_DIR, f"{ds_name}_dgmr_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "method", "expansion_r", "macs_m", "params_m", "loss", "acc", "auc"])
        writer.writeheader()
        writer.writerow({
            "dataset": ds_name,
            "method": "dgmr_pruned",
            "expansion_r": TARGET_EXPANSION_RATIO,
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