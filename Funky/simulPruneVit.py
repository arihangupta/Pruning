"""
Vision Transformer Structured Pruning for MedMNIST with Physical Model Compression

This implementation follows the Hikvision paper methodology with complete deployment conversion:
1. Train with gates to identify important components
2. Convert to physically smaller models (actual compression)
3. Measure energy on deployment models (real speedup)

Four methods compared:
- Baseline: Full ViT trained for 20 epochs
- Progressive Pruning: Gradual pruning during training -> physically smaller model
- Small Model From Scratch: Same dimensions as pruned model, initialized from pretrained,
  trained with baseline hyperparameters (fair comparison to see if pruning helps)
- Enhanced Progressive Pruning: Progressive pruning + LR warmup/rewarm + RandAugment + Mixup

Configuration: gentle_40pct (best performing from hyperparameter tuning)
- 40% target sparsity (keep 60% of dimensions)
- Gentle sparsity schedule: [0.08, 0.16, 0.28, 0.4]
- Low gate L1 weight: 3e-5
- 20 total epochs, no deployment finetuning

Metrics tracked:
- Accuracy, AUC, Precision, Recall, F1
- Parameters, FLOPs, Model Size
- Training & Inference Energy (CodeCarbon)

Features:
- Supports resuming: skips experiments if final .pth files already exist
- Generates summary statistics (mean/std) across trials
- Use organize_vit_outputs.py to sort results into dataset folders

Date: 2026
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import timm
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import copy
import gc
from functools import partial

# Try codecarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("CodeCarbon not available - energy metrics will be NaN")

# ==================== CONFIGURATION ====================

# Paths
DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/simultaneousPruning/vit/FourMethodMadness"

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224

# Training configuration (from gentle_40pct - best performing config)
FIXED_EPOCHS = 20
INITIAL_LR = 1e-4
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-6

# Pruning configuration (from gentle_40pct in ViT_hyper.py)
WARMUP_EPOCHS = 3
EPOCHS_BETWEEN_PRUNES = 4
NUM_PRUNE_STEPS = 4
TARGET_FLOPS_REDUCTION = 0.4  # 40% max compression (keep 60%)
PRUNING_MOMENTUM = 0.9
IMPORTANCE_CAL_BATCHES = 50
LR_REDUCTION_AFTER_PRUNE = 0.85  # Gentler LR reduction

# Gate regularization weight (from gentle_40pct)
GATE_L1_WEIGHT = 3e-5  # Lower L1 weight for gentler pruning

# No separate deployment finetuning - convert mid-training and continue
DEPLOY_FINETUNE_EPOCHS = 0
DEPLOY_FINETUNE_LR = 1e-4

# Use mixed precision training (reduces memory usage significantly)
USE_AMP = True

# ==================== EXPERIMENT CONFIGURATION ====================
# Number of replicates for each (model, method) combination
# Total runs = NUM_TRIALS × len(MODELS_TO_RUN) × 3 methods × len(datasets)
# Default: 3 trials × 2 models × 3 methods × 3 datasets = 54 total runs
NUM_TRIALS = 3

# Models to test (ViT-Tiny and ViT-Base) with model-specific batch sizes
# ViT-Base needs smaller batch size due to larger memory footprint (~86M vs ~5.7M params)
MODEL_CONFIGS = {
    'vit_tiny_patch16_224': {'embed_dim': 192, 'depth': 12, 'num_heads': 3, 'mlp_ratio': 4, 'batch_size': 64},
    'vit_base_patch16_224': {'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'mlp_ratio': 4, 'batch_size': 16},
}

# Which models to run (can be modified via command line)
MODELS_TO_RUN = ['vit_tiny_patch16_224', 'vit_base_patch16_224']

# Datasets to process
DATASETS = ['bloodmnist', 'pathmnist', 'dermamnist']

os.makedirs(SAVE_DIR, exist_ok=True)

PRUNE_EPOCHS_SCHEDULE = [
    WARMUP_EPOCHS + 1 + i * EPOCHS_BETWEEN_PRUNES 
    for i in range(NUM_PRUNE_STEPS)
]

# Gentle sparsity schedule (from gentle_40pct - stops at 40%)
SPARSITY_SCHEDULE = [0.08, 0.16, 0.28, 0.4]  # Cumulative sparsity per step

# ==================== UTILITIES ====================

def set_seed(seed=SEED):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup_memory():
    """Clean up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def get_final_model_path(save_dir, dataset_name, model_name, method_key, trial_num):
    """Get the path to the final model file for a given method and trial."""
    paths = {
        'baseline': f"{dataset_name}_{model_name}_baseline_deploy_trial{trial_num}_final.pth",
        'progressive': f"{dataset_name}_{model_name}_progressive_deploy_trial{trial_num}_final.pth",
        'small_model': f"{dataset_name}_{model_name}_small_model_trial{trial_num}_final.pth",
        'enhanced': f"{dataset_name}_{model_name}_enhanced_deploy_trial{trial_num}_final.pth",
    }
    return os.path.join(save_dir, paths[method_key])

def check_experiment_exists(save_dir, dataset_name, model_name, method_key, trial_num):
    """Check if a final model file already exists for this experiment."""
    path = get_final_model_path(save_dir, dataset_name, model_name, method_key, trial_num)
    return os.path.exists(path), path

def count_parameters(model):
    """Count trainable parameters in millions"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def model_size_mb(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2

def compute_flops_vit(embed_dim, num_heads, depth, mlp_ratio=4, seq_len=197):
    """
    Compute theoretical FLOPs for Vision Transformer
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        depth: Number of transformer blocks
        mlp_ratio: MLP expansion ratio
        seq_len: Sequence length (patches + cls token)
    """
    n = seq_len
    d = embed_dim
    h = num_heads
    mlp_dim = int(embed_dim * mlp_ratio)
    
    # Per layer FLOPs
    # MHSA: QKV projection + attention + output projection
    mhsa_flops = (3 * n * d * d) + (2 * n * n * d) + (n * d * d)
    
    # MLP: two linear layers
    mlp_flops = 2 * n * d * mlp_dim
    
    # Total
    total_flops = depth * (mhsa_flops + mlp_flops)
    
    return total_flops

def get_gate_sparsity(model):
    """Get sparsity statistics for gates"""
    gate_info = {'attn': [], 'mlp': [], 'res': []}
    for name, param in model.named_parameters():
        if 'gate' in name and 'weight' in name:
            w = param.data.cpu()
            if 'attn_gate' in name:
                gate_info['attn'].append(w)
            elif 'hidden_gate' in name:
                gate_info['mlp'].append(w)
            elif 'res_gate' in name:
                gate_info['res'].append(w)

    stats = {}
    for k, v in gate_info.items():
        if v:
            all_w = torch.cat([x.flatten() for x in v])
            stats[k] = (all_w.abs() < 0.01).sum().item() / all_w.numel() * 100
        else:
            stats[k] = 0
    return stats

# ==================== ENERGY TRACKING ====================

def start_energy_tracker(save_dir, project_name):
    """Start energy tracking"""
    if not CODECARBON_AVAILABLE:
        return None
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=save_dir,
        log_level="error"
    )
    tracker.start()
    return tracker

def stop_energy_tracker(tracker, save_dir, project_name):
    """Stop energy tracking and return metrics"""
    if tracker is None:
        return {
            'energy_kwh': float('nan'),
            'emissions_kg': float('nan'),
            'duration_s': float('nan')
        }
    emissions = tracker.stop()
    return {
        'energy_kwh': emissions,
        'emissions_kg': emissions * 0.475,
        'duration_s': tracker._total_duration.total_seconds() 
                      if hasattr(tracker, '_total_duration') else 0
    }

# ==================== DATASET ====================

class NumpyMemmapDataset(Dataset):
    """Dataset wrapper for NPZ files"""
    def __init__(self, imgs_np, labels_np, img_size=224, is_train=False, use_randaugment=False):
        self.imgs = imgs_np
        self.labels = labels_np
        self.img_size = img_size
        self.is_train = is_train

        if is_train and use_randaugment:
            self.base_tfms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
            ])
        elif is_train:
            self.base_tfms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            self.base_tfms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = int(self.labels[idx])
        x = self.base_tfms(img)
        
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        
        x = self.normalize(x)
        return x, label

def load_dataset(npz_path, batch_size=64):
    """Load dataset from NPZ file with specified batch size"""
    print(f"Loading {npz_path} (batch_size={batch_size})...")
    data = np.load(npz_path, mmap_mode="r")

    X_train, y_train = data["train_images"], data["train_labels"].flatten()
    X_val, y_val = data["val_images"], data["val_labels"].flatten()
    X_test, y_test = data["test_images"], data["test_labels"].flatten()

    print(f"Dataset sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE, is_train=True)
    val_ds = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE, is_train=False)
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    num_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    dataset_name = os.path.splitext(os.path.basename(npz_path))[0].replace('_224', '')

    return train_loader, val_loader, test_loader, num_classes, dataset_name

def load_dataset_enhanced(npz_path, batch_size=64):
    """Load dataset with stronger augmentation (RandAugment) for enhanced pruning method"""
    print(f"Loading {npz_path} with enhanced augmentation (batch_size={batch_size})...")
    data = np.load(npz_path, mmap_mode="r")

    X_train, y_train = data["train_images"], data["train_labels"].flatten()
    X_val, y_val = data["val_images"], data["val_labels"].flatten()
    X_test, y_test = data["test_images"], data["test_labels"].flatten()

    print(f"Dataset sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE, is_train=True,
                                   use_randaugment=True)
    val_ds = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE, is_train=False)
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    num_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    dataset_name = os.path.splitext(os.path.basename(npz_path))[0].replace('_224', '')

    return train_loader, val_loader, test_loader, num_classes, dataset_name

def mixup_data(x, y, alpha=0.2):
    """Apply mixup to a batch of data.

    Returns mixed inputs, pairs of targets, and the lambda value.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss as weighted combination of losses for both targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==================== METRICS ====================

def evaluate_model(net, test_loader, device):
    """Evaluate model and return metrics"""
    net.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            probs = torch.softmax(outputs, dim=1)
            predict_y = torch.max(probs, dim=1)[1]
            all_preds.extend(predict_y.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = sum([1 for i in range(len(all_preds)) if all_preds[i] == all_labels[i]]) / len(all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    n_classes = len(conf_matrix)
    all_labels_one_hot = label_binarize(all_labels, classes=list(range(n_classes)))
    
    try:
        auc = roc_auc_score(all_labels_one_hot, all_probs, multi_class='ovr')
    except ValueError:
        auc = float('nan')
    
    return acc, auc, precision, recall, f1

# ==================== GATE LAYER ====================

class GateLayer(nn.Module):
    """Gating layer for structured pruning (from Hikvision paper)"""
    def __init__(self, input_features, output_features, size_mask):
        super(GateLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.size_mask = size_mask
        self.weight = nn.Parameter(torch.ones(output_features))
        self.do_not_update = True  # Flag for identification
        self.mask = None
    
    def forward(self, input, mask=None):
        if mask is not None:
            self.mask = mask
        return input * self.weight.view(*self.size_mask)

# ==================== GATED VIT COMPONENTS ====================

class GatedMlp(nn.Module):
    """MLP with gate on hidden dimension"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.hidden_gate = GateLayer(hidden_features, hidden_features, [1, 1, -1])
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.hidden_gate(x)  # Gate AFTER fc1, BEFORE activation
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GatedAttention(nn.Module):
    """Multi-head attention with per-head gating"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gate = GateLayer(num_heads, num_heads, [1, -1, 1, 1])

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # [B, num_heads, N, head_dim]
        x = self.attn_gate(x)  # Gate per head BEFORE transpose
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GatedBlock(nn.Module):
    """Transformer block with residual gating"""
    def __init__(self, dim, num_heads, res_gate, mlp_ratio=4., qkv_bias=False, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GatedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GatedMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, drop=drop
        )
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.res_gate = res_gate  # Shared across all blocks

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.res_gate(x)  # Gate after first residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.res_gate(x)  # Gate after second residual
        return x

# ==================== GATED VIT MODEL ====================

class GatedVisionTransformer(nn.Module):
    """ViT with structured pruning gates (training model)"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.img_size = img_size
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        
        # Patch embedding
        from timm.models.layers import PatchEmbed
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # SHARED residual gate (critical Hikvision design)
        self.res_gate = GateLayer(embed_dim, embed_dim, [1, 1, -1])
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GatedBlock(
                dim=embed_dim, num_heads=num_heads, res_gate=self.res_gate,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=None,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        from timm.models.layers import trunc_normal_
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            from timm.models.layers import trunc_normal_
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.res_gate(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x[:, 0]
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# ==================== DEPLOYMENT VIT COMPONENTS ====================

class DeployMlp(nn.Module):
    """MLP without gates - physically smaller"""
    def __init__(self, in_features, hidden_features, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # Note: hidden_features is ACTUAL size
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DeployAttention(nn.Module):
    """Multi-head attention without gates - physically smaller"""
    def __init__(self, dim, num_heads, head_dim, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads  # Note: num_heads is ACTUAL count
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        
        self.inner_dim = head_dim * num_heads

        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.inner_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DeployBlock(nn.Module):
    """Transformer block without gates - physically smaller"""
    def __init__(self, dim, num_heads, head_dim, mlp_hidden_dim, qkv_bias=False, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DeployAttention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        
        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        self.mlp = DeployMlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ==================== DEPLOYMENT VIT MODEL ====================

class DeployVisionTransformer(nn.Module):
    """
    Deployment ViT - physically smaller model without gates
    This is what actually gets deployed for inference
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=192, depth=12, head_dim=64, 
                 per_layer_num_heads=None, per_layer_mlp_dim=None,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.depth = depth
        self.head_dim = head_dim
        
        # Patch embedding
        from timm.models.layers import PatchEmbed
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks with per-layer dimensions
        if per_layer_num_heads is None:
            per_layer_num_heads = [embed_dim // head_dim] * depth
        if per_layer_mlp_dim is None:
            per_layer_mlp_dim = [embed_dim * 4] * depth
        
        self.blocks = nn.ModuleList([
            DeployBlock(
                dim=embed_dim, 
                num_heads=per_layer_num_heads[i],
                head_dim=head_dim,
                mlp_hidden_dim=per_layer_mlp_dim[i],
                qkv_bias=qkv_bias, qk_scale=None,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Store config for FLOPs calculation
        self.per_layer_num_heads = per_layer_num_heads
        self.per_layer_mlp_dim = per_layer_mlp_dim
        
        # Initialize weights
        from timm.models.layers import trunc_normal_
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            from timm.models.layers import trunc_normal_
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x[:, 0]
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    def compute_flops(self):
        """Compute actual FLOPs for this deployment model"""
        n_patches = (self.img_size // self.patch_size) ** 2 + 1
        total_flops = 0
        
        for i, block in enumerate(self.blocks):
            num_heads = self.per_layer_num_heads[i]
            mlp_dim = self.per_layer_mlp_dim[i]
            d = self.embed_dim
            h_dim = self.head_dim
            
            # Attention FLOPs
            # QKV: n * d * (num_heads * h_dim * 3)
            qkv_flops = n_patches * d * (num_heads * h_dim * 3)
            # Attention matrix: n^2 * (num_heads * h_dim)
            attn_flops = n_patches * n_patches * (num_heads * h_dim)
            # Output projection: n * (num_heads * h_dim) * d
            out_flops = n_patches * (num_heads * h_dim) * d
            
            # MLP FLOPs
            mlp_flops = 2 * n_patches * d * mlp_dim
            
            total_flops += qkv_flops + attn_flops + out_flops + mlp_flops
        
        # Head FLOPs
        total_flops += self.embed_dim * self.num_classes
        
        return total_flops

# ==================== MODEL CREATION ====================

def create_gated_vit_from_timm(model_name, num_classes, pretrained=True):
    """Create gated ViT and load pretrained weights from TIMM"""
    print(f"Creating gated {model_name} (pretrained={pretrained})...")
    
    configs = {
        'vit_tiny_patch16_224': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
        'vit_small_patch16_224': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'vit_base_patch16_224': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}")
    
    cfg = configs[model_name]
    
    model = GatedVisionTransformer(
        img_size=IMG_SIZE,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=cfg['embed_dim'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1
    )
    
    if pretrained:
        print("  Loading pretrained weights from TIMM...")
        timm_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        model_dict = model.state_dict()
        timm_dict = timm_model.state_dict()
        
        # Transfer matching weights
        transfer_keys = ['patch_embed.proj.weight', 'patch_embed.proj.bias',
                        'pos_embed', 'cls_token', 'norm.weight', 'norm.bias',
                        'head.weight', 'head.bias']
        
        for key in transfer_keys:
            if key in timm_dict and key in model_dict:
                model_dict[key].copy_(timm_dict[key])
        
        # Transfer block weights
        for i in range(cfg['depth']):
            for component in ['norm1', 'norm2', 'attn.qkv', 'attn.proj', 
                            'mlp.fc1', 'mlp.fc2']:
                for suffix in ['weight', 'bias']:
                    src_key = f'blocks.{i}.{component}.{suffix}'
                    if src_key in timm_dict and src_key in model_dict:
                        model_dict[src_key].copy_(timm_dict[src_key])
        
        print("  ✓ Pretrained weights loaded")

    return model


def create_small_vit_from_pretrained(model_name, num_classes, target_sparsity=0.4, pretrained=True):
    """
    Create a small DeployVisionTransformer with dimensions matching progressive pruning output.
    Initialize from ImageNet pretrained weights where possible.

    Args:
        model_name: Base model name ('vit_tiny_patch16_224' or 'vit_base_patch16_224')
        num_classes: Number of output classes
        target_sparsity: Target sparsity (0.4 = keep 60% of each dimension)
        pretrained: Whether to initialize from pretrained weights

    Returns:
        deploy_model: Small DeployVisionTransformer
        config: Configuration dictionary
    """
    # Get original model dimensions
    model_configs = {
        'vit_tiny_patch16_224': {'embed_dim': 192, 'num_heads': 3, 'mlp_ratio': 4, 'depth': 12},
        'vit_base_patch16_224': {'embed_dim': 768, 'num_heads': 12, 'mlp_ratio': 4, 'depth': 12},
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}")

    cfg = model_configs[model_name]
    orig_embed_dim = cfg['embed_dim']
    orig_num_heads = cfg['num_heads']
    orig_mlp_ratio = cfg['mlp_ratio']
    orig_depth = cfg['depth']
    orig_head_dim = orig_embed_dim // orig_num_heads

    # Calculate pruned dimensions (keep 1 - target_sparsity of each)
    keep_ratio = 1.0 - target_sparsity

    new_embed_dim = int(orig_embed_dim * keep_ratio)
    new_num_heads = max(1, round(orig_num_heads * keep_ratio))  # At least 1 head
    new_mlp_dim = int(orig_embed_dim * orig_mlp_ratio * keep_ratio)

    # Use same dimensions for all layers (uniform pruning)
    per_layer_num_heads = [new_num_heads] * orig_depth
    per_layer_mlp_dim = [new_mlp_dim] * orig_depth

    print(f"\nCreating small ViT matching {target_sparsity:.0%} pruned progressive model:")
    print(f"  Original: embed_dim={orig_embed_dim}, heads={orig_num_heads}, mlp_dim={orig_embed_dim * orig_mlp_ratio}")
    print(f"  Small:    embed_dim={new_embed_dim}, heads={new_num_heads}, mlp_dim={new_mlp_dim}")

    # Create small deployment model
    model = DeployVisionTransformer(
        img_size=IMG_SIZE,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=new_embed_dim,
        depth=orig_depth,
        head_dim=orig_head_dim,
        per_layer_num_heads=per_layer_num_heads,
        per_layer_mlp_dim=per_layer_mlp_dim,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1
    ).to(DEVICE)

    if pretrained:
        print(f"  Initializing from ImageNet pretrained {model_name} (partial transfer)...")
        timm_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        timm_dict = timm_model.state_dict()
        model_dict = model.state_dict()

        with torch.no_grad():
            # Patch embedding - take first new_embed_dim channels
            orig_proj_weight = timm_dict['patch_embed.proj.weight']  # [192, 3, 16, 16]
            model_dict['patch_embed.proj.weight'].copy_(orig_proj_weight[:new_embed_dim])
            model_dict['patch_embed.proj.bias'].copy_(timm_dict['patch_embed.proj.bias'][:new_embed_dim])

            # Position embedding - take first new_embed_dim dims
            orig_pos_embed = timm_dict['pos_embed']  # [1, 197, 192]
            model_dict['pos_embed'].copy_(orig_pos_embed[:, :, :new_embed_dim])

            # CLS token
            orig_cls = timm_dict['cls_token']  # [1, 1, 192]
            model_dict['cls_token'].copy_(orig_cls[:, :, :new_embed_dim])

            # Transfer block weights (partial)
            for i in range(orig_depth):
                # Norm layers
                model_dict[f'blocks.{i}.norm1.weight'].copy_(timm_dict[f'blocks.{i}.norm1.weight'][:new_embed_dim])
                model_dict[f'blocks.{i}.norm1.bias'].copy_(timm_dict[f'blocks.{i}.norm1.bias'][:new_embed_dim])
                model_dict[f'blocks.{i}.norm2.weight'].copy_(timm_dict[f'blocks.{i}.norm2.weight'][:new_embed_dim])
                model_dict[f'blocks.{i}.norm2.bias'].copy_(timm_dict[f'blocks.{i}.norm2.bias'][:new_embed_dim])

                # Attention QKV - need to select heads and embed dims
                orig_qkv_weight = timm_dict[f'blocks.{i}.attn.qkv.weight']  # [576, 192]
                orig_qkv_bias = timm_dict[f'blocks.{i}.attn.qkv.bias']  # [576]

                # Reshape to [3, num_heads, head_dim, embed_dim]
                q_w, k_w, v_w = orig_qkv_weight.chunk(3, dim=0)
                q_b, k_b, v_b = orig_qkv_bias.chunk(3, dim=0)

                q_w = q_w.view(orig_num_heads, orig_head_dim, orig_embed_dim)[:new_num_heads, :, :new_embed_dim]
                k_w = k_w.view(orig_num_heads, orig_head_dim, orig_embed_dim)[:new_num_heads, :, :new_embed_dim]
                v_w = v_w.view(orig_num_heads, orig_head_dim, orig_embed_dim)[:new_num_heads, :, :new_embed_dim]
                q_b = q_b.view(orig_num_heads, orig_head_dim)[:new_num_heads]
                k_b = k_b.view(orig_num_heads, orig_head_dim)[:new_num_heads]
                v_b = v_b.view(orig_num_heads, orig_head_dim)[:new_num_heads]

                new_qkv_weight = torch.cat([q_w.flatten(0, 1), k_w.flatten(0, 1), v_w.flatten(0, 1)], dim=0)
                new_qkv_bias = torch.cat([q_b.flatten(), k_b.flatten(), v_b.flatten()], dim=0)

                model_dict[f'blocks.{i}.attn.qkv.weight'].copy_(new_qkv_weight)
                model_dict[f'blocks.{i}.attn.qkv.bias'].copy_(new_qkv_bias)

                # Attention projection
                orig_proj_weight = timm_dict[f'blocks.{i}.attn.proj.weight']  # [192, 192]
                orig_proj_weight = orig_proj_weight.view(orig_embed_dim, orig_num_heads, orig_head_dim)
                new_proj_weight = orig_proj_weight[:new_embed_dim, :new_num_heads, :].flatten(1)
                model_dict[f'blocks.{i}.attn.proj.weight'].copy_(new_proj_weight)
                model_dict[f'blocks.{i}.attn.proj.bias'].copy_(timm_dict[f'blocks.{i}.attn.proj.bias'][:new_embed_dim])

                # MLP fc1
                orig_fc1_weight = timm_dict[f'blocks.{i}.mlp.fc1.weight']  # [768, 192]
                model_dict[f'blocks.{i}.mlp.fc1.weight'].copy_(orig_fc1_weight[:new_mlp_dim, :new_embed_dim])
                model_dict[f'blocks.{i}.mlp.fc1.bias'].copy_(timm_dict[f'blocks.{i}.mlp.fc1.bias'][:new_mlp_dim])

                # MLP fc2
                orig_fc2_weight = timm_dict[f'blocks.{i}.mlp.fc2.weight']  # [192, 768]
                model_dict[f'blocks.{i}.mlp.fc2.weight'].copy_(orig_fc2_weight[:new_embed_dim, :new_mlp_dim])
                model_dict[f'blocks.{i}.mlp.fc2.bias'].copy_(timm_dict[f'blocks.{i}.mlp.fc2.bias'][:new_embed_dim])

            # Final norm
            model_dict['norm.weight'].copy_(timm_dict['norm.weight'][:new_embed_dim])
            model_dict['norm.bias'].copy_(timm_dict['norm.bias'][:new_embed_dim])

            # Head - random init since num_classes may differ
            # (already initialized by model's _init_weights)

        print("  ✓ Partial pretrained weights loaded")

    config = {
        'embed_dim': new_embed_dim,
        'num_heads': new_num_heads,
        'mlp_dim': new_mlp_dim,
        'depth': orig_depth,
        'head_dim': orig_head_dim,
        'target_sparsity': target_sparsity,
        'per_layer_num_heads': per_layer_num_heads,
        'per_layer_mlp_dim': per_layer_mlp_dim
    }

    print(f"  Parameters: {count_parameters(model):.2f}M")
    print(f"  Model size: {model_size_mb(model):.2f}MB")

    return model, config


# ==================== DEPLOYMENT CONVERSION ====================

def convert_to_deployment_model(gated_model, num_classes, save_masks=False, save_dir=None):
    """Convert gated model to physically smaller deployment model."""
    GATE_THRESHOLD = 0.01

    # Extract masks
    res_gate_weights = gated_model.res_gate.weight.data
    res_mask = (res_gate_weights.abs() > GATE_THRESHOLD).cpu()
    new_embed_dim = res_mask.sum().item()

    head_masks, mlp_masks = [], []
    for block in gated_model.blocks:
        head_masks.append((block.attn.attn_gate.weight.data.abs() > GATE_THRESHOLD).cpu())
        mlp_masks.append((block.mlp.hidden_gate.weight.data.abs() > GATE_THRESHOLD).cpu())

    per_layer_num_heads = [m.sum().item() for m in head_masks]
    per_layer_mlp_dim = [m.sum().item() for m in mlp_masks]
    head_dim = gated_model.embed_dim // gated_model.num_heads

    # Create deployment model
    deploy_model = DeployVisionTransformer(
        img_size=IMG_SIZE, patch_size=16, in_chans=3, num_classes=num_classes,
        embed_dim=new_embed_dim, depth=gated_model.depth, head_dim=head_dim,
        per_layer_num_heads=per_layer_num_heads, per_layer_mlp_dim=per_layer_mlp_dim,
        qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1
    ).to(DEVICE)

    # Transfer weights
    with torch.no_grad():
        gated_dict = gated_model.state_dict()
        deploy_dict = deploy_model.state_dict()

        deploy_dict['patch_embed.proj.weight'].copy_(gated_dict['patch_embed.proj.weight'][res_mask])
        deploy_dict['patch_embed.proj.bias'].copy_(gated_dict['patch_embed.proj.bias'][res_mask])
        deploy_dict['pos_embed'].copy_(gated_dict['pos_embed'][:, :, res_mask])
        deploy_dict['cls_token'].copy_(gated_dict['cls_token'][:, :, res_mask])

        orig_num_heads = gated_model.num_heads
        for i, (head_mask, mlp_mask) in enumerate(zip(head_masks, mlp_masks)):
            deploy_dict[f'blocks.{i}.norm1.weight'].copy_(gated_dict[f'blocks.{i}.norm1.weight'][res_mask])
            deploy_dict[f'blocks.{i}.norm1.bias'].copy_(gated_dict[f'blocks.{i}.norm1.bias'][res_mask])
            deploy_dict[f'blocks.{i}.norm2.weight'].copy_(gated_dict[f'blocks.{i}.norm2.weight'][res_mask])
            deploy_dict[f'blocks.{i}.norm2.bias'].copy_(gated_dict[f'blocks.{i}.norm2.bias'][res_mask])

            # QKV
            q_w, k_w, v_w = gated_dict[f'blocks.{i}.attn.qkv.weight'].chunk(3, dim=0)
            q_b, k_b, v_b = gated_dict[f'blocks.{i}.attn.qkv.bias'].chunk(3, dim=0)
            q_w = q_w.view(orig_num_heads, head_dim, gated_model.embed_dim)[head_mask][:, :, res_mask]
            k_w = k_w.view(orig_num_heads, head_dim, gated_model.embed_dim)[head_mask][:, :, res_mask]
            v_w = v_w.view(orig_num_heads, head_dim, gated_model.embed_dim)[head_mask][:, :, res_mask]
            q_b = q_b.view(orig_num_heads, head_dim)[head_mask]
            k_b = k_b.view(orig_num_heads, head_dim)[head_mask]
            v_b = v_b.view(orig_num_heads, head_dim)[head_mask]
            deploy_dict[f'blocks.{i}.attn.qkv.weight'].copy_(torch.cat([q_w.flatten(0,1), k_w.flatten(0,1), v_w.flatten(0,1)], dim=0))
            deploy_dict[f'blocks.{i}.attn.qkv.bias'].copy_(torch.cat([q_b.flatten(), k_b.flatten(), v_b.flatten()], dim=0))

            # Proj
            proj_w = gated_dict[f'blocks.{i}.attn.proj.weight'].view(gated_model.embed_dim, orig_num_heads, head_dim)
            deploy_dict[f'blocks.{i}.attn.proj.weight'].copy_(proj_w[res_mask][:, head_mask, :].flatten(1))
            deploy_dict[f'blocks.{i}.attn.proj.bias'].copy_(gated_dict[f'blocks.{i}.attn.proj.bias'][res_mask])

            # MLP
            deploy_dict[f'blocks.{i}.mlp.fc1.weight'].copy_(gated_dict[f'blocks.{i}.mlp.fc1.weight'][mlp_mask][:, res_mask])
            deploy_dict[f'blocks.{i}.mlp.fc1.bias'].copy_(gated_dict[f'blocks.{i}.mlp.fc1.bias'][mlp_mask])
            deploy_dict[f'blocks.{i}.mlp.fc2.weight'].copy_(gated_dict[f'blocks.{i}.mlp.fc2.weight'][res_mask][:, mlp_mask])
            deploy_dict[f'blocks.{i}.mlp.fc2.bias'].copy_(gated_dict[f'blocks.{i}.mlp.fc2.bias'][res_mask])

        deploy_dict['norm.weight'].copy_(gated_dict['norm.weight'][res_mask])
        deploy_dict['norm.bias'].copy_(gated_dict['norm.bias'][res_mask])
        deploy_dict['head.weight'].copy_(gated_dict['head.weight'][:, res_mask])
        deploy_dict['head.bias'].copy_(gated_dict['head.bias'])

    pruning_config = {
        'original_embed_dim': gated_model.embed_dim,
        'original_num_heads': gated_model.num_heads,
        'deploy_embed_dim': new_embed_dim,
        'deploy_per_layer_num_heads': per_layer_num_heads,
        'deploy_per_layer_mlp_dim': per_layer_mlp_dim,
    }

    return deploy_model, pruning_config

# ==================== TAYLOR IMPORTANCE SCORING ====================

def forward_hook(self, input, output):
    """Hook to store output"""
    self.output = output.detach()

def backward_hook(self, grad_input, grad_output):
    """Hook to store gradient"""
    self.grad = (grad_output[0].detach(),)

def taylor2Scorer(gate_module):
    """
    Taylor2 scorer from CVPR2019 paper (best method)
    Score = |weight × weight.grad|²
    """
    score = (gate_module.weight * gate_module.weight.grad).data.pow(2)
    return score, 0

def prepare_pruning_list(model, pruning_layer_type):
    """
    Collect gate modules for pruning
    pruning_layer_type: 0=attn, 1=mlp, 2=residual
    """
    pruning_modules = []
    
    for module_name, m in model.named_modules():
        if hasattr(m, "do_not_update"):
            if pruning_layer_type == 0 and 'attn_gate' in module_name:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                pruning_modules.append(m)
            elif pruning_layer_type == 1 and 'hidden_gate' in module_name:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                pruning_modules.append(m)
            elif pruning_layer_type == 2 and 'res_gate' in module_name:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                pruning_modules.append(m)
    
    return pruning_modules

# ==================== PRUNING ENGINE ====================

class StructuredPruner:
    """Structured pruning engine"""
    def __init__(self, model, pruning_modules, pruning_momentum=0.9):
        self.model = model
        self.pruning_modules = pruning_modules
        self.pruning_parameters = [m.weight for m in pruning_modules]
        self.momentum = pruning_momentum
        
        self.iterations_done = 0
        self.pruning_scores = {
            'score': [list() for _ in range(len(self.pruning_parameters))],
            'averaged': [list() for _ in range(len(self.pruning_parameters))]
        }
        
        self.pruning_gates = [
            np.ones(len(param),) for param in self.pruning_parameters
        ]
        
        self.all_neuron_units = sum(len(param) for param in self.pruning_parameters)
    
    def do_step(self, loss=None):
        """Collect importance scores for one batch"""
        for layer, module in enumerate(self.pruning_modules):
            scores, _ = taylor2Scorer(module)
            
            if self.iterations_done == 0:
                self.pruning_scores['score'][layer] = scores
            else:
                self.pruning_scores['score'][layer] += scores
        
        self.iterations_done += 1
    
    def finalize_scores(self):
        """Average and apply momentum"""
        for layer, score in enumerate(self.pruning_scores['score']):
            contribution = self.pruning_scores['score'][layer] / self.iterations_done
            
            if len(self.pruning_scores["averaged"][layer]) == 0 or not self.momentum:
                self.pruning_scores["averaged"][layer] = contribution
            else:
                self.pruning_scores["averaged"][layer] = (
                    self.momentum * self.pruning_scores["averaged"][layer] +
                    (1 - self.momentum) * contribution
                )
        
        criteria = [
            score.detach().cpu().numpy() if torch.is_tensor(score) else score
            for score in self.pruning_scores['averaged']
        ]
        
        return criteria
    
    def reset(self):
        """Reset gates to all ones"""
        for layer in range(len(self.pruning_parameters)):
            self.pruning_parameters[layer].data = torch.ones_like(self.pruning_parameters[layer])
            self.pruning_gates[layer] = np.ones(len(self.pruning_parameters[layer]),)
    
    def prune(self, criteria, threshold):
        """Apply pruning based on threshold"""
        for layer in range(len(criteria)):
            index = np.where(criteria[layer] <= threshold)
            self.pruning_gates[layer][index] *= 0.0
            self.pruning_parameters[layer].data[index] *= 0.0
    
    def get_num_active(self):
        """Count active (unpruned) units"""
        return sum(np.count_nonzero(gate) for gate in self.pruning_gates)

# ==================== IMPORTANCE COLLECTION ====================

def collect_importance_scores(model, pruners, train_loader, max_batches=50):
    """
    Phase 1: Collect importance scores
    Model in eval mode, gradients collected but no weight updates
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    print(f"Collecting importance scores over {max_batches} batches...")
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
        
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Zero all gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        
        # Forward + backward (no optimizer step!)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Collect scores
        for pruner in pruners:
            pruner.do_step(loss.item())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{max_batches}")
    
    print("✓ Importance collection complete")

def apply_pruning_per_component(model, pruners, target_sparsity=0.5):
    """
    FIX: Prune each component type independently for balanced pruning
    This ensures attention, MLP, and residual are all pruned equally
    """
    print(f"\nApplying per-component pruning (target: {target_sparsity:.1%})...")
    
    component_names = ['Attention Heads', 'MLP Hidden', 'Residual Dims']
    
    for i, pruner in enumerate(pruners):
        criteria = pruner.finalize_scores()
        
        # Flatten scores for THIS component only
        all_scores = []
        for layer_scores in criteria:
            all_scores.extend(layer_scores.flatten().tolist())
        
        all_scores_array = np.array(all_scores)
        
        # Calculate threshold for THIS component
        num_to_prune = int(len(all_scores_array) * target_sparsity)
        if num_to_prune > 0:
            threshold = np.sort(all_scores_array)[num_to_prune]
        else:
            threshold = -np.inf
        
        print(f"  {component_names[i]}:")
        print(f"    Total units: {len(all_scores_array)}")
        print(f"    Threshold: {threshold:.6f}")
        print(f"    Pruning: {num_to_prune} units ({target_sparsity*100:.1f}%)")
        
        # Apply pruning
        pruner.prune(criteria, threshold)
        
        active = pruner.get_num_active()
        total = pruner.all_neuron_units
        print(f"    Result: {active}/{total} active ({active/total*100:.1f}%)")

# ==================== TRAINING FUNCTIONS ====================

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, gate_l1_weight=0.0, scaler=None):
    """
    Training epoch with L1 regularization on gates and optional AMP (mixed precision)
    """
    model.train()
    running_loss = 0.0
    running_gate_loss = 0.0
    correct = 0
    total = 0

    use_amp = scaler is not None

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            task_loss = nn.functional.cross_entropy(outputs, labels)

            # L1 regularization on gates
            gate_l1_loss = 0.0
            if gate_l1_weight > 0:
                for name, param in model.named_parameters():
                    if 'gate' in name and 'weight' in name:
                        gate_l1_loss += torch.abs(param).sum()

            total_loss = task_loss + gate_l1_weight * gate_l1_loss

        # Backward with or without scaler
        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        running_loss += task_loss.item()
        running_gate_loss += gate_l1_loss.item() if isinstance(gate_l1_loss, torch.Tensor) else 0.0
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    avg_gate_loss = running_gate_loss / len(train_loader)
    acc = correct / total

    if gate_l1_weight > 0:
        print(f"    Task Loss: {avg_loss:.4f}, Gate L1: {avg_gate_loss:.4f}")

    return avg_loss, acc

def train_one_epoch_enhanced(model, train_loader, optimizer, scheduler, epoch,
                              gate_l1_weight=0.0, scaler=None, mixup_alpha=0.2):
    """
    Training epoch with mixup augmentation and L1 regularization on gates.
    Mixup blends pairs of samples and uses a weighted loss on both targets.
    """
    model.train()
    running_loss = 0.0
    running_gate_loss = 0.0
    correct = 0
    total = 0

    use_amp = scaler is not None
    criterion = nn.CrossEntropyLoss()

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Apply mixup
        if mixup_alpha > 0:
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
        else:
            targets_a, targets_b, lam = labels, labels, 1.0

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            task_loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # L1 regularization on gates
            gate_l1_loss = 0.0
            if gate_l1_weight > 0:
                for name, param in model.named_parameters():
                    if 'gate' in name and 'weight' in name:
                        gate_l1_loss += torch.abs(param).sum()

            total_loss = task_loss + gate_l1_weight * gate_l1_loss

        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        running_loss += task_loss.item()
        running_gate_loss += gate_l1_loss.item() if isinstance(gate_l1_loss, torch.Tensor) else 0.0
        # For accuracy tracking with mixup, count based on the dominant target
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (lam * predicted.eq(targets_a).sum().item()
                     + (1 - lam) * predicted.eq(targets_b).sum().item())

    avg_loss = running_loss / len(train_loader)
    avg_gate_loss = running_gate_loss / len(train_loader)
    acc = correct / total

    if gate_l1_weight > 0:
        print(f"    Task Loss: {avg_loss:.4f}, Gate L1: {avg_gate_loss:.4f}")

    return avg_loss, acc

def finetune_deployment_model(deploy_model, train_loader, val_loader, epochs=5, lr=5e-5):
    """
    Finetune deployment model after conversion
    This helps recover any accuracy lost during weight transfer
    """
    print(f"\nFinetuning deployment model for {epochs} epochs...")
    
    optimizer = optim.AdamW(deploy_model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)
    
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(deploy_model, train_loader, optimizer, scheduler, epoch, gate_l1_weight=0.0)
        val_acc, _, _, _, _ = evaluate_model(deploy_model, val_loader, DEVICE)
        
        print(f"  Epoch {epoch}/{epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(deploy_model.state_dict())
    
    if best_state is not None:
        deploy_model.load_state_dict(best_state)
        print(f"✓ Best val acc: {best_val_acc:.4f}")
    
    return deploy_model

# ==================== METHOD 1: BASELINE ====================

def train_baseline_vit(dataset_name, model_name, train_loader, val_loader, test_loader,
                       num_classes, save_dir, trial_num):
    """
    METHOD 1: Baseline (no pruning)
    Creates deployment model (same size as training model)
    """
    print("\n" + "="*80)
    print(f"METHOD 1: BASELINE - {dataset_name.upper()} - {model_name} - TRIAL {trial_num}")
    print("="*80)

    # Create gated model (gates stay at 1.0)
    model = create_gated_vit_from_timm(model_name, num_classes, pretrained=True).to(DEVICE)
    print(f"Parameters: {count_parameters(model):.2f}M")

    # Freeze gates
    for name, param in model.named_parameters():
        if 'gate' in name:
            param.requires_grad = False

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    total_steps = FIXED_EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)

    # AMP scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if USE_AMP and DEVICE.type == 'cuda' else None

    # Track metrics
    history = []
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0

    # Start energy tracking
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_baseline_trial{trial_num}")

    print(f"\nTraining for {FIXED_EPOCHS} epochs (AMP={scaler is not None})...")

    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, epoch,
                                                 gate_l1_weight=0.0, scaler=scaler)
        val_acc, val_auc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, DEVICE)
        test_acc, test_auc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, DEVICE)
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"  ✓ New best validation accuracy: {val_acc:.4f}")
        
        history.append({
            'trial': trial_num, 'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_acc': val_acc, 'test_acc': test_acc
        })
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch}")
    
    # Convert to deployment model (no pruning, so same size)
    print("\nConverting to deployment model (no pruning applied)...")
    deploy_model, _ = convert_to_deployment_model(model, num_classes, save_masks=False, save_dir=save_dir)
    
    # Measure energy on deployment model
    print("\n📊 Measuring energy on deployment model...")
    deploy_tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_baseline_deploy_trial{trial_num}")
    test_acc, test_auc, test_precision, test_recall, test_f1 = evaluate_model(deploy_model, test_loader, DEVICE)
    energy_metrics = stop_energy_tracker(deploy_tracker, save_dir, f"{dataset_name}_{model_name}_baseline_deploy_trial{trial_num}")
    
    # Stop training energy tracker
    training_energy = stop_energy_tracker(tracker, save_dir, f"{dataset_name}_{model_name}_baseline_trial{trial_num}")
    
    # Save
    torch.save(deploy_model.state_dict(), os.path.join(save_dir, 
               f"{dataset_name}_{model_name}_baseline_deploy_trial{trial_num}_final.pth"))
    pd.DataFrame(history).to_csv(os.path.join(save_dir,
               f"{dataset_name}_{model_name}_baseline_trial{trial_num}_history.csv"), index=False)
    
    final_metrics = {
        'method': 'baseline',
        'dataset': dataset_name,
        'model': model_name,
        'trial': trial_num,
        'total_epochs': FIXED_EPOCHS,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'params_m': count_parameters(deploy_model),
        'model_size_mb': model_size_mb(deploy_model),
        'flops_g': deploy_model.compute_flops() / 1e9,
        'training_energy_kwh': training_energy['energy_kwh'],
        'inference_energy_kwh': energy_metrics['energy_kwh']
    }
    
    print(f"\n✓ Final Test - Acc: {test_acc:.4f}, Precision: {test_precision:.4f}")
    
    cleanup_memory()
    return deploy_model, final_metrics

# ==================== METHOD 2: PROGRESSIVE PRUNING ====================

def train_progressive_structured_pruning(dataset_name, model_name, train_loader, val_loader, test_loader,
                                        num_classes, save_dir, trial_num):
    """
    METHOD 2: Progressive Structured Pruning (gentle_40pct from ViT_hyper.py)

    Flow:
    1. Train gated model with progressive pruning (multiple steps to 40% sparsity)
    2. After final prune, convert to deployment model (physically smaller)
    3. Continue training deployment model for remaining epochs
    4. Track and restore best deployment model
    """
    print("\n" + "="*80)
    print(f"METHOD 2: PROGRESSIVE STRUCTURED PRUNING - {dataset_name.upper()} - {model_name} - TRIAL {trial_num}")
    print("="*80)

    prune_epochs = PRUNE_EPOCHS_SCHEDULE.copy()
    final_prune_epoch = max(prune_epochs)
    convert_to_deploy_epoch = final_prune_epoch + 1  # Convert right after final prune

    print(f"Pruning schedule: {prune_epochs}")
    print(f"Sparsity schedule: {SPARSITY_SCHEDULE}")
    print(f"Convert to deployment after epoch: {final_prune_epoch}")

    # Create gated model
    model = create_gated_vit_from_timm(model_name, num_classes, pretrained=True).to(DEVICE)
    print(f"Initial parameters: {count_parameters(model):.2f}M")

    # Setup pruners
    pruners = []
    for component_type in [0, 1, 2]:
        modules = prepare_pruning_list(model, component_type)
        if len(modules) > 0:
            pruner = StructuredPruner(model, modules, pruning_momentum=PRUNING_MOMENTUM)
            pruners.append(pruner)

    print(f"Created {len(pruners)} pruning engines")

    # Optimizer and scheduler for gated phase
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    current_lr = INITIAL_LR
    total_steps = FIXED_EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)

    # AMP scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if USE_AMP and DEVICE.type == 'cuda' else None

    # Track metrics
    all_metrics = []
    deploy_model = None
    using_deploy = False
    best_deploy_val_acc = 0.0
    best_deploy_state = None
    best_deploy_epoch = 0
    pruning_config = None

    # Start energy tracking
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_progressive_trial{trial_num}")

    print(f"Training with AMP={scaler is not None}")

    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")

        # Pruning steps (only during gated phase)
        if not using_deploy and epoch in prune_epochs:
            prune_step = prune_epochs.index(epoch) + 1
            target_sparsity = SPARSITY_SCHEDULE[prune_step - 1]
            print(f"  [Pruning step {prune_step}/{NUM_PRUNE_STEPS} -> {target_sparsity*100:.0f}% sparsity]")

            # Reset pruners
            for pruner in pruners:
                pruner.iterations_done = 0
                pruner.pruning_scores['score'] = [list() for _ in range(len(pruner.pruning_parameters))]

            # Collect importance and prune
            collect_importance_scores(model, pruners, train_loader, max_batches=IMPORTANCE_CAL_BATCHES)
            apply_pruning_per_component(model, pruners, target_sparsity=target_sparsity)

            # Print gate statistics
            gate_stats = get_gate_sparsity(model)
            print(f"  Gate sparsity: Attn={gate_stats['attn']:.1f}%, MLP={gate_stats['mlp']:.1f}%, Res={gate_stats['res']:.1f}%")

            # Reduce LR after pruning
            current_lr *= LR_REDUCTION_AFTER_PRUNE
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"  LR reduced to {current_lr:.6f}")

        # Convert to deployment model after final pruning
        if epoch == convert_to_deploy_epoch and not using_deploy:
            print(f"  [All pruning complete - converting to deployment model]")

            # Final gate statistics before conversion
            gate_stats = get_gate_sparsity(model)
            print(f"  Final gate sparsity: Attn={gate_stats['attn']:.1f}%, MLP={gate_stats['mlp']:.1f}%, Res={gate_stats['res']:.1f}%")

            # Convert to deployment
            deploy_model, pruning_config = convert_to_deployment_model(model, num_classes)
            using_deploy = True

            # Create new optimizer and scheduler for deployment model
            remaining_epochs = FIXED_EPOCHS - epoch + 1
            remaining_steps = remaining_epochs * len(train_loader)
            optimizer = optim.AdamW(deploy_model.parameters(), lr=current_lr, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_steps, eta_min=MIN_LR)

            print(f"  [Deployment: {count_parameters(deploy_model):.2f}M params (was {count_parameters(model):.2f}M)]")

        # Select active model
        active_model = deploy_model if using_deploy else model
        gate_l1 = 0.0 if using_deploy else GATE_L1_WEIGHT

        # Train
        train_loss, train_acc = train_one_epoch(
            active_model, train_loader, optimizer, scheduler, epoch,
            gate_l1_weight=gate_l1, scaler=scaler
        )

        # Evaluate
        val_acc, val_auc, val_precision, val_recall, val_f1 = evaluate_model(active_model, val_loader, DEVICE)
        test_acc, test_auc, test_precision, test_recall, test_f1 = evaluate_model(active_model, test_loader, DEVICE)

        model_type = 'deploy' if using_deploy else 'gated'
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Acc: {val_acc:.4f}, AUC: {val_auc:.4f} [{model_type}]")

        # Track best deployment model
        if using_deploy and val_acc > best_deploy_val_acc:
            best_deploy_val_acc = val_acc
            best_deploy_state = copy.deepcopy(deploy_model.state_dict())
            best_deploy_epoch = epoch
            print(f"  ✓ New best deployment val_acc: {val_acc:.4f}")

        all_metrics.append({
            'trial': trial_num, 'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_acc': val_acc, 'val_auc': val_auc, 'test_acc': test_acc, 'test_auc': test_auc,
            'params_m': count_parameters(active_model), 'lr': current_lr, 'model_type': model_type
        })

    # Restore best deployment model
    if deploy_model is None:
        print("  [WARNING: No deployment conversion happened, converting now]")
        deploy_model, pruning_config = convert_to_deployment_model(model, num_classes)
    elif best_deploy_state is not None:
        deploy_model.load_state_dict(best_deploy_state)
        print(f"\n✓ Restored best deployment model from epoch {best_deploy_epoch} (val_acc={best_deploy_val_acc:.4f})")

    # Measure inference energy on deployment model
    print("\nMeasuring inference energy on deployment model...")
    deploy_tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_progressive_deploy_trial{trial_num}")
    test_acc, test_auc, test_precision, test_recall, test_f1 = evaluate_model(deploy_model, test_loader, DEVICE)
    energy_metrics = stop_energy_tracker(deploy_tracker, save_dir, f"{dataset_name}_{model_name}_progressive_deploy_trial{trial_num}")

    # Stop training energy tracker
    training_energy = stop_energy_tracker(tracker, save_dir, f"{dataset_name}_{model_name}_progressive_trial{trial_num}")

    # Save
    torch.save(deploy_model.state_dict(), os.path.join(save_dir,
               f"{dataset_name}_{model_name}_progressive_deploy_trial{trial_num}_final.pth"))
    pd.DataFrame(all_metrics).to_csv(os.path.join(save_dir,
               f"{dataset_name}_{model_name}_progressive_trial{trial_num}_metrics.csv"), index=False)

    # Get effective dimensions
    avg_heads = np.mean(pruning_config['deploy_per_layer_num_heads'])
    avg_mlp = np.mean(pruning_config['deploy_per_layer_mlp_dim'])

    final_metrics = {
        'method': 'progressive_structured_pruning',
        'dataset': dataset_name,
        'model': model_name,
        'trial': trial_num,
        'total_epochs': FIXED_EPOCHS,
        'best_epoch': best_deploy_epoch,
        'best_val_acc': best_deploy_val_acc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'params_m': count_parameters(deploy_model),
        'model_size_mb': model_size_mb(deploy_model),
        'flops_g': deploy_model.compute_flops() / 1e9,
        'training_energy_kwh': training_energy['energy_kwh'],
        'inference_energy_kwh': energy_metrics['energy_kwh'],
        'deploy_embed_dim': pruning_config['deploy_embed_dim'],
        'deploy_avg_heads': avg_heads,
        'deploy_avg_mlp': avg_mlp,
        'total_prune_steps': NUM_PRUNE_STEPS
    }

    print(f"\n✓ Final Test - Acc: {test_acc:.4f}, Precision: {test_precision:.4f}")
    print(f"Deployment: {pruning_config['deploy_embed_dim']} embed, {avg_heads:.1f} avg heads, {avg_mlp:.0f} avg MLP")

    cleanup_memory()
    return deploy_model, final_metrics

# ==================== METHOD 3: SMALL MODEL FROM SCRATCH ====================

def train_small_model_from_scratch(dataset_name, model_name, train_loader, val_loader, test_loader,
                                   num_classes, save_dir, trial_num):
    """
    METHOD 3: Train Small Model From Scratch

    Creates a small DeployVisionTransformer with dimensions matching the progressive
    pruning output (40% sparsity = 60% of original dimensions), initializes from
    ImageNet pretrained weights, and trains with baseline hyperparameters.

    This provides a fair comparison to see if progressive pruning outperforms
    simply training a smaller model from scratch.
    """
    print("\n" + "="*80)
    print(f"METHOD 3: SMALL MODEL FROM SCRATCH - {dataset_name.upper()} - {model_name} - TRIAL {trial_num}")
    print("="*80)

    # Create small model with dimensions matching 40% pruned progressive model
    model, model_config = create_small_vit_from_pretrained(
        model_name=model_name,
        num_classes=num_classes,
        target_sparsity=TARGET_FLOPS_REDUCTION,  # 0.4 = 40% pruned
        pretrained=True
    )

    print(f"\nSmall model architecture:")
    print(f"  Embed dim: {model_config['embed_dim']}")
    print(f"  Num heads: {model_config['num_heads']}")
    print(f"  MLP dim: {model_config['mlp_dim']}")
    print(f"  Parameters: {count_parameters(model):.2f}M")

    # Start energy tracking
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_small_model_trial{trial_num}")

    # Optimizer (same as baseline)
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    total_steps = FIXED_EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)

    # AMP scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if USE_AMP and DEVICE.type == 'cuda' else None
    use_amp = scaler is not None

    # Track metrics
    all_metrics = []
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0

    print(f"\nTraining small model for {FIXED_EPOCHS} epochs (AMP={use_amp})...")

    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")

        # Train (no gate L1 since this is a deploy model without gates)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = nn.functional.cross_entropy(outputs, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Evaluate
        val_acc, val_auc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, DEVICE)
        test_acc, test_auc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, DEVICE)

        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"  ✓ New best validation accuracy: {val_acc:.4f}")

        all_metrics.append({
            'trial': trial_num,
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'params_m': count_parameters(model),
            'lr': optimizer.param_groups[0]['lr']
        })

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch}")

    # Measure inference energy
    print("\nMeasuring inference energy...")
    inference_tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_small_model_inference_trial{trial_num}")
    test_acc, test_auc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, DEVICE)
    inference_energy = stop_energy_tracker(inference_tracker, save_dir, f"{dataset_name}_{model_name}_small_model_inference_trial{trial_num}")

    # Stop training energy tracker
    training_energy = stop_energy_tracker(tracker, save_dir, f"{dataset_name}_{model_name}_small_model_trial{trial_num}")

    # Save model and metrics
    torch.save(model.state_dict(), os.path.join(save_dir,
               f"{dataset_name}_{model_name}_small_model_trial{trial_num}_final.pth"))
    pd.DataFrame(all_metrics).to_csv(os.path.join(save_dir,
               f"{dataset_name}_{model_name}_small_model_trial{trial_num}_history.csv"), index=False)

    # Final metrics
    final_metrics = {
        'method': 'small_model_from_scratch',
        'dataset': dataset_name,
        'model': f'{model_name}_small',
        'trial': trial_num,
        'total_epochs': FIXED_EPOCHS,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'params_m': count_parameters(model),
        'model_size_mb': model_size_mb(model),
        'flops_g': model.compute_flops() / 1e9,
        'training_energy_kwh': training_energy['energy_kwh'],
        'inference_energy_kwh': inference_energy['energy_kwh'],
        'embed_dim': model_config['embed_dim'],
        'num_heads': model_config['num_heads'],
        'mlp_dim': model_config['mlp_dim'],
        'target_sparsity': model_config['target_sparsity']
    }

    print(f"\n✓ Final Test Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  AUC:       {test_auc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1:        {test_f1:.4f}")
    print(f"  Params:    {count_parameters(model):.2f}M")
    print(f"  FLOPs:     {model.compute_flops() / 1e9:.2f}G")

    cleanup_memory()
    return model, final_metrics

# ==================== METHOD 4: ENHANCED PROGRESSIVE PRUNING ====================

def train_enhanced_progressive_pruning(dataset_name, model_name, train_loader, val_loader, test_loader,
                                        num_classes, save_dir, trial_num):
    """
    METHOD 4: Enhanced Progressive Structured Pruning

    Same pruning schedule and epoch budget as Method 2, with three enhancements:
    1. LR warmup: Linear warmup for the first epoch before cosine decay
    2. LR rewarm: After deployment conversion, reset LR to INITIAL_LR with warmup + cosine
    3. Stronger augmentation: RandAugment + Mixup (alpha=0.2) during training

    The train_loader passed in uses standard augmentation (for importance collection).
    We reload an enhanced train_loader internally for training steps.
    """
    print("\n" + "="*80)
    print(f"METHOD 4: ENHANCED PROGRESSIVE PRUNING - {dataset_name.upper()} - {model_name} - TRIAL {trial_num}")
    print("="*80)

    # Load enhanced training data (RandAugment)
    model_batch_size = MODEL_CONFIGS.get(model_name, {}).get('batch_size', 64)
    npz_path = os.path.join(DATASET_DIR, f"{dataset_name}_224.npz")
    enhanced_train_loader, _, _, _, _ = load_dataset_enhanced(npz_path, batch_size=model_batch_size)

    prune_epochs = PRUNE_EPOCHS_SCHEDULE.copy()
    final_prune_epoch = max(prune_epochs)
    convert_to_deploy_epoch = final_prune_epoch + 1

    print(f"Pruning schedule: {prune_epochs}")
    print(f"Sparsity schedule: {SPARSITY_SCHEDULE}")
    print(f"Convert to deployment after epoch: {final_prune_epoch}")
    print(f"Enhancements: LR warmup, LR rewarm after conversion, RandAugment, Mixup(alpha=0.2)")

    # Create gated model
    model = create_gated_vit_from_timm(model_name, num_classes, pretrained=True).to(DEVICE)
    print(f"Initial parameters: {count_parameters(model):.2f}M")

    # Setup pruners
    pruners = []
    for component_type in [0, 1, 2]:
        modules = prepare_pruning_list(model, component_type)
        if len(modules) > 0:
            pruner = StructuredPruner(model, modules, pruning_momentum=PRUNING_MOMENTUM)
            pruners.append(pruner)

    print(f"Created {len(pruners)} pruning engines")

    # Optimizer with LR warmup + cosine schedule
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    current_lr = INITIAL_LR
    total_steps = FIXED_EPOCHS * len(enhanced_train_loader)
    warmup_steps = len(enhanced_train_loader)  # 1 epoch of warmup

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=MIN_LR
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )
    print(f"LR schedule: linear warmup ({warmup_steps} steps) -> cosine decay")

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if USE_AMP and DEVICE.type == 'cuda' else None

    # Track metrics
    all_metrics = []
    deploy_model = None
    using_deploy = False
    best_deploy_val_acc = 0.0
    best_deploy_state = None
    best_deploy_epoch = 0
    pruning_config = None

    # Start energy tracking
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_enhanced_trial{trial_num}")

    print(f"Training with AMP={scaler is not None}")

    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")

        # Pruning steps (only during gated phase)
        # Use standard train_loader for importance collection (no mixup/randaug)
        if not using_deploy and epoch in prune_epochs:
            prune_step = prune_epochs.index(epoch) + 1
            target_sparsity = SPARSITY_SCHEDULE[prune_step - 1]
            print(f"  [Pruning step {prune_step}/{NUM_PRUNE_STEPS} -> {target_sparsity*100:.0f}% sparsity]")

            # Reset pruners
            for pruner in pruners:
                pruner.iterations_done = 0
                pruner.pruning_scores['score'] = [list() for _ in range(len(pruner.pruning_parameters))]

            # Collect importance using standard loader (clean data)
            collect_importance_scores(model, pruners, train_loader, max_batches=IMPORTANCE_CAL_BATCHES)
            apply_pruning_per_component(model, pruners, target_sparsity=target_sparsity)

            gate_stats = get_gate_sparsity(model)
            print(f"  Gate sparsity: Attn={gate_stats['attn']:.1f}%, MLP={gate_stats['mlp']:.1f}%, Res={gate_stats['res']:.1f}%")

            # Reduce LR after pruning
            current_lr *= LR_REDUCTION_AFTER_PRUNE
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"  LR reduced to {current_lr:.6f}")

        # Convert to deployment model after final pruning
        if epoch == convert_to_deploy_epoch and not using_deploy:
            print(f"  [All pruning complete - converting to deployment model]")

            gate_stats = get_gate_sparsity(model)
            print(f"  Final gate sparsity: Attn={gate_stats['attn']:.1f}%, MLP={gate_stats['mlp']:.1f}%, Res={gate_stats['res']:.1f}%")

            deploy_model, pruning_config = convert_to_deployment_model(model, num_classes)
            using_deploy = True

            # LR REWARM: Reset optimizer with warmup + cosine for remaining epochs
            remaining_epochs = FIXED_EPOCHS - epoch + 1
            remaining_steps = remaining_epochs * len(enhanced_train_loader)
            rewarm_steps = len(enhanced_train_loader)  # 1 epoch of warmup

            optimizer = optim.AdamW(deploy_model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)

            rewarm_warmup = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=rewarm_steps
            )
            rewarm_cosine = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_steps - rewarm_steps, eta_min=MIN_LR
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[rewarm_warmup, rewarm_cosine], milestones=[rewarm_steps]
            )

            print(f"  LR rewarmed to {INITIAL_LR} with warmup ({rewarm_steps} steps) -> cosine")
            print(f"  [Deployment: {count_parameters(deploy_model):.2f}M params (was {count_parameters(model):.2f}M)]")

        # Select active model
        active_model = deploy_model if using_deploy else model
        gate_l1 = 0.0 if using_deploy else GATE_L1_WEIGHT

        # Train with enhanced loop (mixup + RandAugment via enhanced_train_loader)
        train_loss, train_acc = train_one_epoch_enhanced(
            active_model, enhanced_train_loader, optimizer, scheduler, epoch,
            gate_l1_weight=gate_l1, scaler=scaler, mixup_alpha=0.2
        )

        # Evaluate (no mixup during eval)
        val_acc, val_auc, val_precision, val_recall, val_f1 = evaluate_model(active_model, val_loader, DEVICE)
        test_acc, test_auc, test_precision, test_recall, test_f1 = evaluate_model(active_model, test_loader, DEVICE)

        model_type = 'deploy' if using_deploy else 'gated'
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Acc: {val_acc:.4f}, AUC: {val_auc:.4f} [{model_type}]")

        # Track best deployment model
        if using_deploy and val_acc > best_deploy_val_acc:
            best_deploy_val_acc = val_acc
            best_deploy_state = copy.deepcopy(deploy_model.state_dict())
            best_deploy_epoch = epoch
            print(f"  ✓ New best deployment val_acc: {val_acc:.4f}")

        all_metrics.append({
            'trial': trial_num, 'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_acc': val_acc, 'val_auc': val_auc, 'test_acc': test_acc, 'test_auc': test_auc,
            'params_m': count_parameters(active_model), 'lr': optimizer.param_groups[0]['lr'],
            'model_type': model_type
        })

    # Restore best deployment model
    if deploy_model is None:
        print("  [WARNING: No deployment conversion happened, converting now]")
        deploy_model, pruning_config = convert_to_deployment_model(model, num_classes)
    elif best_deploy_state is not None:
        deploy_model.load_state_dict(best_deploy_state)
        print(f"\n✓ Restored best deployment model from epoch {best_deploy_epoch} (val_acc={best_deploy_val_acc:.4f})")

    # Measure inference energy on deployment model
    print("\nMeasuring inference energy on deployment model...")
    deploy_tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_enhanced_deploy_trial{trial_num}")
    test_acc, test_auc, test_precision, test_recall, test_f1 = evaluate_model(deploy_model, test_loader, DEVICE)
    energy_metrics = stop_energy_tracker(deploy_tracker, save_dir, f"{dataset_name}_{model_name}_enhanced_deploy_trial{trial_num}")

    # Stop training energy tracker
    training_energy = stop_energy_tracker(tracker, save_dir, f"{dataset_name}_{model_name}_enhanced_trial{trial_num}")

    # Save
    torch.save(deploy_model.state_dict(), os.path.join(save_dir,
               f"{dataset_name}_{model_name}_enhanced_deploy_trial{trial_num}_final.pth"))
    pd.DataFrame(all_metrics).to_csv(os.path.join(save_dir,
               f"{dataset_name}_{model_name}_enhanced_trial{trial_num}_metrics.csv"), index=False)

    # Get effective dimensions
    avg_heads = np.mean(pruning_config['deploy_per_layer_num_heads'])
    avg_mlp = np.mean(pruning_config['deploy_per_layer_mlp_dim'])

    final_metrics = {
        'method': 'enhanced_progressive_pruning',
        'dataset': dataset_name,
        'model': model_name,
        'trial': trial_num,
        'total_epochs': FIXED_EPOCHS,
        'best_epoch': best_deploy_epoch,
        'best_val_acc': best_deploy_val_acc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'params_m': count_parameters(deploy_model),
        'model_size_mb': model_size_mb(deploy_model),
        'flops_g': deploy_model.compute_flops() / 1e9,
        'training_energy_kwh': training_energy['energy_kwh'],
        'inference_energy_kwh': energy_metrics['energy_kwh'],
        'deploy_embed_dim': pruning_config['deploy_embed_dim'],
        'deploy_avg_heads': avg_heads,
        'deploy_avg_mlp': avg_mlp,
        'total_prune_steps': NUM_PRUNE_STEPS
    }

    print(f"\n✓ Final Test - Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
    print(f"Deployment: {pruning_config['deploy_embed_dim']} embed, {avg_heads:.1f} avg heads, {avg_mlp:.0f} avg MLP")

    cleanup_memory()
    return deploy_model, final_metrics

# ==================== PROCESS DATASET ====================

def process_dataset(dataset_name, model_name, train_loader, val_loader, test_loader, num_classes, save_dir):
    """Process one dataset with one model through all methods, skipping existing experiments"""
    print("\n" + "="*100)
    print(f"PROCESSING: {dataset_name.upper()} with {model_name}")
    print("="*100)

    # Check for existing files
    all_trials_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_all_trials_metrics.csv")
    summary_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_summary_statistics.csv")

    existing_trials = None
    existing_summary = None
    if os.path.exists(all_trials_path):
        existing_trials = pd.read_csv(all_trials_path)
        print(f"Found existing trials data with {len(existing_trials)} rows")
    if os.path.exists(summary_path):
        existing_summary = pd.read_csv(summary_path)
        print(f"Found existing summary with {len(existing_summary)} methods")

    all_baseline_metrics = []
    all_progressive_metrics = []
    all_small_model_metrics = []
    all_enhanced_metrics = []

    # Method configurations: (key, method_name, train_func, results_list)
    method_configs = [
        ('baseline', 'baseline', train_baseline_vit, all_baseline_metrics),
        ('progressive', 'progressive_structured_pruning', train_progressive_structured_pruning, all_progressive_metrics),
        ('small_model', 'small_model_from_scratch', train_small_model_from_scratch, all_small_model_metrics),
        ('enhanced', 'enhanced_progressive_pruning', train_enhanced_progressive_pruning, all_enhanced_metrics),
    ]

    for trial in range(1, NUM_TRIALS + 1):
        print(f"\n{'~'*100}")
        print(f"~ TRIAL {trial}/{NUM_TRIALS}")
        print(f"{'~'*100}")

        trial_seed = SEED + trial * 100
        set_seed(trial_seed)

        for method_key, method_name, train_func, results_list in method_configs:
            exists, existing_path = check_experiment_exists(save_dir, dataset_name, model_name, method_key, trial)

            if exists:
                print(f"\n  Skipping {method_key} trial {trial} (already exists)")

                # Load metrics from existing trials CSV if available
                if existing_trials is not None:
                    trial_data = existing_trials[
                        (existing_trials['method'] == method_name) &
                        (existing_trials['trial'] == trial)
                    ]
                    if len(trial_data) > 0:
                        results_list.append(trial_data.iloc[0].to_dict())
                        print(f"    Loaded metrics from existing CSV")
                    else:
                        print(f"    Warning: No matching metrics found in CSV for {method_key} trial {trial}")
            else:
                # Run the training
                _, metrics = train_func(
                    dataset_name, model_name, train_loader, val_loader, test_loader,
                    num_classes, save_dir, trial
                )
                results_list.append(metrics)

    # Combine all metrics
    all_new_metrics = all_baseline_metrics + all_progressive_metrics + all_small_model_metrics + all_enhanced_metrics

    if all_new_metrics:
        new_metrics_df = pd.DataFrame(all_new_metrics)

        # Handle all_trials file - append new entries
        if existing_trials is not None:
            existing_keys = set(zip(existing_trials['method'], existing_trials['trial']))
            new_rows = [m for m in all_new_metrics if (m['method'], m['trial']) not in existing_keys]
            if new_rows:
                combined_trials = pd.concat([existing_trials, pd.DataFrame(new_rows)], ignore_index=True)
            else:
                combined_trials = existing_trials
        else:
            combined_trials = new_metrics_df

        combined_trials.to_csv(all_trials_path, index=False)

    # Compute summary from collected metrics
    def compute_summary(metrics_list, method_name):
        if not metrics_list:
            return None
        df = pd.DataFrame(metrics_list)
        summary = {'method': method_name, 'dataset': dataset_name, 'model': model_name, 'num_trials': len(df)}
        for col in ['test_acc', 'test_auc', 'test_precision', 'test_recall', 'test_f1',
                    'params_m', 'flops_g', 'inference_energy_kwh', 'training_energy_kwh',
                    'model_size_mb', 'best_val_acc']:
            if col in df.columns:
                summary[f'{col}_mean'] = df[col].mean()
                summary[f'{col}_std'] = df[col].std()
        return summary

    summary_rows = []
    for method_key, method_name, _, results_list in method_configs:
        s = compute_summary(results_list, method_name)
        if s:
            summary_rows.append(s)

    if summary_rows:
        new_summary_df = pd.DataFrame(summary_rows)

        # Merge with existing summary if present
        if existing_summary is not None:
            existing_methods = set(existing_summary['method'])
            new_methods = set(new_summary_df['method'])
            rows_to_keep = existing_summary[~existing_summary['method'].isin(new_methods)]
            final_summary = pd.concat([rows_to_keep, new_summary_df], ignore_index=True)
        else:
            final_summary = new_summary_df

        final_summary.to_csv(summary_path, index=False)

        # Print comparison
        print("\n" + "="*150)
        print(f"SUMMARY - {dataset_name.upper()} - {model_name}")
        print("="*150)
        print(f"{'Metric':<25} {'Baseline (Full)':<30} {'Progressive':<30} {'Small Model':<30} {'Enhanced Prog.':<30}")
        print("-"*165)

        # Get summary rows in order
        base_row = next((r for r in summary_rows if r['method'] == 'baseline'), {})
        prog_row = next((r for r in summary_rows if r['method'] == 'progressive_structured_pruning'), {})
        small_row = next((r for r in summary_rows if r['method'] == 'small_model_from_scratch'), {})
        enhanced_row = next((r for r in summary_rows if r['method'] == 'enhanced_progressive_pruning'), {})

        for col in ['test_acc', 'test_auc', 'test_f1', 'params_m', 'flops_g', 'inference_energy_kwh', 'model_size_mb']:
            vals = []
            for row in [base_row, prog_row, small_row, enhanced_row]:
                mean = row.get(f'{col}_mean', float('nan'))
                std = row.get(f'{col}_std', float('nan'))
                vals.append(f"{mean:.4f}±{std:.4f}")

            print(f"{col:<25} {vals[0]:<30} {vals[1]:<30} {vals[2]:<30} {vals[3]:<30}")
        print("="*165)

    return all_baseline_metrics, all_progressive_metrics, all_small_model_metrics, all_enhanced_metrics

# ==================== MAIN ====================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='ViT Structured Pruning Experiments (ViT-Tiny and ViT-Base)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python simulPruneVit.py --trials 3                    # Run 3 trials per (model, method)
  python simulPruneVit.py --trials 5 --models tiny      # Run 5 trials, only ViT-Tiny
  python simulPruneVit.py --trials 1 --models base      # Quick test with ViT-Base only
  python simulPruneVit.py --datasets bloodmnist         # Run on single dataset

Experiment structure (per dataset):
  - 4 methods: baseline, progressive_pruning, small_model_from_scratch, enhanced_progressive
  - 2 models: vit_tiny_patch16_224, vit_base_patch16_224
  - Total runs per dataset: NUM_TRIALS × 2 models × 4 methods = NUM_TRIALS × 8
        """
    )
    parser.add_argument('--trials', type=int, default=NUM_TRIALS,
                        help=f'Number of trials/replicates per (model, method) combination (default: {NUM_TRIALS})')
    parser.add_argument('--models', type=str, nargs='+', default=['tiny', 'base'],
                        choices=['tiny', 'base'],
                        help='Which models to run: tiny, base, or both (default: both)')
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS,
                        help=f'Datasets to process (default: {DATASETS})')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR,
                        help=f'Directory to save results (default: {SAVE_DIR})')
    return parser.parse_args()


def main():
    global NUM_TRIALS, MODELS_TO_RUN, SAVE_DIR

    args = parse_args()

    # Update global config from args
    NUM_TRIALS = args.trials
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Map model shortcuts to full names
    model_map = {
        'tiny': 'vit_tiny_patch16_224',
        'base': 'vit_base_patch16_224'
    }
    models_to_run = [model_map[m] for m in args.models]
    datasets = args.datasets

    set_seed(SEED)

    print("="*100)
    print("STRUCTURED VISION TRANSFORMER PRUNING WITH DEPLOYMENT CONVERSION")
    print("="*100)

    # Calculate total runs
    total_runs = NUM_TRIALS * len(models_to_run) * 4 * len(datasets)
    print(f"\nExperiment Overview:")
    print(f"  Models: {models_to_run}")
    print(f"  Datasets: {datasets}")
    print(f"  Methods: baseline, progressive_pruning, small_model_from_scratch, enhanced_progressive_pruning")
    print(f"  Trials per (model, method): {NUM_TRIALS}")
    print(f"  Total runs: {NUM_TRIALS} trials x {len(models_to_run)} models x 4 methods x {len(datasets)} datasets = {total_runs}")

    print(f"\nConfiguration (gentle_40pct from ViT_hyper.py):")
    print(f"  Device: {DEVICE}")
    print(f"  Total Epochs: {FIXED_EPOCHS}")
    print(f"  Pruning Schedule: epochs {PRUNE_EPOCHS_SCHEDULE}")
    print(f"  Sparsity Schedule: {SPARSITY_SCHEDULE} (40% max compression)")
    print(f"  Gate L1 Weight: {GATE_L1_WEIGHT}")
    print(f"  LR Reduction After Prune: {LR_REDUCTION_AFTER_PRUNE}")

    print(f"\nMetrics Tracked:")
    print(f"  - Accuracy, AUC, Precision, Recall, F1")
    print(f"  - Parameters (M), FLOPs (G), Model Size (MB)")
    print(f"  - Training Energy (kWh), Inference Energy (kWh) [CodeCarbon]")

    if not CODECARBON_AVAILABLE:
        print(f"\n  WARNING: CodeCarbon not available - energy metrics will be NaN")

    # Collect all results for final summary
    all_results = []

    for dataset in datasets:
        npz_path = os.path.join(DATASET_DIR, f"{dataset}_224.npz")

        if not os.path.exists(npz_path):
            print(f"\nDataset not found: {npz_path}")
            continue

        # Run experiments for each model (with model-specific batch size)
        for model_name in models_to_run:
            # Get model-specific batch size
            model_batch_size = MODEL_CONFIGS.get(model_name, {}).get('batch_size', 64)

            # Load dataset with model-specific batch size
            train_loader, val_loader, test_loader, num_classes, dataset_name = load_dataset(
                npz_path, batch_size=model_batch_size
            )

            print(f"\n" + "#"*100)
            print(f"# RUNNING: {dataset_name} with {model_name} (batch_size={model_batch_size})")
            print(f"# Trials: {NUM_TRIALS}, Methods: 4 (baseline, progressive, small_model, enhanced)")
            print(f"#"*100)

            baseline_metrics, progressive_metrics, small_model_metrics, enhanced_metrics = process_dataset(
                dataset_name, model_name, train_loader, val_loader, test_loader, num_classes, SAVE_DIR
            )

            # Clear GPU memory between models
            cleanup_memory()

            all_results.extend(baseline_metrics)
            all_results.extend(progressive_metrics)
            all_results.extend(small_model_metrics)
            all_results.extend(enhanced_metrics)

    # Save all results
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv(os.path.join(SAVE_DIR, "all_experiments_combined.csv"), index=False)

        # Print final summary table
        print("\n" + "="*120)
        print("FINAL SUMMARY - ALL EXPERIMENTS")
        print("="*120)

        summary_data = []
        for model_name in models_to_run:
            for method in ['baseline', 'progressive_structured_pruning', 'small_model_from_scratch', 'enhanced_progressive_pruning']:
                model_filter = model_name if method != 'small_model_from_scratch' else f'{model_name}_small'
                subset = all_results_df[(all_results_df['model'] == model_filter) &
                                        (all_results_df['method'] == method)]
                if len(subset) > 0:
                    summary_data.append({
                        'model': model_name.replace('_patch16_224', ''),
                        'method': method.replace('_structured_pruning', '').replace('_from_scratch', ''),
                        'test_acc': f"{subset['test_acc'].mean():.4f}±{subset['test_acc'].std():.4f}",
                        'params_m': f"{subset['params_m'].mean():.2f}",
                        'flops_g': f"{subset['flops_g'].mean():.2f}",
                        'train_energy': f"{subset['training_energy_kwh'].mean():.6f}",
                        'infer_energy': f"{subset['inference_energy_kwh'].mean():.6f}"
                    })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print(summary_df.to_string(index=False))

    print("\n" + "="*100)
    print("EXPERIMENT COMPLETED")
    print("="*100)
    print(f"\nMethods Compared:")
    print(f"   1. Baseline: Full model ({FIXED_EPOCHS} epochs)")
    print(f"   2. Progressive Pruning: Prune during training -> convert to deploy -> continue training")
    print(f"   3. Small Model: Same size as pruned, trained from scratch")
    print(f"   4. Enhanced Progressive: Progressive pruning + LR warmup/rewarm + RandAugment + Mixup")
    print(f"\nModels Tested:")
    for model_name in models_to_run:
        cfg = MODEL_CONFIGS.get(model_name, {})
        print(f"   - {model_name}: embed_dim={cfg.get('embed_dim')}, heads={cfg.get('num_heads')}")
    print(f"\nMetrics Tracked:")
    print(f"   - Accuracy, AUC, Precision, Recall, F1")
    print(f"   - Parameters (M), FLOPs (G), Model Size (MB)")
    print(f"   - Training & Inference Energy (kWh) via CodeCarbon")
    print(f"\nResults saved to: {SAVE_DIR}")
    print("="*100)


if __name__ == "__main__":
    main()
