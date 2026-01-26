"""
Vision Transformer Structured Pruning for MedMNIST with Physical Model Compression

This implementation follows the Hikvision paper methodology with complete deployment conversion:
1. Train with gates to identify important components
2. Convert to physically smaller models (actual compression)
3. Measure energy on deployment models (real speedup)

Three methods compared:
- Baseline: No pruning
- Progressive Pruning: Gradual pruning during training
- One-Shot Pruning: Prune at initialization then finetune

FIXES APPLIED:
- L1 regularization on gates to induce sparsity
- Per-component pruning for balanced reduction
- Improved sparsity scheduling
- Gate statistics monitoring

Date: 2026
"""

import os
import sys
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
    print("⚠️  CodeCarbon not available - energy metrics will be NaN")

# ==================== CONFIGURATION ====================

DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/vit_deployment_pruning"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224

# Training configuration
FIXED_EPOCHS = 15
BATCH_SIZE = 64
INITIAL_LR = 1e-4
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-6

# Pruning configuration
WARMUP_EPOCHS = 2
EPOCHS_BETWEEN_PRUNES = 3
NUM_PRUNE_STEPS = 4
TARGET_FLOPS_REDUCTION = 0.5
PRUNING_MOMENTUM = 0.9
IMPORTANCE_CAL_BATCHES = 50
LR_REDUCTION_AFTER_PRUNE = 0.5

# CRITICAL: Gate regularization weight (induces sparsity)
GATE_L1_WEIGHT = 1e-4

# Deployment finetuning (after conversion to smaller model)
DEPLOY_FINETUNE_EPOCHS = 5
DEPLOY_FINETUNE_LR = 5e-5

# Experimental configuration
NUM_TRIALS = 1

os.makedirs(SAVE_DIR, exist_ok=True)

PRUNE_EPOCHS_SCHEDULE = [
    WARMUP_EPOCHS + 1 + i * EPOCHS_BETWEEN_PRUNES 
    for i in range(NUM_PRUNE_STEPS)
]

# Aggressive sparsity schedule (fixes slow pruning)
SPARSITY_SCHEDULE = [0.2, 0.35, 0.45, 0.5]  # Cumulative sparsity per step

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

def print_gate_statistics(model):
    """Print statistics about gate values - CRITICAL for debugging"""
    print("\n" + "="*80)
    print("GATE STATISTICS")
    print("="*80)
    
    gate_info = {}
    for name, param in model.named_parameters():
        if 'gate' in name and 'weight' in name:
            weights = param.data.cpu()
            gate_type = None
            if 'attn_gate' in name:
                gate_type = 'Attention'
            elif 'hidden_gate' in name:
                gate_type = 'MLP'
            elif 'res_gate' in name:
                gate_type = 'Residual'
            
            if gate_type:
                if gate_type not in gate_info:
                    gate_info[gate_type] = []
                gate_info[gate_type].append(weights)
    
    for gate_type, weight_list in gate_info.items():
        all_weights = torch.cat([w.flatten() for w in weight_list])
        print(f"\n{gate_type} Gates:")
        print(f"  Total units: {all_weights.numel()}")
        print(f"  Mean: {all_weights.mean():.4f}, Std: {all_weights.std():.4f}")
        print(f"  Min: {all_weights.min():.4f}, Max: {all_weights.max():.4f}")
        print(f"  Zeros: {(all_weights == 0).sum().item()} ({(all_weights == 0).sum().item() / all_weights.numel() * 100:.1f}%)")
        print(f"  Near-zero (<0.01): {(all_weights.abs() < 0.01).sum().item()} ({(all_weights.abs() < 0.01).sum().item() / all_weights.numel() * 100:.1f}%)")
    
    print("="*80)

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
    def __init__(self, imgs_np, labels_np, img_size=224, is_train=False):
        self.imgs = imgs_np
        self.labels = labels_np
        self.img_size = img_size
        self.is_train = is_train
        
        if is_train:
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

def load_dataset(npz_path):
    """Load dataset from NPZ file"""
    print(f"Loading {npz_path}...")
    data = np.load(npz_path, mmap_mode="r")
    
    X_train, y_train = data["train_images"], data["train_labels"].flatten()
    X_val, y_val = data["val_images"], data["val_labels"].flatten()
    X_test, y_test = data["test_images"], data["test_labels"].flatten()
    
    print(f"Dataset sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    
    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE, is_train=True)
    val_ds = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE, is_train=False)
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=4, pin_memory=True)
    
    num_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    dataset_name = os.path.splitext(os.path.basename(npz_path))[0].replace('_224', '')
    
    return train_loader, val_loader, test_loader, num_classes, dataset_name

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

# ==================== DEPLOYMENT CONVERSION ====================

def convert_to_deployment_model(gated_model, num_classes, save_masks=True, save_dir=None):
    """
    Convert gated training model to deployment model
    This creates a PHYSICALLY SMALLER model
    
    Returns:
        deploy_model: Smaller model without gates
        pruning_config: Configuration dictionary for reproducibility
    """
    print("\n" + "="*80)
    print("CONVERTING TO DEPLOYMENT MODEL (Physical Compression)")
    print("="*80)
    
    # CRITICAL FIX: Use absolute threshold instead of != 0
    GATE_THRESHOLD = 0.01  # Gates below this are considered pruned
    
    # Extract gate masks with debugging
    res_gate_weights = gated_model.res_gate.weight.data
    print(f"\nDEBUG - Residual Gates:")
    print(f"  Weight range: [{res_gate_weights.min():.6f}, {res_gate_weights.max():.6f}]")
    print(f"  Threshold: {GATE_THRESHOLD}")
    print(f"  Weights > threshold: {(res_gate_weights.abs() > GATE_THRESHOLD).sum()}/{len(res_gate_weights)}")
    
    res_mask = (res_gate_weights.abs() > GATE_THRESHOLD).cpu()
    new_embed_dim = res_mask.sum().item()
    
    print(f"  Result: Keeping {new_embed_dim}/{len(res_mask)} embedding dims ({new_embed_dim/len(res_mask)*100:.1f}%)")
    
    head_masks = []
    mlp_masks = []
    
    print(f"\nDEBUG - Per-Block Gates:")
    for i, block in enumerate(gated_model.blocks):
        # Attention heads
        attn_gate_weights = block.attn.attn_gate.weight.data
        head_mask = (attn_gate_weights.abs() > GATE_THRESHOLD).cpu()
        head_masks.append(head_mask)
        
        # MLP hidden dims
        mlp_gate_weights = block.mlp.hidden_gate.weight.data
        mlp_mask = (mlp_gate_weights.abs() > GATE_THRESHOLD).cpu()
        mlp_masks.append(mlp_mask)
        
        if i < 3 or i >= len(gated_model.blocks) - 1:  # Show first 3 and last block
            print(f"  Block {i}:")
            print(f"    Attention: {head_mask.sum()}/{len(head_mask)} heads (weights: [{attn_gate_weights.min():.4f}, {attn_gate_weights.max():.4f}])")
            print(f"    MLP: {mlp_mask.sum()}/{len(mlp_mask)} dims (weights: [{mlp_gate_weights.min():.4f}, {mlp_gate_weights.max():.4f}])")
        elif i == 3:
            print(f"  ... (blocks 3-{len(gated_model.blocks)-2} omitted) ...")
    
    # Calculate per-layer dimensions
    per_layer_num_heads = [mask.sum().item() for mask in head_masks]
    per_layer_mlp_dim = [mask.sum().item() for mask in mlp_masks]
    
    # Original head dimension (doesn't change)
    head_dim = gated_model.embed_dim // gated_model.num_heads
    
    # Report compression
    print(f"\nOriginal architecture:")
    print(f"  Embed dim: {gated_model.embed_dim}")
    print(f"  Num heads: {gated_model.num_heads}")
    print(f"  MLP ratio: {gated_model.mlp_ratio}")
    print(f"  Depth: {gated_model.depth}")
    
    print(f"\nDeployment architecture:")
    print(f"  Embed dim: {new_embed_dim} ({new_embed_dim/gated_model.embed_dim*100:.1f}%)")
    print(f"  Avg heads: {np.mean(per_layer_num_heads):.1f} ({np.mean(per_layer_num_heads)/gated_model.num_heads*100:.1f}%)")
    print(f"  Avg MLP dim: {np.mean(per_layer_mlp_dim):.0f} ({np.mean(per_layer_mlp_dim)/(gated_model.embed_dim*4)*100:.1f}%)")
    
    # CRITICAL CHECK: If nothing was pruned, warn user
    if new_embed_dim == gated_model.embed_dim and all(h == gated_model.num_heads for h in per_layer_num_heads):
        print("\n" + "!"*80)
        print("WARNING: NO PRUNING DETECTED!")
        print("  All gates are above threshold - deployment model will be same size as training model")
        print("  This is expected for baseline method, but indicates a problem for pruning methods")
        print("!"*80)
    
    # Create deployment model
    deploy_model = DeployVisionTransformer(
        img_size=IMG_SIZE,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=new_embed_dim,
        depth=gated_model.depth,
        head_dim=head_dim,
        per_layer_num_heads=per_layer_num_heads,
        per_layer_mlp_dim=per_layer_mlp_dim,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1
    ).to(DEVICE)
    
    # Transfer weights
    print("\nTransferring weights from gated model...")
    
    with torch.no_grad():
        # Transfer patch embedding (only surviving embed dims)
        gated_dict = gated_model.state_dict()
        deploy_dict = deploy_model.state_dict()
        
        # Patch embedding projection
        orig_proj_weight = gated_dict['patch_embed.proj.weight']  # [embed_dim, 3, 16, 16]
        deploy_dict['patch_embed.proj.weight'].copy_(orig_proj_weight[res_mask])
        deploy_dict['patch_embed.proj.bias'].copy_(gated_dict['patch_embed.proj.bias'][res_mask])
        
        # Position embedding
        orig_pos_embed = gated_dict['pos_embed']  # [1, n_patches+1, embed_dim]
        deploy_dict['pos_embed'].copy_(orig_pos_embed[:, :, res_mask])
        
        # CLS token
        orig_cls = gated_dict['cls_token']  # [1, 1, embed_dim]
        deploy_dict['cls_token'].copy_(orig_cls[:, :, res_mask])
        
        # Transfer each block
        for i, (block, head_mask, mlp_mask) in enumerate(zip(gated_model.blocks, head_masks, mlp_masks)):
            if i < 2 or i >= len(gated_model.blocks) - 1:
                print(f"  Block {i}: {per_layer_num_heads[i]} heads, {per_layer_mlp_dim[i]} MLP dims")
            elif i == 2:
                print(f"  ... (blocks 2-{len(gated_model.blocks)-2} pruned similarly) ...")
            
            # Norm layers
            deploy_dict[f'blocks.{i}.norm1.weight'].copy_(gated_dict[f'blocks.{i}.norm1.weight'][res_mask])
            deploy_dict[f'blocks.{i}.norm1.bias'].copy_(gated_dict[f'blocks.{i}.norm1.bias'][res_mask])
            deploy_dict[f'blocks.{i}.norm2.weight'].copy_(gated_dict[f'blocks.{i}.norm2.weight'][res_mask])
            deploy_dict[f'blocks.{i}.norm2.bias'].copy_(gated_dict[f'blocks.{i}.norm2.bias'][res_mask])
            
            # Attention QKV
            orig_qkv_weight = gated_dict[f'blocks.{i}.attn.qkv.weight']  # [3*embed_dim, embed_dim]
            orig_qkv_bias = gated_dict[f'blocks.{i}.attn.qkv.bias']  # [3*embed_dim]
            
            # Split into Q, K, V
            q_w, k_w, v_w = orig_qkv_weight.chunk(3, dim=0)
            q_b, k_b, v_b = orig_qkv_bias.chunk(3, dim=0)
            
            # Reshape to [num_heads, head_dim, embed_dim] for selection
            orig_num_heads = gated_model.num_heads
            q_w = q_w.view(orig_num_heads, head_dim, gated_model.embed_dim)
            k_w = k_w.view(orig_num_heads, head_dim, gated_model.embed_dim)
            v_w = v_w.view(orig_num_heads, head_dim, gated_model.embed_dim)
            q_b = q_b.view(orig_num_heads, head_dim)
            k_b = k_b.view(orig_num_heads, head_dim)
            v_b = v_b.view(orig_num_heads, head_dim)
            
            # Select surviving heads and embed dims
            q_w = q_w[head_mask][:, :, res_mask]
            k_w = k_w[head_mask][:, :, res_mask]
            v_w = v_w[head_mask][:, :, res_mask]
            q_b = q_b[head_mask]
            k_b = k_b[head_mask]
            v_b = v_b[head_mask]
            
            # Concatenate back
            new_qkv_weight = torch.cat([q_w.flatten(0, 1), k_w.flatten(0, 1), v_w.flatten(0, 1)], dim=0)
            new_qkv_bias = torch.cat([q_b.flatten(), k_b.flatten(), v_b.flatten()], dim=0)
            
            deploy_dict[f'blocks.{i}.attn.qkv.weight'].copy_(new_qkv_weight)
            deploy_dict[f'blocks.{i}.attn.qkv.bias'].copy_(new_qkv_bias)
            
            # Attention projection
            orig_proj_weight = gated_dict[f'blocks.{i}.attn.proj.weight']  # [embed_dim, embed_dim]
            orig_proj_weight = orig_proj_weight.view(gated_model.embed_dim, orig_num_heads, head_dim)
            new_proj_weight = orig_proj_weight[res_mask][:, head_mask, :].flatten(1)
            deploy_dict[f'blocks.{i}.attn.proj.weight'].copy_(new_proj_weight)
            deploy_dict[f'blocks.{i}.attn.proj.bias'].copy_(gated_dict[f'blocks.{i}.attn.proj.bias'][res_mask])
            
            # MLP fc1
            orig_fc1_weight = gated_dict[f'blocks.{i}.mlp.fc1.weight']  # [mlp_dim, embed_dim]
            new_fc1_weight = orig_fc1_weight[mlp_mask][:, res_mask]
            deploy_dict[f'blocks.{i}.mlp.fc1.weight'].copy_(new_fc1_weight)
            deploy_dict[f'blocks.{i}.mlp.fc1.bias'].copy_(gated_dict[f'blocks.{i}.mlp.fc1.bias'][mlp_mask])
            
            # MLP fc2
            orig_fc2_weight = gated_dict[f'blocks.{i}.mlp.fc2.weight']  # [embed_dim, mlp_dim]
            new_fc2_weight = orig_fc2_weight[res_mask][:, mlp_mask]
            deploy_dict[f'blocks.{i}.mlp.fc2.weight'].copy_(new_fc2_weight)
            deploy_dict[f'blocks.{i}.mlp.fc2.bias'].copy_(gated_dict[f'blocks.{i}.mlp.fc2.bias'][res_mask])
        
        # Final norm
        deploy_dict['norm.weight'].copy_(gated_dict['norm.weight'][res_mask])
        deploy_dict['norm.bias'].copy_(gated_dict['norm.bias'][res_mask])
        
        # Head
        orig_head_weight = gated_dict['head.weight']  # [num_classes, embed_dim]
        deploy_dict['head.weight'].copy_(orig_head_weight[:, res_mask])
        deploy_dict['head.bias'].copy_(gated_dict['head.bias'])
    
    print("✓ Weight transfer complete")
    
    # Create pruning config for reproducibility
    pruning_config = {
        'original_embed_dim': gated_model.embed_dim,
        'original_num_heads': gated_model.num_heads,
        'original_mlp_ratio': gated_model.mlp_ratio,
        'deploy_embed_dim': new_embed_dim,
        'deploy_per_layer_num_heads': per_layer_num_heads,
        'deploy_per_layer_mlp_dim': per_layer_mlp_dim,
        'head_masks': [mask.numpy() for mask in head_masks],
        'mlp_masks': [mask.numpy() for mask in mlp_masks],
        'res_mask': res_mask.numpy()
    }
    
    # Save masks if requested
    if save_masks and save_dir is not None:
        mask_path = os.path.join(save_dir, 'pruning_config.pt')
        torch.save(pruning_config, mask_path)
        print(f"✓ Pruning config saved to {mask_path}")
    
    # Report size reduction
    orig_params = count_parameters(gated_model)
    deploy_params = count_parameters(deploy_model)
    print(f"\nParameter reduction: {orig_params:.2f}M → {deploy_params:.2f}M ({deploy_params/orig_params*100:.1f}%)")
    
    orig_size = model_size_mb(gated_model)
    deploy_size = model_size_mb(deploy_model)
    print(f"Model size reduction: {orig_size:.2f}MB → {deploy_size:.2f}MB ({deploy_size/orig_size*100:.1f}%)")
    
    orig_flops = compute_flops_vit(gated_model.embed_dim, gated_model.num_heads, 
                                    gated_model.depth, gated_model.mlp_ratio)
    deploy_flops = deploy_model.compute_flops()
    print(f"FLOPs reduction: {orig_flops/1e9:.2f}G → {deploy_flops/1e9:.2f}G ({deploy_flops/orig_flops*100:.1f}%)")
    
    print("="*80)
    
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
            score.cpu().numpy() if torch.is_tensor(score) else score
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

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, gate_l1_weight=0.0):
    """
    Training epoch with L1 regularization on gates
    FIX: Adds sparsity-inducing penalty to force gates toward zero
    """
    model.train()
    running_loss = 0.0
    running_gate_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        task_loss = nn.functional.cross_entropy(outputs, labels)
        
        # CRITICAL FIX: Add L1 regularization to gates
        gate_l1_loss = 0.0
        if gate_l1_weight > 0:
            for name, param in model.named_parameters():
                if 'gate' in name and 'weight' in name:
                    gate_l1_loss += torch.abs(param).sum()
        
        total_loss = task_loss + gate_l1_weight * gate_l1_loss
        
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
    
    # Track metrics
    history = []
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Start energy tracking
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_baseline_trial{trial_num}")
    
    print(f"\nTraining for {FIXED_EPOCHS} epochs...")
    
    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, gate_l1_weight=0.0)
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
    METHOD 2: Progressive Structured Pruning
    Converts to deployment model after pruning
    """
    print("\n" + "="*80)
    print(f"METHOD 2: PROGRESSIVE STRUCTURED PRUNING - {dataset_name.upper()} - {model_name} - TRIAL {trial_num}")
    print("="*80)
    
    prune_epochs = PRUNE_EPOCHS_SCHEDULE.copy()
    print(f"Pruning schedule: {prune_epochs}")
    print(f"Sparsity schedule: {SPARSITY_SCHEDULE}")
    
    # Create model
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
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    current_lr = INITIAL_LR
    
    # Track metrics
    all_metrics = []
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Start energy tracking
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_progressive_trial{trial_num}")
    
    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")
        
        # Check if pruning epoch
        if epoch in prune_epochs:
            prune_step = prune_epochs.index(epoch) + 1
            print(f"\n*** PRUNING STEP {prune_step}/{NUM_PRUNE_STEPS} ***")
            
            # Reset pruners
            for pruner in pruners:
                pruner.iterations_done = 0
                pruner.pruning_scores['score'] = [list() for _ in range(len(pruner.pruning_parameters))]
            
            # Collect importance
            collect_importance_scores(model, pruners, train_loader, max_batches=IMPORTANCE_CAL_BATCHES)
            
            # FIX: Use aggressive sparsity schedule
            target_sparsity = SPARSITY_SCHEDULE[prune_step - 1]
            
            # FIX: Apply per-component pruning
            apply_pruning_per_component(model, pruners, target_sparsity=target_sparsity)
            
            # Debug: Print gate statistics
            print_gate_statistics(model)
            
            # Reduce LR
            current_lr *= LR_REDUCTION_AFTER_PRUNE
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"  LR reduced to {current_lr:.6f}")
        
        # Setup scheduler
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch, eta_min=MIN_LR)
        
        # Train with gate regularization
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, epoch, 
            gate_l1_weight=GATE_L1_WEIGHT
        )
        
        # Validate
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
            'trial': trial_num, 'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_acc': val_acc, 'test_acc': test_acc, 'params_m': count_parameters(model), 'lr': current_lr
        })
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch}")
    
    # Final gate statistics
    print("\nFinal gate statistics before deployment conversion:")
    print_gate_statistics(model)
    
    # Convert to deployment model
    deploy_model, pruning_config = convert_to_deployment_model(
        model, num_classes, save_masks=True, save_dir=save_dir
    )
    
    # Finetune deployment model
    deploy_model = finetune_deployment_model(
        deploy_model, train_loader, val_loader, 
        epochs=DEPLOY_FINETUNE_EPOCHS, lr=DEPLOY_FINETUNE_LR
    )
    
    # Measure energy on deployment model
    print("\n📊 Measuring energy on deployment model...")
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

# ==================== METHOD 3: ONE-SHOT PRUNING ====================

def train_prune_then_finetune(dataset_name, model_name, train_loader, val_loader, test_loader,
                              num_classes, save_dir, trial_num):
    """
    METHOD 3: One-Shot Pruning then Finetune
    Converts to deployment model after pruning
    """
    print("\n" + "="*80)
    print(f"METHOD 3: ONE-SHOT PRUNE-THEN-TRAIN - {dataset_name.upper()} - {model_name} - TRIAL {trial_num}")
    print("="*80)
    
    # Create model
    model = create_gated_vit_from_timm(model_name, num_classes, pretrained=True).to(DEVICE)
    print(f"Initial parameters: {count_parameters(model):.2f}M")
    
    # Setup pruners
    pruners = []
    for component_type in [0, 1, 2]:
        modules = prepare_pruning_list(model, component_type)
        if len(modules) > 0:
            pruner = StructuredPruner(model, modules, pruning_momentum=PRUNING_MOMENTUM)
            pruners.append(pruner)
    
    # Start energy tracking (includes pruning)
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_prune_then_train_trial{trial_num}")
    
    # ONE-SHOT PRUNING
    print("\n*** ONE-SHOT PRUNING ***")
    collect_importance_scores(model, pruners, train_loader, max_batches=IMPORTANCE_CAL_BATCHES)
    
    target_sparsity = TARGET_FLOPS_REDUCTION
    apply_pruning_per_component(model, pruners, target_sparsity=target_sparsity)
    
    # Debug: Print gate statistics
    print_gate_statistics(model)
    
    print(f"\nNow training pruned model for {FIXED_EPOCHS} epochs...")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    total_steps = FIXED_EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)
    
    # Track metrics
    all_metrics = []
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, epoch,
            gate_l1_weight=GATE_L1_WEIGHT
        )
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
            'trial': trial_num, 'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_acc': val_acc, 'test_acc': test_acc, 'params_m': count_parameters(model)
        })
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch}")
    
    # Final gate statistics
    print("\nFinal gate statistics before deployment conversion:")
    print_gate_statistics(model)
    
    # Convert to deployment model
    deploy_model, pruning_config = convert_to_deployment_model(
        model, num_classes, save_masks=True, save_dir=save_dir
    )
    
    # Finetune deployment model
    deploy_model = finetune_deployment_model(
        deploy_model, train_loader, val_loader,
        epochs=DEPLOY_FINETUNE_EPOCHS, lr=DEPLOY_FINETUNE_LR
    )
    
    # Measure energy on deployment model
    print("\n📊 Measuring energy on deployment model...")
    deploy_tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_prune_then_train_deploy_trial{trial_num}")
    test_acc, test_auc, test_precision, test_recall, test_f1 = evaluate_model(deploy_model, test_loader, DEVICE)
    energy_metrics = stop_energy_tracker(deploy_tracker, save_dir, f"{dataset_name}_{model_name}_prune_then_train_deploy_trial{trial_num}")
    
    # Stop training energy tracker
    training_energy = stop_energy_tracker(tracker, save_dir, f"{dataset_name}_{model_name}_prune_then_train_trial{trial_num}")
    
    # Save
    torch.save(deploy_model.state_dict(), os.path.join(save_dir,
               f"{dataset_name}_{model_name}_prune_then_train_deploy_trial{trial_num}_final.pth"))
    pd.DataFrame(all_metrics).to_csv(os.path.join(save_dir,
               f"{dataset_name}_{model_name}_prune_then_train_trial{trial_num}_metrics.csv"), index=False)
    
    # Get effective dimensions
    avg_heads = np.mean(pruning_config['deploy_per_layer_num_heads'])
    avg_mlp = np.mean(pruning_config['deploy_per_layer_mlp_dim'])
    
    final_metrics = {
        'method': 'prune_then_train',
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
        'inference_energy_kwh': energy_metrics['energy_kwh'],
        'deploy_embed_dim': pruning_config['deploy_embed_dim'],
        'deploy_avg_heads': avg_heads,
        'deploy_avg_mlp': avg_mlp
    }
    
    print(f"\n✓ Final Test - Acc: {test_acc:.4f}, Precision: {test_precision:.4f}")
    print(f"Deployment: {pruning_config['deploy_embed_dim']} embed, {avg_heads:.1f} avg heads, {avg_mlp:.0f} avg MLP")
    
    cleanup_memory()
    return deploy_model, final_metrics

# ==================== PROCESS DATASET ====================

def process_dataset(dataset_name, train_loader, val_loader, test_loader, num_classes, save_dir):
    """Process one dataset through all methods"""
    print("\n" + "="*100)
    print(f"PROCESSING DATASET: {dataset_name.upper()}")
    print("="*100)
    
    all_baseline_metrics = []
    all_progressive_metrics = []
    all_prune_then_train_metrics = []
    
    for trial in range(1, NUM_TRIALS + 1):
        print(f"\n{'~'*100}")
        print(f"~ TRIAL {trial}/{NUM_TRIALS}")
        print(f"{'~'*100}")
        
        trial_seed = SEED + trial * 100
        set_seed(trial_seed)
        
        # Method 2: Progressive
        _, progressive_metrics = train_progressive_structured_pruning(
            dataset_name, 'vit_tiny_patch16_224', train_loader, val_loader, test_loader,
            num_classes, save_dir, trial
        )
        all_progressive_metrics.append(progressive_metrics)
        
        # Method 1: Baseline
        _, baseline_metrics = train_baseline_vit(
            dataset_name, 'vit_tiny_patch16_224', train_loader, val_loader, test_loader,
            num_classes, save_dir, trial
        )
        all_baseline_metrics.append(baseline_metrics)
        
        # Method 3: Prune-then-train
        _, prune_then_train_metrics = train_prune_then_finetune(
            dataset_name, 'vit_tiny_patch16_224', train_loader, val_loader, test_loader,
            num_classes, save_dir, trial
        )
        all_prune_then_train_metrics.append(prune_then_train_metrics)
    
    # Save combined results
    all_metrics_df = pd.DataFrame(all_baseline_metrics + all_progressive_metrics + all_prune_then_train_metrics)
    all_metrics_df.to_csv(os.path.join(save_dir, f"{dataset_name}_all_trials_metrics.csv"), index=False)
    
    # Compute summary
    baseline_df = pd.DataFrame(all_baseline_metrics)
    progressive_df = pd.DataFrame(all_progressive_metrics)
    prune_then_train_df = pd.DataFrame(all_prune_then_train_metrics)
    
    summary_rows = []
    
    for df, method_name in [(baseline_df, 'baseline'), 
                            (progressive_df, 'progressive_structured_pruning'),
                            (prune_then_train_df, 'prune_then_train')]:
        summary = {'method': method_name, 'dataset': dataset_name, 'num_trials': NUM_TRIALS}
        for col in ['test_acc', 'test_precision', 'params_m', 'flops_g', 
                    'inference_energy_kwh', 'model_size_mb']:
            if col in df.columns:
                summary[f'{col}_mean'] = df[col].mean()
                summary[f'{col}_std'] = df[col].std()
        summary_rows.append(summary)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(save_dir, f"{dataset_name}_summary_statistics.csv"), index=False)
    
    # Print comparison
    print("\n" + "="*140)
    print(f"SUMMARY - {dataset_name.upper()}")
    print("="*140)
    print(f"{'Metric':<30} {'Baseline':<35} {'Progressive':<35} {'Prune-Then-Train':<35}")
    print("-"*140)
    for col in ['test_acc', 'params_m', 'flops_g', 'inference_energy_kwh', 'model_size_mb']:
        base_mean = summary_rows[0].get(f'{col}_mean', float('nan'))
        base_std = summary_rows[0].get(f'{col}_std', float('nan'))
        prog_mean = summary_rows[1].get(f'{col}_mean', float('nan'))
        prog_std = summary_rows[1].get(f'{col}_std', float('nan'))
        ptt_mean = summary_rows[2].get(f'{col}_mean', float('nan'))
        ptt_std = summary_rows[2].get(f'{col}_std', float('nan'))
        
        print(f"{col:<30} {base_mean:.4f}±{base_std:.4f}  "
              f"{prog_mean:.4f}±{prog_std:.4f}  "
              f"{ptt_mean:.4f}±{ptt_std:.4f}")
    print("="*140)

# ==================== MAIN ====================

def main():
    set_seed(SEED)
    
    print("="*100)
    print("STRUCTURED VISION TRANSFORMER PRUNING WITH DEPLOYMENT CONVERSION")
    print("="*100)
    print(f"\n⚡ Key Feature: Physical model compression with deployment conversion")
    print(f"   - Training models use gates to identify important components")
    print(f"   - Deployment models are PHYSICALLY SMALLER (actual speedup)")
    print(f"   - Energy measured on deployment models (real-world applicable)")
    print(f"\n🔧 FIXES APPLIED:")
    print(f"   - L1 regularization on gates (weight={GATE_L1_WEIGHT})")
    print(f"   - Per-component pruning for balanced reduction")
    print(f"   - Aggressive sparsity schedule: {SPARSITY_SCHEDULE}")
    print(f"   - Gate statistics monitoring")
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Fixed Epochs: {FIXED_EPOCHS}")
    print(f"  Pruning Schedule: {PRUNE_EPOCHS_SCHEDULE}")
    print(f"  Target FLOPs Reduction: {TARGET_FLOPS_REDUCTION:.1%}")
    print(f"  Deployment Finetune Epochs: {DEPLOY_FINETUNE_EPOCHS}")
    print(f"  Number of Trials: {NUM_TRIALS}")
    
    datasets = ['bloodmnist', 'pathmnist', 'dermamnist']
    
    for dataset in datasets:
        npz_path = os.path.join(DATASET_DIR, f"{dataset}_224.npz")
        
        if not os.path.exists(npz_path):
            print(f"\n⚠️  Dataset not found: {npz_path}")
            continue
        
        train_loader, val_loader, test_loader, num_classes, dataset_name = load_dataset(npz_path)
        process_dataset(dataset_name, train_loader, val_loader, test_loader, num_classes, SAVE_DIR)
    
    print("\n" + "="*100)
    print("✅ EXPERIMENT COMPLETED")
    print("="*100)
    print(f"\n🎯 Key Results:")
    print(f"   - All methods create deployment models")
    print(f"   - Baseline: Full-size deployment model")
    print(f"   - Progressive/One-shot: Physically smaller deployment models")
    print(f"   - Energy measured on actual deployment models")
    print(f"   - Gate regularization ensures actual pruning occurs")
    print(f"   - Results saved to: {SAVE_DIR}")
    print("="*100)

if __name__ == "__main__":
    main()
