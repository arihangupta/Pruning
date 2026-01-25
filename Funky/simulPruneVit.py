"""
Vision Transformer Structured Pruning for MedMNIST
Implements Hikvision-style structured pruning with three methods:
1. Baseline - No pruning
2. Progressive Structured Pruning - Gradual Taylor-based pruning with gates
3. One-Shot EA Pruning - Evolutionary algorithm search at initialization
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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import pandas as pd
import copy
import gc
import math
import pickle
from functools import partial

# Try codecarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("CodeCarbon not available - energy metrics will be NaN")

# ==================== CONFIGURATION ====================

DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/vit_structured_pruning"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224

# Training configuration
FIXED_EPOCHS = 15
BATCH_SIZE = 64
INITIAL_LR = 1e-4
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-6

# Structured pruning configuration (matching Hikvision paper)
WARMUP_EPOCHS = 2
EPOCHS_BETWEEN_PRUNES = 3
NUM_PRUNE_STEPS = 4  # Prune at epochs 3, 6, 9, 12
TARGET_FLOPS_REDUCTION = 0.5  # Target 50% FLOPs reduction
PRUNING_MOMENTUM = 0.9
IMPORTANCE_CAL_BATCHES = 50
LR_REDUCTION_AFTER_PRUNE = 0.5

# EA configuration
USE_HESSIAN = False  # Set to True if you want Hessian-aware EA (slower)
EA_POPULATION_SIZE = 50
EA_GENERATIONS = 30
EA_N_KIDS = 25

# Experimental configuration
NUM_TRIALS = 1

os.makedirs(SAVE_DIR, exist_ok=True)

# Calculate pruning schedule
PRUNE_EPOCHS_SCHEDULE = [WARMUP_EPOCHS + 1 + i * EPOCHS_BETWEEN_PRUNES for i in range(NUM_PRUNE_STEPS)]

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

# ==================== GATE LAYER ====================

class GateLayer(nn.Module):
    """Gating layer for structured pruning (from Hikvision paper)"""
    def __init__(self, input_features, output_features, size_mask):
        super(GateLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.size_mask = size_mask
        self.weight = nn.Parameter(torch.ones(output_features))
        self.do_not_update = True  # For easy identification
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
    """ViT with structured pruning gates"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.img_size = img_size
        
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
        
        # SHARED residual gate (critical detail from Hikvision paper!)
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
        
        x = self.res_gate(x)  # Initial gate application
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x[:, 0]  # Return CLS token
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    def flops(self, component=2):
        """
        Calculate FLOPs considering active gates
        component: 0=attn only, 1=mlp only, 2=total
        """
        n_patches = (self.img_size // self.patch_size) ** 2 + 1  # +1 for cls
        total_flops = 0
        
        for block in self.blocks:
            # Count active components
            active_heads = (block.attn.attn_gate.weight != 0).sum().item()
            active_hidden = (block.mlp.hidden_gate.weight != 0).sum().item()
            active_res = (block.res_gate.weight != 0).sum().item()
            
            if component in [0, 2]:  # Attention FLOPs
                # QKV: 3 * N * D * D
                # Attention: N^2 * D  
                # Output: N * D * D
                attn_flops = (
                    3 * n_patches * self.embed_dim * self.embed_dim +
                    n_patches * n_patches * self.embed_dim +
                    n_patches * self.embed_dim * self.embed_dim
                )
                attn_flops *= (active_heads / self.num_heads)
                total_flops += attn_flops
            
            if component in [1, 2]:  # MLP FLOPs
                mlp_hidden = int(self.embed_dim * 4)
                mlp_flops = (
                    n_patches * self.embed_dim * active_hidden +
                    n_patches * active_hidden * self.embed_dim
                )
                total_flops += mlp_flops
        
        return torch.tensor(total_flops)

def create_gated_vit_from_timm(model_name, num_classes, pretrained=True):
    """Load pretrained TIMM weights into gated architecture"""
    print(f"Creating gated {model_name} (pretrained={pretrained})...")
    
    # Map model names to architectures
    configs = {
        'vit_tiny_patch16_224': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
        'vit_small_patch16_224': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'vit_base_patch16_224': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}")
    
    cfg = configs[model_name]
    
    # Create gated model
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
        # Load TIMM weights
        print("  Loading pretrained weights from TIMM...")
        timm_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # Transfer weights (matching components)
        model_dict = model.state_dict()
        timm_dict = timm_model.state_dict()
        
        # Transfer patch embedding, pos embedding, cls token
        transfer_keys = ['patch_embed.proj.weight', 'patch_embed.proj.bias',
                        'pos_embed', 'cls_token', 'norm.weight', 'norm.bias',
                        'head.weight', 'head.bias']
        
        for key in transfer_keys:
            if key in timm_dict and key in model_dict:
                model_dict[key].copy_(timm_dict[key])
        
        # Transfer block weights (attn and mlp, but not gates)
        for i in range(cfg['depth']):
            for component in ['norm1', 'norm2', 'attn.qkv', 'attn.proj', 
                            'mlp.fc1', 'mlp.fc2']:
                src_key = f'blocks.{i}.{component}.weight'
                if src_key in timm_dict and src_key in model_dict:
                    model_dict[src_key].copy_(timm_dict[src_key])
                
                src_key_bias = f'blocks.{i}.{component}.bias'
                if src_key_bias in timm_dict and src_key_bias in model_dict:
                    model_dict[src_key_bias].copy_(timm_dict[src_key_bias])
        
        print("  ✓ Pretrained weights loaded")
    
    return model

# ==================== TAYLOR IMPORTANCE SCORERS ====================

def forward_hook(self, input, output):
    """Hook to store output"""
    self.output = output.detach()

def backward_hook(self, grad_input, grad_output):
    """Hook to store gradient"""
    self.grad = (grad_output[0].detach(),)

def taylor2Scorer(gate_module):
    """
    Method 22 from CVPR2019 paper (best in their experiments)
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
        if hasattr(m, "do_not_update"):  # Is a GateLayer
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
    """Structured pruning engine (simplified from Hikvision BaseUnitPruner)"""
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
        
        # Convert to CPU for easier manipulation
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

# ==================== ENERGY & DATASET (SAME AS BEFORE) ====================

def start_energy_tracker(save_dir, project_name):
    if not CODECARBON_AVAILABLE:
        return None
    tracker = EmissionsTracker(project_name=project_name, output_dir=save_dir, log_level="error")
    tracker.start()
    return tracker

def stop_energy_tracker(tracker, save_dir, project_name):
    if tracker is None:
        return {'energy_kwh': float('nan'), 'emissions_kg': float('nan'), 'duration_s': float('nan')}
    emissions = tracker.stop()
    return {'energy_kwh': emissions, 'emissions_kg': emissions * 0.475, 
            'duration_s': tracker._total_duration.total_seconds() if hasattr(tracker, '_total_duration') else 0}

class NumpyMemmapDataset(Dataset):
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
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
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
    print(f"Loading {npz_path}...")
    data = np.load(npz_path, mmap_mode="r")
    
    X_train, y_train = data["train_images"], data["train_labels"].flatten()
    X_val, y_val = data["val_images"], data["val_labels"].flatten()
    X_test, y_test = data["test_images"], data["test_labels"].flatten()
    
    print(f"Dataset sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    
    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE, is_train=True)
    val_ds = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE, is_train=False)
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True)
    
    num_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    dataset_name = os.path.splitext(os.path.basename(npz_path))[0].replace('_224', '')
    
    return train_loader, val_loader, test_loader, num_classes, dataset_name

def evaluate_model(net, test_loader, device):
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

# ==================== TRAINING FUNCTIONS ====================

def collect_importance_scores(model, pruners, train_loader, max_batches=50):
    """Phase 1: Collect importance scores (eval mode, no weight updates)"""
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

def apply_pruning_simple(model, pruners, target_sparsity=0.5):
    """
    Simple pruning: calculate global threshold across all components
    """
    print(f"\nApplying pruning (target sparsity: {target_sparsity:.1%})...")
    
    # Finalize scores
    all_criteria = []
    for pruner in pruners:
        criteria = pruner.finalize_scores()
        all_criteria.extend([c for layer in criteria for c in layer])
    
    # Calculate threshold
    all_criteria_array = np.array(all_criteria)
    num_to_prune = int(len(all_criteria_array) * target_sparsity)
    threshold = np.sort(all_criteria_array)[num_to_prune]
    
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Pruning {num_to_prune}/{len(all_criteria_array)} units")
    
    # Apply pruning
    for pruner in pruners:
        criteria = pruner.finalize_scores()
        pruner.prune(criteria, threshold)
    
    # Report
    for i, pruner in enumerate(pruners):
        active = pruner.get_num_active()
        total = pruner.all_neuron_units
        print(f"  Component {i}: {active}/{total} active ({active/total*100:.1f}%)")

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch):
    """Standard training epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.functional.cross_entropy(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), correct / total

# ==================== METHOD 1: BASELINE ====================

def train_baseline_vit(dataset_name, model_name, train_loader, val_loader, test_loader,
                       num_classes, save_dir, trial_num):
    """METHOD 1: Baseline (no pruning)"""
    print("\n" + "="*80)
    print(f"METHOD 1: BASELINE - {dataset_name.upper()} - {model_name} - TRIAL {trial_num}")
    print("="*80)
    
    # Create gated model (gates stay at 1.0)
    model = create_gated_vit_from_timm(model_name, num_classes, pretrained=True).to(DEVICE)
    print(f"Parameters: {count_parameters(model):.2f}M")
    
    # Freeze gates (they don't train in baseline)
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
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
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
    
    energy_metrics = stop_energy_tracker(tracker, save_dir, f"{dataset_name}_{model_name}_baseline_trial{trial_num}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch}")
    
    # Save
    torch.save(model.state_dict(), os.path.join(save_dir, 
               f"{dataset_name}_{model_name}_baseline_trial{trial_num}_final.pth"))
    pd.DataFrame(history).to_csv(os.path.join(save_dir,
               f"{dataset_name}_{model_name}_baseline_trial{trial_num}_history.csv"), index=False)
    
    # Final test
    test_acc, test_auc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, DEVICE)
    
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
        'params_m': count_parameters(model),
        'model_size_mb': model_size_mb(model),
        'flops_g': model.flops(2).item() / 1e9,
        'training_energy_kwh': energy_metrics['energy_kwh']
    }
    
    print(f"\nFinal Test - Acc: {test_acc:.4f}, Precision: {test_precision:.4f}")
    
    return model, final_metrics

# ==================== METHOD 2: PROGRESSIVE STRUCTURED PRUNING ====================

def train_progressive_structured_pruning(dataset_name, model_name, train_loader, val_loader, test_loader,
                                        num_classes, save_dir, trial_num):
    """METHOD 2: Progressive Structured Pruning with Taylor scores"""
    print("\n" + "="*80)
    print(f"METHOD 2: PROGRESSIVE STRUCTURED PRUNING - {dataset_name.upper()} - {model_name} - TRIAL {trial_num}")
    print("="*80)
    
    prune_epochs = PRUNE_EPOCHS_SCHEDULE.copy()
    print(f"Pruning schedule: {prune_epochs}")
    
    # Create model
    model = create_gated_vit_from_timm(model_name, num_classes, pretrained=True).to(DEVICE)
    print(f"Initial parameters: {count_parameters(model):.2f}M")
    print(f"Initial FLOPs: {model.flops(2).item() / 1e9:.2f} GFLOPs")
    
    # Setup pruners for each component
    pruners = []
    for component_type in [0, 1, 2]:  # attention, mlp, residual
        modules = prepare_pruning_list(model, component_type)
        if len(modules) > 0:
            pruner = StructuredPruner(model, modules, pruning_momentum=PRUNING_MOMENTUM)
            pruners.append(pruner)
    
    print(f"Created {len(pruners)} pruning engines")
    
    # Optimizer (gates will be pruned, other params trained)
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    current_lr = INITIAL_LR
    
    # Track metrics
    all_metrics = []
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Start energy tracking
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_progressive_trial{trial_num}")
    
    cumulative_sparsity = 0.0
    
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
            
            # Calculate cumulative sparsity
            cumulative_sparsity += (1 - cumulative_sparsity) * (TARGET_FLOPS_REDUCTION / NUM_PRUNE_STEPS)
            
            # Apply pruning
            apply_pruning_simple(model, pruners, target_sparsity=cumulative_sparsity)
            
            # Reduce LR
            current_lr *= LR_REDUCTION_AFTER_PRUNE
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"  LR reduced to {current_lr:.6f}")
            
            # Report FLOPs
            current_flops = model.flops(2).item() / 1e9
            print(f"  Current FLOPs: {current_flops:.2f} GFLOPs")
        
        # Setup scheduler
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch, eta_min=MIN_LR)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
        
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
    
    energy_metrics = stop_energy_tracker(tracker, save_dir, f"{dataset_name}_{model_name}_progressive_trial{trial_num}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch}")
    
    # Save
    torch.save(model.state_dict(), os.path.join(save_dir,
               f"{dataset_name}_{model_name}_progressive_trial{trial_num}_final.pth"))
    pd.DataFrame(all_metrics).to_csv(os.path.join(save_dir,
               f"{dataset_name}_{model_name}_progressive_trial{trial_num}_metrics.csv"), index=False)
    
    # Final test
    test_acc, test_auc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, DEVICE)
    
    # Count effective dimensions
    total_heads, active_heads = 0, 0
    total_mlp, active_mlp = 0, 0
    for block in model.blocks:
        total_heads += model.num_heads
        active_heads += (block.attn.attn_gate.weight != 0).sum().item()
        mlp_dim = block.mlp.hidden_gate.weight.shape[0]
        total_mlp += mlp_dim
        active_mlp += (block.mlp.hidden_gate.weight != 0).sum().item()
    
    effective_heads = active_heads / len(model.blocks)
    effective_mlp = active_mlp / len(model.blocks)
    effective_embed = (model.res_gate.weight != 0).sum().item()
    
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
        'params_m': count_parameters(model),
        'model_size_mb': model_size_mb(model),
        'flops_g': model.flops(2).item() / 1e9,
        'training_energy_kwh': energy_metrics['energy_kwh'],
        'effective_heads': effective_heads,
        'effective_mlp': effective_mlp,
        'effective_embed': effective_embed,
        'total_prune_steps': NUM_PRUNE_STEPS
    }
    
    print(f"\nFinal Test - Acc: {test_acc:.4f}, Precision: {test_precision:.4f}")
    print(f"Effective: heads={effective_heads:.1f}, mlp={effective_mlp:.0f}, embed={effective_embed:.0f}")
    
    return model, final_metrics

# ==================== METHOD 3: ONE-SHOT PRUNING ====================

def train_prune_then_finetune(dataset_name, model_name, train_loader, val_loader, test_loader,
                              num_classes, save_dir, trial_num):
    """METHOD 3: One-Shot Pruning then Finetune"""
    print("\n" + "="*80)
    print(f"METHOD 3: ONE-SHOT PRUNE-THEN-TRAIN - {dataset_name.upper()} - {model_name} - TRIAL {trial_num}")
    print("="*80)
    
    # Create model
    model = create_gated_vit_from_timm(model_name, num_classes, pretrained=True).to(DEVICE)
    print(f"Initial parameters: {count_parameters(model):.2f}M")
    print(f"Initial FLOPs: {model.flops(2).item() / 1e9:.2f} GFLOPs")
    
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
    
    # Calculate target sparsity (cumulative from 4 steps of pruning)
    cumulative_sparsity = TARGET_FLOPS_REDUCTION
    apply_pruning_simple(model, pruners, target_sparsity=cumulative_sparsity)
    
    print(f"\nPruned FLOPs: {model.flops(2).item() / 1e9:.2f} GFLOPs")
    print(f"Now finetuning for {FIXED_EPOCHS} epochs...")
    
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
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
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
    
    energy_metrics = stop_energy_tracker(tracker, save_dir, f"{dataset_name}_{model_name}_prune_then_train_trial{trial_num}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch}")
    
    # Save
    torch.save(model.state_dict(), os.path.join(save_dir,
               f"{dataset_name}_{model_name}_prune_then_train_trial{trial_num}_final.pth"))
    pd.DataFrame(all_metrics).to_csv(os.path.join(save_dir,
               f"{dataset_name}_{model_name}_prune_then_train_trial{trial_num}_metrics.csv"), index=False)
    
    # Final test
    test_acc, test_auc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, DEVICE)
    
    # Count effective dimensions
    total_heads, active_heads = 0, 0
    total_mlp, active_mlp = 0, 0
    for block in model.blocks:
        total_heads += model.num_heads
        active_heads += (block.attn.attn_gate.weight != 0).sum().item()
        mlp_dim = block.mlp.hidden_gate.weight.shape[0]
        total_mlp += mlp_dim
        active_mlp += (block.mlp.hidden_gate.weight != 0).sum().item()
    
    effective_heads = active_heads / len(model.blocks)
    effective_mlp = active_mlp / len(model.blocks)
    effective_embed = (model.res_gate.weight != 0).sum().item()
    
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
        'params_m': count_parameters(model),
        'model_size_mb': model_size_mb(model),
        'flops_g': model.flops(2).item() / 1e9,
        'training_energy_kwh': energy_metrics['energy_kwh'],
        'effective_heads': effective_heads,
        'effective_mlp': effective_mlp,
        'effective_embed': effective_embed
    }
    
    print(f"\nFinal Test - Acc: {test_acc:.4f}, Precision: {test_precision:.4f}")
    print(f"Effective: heads={effective_heads:.1f}, mlp={effective_mlp:.0f}, embed={effective_embed:.0f}")
    
    return model, final_metrics

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
        cleanup_memory()
        
        # Method 1: Baseline
        _, baseline_metrics = train_baseline_vit(
            dataset_name, 'vit_tiny_patch16_224', train_loader, val_loader, test_loader,
            num_classes, save_dir, trial
        )
        all_baseline_metrics.append(baseline_metrics)
        cleanup_memory()
        
        # Method 3: Prune-then-train
        _, prune_then_train_metrics = train_prune_then_finetune(
            dataset_name, 'vit_tiny_patch16_224', train_loader, val_loader, test_loader,
            num_classes, save_dir, trial
        )
        all_prune_then_train_metrics.append(prune_then_train_metrics)
        cleanup_memory()
    
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
        for col in ['test_acc', 'test_precision', 'params_m', 'flops_g', 'training_energy_kwh']:
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
    for col in ['test_acc', 'params_m', 'flops_g', 'training_energy_kwh']:
        base_mean = summary_rows[0].get(f'{col}_mean', float('nan'))
        base_std = summary_rows[0].get(f'{col}_std', float('nan'))
        prog_mean = summary_rows[1].get(f'{col}_mean', float('nan'))
        prog_std = summary_rows[1].get(f'{col}_std', float('nan'))
        ptt_mean = summary_rows[2].get(f'{col}_mean', float('nan'))
        ptt_std = summary_rows[2].get(f'{col}_std', float('nan'))
        
        print(f"{col:<30} Baseline: {base_mean:.4f}±{base_std:.4f}  "
              f"Progressive: {prog_mean:.4f}±{prog_std:.4f}  "
              f"Prune-Then-Train: {ptt_mean:.4f}±{ptt_std:.4f}")
    print("="*140)

# ==================== MAIN ====================

def main():
    set_seed(SEED)
    
    print("="*100)
    print("STRUCTURED VISION TRANSFORMER PRUNING (Hikvision-Style)")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Fixed Epochs: {FIXED_EPOCHS}")
    print(f"  Pruning Schedule: {PRUNE_EPOCHS_SCHEDULE}")
    print(f"  Target FLOPs Reduction: {TARGET_FLOPS_REDUCTION:.1%}")
    print(f"  Number of Trials: {NUM_TRIALS}")
    
    datasets = ['bloodmnist', 'pathmnist', 'dermamnist']
    
    for dataset in datasets:
        npz_path = os.path.join(DATASET_DIR, f"{dataset}_224.npz")
        
        if not os.path.exists(npz_path):
            print(f"\nDataset not found: {npz_path}")
            continue
        
        train_loader, val_loader, test_loader, num_classes, dataset_name = load_dataset(npz_path)
        process_dataset(dataset_name, train_loader, val_loader, test_loader, num_classes, SAVE_DIR)
    
    print("\n" + "="*100)
    print("EXPERIMENT COMPLETED")
    print("="*100)

if __name__ == "__main__":
    main()