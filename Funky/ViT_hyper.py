"""
ViT Hyperparameter Tuning - Baseline & Progressive Pruning
CORRECTED: Both methods convert to deployment model and finetune for fair comparison
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import timm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd
import copy
import gc
from functools import partial
import time
import warnings

# Suppress the backward hook warning
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.modules.module')

# ==================== CONFIGURATION ====================

DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Hyper/vit_hyperparam_tuning"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224
MIN_LR = 1e-6

os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== HYPERPARAMETER CONFIGS ====================

CONFIGS = [
    # Config 1: Conservative - proven best baseline LR with gentle 50% compression
    {
        'name': 'conservative_50pct',
        'description': 'Low LR (1e-4), gentle reduction, 50% target, long finetune',
        'INITIAL_LR': 1e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.8,
        'GATE_L1_WEIGHT': 5e-5,
        'WEIGHT_DECAY': 1e-4,
        'BATCH_SIZE': 64,
        'SPARSITY_SCHEDULE': [0.1, 0.2, 0.35, 0.5],  # Very gradual to 50%
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 3,
        'EPOCHS_BETWEEN_PRUNES': 4,
        'DEPLOY_FINETUNE_EPOCHS': 20,
        'DEPLOY_FINETUNE_LR': 1e-4,
        'TOTAL_EPOCHS': 20  # FIXED: All configs same epochs
    },
    
    # Config 2: Extra gentle - 40% max compression for better retention
    {
        'name': 'gentle_40pct',
        'description': 'Low LR, 40% max compression, extended training',
        'INITIAL_LR': 1e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.85,
        'GATE_L1_WEIGHT': 3e-5,
        'WEIGHT_DECAY': 1e-4,
        'BATCH_SIZE': 64,
        'SPARSITY_SCHEDULE': [0.08, 0.16, 0.28, 0.4],  # Stop at 40%
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 3,
        'EPOCHS_BETWEEN_PRUNES': 4,
        'DEPLOY_FINETUNE_EPOCHS': 20,
        'DEPLOY_FINETUNE_LR': 1e-4,
        'TOTAL_EPOCHS': 20  # FIXED: All configs same epochs
    },
    
    # Config 3: Many small steps - 6 pruning steps to 50%
    {
        'name': 'many_steps_50pct',
        'description': '6 tiny pruning steps, minimal LR changes',
        'INITIAL_LR': 1e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.9,
        'GATE_L1_WEIGHT': 5e-5,
        'WEIGHT_DECAY': 1e-4,
        'BATCH_SIZE': 64,
        'SPARSITY_SCHEDULE': [0.08, 0.16, 0.25, 0.33, 0.42, 0.5],  # 6 small steps
        'NUM_PRUNE_STEPS': 6,
        'WARMUP_EPOCHS': 3,
        'EPOCHS_BETWEEN_PRUNES': 2,
        'DEPLOY_FINETUNE_EPOCHS': 20,
        'DEPLOY_FINETUNE_LR': 1e-4,
        'TOTAL_EPOCHS': 20  # FIXED: All configs same epochs
    },
    
    # Config 4: Moderate with strong recovery - 45% compression
    {
        'name': 'moderate_45pct',
        'description': 'Balance compression/accuracy, heavy deployment finetune',
        'INITIAL_LR': 1e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.75,
        'GATE_L1_WEIGHT': 5e-5,
        'WEIGHT_DECAY': 1e-4,
        'BATCH_SIZE': 64,
        'SPARSITY_SCHEDULE': [0.1, 0.22, 0.35, 0.45],  # Stop at 45%
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 3,
        'EPOCHS_BETWEEN_PRUNES': 4,
        'DEPLOY_FINETUNE_EPOCHS': 20,
        'DEPLOY_FINETUNE_LR': 1e-4,
        'TOTAL_EPOCHS': 20  # FIXED: All configs same epochs
    },
]

# ==================== UTILITIES ====================

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2

# ==================== DATASET ====================

class NumpyMemmapDataset(Dataset):
    def __init__(self, imgs_np, labels_np, img_size=224, is_train=False):
        self.imgs = imgs_np
        self.labels = labels_np
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

def load_dataset(npz_path, batch_size):
    data = np.load(npz_path, mmap_mode="r")
    X_train, y_train = data["train_images"], data["train_labels"].flatten()
    X_val, y_val = data["val_images"], data["val_labels"].flatten()
    X_test, y_test = data["test_images"], data["test_labels"].flatten()

    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE, is_train=True)
    val_ds = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE, is_train=False)
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    dataset_name = os.path.splitext(os.path.basename(npz_path))[0].replace('_224', '')
    return train_loader, val_loader, test_loader, num_classes, dataset_name

# ==================== METRICS ====================

def evaluate_model(net, loader, device):
    net.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            probs = torch.softmax(outputs, dim=1)
            predict_y = torch.max(probs, dim=1)[1]
            all_preds.extend(predict_y.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = sum([1 for i in range(len(all_preds)) if all_preds[i] == all_labels[i]]) / len(all_preds)
    n_classes = len(set(all_labels))
    all_labels_one_hot = label_binarize(all_labels, classes=list(range(n_classes)))
    try:
        auc = roc_auc_score(all_labels_one_hot, all_probs, multi_class='ovr')
    except ValueError:
        auc = float('nan')
    return acc, auc

# ==================== GATE LAYER ====================

class GateLayer(nn.Module):
    def __init__(self, input_features, output_features, size_mask):
        super(GateLayer, self).__init__()
        self.size_mask = size_mask
        self.weight = nn.Parameter(torch.ones(output_features))
        self.do_not_update = True

    def forward(self, input, mask=None):
        return input * self.weight.view(*self.size_mask)

# ==================== GATED VIT COMPONENTS ====================

class GatedMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
        x = self.hidden_gate(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GatedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
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
        x = attn @ v
        x = self.attn_gate(x)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GatedBlock(nn.Module):
    def __init__(self, dim, num_heads, res_gate, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GatedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   attn_drop=attn_drop, proj_drop=drop)
        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GatedMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.res_gate = res_gate

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.res_gate(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.res_gate(x)
        return x

class GatedVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio

        from timm.models.layers import PatchEmbed
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.res_gate = GateLayer(embed_dim, embed_dim, [1, 1, -1])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            GatedBlock(dim=embed_dim, num_heads=num_heads, res_gate=self.res_gate,
                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=None,
                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                       norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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

    def forward(self, x):
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
        x = x[:, 0]
        x = self.head(x)
        return x

# ==================== DEPLOYMENT MODEL ====================

class DeployMlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
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
    def __init__(self, dim, num_heads, head_dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
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
    def __init__(self, dim, num_heads, head_dim, mlp_hidden_dim, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DeployAttention(dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = DeployMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DeployVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=192, depth=12, head_dim=64, per_layer_num_heads=None,
                 per_layer_mlp_dim=None, qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.head_dim = head_dim
        self.img_size = img_size
        self.patch_size = patch_size

        from timm.models.layers import PatchEmbed
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if per_layer_num_heads is None:
            per_layer_num_heads = [embed_dim // head_dim] * depth
        if per_layer_mlp_dim is None:
            per_layer_mlp_dim = [embed_dim * 4] * depth

        self.blocks = nn.ModuleList([
            DeployBlock(dim=embed_dim, num_heads=per_layer_num_heads[i], head_dim=head_dim,
                        mlp_hidden_dim=per_layer_mlp_dim[i], qkv_bias=qkv_bias, qk_scale=None,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.per_layer_num_heads = per_layer_num_heads
        self.per_layer_mlp_dim = per_layer_mlp_dim

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

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x

    def compute_flops(self):
        n_patches = (self.img_size // self.patch_size) ** 2 + 1
        total_flops = 0
        for i in range(self.depth):
            num_heads = self.per_layer_num_heads[i]
            mlp_dim = self.per_layer_mlp_dim[i]
            d = self.embed_dim
            qkv_flops = n_patches * d * (num_heads * self.head_dim * 3)
            attn_flops = n_patches * n_patches * (num_heads * self.head_dim)
            out_flops = n_patches * (num_heads * self.head_dim) * d
            mlp_flops = 2 * n_patches * d * mlp_dim
            total_flops += qkv_flops + attn_flops + out_flops + mlp_flops
        total_flops += self.embed_dim * self.num_classes
        return total_flops

# ==================== MODEL CREATION ====================

def create_gated_vit(num_classes, pretrained=True):
    model = GatedVisionTransformer(img_size=IMG_SIZE, patch_size=16, in_chans=3, num_classes=num_classes,
                                   embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
                                   drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1)
    if pretrained:
        timm_model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
        model_dict = model.state_dict()
        timm_dict = timm_model.state_dict()

        for key in ['patch_embed.proj.weight', 'patch_embed.proj.bias', 'pos_embed', 'cls_token',
                    'norm.weight', 'norm.bias', 'head.weight', 'head.bias']:
            if key in timm_dict and key in model_dict:
                model_dict[key].copy_(timm_dict[key])

        for i in range(12):
            for component in ['norm1', 'norm2', 'attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2']:
                for suffix in ['weight', 'bias']:
                    src_key = f'blocks.{i}.{component}.{suffix}'
                    if src_key in timm_dict and src_key in model_dict:
                        model_dict[src_key].copy_(timm_dict[src_key])
    return model

# ==================== PRUNING ====================

def forward_hook(self, input, output):
    self.output = output.detach()

def backward_hook(self, grad_input, grad_output):
    self.grad = (grad_output[0].detach(),)

def taylor2Scorer(gate_module):
    score = (gate_module.weight * gate_module.weight.grad).pow(2)
    return score

def prepare_pruning_list(model, pruning_layer_type):
    """
    Prepare pruning modules and register hooks.
    Returns (pruning_modules, hook_handles) - caller must remove hooks when done.
    """
    pruning_modules = []
    hook_handles = []
    for module_name, m in model.named_modules():
        if hasattr(m, "do_not_update"):
            if pruning_layer_type == 0 and 'attn_gate' in module_name:
                hook_handles.append(m.register_forward_hook(forward_hook))
                hook_handles.append(m.register_full_backward_hook(backward_hook))
                pruning_modules.append(m)
            elif pruning_layer_type == 1 and 'hidden_gate' in module_name:
                hook_handles.append(m.register_forward_hook(forward_hook))
                hook_handles.append(m.register_full_backward_hook(backward_hook))
                pruning_modules.append(m)
            elif pruning_layer_type == 2 and 'res_gate' in module_name:
                hook_handles.append(m.register_forward_hook(forward_hook))
                hook_handles.append(m.register_full_backward_hook(backward_hook))
                pruning_modules.append(m)
    return pruning_modules, hook_handles

class StructuredPruner:
    def __init__(self, model, pruning_modules, pruning_momentum=0.9):
        self.pruning_modules = pruning_modules
        self.pruning_parameters = [m.weight for m in pruning_modules]
        self.momentum = pruning_momentum
        self.iterations_done = 0
        self.pruning_scores = {'score': [list() for _ in range(len(self.pruning_parameters))],
                               'averaged': [list() for _ in range(len(self.pruning_parameters))]}
        self.pruning_gates = [np.ones(len(param),) for param in self.pruning_parameters]
        self.all_neuron_units = sum(len(param) for param in self.pruning_parameters)

    def do_step(self):
        for layer, module in enumerate(self.pruning_modules):
            scores = taylor2Scorer(module)
            if self.iterations_done == 0:
                self.pruning_scores['score'][layer] = scores
            else:
                self.pruning_scores['score'][layer] += scores
        self.iterations_done += 1

    def finalize_scores(self):
        for layer in range(len(self.pruning_scores['score'])):
            contribution = self.pruning_scores['score'][layer] / self.iterations_done
            if len(self.pruning_scores["averaged"][layer]) == 0 or not self.momentum:
                self.pruning_scores["averaged"][layer] = contribution
            else:
                self.pruning_scores["averaged"][layer] = (self.momentum * self.pruning_scores["averaged"][layer] +
                                                          (1 - self.momentum) * contribution)
        criteria = [score.detach().cpu().numpy() if torch.is_tensor(score) else score
                    for score in self.pruning_scores['averaged']]
        return criteria

    def prune(self, criteria, threshold):
        for layer in range(len(criteria)):
            index = np.where(criteria[layer] <= threshold)
            self.pruning_gates[layer][index] *= 0.0
            self.pruning_parameters[layer].data[index] *= 0.0

    def get_num_active(self):
        return sum(np.count_nonzero(gate) for gate in self.pruning_gates)

def collect_importance_scores(model, pruners, train_loader, max_batches=50):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        for pruner in pruners:
            pruner.do_step()

def apply_pruning(model, pruners, target_sparsity):
    for pruner in pruners:
        criteria = pruner.finalize_scores()
        all_scores = []
        for layer_scores in criteria:
            all_scores.extend(layer_scores.flatten().tolist())
        all_scores_array = np.array(all_scores)
        num_to_prune = int(len(all_scores_array) * target_sparsity)
        if num_to_prune > 0:
            threshold = np.sort(all_scores_array)[num_to_prune]
        else:
            threshold = -np.inf
        pruner.prune(criteria, threshold)

def get_gate_stats(model):
    stats = {'attn': [], 'mlp': [], 'res': []}
    for name, param in model.named_parameters():
        if 'gate' in name and 'weight' in name:
            w = param.data.cpu().numpy()
            if 'attn_gate' in name:
                stats['attn'].extend(w.flatten())
            elif 'hidden_gate' in name:
                stats['mlp'].extend(w.flatten())
            elif 'res_gate' in name:
                stats['res'].extend(w.flatten())
    result = {}
    for k, v in stats.items():
        if len(v) > 0:
            arr = np.array(v)
            result[f'{k}_mean'] = arr.mean()
            result[f'{k}_zeros_pct'] = (np.abs(arr) < 0.01).sum() / len(arr) * 100
        else:
            result[f'{k}_mean'] = 0
            result[f'{k}_zeros_pct'] = 0
    return result

# ==================== DEPLOYMENT CONVERSION ====================

def convert_to_deployment(gated_model, num_classes):
    GATE_THRESHOLD = 0.01
    res_gate_weights = gated_model.res_gate.weight.data
    res_mask = (res_gate_weights.abs() > GATE_THRESHOLD).cpu()
    new_embed_dim = res_mask.sum().item()

    head_masks, mlp_masks = [], []
    for block in gated_model.blocks:
        attn_gate_weights = block.attn.attn_gate.weight.data
        head_mask = (attn_gate_weights.abs() > GATE_THRESHOLD).cpu()
        head_masks.append(head_mask)
        mlp_gate_weights = block.mlp.hidden_gate.weight.data
        mlp_mask = (mlp_gate_weights.abs() > GATE_THRESHOLD).cpu()
        mlp_masks.append(mlp_mask)

    per_layer_num_heads = [mask.sum().item() for mask in head_masks]
    per_layer_mlp_dim = [mask.sum().item() for mask in mlp_masks]
    head_dim = gated_model.embed_dim // gated_model.num_heads

    deploy_model = DeployVisionTransformer(
        img_size=IMG_SIZE, patch_size=16, in_chans=3, num_classes=num_classes,
        embed_dim=new_embed_dim, depth=gated_model.depth, head_dim=head_dim,
        per_layer_num_heads=per_layer_num_heads, per_layer_mlp_dim=per_layer_mlp_dim,
        qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1
    ).to(DEVICE)

    with torch.no_grad():
        gated_dict = gated_model.state_dict()
        deploy_dict = deploy_model.state_dict()

        deploy_dict['patch_embed.proj.weight'].copy_(gated_dict['patch_embed.proj.weight'][res_mask])
        deploy_dict['patch_embed.proj.bias'].copy_(gated_dict['patch_embed.proj.bias'][res_mask])
        deploy_dict['pos_embed'].copy_(gated_dict['pos_embed'][:, :, res_mask])
        deploy_dict['cls_token'].copy_(gated_dict['cls_token'][:, :, res_mask])

        for i, (head_mask, mlp_mask) in enumerate(zip(head_masks, mlp_masks)):
            deploy_dict[f'blocks.{i}.norm1.weight'].copy_(gated_dict[f'blocks.{i}.norm1.weight'][res_mask])
            deploy_dict[f'blocks.{i}.norm1.bias'].copy_(gated_dict[f'blocks.{i}.norm1.bias'][res_mask])
            deploy_dict[f'blocks.{i}.norm2.weight'].copy_(gated_dict[f'blocks.{i}.norm2.weight'][res_mask])
            deploy_dict[f'blocks.{i}.norm2.bias'].copy_(gated_dict[f'blocks.{i}.norm2.bias'][res_mask])

            orig_qkv_weight = gated_dict[f'blocks.{i}.attn.qkv.weight']
            orig_qkv_bias = gated_dict[f'blocks.{i}.attn.qkv.bias']
            q_w, k_w, v_w = orig_qkv_weight.chunk(3, dim=0)
            q_b, k_b, v_b = orig_qkv_bias.chunk(3, dim=0)

            orig_num_heads = gated_model.num_heads
            q_w = q_w.view(orig_num_heads, head_dim, gated_model.embed_dim)
            k_w = k_w.view(orig_num_heads, head_dim, gated_model.embed_dim)
            v_w = v_w.view(orig_num_heads, head_dim, gated_model.embed_dim)
            q_b = q_b.view(orig_num_heads, head_dim)
            k_b = k_b.view(orig_num_heads, head_dim)
            v_b = v_b.view(orig_num_heads, head_dim)

            q_w = q_w[head_mask][:, :, res_mask]
            k_w = k_w[head_mask][:, :, res_mask]
            v_w = v_w[head_mask][:, :, res_mask]
            q_b = q_b[head_mask]
            k_b = k_b[head_mask]
            v_b = v_b[head_mask]

            new_qkv_weight = torch.cat([q_w.flatten(0, 1), k_w.flatten(0, 1), v_w.flatten(0, 1)], dim=0)
            new_qkv_bias = torch.cat([q_b.flatten(), k_b.flatten(), v_b.flatten()], dim=0)
            deploy_dict[f'blocks.{i}.attn.qkv.weight'].copy_(new_qkv_weight)
            deploy_dict[f'blocks.{i}.attn.qkv.bias'].copy_(new_qkv_bias)

            orig_proj_weight = gated_dict[f'blocks.{i}.attn.proj.weight']
            orig_proj_weight = orig_proj_weight.view(gated_model.embed_dim, orig_num_heads, head_dim)
            new_proj_weight = orig_proj_weight[res_mask][:, head_mask, :].flatten(1)
            deploy_dict[f'blocks.{i}.attn.proj.weight'].copy_(new_proj_weight)
            deploy_dict[f'blocks.{i}.attn.proj.bias'].copy_(gated_dict[f'blocks.{i}.attn.proj.bias'][res_mask])

            orig_fc1_weight = gated_dict[f'blocks.{i}.mlp.fc1.weight']
            deploy_dict[f'blocks.{i}.mlp.fc1.weight'].copy_(orig_fc1_weight[mlp_mask][:, res_mask])
            deploy_dict[f'blocks.{i}.mlp.fc1.bias'].copy_(gated_dict[f'blocks.{i}.mlp.fc1.bias'][mlp_mask])

            orig_fc2_weight = gated_dict[f'blocks.{i}.mlp.fc2.weight']
            deploy_dict[f'blocks.{i}.mlp.fc2.weight'].copy_(orig_fc2_weight[res_mask][:, mlp_mask])
            deploy_dict[f'blocks.{i}.mlp.fc2.bias'].copy_(gated_dict[f'blocks.{i}.mlp.fc2.bias'][res_mask])

        deploy_dict['norm.weight'].copy_(gated_dict['norm.weight'][res_mask])
        deploy_dict['norm.bias'].copy_(gated_dict['norm.bias'][res_mask])
        deploy_dict['head.weight'].copy_(gated_dict['head.weight'][:, res_mask])
        deploy_dict['head.bias'].copy_(gated_dict['head.bias'])

    return deploy_model, {'embed_dim': new_embed_dim, 'avg_heads': np.mean(per_layer_num_heads),
                          'avg_mlp': np.mean(per_layer_mlp_dim)}

# ==================== TRAINING ====================

def train_one_epoch(model, train_loader, optimizer, scheduler, gate_l1_weight=0.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        task_loss = nn.functional.cross_entropy(outputs, labels)

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
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), correct / total

def finetune_deploy(deploy_model, train_loader, val_loader, epochs, lr, weight_decay):
    """Finetune deployment model after conversion"""
    optimizer = optim.AdamW(deploy_model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)
    best_val_acc = 0.0
    best_state = None

    print(f"  [Finetuning deployment model for {epochs} epochs]")
    for epoch in range(1, epochs + 1):
        train_one_epoch(deploy_model, train_loader, optimizer, scheduler, gate_l1_weight=0.0)
        val_acc, _ = evaluate_model(deploy_model, val_loader, DEVICE)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(deploy_model.state_dict())

    if best_state is not None:
        deploy_model.load_state_dict(best_state)
    print(f"  [Deployment model finetuned: best val acc = {best_val_acc:.4f}]")
    return deploy_model

# ==================== BASELINE TRAINING ====================

def train_baseline(config, train_loader, val_loader, test_loader, num_classes, epoch_log):
    """
    Baseline training - NO pruning, trains for TOTAL_EPOCHS only.
    Converts to deployment model at the end for fair size/FLOPs comparison.
    """
    model = create_gated_vit(num_classes, pretrained=True).to(DEVICE)

    # Freeze gates (no pruning for baseline)
    for name, param in model.named_parameters():
        if 'gate' in name:
            param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=config['INITIAL_LR'], weight_decay=config['WEIGHT_DECAY'])
    total_steps = config['TOTAL_EPOCHS'] * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, config['TOTAL_EPOCHS'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, gate_l1_weight=0.0)
        val_acc, val_auc = evaluate_model(model, val_loader, DEVICE)
        test_acc, test_auc = evaluate_model(model, test_loader, DEVICE)

        print(f"  E{epoch:02d} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")
        epoch_log.append({'epoch': epoch, 'val_acc': val_acc, 'val_auc': val_auc, 'test_acc': test_acc, 'test_auc': test_auc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Convert to deployment model (no pruning, so same size - just removes gate layers)
    deploy_model, deploy_config = convert_to_deployment(model, num_classes)

    # Final evaluation (no additional finetuning - all training within TOTAL_EPOCHS)
    test_acc, test_auc = evaluate_model(deploy_model, test_loader, DEVICE)

    return {
        'test_acc': test_acc, 'test_auc': test_auc, 'best_val_acc': best_val_acc,
        'params_m': count_parameters(deploy_model), 'size_mb': model_size_mb(deploy_model),
        'flops_g': deploy_model.compute_flops() / 1e9, 'deploy_embed': deploy_config['embed_dim'],
        'deploy_avg_heads': deploy_config['avg_heads'], 'deploy_avg_mlp': deploy_config['avg_mlp']
    }

# ==================== PROGRESSIVE PRUNING ====================

def train_progressive(config, train_loader, val_loader, test_loader, num_classes, epoch_log):
    """
    Progressive pruning - multiple pruning steps, then convert to deployment and train that

    Flow:
    1. Train gated model with progressive pruning (multiple steps to target sparsity)
    2. Convert final pruned gated model to deployment model (physically smaller)
    3. Train the deployment model for remaining epochs
    4. Final deployment finetuning
    """
    prune_epochs = [config['WARMUP_EPOCHS'] + 1 + i * config['EPOCHS_BETWEEN_PRUNES']
                    for i in range(config['NUM_PRUNE_STEPS'])]

    # Determine when to convert to deployment (after last prune)
    final_prune_epoch = max(prune_epochs)
    convert_to_deploy_epoch = final_prune_epoch + 1  # Convert right after final prune

    # Phase 1: Train gated model with progressive pruning
    model = create_gated_vit(num_classes, pretrained=True).to(DEVICE)

    pruners = []
    all_hook_handles = []  # Track all hooks for cleanup
    for component_type in [0, 1, 2]:
        modules, hook_handles = prepare_pruning_list(model, component_type)
        all_hook_handles.extend(hook_handles)
        if len(modules) > 0:
            pruner = StructuredPruner(model, modules, pruning_momentum=0.9)
            pruners.append(pruner)

    optimizer = optim.AdamW(model.parameters(), lr=config['INITIAL_LR'], weight_decay=config['WEIGHT_DECAY'])
    current_lr = config['INITIAL_LR']

    # Create scheduler spanning entire gated training phase (consistent with baseline)
    total_steps = config['TOTAL_EPOCHS'] * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)

    deploy_model = None
    using_deploy = False
    best_deploy_val_acc = 0.0
    best_deploy_state = None

    for epoch in range(1, config['TOTAL_EPOCHS'] + 1):
        # Handle pruning steps
        if epoch in prune_epochs:
            prune_step = prune_epochs.index(epoch) + 1
            target_sparsity = config['SPARSITY_SCHEDULE'][prune_step - 1]
            print(f"  [Pruning step {prune_step}/{config['NUM_PRUNE_STEPS']} → {target_sparsity*100:.0f}% sparsity]")

            for pruner in pruners:
                pruner.iterations_done = 0
                pruner.pruning_scores['score'] = [list() for _ in range(len(pruner.pruning_parameters))]

            collect_importance_scores(model, pruners, train_loader, max_batches=50)
            apply_pruning(model, pruners, target_sparsity)

            current_lr *= config['LR_REDUCTION_AFTER_PRUNE']
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        # Convert to deployment model after final pruning step
        if epoch == convert_to_deploy_epoch and not using_deploy:
            print(f"  [All pruning complete - converting to deployment model]")

            # Remove hooks before conversion (no longer needed)
            for handle in all_hook_handles:
                handle.remove()
            all_hook_handles.clear()

            deploy_model, deploy_config = convert_to_deployment(model, num_classes)
            using_deploy = True

            # Create new optimizer and scheduler for deployment model
            remaining_epochs = config['TOTAL_EPOCHS'] - epoch + 1
            remaining_steps = remaining_epochs * len(train_loader)
            optimizer = optim.AdamW(deploy_model.parameters(), lr=current_lr, weight_decay=config['WEIGHT_DECAY'])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_steps, eta_min=MIN_LR)
            print(f"  [Deployment: {count_parameters(deploy_model):.2f}M params (was {count_parameters(model):.2f}M)]")

        active_model = deploy_model if using_deploy else model
        train_loss, train_acc = train_one_epoch(active_model, train_loader, optimizer, scheduler,
                                                 gate_l1_weight=0.0 if using_deploy else config['GATE_L1_WEIGHT'])

        # Evaluate
        val_acc, val_auc = evaluate_model(active_model, val_loader, DEVICE)
        test_acc, test_auc = evaluate_model(active_model, test_loader, DEVICE)

        # Track best deployment model
        if using_deploy and val_acc > best_deploy_val_acc:
            best_deploy_val_acc = val_acc
            best_deploy_state = copy.deepcopy(deploy_model.state_dict())

        gate_stats = get_gate_stats(model) if not using_deploy else {
            'attn_mean': 0, 'attn_zeros_pct': 0, 'mlp_mean': 0, 'mlp_zeros_pct': 0,
            'res_mean': 0, 'res_zeros_pct': 0
        }
        model_type = 'deploy' if using_deploy else 'gated'
        params_now = count_parameters(active_model)
        sparsity_pct = gate_stats.get('res_zeros_pct', 0)

        print(f"  E{epoch:02d} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | "
              f"Params: {params_now:.2f}M | Sparse: {sparsity_pct:.0f}% | {model_type}")

        epoch_log.append({
            'epoch': epoch, 'val_acc': val_acc, 'val_auc': val_auc, 'test_acc': test_acc,
            'test_auc': test_auc, 'eval_model': model_type, 'params_m': params_now, **gate_stats
        })

    # Clean up any remaining hooks (safety)
    for handle in all_hook_handles:
        handle.remove()

    # If we never converted to deploy (shouldn't happen with proper config), do it now
    if deploy_model is None:
        print("  [WARNING: No deployment conversion happened, converting now]")
        deploy_model, _ = convert_to_deployment(model, num_classes)
    elif best_deploy_state is not None:
        # Restore best deployment model state
        deploy_model.load_state_dict(best_deploy_state)
        print(f"  [Restored best deployment model with val_acc={best_deploy_val_acc:.4f}]")

    # Final evaluation (no additional finetuning - all training within TOTAL_EPOCHS)
    test_acc, test_auc = evaluate_model(deploy_model, test_loader, DEVICE)

    deploy_config_final = {
        'embed_dim': deploy_model.embed_dim,
        'avg_heads': np.mean(deploy_model.per_layer_num_heads),
        'avg_mlp': np.mean(deploy_model.per_layer_mlp_dim)
    }

    return {
        'test_acc': test_acc, 'test_auc': test_auc, 'best_val_acc': best_deploy_val_acc,
        'params_m': count_parameters(deploy_model), 'size_mb': model_size_mb(deploy_model),
        'flops_g': deploy_model.compute_flops() / 1e9, 'deploy_embed': deploy_config_final['embed_dim'],
        'deploy_avg_heads': deploy_config_final['avg_heads'], 'deploy_avg_mlp': deploy_config_final['avg_mlp']
    }

# ==================== MAIN ====================

def main():
    set_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Configs: {len(CONFIGS)} optimized configurations")
    print("=" * 80)
    print("OPTIMIZED CONFIGS - Max 50% Compression")
    print("=" * 80)
    for i, cfg in enumerate(CONFIGS, 1):
        print(f"\n{i}. {cfg['name']}")
        print(f"   {cfg['description']}")
        print(f"   Sparsity: {cfg['SPARSITY_SCHEDULE'][-1]*100:.0f}% | "
              f"Deploy FT: {cfg['DEPLOY_FINETUNE_EPOCHS']} epochs | "
              f"LR: {cfg['INITIAL_LR']:.0e}")
    print("\n" + "=" * 80)

    all_results = []
    all_epoch_logs = []

    datasets = ['dermamnist']  # Start with one dataset

    for dataset in datasets:
        npz_path = os.path.join(DATASET_DIR, f"{dataset}_224.npz")
        if not os.path.exists(npz_path):
            print(f"Dataset not found: {npz_path}")
            continue

        print(f"\n{'='*60}\nDataset: {dataset}\n{'='*60}")

        for config in CONFIGS:
            batch_size = config.get('BATCH_SIZE', 64)
            train_loader, val_loader, test_loader, num_classes, ds_name = load_dataset(npz_path, batch_size)

            # Baseline
            print(f"\n[BASELINE] {config['name']}")
            epoch_log = []
            try:
                start_time = time.time()
                result = train_baseline(config, train_loader, val_loader, test_loader, num_classes, epoch_log)
                elapsed = (time.time() - start_time) / 60
                result.update({'config': config['name'], 'method': 'baseline', 'dataset': ds_name,
                               'time_min': elapsed, **{k.lower(): v for k, v in config.items()}})
                all_results.append(result)
                for e in epoch_log:
                    e.update({'config': config['name'], 'method': 'baseline', 'dataset': ds_name})
                all_epoch_logs.extend(epoch_log)
                print(f"  Final (after deploy finetune): Acc={result['test_acc']:.4f} AUC={result['test_auc']:.4f}")
            except Exception as ex:
                print(f"  ERROR: {ex}")
            cleanup_memory()

            # Progressive
            print(f"\n[PROGRESSIVE] {config['name']}")
            epoch_log = []
            try:
                start_time = time.time()
                result = train_progressive(config, train_loader, val_loader, test_loader, num_classes, epoch_log)
                elapsed = (time.time() - start_time) / 60
                result.update({'config': config['name'], 'method': 'progressive', 'dataset': ds_name,
                               'time_min': elapsed, **{k.lower(): v for k, v in config.items()}})
                all_results.append(result)
                for e in epoch_log:
                    e.update({'config': config['name'], 'method': 'progressive', 'dataset': ds_name})
                all_epoch_logs.extend(epoch_log)
                print(f"  Final (after deploy finetune): Acc={result['test_acc']:.4f} AUC={result['test_auc']:.4f} Params={result['params_m']:.2f}M")
            except Exception as ex:
                print(f"  ERROR: {ex}")
            cleanup_memory()

            # Save incrementally
            pd.DataFrame(all_results).to_csv(os.path.join(SAVE_DIR, 'results.csv'), index=False)
            pd.DataFrame(all_epoch_logs).to_csv(os.path.join(SAVE_DIR, 'epoch_logs.csv'), index=False)

    # Final summary
    df = pd.DataFrame(all_results)
    summary = df.groupby(['config', 'method']).agg({
        'test_acc': ['mean', 'std'], 'test_auc': ['mean', 'std'],
        'params_m': 'mean', 'flops_g': 'mean', 'time_min': 'mean'
    }).round(4)
    summary.to_csv(os.path.join(SAVE_DIR, 'summary.csv'))

    print(f"\nResults saved to {SAVE_DIR}")
    print("\nTop configs by test accuracy (progressive):")
    prog_df = df[df['method'] == 'progressive'].sort_values('test_acc', ascending=False)
    for _, row in prog_df.head(5).iterrows():
        print(f"  {row['config']}: Acc={row['test_acc']:.4f} AUC={row['test_auc']:.4f} Params={row['params_m']:.2f}M")

if __name__ == "__main__":
    main()