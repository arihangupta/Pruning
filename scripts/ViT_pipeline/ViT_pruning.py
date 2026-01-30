import os
import sys
import time
import math
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import timm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import csv
import shutil
from torchprofile import profile_macs
import psutil
import gc
import logging
import copy
from functools import partial

# CodeCarbon - Configure to be QUIET
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
    # Suppress CodeCarbon's internal logging
    logging.getLogger("codecarbon").setLevel(logging.ERROR)
except Exception:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    print("WARNING: codecarbon not available. Energy/emissions will be NaN.")

# -------------------------
# Config
# -------------------------
DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
BASELINE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/new_baseline"
SAVE_DIR_BASE = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/rerun/pruned_models"
EPOCHS_KD = 50  # Epochs for knowledge distillation
BATCH_SIZE = 64
LR_KD = 0.0005  # Learning rate for student
MIN_LR = 1e-6
WEIGHT_DECAY = 0.01
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EARLY_STOP_PATIENCE = 10

# Knowledge Distillation Parameters
TEMPERATURE = 4.0  # Temperature for softening probabilities
ALPHA = 0.7  # Weight for distillation loss (1-alpha for student loss)

# Benchmarking Parameters
TIMING_BATCHES = 100
WARMUP = 5
NUM_BASELINE_RUNS = 3
PREDICTION_IMAGES = 50

# One-Shot Pruning Parameters
ONESHOT_TARGET_SPARSITY = 0.5  # 40% sparsity (keep 60% of dimensions)
ONESHOT_RECOVERY_EPOCHS = 10   # Epochs for recovery training after pruning
ONESHOT_RECOVERY_LR = 1e-4    # Learning rate for recovery training
IMPORTANCE_CAL_BATCHES = 50   # Number of batches to compute importance scores
GATE_THRESHOLD = 0.01         # Threshold for considering a gate as pruned

os.makedirs(SAVE_DIR_BASE, exist_ok=True)


# -------------------------
# Gate Layer for Structured Pruning
# -------------------------
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


# -------------------------
# Gated ViT Components
# -------------------------
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
        x = self.hidden_gate(x)
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

        x = attn @ v
        x = self.attn_gate(x)
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
        self.res_gate = res_gate

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.res_gate(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.res_gate(x)
        return x


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

        from timm.models.layers import PatchEmbed
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.res_gate = GateLayer(embed_dim, embed_dim, [1, 1, -1])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

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


# -------------------------
# Deployment ViT Components (Physically Smaller)
# -------------------------
class DeployMlp(nn.Module):
    """MLP without gates - physically smaller"""
    def __init__(self, in_features, hidden_features, out_features=None,
                 act_layer=nn.GELU, drop=0.):
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
    """Multi-head attention without gates - physically smaller"""
    def __init__(self, dim, num_heads, head_dim, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
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


class DeployVisionTransformer(nn.Module):
    """Deployment ViT - physically smaller model without gates"""
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

        from timm.models.layers import PatchEmbed
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
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

            qkv_flops = n_patches * d * (num_heads * h_dim * 3)
            attn_flops = n_patches * n_patches * (num_heads * h_dim)
            out_flops = n_patches * (num_heads * h_dim) * d
            mlp_flops = 2 * n_patches * d * mlp_dim

            total_flops += qkv_flops + attn_flops + out_flops + mlp_flops

        total_flops += self.embed_dim * self.num_classes

        return total_flops


# -------------------------
# Taylor Importance Scoring
# -------------------------
def forward_hook(self, input, output):
    """Hook to store output"""
    self.output = output.detach()


def backward_hook(self, grad_input, grad_output):
    """Hook to store gradient"""
    self.grad = (grad_output[0].detach(),)


def taylor2Scorer(gate_module):
    """Taylor2 scorer: Score = |weight × weight.grad|²"""
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

    def prune(self, criteria, threshold):
        """Apply pruning based on threshold"""
        for layer in range(len(criteria)):
            index = np.where(criteria[layer] <= threshold)
            self.pruning_gates[layer][index] *= 0.0
            self.pruning_parameters[layer].data[index] *= 0.0

    def get_num_active(self):
        """Count active (unpruned) units"""
        return sum(np.count_nonzero(gate) for gate in self.pruning_gates)


# -------------------------
# Gated ViT Creation from TIMM
# -------------------------
def create_gated_vit_from_timm(model_name, num_classes, pretrained=True):
    """Create gated ViT and load pretrained weights from TIMM"""

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
        timm_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

        model_dict = model.state_dict()
        timm_dict = timm_model.state_dict()

        transfer_keys = ['patch_embed.proj.weight', 'patch_embed.proj.bias',
                        'pos_embed', 'cls_token', 'norm.weight', 'norm.bias',
                        'head.weight', 'head.bias']

        for key in transfer_keys:
            if key in timm_dict and key in model_dict:
                model_dict[key].copy_(timm_dict[key])

        for i in range(cfg['depth']):
            block_keys = [
                (f'blocks.{i}.norm1.weight', f'blocks.{i}.norm1.weight'),
                (f'blocks.{i}.norm1.bias', f'blocks.{i}.norm1.bias'),
                (f'blocks.{i}.attn.qkv.weight', f'blocks.{i}.attn.qkv.weight'),
                (f'blocks.{i}.attn.qkv.bias', f'blocks.{i}.attn.qkv.bias'),
                (f'blocks.{i}.attn.proj.weight', f'blocks.{i}.attn.proj.weight'),
                (f'blocks.{i}.attn.proj.bias', f'blocks.{i}.attn.proj.bias'),
                (f'blocks.{i}.norm2.weight', f'blocks.{i}.norm2.weight'),
                (f'blocks.{i}.norm2.bias', f'blocks.{i}.norm2.bias'),
                (f'blocks.{i}.mlp.fc1.weight', f'blocks.{i}.mlp.fc1.weight'),
                (f'blocks.{i}.mlp.fc1.bias', f'blocks.{i}.mlp.fc1.bias'),
                (f'blocks.{i}.mlp.fc2.weight', f'blocks.{i}.mlp.fc2.weight'),
                (f'blocks.{i}.mlp.fc2.bias', f'blocks.{i}.mlp.fc2.bias'),
            ]

            for timm_key, model_key in block_keys:
                if timm_key in timm_dict and model_key in model_dict:
                    model_dict[model_key].copy_(timm_dict[timm_key])

        model.load_state_dict(model_dict)
        del timm_model
        cleanup_memory()

    return model


def convert_to_deployment_model(gated_model, num_classes, save_masks=False, save_dir=None):
    """Convert gated model to physically smaller deployment model."""
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

        orig_num_heads = gated_model.num_heads
        for i, (head_mask, mlp_mask) in enumerate(zip(head_masks, mlp_masks)):
            deploy_dict[f'blocks.{i}.norm1.weight'].copy_(gated_dict[f'blocks.{i}.norm1.weight'][res_mask])
            deploy_dict[f'blocks.{i}.norm1.bias'].copy_(gated_dict[f'blocks.{i}.norm1.bias'][res_mask])

            orig_qkv = gated_dict[f'blocks.{i}.attn.qkv.weight']
            orig_qkv_bias = gated_dict[f'blocks.{i}.attn.qkv.bias']
            q_w, k_w, v_w = orig_qkv.chunk(3, dim=0)
            q_b, k_b, v_b = orig_qkv_bias.chunk(3, dim=0)

            head_indices = torch.where(head_mask)[0]
            q_selected, k_selected, v_selected = [], [], []
            qb_selected, kb_selected, vb_selected = [], [], []

            for h in head_indices:
                start = h * head_dim
                end = (h + 1) * head_dim
                q_selected.append(q_w[start:end, res_mask])
                k_selected.append(k_w[start:end, res_mask])
                v_selected.append(v_w[start:end, res_mask])
                qb_selected.append(q_b[start:end])
                kb_selected.append(k_b[start:end])
                vb_selected.append(v_b[start:end])

            if q_selected:
                new_qkv = torch.cat([torch.cat(q_selected), torch.cat(k_selected), torch.cat(v_selected)])
                new_qkv_bias = torch.cat([torch.cat(qb_selected), torch.cat(kb_selected), torch.cat(vb_selected)])
                deploy_dict[f'blocks.{i}.attn.qkv.weight'].copy_(new_qkv)
                deploy_dict[f'blocks.{i}.attn.qkv.bias'].copy_(new_qkv_bias)

            orig_proj = gated_dict[f'blocks.{i}.attn.proj.weight']
            orig_proj_bias = gated_dict[f'blocks.{i}.attn.proj.bias']
            proj_selected = []
            for h in head_indices:
                start = h * head_dim
                end = (h + 1) * head_dim
                proj_selected.append(orig_proj[res_mask][:, start:end])

            if proj_selected:
                deploy_dict[f'blocks.{i}.attn.proj.weight'].copy_(torch.cat(proj_selected, dim=1))
                deploy_dict[f'blocks.{i}.attn.proj.bias'].copy_(orig_proj_bias[res_mask])

            deploy_dict[f'blocks.{i}.norm2.weight'].copy_(gated_dict[f'blocks.{i}.norm2.weight'][res_mask])
            deploy_dict[f'blocks.{i}.norm2.bias'].copy_(gated_dict[f'blocks.{i}.norm2.bias'][res_mask])
            deploy_dict[f'blocks.{i}.mlp.fc1.weight'].copy_(gated_dict[f'blocks.{i}.mlp.fc1.weight'][mlp_mask][:, res_mask])
            deploy_dict[f'blocks.{i}.mlp.fc1.bias'].copy_(gated_dict[f'blocks.{i}.mlp.fc1.bias'][mlp_mask])
            deploy_dict[f'blocks.{i}.mlp.fc2.weight'].copy_(gated_dict[f'blocks.{i}.mlp.fc2.weight'][res_mask][:, mlp_mask])
            deploy_dict[f'blocks.{i}.mlp.fc2.bias'].copy_(gated_dict[f'blocks.{i}.mlp.fc2.bias'][res_mask])

        deploy_dict['norm.weight'].copy_(gated_dict['norm.weight'][res_mask])
        deploy_dict['norm.bias'].copy_(gated_dict['norm.bias'][res_mask])
        deploy_dict['head.weight'].copy_(gated_dict['head.weight'][:, res_mask])
        deploy_dict['head.bias'].copy_(gated_dict['head.bias'])

        deploy_model.load_state_dict(deploy_dict)

    pruning_config = {
        'deploy_embed_dim': new_embed_dim,
        'deploy_per_layer_num_heads': per_layer_num_heads,
        'deploy_per_layer_mlp_dim': per_layer_mlp_dim,
        'head_dim': head_dim
    }

    return deploy_model, pruning_config


# -------------------------
# Early Stopping Class
# -------------------------
class EarlyStopping:
    """Early stops training if validation accuracy doesn't improve after patience epochs."""
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_acc = 0

    def __call__(self, val_acc):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.best_acc = val_acc
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_acc = val_acc
            self.counter = 0


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Memory monitoring
# -------------------------
def log_memory_usage(prefix=""):
    process = psutil.Process()
    mem_info = process.memory_info()
    gpu_mem = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    print(f"{prefix}Memory Usage: RSS={mem_info.rss/(1024**2):.2f}MB, GPU={gpu_mem:.2f}MB")

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# -------------------------
# Dataset utilities
# -------------------------
class NumpyMemmapDataset(Dataset):
    """Wraps a numpy array (H,W[,C]) and a label array."""
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
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

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


def load_dataset(npz_path: str):
    """Load dataset from NPZ file."""
    data = np.load(npz_path, mmap_mode="r")

    X_train = data["train_images"]
    y_train = data["train_labels"].flatten()
    X_val = data["val_images"]
    y_val = data["val_labels"].flatten()
    X_test = data["test_images"]
    y_test = data["test_labels"].flatten()

    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)

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


def specificity_per_class(conf_matrix):
    """Calculates specificity for each class."""
    specificity = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        if (tn + fp) > 0:
            specificity.append(tn / (tn + fp))
        else:
            specificity.append(0.0)
    return specificity


def evaluate_model(net, test_loader, device, use_amp=False):
    """Evaluate the model."""
    net.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        val_bar = tqdm(test_loader, file=sys.stdout, desc="Evaluating", leave=False)
        for inputs, targets in val_bar:
            inputs = inputs.to(device)
            
            if use_amp:
                with autocast():
                    outputs = net(inputs)
            else:
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
    specificity = specificity_per_class(conf_matrix)
    avg_specificity = sum(specificity) / len(specificity) if specificity else 0.0
    
    n_classes = len(conf_matrix)
    all_labels_one_hot = label_binarize(all_labels, classes=list(range(n_classes)))
    
    try:
        auc = roc_auc_score(all_labels_one_hot, all_probs, multi_class='ovr')
    except ValueError:
        auc = float('nan')
    
    metrics = {
        'acc': acc,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'specificity': avg_specificity,
        'f1': f1
    }
    
    return metrics


# -------------------------
# Benchmarking utilities
# -------------------------
def params_count(model):
    return sum(p.numel() for p in model.parameters())

def model_size_bytes(model):
    fd, tmp = tempfile.mkstemp(suffix=".pth")
    os.close(fd)
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp)
    os.remove(tmp)
    return size

def compute_flops(model):
    model.eval()
    try:
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
    except StopIteration:
        model_dtype = torch.float32
        model_device = DEVICE
    
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='No handlers found')
            inputs = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(model_device)
            if model_dtype == torch.half:
                inputs = inputs.half()
            macs = profile_macs(model, inputs)
            flops = macs * 2
            return float(flops)
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        return float("nan")

def inference_time_per_batch(model, loader, warmup=WARMUP, timed=TIMING_BATCHES):
    model.eval()
    try:
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_device = DEVICE
        model_dtype = torch.float32
    
    use_cuda = model_device.type == "cuda"
    
    # Calculate max possible batches
    total_batches = len(loader)
    actual_warmup = min(warmup, max(1, total_batches // 4))
    actual_timed = min(timed, max(1, total_batches - actual_warmup))
    
    it = iter(loader)
    
    # Warmup
    try:
        for _ in range(actual_warmup):
            imgs, _ = next(it)
            imgs = imgs.to(model_device)
            if model_dtype == torch.half:
                imgs = imgs.half()
            with torch.no_grad():
                _ = model(imgs)
            if use_cuda:
                torch.cuda.synchronize()
    except StopIteration:
        it = iter(loader)
    
    # Timed run
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    batches_done = 0
    images_processed = 0
    try:
        for _ in range(actual_timed):
            imgs, _ = next(it)
            imgs = imgs.to(model_device)
            if model_dtype == torch.half:
                imgs = imgs.half()
            with torch.no_grad():
                _ = model(imgs)
            if use_cuda:
                torch.cuda.synchronize()
            batches_done += 1
            images_processed += imgs.size(0)
    except StopIteration:
        pass
    
    elapsed = time.time() - start
    avg_batch = elapsed / max(1, batches_done)
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if use_cuda else params_count(model)*4.0/(1024**2)
    return avg_batch, peak_mb, images_processed


# -------------------------
# CodeCarbon helpers - QUIET MODE
# -------------------------
def start_tracker(save_dir: str, project_name: str, measure_power_secs: int=10):
    """Start tracker with dedicated CSV file per experiment - QUIET MODE."""
    if not CODECARBON_AVAILABLE:
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create UNIQUE CSV file for this specific experiment
    output_file = f"emissions_{project_name}.csv"
    csv_path = os.path.join(save_dir, output_file)
    
    # Remove old file if exists
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except Exception:
            pass
    
    try:
        tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=save_dir,
            output_file=output_file,
            measure_power_secs=measure_power_secs,
            save_to_file=True,
            log_level='error'  # Only show errors
        )
        tracker.start()
        return tracker
    except Exception as e:
        print(f"Error starting tracker: {e}")
        return None


def stop_tracker_and_get_metrics(tracker, save_dir: str, project_name: str):
    """Stop tracker and extract metrics - QUIET MODE."""
    if tracker is None:
        return {
            "emissions_kg": float("nan"),
            "energy_kwh": float("nan"),
            "cpu_power_w": float("nan"),
            "gpu_power_w": float("nan"),
            "ram_power_w": float("nan"),
            "raw_row": None
        }
    
    # Stop the tracker
    try:
        emissions_val = tracker.stop()
    except Exception:
        emissions_val = None
    
    # Give CodeCarbon time to flush to disk
    time.sleep(2)
    
    # Read from the UNIQUE CSV file for this experiment
    output_file = f"emissions_{project_name}.csv"
    csv_path = os.path.join(save_dir, output_file)
    
    if not os.path.exists(csv_path):
        return {
            "emissions_kg": float(emissions_val) if emissions_val is not None else float("nan"),
            "energy_kwh": float("nan"),
            "cpu_power_w": float("nan"),
            "gpu_power_w": float("nan"),
            "ram_power_w": float("nan"),
            "raw_row": None
        }
    
    # Read the CSV
    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return {
                "emissions_kg": float(emissions_val) if emissions_val is not None else float("nan"),
                "energy_kwh": float("nan"),
                "cpu_power_w": float("nan"),
                "gpu_power_w": float("nan"),
                "ram_power_w": float("nan"),
                "raw_row": None
            }
        
        # Get the last row (most recent measurement)
        raw = df.iloc[-1].to_dict()
        
        # Extract metrics
        energy_kwh = float(raw.get("energy_consumed", float("nan")))
        cpu_power = float(raw.get("cpu_power", float("nan")))
        gpu_power = float(raw.get("gpu_power", float("nan")))
        ram_power = float(raw.get("ram_power", float("nan")))
        emissions_kg = float(raw.get("emissions", float("nan"))) if raw.get("emissions") is not None else (
            float(emissions_val) if emissions_val is not None else float("nan")
        )
        
        # Archive the CSV instead of deleting (for debugging)
        archive_path = csv_path.replace(".csv", "_archived.csv")
        try:
            os.rename(csv_path, archive_path)
        except Exception:
            pass
        
        return {
            "emissions_kg": emissions_kg,
            "energy_kwh": energy_kwh,
            "cpu_power_w": cpu_power,
            "gpu_power_w": gpu_power,
            "ram_power_w": ram_power,
            "raw_row": raw
        }
        
    except Exception:
        return {
            "emissions_kg": float(emissions_val) if emissions_val is not None else float("nan"),
            "energy_kwh": float("nan"),
            "cpu_power_w": float("nan"),
            "gpu_power_w": float("nan"),
            "ram_power_w": float("nan"),
            "raw_row": None
        }


def measure_prediction_energy(model, test_loader, save_dir, project_name, num_images=PREDICTION_IMAGES):
    """Measure energy for predictions - QUIET MODE."""
    if not CODECARBON_AVAILABLE:
        return float("nan"), float("nan")
    
    tracker = start_tracker(save_dir, project_name, measure_power_secs=10)
    if tracker is None:
        return float("nan"), float("nan")
    
    model.eval()
    images_processed = 0
    it = iter(test_loader)
    
    try:
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_device = DEVICE
        model_dtype = torch.float32
    
    with torch.no_grad():
        while images_processed < num_images:
            try:
                imgs, _ = next(it)
                imgs = imgs.to(model_device)
                if model_dtype == torch.half:
                    imgs = imgs.half()
                batch_size = imgs.size(0)
                if images_processed + batch_size > num_images:
                    imgs = imgs[:num_images - images_processed]
                _ = model(imgs)
                if model_device.type == "cuda":
                    torch.cuda.synchronize()
                images_processed += imgs.size(0)
            except StopIteration:
                break
    
    metrics = stop_tracker_and_get_metrics(tracker, save_dir, project_name)
    energy_kwh = metrics["energy_kwh"]
    emissions_kg = metrics["emissions_kg"]
    
    energy_per_image_kwh = energy_kwh / images_processed if images_processed > 0 and not math.isnan(energy_kwh) else float("nan")
    
    return energy_per_image_kwh, emissions_kg


def calculate_break_even_safe(retrain_energy_kwh, baseline_energy_per_pred_kwh, pruned_energy_per_pred_kwh):
    if (math.isnan(retrain_energy_kwh) or 
        math.isnan(baseline_energy_per_pred_kwh) or 
        math.isnan(pruned_energy_per_pred_kwh)):
        return float("nan")
    
    delta = baseline_energy_per_pred_kwh - pruned_energy_per_pred_kwh
    
    if delta <= 0:
        return float("inf")
    elif retrain_energy_kwh <= 0:
        return 0.0
    else:
        return retrain_energy_kwh / delta


def measure_baseline_energy_averaged(baseline, test_loader, save_dir, model_name, dataset_name):
    """Measure baseline energy with multiple runs - QUIET MODE."""
    energies_total = []
    energies_per_pred = []
    images_per_run = []
    emissions_per_run = []
    
    for run in range(NUM_BASELINE_RUNS):
        proj = f"{dataset_name}_{model_name}_baseline_inference_run{run}_{int(time.time())}"
        tracker = start_tracker(save_dir, proj, measure_power_secs=10) if CODECARBON_AVAILABLE else None
        
        avg_time, _, images = inference_time_per_batch(baseline, test_loader, timed=TIMING_BATCHES)
        
        metrics = stop_tracker_and_get_metrics(tracker, save_dir, proj)
        energy_kwh = metrics["energy_kwh"]
        emissions_kg = metrics["emissions_kg"]
        
        if images > 0 and not math.isnan(energy_kwh):
            energies_total.append(energy_kwh)
            energies_per_pred.append(energy_kwh / images)
        if not math.isnan(emissions_kg):
            emissions_per_run.append(emissions_kg)
        images_per_run.append(images)
    
    avg_images = np.mean(images_per_run)
    baseline_energy_kwh = np.mean(energies_total) if len(energies_total) > 0 else float("nan")
    baseline_emissions_kg = np.mean(emissions_per_run) if len(emissions_per_run) > 0 else float("nan")
    baseline_energy_per_pred_kwh = np.mean(energies_per_pred) if len(energies_per_pred) > 0 else float("nan")
    
    # Measure prediction energy on separate images
    baseline_pred_energy_per_image_kwh, baseline_pred_emissions_kg = measure_prediction_energy(
        baseline, test_loader, save_dir, f"{dataset_name}_{model_name}_baseline_pred_50images_{int(time.time())}"
    )
    
    return baseline_energy_kwh, baseline_emissions_kg, baseline_energy_per_pred_kwh, float(avg_images), baseline_pred_energy_per_image_kwh


# -------------------------
def test_codecarbon():
    """Test CodeCarbon functionality. Returns True if working, False otherwise."""
    if not CODECARBON_AVAILABLE:
        print("CodeCarbon not installed")
        return False

    test_dir = "/tmp/codecarbon_test"
    os.makedirs(test_dir, exist_ok=True)

    try:
        tracker = EmissionsTracker(
            project_name="test_run",
            output_dir=test_dir,
            output_file="test_emissions.csv",
            measure_power_secs=1,
            save_to_file=True,
            log_level='error'
        )
        tracker.start()

        if torch.cuda.is_available():
            x = torch.randn(2000, 2000).cuda()
            for _ in range(200):
                y = torch.matmul(x, x)
            torch.cuda.synchronize()
        else:
            x = torch.randn(1000, 1000)
            for _ in range(50):
                y = torch.matmul(x, x)

        tracker.stop()
        time.sleep(3)

        csv_path = os.path.join(test_dir, "test_emissions.csv")
        if not os.path.exists(csv_path):
            return False

        df = pd.read_csv(csv_path)
        if df.empty:
            return False

        energy_kwh = df.iloc[-1].get('energy_consumed', float('nan'))
        if math.isnan(energy_kwh) or energy_kwh == 0:
            print("Warning: Energy measurement may be inaccurate")

        print("CodeCarbon test passed")
        return True

    except Exception as e:
        print(f"CodeCarbon test failed: {e}")
        return False


# -------------------------
# Quantization with AMP
# -------------------------
def quantize_model_amp(dataset_name, model_name, baseline_path, baseline_model, test_loader, num_classes, save_dir):
    """Apply AMP quantization."""
    print(f"Quantizing {model_name} on {dataset_name}")

    save_path = os.path.join(save_dir, f'quantization_{model_name}_{dataset_name}_final.pth')
    if os.path.exists(save_path):
        print(f"Quantized model already exists, skipping")
        return None
    
    # Start conversion energy tracking
    conversion_proj = f"{dataset_name}_quantization_{model_name}_conversion_{int(time.time())}"
    conversion_tracker = start_tracker(save_dir, conversion_proj, measure_power_secs=5)
    
    # Create quantized model (FP16)
    quantized_model = timm.create_model(model_name, pretrained=False, num_classes=num_classes).to(DEVICE)
    checkpoint = torch.load(baseline_path, map_location=DEVICE, weights_only=False)
    quantized_model.load_state_dict(checkpoint['model'])
    quantized_model = quantized_model.half()
    
    time.sleep(2)
    
    conversion_metrics = stop_tracker_and_get_metrics(conversion_tracker, save_dir, conversion_proj)
    conversion_energy_kwh = conversion_metrics["energy_kwh"]
    conversion_emissions_kg = conversion_metrics["emissions_kg"]

    quantized_metrics = evaluate_model(quantized_model, test_loader, DEVICE, use_amp=True)
    baseline_metrics = evaluate_model(baseline_model, test_loader, DEVICE, use_amp=False)
    
    # Measure baseline energy
    baseline_energy_kwh, baseline_emissions_kg, baseline_energy_per_pred_kwh, baseline_images, baseline_pred_energy_per_image_kwh = measure_baseline_energy_averaged(
        baseline_model, test_loader, save_dir, model_name, dataset_name
    )
    
    # Measure inference energy for quantized model
    quantized_inf_proj = f"{dataset_name}_quantization_{model_name}_inference_{int(time.time())}"
    quantized_tracker = start_tracker(save_dir, quantized_inf_proj, measure_power_secs=10)
    inf_time, peak_ram, quantized_images = inference_time_per_batch(quantized_model, test_loader, timed=TIMING_BATCHES)
    quantized_inf_metrics = stop_tracker_and_get_metrics(quantized_tracker, save_dir, quantized_inf_proj)
    quantized_energy_kwh = quantized_inf_metrics["energy_kwh"]
    quantized_emissions_kg = quantized_inf_metrics["emissions_kg"]
    quantized_energy_per_pred_kwh = quantized_energy_kwh / quantized_images if quantized_images > 0 and not math.isnan(quantized_energy_kwh) else float("nan")
    
    # Measure prediction energy
    pred_energy_proj = f"{dataset_name}_quantization_{model_name}_pred_50images_{int(time.time())}"
    pred_energy_per_image_kwh, pred_emissions_kg = measure_prediction_energy(
        quantized_model, test_loader, save_dir, pred_energy_proj
    )
    
    # Calculate metrics
    baseline_params = params_count(baseline_model)
    quantized_params = params_count(quantized_model)
    baseline_size = os.path.getsize(baseline_path) / (1024 * 1024) if os.path.exists(baseline_path) else model_size_bytes(baseline_model) / (1024 * 1024)
    
    # Save model
    state = {
        'model': quantized_model.state_dict(),
        'acc': quantized_metrics['acc'],
        'auc': quantized_metrics['auc'],
        'model_name': model_name,
        'dataset': dataset_name,
        'pruning_method': 'quantization'
    }
    torch.save(state, save_path)
    
    quantized_size = os.path.getsize(save_path) / (1024 * 1024)
    compression_ratio = baseline_size / quantized_size if quantized_size > 0 else 1.0
    
    flops = compute_flops(quantized_model)
    flops_m = flops / 1e6 if not math.isnan(flops) else float("nan")
    
    break_even = calculate_break_even_safe(conversion_energy_kwh, baseline_energy_per_pred_kwh, quantized_energy_per_pred_kwh)
    
    acc_drop = (baseline_metrics['acc'] - quantized_metrics['acc']) * 100
    auc_drop = (baseline_metrics['auc'] - quantized_metrics['auc']) * 100

    print(f"Quantization complete: Acc={quantized_metrics['acc']:.4f}, Drop={acc_drop:.2f}%")
    
    results = {
        'Dataset': dataset_name,
        'Variant': f'quantization_{model_name}',
        'TeacherModel': model_name,
        'StudentModel': f'{model_name}_fp16',
        'Stage': 'final',
        'KeepRatio': 1.0,
        'Acc': quantized_metrics['acc'],
        'AUC': quantized_metrics['auc'],
        'Precision': quantized_metrics['precision'],
        'Recall': quantized_metrics['recall'],
        'Specificity': quantized_metrics['specificity'],
        'F1': quantized_metrics['f1'],
        'BaselineAcc': baseline_metrics['acc'],
        'BaselineAUC': baseline_metrics['auc'],
        'AccDrop_percent': acc_drop,
        'AucDrop_percent': auc_drop,
        'Params': quantized_params,
        'BaselineParams': baseline_params,
        'ParamReduction_percent': ((baseline_params - quantized_params) / baseline_params * 100) if baseline_params > 0 else 0,
        'ModelSizeMB': quantized_size,
        'BaselineSizeMB': baseline_size,
        'CompressionRatio': compression_ratio,
        'FLOPs_per_image': flops,
        'FLOPs_M_per_image': flops_m,
        'InferenceTime_per_batch_s': inf_time,
        'PeakRAM_MB': peak_ram,
        'ImagesProcessedDuringTiming': quantized_images,
        'ConversionEnergy_kWh': conversion_energy_kwh,
        'ConversionEmissions_kg': conversion_emissions_kg,
        'BaselineInferenceEnergy_kWh_total': baseline_energy_kwh,
        'BaselineEnergy_per_pred_kWh': baseline_energy_per_pred_kwh,
        'BaselineEmissions_kg_total': baseline_emissions_kg,
        'BaselinePredictionEnergy_per_image_kWh': baseline_pred_energy_per_image_kwh,
        'InferenceEnergy_kWh_total': quantized_energy_kwh,
        'Energy_per_pred_kWh': quantized_energy_per_pred_kwh,
        'Emissions_kg_total': quantized_emissions_kg,
        'PredictionEnergy_per_image_kWh': pred_energy_per_image_kwh,
        'BreakEvenPredictions': break_even,
        'ModelPath': save_path
    }
    
    return results


# -------------------------
# Knowledge Distillation
# -------------------------
class DistillationLoss(nn.Module):
    """Distillation loss combines soft and hard targets."""
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        soft_targets = torch.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = torch.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = self.kl_loss(soft_student, soft_targets) * (self.temperature ** 2)
        student_loss = self.ce_loss(student_logits, labels)
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        return total_loss


def train_epoch_kd(student, teacher, train_loader, optimizer, scheduler, criterion, device):
    """Train student for one epoch with knowledge distillation."""
    student.train()
    teacher.eval()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout, desc="Training (KD)", leave=False)
    
    for step, (images, labels) in enumerate(train_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            teacher_logits = teacher(images)
        
        student_logits = student(images)
        loss = criterion(student_logits, teacher_logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.3f}")
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def knowledge_distillation(dataset_name, teacher_model_name, student_model_name,
                          teacher_path, teacher_model, train_loader, val_loader, test_loader, num_classes, save_dir):
    """Perform knowledge distillation."""
    print(f"KD: {teacher_model_name} -> {student_model_name} on {dataset_name}")

    save_path = os.path.join(save_dir, f'kd_{teacher_model_name}_to_{student_model_name}_{dataset_name}_final.pth')
    if os.path.exists(save_path):
        print(f"KD model already exists, skipping")
        return None
    
    # Start training energy tracking
    training_proj = f"{dataset_name}_kd_{teacher_model_name}_to_{student_model_name}_training_{int(time.time())}"
    training_tracker = start_tracker(save_dir, training_proj, measure_power_secs=10)
    
    teacher = timm.create_model(teacher_model_name, pretrained=False, num_classes=num_classes).to(DEVICE)
    checkpoint = torch.load(teacher_path, map_location=DEVICE, weights_only=False)
    teacher.load_state_dict(checkpoint['model'])
    teacher.eval()

    teacher_metrics = evaluate_model(teacher, test_loader, DEVICE)
    
    # Measure teacher energy
    teacher_energy_kwh, teacher_emissions_kg, teacher_energy_per_pred_kwh, teacher_images, teacher_pred_energy_per_image_kwh = measure_baseline_energy_averaged(
        teacher, test_loader, save_dir, teacher_model_name, dataset_name
    )
    
    student = timm.create_model(student_model_name, pretrained=False, num_classes=num_classes).to(DEVICE)

    teacher_params = params_count(teacher)
    student_params = params_count(student)
    
    # Setup training
    criterion = DistillationLoss(temperature=TEMPERATURE, alpha=ALPHA)
    optimizer = optim.AdamW(student.parameters(), lr=LR_KD, weight_decay=WEIGHT_DECAY)
    
    total_steps = EPOCHS_KD * len(train_loader)
    warmup_steps = 3 * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, verbose=False)
    best_acc, best_auc = 0.0, 0.0

    for epoch in range(EPOCHS_KD):
        train_loss = train_epoch_kd(student, teacher, train_loader, optimizer, scheduler, criterion, DEVICE)
        metrics = evaluate_model(student, val_loader, DEVICE)

        print(f"Epoch {epoch+1}/{EPOCHS_KD}: Loss={train_loss:.4f}, Val Acc={metrics['acc']:.4f}")

        if metrics['acc'] > best_acc:
            best_acc, best_auc = metrics['acc'], metrics['auc']
            state = {
                'model': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'auc': best_auc,
                'epoch': epoch,
                'student_model': student_model_name,
                'teacher_model': teacher_model_name,
                'dataset': dataset_name,
                'pruning_method': 'knowledge_distillation'
            }
            torch.save(state, save_path)
        
        early_stopping(metrics['acc'])
        if early_stopping.early_stop:
            break
    
    time.sleep(3)
    
    # Stop training energy tracking
    training_metrics = stop_tracker_and_get_metrics(training_tracker, save_dir, training_proj)
    training_energy_kwh = training_metrics["energy_kwh"]
    training_emissions_kg = training_metrics["emissions_kg"]

    # Test evaluation
    checkpoint = torch.load(save_path, map_location=DEVICE, weights_only=False)
    student.load_state_dict(checkpoint['model'])
    test_metrics = evaluate_model(student, test_loader, DEVICE)
    
    # Measure student inference energy
    student_inf_proj = f"{dataset_name}_kd_{teacher_model_name}_to_{student_model_name}_inference_{int(time.time())}"
    student_tracker = start_tracker(save_dir, student_inf_proj, measure_power_secs=10)
    inf_time, peak_ram, student_images = inference_time_per_batch(student, test_loader, timed=TIMING_BATCHES)
    student_inf_metrics = stop_tracker_and_get_metrics(student_tracker, save_dir, student_inf_proj)
    student_energy_kwh = student_inf_metrics["energy_kwh"]
    student_emissions_kg = student_inf_metrics["emissions_kg"]
    student_energy_per_pred_kwh = student_energy_kwh / student_images if student_images > 0 and not math.isnan(student_energy_kwh) else float("nan")
    
    # Measure prediction energy
    pred_energy_proj = f"{dataset_name}_kd_{teacher_model_name}_to_{student_model_name}_pred_50images_{int(time.time())}"
    pred_energy_per_image_kwh, pred_emissions_kg = measure_prediction_energy(
        student, test_loader, save_dir, pred_energy_proj
    )
    
    # Calculate metrics
    teacher_size = os.path.getsize(teacher_path) / (1024 * 1024) if os.path.exists(teacher_path) else model_size_bytes(teacher) / (1024 * 1024)
    student_size = os.path.getsize(save_path) / (1024 * 1024)
    compression_ratio = teacher_size / student_size if student_size > 0 else 1.0
    
    flops = compute_flops(student)
    flops_m = flops / 1e6 if not math.isnan(flops) else float("nan")
    
    break_even = calculate_break_even_safe(training_energy_kwh, teacher_energy_per_pred_kwh, student_energy_per_pred_kwh)
    
    acc_drop = (teacher_metrics['acc'] - test_metrics['acc']) * 100
    auc_drop = (teacher_metrics['auc'] - test_metrics['auc']) * 100

    print(f"KD complete: Acc={test_metrics['acc']:.4f}, Drop={acc_drop:.2f}%")
    
    results = {
        'Dataset': dataset_name,
        'Variant': f'kd_{teacher_model_name}_to_{student_model_name}',
        'TeacherModel': teacher_model_name,
        'StudentModel': student_model_name,
        'Stage': 'final',
        'KeepRatio': student_params / teacher_params if teacher_params > 0 else 1.0,
        'Acc': test_metrics['acc'],
        'AUC': test_metrics['auc'],
        'Precision': test_metrics['precision'],
        'Recall': test_metrics['recall'],
        'Specificity': test_metrics['specificity'],
        'F1': test_metrics['f1'],
        'TeacherAcc': teacher_metrics['acc'],
        'TeacherAUC': teacher_metrics['auc'],
        'AccDrop_percent': acc_drop,
        'AucDrop_percent': auc_drop,
        'Params': student_params,
        'TeacherParams': teacher_params,
        'ParamReduction_percent': (1 - student_params/teacher_params)*100,
        'ModelSizeMB': student_size,
        'TeacherSizeMB': teacher_size,
        'CompressionRatio': compression_ratio,
        'FLOPs_per_image': flops,
        'FLOPs_M_per_image': flops_m,
        'InferenceTime_per_batch_s': inf_time,
        'PeakRAM_MB': peak_ram,
        'ImagesProcessedDuringTiming': student_images,
        'TrainingEnergy_kWh': training_energy_kwh,
        'TrainingEmissions_kg': training_emissions_kg,
        'TeacherInferenceEnergy_kWh_total': teacher_energy_kwh,
        'TeacherEnergy_per_pred_kWh': teacher_energy_per_pred_kwh,
        'TeacherEmissions_kg_total': teacher_emissions_kg,
        'TeacherPredictionEnergy_per_image_kWh': teacher_pred_energy_per_image_kwh,
        'InferenceEnergy_kWh_total': student_energy_kwh,
        'Energy_per_pred_kWh': student_energy_per_pred_kwh,
        'Emissions_kg_total': student_emissions_kg,
        'PredictionEnergy_per_image_kWh': pred_energy_per_image_kwh,
        'BreakEvenPredictions': break_even,
        'ModelPath': save_path
    }
    
    return results


# -------------------------
# One-Shot Pruning
# -------------------------
def collect_importance_scores(model, pruners, train_loader, max_batches=IMPORTANCE_CAL_BATCHES):
    """
    Collect importance scores for one-shot pruning.
    Model in eval mode, gradients collected but no weight updates.
    Uses memory-efficient processing with smaller effective batch sizes.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()


    # Clear memory before starting
    cleanup_memory()

    batch_count = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_count >= max_batches:
            break

        # Process in smaller chunks if batch is large to save memory
        chunk_size = min(8, images.size(0))  # Process max 8 images at a time
        num_chunks = (images.size(0) + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            if batch_count >= max_batches:
                break

            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, images.size(0))

            chunk_images = images[start_idx:end_idx].to(DEVICE)
            chunk_labels = labels[start_idx:end_idx].to(DEVICE)

            # Zero gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # Forward + backward
            outputs = model(chunk_images)
            loss = criterion(outputs, chunk_labels)
            loss.backward()

            # Collect scores
            for pruner in pruners:
                pruner.do_step(loss.item())

            # Free memory immediately
            del outputs, loss, chunk_images, chunk_labels

            batch_count += 1

            if batch_count % 10 == 0:
                cleanup_memory()

    cleanup_memory()


def apply_pruning_per_component(model, pruners, target_sparsity=ONESHOT_TARGET_SPARSITY):
    """
    Prune each component type independently for balanced pruning.
    This ensures attention, MLP, and residual are all pruned equally.
    """
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


def train_epoch_oneshot(model, train_loader, optimizer, scheduler, epoch):
    """Train model for one epoch (recovery training after pruning)"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, file=sys.stdout, desc=f"Recovery Epoch {epoch}", leave=False)

    for images, labels in train_bar:
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

        train_bar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.4f}")

    avg_loss = running_loss / len(train_loader)
    acc = correct / total

    return avg_loss, acc


def oneshot_pruning(dataset_name, model_name, baseline_path, baseline_model,
                    train_loader, val_loader, test_loader, num_classes, save_dir):
    """
    ONE-SHOT PRUNING: Prune model based on importance scores, then recover with training.
    """
    print(f"One-shot pruning: {model_name} on {dataset_name}")

    save_path = os.path.join(save_dir, f'oneshot_{model_name}_{dataset_name}_final.pth')
    if os.path.exists(save_path):
        print(f"One-shot model already exists, skipping")
        return None

    # Start pruning energy tracking
    pruning_proj = f"{dataset_name}_oneshot_{model_name}_pruning_{int(time.time())}"
    pruning_tracker = start_tracker(save_dir, pruning_proj, measure_power_secs=10)

    baseline_dict = {k: v.cpu().clone() for k, v in baseline_model.state_dict().items()}
    baseline_params = params_count(baseline_model)

    baseline_model.cpu()
    cleanup_memory()

    gated_model = create_gated_vit_from_timm(model_name, num_classes, pretrained=True).to(DEVICE)
    gated_dict = gated_model.state_dict()

    # Transfer matching weights from baseline
    for key in baseline_dict.keys():
        if key in gated_dict and gated_dict[key].shape == baseline_dict[key].shape:
            gated_dict[key].copy_(baseline_dict[key].to(DEVICE))

    gated_model.load_state_dict(gated_dict)

    del baseline_dict
    cleanup_memory()

    pruners = []
    for component_type in [0, 1, 2]:  # attn, mlp, residual
        modules = prepare_pruning_list(gated_model, component_type)
        if len(modules) > 0:
            pruner = StructuredPruner(gated_model, modules, pruning_momentum=0.9)
            pruners.append(pruner)

    collect_importance_scores(gated_model, pruners, train_loader, max_batches=IMPORTANCE_CAL_BATCHES)
    apply_pruning_per_component(gated_model, pruners, target_sparsity=ONESHOT_TARGET_SPARSITY)

    deploy_model, pruning_config = convert_to_deployment_model(gated_model, num_classes)

    del gated_model, pruners
    cleanup_memory()

    deploy_params = params_count(deploy_model)
    optimizer = optim.AdamW(deploy_model.parameters(), lr=ONESHOT_RECOVERY_LR, weight_decay=WEIGHT_DECAY)
    total_steps = ONESHOT_RECOVERY_EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, ONESHOT_RECOVERY_EPOCHS + 1):
        train_loss, train_acc = train_epoch_oneshot(deploy_model, train_loader, optimizer, scheduler, epoch)
        val_metrics = evaluate_model(deploy_model, val_loader, DEVICE)

        print(f"Recovery {epoch}/{ONESHOT_RECOVERY_EPOCHS}: Val Acc={val_metrics['acc']:.4f}")

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_state = copy.deepcopy(deploy_model.state_dict())

    if best_state is not None:
        deploy_model.load_state_dict(best_state)

    time.sleep(2)

    # Stop pruning energy tracking
    pruning_metrics = stop_tracker_and_get_metrics(pruning_tracker, save_dir, pruning_proj)
    pruning_energy_kwh = pruning_metrics["energy_kwh"]
    pruning_emissions_kg = pruning_metrics["emissions_kg"]

    test_metrics = evaluate_model(deploy_model, test_loader, DEVICE)

    baseline_model.to(DEVICE)

    # Get baseline metrics for comparison
    baseline_metrics = evaluate_model(baseline_model, test_loader, DEVICE)

    # Measure baseline energy
    baseline_energy_kwh, baseline_emissions_kg, baseline_energy_per_pred_kwh, baseline_images, baseline_pred_energy_per_image_kwh = measure_baseline_energy_averaged(
        baseline_model, test_loader, save_dir, model_name, dataset_name
    )

    # Move baseline back to CPU to free GPU memory for remaining operations
    baseline_model.cpu()
    cleanup_memory()

    # Measure pruned model inference energy
    pruned_inf_proj = f"{dataset_name}_oneshot_{model_name}_inference_{int(time.time())}"
    pruned_tracker = start_tracker(save_dir, pruned_inf_proj, measure_power_secs=10)
    inf_time, peak_ram, pruned_images = inference_time_per_batch(deploy_model, test_loader, timed=TIMING_BATCHES)
    pruned_inf_metrics = stop_tracker_and_get_metrics(pruned_tracker, save_dir, pruned_inf_proj)
    pruned_energy_kwh = pruned_inf_metrics["energy_kwh"]
    pruned_emissions_kg = pruned_inf_metrics["emissions_kg"]
    pruned_energy_per_pred_kwh = pruned_energy_kwh / pruned_images if pruned_images > 0 and not math.isnan(pruned_energy_kwh) else float("nan")

    # Measure prediction energy
    pred_energy_proj = f"{dataset_name}_oneshot_{model_name}_pred_50images_{int(time.time())}"
    pred_energy_per_image_kwh, pred_emissions_kg = measure_prediction_energy(
        deploy_model, test_loader, save_dir, pred_energy_proj
    )

    # Calculate metrics
    baseline_size = os.path.getsize(baseline_path) / (1024 * 1024) if os.path.exists(baseline_path) else model_size_bytes(baseline_model) / (1024 * 1024)

    # Save model
    state = {
        'model': deploy_model.state_dict(),
        'acc': test_metrics['acc'],
        'auc': test_metrics['auc'],
        'model_name': model_name,
        'dataset': dataset_name,
        'pruning_method': 'oneshot',
        'pruning_config': pruning_config,
        'target_sparsity': ONESHOT_TARGET_SPARSITY,
        'recovery_epochs': ONESHOT_RECOVERY_EPOCHS
    }
    torch.save(state, save_path)

    pruned_size = os.path.getsize(save_path) / (1024 * 1024)
    compression_ratio = baseline_size / pruned_size if pruned_size > 0 else 1.0

    flops = deploy_model.compute_flops()
    flops_m = flops / 1e6 if not math.isnan(flops) else float("nan")

    break_even = calculate_break_even_safe(pruning_energy_kwh, baseline_energy_per_pred_kwh, pruned_energy_per_pred_kwh)

    acc_drop = (baseline_metrics['acc'] - test_metrics['acc']) * 100
    auc_drop = (baseline_metrics['auc'] - test_metrics['auc']) * 100

    print(f"One-shot complete: Acc={test_metrics['acc']:.4f}, Drop={acc_drop:.2f}%")

    results = {
        'Dataset': dataset_name,
        'Variant': f'oneshot_{model_name}',
        'TeacherModel': model_name,
        'StudentModel': f'{model_name}_pruned',
        'Stage': 'final',
        'KeepRatio': 1.0 - ONESHOT_TARGET_SPARSITY,
        'Acc': test_metrics['acc'],
        'AUC': test_metrics['auc'],
        'Precision': test_metrics['precision'],
        'Recall': test_metrics['recall'],
        'Specificity': test_metrics['specificity'],
        'F1': test_metrics['f1'],
        'BaselineAcc': baseline_metrics['acc'],
        'BaselineAUC': baseline_metrics['auc'],
        'AccDrop_percent': acc_drop,
        'AucDrop_percent': auc_drop,
        'Params': deploy_params,
        'BaselineParams': baseline_params,
        'ParamReduction_percent': ((baseline_params - deploy_params) / baseline_params * 100) if baseline_params > 0 else 0,
        'ModelSizeMB': pruned_size,
        'BaselineSizeMB': baseline_size,
        'CompressionRatio': compression_ratio,
        'FLOPs_per_image': flops,
        'FLOPs_M_per_image': flops_m,
        'InferenceTime_per_batch_s': inf_time,
        'PeakRAM_MB': peak_ram,
        'ImagesProcessedDuringTiming': pruned_images,
        'PruningEnergy_kWh': pruning_energy_kwh,
        'PruningEmissions_kg': pruning_emissions_kg,
        'BaselineInferenceEnergy_kWh_total': baseline_energy_kwh,
        'BaselineEnergy_per_pred_kWh': baseline_energy_per_pred_kwh,
        'BaselineEmissions_kg_total': baseline_emissions_kg,
        'BaselinePredictionEnergy_per_image_kWh': baseline_pred_energy_per_image_kwh,
        'InferenceEnergy_kWh_total': pruned_energy_kwh,
        'Energy_per_pred_kWh': pruned_energy_per_pred_kwh,
        'Emissions_kg_total': pruned_emissions_kg,
        'PredictionEnergy_per_image_kWh': pred_energy_per_image_kwh,
        'BreakEvenPredictions': break_even,
        'TargetSparsity': ONESHOT_TARGET_SPARSITY,
        'RecoveryEpochs': ONESHOT_RECOVERY_EPOCHS,
        'DeployEmbedDim': pruning_config['deploy_embed_dim'],
        'DeployAvgHeads': np.mean(pruning_config['deploy_per_layer_num_heads']),
        'DeployAvgMlpDim': np.mean(pruning_config['deploy_per_layer_mlp_dim']),
        'ModelPath': save_path
    }

    cleanup_memory()

    return results


def evaluate_baseline_model(model_name, baseline_path, test_loader, num_classes, dataset_name, save_dir):
    """Evaluate a baseline model with full benchmarking."""
    print(f"Evaluating baseline: {model_name} on {dataset_name}")

    net = timm.create_model(model_name, pretrained=False, num_classes=num_classes).to(DEVICE)
    checkpoint = torch.load(baseline_path, map_location=DEVICE, weights_only=False)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    
    # Evaluate
    metrics = evaluate_model(net, test_loader, DEVICE, use_amp=False)
    
    # Measure energy
    baseline_energy_kwh, baseline_emissions_kg, baseline_energy_per_pred_kwh, baseline_images, baseline_pred_energy_per_image_kwh = measure_baseline_energy_averaged(
        net, test_loader, save_dir, model_name, dataset_name
    )
    
    # Additional metrics
    params = params_count(net)
    model_size = os.path.getsize(baseline_path) / (1024 * 1024) if os.path.exists(baseline_path) else model_size_bytes(net) / (1024 * 1024)
    flops = compute_flops(net)
    flops_m = flops / 1e6 if not math.isnan(flops) else float("nan")
    
    inf_time, peak_ram, images = inference_time_per_batch(net, test_loader, timed=TIMING_BATCHES)

    print(f"Baseline complete: Acc={metrics['acc']:.4f}")
    
    results = {
        'Dataset': dataset_name,
        'Variant': f'baseline_{model_name}',
        'TeacherModel': model_name,
        'StudentModel': model_name,
        'Stage': 'baseline',
        'KeepRatio': 1.0,
        'Acc': metrics['acc'],
        'AUC': metrics['auc'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'Specificity': metrics['specificity'],
        'F1': metrics['f1'],
        'TeacherAcc': metrics['acc'],
        'TeacherAUC': metrics['auc'],
        'AccDrop_percent': 0.0,
        'AucDrop_percent': 0.0,
        'Params': params,
        'TeacherParams': params,
        'ParamReduction_percent': 0.0,
        'ModelSizeMB': model_size,
        'TeacherSizeMB': model_size,
        'CompressionRatio': 1.0,
        'FLOPs_per_image': flops,
        'FLOPs_M_per_image': flops_m,
        'InferenceTime_per_batch_s': inf_time,
        'PeakRAM_MB': peak_ram,
        'ImagesProcessedDuringTiming': images,
        'TrainingEnergy_kWh': 0.0,
        'TrainingEmissions_kg': 0.0,
        'TeacherInferenceEnergy_kWh_total': baseline_energy_kwh,
        'TeacherEnergy_per_pred_kWh': baseline_energy_per_pred_kwh,
        'TeacherEmissions_kg_total': baseline_emissions_kg,
        'TeacherPredictionEnergy_per_image_kWh': baseline_pred_energy_per_image_kwh,
        'InferenceEnergy_kWh_total': baseline_energy_kwh,
        'Energy_per_pred_kWh': baseline_energy_per_pred_kwh,
        'Emissions_kg_total': baseline_emissions_kg,
        'PredictionEnergy_per_image_kWh': baseline_pred_energy_per_image_kwh,
        'BreakEvenPredictions': float('nan'),
        'ModelPath': baseline_path
    }
    
    return results, net


def save_results_to_csv(results_list, dataset_name, save_dir):
    """Save results to CSV file."""
    csv_path = os.path.join(save_dir, f"{dataset_name}_pruning_results.csv")
    
    if len(results_list) == 0:
        print(f"Warning: No results to save for {dataset_name}")
        return
    
    df = pd.DataFrame(results_list)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


def main():
    set_seed(SEED)
    print(f"Using {DEVICE}")

    codecarbon_ok = test_codecarbon()

    if not codecarbon_ok:
        print("WARNING: CodeCarbon test failed. Energy measurements may be inaccurate.")
        response = input("Continue? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(1)
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    if not os.path.exists(BASELINE_DIR):
        print(f"Error: Baseline directory not found: {BASELINE_DIR}")
        sys.exit(1)

    datasets = ['bloodmnist', 'pathmnist', 'dermamnist']
    baseline_models = ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224']
    kd_pairs = [
        ('vit_base_patch16_224', 'vit_small_patch16_224'),
        ('vit_base_patch16_224', 'vit_tiny_patch16_224'),
        ('vit_small_patch16_224', 'vit_tiny_patch16_224')
    ]
    models_for_quantization = ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224']
    models_for_oneshot = ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224']
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")

        save_dir = os.path.join(SAVE_DIR_BASE, dataset)
        os.makedirs(save_dir, exist_ok=True)

        # Load existing results if available
        csv_path = os.path.join(save_dir, f"{dataset}_pruning_results.csv")
        existing_results = []
        existing_variants = set()
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                existing_results = existing_df.to_dict('records')
                existing_variants = set(existing_df['Variant'].tolist())
            except Exception:
                pass

        # Load dataset
        npz_path = os.path.join(DATASET_DIR, f"{dataset}_224.npz")
        if not os.path.exists(npz_path):
            print(f"Dataset not found: {npz_path}")
            continue

        try:
            train_loader, val_loader, test_loader, num_classes, dataset_name = load_dataset(npz_path)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            continue

        all_results = list(existing_results)
        baseline_cache = {}

        for model_name in baseline_models:
            variant_name = f'baseline_{model_name}'
            baseline_path = os.path.join(BASELINE_DIR, f'{model_name}_{dataset_name}_pretrained.pth')

            if not os.path.exists(baseline_path):
                print(f"Baseline not found: {baseline_path}")
                continue

            if variant_name in existing_variants:
                try:
                    net = timm.create_model(model_name, pretrained=False, num_classes=num_classes).to(DEVICE)
                    checkpoint = torch.load(baseline_path, map_location=DEVICE, weights_only=False)
                    net.load_state_dict(checkpoint['model'])
                    net.eval()
                    baseline_cache[model_name] = (net, baseline_path)
                except Exception as e:
                    print(f"Error loading baseline: {e}")
                continue

            try:
                results, baseline_model = evaluate_baseline_model(
                    model_name, baseline_path, test_loader, num_classes, dataset_name, save_dir
                )
                all_results.append(results)
                baseline_cache[model_name] = (baseline_model, baseline_path)
                cleanup_memory()
            except Exception as e:
                print(f"Error evaluating baseline: {e}")

        for model_name in models_for_quantization:
            variant_name = f'quantization_{model_name}'

            if variant_name in existing_variants:
                continue

            if model_name not in baseline_cache:
                continue

            baseline_model, baseline_path = baseline_cache[model_name]

            try:
                results = quantize_model_amp(
                    dataset_name, model_name, baseline_path, baseline_model,
                    test_loader, num_classes, save_dir
                )
                if results is not None:
                    all_results.append(results)
                cleanup_memory()
            except Exception as e:
                print(f"Error quantizing: {e}")

        for teacher_model, student_model in kd_pairs:
            variant_name = f'kd_{teacher_model}_to_{student_model}'

            if variant_name in existing_variants:
                continue

            if teacher_model not in baseline_cache:
                continue

            teacher_baseline, teacher_path = baseline_cache[teacher_model]

            try:
                results = knowledge_distillation(
                    dataset_name, teacher_model, student_model,
                    teacher_path, teacher_baseline, train_loader, val_loader, test_loader, num_classes, save_dir
                )
                if results is not None:
                    all_results.append(results)
                cleanup_memory()
            except Exception as e:
                print(f"Error in KD: {e}")

        for model_name in models_for_oneshot:
            variant_name = f'oneshot_{model_name}'

            if variant_name in existing_variants:
                continue

            if model_name not in baseline_cache:
                continue

            baseline_model, baseline_path = baseline_cache[model_name]

            try:
                results = oneshot_pruning(
                    dataset_name, model_name, baseline_path, baseline_model,
                    train_loader, val_loader, test_loader, num_classes, save_dir
                )
                if results is not None:
                    all_results.append(results)
                cleanup_memory()
            except Exception as e:
                print(f"Error in one-shot pruning: {e}")

        # Save results (overwrite with all results including new ones)
        save_results_to_csv(all_results, dataset_name, save_dir)

        del train_loader, val_loader, test_loader, baseline_cache
        cleanup_memory()

    print("All datasets processed.")


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nCritical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)