#!/usr/bin/env python3
"""
Outputs:
- Single CSV per dataset with all variants (Variant column distinguishes).
- emissions.csv in SAVE_DIR.
- Corrected to prune 50%, 75%, or 87.5% of bits (keep 50%, 25%, 12.5% from FP32).
- Measures energy for predictions on 50 test images after training, reports energy per image.
- Includes AMP (FP16) variants for regional_gradients and slim_kd methods.
- Total of 6 model variants: baseline, quantization, regional_gradients, regional_gradients_fp16, slim_kd, slim_kd_fp16
- Fixed: Only saves final FP32 models for non-AMP variants, AMP variants only save FP16 version
- Fixed: Measures actual conversion energy for AMP quantization
- Modified: Uses same AUC calculation logic for all methods (quantization, slim_kd_amp, regional_gradients_amp)
- Modified: Checks if final model and CSV exist; if so, only calculates AUC
"""

import os
import time
import math
import random
import tempfile
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import Bottleneck
from torchvision import models, transforms as T
from sklearn.metrics import roc_auc_score
from torchprofile import profile_macs
import psutil
from torch.quantization import quantize_dynamic
import gc

# CodeCarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except Exception:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    print("Warning: codecarbon not available. Energy/emissions will be NaN.")

# -------------------------
# Config
# -------------------------
SAVE_DIR_BASE = "/home/arihangupta/Pruning/dinov2/Pruning/trial_1/pruned_models"
CNN_EXP1_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/CNN_exp1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224
BATCH_SIZE_DEFAULT = 32
TIMING_BATCHES = 100
WARMUP = 5
NUM_BASELINE_RUNS = 3
PREDICTION_IMAGES = 50

TARGET_COMPRESS_RATIOS = [0.5]

# Methods: quantization, regional_gradients (FP32), regional_gradients_amp (FP16), slim_kd (FP32), slim_kd_amp (FP16)
METHODS = ["quantization", "regional_gradients", "regional_gradients_amp", "slim_kd", "slim_kd_amp"]

CAL_EPOCHS = 1
CAL_MAX_BATCHES = 150
CAL_LR = 3e-4

KD_EPOCHS = 2
KD_LR = 3e-4
KD_ALPHA = 0.7
KD_TEMPERATURE = 3.0
KD_MAX_BATCHES = None

FINAL_FINETUNE_EPOCHS = 2
FINAL_LR = 1e-4

LOG_INTERVAL = 20
RG_CAL_MAX_BATCHES = 50

DATASET_BATCH_SIZES = {
    "dermamnist": 32,
    "pathmnist": 16,
    "bloodmnist": 32,
}

DATASETS = {
    "pathmnist": {
        "path": "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced/pathmnist_224.npz",
        "baseline": "/home/arihangupta/Pruning/dinov2/Pruning/CNN_exp1/pathmnist_224_baseline.pth"
    },
    "dermamnist": {
        "path": "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced/dermamnist_224.npz",
        "baseline": "/home/arihangupta/Pruning/dinov2/Pruning/CNN_exp1/dermamnist_224_baseline.pth"
    },
    "bloodmnist": {
        "path": "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced/bloodmnist_224.npz",
        "baseline": "/home/arihangupta/Pruning/dinov2/Pruning/CNN_exp1/bloodmnist_224_baseline.pth"
    },
}

ORIGINAL_PLANES = [64, 128, 256, 512]

# -------------------------
# Repro
# -------------------------
def set_seed(s=SEED, deterministic=True):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

set_seed(SEED, deterministic=True)

# -------------------------
# Memory monitoring
# -------------------------
def log_memory_usage(prefix=""):
    process = psutil.Process()
    mem_info = process.memory_info()
    gpu_mem = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    print(f"{prefix}Memory Usage: RSS={mem_info.rss/(1024**2):.2f}MB, GPU={gpu_mem:.2f}MB")

def cleanup_memory():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# -------------------------
# Data helpers
# -------------------------
class NumpyMemmapDataset(Dataset):
    def __init__(self, imgs_np, labels_np, img_size=224):
        self.imgs = imgs_np
        self.labels = labels_np
        self.img_size = img_size
        self.base_tfms = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

def make_loaders(npz_path, batch_size):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset file not found: {npz_path}")
    data = np.load(npz_path, mmap_mode="r")
    X_train, y_train = data["train_images"], data["train_labels"].flatten()
    X_val, y_val = data["val_images"], data["val_labels"].flatten()
    X_test, y_test = data["test_images"], data["test_labels"].flatten()
    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    print(f"Dataset sizes: train={n_train}, val={n_val}, test={n_test}, total={n_train + n_val + n_test}")
    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE)
    val_ds = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE)
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    num_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    return train_loader, val_loader, test_loader, num_classes, train_ds

# -------------------------
# Models / builder
# -------------------------
class CustomResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3,4,6,3], stage_planes=[64,128,256,512], num_classes=1000, random_init=False):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage_planes = stage_planes[:]
        self.layers_cfg = layers[:]
        self.layer1 = self._make_layer(block, stage_planes[0], layers[0])
        self.layer2 = self._make_layer(block, stage_planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, stage_planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, stage_planes[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(stage_planes[3] * block.expansion, num_classes)

        if random_init:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x,1); x = self.fc(x)
        return x

def build_resnet50_for_load(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_pruned_or_slim_resnet(keep_indices=None, stage_planes=None, num_classes=1000, random_init=False):
    if keep_indices is not None:
        stage_planes = [len(keep_indices['layer1']), len(keep_indices['layer2']),
                        len(keep_indices['layer3']), len(keep_indices['layer4'])]
        random_init = False
    elif stage_planes is not None:
        stage_planes = [max(1, int(p)) for p in stage_planes]
        random_init = True
    else:
        raise ValueError("Provide keep_indices or stage_planes")
    return CustomResNet(block=Bottleneck, layers=[3,4,6,3], stage_planes=stage_planes,
                        num_classes=num_classes, random_init=random_init).to(DEVICE)

STAGES = ["layer1", "layer2", "layer3", "layer4"]

def stage_orig_channels(model, stage_name):
    first_block = next(getattr(model, stage_name).children())
    return first_block.conv1.out_channels

# -------------------------
# Load final model
# -------------------------
def load_final_model(dataset_name, method, compress_ratio, num_classes):
    """Load the final model checkpoint for AUC calculation."""
    save_dir = os.path.join(SAVE_DIR_BASE, dataset_name)
    base_method = method.replace("_amp", "") if method.endswith("_amp") else method
    variant = f"{base_method}_fp16" if method.endswith("_amp") else base_method
    ckpt_name = f"{base_method}_r{int(compress_ratio*100)}compressed_final{'_amp' if method.endswith('_amp') else ''}.pth"
    if base_method == "regional_gradients":
        ckpt_name = f"pgto_{ckpt_name}"
    ckpt_path = os.path.join(save_dir, ckpt_name)
    
    if not os.path.exists(ckpt_path):
        print(f"Final checkpoint not found: {ckpt_path}")
        return None, None
    
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        is_fp16 = next(iter(state_dict.values())).dtype == torch.float16
        
        # Infer stage planes from state dict
        stage_planes = []
        for i in range(4):
            key = f'layer{i+1}.0.conv1.weight'
            if key in state_dict:
                planes = state_dict[key].shape[0]
                stage_planes.append(planes)
            else:
                print(f"Cannot find {key} in state dict to infer stage planes")
                return None, None
        
        if method == "quantization" and not is_fp16:
            model = build_resnet50_for_load(num_classes)
        else:
            model = CustomResNet(block=Bottleneck, layers=[3,4,6,3], stage_planes=stage_planes, num_classes=num_classes)
        
        model.load_state_dict(state_dict)
        if is_fp16:
            model = model.half()
        model = model.to(DEVICE).eval()
        print(f"Loaded final model from {ckpt_path} (FP16={is_fp16})")
        return model, variant
    except Exception as e:
        print(f"Error loading final model from {ckpt_path}: {e}")
        return None, None

# -------------------------
# Importance scoring
# -------------------------
def compute_stage_importance_and_keeps_regional(model: nn.Module, stage_name: str, keep_k: int,
                                               calib_loader: DataLoader, max_batches: int=RG_CAL_MAX_BATCHES):
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    orig_planes = first_block.conv1.out_channels
    expansion = 4
    device = DEVICE
    act_norms = torch.zeros(orig_planes, device=device)
    grad_norms = torch.zeros(orig_planes, device=device)
    weight_l1 = torch.zeros(orig_planes, device=device)

    for block in stage.children():
        w = block.conv3.weight.detach().abs().cpu().numpy()
        for p in range(orig_planes):
            weight_l1[p] += np.sum(w[p*expansion:(p+1)*expansion])
    weight_l1 = weight_l1.to(device)

    saved = {}
    def hook_fn(module, inp, out):
        saved['act'] = out
    handle = stage.register_forward_hook(hook_fn)

    model.train()
    batch_count = 0
    for bidx, (imgs, _) in enumerate(calib_loader, 1):
        imgs = imgs.to(device)
        model.zero_grad()
        _ = model(imgs)
        if 'act' not in saved:
            continue
        act = saved['act']
        loss = (act ** 2).mean()
        loss.backward(retain_graph=True)
        with torch.no_grad():
            Cexp = act.shape[1]
            act_flat = act.detach().permute(1,0,2,3).reshape(Cexp, -1)
            for p in range(orig_planes):
                idx0 = p*expansion
                idx1 = (p+1)*expansion
                part = act_flat[idx0:idx1]
                act_norms[p] += torch.norm(part)
        for block in stage.children():
            g = block.conv3.weight.grad
            if g is None:
                continue
            g_abs = g.abs()
            g_per_out = g_abs.view(g_abs.shape[0], -1).norm(dim=1)
            for p in range(orig_planes):
                idx0 = p*expansion; idx1 = (p+1)*expansion
                grad_norms[p] += g_per_out[idx0:idx1].norm()
        batch_count += 1
        if batch_count >= max_batches:
            break
    handle.remove()

    with torch.no_grad():
        importance = act_norms * grad_norms * weight_l1
        importance_np = importance.cpu().numpy()
    if keep_k >= len(importance_np):
        keep = np.arange(len(importance_np))
    else:
        keep = np.argsort(importance_np)[-keep_k:]
    return np.sort(keep)

def compute_stage_importance_and_keeps(model: nn.Module, stage_name: str, keep_k: int, method: str="regional_gradients", calib_loader: DataLoader=None, max_batches: int=RG_CAL_MAX_BATCHES):
    if method in ["regional_gradients", "regional_gradients_amp"]:
        if calib_loader is None:
            raise ValueError("calib_loader required for regional_gradients method")
        return compute_stage_importance_and_keeps_regional(model, stage_name, keep_k, calib_loader, max_batches)
    else:
        raise ValueError(f"Unknown method: {method}")

# -------------------------
# Weight copying for pruned ResNet
# -------------------------
def build_pruned_resnet_and_copy_weights_fixed(base_model: nn.Module, keep_indices: dict, num_classes: int):
    """Build a pruned ResNet and copy weights with proper mapping."""
    new_model = build_pruned_or_slim_resnet(keep_indices=keep_indices, num_classes=num_classes, random_init=False)
    
    new_model.conv1.weight.data.copy_(base_model.conv1.weight.data)
    new_model.bn1.weight.data.copy_(base_model.bn1.weight.data)
    new_model.bn1.bias.data.copy_(base_model.bn1.bias.data)
    new_model.bn1.running_mean.copy_(base_model.bn1.running_mean)
    new_model.bn1.running_var.copy_(base_model.bn1.running_var)
    
    prev_kept = torch.arange(64, dtype=torch.long, device=DEVICE)
    
    for stage_idx, stage in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
        kept = torch.tensor(keep_indices[stage], dtype=torch.long, device=DEVICE)
        base_stage = getattr(base_model, stage)
        new_stage = getattr(new_model, stage)
        
        print(f"    Copying weights for {stage}: keeping {len(kept)} out of {stage_orig_channels(base_model, stage)} channels")
        
        expanded_rows = torch.cat([torch.arange(p * 4, (p + 1) * 4, dtype=torch.long, device=DEVICE) for p in kept])
        
        for block_idx, (base_block, new_block) in enumerate(zip(base_stage.children(), new_stage.children())):
            if block_idx == 0:
                in_idx = prev_kept
            else:
                in_idx = expanded_rows
            
            out_idx = kept
            
            new_block.conv1.weight.data.copy_(base_block.conv1.weight.data[out_idx][:, in_idx])
            
            new_block.bn1.weight.data.copy_(base_block.bn1.weight.data[out_idx])
            new_block.bn1.bias.data.copy_(base_block.bn1.bias.data[out_idx])
            new_block.bn1.running_mean.copy_(base_block.bn1.running_mean[out_idx])
            new_block.bn1.running_var.copy_(base_block.bn1.running_var[out_idx])
            
            new_block.conv2.weight.data.copy_(base_block.conv2.weight.data[out_idx][:, out_idx])
            new_block.bn2.weight.data.copy_(base_block.bn2.weight.data[out_idx])
            new_block.bn2.bias.data.copy_(base_block.bn2.bias.data[out_idx])
            new_block.bn2.running_mean.copy_(base_block.bn2.running_mean[out_idx])
            new_model.bn2.running_var.copy_(base_block.bn2.running_var[out_idx])
            
            new_block.conv3.weight.data.copy_(base_block.conv3.weight.data[expanded_rows][:, out_idx])
            new_block.bn3.weight.data.copy_(base_block.bn3.weight.data[expanded_rows])
            new_block.bn3.bias.data.copy_(base_block.bn3.bias.data[expanded_rows])
            new_block.bn3.running_mean.copy_(base_block.bn3.running_mean[expanded_rows])
            new_block.bn3.running_var.copy_(base_block.bn3.running_var[expanded_rows])
            
            if base_block.downsample is not None and new_block.downsample is not None:
                downsample_in_idx = prev_kept if block_idx == 0 else expanded_rows
                new_block.downsample[0].weight.data.copy_(base_block.downsample[0].weight.data[expanded_rows][:, downsample_in_idx])
                new_block.downsample[1].weight.data.copy_(base_block.downsample[1].weight.data[expanded_rows])
                new_block.downsample[1].bias.data.copy_(base_block.downsample[1].bias.data[expanded_rows])
                new_block.downsample[1].running_mean.copy_(base_block.downsample[1].running_mean[expanded_rows])
                new_block.downsample[1].running_var.copy_(base_block.downsample[1].running_var[expanded_rows])
        
        prev_kept = expanded_rows
    
    last_kept = torch.tensor(keep_indices['layer4'], dtype=torch.long, device=DEVICE)
    if last_kept.numel() > 0:
        fc_in_idx = torch.cat([torch.arange(p * 4, (p + 1) * 4, dtype=torch.long, device=DEVICE) for p in last_kept])
        if fc_in_idx.numel() > 0:
            new_model.fc.weight.data.copy_(base_model.fc.weight.data[:, fc_in_idx])
    new_model.fc.bias.data.copy_(base_model.fc.bias.data)
    
    return new_model

# -------------------------
# Metrics & eval
# -------------------------
criterion = nn.CrossEntropyLoss()

def evaluate_model_basic(model, loader):
    model.eval()
    loss_total = 0.0; correct = 0; total = 0
    probs_list = []; labels_list = []
    
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = DEVICE
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(model_device)
            labels = labels.to(model_device)
            
            if model_dtype == torch.float16:
                images = images.half()
            
            outputs = model(images)
            
            if outputs.device != labels.device:
                outputs = outputs.to(labels.device)
            
            loss = criterion(outputs, labels)
            loss_total += float(loss.item()) * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0); correct += int(predicted.eq(labels).sum().item())
            probs_list.append(torch.softmax(outputs, dim=1).float().cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    loss_avg = loss_total / max(1, total)
    acc = correct / max(1, total)
    try:
        auc = roc_auc_score(np.concatenate(labels_list), np.concatenate(probs_list), multi_class="ovr")
    except Exception as e:
        print(f"Warning: AUC calculation failed: {e}")
        auc = float("nan")
    return loss_avg, acc, auc

def count_zeros_and_total(model):
    total = 0; zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += int((p == 0).sum().item())
    return zeros, total

def params_count(model):
    return sum(p.numel() for p in model.parameters())

def model_size_bytes(model):
    fd, tmp = tempfile.mkstemp(suffix=".pth"); os.close(fd)
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
        inputs = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(model_device)
        if model_dtype == torch.float16:
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
    it = iter(loader)
    
    try:
        for _ in range(warmup):
            imgs, _ = next(it)
            imgs = imgs.to(model_device)
            if model_dtype == torch.float16:
                imgs = imgs.half()
            with torch.no_grad(): _ = model(imgs)
            if use_cuda: torch.cuda.synchronize()
    except StopIteration:
        pass
    
    if use_cuda: torch.cuda.reset_peak_memory_stats()
    start = time.time()
    batches_done = 0
    images_processed = 0
    try:
        for _ in range(timed):
            imgs, _ = next(it)
            imgs = imgs.to(model_device)
            if model_dtype == torch.float16:
                imgs = imgs.half()
            with torch.no_grad(): _ = model(imgs)
            if use_cuda: torch.cuda.synchronize()
            batches_done += 1
            images_processed += imgs.size(0)
    except StopIteration:
        pass
    elapsed = time.time() - start
    avg_batch = elapsed / max(1, batches_done)
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if use_cuda else params_count(model)*4.0/(1024**2)
    return avg_batch, peak_mb, images_processed

def measure_prediction_energy(model, test_loader, save_dir, project_name, num_images=PREDICTION_IMAGES):
    """Measure energy for predicting on exactly num_images from test_loader."""
    if not CODECARBON_AVAILABLE:
        return float("nan"), float("nan")
    
    tracker = start_tracker(save_dir, project_name, measure_power_secs=10)
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
                if model_dtype == torch.float16:
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
    
    print(f"  Prediction energy for {images_processed} images: total_kWh={energy_kwh}, per_image_kWh={energy_per_image_kwh}, emissions_kg={emissions_kg}")
    return energy_per_image_kwh, emissions_kg

def collect_metrics_row(variant, stage, ratio, model, test_loader, path_hint):
    loss, acc, auc = evaluate_model_basic(model, test_loader)
    zeros, total = count_zeros_and_total(model) if "slim" not in variant and "quantization" not in variant and "amp" not in variant and "fp16" not in variant else (0, params_count(model))
    params = params_count(model)
    flops = compute_flops(model)
    flops_m = flops / 1e6 if not math.isnan(flops) else float("nan")
    avg_time, peak_ram, images_processed = inference_time_per_batch(model, test_loader, timed=TIMING_BATCHES)
    if path_hint is not None and os.path.exists(path_hint):
        size_mb = os.path.getsize(path_hint)/(1024**2)
    else:
        size_mb = model_size_bytes(model)/(1024**2)
    power_m = (flops * ((total - zeros)/total)) / 1e6 if not math.isnan(flops) and total>0 and "slim" not in variant else float("nan")
    return {
        "Variant": variant, "Stage": stage, "Ratio": ratio,
        "Acc": acc, "AUC": auc, "Loss": loss,
        "Params": params, "Zeros": zeros, "TotalParams": total, "PctZeros": (zeros/total)*100 if total>0 else 0,
        "ModelSizeMB": size_mb, "FLOPs_per_image": flops, "FLOPs_M_per_image": flops_m,
        "InferenceTime_per_batch_s": avg_time, "PeakRAM_MB": peak_ram,
        "PowerProxy_MFLOPs": power_m, "ModelPath": path_hint,
        "ImagesProcessedDuringTiming": images_processed
    }

# -------------------------
# Compute AUC only for existing models
# -------------------------
def compute_auc_only(dataset_name, method, compress_ratio, num_classes, test_loader, save_dir, csv_path):
    """Load final model and compute AUC, updating CSV if necessary."""
    model, variant = load_final_model(dataset_name, method, compress_ratio, num_classes)
    if model is None:
        print(f"Skipping AUC calculation for {method} at {compress_ratio*100}% compression: model loading failed")
        return False
    
    print(f"Computing AUC for {variant} at {compress_ratio*100}% compression")
    row = collect_metrics_row(variant, "after_global_finetune" + ("_amp" if method.endswith("_amp") else ""), 1 - compress_ratio, model, test_loader, None)
    print(f"AUC calculated: {row['AUC']:.4f}, Acc: {row['Acc']:.4f}, Loss: {row['Loss']:.4f}")
    
    # Load existing CSV and update AUC
    try:
        df = pd.read_csv(csv_path)
        stage = "after_global_finetune" + ("_amp" if method.endswith("_amp") else "")
        mask = (df["Variant"] == variant) & (df["Stage"] == stage) & (df["Ratio"] == 1 - compress_ratio)
        if mask.any():
            df.loc[mask, "AUC"] = row["AUC"]
            df.loc[mask, "Acc"] = row["Acc"]
            df.loc[mask, "Loss"] = row["Loss"]
            df.to_csv(csv_path, index=False)
            print(f"Updated CSV at {csv_path} with new AUC for {variant}")
        else:
            print(f"No matching row found in CSV for {variant} at stage {stage}. Appending new row.")
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(csv_path, index=False)
            print(f"Appended new row to CSV at {csv_path}")
    except Exception as e:
        print(f"Error updating CSV {csv_path}: {e}. Creating new CSV with AUC row.")
        pd.DataFrame([row]).to_csv(csv_path, index=False)
    
    del model
    cleanup_memory()
    return True

# -------------------------
# Freeze / unfreeze & local calibration
# -------------------------
def freeze_all(model):
    for p in model.parameters(): p.requires_grad = False

def unfreeze_stage(model, stage_name, allow_fc_bn1=False):
    for name, p in model.named_parameters():
        if name.startswith(stage_name):
            p.requires_grad = True
        if allow_fc_bn1 and (name.startswith("fc.") or name.startswith("bn1.")):
            p.requires_grad = True

def calibrate_stage(model, stage_name, train_loader, epochs=CAL_EPOCHS, max_batches=CAL_MAX_BATCHES, lr=CAL_LR, allow_fc_bn1=False):
    freeze_all(model)
    unfreeze_stage(model, stage_name, allow_fc_bn1=allow_fc_bn1)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    steps = 0
    for ep in range(epochs):
        running_loss = 0.0; total = 0; correct = 0
        for bidx, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if model.parameters().__next__().dtype == torch.float16:
                imgs = imgs.half()
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward(); opt.step()
            running_loss += float(loss.item()) * imgs.size(0)
            _, preds = out.max(1)
            total += labels.size(0); correct += int(preds.eq(labels).sum().item())
            steps += 1
            if bidx % LOG_INTERVAL == 0:
                print(f"      Calib {stage_name} ep{ep+1} batch{bidx} - loss {running_loss/max(1,total):.4f}, acc {correct/max(1,total):.4f}")
            if steps >= max_batches:
                return model
    return model

# -------------------------
# KD
# -------------------------
def distill_student(student: nn.Module, teacher: nn.Module, train_loader: DataLoader,
                    epochs: int=KD_EPOCHS, lr: float=KD_LR, alpha: float=KD_ALPHA, T: float=KD_TEMPERATURE,
                    max_batches: int=KD_MAX_BATCHES):
    teacher.eval()
    student.train()
    opt = optim.Adam(student.parameters(), lr=lr)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    device = DEVICE
    
    student_dtype = next(student.parameters()).dtype
    teacher_dtype = next(teacher.parameters()).dtype
    
    use_fp16 = (student_dtype == torch.float16)
    
    for ep in range(epochs):
        running_loss = 0.0; total = 0; correct = 0
        for bidx, (imgs, labels) in enumerate(train_loader, 1):
            if max_batches is not None and bidx > max_batches:
                break
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            if use_fp16:
                with torch.no_grad():
                    if teacher_dtype == torch.float16:
                        t_logits = teacher(imgs.half()).float()
                    else:
                        t_logits = teacher(imgs)
                
                s_logits = student(imgs.half()).float()
            else:
                with torch.no_grad():
                    if teacher_dtype == torch.float16:
                        t_logits = teacher(imgs.half()).float()
                    else:
                        t_logits = teacher(imgs)
                
                s_logits = student(imgs)
            
            loss_ce = criterion(s_logits, labels)
            
            s_log_soft = F.log_softmax(s_logits / T, dim=1)
            with torch.no_grad():
                t_soft = F.softmax(t_logits / T, dim=1)
            
            s_log_soft = torch.clamp(s_log_soft, min=-100)
            t_soft = torch.clamp(t_soft, min=1e-8)
            
            loss_kd = kl_loss(s_log_soft, t_soft) * (T * T)
            
            if torch.isnan(loss_ce) or torch.isnan(loss_kd):
                print(f"      Warning: NaN detected in losses (CE: {loss_ce.item()}, KD: {loss_kd.item()}), skipping batch")
                continue
            
            loss = alpha * loss_ce + (1.0 - alpha) * loss_kd
            
            opt.zero_grad()
            loss.backward()
            
            if use_fp16:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            
            opt.step()
            
            running_loss += float(loss.item()) * imgs.size(0)
            _, preds = s_logits.max(1)
            total += labels.size(0)
            correct += int(preds.eq(labels).sum().item())
            
            if bidx % LOG_INTERVAL == 0:
                avg_loss = running_loss/max(1,total)
                avg_acc = correct/max(1,total)
                print(f"      KD ep{ep+1} batch{bidx} - loss {avg_loss:.4f}, acc {avg_acc:.4f}")
    
    student.eval()
    return student

# -------------------------
# Global finetune with FP16 support
# -------------------------
def global_finetune(model, train_loader, val_loader, epochs=FINAL_FINETUNE_EPOCHS, lr=FINAL_LR):
    model.train()
    
    model_dtype = next(model.parameters()).dtype
    is_fp16 = (model_dtype == torch.float16)
    
    if is_fp16:
        print("    Converting FP16 model to FP32 for stable finetuning...")
        model = model.float()
        model_dtype = torch.float32
    
    opt = optim.Adam(model.parameters(), lr=lr)
    
    for ep in range(epochs):
        running_loss = 0.0; total = 0; correct = 0
        for bidx, (imgs, labels) in enumerate(train_loader, 1):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            
            running_loss += float(loss.item()) * imgs.size(0)
            _, preds = out.max(1)
            total += labels.size(0); correct += int(preds.eq(labels).sum().item())
            
            if bidx % LOG_INTERVAL == 0:
                print(f"    Global FT ep{ep+1} batch{bidx} - loss {running_loss/max(1,total):.4f}, acc {correct/max(1,total):.4f}")
        
        vloss, vacc, vauc = evaluate_model_basic(model, val_loader)
        print(f"    Global FT epoch {ep+1}: ValLoss {vloss:.4f}, ValAcc {vacc:.4f}, ValAUC {vauc:.4f}")
    
    if is_fp16:
        print("    Converting model back to FP16 after finetuning...")
        model = model.half()
    
    model.eval()
    return model

# -------------------------
# Quantization
# -------------------------
def symmetric_quantize_model(model, train_loader, bit_width=8):
    model.eval()
    if bit_width == 16:
        quantized_model = copy.deepcopy(model)
        quantized_model = quantized_model.half()
        quantized_model = quantized_model.to(DEVICE)
        print("Applied FP16 quantization.")
    elif bit_width == 8:
        quantized_model = copy.deepcopy(model)
        quantized_model = quantized_model.cpu()
        
        quantized_model = torch.quantization.quantize_dynamic(
            quantized_model,
            {nn.Conv2d, nn.Linear},
            dtype=torch.qint8
        )
        
        print("Applied INT8 dynamic quantization.")
        return quantized_model
    elif bit_width == 4:
        quantized_model = copy.deepcopy(model)
        for name, module in quantized_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                w_max = torch.max(torch.abs(weight))
                if w_max == 0:
                    continue
                scale = w_max / 7.0
                q_weight = torch.round(weight / scale).clamp(-8, 7)
                module.weight.data = q_weight * scale
        print("Applied custom INT4 quantization (simulated).")
    else:
        print(f"Warning: {bit_width}-bit quantization not implemented. Using model as-is.")
        quantized_model = model
    return quantized_model.to(DEVICE)

def create_amp_quantized_version(model, save_dir, project_name):
    """Create an FP16 (AMP) quantized version and measure conversion energy."""
    if not CODECARBON_AVAILABLE:
        print("  Warning: Cannot measure conversion energy without codecarbon")
        conversion_energy_kwh = float("nan")
        conversion_emissions_kg = float("nan")
    else:
        conversion_tracker = start_tracker(save_dir, project_name, measure_power_secs=10)
    
    amp_model = copy.deepcopy(model)
    amp_model = amp_model.half()
    amp_model = amp_model.to(DEVICE)
    
    if CODECARBON_AVAILABLE:
        conversion_metrics = stop_tracker_and_get_metrics(conversion_tracker, save_dir, project_name)
        conversion_energy_kwh = conversion_metrics["energy_kwh"]
        conversion_emissions_kg = conversion_metrics["emissions_kg"]
        print(f"  Created FP16 (AMP) quantized version. Conversion energy: {conversion_energy_kwh} kWh, emissions: {conversion_emissions_kg} kg")
    else:
        conversion_energy_kwh = float("nan")
        conversion_emissions_kg = float("nan")
        print("  Created FP16 (AMP) quantized version.")
    
    return amp_model, conversion_energy_kwh, conversion_emissions_kg

# -------------------------
# CodeCarbon helpers
# -------------------------
def start_tracker(save_dir: str, project_name: str, output_file: str="emissions.csv", measure_power_secs: int=10):
    if not CODECARBON_AVAILABLE:
        return None
    os.makedirs(save_dir, exist_ok=True)
    tracker = EmissionsTracker(project_name=project_name,
                              output_dir=save_dir,
                              output_file=output_file,
                              measure_power_secs=measure_power_secs,
                              save_to_file=True)
    tracker.start()
    return tracker

def _read_latest_tracker_row(save_dir: str, project_name: str):
    csv_path = os.path.join(save_dir, "emissions.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    try:
        df_match = df[df["project_name"] == project_name]
        if df_match.shape[0] == 0:
            return None
        last = df_match.iloc[-1].to_dict()
        return last
    except Exception:
        try:
            return df.iloc[-1].to_dict()
        except Exception:
            return None

def stop_tracker_and_get_metrics(tracker, save_dir: str, project_name: str):
    if tracker is None:
        return {"emissions_kg": float("nan"), "energy_kwh": float("nan"),
                "cpu_power_w": float("nan"), "gpu_power_w": float("nan"), "ram_power_w": float("nan"),
                "raw_row": None}
    try:
        emissions_val = tracker.stop()
    except Exception as e:
        print(f"Error stopping CodeCarbon tracker: {e}")
        emissions_val = None
    raw = _read_latest_tracker_row(save_dir, project_name)
    if raw is None:
        return {
            "emissions_kg": float(emissions_val) if emissions_val is not None else float("nan"),
            "energy_kwh": float("nan"),
            "cpu_power_w": float("nan"),
            "gpu_power_w": float("nan"),
            "ram_power_w": float("nan"),
            "raw_row": None
        }
    energy_kwh = float(raw.get("energy_consumed", float("nan")))
    cpu_power = float(raw.get("cpu_power", float("nan")))
    gpu_power = float(raw.get("gpu_power", float("nan")))
    ram_power = float(raw.get("ram_power", float("nan")))
    emissions_kg = float(raw.get("emissions", float("nan"))) if raw.get("emissions") is not None else (float(emissions_val) if emissions_val is not None else float("nan"))
    return {
        "emissions_kg": emissions_kg,
        "energy_kwh": energy_kwh,
        "cpu_power_w": cpu_power,
        "gpu_power_w": gpu_power,
        "ram_power_w": ram_power,
        "raw_row": raw
    }

# -------------------------
# Break-even calculation
# -------------------------
def calculate_break_even_safe(retrain_energy_kwh, baseline_energy_per_pred_kwh, pruned_energy_per_pred_kwh):
    """Safely calculate break-even predictions with proper error handling"""
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

# -------------------------
# Energy row creation
# -------------------------
def create_energy_row(method, compress_ratio, keep_ratio, retrain_energy_kwh, 
                     retrain_emissions_kg, baseline_energy_kwh, baseline_energy_per_pred_kwh,
                     baseline_emissions_kg, pruned_energy_kwh, pruned_energy_per_pred_kwh,
                     pruned_emissions_kg, pred_energy_per_image_kwh, break_even):
    return {
        "Variant": method, 
        "Stage": f"energy_summary_r{int(compress_ratio*100)}compressed", 
        "Ratio": keep_ratio,
        "RetrainEnergy_kWh": retrain_energy_kwh, 
        "RetrainEmissions_kg": retrain_emissions_kg,
        "BaselineInferenceEnergy_kWh_total": baseline_energy_kwh,
        "BaselineEnergy_per_pred_kWh": baseline_energy_per_pred_kwh,
        "BaselineEmissions_kg_total": baseline_emissions_kg,
        "PrunedInferenceEnergy_kWh_total": pruned_energy_kwh,
        "PrunedEnergy_per_pred_kWh": pruned_energy_per_pred_kwh,
        "PrunedEmissions_kg_total": pruned_emissions_kg,
        "PredictionEnergy_per_image_kWh": pred_energy_per_image_kwh,
        "BreakEvenPredictions": break_even
    }

# -------------------------
# Averaged baseline energy measurement
# -------------------------
def measure_baseline_energy_averaged(baseline, test_loader, save_dir, dataset_name):
    energies_total = []
    energies_per_pred = []
    images_per_run = []
    emissions_per_run = []
    
    for run in range(NUM_BASELINE_RUNS):
        proj = f"{dataset_name}_baseline_inference_run{run}"
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
    
    print(f"Averaged baseline ({NUM_BASELINE_RUNS} runs): images={avg_images:.0f}, energy_kWh={baseline_energy_kwh}, emissions_kg={baseline_emissions_kg}, per_pred={baseline_energy_per_pred_kwh}")
    
    baseline_pred_energy_per_image_kwh, baseline_pred_emissions_kg = measure_prediction_energy(
        baseline, test_loader, save_dir, f"{dataset_name}_baseline_pred_50images"
    )
    
    return baseline_energy_kwh, baseline_emissions_kg, baseline_energy_per_pred_kwh, float(avg_images), baseline_pred_energy_per_image_kwh

# -------------------------
# Load baseline checkpoint
# -------------------------
def load_baseline_ckpt_safe(path, num_classes):
    """Safely load baseline checkpoint with better error handling"""
    model = build_resnet50_for_load(num_classes)
    if not os.path.exists(path):
        print(f"ERROR: Baseline checkpoint not found: {path}")
        print("Please ensure baseline models are trained first.")
        raise FileNotFoundError(f"Baseline checkpoint missing: {path}")
    
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"Successfully loaded baseline from: {path}")
    except Exception as e:
        print(f"ERROR loading checkpoint {path}: {e}")
        raise
    
    return model.to(DEVICE).eval()

# -------------------------
# Dataset processing with error handling
# -------------------------
# -------------------------
# Dataset processing with error handling
# -------------------------
# -------------------------
# Dataset processing with error handling
# -------------------------
def process_dataset_safely(dataset_name, cfg):
    """Process a single dataset with comprehensive error handling"""
    try:
        print(f"\n\n===================== DATASET: {dataset_name.upper()} =====================")
        log_memory_usage(f"Before loading {dataset_name}: ")

        SAVE_DIR = os.path.join(SAVE_DIR_BASE, dataset_name)
        os.makedirs(SAVE_DIR, exist_ok=True)

        csv_path = os.path.join(SAVE_DIR, f"{dataset_name}_combined_pruning_kd_metrics_with_energy.csv")
        
        batch_size = DATASET_BATCH_SIZES.get(dataset_name, BATCH_SIZE_DEFAULT)
        train_loader, val_loader, test_loader, NUM_CLASSES, train_ds = make_loaders(cfg["path"], batch_size)
        print(f"Data loaded for {dataset_name}. NUM_CLASSES={NUM_CLASSES}, device={DEVICE}, batch_size={batch_size}")
        log_memory_usage(f"After loading data for {dataset_name}: ")

        baseline = load_baseline_ckpt_safe(cfg["baseline"], NUM_CLASSES)
        print("Baseline loaded.")
        log_memory_usage(f"After loading baseline for {dataset_name}: ")

        rows = []

        # Check for existing models and CSV
        methods_to_process = []
        for method in METHODS:
            for compress_ratio in TARGET_COMPRESS_RATIOS:
                base_method = method.replace("_amp", "") if method.endswith("_amp") else method
                ckpt_name = f"{base_method}_r{int(compress_ratio*100)}compressed_final{'_amp' if method.endswith('_amp') else ''}.pth"
                if base_method == "regional_gradients":
                    ckpt_name = f"pgto_{ckpt_name}"
                ckpt_path = os.path.join(SAVE_DIR, ckpt_name)
                if os.path.exists(ckpt_path) and os.path.exists(csv_path):
                    print(f"Found final model {ckpt_path} and CSV {csv_path} for {method}. Computing AUC only.")
                    success = compute_auc_only(dataset_name, method, compress_ratio, NUM_CLASSES, test_loader, SAVE_DIR, csv_path)
                    if not success:
                        methods_to_process.append((method, compress_ratio))
                else:
                    methods_to_process.append((method, compress_ratio))

        # If all methods have final models and CSV, skip further processing
        if not methods_to_process and os.path.exists(csv_path):
            print(f"All methods for {dataset_name} have final models and CSV. Skipping dataset.")
            return True

        # Process baseline if not already in CSV
        if not os.path.exists(csv_path):
            print("=== EVALUATE BASELINE ===")
            base_ckpt = os.path.join(SAVE_DIR, "baseline.pth")
            torch.save(baseline.state_dict(), base_ckpt)
            row = collect_metrics_row("baseline", "baseline", 0.0, baseline, test_loader, base_ckpt)
            rows.append(row)
            print("Baseline done:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

            baseline_energy_kwh, baseline_emissions_kg, baseline_energy_per_pred_kwh, baseline_images, baseline_pred_energy_per_image_kwh = measure_baseline_energy_averaged(baseline, test_loader, SAVE_DIR, dataset_name)
            print(f"Final averaged baseline inference: images={baseline_images:.0f}, energy_kWh={baseline_energy_kwh}, emissions_kg={baseline_emissions_kg}, per_pred={baseline_energy_per_pred_kwh}")

            energy_row = {
                "Variant": "baseline", "Stage": "energy_summary_r0pruned", "Ratio": 1.0,
                "RetrainEnergy_kWh": 0.0, "RetrainEmissions_kg": 0.0,
                "BaselineInferenceEnergy_kWh_total": baseline_energy_kwh,
                "BaselineEnergy_per_pred_kWh": baseline_energy_per_pred_kwh,
                "BaselineEmissions_kg_total": baseline_emissions_kg,
                "PrunedInferenceEnergy_kWh_total": baseline_energy_kwh,
                "PrunedEnergy_per_pred_kWh": baseline_energy_per_pred_kwh,
                "PrunedEmissions_kg_total": baseline_emissions_kg,
                "PredictionEnergy_per_image_kWh": baseline_pred_energy_per_image_kwh,
                "BreakEvenPredictions": float("nan")
            }
            rows.append(energy_row)
            print("  Baseline energy summary:", energy_row)
        else:
            print(f"Baseline already processed in CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            baseline_row = df[(df["Variant"] == "baseline") & (df["Stage"] == "energy_summary_r0pruned")]
            if not baseline_row.empty:
                baseline_energy_kwh = baseline_row["BaselineInferenceEnergy_kWh_total"].iloc[0]
                baseline_emissions_kg = baseline_row["BaselineEmissions_kg_total"].iloc[0]
                baseline_energy_per_pred_kwh = baseline_row["BaselineEnergy_per_pred_kWh"].iloc[0]
                baseline_pred_energy_per_image_kwh = baseline_row["PredictionEnergy_per_image_kWh"].iloc[0]
                baseline_images = float("nan")
            else:
                baseline_energy_kwh, baseline_emissions_kg, baseline_energy_per_pred_kwh, baseline_images, baseline_pred_energy_per_image_kwh = measure_baseline_energy_averaged(baseline, test_loader, SAVE_DIR, dataset_name)
                energy_row = {
                    "Variant": "baseline", "Stage": "energy_summary_r0pruned", "Ratio": 1.0,
                    "RetrainEnergy_kWh": 0.0, "RetrainEmissions_kg": 0.0,
                    "BaselineInferenceEnergy_kWh_total": baseline_energy_kwh,
                    "BaselineEnergy_per_pred_kWh": baseline_energy_per_pred_kwh,
                    "BaselineEmissions_kg_total": baseline_emissions_kg,
                    "PrunedInferenceEnergy_kWh_total": baseline_energy_kwh,
                    "PrunedEnergy_per_pred_kWh": baseline_energy_per_pred_kwh,
                    "PrunedEmissions_kg_total": baseline_emissions_kg,
                    "PredictionEnergy_per_image_kWh": baseline_pred_energy_per_image_kwh,
                    "BreakEvenPredictions": float("nan")
                }
                rows.append(energy_row)
                print("  Baseline energy summary:", energy_row)

        # Resume processing from regional_gradients_amp
        for method, compress_ratio in methods_to_process:
            keep_ratio = 1 - compress_ratio
            if method == "regional_gradients_amp":
                print(f"\n=== PROGRESSIVE PRUNING: method={method} ===")
                is_amp_variant = True
                base_method = "regional_gradients"
                
                prune_retrain_proj = f"{dataset_name}_{base_method}_r{int(compress_ratio*100)}compressed_prune_retrain"
                prune_retrain_tracker = start_tracker(SAVE_DIR, prune_retrain_proj, measure_power_secs=10) if CODECARBON_AVAILABLE else None

                current_model = copy.deepcopy(baseline).to(DEVICE)
                keep_indices = {s: np.arange(stage_orig_channels(current_model, s)) for s in STAGES}
                log_memory_usage(f"Before pruning loop for {method}, compress_ratio={compress_ratio}: ")

                for s in STAGES:
                    orig = stage_orig_channels(current_model, s)
                    keep_k = max(1, int(math.floor(orig * keep_ratio)))
                    keeps = compute_stage_importance_and_keeps(current_model, s, keep_k, method=base_method, calib_loader=train_loader, max_batches=RG_CAL_MAX_BATCHES)
                    keep_indices[s] = keeps
                    print(f"  Stage {s}: keep {len(keeps)}/{orig} ({100*len(keeps)/orig:.1f}% kept)")
                    
                    stage_specific_indices = {k: keep_indices[k] if k==s else np.arange(stage_orig_channels(current_model, k)) for k in STAGES}
                    pruned_model = build_pruned_resnet_and_copy_weights_fixed(current_model, stage_specific_indices, num_classes=NUM_CLASSES)
                    pruned_model = pruned_model.to(DEVICE).eval()
                    
                    with torch.no_grad():
                        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
                        _ = pruned_model(dummy_input)
                    
                    stage_pruned_ckpt = os.path.join(SAVE_DIR, f"pgto_{base_method}_r{int(compress_ratio*100)}compressed_{s}_postprune.pth")
                    torch.save(pruned_model.state_dict(), stage_pruned_ckpt)
                    row = collect_metrics_row(base_method, f"{s}_postprune", keep_ratio, pruned_model, test_loader, stage_pruned_ckpt)
                    rows.append(row)
                    print("    Post-prune metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})
                    
                    print(f"    Calibrating {s} (local)...")
                    pruned_model = calibrate_stage(pruned_model, s, train_loader, epochs=CAL_EPOCHS, max_batches=CAL_MAX_BATCHES, lr=CAL_LR, allow_fc_bn1=False)
                    stage_calib_ckpt = os.path.join(SAVE_DIR, f"pgto_{base_method}_r{int(compress_ratio*100)}compressed_{s}_calibrated.pth")
                    torch.save(pruned_model.state_dict(), stage_calib_ckpt)
                    row = collect_metrics_row(base_method, f"{s}_calibrated", keep_ratio, pruned_model, test_loader, stage_calib_ckpt)
                    rows.append(row)
                    print("    Post-calibration metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})
                    
                    current_model = pruned_model
                    del pruned_model
                    cleanup_memory()
                    log_memory_usage(f"After calibrating {s} for {method}: ")

                print("  Final global finetune...")
                current_model = global_finetune(current_model, train_loader, val_loader, epochs=FINAL_FINETUNE_EPOCHS, lr=FINAL_LR)
                
                # Measure retrain energy
                prune_retrain_metrics = stop_tracker_and_get_metrics(prune_retrain_tracker, SAVE_DIR, prune_retrain_proj)
                retrain_energy_kwh = prune_retrain_metrics["energy_kwh"]
                retrain_emissions_kg = prune_retrain_metrics["emissions_kg"]
                print(f"  Retrain energy_kWh={retrain_energy_kwh}, emissions_kg={retrain_emissions_kg}")

                # Create AMP (FP16) version
                print("\n  === Creating AMP (FP16) version for regional_gradients ===")
                conversion_proj = f"{dataset_name}_{method}_r{int(compress_ratio*100)}compressed_conversion"
                amp_model, conversion_energy_kwh, conversion_emissions_kg = create_amp_quantized_version(
                    current_model, SAVE_DIR, conversion_proj
                )
                
                # Save FP16 version
                amp_ckpt = os.path.join(SAVE_DIR, f"pgto_{base_method}_r{int(compress_ratio*100)}compressed_final_amp.pth")
                torch.save(amp_model.state_dict(), amp_ckpt)
                row = collect_metrics_row(f"{base_method}_fp16", "after_global_finetune_amp", keep_ratio, amp_model, test_loader, amp_ckpt)
                rows.append(row)
                print("  AMP Final metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

                # Measure AMP inference energy
                amp_inf_proj = f"{dataset_name}_{method}_r{int(compress_ratio*100)}compressed_inference"
                amp_tracker = start_tracker(SAVE_DIR, amp_inf_proj, measure_power_secs=10) if CODECARBON_AVAILABLE else None
                _, _, amp_images = inference_time_per_batch(amp_model, test_loader, timed=TIMING_BATCHES)
                amp_inf_metrics = stop_tracker_and_get_metrics(amp_tracker, SAVE_DIR, amp_inf_proj)
                amp_energy_kwh = amp_inf_metrics["energy_kwh"]
                amp_emissions_kg = amp_inf_metrics["emissions_kg"]
                amp_energy_per_pred_kwh = amp_energy_kwh / amp_images if amp_images > 0 and not math.isnan(amp_energy_kwh) else float("nan")
                print(f"  AMP inference: images={amp_images}, energy_kWh={amp_energy_kwh}, emissions_kg={amp_emissions_kg}")

                amp_pred_energy_proj = f"{dataset_name}_{method}_r{int(compress_ratio*100)}compressed_pred_50images"
                amp_pred_energy_per_image_kwh, amp_pred_emissions_kg = measure_prediction_energy(
                    amp_model, test_loader, SAVE_DIR, amp_pred_energy_proj
                )

                amp_retrain_energy_kwh = conversion_energy_kwh
                amp_retrain_emissions_kg = conversion_emissions_kg
                amp_break_even = calculate_break_even_safe(amp_retrain_energy_kwh, baseline_energy_per_pred_kwh, amp_energy_per_pred_kwh)

                amp_energy_row = create_energy_row(
                    f"{base_method}_fp16", compress_ratio, keep_ratio, amp_retrain_energy_kwh,
                    amp_retrain_emissions_kg, baseline_energy_kwh, baseline_energy_per_pred_kwh, baseline_emissions_kg,
                    amp_energy_kwh, amp_energy_per_pred_kwh, amp_emissions_kg,
                    amp_pred_energy_per_image_kwh, amp_break_even
                )
                rows.append(amp_energy_row)
                print("  AMP Energy summary:", amp_energy_row)

                del amp_model, current_model
                cleanup_memory()
                log_memory_usage(f"After completing {method} for {dataset_name}: ")

            elif method == "slim_kd_amp":
                print(f"\n=== SLIM KD VARIANT: {method} ===")
                is_amp_variant = True
                base_method = "slim_kd"
                
                ckpt_name = f"{base_method}_r{int(compress_ratio*100)}compressed_final_amp.pth"
                ckpt_path = os.path.join(SAVE_DIR, ckpt_name)
                if os.path.exists(ckpt_path) and os.path.exists(csv_path):
                    print(f"Found final model {ckpt_path} and CSV {csv_path} for {method}. Computing AUC only.")
                    success = compute_auc_only(dataset_name, method, compress_ratio, NUM_CLASSES, test_loader, SAVE_DIR, csv_path)
                    if success:
                        continue
                
                kd_ft_proj = f"{dataset_name}_{base_method}_r{int(compress_ratio*100)}compressed_kd_ft"
                kd_ft_tracker = start_tracker(SAVE_DIR, kd_ft_proj, measure_power_secs=10) if CODECARBON_AVAILABLE else None

                stage_planes = [max(1, int(p * keep_ratio)) for p in ORIGINAL_PLANES]
                current_model = build_pruned_or_slim_resnet(stage_planes=stage_planes, num_classes=NUM_CLASSES, random_init=True)
                print(f"  Slim student built (random init, planes: {stage_planes}).")

                pre_kd_ckpt = os.path.join(SAVE_DIR, f"{base_method}_r{int(compress_ratio*100)}compressed_pre_kd.pth")
                torch.save(current_model.state_dict(), pre_kd_ckpt)
                row = collect_metrics_row(base_method, "pre_kd", keep_ratio, current_model, test_loader, pre_kd_ckpt)
                rows.append(row)
                print("  Pre-KD metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

                print("  Knowledge distillation...")
                current_model = distill_student(current_model, baseline, train_loader, epochs=KD_EPOCHS, lr=KD_LR, alpha=KD_ALPHA, T=KD_TEMPERATURE, max_batches=KD_MAX_BATCHES)
                kd_ckpt = os.path.join(SAVE_DIR, f"{base_method}_r{int(compress_ratio*100)}compressed_afterKD.pth")
                torch.save(current_model.state_dict(), kd_ckpt)
                row = collect_metrics_row(base_method, "after_kd", keep_ratio, current_model, test_loader, kd_ckpt)
                rows.append(row)
                print("  KD metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

                print("  Final global finetune...")
                current_model = global_finetune(current_model, train_loader, val_loader, epochs=FINAL_FINETUNE_EPOCHS, lr=FINAL_LR)
                
                kd_ft_metrics = stop_tracker_and_get_metrics(kd_ft_tracker, SAVE_DIR, kd_ft_proj)
                retrain_energy_kwh = kd_ft_metrics["energy_kwh"]
                retrain_emissions_kg = kd_ft_metrics["emissions_kg"]
                print(f"  Retrain energy_kWh={retrain_energy_kwh}, emissions_kg={retrain_emissions_kg}")

                print("\n  === Creating AMP (FP16) version for slim_kd ===")
                conversion_proj = f"{dataset_name}_{method}_r{int(compress_ratio*100)}compressed_conversion"
                amp_model, conversion_energy_kwh, conversion_emissions_kg = create_amp_quantized_version(
                    current_model, SAVE_DIR, conversion_proj
                )
                
                # Save FP16 version
                amp_ckpt = os.path.join(SAVE_DIR, f"{base_method}_r{int(compress_ratio*100)}compressed_final_amp.pth")
                torch.save(amp_model.state_dict(), amp_ckpt)
                row = collect_metrics_row(f"{base_method}_fp16", "after_global_finetune_amp", keep_ratio, amp_model, test_loader, amp_ckpt)
                rows.append(row)
                print("  AMP Final metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

                # Measure AMP inference energy
                amp_inf_proj = f"{dataset_name}_{method}_r{int(compress_ratio*100)}compressed_inference"
                amp_tracker = start_tracker(SAVE_DIR, amp_inf_proj, measure_power_secs=10) if CODECARBON_AVAILABLE else None
                _, _, amp_images = inference_time_per_batch(amp_model, test_loader, timed=TIMING_BATCHES)
                amp_inf_metrics = stop_tracker_and_get_metrics(amp_tracker, SAVE_DIR, amp_inf_proj)
                amp_energy_kwh = amp_inf_metrics["energy_kwh"]
                amp_emissions_kg = amp_inf_metrics["emissions_kg"]
                amp_energy_per_pred_kwh = amp_energy_kwh / amp_images if amp_images > 0 and not math.isnan(amp_energy_kwh) else float("nan")
                print(f"  AMP inference: images={amp_images}, energy_kWh={amp_energy_kwh}, emissions_kg={amp_emissions_kg}")

                amp_pred_energy_proj = f"{dataset_name}_{method}_r{int(compress_ratio*100)}compressed_pred_50images"
                amp_pred_energy_per_image_kwh, amp_pred_emissions_kg = measure_prediction_energy(
                    amp_model, test_loader, SAVE_DIR, amp_pred_energy_proj
                )

                amp_retrain_energy_kwh = conversion_energy_kwh
                amp_retrain_emissions_kg = conversion_emissions_kg
                amp_break_even = calculate_break_even_safe(amp_retrain_energy_kwh, baseline_energy_per_pred_kwh, amp_energy_per_pred_kwh)

                amp_energy_row = create_energy_row(
                    f"{base_method}_fp16", compress_ratio, keep_ratio, amp_retrain_energy_kwh,
                    amp_retrain_emissions_kg, baseline_energy_kwh, baseline_energy_per_pred_kwh, baseline_emissions_kg,
                    amp_energy_kwh, amp_energy_per_pred_kwh, amp_emissions_kg,
                    amp_pred_energy_per_image_kwh, amp_break_even
                )
                rows.append(amp_energy_row)
                print("  AMP Energy summary:", amp_energy_row)

                del amp_model, current_model
                cleanup_memory()
                log_memory_usage(f"After completing {method} for {dataset_name}: ")

        # Save results to CSV
        try:
            if os.path.exists(csv_path):
                print(f"Appending to existing CSV: {csv_path}")
                existing_df = pd.read_csv(csv_path)
                new_df = pd.DataFrame(rows)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_csv(csv_path, index=False)
                print(f"Updated CSV saved with {len(combined_df)} rows.")
            else:
                print(f"Creating new CSV: {csv_path}")
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"New CSV saved with {len(rows)} rows.")
        except Exception as e:
            print(f"Error saving CSV {csv_path}: {e}")
            fallback_csv = os.path.join(SAVE_DIR, f"{dataset_name}_combined_pruning_kd_metrics_with_energy_fallback.csv")
            print(f"Saving to fallback CSV: {fallback_csv}")
            pd.DataFrame(rows).to_csv(fallback_csv, index=False)
        
        del baseline, train_loader, val_loader, test_loader, train_ds
        cleanup_memory()
        log_memory_usage(f"After completing {dataset_name}: ")
        print(f"===================== DATASET {dataset_name.upper()} DONE =====================")
        return True

    except Exception as e:
        print(f"\nERROR processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

# -------------------------
# Main execution
# -------------------------
def main():
    successes = []
    for dataset_name, cfg in DATASETS.items():
        success = process_dataset_safely(dataset_name, cfg)
        successes.append(success)
    print("\n=== SUMMARY ===")
    for dataset_name, success in zip(DATASETS.keys(), successes):
        print(f"{dataset_name}: {'Success' if success else 'Failed'}")
    if all(successes):
        print("All datasets processed successfully.")
    else:
        print("Some datasets failed to process.")
        raise RuntimeError("One or more datasets failed to process.")

if __name__ == "__main__":
    main()