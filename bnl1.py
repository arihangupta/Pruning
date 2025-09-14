#!/usr/bin/env python3
"""
Outputs:
- Single CSV per dataset with all variants (Variant column distinguishes).
- emissions.csv in SAVE_DIR.

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
SAVE_DIR_BASE = "/home/arihangupta/Pruning/dinov2/Pruning/combined_pruning_kd_experiment"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224
BATCH_SIZE_DEFAULT = 32
TIMING_BATCHES = 100  # Increased for better energy stability
WARMUP = 5
NUM_BASELINE_RUNS = 3  # Average over 3 runs for baseline consistency

# Pruning ratios (keep ratios)
TARGET_RATIOS = [0.5, 0.6, 0.7]

# Methods: pruning + slim_kd
METHODS = ["regional_gradients", "l1", "bn_gamma", "slim_kd"]

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
    "octmnist": 16,
    "tissuemnist": 8,
}

DATASETS = {
    "pathmnist": {
        "path": "/home/arihangupta/Pruning/dinov2/Pruning/datasets/pathmnist_224.npz",
        "baseline": "/home/arihangupta/Pruning/dinov2/Pruning/exp1_saved_models/pathmnist_224_baseline.pth"
    },
    "dermamnist": {
        "path": "/home/arihangupta/Pruning/dinov2/Pruning/datasets/dermamnist_224.npz",
        "baseline": "/home/arihangupta/Pruning/dinov2/Pruning/exp1_saved_models/dermamnist_224_baseline.pth"
    },
    "bloodmnist": {
        "path": "/home/arihangupta/Pruning/dinov2/Pruning/datasets/bloodmnist_224.npz",
        "baseline": "/home/arihangupta/Pruning/dinov2/Pruning/exp1_saved_models/bloodmnist_224_baseline.pth"
    },
    "octmnist": {
        "path": "/home/arihangupta/Pruning/dinov2/Pruning/datasets/octmnist_224.npz",
        "baseline": "/home/arihangupta/Pruning/dinov2/Pruning/exp1_saved_models/octmnist_224_baseline.pth"
    },
    "tissuemnist": {
        "path": "/home/arihangupta/Pruning/dinov2/Pruning/datasets/tissuemnist_224.npz",
        "baseline": "/home/arihangupta/Pruning/dinov2/Pruning/exp1_saved_models/tissuemnist_224_baseline.pth"
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
        # For pruning: use keep_indices to set planes
        stage_planes = [len(keep_indices['layer1']), len(keep_indices['layer2']),
                        len(keep_indices['layer3']), len(keep_indices['layer4'])]
        random_init = False
    elif stage_planes is not None:
        # For slim: use scaled planes
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
# Importance scoring (pruning only)
# -------------------------
def compute_stage_importance_and_keeps_l1(model: nn.Module, stage_name: str, keep_k: int):
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    orig_planes = first_block.conv1.out_channels
    expansion = 4
    weight_l1 = np.zeros(orig_planes, dtype=float)
    for block in stage.children():
        w = block.conv3.weight.detach().cpu().numpy()  # shape (C_out, C_in, k, k)
        for p in range(orig_planes):
            weight_l1[p] += float(np.sum(np.abs(w[p*expansion:(p+1)*expansion])))
    if keep_k >= len(weight_l1):
        keep = np.arange(len(weight_l1))
    else:
        keep = np.argsort(weight_l1)[-keep_k:]
    return np.sort(keep)

def compute_stage_importance_and_keeps_bn_gamma(model: nn.Module, stage_name: str, keep_k: int):
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    orig_planes = first_block.conv1.out_channels
    expansion = 4
    agg = np.zeros(orig_planes, dtype=float)
    for block in stage.children():
        gammas = block.bn3.weight.detach().abs().cpu().numpy()  # length = C_out (expanded)
        for p in range(orig_planes):
            vals = gammas[p*expansion:(p+1)*expansion]
            agg[p] += float(np.mean(vals))
    if keep_k >= len(agg):
        keep = np.arange(len(agg))
    else:
        keep = np.argsort(agg)[-keep_k:]
    return np.sort(keep)

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

    # Weight L1 proxy
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
    for bidx, (imgs, _) in enumerate(calib_loader):
        if bidx >= max_batches:
            break
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
            g_abs = g.detach().abs()
            g_per_out = g_abs.view(g_abs.shape[0], -1).norm(dim=1)
            for p in range(orig_planes):
                idx0 = p*expansion; idx1 = (p+1)*expansion
                grad_norms[p] += g_per_out[idx0:idx1].norm()
        batch_count += 1
        saved.pop('act', None)
    handle.remove()

    if batch_count == 0:
        agg = weight_l1.cpu().numpy()
    else:
        act_norms /= batch_count
        grad_norms /= batch_count
        agg = (act_norms * grad_norms * weight_l1).cpu().numpy()
    keep = np.arange(len(agg)) if keep_k >= len(agg) else np.argsort(agg)[-keep_k:]
    return np.sort(keep)

def compute_stage_importance_and_keeps(model: nn.Module, stage_name: str, keep_k: int,
                                      method="regional_gradients", calib_loader=None, max_batches=RG_CAL_MAX_BATCHES):
    if method == "l1":
        return compute_stage_importance_and_keeps_l1(model, stage_name, keep_k)
    elif method == "bn_gamma":
        return compute_stage_importance_and_keeps_bn_gamma(model, stage_name, keep_k)
    elif method == "regional_gradients":
        assert calib_loader is not None, "calib_loader required for regional_gradients"
        return compute_stage_importance_and_keeps_regional(model, stage_name, keep_k, calib_loader, max_batches)
    else:
        raise ValueError(f"Unknown method {method}")

# -------------------------
# Surgery (pruning only)
# -------------------------
def copy_bn_params(new_bn, old_bn, indices):
    new_bn.weight.data.copy_(old_bn.weight.data[indices])
    new_bn.bias.data.copy_(old_bn.bias.data[indices])
    new_bn.running_mean.data.copy_(old_bn.running_mean.data[indices])
    new_bn.running_var.data.copy_(old_bn.running_var.data[indices])

def build_pruned_resnet_and_copy_weights_fixed(base_model: nn.Module, keep_indices: dict, num_classes: int):
    expansion = 4
    stage_planes = [len(keep_indices['layer1']), len(keep_indices['layer2']),
                    len(keep_indices['layer3']), len(keep_indices['layer4'])]
    new_model = CustomResNet(block=Bottleneck, layers=[3,4,6,3], stage_planes=stage_planes, num_classes=num_classes).to(DEVICE)
    new_model.eval()
    base_model = base_model.to(DEVICE)
    prev_out_idx = torch.arange(base_model.conv1.out_channels, dtype=torch.long, device=DEVICE)
    
    for stage_name in ["layer1", "layer2", "layer3", "layer4"]:
        old_stage = getattr(base_model, stage_name)
        new_stage = getattr(new_model, stage_name)
        kept_planes = torch.tensor(keep_indices[stage_name], dtype=torch.long, device=DEVICE)
        for block_idx, (old_block, new_block) in enumerate(zip(old_stage, new_stage)):
            in_idx = prev_out_idx
            out_planes = kept_planes
            expanded_rows = torch.cat([ (k * expansion + torch.arange(expansion, device=DEVICE)) for k in out_planes ]) if len(out_planes) > 0 else torch.tensor([], dtype=torch.long, device=DEVICE)
            
            # conv1
            old_w = old_block.conv1.weight.data
            if out_planes.numel() > 0 and in_idx.numel() > 0:
                new_block.conv1.weight.data.copy_(old_w[out_planes][:, in_idx, :, :])
            if getattr(old_block.conv1, 'bias', None) is not None and out_planes.numel() > 0:
                new_block.conv1.bias.data.copy_(old_block.conv1.bias.data[out_planes])
            if out_planes.numel() > 0:
                copy_bn_params(new_block.bn1, old_block.bn1, out_planes)
            
            # conv2
            if out_planes.numel() > 0:
                new_block.conv2.weight.data.copy_(old_block.conv2.weight.data[out_planes][:, out_planes, :, :])
            if getattr(old_block.conv2, 'bias', None) is not None and out_planes.numel() > 0:
                new_block.conv2.bias.data.copy_(old_block.conv2.bias.data[out_planes])
            if out_planes.numel() > 0:
                copy_bn_params(new_block.bn2, old_block.bn2, out_planes)
            
            # conv3
            if expanded_rows.numel() > 0 and out_planes.numel() > 0:
                new_block.conv3.weight.data.copy_(old_block.conv3.weight.data[expanded_rows][:, out_planes, :, :])
            if getattr(old_block.conv3, 'bias', None) is not None and expanded_rows.numel() > 0:
                new_block.conv3.bias.data.copy_(old_block.conv3.bias.data[expanded_rows])
            if expanded_rows.numel() > 0:
                copy_bn_params(new_block.bn3, old_block.bn3, expanded_rows)
            
            # downsample
            if old_block.downsample is not None:
                ds_conv = old_block.downsample[0]
                ds_bn = old_block.downsample[1]
                if expanded_rows.numel() > 0 and in_idx.numel() > 0:
                    new_block.downsample[0].weight.data.copy_(ds_conv.weight.data[expanded_rows][:, in_idx, :, :])
                if expanded_rows.numel() > 0:
                    copy_bn_params(new_block.downsample[1], ds_bn, expanded_rows)
            
            prev_out_idx = expanded_rows
            
    # Final FC layer - handle case where no channels remain
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
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += float(loss.item()) * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0); correct += int(predicted.eq(labels).sum().item())
            probs_list.append(torch.softmax(outputs, dim=1).cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    loss_avg = loss_total / max(1, total)
    acc = correct / max(1, total)
    try:
        auc = roc_auc_score(np.concatenate(labels_list), np.concatenate(probs_list), multi_class="ovr")
    except Exception:
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
        inputs = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        macs = profile_macs(model, inputs)
        flops = macs * 2
        return float(flops)
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        return float("nan")

def inference_time_per_batch(model, loader, warmup=WARMUP, timed=TIMING_BATCHES):
    model.eval()
    use_cuda = DEVICE.type == "cuda"
    it = iter(loader)
    try:
        for _ in range(warmup):
            imgs, _ = next(it)
            imgs = imgs.to(DEVICE)
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
            imgs = imgs.to(DEVICE)
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

def collect_metrics_row(variant, stage, ratio, model, test_loader, path_hint):
    loss, acc, auc = evaluate_model_basic(model, test_loader)
    zeros, total = count_zeros_and_total(model) if variant != "slim_kd" else (0, params_count(model))
    params = params_count(model)
    flops = compute_flops(model)
    flops_m = flops / 1e6 if not math.isnan(flops) else float("nan")
    avg_time, peak_ram, images_processed = inference_time_per_batch(model, test_loader, timed=TIMING_BATCHES)
    if path_hint is not None and os.path.exists(path_hint):
        size_mb = os.path.getsize(path_hint)/(1024**2)
    else:
        size_mb = model_size_bytes(model)/(1024**2)
    power_m = (flops * ((total - zeros)/total)) / 1e6 if not math.isnan(flops) and total>0 and variant != "slim_kd" else float("nan")
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
# Freeze / unfreeze & local calibration (pruning only)
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
    for ep in range(epochs):
        running_loss = 0.0; total = 0; correct = 0
        for bidx, (imgs, labels) in enumerate(train_loader, 1):
            if max_batches is not None and bidx > max_batches:
                break
            imgs = imgs.to(device); labels = labels.to(device)
            with torch.no_grad():
                t_logits = teacher(imgs)
            s_logits = student(imgs)
            loss_ce = criterion(s_logits, labels)
            s_log_soft = F.log_softmax(s_logits / T, dim=1)
            with torch.no_grad():
                t_soft = F.softmax(t_logits / T, dim=1)
            loss_kd = kl_loss(s_log_soft, t_soft) * (T * T)
            loss = alpha * loss_ce + (1.0 - alpha) * loss_kd
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += float(loss.item()) * imgs.size(0)
            _, preds = s_logits.max(1)
            total += labels.size(0); correct += int(preds.eq(labels).sum().item())
            if bidx % LOG_INTERVAL == 0:
                print(f"      KD ep{ep+1} batch{bidx} - loss {running_loss/max(1,total):.4f}, acc {correct/max(1,total):.4f}")
    student.eval()
    return student

# -------------------------
# Global finetune
# -------------------------
def global_finetune(model, train_loader, val_loader, epochs=FINAL_FINETUNE_EPOCHS, lr=FINAL_LR):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        running_loss = 0.0; total = 0; correct = 0
        for bidx, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward(); opt.step()
            running_loss += float(loss.item()) * imgs.size(0)
            _, preds = out.max(1)
            total += labels.size(0); correct += int(preds.eq(labels).sum().item())
            if bidx % LOG_INTERVAL == 0:
                print(f"    Global FT ep{ep+1} batch{bidx} - loss {running_loss/max(1,total):.4f}, acc {correct/max(1,total):.4f}")
        vloss, vacc, vauc = evaluate_model_basic(model, val_loader)
        print(f"    Global FT epoch {ep+1}: ValLoss {vloss:.4f}, ValAcc {vacc:.4f}, ValAUC {vauc:.4f}")
    model.eval()
    return model

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
    energy_kwh = float(raw.get("energy_consumed", float("nan"))) if raw is not None and "energy_consumed" in raw else float("nan")
    cpu_power = float(raw.get("cpu_power", float("nan"))) if raw is not None and "cpu_power" in raw else float("nan")
    gpu_power = float(raw.get("gpu_power", float("nan"))) if raw is not None and "gpu_power" in raw else float("nan")
    ram_power = float(raw.get("ram_power", float("nan"))) if raw is not None and "ram_power" in raw else float("nan")
    emissions_kg = float(raw.get("emissions", float("nan"))) if raw is not None and "emissions" in raw else (float(emissions_val) if emissions_val is not None else float("nan"))
    return {
        "emissions_kg": emissions_kg,
        "energy_kwh": energy_kwh,
        "cpu_power_w": cpu_power,
        "gpu_power_w": gpu_power,
        "ram_power_w": ram_power,
        "raw_row": raw
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
    
    # Average
    avg_images = np.mean(images_per_run)
    baseline_energy_kwh = np.mean(energies_total) if len(energies_total) > 0 else float("nan")
    baseline_emissions_kg = np.mean(emissions_per_run) if len(emissions_per_run) > 0 else float("nan")
    baseline_energy_per_pred_kwh = np.mean(energies_per_pred) if len(energies_per_pred) > 0 else float("nan")
    
    print(f"Averaged baseline ({NUM_BASELINE_RUNS} runs): images={avg_images:.0f}, energy_kWh={baseline_energy_kwh}, emissions_kg={baseline_emissions_kg}, per_pred={baseline_energy_per_pred_kwh}")
    return baseline_energy_kwh, baseline_emissions_kg, baseline_energy_per_pred_kwh, float(avg_images)

# -------------------------
# Main pipeline
# -------------------------
for dataset_name, cfg in DATASETS.items():
    try:
        print(f"\n\n===================== DATASET: {dataset_name.upper()} =====================")
        log_memory_usage(f"Before loading {dataset_name}: ")

        SAVE_DIR = f"{SAVE_DIR_BASE}/{dataset_name}"
        os.makedirs(SAVE_DIR, exist_ok=True)

        csv_path = os.path.join(SAVE_DIR, f"{dataset_name}_combined_pruning_kd_metrics_with_energy.csv")
        if os.path.exists(csv_path):
            print(f"Skipping {dataset_name}: CSV already exists at {csv_path}")
            continue

        batch_size = DATASET_BATCH_SIZES.get(dataset_name, BATCH_SIZE_DEFAULT)
        train_loader, val_loader, test_loader, NUM_CLASSES, train_ds = make_loaders(cfg["path"], batch_size)
        print(f"Data loaded for {dataset_name}. NUM_CLASSES={NUM_CLASSES}, device={DEVICE}, batch_size={batch_size}")
        log_memory_usage(f"After loading data for {dataset_name}: ")

        def load_baseline_ckpt(path):
            model = build_resnet50_for_load(NUM_CLASSES)
            if os.path.exists(path):
                state = torch.load(path, map_location="cpu")
                model.load_state_dict(state)
            return model.to(DEVICE).eval()

        baseline = load_baseline_ckpt(cfg["baseline"])
        print("Baseline loaded.")
        log_memory_usage(f"After loading baseline for {dataset_name}: ")

        rows = []

        print("=== EVALUATE BASELINE ===")
        base_ckpt = os.path.join(SAVE_DIR, "baseline.pth")
        torch.save(baseline.state_dict(), base_ckpt)
        row = collect_metrics_row("baseline", "baseline", 0.0, baseline, test_loader, base_ckpt)
        rows.append(row)
        print("Baseline done:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

        # Averaged baseline energy
        baseline_energy_kwh, baseline_emissions_kg, baseline_energy_per_pred_kwh, baseline_images = measure_baseline_energy_averaged(baseline, test_loader, SAVE_DIR, dataset_name)
        print(f"Final averaged baseline inference: images={baseline_images}, energy_kWh={baseline_energy_kwh}, emissions_kg={baseline_emissions_kg}")

        for method in METHODS:
            if method == "slim_kd":
                print(f"\n=== SLIM KD VARIANT ===")
                for target_ratio in TARGET_RATIOS:  # Reuse ratios as keep ratios
                    reduction = 1 - target_ratio
                    print(f"  Reduction: {reduction} (keep ratio {target_ratio})")

                    # Tracker for KD+FT
                    kd_ft_proj = f"{dataset_name}_{method}_r{int(target_ratio*100)}_kd_ft"
                    kd_ft_tracker = start_tracker(SAVE_DIR, kd_ft_proj, measure_power_secs=10) if CODECARBON_AVAILABLE else None

                    # Scaled planes for slim
                    stage_planes = [max(1, int(p * target_ratio)) for p in ORIGINAL_PLANES]
                    current_model = build_pruned_or_slim_resnet(stage_planes=stage_planes, num_classes=NUM_CLASSES, random_init=True)
                    print(f"  Slim student built (random init, planes: {stage_planes}).")

                    # Pre-KD eval
                    pre_kd_ckpt = os.path.join(SAVE_DIR, f"{method}_r{int(target_ratio*100)}_pre_kd.pth")
                    torch.save(current_model.state_dict(), pre_kd_ckpt)
                    row = collect_metrics_row(method, "pre_kd", target_ratio, current_model, test_loader, pre_kd_ckpt)
                    rows.append(row)
                    print("  Pre-KD metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

                    # KD
                    print("  Knowledge distillation...")
                    current_model = distill_student(current_model, baseline, train_loader, epochs=KD_EPOCHS, lr=KD_LR, alpha=KD_ALPHA, T=KD_TEMPERATURE, max_batches=KD_MAX_BATCHES)
                    kd_ckpt = os.path.join(SAVE_DIR, f"{method}_r{int(target_ratio*100)}_afterKD.pth")
                    torch.save(current_model.state_dict(), kd_ckpt)
                    row = collect_metrics_row(method, "after_kd", target_ratio, current_model, test_loader, kd_ckpt)
                    rows.append(row)
                    print("  KD metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

                    # Global finetune
                    print("  Final global finetune...")
                    current_model = global_finetune(current_model, train_loader, val_loader, epochs=FINAL_FINETUNE_EPOCHS, lr=FINAL_LR)
                    final_ckpt = os.path.join(SAVE_DIR, f"{method}_r{int(target_ratio*100)}_final.pth")
                    torch.save(current_model.state_dict(), final_ckpt)
                    row = collect_metrics_row(method, "after_global_finetune", target_ratio, current_model, test_loader, final_ckpt)
                    rows.append(row)
                    print("  Final metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

                    # Stop tracker
                    kd_ft_metrics = stop_tracker_and_get_metrics(kd_ft_tracker, SAVE_DIR, kd_ft_proj)
                    retrain_energy_kwh = kd_ft_metrics["energy_kwh"]
                    retrain_emissions_kg = kd_ft_metrics["emissions_kg"]
                    print(f"  Retrain energy_kWh={retrain_energy_kwh}, emissions_kg={retrain_emissions_kg}")

                    # Pruned inference energy (single run, but consistent with baseline)
                    pruned_inf_proj = f"{dataset_name}_{method}_r{int(target_ratio*100)}_inference"
                    pruned_tracker = start_tracker(SAVE_DIR, pruned_inf_proj, measure_power_secs=10) if CODECARBON_AVAILABLE else None
                    _, _, pruned_images = inference_time_per_batch(current_model, test_loader, timed=TIMING_BATCHES)
                    pruned_inf_metrics = stop_tracker_and_get_metrics(pruned_tracker, SAVE_DIR, pruned_inf_proj)
                    pruned_energy_kwh = pruned_inf_metrics["energy_kwh"]
                    pruned_emissions_kg = pruned_inf_metrics["emissions_kg"]
                    pruned_energy_per_pred_kwh = pruned_energy_kwh / pruned_images if pruned_images > 0 and not math.isnan(pruned_energy_kwh) else float("nan")
                    print(f"  Pruned inference: images={pruned_images}, energy_kWh={pruned_energy_kwh}, emissions_kg={pruned_emissions_kg}")

                    # Break-even
                    if math.isnan(retrain_energy_kwh) or math.isnan(baseline_energy_per_pred_kwh) or math.isnan(pruned_energy_per_pred_kwh):
                        break_even = float("nan")
                    else:
                        delta = baseline_energy_per_pred_kwh - pruned_energy_per_pred_kwh
                        break_even = float("inf") if delta <= 0 else retrain_energy_kwh / delta

                    # Energy summary
                    energy_row = {
                        "Variant": method, "Stage": f"energy_summary_r{int(target_ratio*100)}", "Ratio": target_ratio,
                        "RetrainEnergy_kWh": retrain_energy_kwh, "RetrainEmissions_kg": retrain_emissions_kg,
                        "BaselineInferenceEnergy_kWh_total": baseline_energy_kwh,
                        "BaselineEnergy_per_pred_kWh": baseline_energy_per_pred_kwh,
                        "BaselineEmissions_kg_total": baseline_emissions_kg,
                        "PrunedInferenceEnergy_kWh_total": pruned_energy_kwh,
                        "PrunedEnergy_per_pred_kWh": pruned_energy_per_pred_kwh,
                        "PrunedEmissions_kg_total": pruned_emissions_kg,
                        "BreakEvenPredictions": break_even
                    }
                    rows.append(energy_row)
                    print("  Energy summary:", energy_row)

                    del current_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            else:
                # Pruning methods
                print(f"\n=== PROGRESSIVE PGTO: method={method} ===")
                for target_ratio in TARGET_RATIOS:
                    print(f"  Target keep ratio: {target_ratio}")

                    # Tracker for prune+retrain
                    prune_retrain_proj = f"{dataset_name}_{method}_r{int(target_ratio*100)}_prune_retrain"
                    prune_retrain_tracker = start_tracker(SAVE_DIR, prune_retrain_proj, measure_power_secs=10) if CODECARBON_AVAILABLE else None

                    current_model = copy.deepcopy(baseline).to(DEVICE)
                    keep_indices = {s: np.arange(stage_orig_channels(baseline, s)) for s in STAGES}
                    log_memory_usage(f"Before pruning loop for {method}, ratio={target_ratio}: ")

                    for s in STAGES:
                        orig = stage_orig_channels(current_model, s)
                        keep_k = max(1, int(math.floor(orig * target_ratio)))
                        keeps = compute_stage_importance_and_keeps(current_model, s, keep_k, method=method, calib_loader=train_loader, max_batches=RG_CAL_MAX_BATCHES)
                        keep_indices[s] = keeps
                        print(f"  Stage {s}: keep {len(keeps)}/{orig} ({100*len(keeps)/orig:.1f}% kept)")
                        
                        # Create pruned model with only this stage pruned
                        stage_specific_indices = {k: keep_indices[k] if k==s else np.arange(stage_orig_channels(current_model, k)) for k in STAGES}
                        pruned_model = build_pruned_resnet_and_copy_weights_fixed(current_model, stage_specific_indices, num_classes=NUM_CLASSES)
                        pruned_model = pruned_model.to(DEVICE).eval()
                        
                        # Dummy forward pass to ensure model is valid
                        with torch.no_grad():
                            dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
                            _ = pruned_model(dummy_input)
                        
                        stage_pruned_ckpt = os.path.join(SAVE_DIR, f"pgto_{method}_r{int(target_ratio*100)}_{s}_postprune.pth")
                        torch.save(pruned_model.state_dict(), stage_pruned_ckpt)
                        row = collect_metrics_row(method, f"{s}_postprune", target_ratio, pruned_model, test_loader, stage_pruned_ckpt)
                        rows.append(row)
                        print("    Post-prune metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})
                        
                        print(f"    Calibrating {s} (local)...")
                        pruned_model = calibrate_stage(pruned_model, s, train_loader, epochs=CAL_EPOCHS, max_batches=CAL_MAX_BATCHES, lr=CAL_LR, allow_fc_bn1=False)
                        stage_calib_ckpt = os.path.join(SAVE_DIR, f"pgto_{method}_r{int(target_ratio*100)}_{s}_calibrated.pth")
                        torch.save(pruned_model.state_dict(), stage_calib_ckpt)
                        row = collect_metrics_row(method, f"{s}_postcalib", target_ratio, pruned_model, test_loader, stage_calib_ckpt)
                        rows.append(row)
                        print("    Post-calib metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})
                        
                        current_model = pruned_model
                        log_memory_usage(f"After stage {s} for {method}, ratio={target_ratio}: ")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # All-pruned pre-KD
                    all_pruned_ckpt = os.path.join(SAVE_DIR, f"pgto_{method}_r{int(target_ratio*100)}_allpruned_preKD.pth")
                    torch.save(current_model.state_dict(), all_pruned_ckpt)
                    row = collect_metrics_row(method, "all_pruned_preKD", target_ratio, current_model, test_loader, all_pruned_ckpt)
                    rows.append(row)
                    print("  All-pruned (pre-KD) metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

                    # KD
                    print("  Knowledge distillation...")
                    current_model = distill_student(current_model, baseline, train_loader, epochs=KD_EPOCHS, lr=KD_LR, alpha=KD_ALPHA, T=KD_TEMPERATURE, max_batches=KD_MAX_BATCHES)
                    kd_ckpt = os.path.join(SAVE_DIR, f"pgto_{method}_r{int(target_ratio*100)}_afterKD.pth")
                    torch.save(current_model.state_dict(), kd_ckpt)
                    row = collect_metrics_row(method, "after_kd", target_ratio, current_model, test_loader, kd_ckpt)
                    rows.append(row)
                    print("  KD metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

                    # Global finetune
                    print("  Final global finetune...")
                    current_model = global_finetune(current_model, train_loader, val_loader, epochs=FINAL_FINETUNE_EPOCHS, lr=FINAL_LR)
                    final_ckpt = os.path.join(SAVE_DIR, f"pgto_{method}_r{int(target_ratio*100)}_final.pth")
                    torch.save(current_model.state_dict(), final_ckpt)
                    row = collect_metrics_row(method, "after_global_finetune", target_ratio, current_model, test_loader, final_ckpt)
                    rows.append(row)
                    print("  Final metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

                    # Stop tracker
                    prune_retrain_metrics = stop_tracker_and_get_metrics(prune_retrain_tracker, SAVE_DIR, prune_retrain_proj)
                    retrain_energy_kwh = prune_retrain_metrics["energy_kwh"]
                    retrain_emissions_kg = prune_retrain_metrics["emissions_kg"]
                    print(f"  Prune+retrain energy_kWh={retrain_energy_kwh}, emissions_kg={retrain_emissions_kg}")

                    # Pruned inference energy
                    pruned_inf_proj = f"{dataset_name}_{method}_r{int(target_ratio*100)}_pruned_inference"
                    pruned_tracker = start_tracker(SAVE_DIR, pruned_inf_proj, measure_power_secs=10) if CODECARBON_AVAILABLE else None
                    _, _, pruned_images = inference_time_per_batch(current_model, test_loader, timed=TIMING_BATCHES)
                    pruned_inf_metrics = stop_tracker_and_get_metrics(pruned_tracker, SAVE_DIR, pruned_inf_proj)
                    pruned_energy_kwh = pruned_inf_metrics["energy_kwh"]
                    pruned_emissions_kg = pruned_inf_metrics["emissions_kg"]
                    pruned_energy_per_pred_kwh = pruned_energy_kwh / pruned_images if pruned_images>0 and not math.isnan(pruned_energy_kwh) else float("nan")
                    print(f"  Pruned inference: images={pruned_images}, energy_kWh={pruned_energy_kwh}, emissions_kg={pruned_emissions_kg}")

                    # Break-even
                    if math.isnan(retrain_energy_kwh) or math.isnan(baseline_energy_per_pred_kwh) or math.isnan(pruned_energy_per_pred_kwh):
                        break_even = float("nan")
                    else:
                        delta = baseline_energy_per_pred_kwh - pruned_energy_per_pred_kwh
                        break_even = float("inf") if delta <= 0 else retrain_energy_kwh / delta

                    # Energy summary
                    energy_row = {
                        "Variant": method, "Stage": f"energy_summary_r{int(target_ratio*100)}", "Ratio": target_ratio,
                        "RetrainEnergy_kWh": retrain_energy_kwh, "RetrainEmissions_kg": retrain_emissions_kg,
                        "BaselineInferenceEnergy_kWh_total": baseline_energy_kwh,
                        "BaselineEnergy_per_pred_kWh": baseline_energy_per_pred_kwh,
                        "BaselineEmissions_kg_total": baseline_emissions_kg,
                        "PrunedInferenceEnergy_kWh_total": pruned_energy_kwh,
                        "PrunedEnergy_per_pred_kWh": pruned_energy_per_pred_kwh,
                        "PrunedEmissions_kg_total": pruned_emissions_kg,
                        "BreakEvenPredictions": break_even
                    }
                    rows.append(energy_row)
                    print("  Energy summary:", energy_row)

                    del current_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Save CSV
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"All done for {dataset_name}. CSV: {csv_path}")
        del baseline, train_loader, val_loader, test_loader, train_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_memory_usage(f"After completing {dataset_name}: ")
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"Continuing to next dataset...")

print("All datasets processed.")