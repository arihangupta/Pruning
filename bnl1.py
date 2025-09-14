#!/usr/bin/env python3
"""
Progressive PGTO pruning for multiple MedMNIST datasets.

- Supports importance methods: regional_gradients, l1, bn_gamma
- Surgery to build pruned ResNet-50 and copy weights
- KD (knowledge distillation) + global finetune
- Metrics: Acc, AUC, Loss, Params, FLOPs, Inference time, Peak RAM, Model size
- CodeCarbon energy tracking (if installed). Writes emissions.csv in SAVE_DIR per dataset.
- Outputs CSV per dataset with rows for baseline, per-stage postprune/postcalib, after KD, final, and energy summary rows.

Usage: python3 pgto_prune_all_datasets_with_bn_l1.py
"""
import os
import time
import math
import random
import tempfile
import copy
import traceback
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
from torchvision.models.resnet import Bottleneck
from sklearn.metrics import roc_auc_score

# optional macs profiler
try:
    from torchprofile import profile_macs
    HAS_PROFILE_MACS = True
except Exception:
    HAS_PROFILE_MACS = False

# CodeCarbon optional
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except Exception:
    CODECARBON_AVAILABLE = False
    EmissionsTracker = None
    print("Warning: codecarbon not available. Energy/emissions will be NaN.")

# -------------------------
# Config (edit paths / hyperparams here)
# -------------------------
SAVE_DIR_BASE = "/home/arihangupta/Pruning/dinov2/Pruning/experiment2_pgto_bn_l1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224

# pruning targets (fraction kept = 1 - ratio)
TARGET_RATIOS = [0.5, 0.6, 0.7]  # fraction pruned per stage
METHODS = ["regional_gradients", "l1", "bn_gamma"]

# calibration / kd / finetune
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

LOG_INTERVAL = 50
WARMUP = 5
TIMING_BATCHES = 30
RG_CAL_MAX_BATCHES = 50  # used by regional gradients

# batch sizes per dataset (tune to your GPU/RAM)
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

# -------------------------
# Repro
# -------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# -------------------------
# Data loader (memory mapped friendly)
# -------------------------
class NumpyMemmapDataset(Dataset):
    def __init__(self, imgs_np, labels_np, img_size=IMG_SIZE):
        self.imgs = imgs_np
        self.labels = labels_np
        self.img_size = img_size
        self.to_pil = T.ToPILImage()
        self.tfms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])
        # normalize to typical ImageNet norm
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]  # shape H,W,C or H,W (grayscale)
        label = int(self.labels[idx])
        # ensure HWC
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        # If single channel, repeat to 3 for transform; later model conv1 will be adapted if needed
        if img.shape[2] == 1:
            img3 = np.repeat(img, 3, axis=2)
        else:
            img3 = img
        x = self.to_pil(img3.astype(np.uint8))
        x = self.tfms(x)
        x = self.norm(x)
        return x, label

def make_loaders(npz_path, batch_size):
    data = np.load(npz_path, mmap_mode="r")
    X_train, y_train = data["train_images"], data["train_labels"].flatten()
    X_val, y_val = data["val_images"], data["val_labels"].flatten()
    X_test, y_test = data["test_images"], data["test_labels"].flatten()
    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    print(f"Dataset sizes: train={n_train}, val={n_val}, test={n_test}")
    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE)
    val_ds = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE)
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    num_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    return train_loader, val_loader, test_loader, num_classes, train_ds

# -------------------------
# model builders
# -------------------------
class CustomResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3,4,6,3], stage_planes=[64,128,256,512], num_classes=1000):
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

def build_resnet50_for_load(num_classes, in_channels=3):
    model = models.resnet50(weights=None)
    if in_channels == 1:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(1, old_conv.out_channels,
                                kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride,
                                padding=old_conv.padding,
                                bias=old_conv.bias)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -------------------------
# helpers: metrics / flops / time / model size
# -------------------------
criterion = nn.CrossEntropyLoss()

def stage_orig_channels(model, stage):
    stage_dict = {
        'layer1': model.layer1[0].conv1.out_channels,
        'layer2': model.layer2[0].conv1.out_channels,
        'layer3': model.layer3[0].conv1.out_channels,
        'layer4': model.layer4[0].conv1.out_channels
    }
    return stage_dict[stage]

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
            total += labels.size(0)
            correct += int(predicted.eq(labels).sum().item())
            probs_list.append(torch.softmax(outputs, dim=1).cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    loss_avg = loss_total / max(1, total)
    acc = correct / max(1, total)
    try:
        probs_all = np.concatenate(probs_list, axis=0)
        labels_all = np.concatenate(labels_list, axis=0)
        auc = roc_auc_score(labels_all, probs_all, multi_class="ovr")
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
    fd, tmp = tempfile.mkstemp(suffix=".pth")
    os.close(fd)
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp)
    os.remove(tmp)
    return size

def compute_flops(model):
    model.eval()
    try:
        if HAS_PROFILE_MACS:
            inputs = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            macs = profile_macs(model, inputs)
            flops = macs * 2
            return float(flops)
        else:
            # a rough fallback: run a forward with hooks to sum conv/linear ops
            flops = 0
            hooks = []
            def conv_hook(self, inp, out):
                nonlocal flops
                in_t = inp[0]
                kh, kw = self.kernel_size
                C_in = in_t.shape[1]
                C_out = out.shape[1]
                H_out = out.shape[2]; W_out = out.shape[3]
                flops += kh * kw * C_in * C_out * H_out * W_out
            def linear_hook(self, inp, out):
                nonlocal flops
                in_features = inp[0].shape[1]; out_features = out.shape[1]
                flops += in_features * out_features
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    hooks.append(m.register_forward_hook(conv_hook))
                if isinstance(m, nn.Linear):
                    hooks.append(m.register_forward_hook(linear_hook))
            device = next(model.parameters()).device
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
            with torch.no_grad():
                _ = model(dummy)
            for h in hooks: h.remove()
            return float(flops)
    except Exception as e:
        print("FLOPs estimation failed:", e)
        return float("nan")

def inference_time_and_peak_ram(model, loader, warmup=WARMUP, timed=TIMING_BATCHES):
    model.eval()
    use_cuda = DEVICE.type == "cuda"
    it = iter(loader)
    try:
        for _ in range(warmup):
            imgs, _ = next(it)
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                _ = model(imgs)
            if use_cuda: torch.cuda.synchronize()
    except StopIteration:
        pass
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    batches_done = 0
    images_processed = 0
    try:
        for _ in range(timed):
            imgs, _ = next(it)
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                _ = model(imgs)
            if use_cuda: torch.cuda.synchronize()
            batches_done += 1
            images_processed += imgs.size(0)
    except StopIteration:
        pass
    elapsed = time.time() - start
    avg_batch = elapsed / max(1, batches_done)
    if use_cuda:
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_mb = peak_bytes / (1024**2)
    else:
        peak_mb = params_count(model) * 4.0 / (1024**2)
    return avg_batch, peak_mb, batches_done, images_processed

def collect_metrics_row(tag_variant, tag_stage, ratio, model, test_loader, path_hint):
    loss, acc, auc = evaluate_model_basic(model, test_loader)
    zeros, total = count_zeros_and_total(model)
    params = params_count(model)
    flops = compute_flops(model)
    flops_m = flops / 1e6 if not math.isnan(flops) else float("nan")
    avg_time, peak_ram, _, images_processed = inference_time_and_peak_ram(model, test_loader)
    if path_hint is not None and os.path.exists(path_hint):
        size_mb = os.path.getsize(path_hint)/(1024**2)
    else:
        size_mb = model_size_bytes(model)/(1024**2)
    power_m = (flops * ((total - zeros)/total)) / 1e6 if not math.isnan(flops) and total>0 else float("nan")
    return {
        "Variant": tag_variant, "Stage": tag_stage, "Ratio": ratio,
        "Acc": acc, "AUC": auc, "Loss": loss,
        "Params": params, "Zeros": zeros, "TotalParams": total, "PctZeros": (zeros/total)*100 if total>0 else 0,
        "ModelSizeMB": size_mb, "FLOPs_per_image": flops, "FLOPs_M_per_image": flops_m,
        "InferenceTime_per_batch_s": avg_time, "PeakRAM_MB": peak_ram,
        "PowerProxy_MFLOPs": power_m, "ModelPath": path_hint,
        "ImagesProcessedDuringTiming": images_processed
    }

# -------------------------
# Importance scoring methods
# -------------------------
def compute_stage_importance_and_keeps_l1(model: nn.Module, stage_name: str, keep_k: int):
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    orig_planes = first_block.conv1.out_channels
    expansion = 4
    weight_l1 = np.zeros(orig_planes, dtype=float)
    for block in stage.children():
        w = block.conv3.weight.detach().cpu().numpy()  # shape (C_out, C_in, k, k)
        # partition by output groups (expanded)
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
    # use bn3 gamma per expanded outputs averaged per plane
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

def compute_stage_importance_and_keeps_regional(model: nn.Module, stage_name: str, keep_k: int, calib_loader: DataLoader, max_batches=RG_CAL_MAX_BATCHES):
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    orig_planes = first_block.conv1.out_channels
    expansion = 4
    device = DEVICE
    act_norms = torch.zeros(orig_planes, device=device)
    grad_norms = torch.zeros(orig_planes, device=device)
    weight_l1 = torch.zeros(orig_planes, device=device)

    # weight l1 proxy
    for block in stage.children():
        w = block.conv3.weight.detach().abs().cpu().numpy()
        for p in range(orig_planes):
            weight_l1[p] += np.sum(w[p*expansion:(p+1)*expansion])
    weight_l1 = weight_l1.to(device)

    # We'll hook the stage output activation to form a regional gradient proxy
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
        act = saved['act']  # shape (B, Cexp, H, W)
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
            if block.conv3.weight.grad is None:
                continue
            g_abs = block.conv3.weight.grad.detach().abs()
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
        act_norms /= max(1, batch_count)
        grad_norms /= max(1, batch_count)
        agg = (act_norms * grad_norms * weight_l1).cpu().numpy()
    if keep_k >= len(agg):
        keep = np.arange(len(agg))
    else:
        keep = np.argsort(agg)[-keep_k:]
    return np.sort(keep)

def compute_stage_importance_and_keeps(model: nn.Module, stage_name: str, keep_k: int,
                                      method="regional_gradients", calib_loader=None, max_batches=RG_CAL_MAX_BATCHES):
    if method == "l1":
        return compute_stage_importance_and_keeps_l1(model, stage_name, keep_k)
    elif method == "bn_gamma":
        return compute_stage_importance_and_keeps_bn_gamma(model, stage_name, keep_k)
    elif method == "regional_gradients":
        assert calib_loader is not None, "calib_loader required for regional_gradients"
        return compute_stage_importance_and_keeps_regional(model, stage_name, keep_k, calib_loader, max_batches=max_batches)
    else:
        raise ValueError(f"Unknown method {method}")

# -------------------------
# Surgery: build pruned network and copy weights
# -------------------------
def build_pruned_resnet(keep_indices, num_classes):
    stage_planes = [
        len(keep_indices['layer1']),
        len(keep_indices['layer2']),
        len(keep_indices['layer3']),
        len(keep_indices['layer4'])
    ]
    return CustomResNet(block=Bottleneck, layers=[3,4,6,3], stage_planes=stage_planes, num_classes=num_classes)

def build_pruned_resnet_and_copy_weights_fixed(base_model: nn.Module, keep_indices: Dict[str, np.ndarray], num_classes: int):
    """
    Copy weights from base_model into a smaller CustomResNet according to keep_indices dict.
    This is adapted from earlier surgery code; handles conv1, conv2, conv3 and downsample.
    """
    expansion = 4
    new_model = build_pruned_resnet(keep_indices, num_classes).to(DEVICE)
    new_model.eval()
    base_model = base_model.to(DEVICE)
    prev_out_idx = torch.arange(base_model.conv1.out_channels, dtype=torch.long, device=DEVICE)

    for stage_name in ["layer1", "layer2", "layer3", "layer4"]:
        old_stage = getattr(base_model, stage_name)
        new_stage = getattr(new_model, stage_name)
        kept_planes = torch.tensor(keep_indices[stage_name], dtype=torch.long, device=DEVICE)
        # For each block pair
        for block_idx, (old_block, new_block) in enumerate(zip(old_stage, new_stage)):
            in_idx = prev_out_idx
            out_planes = kept_planes
            # expanded rows for conv3 outputs
            expanded_rows = torch.cat([ (k * expansion + torch.arange(expansion, device=DEVICE)) for k in out_planes ]) if len(out_planes)>0 else torch.tensor([], dtype=torch.long, device=DEVICE)

            # conv1: take out_channels selected, input channels from in_idx
            old_w1 = old_block.conv1.weight.data
            new_block.conv1.weight.data.copy_(old_w1[out_planes][:, in_idx, :, :])
            if getattr(old_block.conv1, 'bias', None) is not None:
                new_block.conv1.bias.data.copy_(old_block.conv1.bias.data[out_planes])

            # bn1
            copy_bn_params(new_block.bn1, old_block.bn1, out_planes)

            # conv2
            old_w2 = old_block.conv2.weight.data
            new_block.conv2.weight.data.copy_(old_w2[out_planes][:, out_planes, :, :])
            if getattr(old_block.conv2, 'bias', None) is not None:
                new_block.conv2.bias.data.copy_(old_block.conv2.bias.data[out_planes])
            # bn2
            copy_bn_params(new_block.bn2, old_block.bn2, out_planes)

            # conv3: rows correspond to expanded outputs (C_out original * expansion)
            old_w3 = old_block.conv3.weight.data
            if expanded_rows.numel() > 0:
                new_block.conv3.weight.data.copy_(old_w3[expanded_rows][:, out_planes, :, :])
                if getattr(old_block.conv3, 'bias', None) is not None:
                    new_block.conv3.bias.data.copy_(old_block.conv3.bias.data[expanded_rows])
                # bn3 uses expanded rows
                copy_bn_params(new_block.bn3, old_block.bn3, expanded_rows)
            else:
                # no channels kept (shouldn't happen)
                pass

            # downsample if present
            if old_block.downsample is not None:
                ds_conv_old = old_block.downsample[0].weight.data
                ds_bn_old = old_block.downsample[1]
                # For downsample conv: rows = expanded_rows (output), columns = in_idx
                new_block.downsample[0].weight.data.copy_(ds_conv_old[expanded_rows][:, in_idx, :, :])
                copy_bn_params(new_block.downsample[1], ds_bn_old, expanded_rows)

            prev_out_idx = expanded_rows  # next block input mapping

    # fc: map columns corresponding to last expanded_rows
    last_kept = torch.tensor(keep_indices['layer4'], dtype=torch.long, device=DEVICE)
    fc_in_idx = torch.cat([torch.arange(p * expansion, (p + 1) * expansion, dtype=torch.long, device=DEVICE) for p in last_kept]) if last_kept.numel()>0 else torch.tensor([], dtype=torch.long, device=DEVICE)
    if fc_in_idx.numel() > 0:
        new_model.fc.weight.data.copy_(base_model.fc.weight.data[:, fc_in_idx])
    new_model.fc.bias.data.copy_(base_model.fc.bias.data)
    return new_model

# -------------------------
# freeze / calibrate
# -------------------------
def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_stage(model, stage_name, allow_fc_bn1=False):
    for name, p in model.named_parameters():
        if name.startswith(stage_name):
            p.requires_grad = True
        if allow_fc_bn1 and (name.startswith("fc.") or name.startswith("bn1.")):
            p.requires_grad = True

def calibrate_stage(model, stage_name, train_loader, epochs=CAL_EPOCHS, max_batches=CAL_MAX_BATCHES, lr=CAL_LR, allow_fc_bn1=False):
    freeze_all(model)
    unfreeze_stage(model, stage_name, allow_fc_bn1=allow_fc_bn1)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    steps = 0
    for ep in range(epochs):
        running_loss = 0.0; total = 0; correct = 0
        for bidx, (imgs, labels) in enumerate(train_loader, 1):
            if bidx > max_batches:
                break
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            running_loss += float(loss.item()) * imgs.size(0)
            _, preds = out.max(1)
            total += labels.size(0); correct += int(preds.eq(labels).sum().item())
            steps += 1
            if bidx % LOG_INTERVAL == 0:
                print(f"      Calib {stage_name} ep{ep+1} batch{bidx} - loss {running_loss/max(1,total):.4f}, acc {correct/max(1,total):.4f}")
    return model

# -------------------------
# KD and finetune
# -------------------------
def distill_student(student: nn.Module, teacher: nn.Module, train_loader: DataLoader,
                    epochs: int=KD_EPOCHS, lr: float=KD_LR, alpha: float=KD_ALPHA, T: float=KD_TEMPERATURE,
                    max_batches: int=None):
    teacher.eval()
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    device = DEVICE
    for ep in range(epochs):
        running_loss = 0.0; total = 0; correct = 0
        for bidx, (imgs, labels) in enumerate(train_loader, 1):
            if max_batches is not None and bidx > max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
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

def global_finetune(model, train_loader, val_loader, epochs=FINAL_FINETUNE_EPOCHS, lr=FINAL_LR):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        running_loss = 0.0; total = 0; correct = 0
        for bidx, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
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
    model.eval()
    return model

# -------------------------
# CodeCarbon helpers
# -------------------------
def start_tracker(save_dir: str, project_name: str, output_file: str="emissions.csv", measure_power_secs: int=15):
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
        return df_match.iloc[-1].to_dict()
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
# Main pipeline over datasets
# -------------------------
def main():
    set_seed(SEED)
    for dataset_name, cfg in DATASETS.items():
        try:
            print("\n" + "="*80)
            print(f"DATASET: {dataset_name}")
            save_dir = os.path.join(SAVE_DIR_BASE, dataset_name)
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, f"{dataset_name}_pgto_pruning_metrics_with_energy.csv")
            if os.path.exists(csv_path):
                print(f"CSV already exists at {csv_path} -- skipping dataset.")
                continue

            batch_size = DATASET_BATCH_SIZES.get(dataset_name, 32)
            train_loader, val_loader, test_loader, NUM_CLASSES, train_ds = make_loaders(cfg["path"], batch_size)
            print(f"Loaded {dataset_name}: NUM_CLASSES={NUM_CLASSES}, batch_size={batch_size}")

            # load baseline (if provided)
            def load_baseline_ckpt(path):
                model = build_resnet50_for_load(NUM_CLASSES)
                if os.path.exists(path):
                    state = torch.load(path, map_location="cpu")
                    try:
                        model.load_state_dict(state)
                    except Exception as e:
                        print("Warning loading baseline state_dict:", e)
                return model.to(DEVICE).eval()

            baseline = load_baseline_ckpt(cfg.get("baseline", ""))
            print("Baseline ready.")

            rows = []

            # baseline metrics & save
            base_ckpt = os.path.join(save_dir, "baseline.pth")
            torch.save(baseline.state_dict(), base_ckpt)
            row = collect_metrics_row("baseline", "baseline", 0.0, baseline, test_loader, base_ckpt)
            rows.append(row)
            print("Baseline metric row appended:", {k: row[k] for k in ["Acc","AUC","ModelSizeMB","FLOPs_M_per_image"]})

            # baseline inference energy measurement
            baseline_proj = f"{dataset_name}_baseline_inference"
            baseline_tracker = start_tracker(save_dir, baseline_proj, measure_power_secs=15) if CODECARBON_AVAILABLE else None
            baseline_images = 0
            try:
                it = iter(test_loader)
                for _ in range(WARMUP):
                    imgs, _ = next(it)
                    imgs = imgs.to(DEVICE)
                    with torch.no_grad(): _ = baseline(imgs)
                    if DEVICE.type == "cuda": torch.cuda.synchronize()
                for _ in range(TIMING_BATCHES):
                    imgs, _ = next(it)
                    imgs = imgs.to(DEVICE)
                    with torch.no_grad(): _ = baseline(imgs)
                    if DEVICE.type == "cuda": torch.cuda.synchronize()
                    baseline_images += imgs.size(0)
            except StopIteration:
                pass
            baseline_res = stop_tracker_and_get_metrics(baseline_tracker, save_dir, baseline_proj)
            baseline_energy_kwh = baseline_res["energy_kwh"]
            baseline_emissions_kg = baseline_res["emissions_kg"]
            baseline_energy_per_pred_kwh = baseline_energy_kwh / baseline_images if baseline_images>0 and not math.isnan(baseline_energy_kwh) else float("nan")
            print(f"Baseline inference: images={baseline_images}, energy_kWh={baseline_energy_kwh}")

            # STAGES
            STAGES = ["layer1","layer2","layer3","layer4"]

            # per-method loops
            for method in METHODS:
                for ratio in TARGET_RATIOS:
                    print("\n" + "-"*60)
                    print(f"METHOD={method}  RATIO={ratio}")

                    prune_retrain_proj = f"{dataset_name}_{method}_r{int(ratio*100)}_prune_retrain"
                    prune_retrain_tracker = start_tracker(save_dir, prune_retrain_proj, measure_power_secs=15) if CODECARBON_AVAILABLE else None

                    current_model = copy.deepcopy(baseline).to(DEVICE)
                    keep_indices = {s: np.arange(stage_orig_channels(baseline, s)) for s in STAGES}

                    # progressive stage pruning
                    for s in STAGES:
                        orig = stage_orig_channels(current_model, s)
                        keep_k = max(1, int(math.floor(orig * (1.0 - ratio))))
                        if method == "regional_gradients":
                            keeps = compute_stage_importance_and_keeps(current_model, s, keep_k, method=method, calib_loader=train_loader, max_batches=RG_CAL_MAX_BATCHES)
                        else:
                            keeps = compute_stage_importance_and_keeps(current_model, s, keep_k, method=method)
                        keep_indices[s] = keeps
                        print(f" Stage {s}: keep {len(keeps)}/{orig}")

                        # build pruned model where we only prune stage s (others keep original indices)
                        partial_keep = {k: (keep_indices[k] if k==s else np.arange(stage_orig_channels(current_model,k))) for k in STAGES}
                        pruned_model = build_pruned_resnet_and_copy_weights_fixed(current_model, partial_keep, NUM_CLASSES).to(DEVICE).eval()
                        # quick forward check
                        try:
                            with torch.no_grad():
                                _ = pruned_model(torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE))
                            print("  dummy forward OK")
                        except Exception as e:
                            print("  dummy forward failed:", e)
                        # save postprune
                        stage_pruned_ckpt = os.path.join(save_dir, f"pgto_{method}_r{int(ratio*100)}_{s}_postprune.pth")
                        torch.save(pruned_model.state_dict(), stage_pruned_ckpt)
                        row = collect_metrics_row(method, f"{s}_postprune", ratio, pruned_model, test_loader, stage_pruned_ckpt)
                        rows.append(row)
                        print("  post-prune metrics appended")

                        # local calibration
                        print(f"  calibrating {s} (local)")
                        pruned_model = calibrate_stage(pruned_model, s, train_loader, epochs=CAL_EPOCHS, max_batches=CAL_MAX_BATCHES, lr=CAL_LR, allow_fc_bn1=False)
                        stage_calib_ckpt = os.path.join(save_dir, f"pgto_{method}_r{int(ratio*100)}_{s}_calibrated.pth")
                        torch.save(pruned_model.state_dict(), stage_calib_ckpt)
                        row = collect_metrics_row(method, f"{s}_postcalib", ratio, pruned_model, test_loader, stage_calib_ckpt)
                        rows.append(row)
                        print("  post-calib metrics appended")
                        current_model = pruned_model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # all-pruned pre KD
                    all_pruned_ckpt = os.path.join(save_dir, f"pgto_{method}_r{int(ratio*100)}_allpruned_preKD.pth")
                    torch.save(current_model.state_dict(), all_pruned_ckpt)
                    row = collect_metrics_row(method, "all_pruned_preKD", ratio, current_model, test_loader, all_pruned_ckpt)
                    rows.append(row)
                    print("All-pruned preKD metrics appended")

                    # Knowledge distillation
                    print(" Performing KD...")
                    current_model = distill_student(current_model, baseline, train_loader, epochs=KD_EPOCHS, lr=KD_LR, alpha=KD_ALPHA, T=KD_TEMPERATURE, max_batches=KD_MAX_BATCHES)
                    kd_ckpt = os.path.join(save_dir, f"pgto_{method}_r{int(ratio*100)}_afterKD.pth")
                    torch.save(current_model.state_dict(), kd_ckpt)
                    row = collect_metrics_row(method, "after_kd", ratio, current_model, test_loader, kd_ckpt)
                    rows.append(row)
                    print("KD metrics appended")

                    # Global finetune
                    print(" Final global finetune...")
                    current_model = global_finetune(current_model, train_loader, val_loader, epochs=FINAL_FINETUNE_EPOCHS, lr=FINAL_LR)
                    final_ckpt = os.path.join(save_dir, f"pgto_{method}_r{int(ratio*100)}_final.pth")
                    torch.save(current_model.state_dict(), final_ckpt)
                    row = collect_metrics_row(method, "after_global_finetune", ratio, current_model, test_loader, final_ckpt)
                    rows.append(row)
                    print(" Final metrics appended")

                    # stop prune+retrain tracker
                    prune_retrain_metrics = stop_tracker_and_get_metrics(prune_retrain_tracker, save_dir, prune_retrain_proj)
                    retrain_energy_kwh = prune_retrain_metrics["energy_kwh"]
                    retrain_emissions_kg = prune_retrain_metrics["emissions_kg"]
                    print(f"Prune+retrain energy (kWh) = {retrain_energy_kwh}")

                    # measure pruned model inference energy
                    pruned_inf_proj = f"{dataset_name}_{method}_r{int(ratio*100)}_pruned_inference"
                    pruned_tracker = start_tracker(save_dir, pruned_inf_proj, measure_power_secs=15) if CODECARBON_AVAILABLE else None
                    pruned_images = 0
                    try:
                        it = iter(test_loader)
                        for _ in range(WARMUP):
                            imgs, _ = next(it)
                            imgs = imgs.to(DEVICE)
                            with torch.no_grad(): _ = current_model(imgs)
                            if DEVICE.type == "cuda": torch.cuda.synchronize()
                        for _ in range(TIMING_BATCHES):
                            imgs, _ = next(it)
                            imgs = imgs.to(DEVICE)
                            with torch.no_grad(): _ = current_model(imgs)
                            if DEVICE.type == "cuda": torch.cuda.synchronize()
                            pruned_images += imgs.size(0)
                    except StopIteration:
                        pass
                    pruned_inf_metrics = stop_tracker_and_get_metrics(pruned_tracker, save_dir, pruned_inf_proj)
                    pruned_energy_kwh = pruned_inf_metrics["energy_kwh"]
                    pruned_emissions_kg = pruned_inf_metrics["emissions_kg"]
                    pruned_energy_per_pred_kwh = pruned_energy_kwh / pruned_images if pruned_images>0 and not math.isnan(pruned_energy_kwh) else float("nan")
                    print(f"Pruned inference energy kWh = {pruned_energy_kwh}")

                    # compute break-even predictions
                    if math.isnan(retrain_energy_kwh) or math.isnan(baseline_energy_per_pred_kwh) or math.isnan(pruned_energy_per_pred_kwh):
                        break_even = float("nan")
                    else:
                        delta = baseline_energy_per_pred_kwh - pruned_energy_per_pred_kwh
                        if delta <= 0:
                            break_even = float("inf")
                        else:
                            break_even = retrain_energy_kwh / delta

                    energy_row = {
                        "Variant": method,
                        "Stage": f"energy_summary_r{int(ratio*100)}",
                        "Ratio": ratio,
                        "RetrainEnergy_kWh": retrain_energy_kwh,
                        "RetrainEmissions_kg": retrain_emissions_kg,
                        "BaselineInferenceEnergy_kWh_total": baseline_energy_kwh,
                        "BaselineEnergy_per_pred_kWh": baseline_energy_per_pred_kwh,
                        "BaselineEmissions_kg_total": baseline_emissions_kg,
                        "PrunedInferenceEnergy_kWh_total": pruned_energy_kwh,
                        "PrunedEnergy_per_pred_kWh": pruned_energy_per_pred_kwh,
                        "PrunedEmissions_kg_total": pruned_emissions_kg,
                        "BreakEvenPredictions": break_even
                    }
                    rows.append(energy_row)
                    print("Energy summary appended.")

                    # free memory
                    del current_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # write CSV
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV for {dataset_name}: {csv_path}")

            # cleanup for this dataset
            del baseline, train_loader, val_loader, test_loader, train_ds, rows
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR processing {dataset_name}: {e}")
            traceback.print_exc()
            print("Skipping to next dataset...")
            continue

if __name__ == "__main__":
    main()
