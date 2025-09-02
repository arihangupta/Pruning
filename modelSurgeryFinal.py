#!/usr/bin/env python3
"""
Dermamnist ResNet50 model-surgery pruning (structured channel removal) with metrics.

Saves:
 - pruned model before/after finetune (.pth)
 - CSV summary with metrics for baseline and all pruned variants.

Requirements:
 - torch, torchvision
 - fvcore (optional, for FLOPs). If missing, script falls back to hook-based FLOP estimate.
 - sklearn (for AUC)
"""
import os
import time
import math
import random
import copy
import csv
import numpy as np
import pandas as pd
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.resnet import Bottleneck
from torchvision import models

from sklearn.metrics import roc_auc_score

# Try fvcore for accurate FLOPs
HAS_FVCORE = True
try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    HAS_FVCORE = False

# -------------------------
# Config
# -------------------------
DERMA_PATH = "/home/arihangupta/Pruning/dinov2/Pruning/datasets/dermamnist_224.npz"
SAVE_DIR   = "/home/arihangupta/Pruning/dinov2/Pruning/saved_models"
BASELINE_CKPT = os.path.join(SAVE_DIR, "dermamnist_resnet50_BASELINE.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 32
FINETUNE_EPOCHS = 3
LEARNING_RATE = 1e-4
IMG_SIZE = 224
RATIOS = [0.1, 0.3, 0.5]   # per-stage prune fractions
LOG_INTERVAL = 20
WARMUP = 5
TIMING_BATCHES = 30

# -------------------------
# Reproducibility
# -------------------------
def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# -------------------------
# Data helpers
# -------------------------
def preprocess(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32") / 255.0
    if arr.ndim == 3 or arr.shape[-1] == 1:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    return np.transpose(arr, (0, 3, 1, 2))  # NHWC -> NCHW

def make_loaders(npz_path):
    data = np.load(npz_path)
    X_train, y_train = data["train_images"], data["train_labels"].flatten()
    X_val, y_val     = data["val_images"], data["val_labels"].flatten()
    X_test, y_test   = data["test_images"], data["test_labels"].flatten()
    X_train = preprocess(X_train); X_val = preprocess(X_val); X_test = preprocess(X_test)
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader, int(len(np.unique(y_train)))

train_loader, val_loader, test_loader, NUM_CLASSES = make_loaders(DERMA_PATH)
print(f"Data loaded. NUM_CLASSES={NUM_CLASSES}, device={DEVICE}")

# -------------------------
# CustomResNet builder (smaller stage widths)
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
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

# -------------------------
# Baseline load
# -------------------------
def build_resnet50_for_load(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_baseline_ckpt(path=BASELINE_CKPT):
    model = build_resnet50_for_load(NUM_CLASSES)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model

baseline = load_baseline_ckpt()
print("Baseline loaded.")

# -------------------------
# Importance & keep computation
# -------------------------
def compute_stage_importance_and_keeps(model: nn.Module, stage_name: str, keep_k: int, method: str="l1"):
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    orig_planes = first_block.conv1.out_channels
    block_importances = []
    for block in stage.children():
        if method == "bn_gamma":
            bn3 = block.bn3
            gammas = bn3.weight.detach().abs().cpu().numpy()
            exp = block.conv3.out_channels // block.conv3.in_channels
            per_plane = []
            for p in range(orig_planes):
                vals = gammas[p*exp:(p+1)*exp]
                per_plane.append(np.mean(vals))
            block_importances.append(np.array(per_plane))
        else:
            conv3 = block.conv3.weight.detach().abs().cpu().numpy()
            exp = block.conv3.out_channels // block.conv3.in_channels
            per_plane = []
            for p in range(orig_planes):
                group = conv3[p*exp:(p+1)*exp, :, :, :]
                per_plane.append(np.sum(np.abs(group)))
            block_importances.append(np.array(per_plane))
    agg = np.mean(np.stack(block_importances, axis=0), axis=0)
    if keep_k >= len(agg):
        keep = np.arange(len(agg))
    else:
        keep = np.argsort(agg)[-keep_k:]
    keep = np.sort(keep)
    return keep

# -------------------------
# Build pruned model and copy weights (surgery)
# -------------------------
def build_pruned_resnet_and_copy_weights(old_model: nn.Module, keep_indices_per_stage: Dict[str, np.ndarray], num_classes: int):
    # orig planes
    STAGE_NAMES = ["layer1", "layer2", "layer3", "layer4"]
    orig_stage_planes = []
    for s in STAGE_NAMES:
        first_block = next(getattr(old_model, s).children())
        orig_stage_planes.append(first_block.conv1.out_channels)
    new_stage_planes = [len(keep_indices_per_stage[s]) for s in STAGE_NAMES]
    layers_cfg = [len(list(getattr(old_model, s).children())) for s in STAGE_NAMES]

    new_model = CustomResNet(block=Bottleneck, layers=layers_cfg, stage_planes=new_stage_planes, num_classes=num_classes)
    new_model = new_model.to(DEVICE)
    new_model.eval()

    def expanded_indices(plane_indices, expansion=4):
        out = []
        for p in plane_indices:
            for r in range(expansion):
                out.append(int(p*expansion + r))
        return out

    # copy stem conv1 & bn1
    with torch.no_grad():
        new_model.conv1.weight.copy_(old_model.conv1.weight)
        new_model.bn1.weight.copy_(old_model.bn1.weight)
        new_model.bn1.bias.copy_(old_model.bn1.bias)
        new_model.bn1.running_mean.copy_(old_model.bn1.running_mean)
        new_model.bn1.running_var.copy_(old_model.bn1.running_var)

    prev_expanded_idxs = None

    for s_idx, s in enumerate(STAGE_NAMES):
        old_stage = getattr(old_model, s)
        new_stage = getattr(new_model, s)
        keep = list(keep_indices_per_stage[s])
        expansion = Bottleneck.expansion

        for b_idx, (old_blk, new_blk) in enumerate(zip(old_stage.children(), new_stage.children())):
            # conv1 copying
            old_conv1_w = old_blk.conv1.weight.data  # (out_planes, in_ch,1,1)
            new_conv1_w = new_blk.conv1.weight.data
            if prev_expanded_idxs is None:
                # copy channels from old indices into new positions
                for j, old_p in enumerate(keep):
                    new_conv1_w[j].copy_(old_conv1_w[old_p])
            else:
                for j, old_p in enumerate(keep):
                    new_conv1_w[j].copy_(old_conv1_w[old_p][prev_expanded_idxs, ...])

            # bn1 copy for kept channels
            for attr in ["weight", "bias", "running_mean", "running_var"]:
                old_val = getattr(old_blk.bn1, attr).data
                new_val = getattr(new_blk.bn1, attr).data
                new_val.copy_(old_val[keep])

            # conv2: keep rows and cols corresponding to keep
            old_conv2_w = old_blk.conv2.weight.data
            new_conv2_w = new_blk.conv2.weight.data
            for j_out, old_p_out in enumerate(keep):
                for j_in, old_p_in in enumerate(keep):
                    new_conv2_w[j_out, j_in].copy_(old_conv2_w[old_p_out, old_p_in])

            # bn2
            for attr in ["weight", "bias", "running_mean", "running_var"]:
                old_val = getattr(old_blk.bn2, attr).data
                new_val = getattr(new_blk.bn2, attr).data
                new_val.copy_(old_val[keep])

            # conv3: expanded outputs
            old_conv3_w = old_blk.conv3.weight.data
            new_conv3_w = new_blk.conv3.weight.data
            for j_new, old_plane_idx in enumerate(keep):
                for r in range(expansion):
                    new_idx = j_new * expansion + r
                    old_idx = old_plane_idx * expansion + r
                    if prev_expanded_idxs is None:
                        new_conv3_w[new_idx].copy_(old_conv3_w[old_idx])
                    else:
                        new_conv3_w[new_idx].copy_(old_conv3_w[old_idx][prev_expanded_idxs, ...])

            # bn3
            for attr in ["weight", "bias", "running_mean", "running_var"]:
                old_val = getattr(old_blk.bn3, attr).data
                new_val = getattr(new_blk.bn3, attr).data
                old_idxs = []
                for old_plane_idx in keep:
                    for r in range(expansion):
                        old_idxs.append(old_plane_idx * expansion + r)
                new_val.copy_(old_val[old_idxs])

            # downsample if present
            if hasattr(old_blk, "downsample") and old_blk.downsample is not None:
                old_ds = old_blk.downsample
                new_ds = new_blk.downsample
                old_ds_conv_w = old_ds[0].weight.data
                new_ds_conv_w = new_ds[0].weight.data
                old_out_idxs = []
                for old_plane_idx in keep:
                    for r in range(expansion):
                        old_out_idxs.append(old_plane_idx * expansion + r)
                if prev_expanded_idxs is None:
                    for j_new, out_idx in enumerate(old_out_idxs):
                        new_ds_conv_w[j_new].copy_(old_ds_conv_w[out_idx])
                else:
                    for j_new, out_idx in enumerate(old_out_idxs):
                        new_ds_conv_w[j_new].copy_(old_ds_conv_w[out_idx][prev_expanded_idxs, ...])
                # bn in downsample
                for attr in ["weight", "bias", "running_mean", "running_var"]:
                    old_val = getattr(old_ds[1], attr).data
                    new_val = getattr(new_ds[1], attr).data
                    new_val.copy_(old_val[old_out_idxs])

            prev_expanded_idxs = expanded_indices(keep, expansion=expansion)

    # copy fc weights (select columns corresponding to final kept expanded indices)
    final_keep = keep_indices_per_stage["layer4"]
    final_expanded = []
    for p in final_keep:
        for r in range(Bottleneck.expansion):
            final_expanded.append(p * Bottleneck.expansion + r)
    old_fc_w = old_model.fc.weight.data  # shape (num_classes, old_in)
    new_fc_w = new_model.fc.weight.data
    for out_i in range(old_fc_w.shape[0]):
        new_fc_w[out_i].copy_(old_fc_w[out_i, final_expanded])
    new_model.fc.bias.data.copy_(old_model.fc.bias.data)

    return new_model

# -------------------------
# Metrics: params, zeros, flops, timing, RAM, model size
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
            total += labels.size(0)
            correct += int(predicted.eq(labels).sum().item())
            probs_list.append(torch.softmax(outputs, dim=1).cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    loss_avg = loss_total / total
    acc = correct / total
    probs_all = np.concatenate(probs_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)
    try:
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
    # save to temp file to measure on-disk size
    import tempfile, torch
    fd, tmp = tempfile.mkstemp(suffix=".pth")
    os.close(fd)
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp)
    os.remove(tmp)
    return size

def compute_flops(model):
    model = model.eval()
    try:
        if HAS_FVCORE:
            example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            fca = FlopCountAnalysis(model, example)
            return float(fca.total())  # raw flops (MACs-like)
        else:
            # fallback approximate (per-image)
            flops = 0
            hooks = []
            def conv_hook(self, inp, out):
                nonlocal flops
                in_t = inp[0]
                out_t = out
                kh, kw = self.kernel_size
                C_in = in_t.shape[1]; C_out = out_t.shape[1]
                H_out = out_t.shape[2]; W_out = out_t.shape[3]
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
        print("FLOPs estimate failed:", e)
        return float("nan")

def measure_inference_time_and_peak_ram(model, loader, batch_size=BATCH_SIZE, warmup=WARMUP, timed=TIMING_BATCHES):
    model.eval()
    use_cuda = DEVICE.type == "cuda"
    it = iter(loader)
    # warmup
    try:
        for _ in range(warmup):
            imgs, _ = next(it)
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                _ = model(imgs)
            if use_cuda:
                torch.cuda.synchronize()
    except StopIteration:
        pass
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    batches_done = 0
    try:
        for _ in range(timed):
            imgs, _ = next(it)
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                _ = model(imgs)
            if use_cuda:
                torch.cuda.synchronize()
            batches_done += 1
    except StopIteration:
        pass
    end = time.time()
    elapsed = end - start
    avg_batch = elapsed / max(1, batches_done)
    if use_cuda:
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_mb = peak_bytes / (1024**2)
    else:
        peak_mb = params_count(model) * 4.0 / (1024**2)
    return avg_batch, peak_mb, batches_done

# -------------------------
# Pipeline
# -------------------------
rows = []

# Baseline metrics
print("\n=== EVALUATE BASELINE ===")
loss_b, acc_b, auc_b = evaluate_model_basic(baseline, test_loader)
zeros_b, total_b = count_zeros_and_total(baseline)
params_b = params_count(baseline)
flops_b = compute_flops(baseline)
flops_b_m = flops_b / 1e6 if not math.isnan(flops_b) else float("nan")
avg_time_b, peak_ram_b, done_b = measure_inference_time_and_peak_ram(baseline, test_loader)
size_b = os.path.getsize(BASELINE_CKPT) / (1024**2) if os.path.exists(BASELINE_CKPT) else (model_size_bytes(baseline)/(1024**2))
power_b_m = (flops_b * ((total_b - zeros_b) / total_b)) / 1e6 if not math.isnan(flops_b) else float("nan")

rows.append({
    "Variant": "baseline", "Stage": "baseline", "Ratio": 0.0,
    "Acc": acc_b, "AUC": auc_b, "Loss": loss_b,
    "Params": params_b, "Zeros": zeros_b, "TotalParams": total_b, "PctZeros": (zeros_b/total_b)*100 if total_b>0 else 0,
    "ModelSizeMB": size_b,
    "FLOPs_per_image": flops_b, "FLOPs_M_per_image": flops_b_m,
    "InferenceTime_per_batch32_s": avg_time_b, "PeakRAM_MB": peak_ram_b,
    "PowerProxy_MFLOPs": power_b_m, "ModelPath": BASELINE_CKPT
})
print("Baseline done:", rows[-1])

# For each method & ratio, do surgery, evaluate before/after finetune
for method in ["l1", "bn_gamma"]:
    for ratio in RATIOS:
        print(f"\n=== SURGERY prune method={method} ratio={ratio} ===")
        # compute per-stage keep idxs
        keep_indices = {}
        STAGE_NAMES = ["layer1", "layer2", "layer3", "layer4"]
        for s in STAGE_NAMES:
            first_block = next(getattr(baseline, s).children())
            orig = first_block.conv1.out_channels
            keep_k = max(1, int(math.floor(orig * (1.0 - ratio))))
            keep = compute_stage_importance_and_keeps(baseline, s, keep_k, method="bn_gamma" if method=="bn_gamma" else "l1")
            keep_indices[s] = keep
            print(f"  stage {s}: keep {len(keep)}/{orig}")

        # build pruned model
        pruned = build_pruned_resnet_and_copy_weights(baseline, keep_indices, NUM_CLASSES)
        pruned = pruned.to(DEVICE)
        # dummy forward check
        try:
            pruned.eval()
            with torch.no_grad():
                _ = pruned(torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE))
            print("  Dummy forward OK")
        except Exception as e:
            print("  Dummy forward failed:", e)
            # continue but likely will error later

        # evaluate pre-finetune
        loss_pre, acc_pre, auc_pre = evaluate_model_basic(pruned, test_loader)
        zeros_pre, total_pre = count_zeros_and_total(pruned)
        params_pre = params_count(pruned)
        flops_pre = compute_flops(pruned)
        flops_pre_m = flops_pre / 1e6 if not math.isnan(flops_pre) else float("nan")
        avg_time_pre, peak_ram_pre, done_pre = measure_inference_time_and_peak_ram(pruned, test_loader)
        # save before model
        before_path = os.path.join(SAVE_DIR, f"dermamnist_resnet50_{method}_r{int(ratio*100)}_before.pth")
        torch.save(pruned.state_dict(), before_path)
        size_before_mb = os.path.getsize(before_path)/(1024**2)
        power_pre_m = (flops_pre * ((total_pre - zeros_pre)/total_pre)) / 1e6 if not math.isnan(flops_pre) else float("nan")

        rows.append({
            "Variant": method, "Stage": "before_finetune", "Ratio": ratio,
            "Acc": acc_pre, "AUC": auc_pre, "Loss": loss_pre,
            "Params": params_pre, "Zeros": zeros_pre, "TotalParams": total_pre, "PctZeros": (zeros_pre/total_pre)*100 if total_pre>0 else 0,
            "ModelSizeMB": size_before_mb, "FLOPs_per_image": flops_pre, "FLOPs_M_per_image": flops_pre_m,
            "InferenceTime_per_batch32_s": avg_time_pre, "PeakRAM_MB": peak_ram_pre,
            "PowerProxy_MFLOPs": power_pre_m, "ModelPath": before_path
        })
        print(f"  Pre-finetune appended: Acc={acc_pre:.4f}, AUC={auc_pre:.4f}")

        # finetune
        print("  Fine-tuning pruned model...")
        opt = optim.Adam(pruned.parameters(), lr=LEARNING_RATE)
        for ep in range(FINETUNE_EPOCHS):
            pruned.train()
            running_loss = 0.0; total = 0; correct = 0
            for bidx, (imgs, labels) in enumerate(train_loader, 1):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad()
                out = pruned(imgs)
                loss = criterion(out, labels)
                loss.backward()
                opt.step()
                running_loss += float(loss.item()) * imgs.size(0)
                _, preds = out.max(1)
                total += labels.size(0); correct += int(preds.eq(labels).sum().item())
                if bidx % LOG_INTERVAL == 0:
                    print(f"    Finetune ep{ep+1} batch {bidx}/{len(train_loader)} - loss {running_loss/total:.4f}, acc {correct/total:.4f}")
            vloss, vacc, vauc = evaluate_model_basic(pruned, val_loader)
            print(f"    Finetune epoch {ep+1} done: ValLoss {vloss:.4f}, ValAcc {vacc:.4f}, ValAUC {vauc:.4f}")

        # evaluate post-finetune
        loss_post, acc_post, auc_post = evaluate_model_basic(pruned, test_loader)
        zeros_post, total_post = count_zeros_and_total(pruned)
        params_post = params_count(pruned)
        flops_post = compute_flops(pruned)
        flops_post_m = flops_post / 1e6 if not math.isnan(flops_post) else float("nan")
        avg_time_post, peak_ram_post, done_post = measure_inference_time_and_peak_ram(pruned, test_loader)
        after_path = os.path.join(SAVE_DIR, f"dermamnist_resnet50_{method}_r{int(ratio*100)}_finetuned.pth")
        torch.save(pruned.state_dict(), after_path)
        size_after_mb = os.path.getsize(after_path)/(1024**2)
        power_post_m = (flops_post * ((total_post - zeros_post)/total_post)) / 1e6 if not math.isnan(flops_post) else float("nan")

        rows.append({
            "Variant": method, "Stage": "after_finetune", "Ratio": ratio,
            "Acc": acc_post, "AUC": auc_post, "Loss": loss_post,
            "Params": params_post, "Zeros": zeros_post, "TotalParams": total_post, "PctZeros": (zeros_post/total_post)*100 if total_post>0 else 0,
            "ModelSizeMB": size_after_mb, "FLOPs_per_image": flops_post, "FLOPs_M_per_image": flops_post_m,
            "InferenceTime_per_batch32_s": avg_time_post, "PeakRAM_MB": peak_ram_post,
            "PowerProxy_MFLOPs": power_post_m, "ModelPath": after_path
        })
        print(f"  Post-finetune appended: Acc={acc_post:.4f}, AUC={auc_post:.4f}")

# Save CSV
csv_path = os.path.join(SAVE_DIR, "dermamnist_surgery_pruning_metrics.csv")
df = pd.DataFrame(rows)
df.to_csv(csv_path, index=False)
print("\nAll done. CSV:", csv_path)
print(df.to_string(index=False))
