#!/usr/bin/env python3
"""
Dermamnist ResNet50 model-surgery pruning (PGTO: stage-by-stage) with metrics.

What’s new vs your previous script:
 - Stage-by-stage pruning (layer1..layer4) with short local calibration after each stage.
 - Freezes all layers except the just-pruned stage during calibration (gentle + localized).
 - Saves & logs metrics after each stage, then runs a short global finetune and logs again.

Outputs:
 - .pth after every stage and after final finetune
 - CSV summary with metrics for baseline + every stage checkpoint

Requirements:
 - torch, torchvision, sklearn
 - fvcore (optional, for FLOPs). Fallback hook estimator is used if missing.
"""

import os, time, math, random, copy, csv, tempfile
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
import torch.nn.functional as F

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
SAVE_DIR   = "/home/arihangupta/Pruning/dinov2/Pruning/saved_models_pgto"
BASELINE_CKPT = os.path.join("/home/arihangupta/Pruning/dinov2/Pruning/saved_models",
                             "dermamnist_resnet50_BASELINE.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 32
IMG_SIZE = 224

# Wanda++-style knobs
TARGET_RATIOS = [0.5, 0.7]     # final prune targets per stage (50%, 70%)
METHODS = ["wanda++", "l1", "bn_gamma"]   # importance criterion
CAL_EPOCHS = 1                 # short local calibration after each stage
CAL_MAX_BATCHES = 150          # cap steps for calibration
CAL_LR = 3e-4
FINAL_FINETUNE_EPOCHS = 2      # short global finetune after all stages
FINAL_LR = 1e-4

# Logging/timing
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
# CustomResNet builder (variable stage widths)
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
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model

baseline = load_baseline_ckpt()
print("Baseline loaded.")

# -------------------------
# Importance & keep computation
# -------------------------
def compute_stage_importance_and_keeps(model: nn.Module, stage_name: str, keep_k: int, method: str="l1", train_loader=None, alpha=1.0):
    """
    Compute importance scores per channel in a stage, blending Wanda++-style activation norms and regional gradients.
    Args:
        model: ResNet model
        stage_name: e.g., "layer1"
        keep_k: number of channels to keep
        method: "wanda++" for the new metric (default="l1"), or "l1"/"bn_gamma"
        train_loader: DataLoader for calibration data
        alpha: weighting factor for gradients (default=1.0)
    Returns:
        keep: sorted indices of channels to keep
    """
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    orig_planes = first_block.conv1.out_channels  # Original input channels to the stage
    block_importances = []

    if method == "wanda++" and train_loader is not None:
        # Compute activation norms and gradients via a single forward/backward pass
        model.eval()
        with torch.no_grad():
            # Collect activations from one batch for calibration
            try:
                imgs, _ = next(iter(train_loader))
                imgs = imgs.to(DEVICE)
                print(f"Debug: Initial BN3 running_mean shape for {stage_name}: {next(iter(stage.children())).bn3.running_mean.shape}")
                print(f"Debug: Initial conv1 in_channels for {stage_name}: {first_block.conv1.in_channels}")

                # Forward pass to update model state with pruned structure
                _ = model(imgs)  # Warm up to propagate pruned structure
                
                # Resize BN running stats to match pruned input channels
                pruned_in_channels = keep_k  # Use the target kept channels as the new input count
                print(f"Debug: Adjusting BN3 for {stage_name}, pruned_in_channels={pruned_in_channels}")
                for block in stage.children():
                    if hasattr(block.bn1, 'running_mean') and block.bn1.running_mean.shape[0] != pruned_in_channels:
                        print(f"Debug: Resizing BN1 from {block.bn1.running_mean.shape[0]} to {pruned_in_channels}")
                        block.bn1.running_mean.data = torch.zeros(pruned_in_channels, device=DEVICE) if pruned_in_channels < block.bn1.running_mean.shape[0] else block.bn1.running_mean.data[:pruned_in_channels]
                        block.bn1.running_var.data = torch.ones(pruned_in_channels, device=DEVICE) if pruned_in_channels < block.bn1.running_var.shape[0] else block.bn1.running_var.data[:pruned_in_channels]
                        block.bn1.weight.data = torch.ones(pruned_in_channels, device=DEVICE) if pruned_in_channels < block.bn1.weight.shape[0] else block.bn1.weight.data[:pruned_in_channels]
                        block.bn1.bias.data = torch.zeros(pruned_in_channels, device=DEVICE) if pruned_in_channels < block.bn1.bias.shape[0] else block.bn1.bias.data[:pruned_in_channels]
                    if hasattr(block.bn3, 'running_mean') and block.bn3.running_mean.shape[0] != pruned_in_channels * 4:  # Adjust for expansion
                        print(f"Debug: Resizing BN3 from {block.bn3.running_mean.shape[0]} to {pruned_in_channels * 4}")
                        block.bn3.running_mean.data = torch.zeros(pruned_in_channels * 4, device=DEVICE) if pruned_in_channels * 4 < block.bn3.running_mean.shape[0] else block.bn3.running_mean.data[:pruned_in_channels * 4]
                        block.bn3.running_var.data = torch.ones(pruned_in_channels * 4, device=DEVICE) if pruned_in_channels * 4 < block.bn3.running_var.shape[0] else block.bn3.running_var.data[:pruned_in_channels * 4]
                        block.bn3.weight.data = torch.ones(pruned_in_channels * 4, device=DEVICE) if pruned_in_channels * 4 < block.bn3.weight.shape[0] else block.bn3.weight.data[:pruned_in_channels * 4]
                        block.bn3.bias.data = torch.zeros(pruned_in_channels * 4, device=DEVICE) if pruned_in_channels * 4 < block.bn3.bias.shape[0] else block.bn3.bias.data[:pruned_in_channels * 4]
                
                print(f"Debug: Post-resize BN1 running_mean shape for {stage_name}: {next(iter(stage.children())).bn1.running_mean.shape}")
                print(f"Debug: Post-resize BN3 running_mean shape for {stage_name}: {next(iter(stage.children())).bn3.running_mean.shape}")

                # Forward pass to get intermediate activations
                def hook_fn(module, input, output):
                    return input[0]  # Store input activations
                hooks = []
                for block in stage.children():
                    hook = block.conv1.register_forward_hook(hook_fn)
                    hooks.append(hook)
                _ = model(imgs)
                # Aggregate activations per channel (across spatial dims and batch)
                act_norms = []
                for hook in hooks:
                    act_input = hook[0]  # Shape: [batch, in_channels, H, W]
                    channel_norms = torch.norm(act_input, p=2, dim=(0, 2, 3))  # L2 norm per channel
                    act_norms.append(channel_norms.cpu().numpy())
                    hook.remove()
                act_norms = np.mean(np.stack(act_norms), axis=0)  # Average across blocks

                # Backward pass for regional gradients (L2 loss on stage output)
                model.zero_grad()
                outputs = []
                def output_hook(module, input, output):
                    outputs.append(output)
                hook_out = stage.register_forward_hook(output_hook)
                _ = model(imgs)
                hook_out.remove()
                stage_output = outputs[0]  # Shape: [batch, out_channels, H, W]
                regional_loss = torch.norm(stage_output, p=2)  # L2 norm of stage output
                regional_loss.backward()

                # Extract gradients for conv3 weights (main output conv in Bottleneck)
                grad_magnitudes = []
                for block in stage.children():
                    grad = block.conv3.weight.grad
                    if grad is not None:
                        grad_mag = torch.abs(grad).mean(dim=(1, 2, 3))  # Average over kernel dims
                        grad_magnitudes.append(grad_mag.cpu().numpy())
                grad_magnitudes = np.mean(np.stack(grad_magnitudes), axis=0)  # Average across blocks
                grad_magnitudes = grad_magnitudes[:orig_planes]  # Align with input channels

                # Combine into Wanda++ score
                weight_mags = []
                for block in stage.children():
                    conv3_w = block.conv3.weight.detach().abs().cpu().numpy()
                    exp = block.conv3.out_channels // block.conv3.in_channels
                    per_plane = [np.sum(conv3_w[p*exp:(p+1)*exp, :, :, :]) for p in range(orig_planes)]
                    weight_mags.append(np.array(per_plane))
                weight_mags = np.mean(np.stack(weight_mags), axis=0)

                # RGS: (alpha * G_ij + ||X_j||_2) * |W_ij|
                importance = (alpha * grad_magnitudes + act_norms) * weight_mags
            except Exception as e:
                print(f"Warning: Wanda++ computation failed for {stage_name}, falling back to l1. Error: {e}")
                method = "l1"  # Fallback if calibration fails

    # Fallback methods based on the resolved method
    if method == "l1":
        for block in stage.children():
            conv3 = block.conv3.weight.detach().abs().cpu().numpy()
            exp = block.conv3.out_channels // block.conv3.in_channels
            per_plane = [np.sum(np.abs(conv3[p*exp:(p+1)*exp, :, :, :])) for p in range(orig_planes)]
            block_importances.append(np.array(per_plane))
        importance = np.mean(np.stack(block_importances, axis=0), axis=0)
    elif method == "bn_gamma":
        for block in stage.children():
            gammas = block.bn3.weight.detach().abs().cpu().numpy()
            exp = block.conv3.out_channels // block.conv3.in_channels
            per_plane = [np.mean(gammas[p*exp:(p+1)*exp]) for p in range(orig_planes)]
            block_importances.append(np.array(per_plane))
        importance = np.mean(np.stack(block_importances, axis=0), axis=0)

    # Select top-k channels to keep
    keep = np.arange(len(importance)) if keep_k >= len(importance) else np.argsort(importance)[-keep_k:]
    return np.sort(keep)

# -------------------------
# Build pruned model and copy weights (surgery)
# -------------------------
def build_pruned_resnet(keep_indices, num_classes):
    stage_planes = [
        len(keep_indices['layer1']),
        len(keep_indices['layer2']),
        len(keep_indices['layer3']),
        len(keep_indices['layer4'])
    ]
    layers = [3, 4, 6, 3]
    return CustomResNet(block=Bottleneck, layers=layers, stage_planes=stage_planes, num_classes=num_classes)

def build_pruned_resnet_and_copy_weights(base_model, keep_indices, num_classes):
    new_model = build_pruned_resnet(keep_indices, num_classes=num_classes).to(DEVICE)
    expansion = 4
    STAGES = ["layer1", "layer2", "layer3", "layer4"]
    prev_expanded_idxs = None

    for stage_name in STAGES:
        old_layer = getattr(base_model, stage_name)
        new_layer = getattr(new_model, stage_name)
        kept = keep_indices[stage_name]
        for block_idx, (old_block, new_block) in enumerate(zip(old_layer, new_layer)):
            # conv1
            if stage_name == "layer1" and block_idx == 0:
                new_block.conv1.weight.data.copy_(old_block.conv1.weight.data[kept])
                prev_expanded_idxs_safe = torch.arange(min(kept.size(0), old_block.conv1.weight.shape[1]))
            else:
                old_w = old_block.conv1.weight.data
                prev_expanded_idxs_safe = torch.arange(min(kept.size(0), old_w.shape[1])) if prev_expanded_idxs is None else prev_expanded_idxs
                new_block.conv1.weight.data.copy_(old_w[kept][:, prev_expanded_idxs_safe, :, :])
            if old_block.conv1.bias is not None:
                new_block.conv1.bias.data.copy_(old_block.conv1.bias.data[kept])
            # bn1
            new_block.bn1.weight.data.copy_(old_block.bn1.weight.data[:kept.size(0)])
            new_block.bn1.bias.data.copy_(old_block.bn1.bias.data[:kept.size(0)])
            new_block.bn1.running_mean.data.copy_(old_block.bn1.running_mean.data[:kept.size(0)])
            new_block.bn1.running_var.data.copy_(old_block.bn1.running_var.data[:kept.size(0)])
            # conv2
            old_w = old_block.conv2.weight.data
            new_block.conv2.weight.data.copy_(old_w[kept][:, :kept.size(0), :, :])
            if old_block.conv2.bias is not None:
                new_block.conv2.bias.data.copy_(old_block.conv2.bias.data[kept])
            # bn2
            new_block.bn2.weight.data.copy_(old_block.bn2.weight.data[:kept.size(0)])
            new_block.bn2.bias.data.copy_(old_block.bn2.bias.data[:kept.size(0)])
            new_block.bn2.running_mean.data.copy_(old_block.bn2.running_mean.data[:kept.size(0)])
            new_block.bn2.running_var.data.copy_(old_block.bn2.running_var.data[:kept.size(0)])
            # conv3
            old_w = old_block.conv3.weight.data
            expanded_idx = np.repeat(kept, expansion)
            old_idx = torch.tensor(expanded_idx, dtype=torch.long)
            new_block.conv3.weight.data.copy_(old_w[old_idx][:, :kept.size(0), :, :])
            if old_block.conv3.bias is not None:
                new_block.conv3.bias.data.copy_(old_block.conv3.bias.data[old_idx])
            # bn3
            new_block.bn3.weight.data.copy_(old_block.bn3.weight.data[:kept.size(0) * expansion])
            new_block.bn3.bias.data.copy_(old_block.bn3.bias.data[:kept.size(0) * expansion])
            new_block.bn3.running_mean.data.copy_(old_block.bn3.running_mean.data[:kept.size(0) * expansion])
            new_block.bn3.running_var.data.copy_(old_block.bn3.running_var.data[:kept.size(0) * expansion])
            # downsample
            if old_block.downsample is not None:
                ds_old = old_block.downsample[0].weight.data
                ds_new = new_block.downsample[0].weight.data
                ds_new.copy_(ds_old[old_idx][:, prev_expanded_idxs_safe, :, :])
                new_block.downsample[1].weight.data.copy_(old_block.downsample[1].weight.data[:kept.size(0) * expansion])
                new_block.downsample[1].bias.data.copy_(old_block.downsample[1].bias.data[:kept.size(0) * expansion])
                new_block.downsample[1].running_mean.data.copy_(old_block.downsample[1].running_mean.data[:kept.size(0) * expansion])
                new_block.downsample[1].running_var.data.copy_(old_block.downsample[1].running_var.data[:kept.size(0) * expansion])
            prev_expanded_idxs = old_idx

    # fc
    new_model.fc.weight.data.copy_(base_model.fc.weight.data[:, prev_expanded_idxs])
    new_model.fc.bias.data.copy_(base_model.fc.bias.data)
    return new_model

# -------------------------
# Metrics & evaluation
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
    model = model.eval()
    try:
        if HAS_FVCORE:
            example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            fca = FlopCountAnalysis(model, example)
            return float(fca.total())
        else:
            flops = 0
            hooks = []
            def conv_hook(self, inp, out):
                nonlocal flops
                in_t = inp[0]; out_t = out
                kh, kw = self.kernel_size
                C_in = in_t.shape[1]; C_out = out_t.shape[1]
                H_out = out_t.shape[2]; W_out = out_t.shape[3]
                flops += kh * kw * C_in * C_out * H_out * W_out
            def linear_hook(self, inp, out):
                nonlocal flops
                flops += inp[0].shape[1] * out.shape[1]
            for m in model.modules():
                if isinstance(m, nn.Conv2d): hooks.append(m.register_forward_hook(conv_hook))
                if isinstance(m, nn.Linear): hooks.append(m.register_forward_hook(linear_hook))
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            with torch.no_grad(): _ = model(dummy)
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
            with torch.no_grad(): _ = model(imgs)
            if use_cuda: torch.cuda.synchronize()
    except StopIteration:
        pass
    if use_cuda: torch.cuda.reset_peak_memory_stats()
    start = time.time()
    batches_done = 0
    try:
        for _ in range(timed):
            imgs, _ = next(it)
            imgs = imgs.to(DEVICE)
            with torch.no_grad(): _ = model(imgs)
            if use_cuda: torch.cuda.synchronize()
            batches_done += 1
    except StopIteration:
        pass
    elapsed = time.time() - start
    avg_batch = elapsed / max(1, batches_done)
    if use_cuda:
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_mb = params_count(model) * 4.0 / (1024**2)
    return avg_batch, peak_mb, batches_done

def collect_metrics_row(tag_variant, tag_stage, ratio, model, test_loader, path_hint):
    loss, acc, auc = evaluate_model_basic(model, test_loader)
    zeros, total = count_zeros_and_total(model)
    params = params_count(model)
    flops = compute_flops(model)
    flops_m = flops / 1e6 if not math.isnan(flops) else float("nan")
    avg_time, peak_ram, _ = measure_inference_time_and_peak_ram(model, test_loader)
    if os.path.exists(path_hint):
        size_mb = os.path.getsize(path_hint)/(1024**2)
    else:
        size_mb = model_size_bytes(model)/(1024**2)
    power_m = (flops * ((total - zeros)/total)) / 1e6 if not math.isnan(flops) and total>0 else float("nan")
    return {
        "Variant": tag_variant, "Stage": tag_stage, "Ratio": ratio,
        "Acc": acc, "AUC": auc, "Loss": loss,
        "Params": params, "Zeros": zeros, "TotalParams": total, "PctZeros": (zeros/total)*100 if total>0 else 0,
        "ModelSizeMB": size_mb, "FLOPs_per_image": flops, "FLOPs_M_per_image": flops_m,
        "InferenceTime_per_batch32_s": avg_time, "PeakRAM_MB": peak_ram,
        "PowerProxy_MFLOPs": power_m, "ModelPath": path_hint
    }

# -------------------------
# Calibration helpers (PGTO)
# -------------------------
def freeze_all(model):
    for p in model.parameters(): p.requires_grad = False

def unfreeze_stage(model, stage_name):
    for name, p in model.named_parameters():
        if name.startswith(stage_name):
            p.requires_grad = True
    # Also allow FC & top BN to breathe a bit
    for name, p in model.named_parameters():
        if name.startswith("fc.") or name.startswith("bn1."):
            p.requires_grad = True

def calibrate_stage(model, stage_name, train_loader, epochs=CAL_EPOCHS, max_batches=CAL_MAX_BATCHES, lr=CAL_LR):
    freeze_all(model)
    unfreeze_stage(model, stage_name)
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
                return

# -------------------------
# PGTO pipeline
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

# Helper to get original stage widths from baseline
def stage_orig_channels(model, stage_name):
    first_block = next(getattr(model, stage_name).children())
    return first_block.conv1.out_channels

STAGES = ["layer1","layer2","layer3","layer4"]

# PGTO loop: for each method and target ratio, prune stage-by-stage with calibration
for method in METHODS:
    for target_ratio in TARGET_RATIOS:
        print(f"\n=== PGTO: method={method}, target_ratio={target_ratio} ===")
        # We will progressively build keep_indices. Start with "keep all".
        keep_indices = {s: np.arange(stage_orig_channels(baseline, s)) for s in STAGES}
        current_base = baseline  # always map from the original baseline weights (stable surgery)

        for s in STAGES:
            orig = stage_orig_channels(baseline, s)
            keep_k = max(1, int(math.floor(orig * (1.0 - target_ratio))))
            # compute fresh keeps for JUST this stage, from the current_base (baseline)
            keep_indices[s] = compute_stage_importance_and_keeps(current_base, s, keep_k, method=method, train_loader=train_loader, alpha=1.0)
            print(f"  -> stage {s}: keep {len(keep_indices[s])}/{orig} ({100*len(keep_indices[s])/orig:.1f}% kept)")

            # Build model with new keep_indices (earlier stages fixed; later stages full)
            pruned_model = build_pruned_resnet_and_copy_weights(baseline, keep_indices, NUM_CLASSES).to(DEVICE)
            # Quick sanity pass
            pruned_model.eval()
            with torch.no_grad():
                _ = pruned_model(torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE))

            # Calibrate ONLY this stage (freeze others)
            print(f"    Calibrating {s} (local, gentle)...")
            calibrate_stage(pruned_model, s, train_loader, epochs=CAL_EPOCHS, max_batches=CAL_MAX_BATCHES, lr=CAL_LR)

            # Save & log metrics AFTER this stage calibration
            stage_ckpt = os.path.join(SAVE_DIR, f"dermamnist_resnet50_pgto_{method}_r{int(target_ratio*100)}_{s}_calibrated.pth")
            torch.save(pruned_model.state_dict(), stage_ckpt)
            row = collect_metrics_row(
                tag_variant=f"{method}",
                tag_stage=f"{s}_after_stage_calibration",
                ratio=target_ratio,
                model=pruned_model,
                test_loader=test_loader,
                path_hint=stage_ckpt
            )
            rows.append(row)
            print("    Stage metrics:", {k: row[k] for k in ["Acc","AUC","ModelSizeMB","FLOPs_M_per_image","InferenceTime_per_batch32_s","PeakRAM_MB","PowerProxy_MFLOPs"]})

            # Update baseline reference for next stage? (Wanda++ scans per block but maps from the same full model.)
            # We keep mapping from the SAME original baseline to avoid compounding copy noise.
            # The latest keep_indices carry the pruning history forward.

        # Final short global finetune (taste once more)
        print("  Final short global finetune...")
        pruned_model = build_pruned_resnet_and_copy_weights(baseline, keep_indices, NUM_CLASSES).to(DEVICE)
        opt = optim.Adam(pruned_model.parameters(), lr=FINAL_LR)
        for ep in range(FINAL_FINETUNE_EPOCHS):
            pruned_model.train()
            running_loss=0; total=0; correct=0
            for bidx,(imgs,labels) in enumerate(train_loader,1):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad(); out = pruned_model(imgs)
                loss = criterion(out, labels); loss.backward(); opt.step()
                running_loss += float(loss.item())*imgs.size(0)
                _, preds = out.max(1); total += labels.size(0); correct += int(preds.eq(labels).sum().item())
                if bidx % LOG_INTERVAL == 0:
                    print(f"    Global FT ep{ep+1} batch{bidx} - loss {running_loss/max(1,total):.4f}, acc {correct/max(1,total):.4f}")
            vloss, vacc, vauc = evaluate_model_basic(pruned_model, val_loader)
            print(f"    Global FT epoch {ep+1}: ValLoss {vloss:.4f}, ValAcc {vacc:.4f}, ValAUC {vauc:.4f}")

        final_ckpt = os.path.join(SAVE_DIR, f"dermamnist_resnet50_pgto_{method}_r{int(target_ratio*100)}_final.pth")
        torch.save(pruned_model.state_dict(), final_ckpt)
        row = collect_metrics_row(
            tag_variant=f"{method}",
            tag_stage="after_global_finetune",
            ratio=target_ratio,
            model=pruned_model,
            test_loader=test_loader,
            path_hint=final_ckpt
        )
        rows.append(row)
        print("  Final metrics:", {k: row[k] for k in ["Acc","AUC","ModelSizeMB","FLOPs_M_per_image","InferenceTime_per_batch32_s","PeakRAM_MB","PowerProxy_MFLOPs"]})

# Save CSV
csv_path = os.path.join(SAVE_DIR, "dermamnist_pgto_pruning_metrics.csv")
df = pd.DataFrame(rows); df.to_csv(csv_path, index=False)
print("\nAll done. CSV:", csv_path)
print(df.to_string(index=False))