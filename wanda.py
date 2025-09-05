#!/usr/bin/env python3
"""
Dermamnist ResNet50 PGTO pruning with Regional Gradients, BN, and L1 methods.

Features:
- Stage-by-stage pruning (layer1..layer4)
- Short local calibration after each stage
- Final short global finetune
- Saves .pth after each stage and final finetune
- CSV summary of all metrics

Requirements:
- torch, torchvision, sklearn, numpy, pandas
- Optional: fvcore for accurate FLOPs
"""

import os, time, math, random, copy, csv, tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.resnet import Bottleneck
from torchvision import models
from sklearn.metrics import roc_auc_score

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

TARGET_RATIOS = [0.5, 0.6, 0.7]
METHODS = ["l1", "bn_gamma", "regional_gradients"]  
CAL_EPOCHS = 1
CAL_MAX_BATCHES = 150
CAL_LR = 3e-4
FINAL_FINETUNE_EPOCHS = 2
FINAL_LR = 1e-4
LOG_INTERVAL = 20
WARMUP = 5
TIMING_BATCHES = 30

# Try fvcore for FLOPs
HAS_FVCORE = True
try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    HAS_FVCORE = False

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
# CustomResNet builder
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
def compute_stage_importance_and_keeps(model: nn.Module, stage_name: str, keep_k: int, method: str="l1"):
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    orig_planes = first_block.conv1.out_channels
    block_importances = []

    for block in stage.children():
        if method == "bn_gamma":
            gammas = block.bn3.weight.detach().abs().cpu().numpy()
            exp = block.conv3.out_channels // block.conv3.in_channels
            per_plane = [np.mean(gammas[p*exp:(p+1)*exp]) for p in range(orig_planes)]
        elif method == "regional_gradients":
            # Compute simple "gradient magnitude" scoring with random input (placeholder)
            inp = torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE).requires_grad_(True)
            out = model(inp)
            loss = out.sum()
            loss.backward()
            grad = block.conv3.weight.grad
            if grad is not None:
                exp = block.conv3.out_channels // block.conv3.in_channels
                grad_score = grad.detach().abs().cpu().numpy()
                per_plane = [np.mean(grad_score[p*exp:(p+1)*exp]) for p in range(orig_planes)]
            else:
                per_plane = [np.sum(np.abs(block.conv3.weight.detach().cpu().numpy()[p*exp:(p+1)*exp,:,:,:])) for p in range(orig_planes)]
        else:  # l1
            conv3 = block.conv3.weight.detach().abs().cpu().numpy()
            exp = block.conv3.out_channels // block.conv3.in_channels
            per_plane = [np.sum(np.abs(conv3[p*exp:(p+1)*exp,:,:,:])) for p in range(orig_planes)]
        block_importances.append(np.array(per_plane))

    agg = np.mean(np.stack(block_importances, axis=0), axis=0)
    keep = np.arange(len(agg)) if keep_k >= len(agg) else np.argsort(agg)[-keep_k:]
    return np.sort(keep)

# -------------------------
# Build pruned model
# -------------------------
def build_pruned_resnet(keep_indices, num_classes):
    stage_planes = [len(keep_indices['layer1']), len(keep_indices['layer2']),
                    len(keep_indices['layer3']), len(keep_indices['layer4'])]
    layers = [3,4,6,3]
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
                prev_expanded_idxs_safe = torch.arange(old_block.conv1.weight.shape[1])
            else:
                old_w = old_block.conv1.weight.data
                prev_expanded_idxs_safe = torch.arange(old_w.shape[1]) if prev_expanded_idxs is None else prev_expanded_idxs
                new_block.conv1.weight.data.copy_(old_w[kept][:, prev_expanded_idxs_safe, :, :])
            # bn1
            new_block.bn1.weight.data.copy_(old_block.bn1.weight.data[kept])
            new_block.bn1.bias.data.copy_(old_block.bn1.bias.data[kept])
            new_block.bn1.running_mean.data.copy_(old_block.bn1.running_mean.data[kept])
            new_block.bn1.running_var.data.copy_(old_block.bn1.running_var.data[kept])
            # conv2
            old_w = old_block.conv2.weight.data
            new_block.conv2.weight.data.copy_(old_w[kept][:, kept, :, :])
            # bn2
            new_block.bn2.weight.data.copy_(old_block.bn2.weight.data[kept])
            new_block.bn2.bias.data.copy_(old_block.bn2.bias.data[kept])
            new_block.bn2.running_mean.data.copy_(old_block.bn2.running_mean.data[kept])
            new_block.bn2.running_var.data.copy_(old_block.bn2.running_var.data[kept])
            # conv3
            old_w = old_block.conv3.weight.data
            expanded_idx = np.repeat(kept, expansion)
            old_idx = torch.tensor(expanded_idx, dtype=torch.long)
            new_block.conv3.weight.data.copy_(old_w[old_idx][:, kept, :, :])
            # bn3
            new_block.bn3.weight.data.copy_(old_block.bn3.weight.data[old_idx])
            new_block.bn3.bias.data.copy_(old_block.bn3.bias.data[old_idx])
            new_block.bn3.running_mean.data.copy_(old_block.bn3.running_mean.data[old_idx])
            new_block.bn3.running_var.data.copy_(old_block.bn3.running_var.data[old_idx])
            # downsample
            if old_block.downsample is not None:
                ds_old = old_block.downsample[0].weight.data
                ds_new = new_block.downsample[0].weight.data
                ds_new.copy_(ds_old[old_idx][:, prev_expanded_idxs_safe, :, :])
                new_block.downsample[1].weight.data.copy_(old_block.downsample[1].weight.data[old_idx])
                new_block.downsample[1].bias.data.copy_(old_block.downsample[1].bias.data[old_idx])
                new_block.downsample[1].running_mean.data.copy_(old_block.downsample[1].running_mean.data[old_idx])
                new_block.downsample[1].running_var.data.copy_(old_block.downsample[1].running_var.data[old_idx])
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
            dummy_input = torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE)
            flops = FlopCountAnalysis(model, dummy_input).total()
        else:
            flops = float("nan")
    except Exception:
        flops = float("nan")
    return flops / 1e6

def inference_time_per_batch(model, loader, batch_size=BATCH_SIZE):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= TIMING_BATCHES + WARMUP: break
            images = images.to(DEVICE)
            start = time.time()
            _ = model(images)
            torch.cuda.synchronize() if DEVICE.type=="cuda" else None
            end = time.time()
            if i >= WARMUP: times.append(end-start)
    return np.mean(times) if times else float("nan")

def collect_metrics_row(tag_variant, tag_stage, ratio, model, test_loader, path_hint=None):
    loss, acc, auc = evaluate_model_basic(model, test_loader)
    zeros, total = count_zeros_and_total(model)
    params = params_count(model)
    model_size = model_size_bytes(model)
    flops = compute_flops(model)
    inf_time = inference_time_per_batch(model, test_loader)
    power_proxy = (flops * params / max(1, inf_time)) if inf_time>0 else float("nan")
    return {
        "Variant": tag_variant, "Stage": tag_stage, "Ratio": ratio,
        "Loss": loss, "Acc": acc, "AUC": auc, "Zeros": zeros, "Total": total,
        "Params": params, "ModelSizeMB": model_size/1e6, "FLOPs_M_per_image": flops,
        "InferenceTime_per_batch32_s": inf_time, "PowerProxy": power_proxy, "Checkpoint": path_hint
    }

# -------------------------
# Stage calibration
# -------------------------
def calibrate_stage(model, stage_name, loader, epochs=1, max_batches=100, lr=3e-4):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= max_batches: break
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# -------------------------
# Main loop
# -------------------------
rows = []

# Baseline metrics
print("\n=== EVALUATE BASELINE ===")
row = collect_metrics_row("baseline", "baseline", 0.0, baseline, test_loader, BASELINE_CKPT)
rows.append(row)
print("Baseline metrics:", {k: row[k] for k in ["Acc","AUC","ModelSizeMB","FLOPs_M_per_image","InferenceTime_per_batch32_s"]})

STAGES = ["layer1","layer2","layer3","layer4"]

for method in METHODS:
    for target_ratio in TARGET_RATIOS:
        print(f"\n=== PGTO: method={method}, target_ratio={target_ratio} ===")
        keep_indices = {s: np.arange(stage_orig_channels(baseline, s)) for s in STAGES}

        for s in STAGES:
            orig_channels = stage_orig_channels(baseline, s)
            keep_k = max(1, int(math.floor(orig_channels * (1.0 - target_ratio))))
            keep_indices[s] = compute_stage_importance_and_keeps(baseline, s, keep_k, method=method)
            print(f"  Stage {s}: keeping {len(keep_indices[s])}/{orig_channels} channels ({100*len(keep_indices[s])/orig_channels:.1f}%)")

            pruned_model = build_pruned_resnet_and_copy_weights(baseline, keep_indices, NUM_CLASSES).to(DEVICE)
            calibrate_stage(pruned_model, s, train_loader, CAL_EPOCHS, CAL_MAX_BATCHES, CAL_LR)

            stage_ckpt = os.path.join(SAVE_DIR, f"dermamnist_resnet50_pgto_{method}_r{int(target_ratio*100)}_{s}_calibrated.pth")
            torch.save(pruned_model.state_dict(), stage_ckpt)
            row = collect_metrics_row(method, f"{s}_after_stage_calibration", target_ratio, pruned_model, test_loader, stage_ckpt)
            rows.append(row)
            print("    Stage metrics:", {k: row[k] for k in ["Acc","AUC","ModelSizeMB","FLOPs_M_per_image"]})

        # Final short global finetune
        print("  Final global finetune...")
        pruned_model = build_pruned_resnet_and_copy_weights(baseline, keep_indices, NUM_CLASSES).to(DEVICE)
        optimizer = optim.Adam(pruned_model.parameters(), lr=FINAL_LR)
        pruned_model.train()
        for ep in range(FINAL_FINETUNE_EPOCHS):
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = pruned_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        final_ckpt = os.path.join(SAVE_DIR, f"dermamnist_resnet50_pgto_{method}_r{int(target_ratio*100)}_final.pth")
        torch.save(pruned_model.state_dict(), final_ckpt)
        row = collect_metrics_row(method, "final_finetune", target_ratio, pruned_model, test_loader, final_ckpt)
        rows.append(row)
        print("    Final finetune metrics:", {k: row[k] for k in ["Acc","AUC","ModelSizeMB","FLOPs_M_per_image"]})

# Save CSV summary
df = pd.DataFrame(rows)
csv_path = os.path.join(SAVE_DIR, "pgto_summary_metrics.csv")
df.to_csv(csv_path, index=False)
print(f"\nAll metrics saved to {csv_path}")

# -------------------------
# Helpers
# -------------------------
def stage_orig_channels(model, stage_name):
    layer = getattr(model, stage_name)
    first_block = next(layer.children())
    return first_block.conv1.out_channels
