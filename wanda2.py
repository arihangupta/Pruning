#!/usr/bin/env python3
"""
Dermamnist ResNet50 PGTO pruning with L1, BN-gamma and Regional Gradients (Wanda++ style).

Features:
 - Stage-by-stage pruning (layer1..layer4)
 - Proper regional-gradients scoring using real calibration data from train_loader
 - Short local calibration after each stage (PGTO-style)
 - Final short global finetune after all stages
 - Saves .pth after each stage and after final finetune
 - CSV summary of metrics for every saved checkpoint

This file is meant to be copy-paste runnable. Edit only the path constants at the top if needed.
"""

import os, time, math, random, tempfile
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
# Config (edit these paths if needed)
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

METHODS = ["l1", "bn_gamma", "regional_gradients"]
TARGET_RATIOS = [0.5, 0.6, 0.7]
CAL_EPOCHS = 1
CAL_MAX_BATCHES = 150   # used for calibration training loops
RG_CAL_MAX_BATCHES = 50 # max batches for regional-gradients scoring
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
    return train_loader, val_loader, test_loader, int(len(np.unique(y_train))), train_ds

train_loader, val_loader, test_loader, NUM_CLASSES, train_ds = make_loaders(DERMA_PATH)
print(f"Data loaded. NUM_CLASSES={NUM_CLASSES}, device={DEVICE}")

# -------------------------
# Model builder (variable stage widths)
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

# -------------------------
# Baseline load
# -------------------------

def build_resnet50_for_load(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_baseline_ckpt(path=BASELINE_CKPT):
    model = build_resnet50_for_load(NUM_CLASSES)
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model

baseline = load_baseline_ckpt()
print("Baseline loaded.")

# -------------------------
# Helper: stage width
# -------------------------
STAGES = ["layer1","layer2","layer3","layer4"]

def stage_orig_channels(model, stage_name):
    first_block = next(getattr(model, stage_name).children())
    return first_block.conv1.out_channels

# -------------------------
# Importance scoring (L1, BN, Regional Gradients)
# -------------------------

def compute_stage_importance_and_keeps(model: nn.Module, stage_name: str, keep_k: int, method: str="l1",
                                      calib_loader: DataLoader=None, max_batches:int=RG_CAL_MAX_BATCHES):
    """
    Returns sorted indices to KEEP for the stage (length keep_k).

    For regional_gradients (Wanda++ style) we compute per-plane:
      - act_norm_j = average over calibration batches of ||X_j||_2 (grouping expanded channels)
      - grad_norm_j = average over calibration batches of ||G_j||_2 (grad of regional loss wrt conv3 weights, grouped)
      - weight_l1_j = L1 norm of conv3 weights for the plane
    Importance_j = act_norm_j * grad_norm_j * weight_l1_j
    """
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    orig_planes = first_block.conv1.out_channels
    expansion = 4

    if method == "l1":
        block_importances = []
        for block in stage.children():
            conv3 = block.conv3.weight.detach().abs().cpu().numpy()
            per_plane = [np.sum(conv3[p*expansion:(p+1)*expansion,:,:,:]) for p in range(orig_planes)]
            block_importances.append(np.array(per_plane))
        agg = np.mean(np.stack(block_importances, axis=0), axis=0)

    elif method == "bn_gamma":
        block_importances = []
        for block in stage.children():
            gammas = block.bn3.weight.detach().abs().cpu().numpy()
            per_plane = [np.mean(gammas[p*expansion:(p+1)*expansion]) for p in range(orig_planes)]
            block_importances.append(np.array(per_plane))
        agg = np.mean(np.stack(block_importances, axis=0), axis=0)

    elif method == "regional_gradients":
        assert calib_loader is not None, "regional_gradients requires a calib_loader"
        device = DEVICE
        act_norms = torch.zeros(orig_planes, device=device)
        grad_norms = torch.zeros(orig_planes, device=device)
        weight_l1 = torch.zeros(orig_planes, device=device)

        # weight L1 per plane (sum over blocks)
        for block in stage.children():
            w = block.conv3.weight.detach().abs().cpu().numpy()
            for p in range(orig_planes):
                weight_l1[p] += np.sum(w[p*expansion:(p+1)*expansion,:,:,:])
        weight_l1 = weight_l1.to(device)

        # register forward hook to capture stage output
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
            act = saved['act']  # shape [B, C_exp, H, W]

            # regional loss = L2 norm (mean) of stage activations
            loss = (act ** 2).mean()
            loss.backward(retain_graph=True)

            # activation norms per plane: group expanded channels
            with torch.no_grad():
                Cexp = act.shape[1]
                act_flat = act.detach().permute(1,0,2,3).reshape(Cexp, -1)  # [Cexp, B*H*W]
                for p in range(orig_planes):
                    idx0 = p*expansion
                    idx1 = (p+1)*expansion
                    part = act_flat[idx0:idx1]
                    act_norms[p] += torch.norm(part)

            # gradient norms: gather conv3.weight.grad from each block and group
            for block in stage.children():
                g = block.conv3.weight.grad
                if g is None:
                    continue
                g_abs = g.detach().abs()
                # g_abs shape [Cexp, Cin, k, k]
                g_per_out = g_abs.view(g_abs.shape[0], -1).norm(dim=1)  # per expanded-out-channel norm
                for p in range(orig_planes):
                    idx0 = p*expansion; idx1 = (p+1)*expansion
                    grad_norms[p] += g_per_out[idx0:idx1].norm()

            batch_count += 1
            # clear saved to avoid stale tensors
            saved.pop('act', None)

        handle.remove()
        if batch_count == 0:
            agg = weight_l1.cpu().numpy()
        else:
            act_norms /= batch_count
            grad_norms /= batch_count
            agg = (act_norms * grad_norms * weight_l1).cpu().numpy()

    else:
        raise ValueError(f"Unknown method {method}")

    keep = np.arange(len(agg)) if keep_k >= len(agg) else np.argsort(agg)[-keep_k:]
    return np.sort(keep)

# -------------------------
# Build pruned model and copy weights (surgery) - robust for Bottleneck
# -------------------------

def build_pruned_resnet(keep_indices, num_classes):
    stage_planes = [len(keep_indices['layer1']), len(keep_indices['layer2']),
                    len(keep_indices['layer3']), len(keep_indices['layer4'])]
    layers = [3, 4, 6, 3]
    return CustomResNet(block=Bottleneck, layers=layers, stage_planes=stage_planes, num_classes=num_classes)


def build_pruned_resnet_and_copy_weights(base_model, keep_indices, num_classes):
    new_model = build_pruned_resnet(keep_indices, num_classes=num_classes).to(DEVICE)
    expansion = 4
    STAGES = ["layer1","layer2","layer3","layer4"]

    prev_out_idx = torch.arange(base_model.conv1.out_channels, dtype=torch.long, device=DEVICE)

    for stage_name in STAGES:
        old_stage = getattr(base_model, stage_name)
        new_stage = getattr(new_model, stage_name)
        kept = torch.tensor(keep_indices[stage_name], dtype=torch.long, device=DEVICE)

        for block_idx, (old_block, new_block) in enumerate(zip(old_stage, new_stage)):
            in_idx = prev_out_idx
            out_idx = kept

            # conv1
            new_block.conv1.weight.data.copy_(old_block.conv1.weight.data[out_idx][:, in_idx, :, :])
            if old_block.conv1.bias is not None:
                new_block.conv1.bias.data.copy_(old_block.conv1.bias.data[out_idx])
            new_block.bn1.weight.data.copy_(old_block.bn1.weight.data[out_idx])
            new_block.bn1.bias.data.copy_(old_block.bn1.bias.data[out_idx])
            new_block.bn1.running_mean.data.copy_(old_block.bn1.running_mean.data[out_idx])
            new_block.bn1.running_var.data.copy_(old_block.bn1.running_var.data[out_idx])

            # conv2
            new_block.conv2.weight.data.copy_(old_block.conv2.weight.data[out_idx][:, out_idx, :, :])
            if old_block.conv2.bias is not None:
                new_block.conv2.bias.data.copy_(old_block.conv2.bias.data[out_idx])
            new_block.bn2.weight.data.copy_(old_block.bn2.weight.data[out_idx])
            new_block.bn2.bias.data.copy_(old_block.bn2.bias.data[out_idx])
            new_block.bn2.running_mean.data.copy_(old_block.bn2.running_mean.data[out_idx])
            new_block.bn2.running_var.data.copy_(old_block.bn2.running_var.data[out_idx])

            # conv3 (expanded out)
            expanded_idx = torch.arange(len(new_block.conv3.weight.data), dtype=torch.long, device=DEVICE)
            new_block.conv3.weight.data.copy_(old_block.conv3.weight.data[expanded_idx][:, out_idx, :, :])
            if old_block.conv3.bias is not None:
                new_block.conv3.bias.data.copy_(old_block.conv3.bias.data[expanded_idx])
            new_block.bn3.weight.data.copy_(old_block.bn3.weight.data[expanded_idx])
            new_block.bn3.bias.data.copy_(old_block.bn3.bias.data[expanded_idx])
            new_block.bn3.running_mean.data.copy_(old_block.bn3.running_mean.data[expanded_idx])
            new_block.bn3.running_var.data.copy_(old_block.bn3.running_var.data[expanded_idx])

            # downsample
            if old_block.downsample is not None:
                ds_w = old_block.downsample[0].weight.data
                new_block.downsample[0].weight.data.copy_(ds_w[expanded_idx][:, in_idx, :, :])
                new_block.downsample[1].weight.data.copy_(old_block.downsample[1].weight.data[expanded_idx])
                new_block.downsample[1].bias.data.copy_(old_block.downsample[1].bias.data[expanded_idx])
                new_block.downsample[1].running_mean.data.copy_(old_block.downsample[1].running_mean.data[expanded_idx])
                new_block.downsample[1].running_var.data.copy_(old_block.downsample[1].running_var.data[expanded_idx])

            prev_out_idx = out_idx.repeat(expansion)

    # fc
    last_keep = torch.tensor(keep_indices['layer4'], dtype=torch.long, device=DEVICE)
    fc_in_idx = last_keep.repeat(expansion)
    new_model.fc.weight.data.copy_(base_model.fc.weight.data[:, fc_in_idx])
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
            return float("nan")
    except Exception:
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
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if use_cuda else params_count(model)*4.0/(1024**2)
    return avg_batch, peak_mb, batches_done


def collect_metrics_row(tag_variant, tag_stage, ratio, model, test_loader, path_hint):
    loss, acc, auc = evaluate_model_basic(model, test_loader)
    zeros, total = count_zeros_and_total(model)
    params = params_count(model)
    flops = compute_flops(model)
    flops_m = flops / 1e6 if not math.isnan(flops) else float("nan")
    avg_time, peak_ram, _ = inference_time_per_batch(model, test_loader)
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
        "InferenceTime_per_batch32_s": avg_time, "PeakRAM_MB": peak_ram,
        "PowerProxy_MFLOPs": power_m, "ModelPath": path_hint
    }

# -------------------------
# Freeze / unfreeze helpers & local calibration (PGTO style)
# -------------------------

def freeze_all(model):
    for p in model.parameters(): p.requires_grad = False


def unfreeze_stage(model, stage_name):
    for name, p in model.named_parameters():
        if name.startswith(stage_name):
            p.requires_grad = True
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
# Main PGTO pipeline
# -------------------------

rows = []

# Baseline metrics
print("=== EVALUATE BASELINE ===")
base_ckpt = os.path.join(SAVE_DIR, "baseline.pth")
torch.save(baseline.state_dict(), base_ckpt)
row = collect_metrics_row("baseline", "baseline", 0.0, baseline, test_loader, base_ckpt)
rows.append(row)
print("Baseline done:", {k: row[k] for k in ["Acc","AUC","ModelSizeMB","FLOPs_M_per_image"]})

for method in METHODS:
    for target_ratio in TARGET_RATIOS:
        print(f"
=== PGTO: method={method}, target_ratio={target_ratio} ===")
        keep_indices = {s: np.arange(stage_orig_channels(baseline, s)) for s in STAGES}

        calib_loader = train_loader

        for s in STAGES:
            orig = stage_orig_channels(baseline, s)
            keep_k = max(1, int(math.floor(orig * (1.0 - target_ratio))))
            keep_indices[s] = compute_stage_importance_and_keeps(baseline, s, keep_k, method=method, calib_loader=calib_loader)
            print(f"  -> stage {s}: keep {len(keep_indices[s])}/{orig} ({100*len(keep_indices[s])/orig:.1f}% kept)")

            pruned_model = build_pruned_resnet_and_copy_weights(baseline, keep_indices, NUM_CLASSES).to(DEVICE)
            pruned_model.eval()
            with torch.no_grad(): _ = pruned_model(torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE))

            print(f"    Calibrating {s} (local, gentle)...")
            calibrate_stage(pruned_model, s, train_loader, epochs=CAL_EPOCHS, max_batches=CAL_MAX_BATCHES, lr=CAL_LR)

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
            print("    Stage metrics:", {k: row[k] for k in ["Acc","AUC","ModelSizeMB","FLOPs_M_per_image"]})

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
        print("  Final metrics:", {k: row[k] for k in ["Acc","AUC","ModelSizeMB","FLOPs_M_per_image"]})

# Save CSV
csv_path = os.path.join(SAVE_DIR, "dermamnist_pgto_pruning_metrics.csv")
df = pd.DataFrame(rows); df.to_csv(csv_path, index=False)
print("All done. CSV:", csv_path)
