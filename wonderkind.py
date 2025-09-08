#!/usr/bin/env python3
"""
Progressive PGTO pruning for multiple MedMNIST datasets (fixed surgery + progressive updates + KD).
Optimized for memory efficiency using memory-mapped data loading.
"""

import os, time, math, random, tempfile, copy
from typing import Dict, List
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

# -------------------------
# Config
# -------------------------
SAVE_DIR_BASE = "/home/arihangupta/Pruning/dinov2/Pruning/experiment1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32  # Default batch size as fallback

METHODS = ["regional_gradients", "l1", "bn_gamma"]
TARGET_RATIOS = [0.5, 0.6, 0.7]

CAL_EPOCHS = 1
CAL_MAX_BATCHES = 150
CAL_LR = 3e-4

KD_EPOCHS = 1
KD_LR = 3e-4
KD_ALPHA = 0.7
KD_TEMPERATURE = 3.0
KD_MAX_BATCHES = None

FINAL_FINETUNE_EPOCHS = 2
FINAL_LR = 1e-4

LOG_INTERVAL = 20
WARMUP = 5
TIMING_BATCHES = 30
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

def build_resnet50_for_load(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

STAGES = ["layer1", "layer2", "layer3", "layer4"]

def stage_orig_channels(model, stage_name):
    first_block = next(getattr(model, stage_name).children())
    return first_block.conv1.out_channels

# -------------------------
# Importance scoring
# -------------------------
def compute_stage_importance_and_keeps(model: nn.Module, stage_name: str, keep_k: int, method: str="l1",
                                      calib_loader: DataLoader=None, max_batches: int=RG_CAL_MAX_BATCHES):
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
        for block in stage.children():
            w = block.conv3.weight.detach().abs().cpu().numpy()
            for p in range(orig_planes):
                weight_l1[p] += np.sum(w[p*expansion:(p+1)*expansion,:,:,:])
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
    else:
        raise ValueError(f"Unknown method {method}")
    keep = np.arange(len(agg)) if keep_k >= len(agg) else np.argsort(agg)[-keep_k:]
    return np.sort(keep)

# -------------------------
# Surgery: fixed copying
# -------------------------
def build_pruned_resnet_and_copy_weights_fixed(base_model: nn.Module, keep_indices: Dict[str, np.ndarray], num_classes: int):
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
            expanded_rows = torch.cat([ (k * expansion + torch.arange(expansion, device=DEVICE)) for k in out_planes ])
            old_w = old_block.conv1.weight.data
            new_block.conv1.weight.data.copy_(old_w[out_planes][:, in_idx, :, :])
            if getattr(old_block.conv1, 'bias', None) is not None:
                new_block.conv1.bias.data.copy_(old_block.conv1.bias.data[out_planes])
            new_block.bn1.weight.data.copy_(old_block.bn1.weight.data[out_planes])
            new_block.bn1.bias.data.copy_(old_block.bn1.bias.data[out_planes])
            new_block.bn1.running_mean.data.copy_(old_block.bn1.running_mean.data[out_planes])
            new_block.bn1.running_var.data.copy_(old_block.bn1.running_var.data[out_planes])
            new_block.conv2.weight.data.copy_(old_block.conv2.weight.data[out_planes][:, out_planes, :, :])
            if getattr(old_block.conv2, 'bias', None) is not None:
                new_block.conv2.bias.data.copy_(old_block.conv2.bias.data[out_planes])
            new_block.bn2.weight.data.copy_(old_block.bn2.weight.data[out_planes])
            new_block.bn2.bias.data.copy_(old_block.bn2.bias.data[out_planes])
            new_block.bn2.running_mean.data.copy_(old_block.bn2.running_mean.data[out_planes])
            new_block.bn2.running_var.data.copy_(old_block.bn2.running_var.data[out_planes])
            new_block.conv3.weight.data.copy_(old_block.conv3.weight.data[expanded_rows][:, out_planes, :, :])
            if getattr(old_block.conv3, 'bias', None) is not None:
                new_block.conv3.bias.data.copy_(old_block.conv3.bias.data[expanded_rows])
            new_block.bn3.weight.data.copy_(old_block.bn3.weight.data[expanded_rows])
            new_block.bn3.bias.data.copy_(old_block.bn3.bias.data[expanded_rows])
            new_block.bn3.running_mean.data.copy_(old_block.bn3.running_mean.data[expanded_rows])
            new_block.bn3.running_var.data.copy_(old_block.bn3.running_var.data[expanded_rows])
            if old_block.downsample is not None:
                ds_conv = old_block.downsample[0]
                ds_bn = old_block.downsample[1]
                new_block.downsample[0].weight.data.copy_(ds_conv.weight.data[expanded_rows][:, in_idx, :, :])
                new_block.downsample[1].weight.data.copy_(ds_bn.weight.data[expanded_rows])
                new_block.downsample[1].bias.data.copy_(ds_bn.bias.data[expanded_rows])
                new_block.downsample[1].running_mean.data.copy_(ds_bn.running_mean.data[expanded_rows])
                new_block.downsample[1].running_var.data.copy_(ds_bn.running_var.data[expanded_rows])
            prev_out_idx = expanded_rows
    last_kept = torch.tensor(keep_indices['layer4'], dtype=torch.long, device=DEVICE)
    fc_in_idx = torch.cat([torch.arange(p * expansion, (p + 1) * expansion, dtype=torch.long, device=DEVICE) for p in last_kept])
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
        "InferenceTime_per_batch_s": avg_time, "PeakRAM_MB": peak_ram,
        "PowerProxy_MFLOPs": power_m, "ModelPath": path_hint
    }

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
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward(); opt.step()
            running_loss += float(loss.item()) * imgs.size(0)
            _, preds = out.max(1)
            total += labels.size(0); correct = int(preds.eq(labels).sum().item())
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
            total += labels.size(0); correct = int(preds.eq(labels).sum().item())
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
            total += labels.size(0); correct = int(preds.eq(labels).sum().item())
            if bidx % LOG_INTERVAL == 0:
                print(f"    Global FT ep{ep+1} batch{bidx} - loss {running_loss/max(1,total):.4f}, acc {correct/max(1,total):.4f}")
        vloss, vacc, vauc = evaluate_model_basic(model, val_loader)
        print(f"    Global FT epoch {ep+1}: ValLoss {vloss:.4f}, ValAcc {vacc:.4f}, ValAUC {vauc:.4f}")
    model.eval()
    return model

# -------------------------
# Main pipeline
# -------------------------
for dataset_name, cfg in DATASETS.items():
    try:
        print(f"\n\n===================== DATASET: {dataset_name.upper()} =====================")
        log_memory_usage(f"Before loading {dataset_name}: ")

        SAVE_DIR = f"{SAVE_DIR_BASE}/{dataset_name}"
        os.makedirs(SAVE_DIR, exist_ok=True)

        csv_path = os.path.join(SAVE_DIR, f"{dataset_name}_pgto_pruning_metrics_progressive_fixed.csv")
        if os.path.exists(csv_path):
            print(f"Skipping {dataset_name}: CSV already exists at {csv_path}")
            continue

        batch_size = DATASET_BATCH_SIZES.get(dataset_name, BATCH_SIZE)
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
        log_memory_usage(f"After baseline evaluation for {dataset_name}: ")

        for method in METHODS:
            for target_ratio in TARGET_RATIOS:
                print(f"\n=== PROGRESSIVE PGTO: method={method}, target_ratio={target_ratio} ===")
                current_model = copy.deepcopy(baseline).to(DEVICE)
                keep_indices = {s: np.arange(stage_orig_channels(baseline, s)) for s in STAGES}
                log_memory_usage(f"Before pruning loop for {method}, ratio={target_ratio}: ")

                for s in STAGES:
                    orig = stage_orig_channels(current_model, s)
                    keep_k = max(1, int(math.floor(orig * (1.0 - target_ratio))))
                    keeps = compute_stage_importance_and_keeps(current_model, s, keep_k, method=method, calib_loader=train_loader)
                    keep_indices[s] = keeps
                    print(f"  Stage {s}: keep {len(keeps)}/{orig} ({100*len(keeps)/orig:.1f}% kept)")
                    pruned_model = build_pruned_resnet_and_copy_weights_fixed(current_model, keep_indices={**{k: keep_indices[k] if k==s else np.arange(stage_orig_channels(current_model, k)) for k in STAGES}}, num_classes=NUM_CLASSES)
                    pruned_model = pruned_model.to(DEVICE)
                    pruned_model.eval()
                    with torch.no_grad(): _ = pruned_model(torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE))
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
                all_pruned_ckpt = os.path.join(SAVE_DIR, f"pgto_{method}_r{int(target_ratio*100)}_allpruned_preKD.pth")
                torch.save(current_model.state_dict(), all_pruned_ckpt)
                row = collect_metrics_row(method, "all_pruned_preKD", target_ratio, current_model, test_loader, all_pruned_ckpt)
                rows.append(row)
                print("  All-pruned (pre-KD) metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})
                print("  Knowledge distillation (student <- teacher)...")
                current_model = distill_student(current_model, baseline, train_loader, epochs=KD_EPOCHS, lr=KD_LR, alpha=KD_ALPHA, T=KD_TEMPERATURE, max_batches=KD_MAX_BATCHES)
                kd_ckpt = os.path.join(SAVE_DIR, f"pgto_{method}_r{int(target_ratio*100)}_afterKD.pth")
                torch.save(current_model.state_dict(), kd_ckpt)
                row = collect_metrics_row(method, "after_kd", target_ratio, current_model, test_loader, kd_ckpt)
                rows.append(row)
                print("  KD metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})
                print("  Final global finetune...")
                current_model = global_finetune(current_model, train_loader, val_loader, epochs=FINAL_FINETUNE_EPOCHS, lr=FINAL_LR)
                final_ckpt = os.path.join(SAVE_DIR, f"pgto_{method}_r{int(target_ratio*100)}_final.pth")
                torch.save(current_model.state_dict(), final_ckpt)
                row = collect_metrics_row(method, "after_global_finetune", target_ratio, current_model, test_loader, final_ckpt)
                rows.append(row)
                print("  Final metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"All done for {dataset_name}. CSV: {csv_path}")
        del baseline, current_model, pruned_model, train_loader, val_loader, test_loader, train_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_memory_usage(f"After completing {dataset_name}: ")
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"Continuing to next dataset...")
print("All datasets processed.")