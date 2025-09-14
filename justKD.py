#!/usr/bin/env python3
"""
Slim ResNet-50 variants for MedMNIST datasets: 50%, 60%, 70% smaller models with random init,
knowledge distillation recovery + global finetune, and energy tracking via CodeCarbon.

Key Changes:
- Models: Uniformly scaled-down stage planes for target size reductions (50%, 60%, 70% params/FLOPs).
- Initialization: Random/empty weights (no surgery/pruning; Kaiming init).
- Recovery: KD (2 epochs) followed by global finetune (1-2 epochs).
- Energy: Tracks KD, finetune, baseline/pruned inference; computes break-even predictions.
- Outputs: Per-dataset CSV with performance + energy metrics; emissions.csv via CodeCarbon.

Requirements: torch, torchvision, sklearn, torchprofile, codecarbon (optional), psutil.
"""

import os
import time
import math
import random
import tempfile
import copy
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
SAVE_DIR_BASE = "/home/arihangupta/Pruning/dinov2/Pruning/experiment3_slim_kd"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224
BATCH_SIZE_DEFAULT = 32

# Target size reductions (fraction of original params/FLOPs)
TARGET_REDUCTIONS = [0.5, 0.6, 0.7]  # 50%, 60%, 70% smaller

KD_EPOCHS = 2
KD_LR = 3e-4
KD_ALPHA = 0.7
KD_TEMPERATURE = 3.0

FINAL_FINETUNE_EPOCHS = 2  # 1-2 epochs
FINAL_LR = 1e-4

LOG_INTERVAL = 20
WARMUP = 5
TIMING_BATCHES = 30

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
# Data helpers (memory-mapped)
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
# Models / builder (with random init)
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

        # Random init (empty weights)
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

def build_slim_resnet(stage_planes, num_classes):
    """Build slim model with given stage_planes, random init."""
    return CustomResNet(block=Bottleneck, layers=[3,4,6,3], stage_planes=stage_planes, num_classes=num_classes).to(DEVICE)

# Original stage planes
ORIGINAL_PLANES = [64, 128, 256, 512]

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
    images_count = 0
    try:
        for _ in range(timed):
            imgs, _ = next(it)
            imgs = imgs.to(DEVICE)
            with torch.no_grad(): _ = model(imgs)
            if use_cuda: torch.cuda.synchronize()
            batches_done += 1
            images_count += imgs.size(0)
    except StopIteration:
        pass
    elapsed = time.time() - start
    avg_batch = elapsed / max(1, batches_done)
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if use_cuda else params_count(model)*4.0/(1024**2)
    return avg_batch, peak_mb, images_count

def collect_metrics_row(tag_variant, tag_stage, reduction, model, test_loader, path_hint):
    loss, acc, auc = evaluate_model_basic(model, test_loader)
    params = params_count(model)
    flops = compute_flops(model)
    flops_m = flops / 1e6 if not math.isnan(flops) else float("nan")
    avg_time, peak_ram, _ = inference_time_per_batch(model, test_loader)
    if path_hint is not None and os.path.exists(path_hint):
        size_mb = os.path.getsize(path_hint)/(1024**2)
    else:
        size_mb = model_size_bytes(model)/(1024**2)
    return {
        "Variant": tag_variant, "Stage": tag_stage, "Reduction": reduction,
        "Acc": acc, "AUC": auc, "Loss": loss,
        "Params": params, "ModelSizeMB": size_mb, "FLOPs_per_image": flops, "FLOPs_M_per_image": flops_m,
        "InferenceTime_per_batch_s": avg_time, "PeakRAM_MB": peak_ram,
        "ModelPath": path_hint
    }

# -------------------------
# KD
# -------------------------
def distill_student(student: nn.Module, teacher: nn.Module, train_loader: DataLoader,
                    epochs: int=KD_EPOCHS, lr: float=KD_LR, alpha: float=KD_ALPHA, T: float=KD_TEMPERATURE):
    teacher.eval()
    student.train()
    opt = optim.Adam(student.parameters(), lr=lr)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    device = DEVICE
    for ep in range(epochs):
        running_loss = 0.0; total = 0; correct = 0
        for bidx, (imgs, labels) in enumerate(train_loader, 1):
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
# Main pipeline
# -------------------------
for dataset_name, cfg in DATASETS.items():
    try:
        print(f"\n\n===================== DATASET: {dataset_name.upper()} =====================")
        log_memory_usage(f"Before loading {dataset_name}: ")

        SAVE_DIR = f"{SAVE_DIR_BASE}/{dataset_name}"
        os.makedirs(SAVE_DIR, exist_ok=True)

        csv_path = os.path.join(SAVE_DIR, f"{dataset_name}_slim_kd_metrics_with_energy.csv")
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

        # Baseline inference energy
        baseline_tracker = start_tracker(SAVE_DIR, f"{dataset_name}_baseline_inference") if CODECARBON_AVAILABLE else None
        _, _, baseline_images = inference_time_per_batch(baseline, test_loader)
        baseline_inf_metrics = stop_tracker_and_get_metrics(baseline_tracker, SAVE_DIR, f"{dataset_name}_baseline_inference")
        baseline_energy_kwh = baseline_inf_metrics["energy_kwh"]
        baseline_emissions_kg = baseline_inf_metrics["emissions_kg"]
        baseline_energy_per_pred_kwh = baseline_energy_kwh / baseline_images if baseline_images > 0 and not math.isnan(baseline_energy_kwh) else float("nan")
        print(f"Baseline inference: images={baseline_images}, energy_kWh={baseline_energy_kwh}, emissions_kg={baseline_emissions_kg}")

        for reduction in TARGET_REDUCTIONS:
            print(f"\n=== SLIM KD: reduction={reduction} ({int(reduction*100)}% smaller) ===")

            # Compute scaled stage planes (uniform reduction)
            stage_planes = [max(1, int(p * (1 - reduction))) for p in ORIGINAL_PLANES]
            print(f"  Scaled planes: {stage_planes}")

            # Start tracker for KD + finetune
            kd_ft_proj = f"{dataset_name}_slim_r{int(reduction*100)}_kd_ft"
            kd_ft_tracker = start_tracker(SAVE_DIR, kd_ft_proj) if CODECARBON_AVAILABLE else None

            # Build slim student with random init
            student = build_slim_resnet(stage_planes, NUM_CLASSES)
            print("  Slim student built (random init).")

            # Pre-KD eval (random weights)
            pre_kd_ckpt = os.path.join(SAVE_DIR, f"slim_r{int(reduction*100)}_pre_kd.pth")
            torch.save(student.state_dict(), pre_kd_ckpt)
            row = collect_metrics_row("slim_kd", "pre_kd", reduction, student, test_loader, pre_kd_ckpt)
            rows.append(row)
            print("  Pre-KD metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

            # KD
            print("  Knowledge distillation...")
            student = distill_student(student, baseline, train_loader, epochs=KD_EPOCHS)
            post_kd_ckpt = os.path.join(SAVE_DIR, f"slim_r{int(reduction*100)}_post_kd.pth")
            torch.save(student.state_dict(), post_kd_ckpt)
            row = collect_metrics_row("slim_kd", "post_kd", reduction, student, test_loader, post_kd_ckpt)
            rows.append(row)
            print("  Post-KD metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

            # Global finetune
            print("  Global finetune...")
            student = global_finetune(student, train_loader, val_loader, epochs=FINAL_FINETUNE_EPOCHS)
            final_ckpt = os.path.join(SAVE_DIR, f"slim_r{int(reduction*100)}_final.pth")
            torch.save(student.state_dict(), final_ckpt)
            row = collect_metrics_row("slim_kd", "post_finetune", reduction, student, test_loader, final_ckpt)
            rows.append(row)
            print("  Final metrics:", {k: row[k] for k in ["Acc", "AUC", "ModelSizeMB", "FLOPs_M_per_image"]})

            # Stop KD+FT tracker
            kd_ft_metrics = stop_tracker_and_get_metrics(kd_ft_tracker, SAVE_DIR, kd_ft_proj)
            training_energy_kwh = kd_ft_metrics["energy_kwh"]
            training_emissions_kg = kd_ft_metrics["emissions_kg"]
            print(f"  Training (KD+FT) energy_kWh={training_energy_kwh}, emissions_kg={training_emissions_kg}")

            # Pruned (final student) inference energy
            pruned_tracker = start_tracker(SAVE_DIR, f"{dataset_name}_slim_r{int(reduction*100)}_inference") if CODECARBON_AVAILABLE else None
            _, _, pruned_images = inference_time_per_batch(student, test_loader)
            pruned_inf_metrics = stop_tracker_and_get_metrics(pruned_tracker, SAVE_DIR, f"{dataset_name}_slim_r{int(reduction*100)}_inference")
            pruned_energy_kwh = pruned_inf_metrics["energy_kwh"]
            pruned_emissions_kg = pruned_inf_metrics["emissions_kg"]
            pruned_energy_per_pred_kwh = pruned_energy_kwh / pruned_images if pruned_images > 0 and not math.isnan(pruned_energy_kwh) else float("nan")
            print(f"  Slim inference: images={pruned_images}, energy_kWh={pruned_energy_kwh}, emissions_kg={pruned_emissions_kg}")

            # Break-even
            if math.isnan(training_energy_kwh) or math.isnan(baseline_energy_per_pred_kwh) or math.isnan(pruned_energy_per_pred_kwh):
                break_even = float("nan")
            else:
                delta = baseline_energy_per_pred_kwh - pruned_energy_per_pred_kwh
                if delta <= 0:
                    break_even = float("inf")
                else:
                    break_even = training_energy_kwh / delta

            # Energy summary row
            energy_row = {
                "Variant": "slim_kd",
                "Stage": f"energy_summary_r{int(reduction*100)}",
                "Reduction": reduction,
                "TrainingEnergy_kWh": training_energy_kwh,
                "TrainingEmissions_kg": training_emissions_kg,
                "BaselineInferenceEnergy_kWh_total": baseline_energy_kwh,
                "BaselineEnergy_per_pred_kWh": baseline_energy_per_pred_kwh,
                "BaselineEmissions_kg_total": baseline_emissions_kg,
                "SlimInferenceEnergy_kWh_total": pruned_energy_kwh,
                "SlimEnergy_per_pred_kWh": pruned_energy_per_pred_kwh,
                "SlimEmissions_kg_total": pruned_emissions_kg,
                "BreakEvenPredictions": break_even
            }
            rows.append(energy_row)
            print("  Energy summary:", energy_row)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_memory_usage(f"After reduction {reduction} for {dataset_name}: ")
            del student

        # Save CSV
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"All done for {dataset_name}. CSV: {csv_path}")
        print(df.to_string(index=False))

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