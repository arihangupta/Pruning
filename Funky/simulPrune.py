#!/usr/bin/env python3
"""
progressive_pruning_during_training.py

Compares two training approaches:
1. Baseline: Standard training with dropout, L2 regularization, early stopping
2. Progressive Pruning: Structured channel pruning during training using importance scoring

Uses gradient × activation × weight importance metric for channel selection.
Tracks comprehensive metrics including energy consumption via CodeCarbon.
"""

import os
import time
import math
import random
import tempfile
import copy
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
import gc

# CodeCarbon for energy tracking
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    print("Warning: codecarbon not available. Energy/emissions will be NaN.")

# -------------------------
# Configuration
# -------------------------
SAVE_DIR_BASE = "/home/arihangupta/Pruning/dinov2/Pruning/progressive_pruning_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224

# Dataset configuration
DATASETS = {
    "pathmnist": {
        "path": "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced/pathmnist_224.npz",
        "batch_size": 16
    },
    "dermamnist": {
        "path": "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced/dermamnist_224.npz",
        "batch_size": 32
    },
    "bloodmnist": {
        "path": "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced/bloodmnist_224.npz",
        "batch_size": 32
    },
}

# Training hyperparameters
INITIAL_LR = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization (applied to ALL training)
FIXED_EPOCHS = 15  # Fixed number of epochs for ALL training (baseline and progressive)

# Progressive pruning configuration
WARMUP_EPOCHS = 2  # Train normally before first prune
EPOCHS_BETWEEN_PRUNES = 3  # Fixed interval between pruning steps
NUM_PRUNE_STEPS = 4  # Number of pruning iterations (epochs 3, 6, 9, 12)
PRUNE_PERCENT = 0.10  # Remove 10% of channels each time
LR_REDUCTION_AFTER_PRUNE = 0.5  # Multiply LR by this after each prune

# Prune-then-train configuration (based on progressive pruning final dimensions)
# These are the target dimensions after 4 pruning steps of 10% each
TARGET_PRUNED_CHANNELS = {
    'layer1': 40,   # ~62.5% of original 64
    'layer2': 82,   # ~64% of original 128
    'layer3': 167,  # ~65% of original 256
    'layer4': 334,  # ~65% of original 512
}

# Experimental configuration
NUM_TRIALS = 1  # Number of trials per dataset for statistical reliability

# Importance calculation
IMPORTANCE_CAL_BATCHES = 50  # Batches to use for importance scoring

# Logging and evaluation
LOG_INTERVAL = 20
TIMING_BATCHES = 100
WARMUP_BATCHES = 5

STAGES = ["layer1", "layer2", "layer3", "layer4"]
ORIGINAL_PLANES = [64, 128, 256, 512]

# -------------------------
# Reproducibility
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
# Memory utilities
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
    return train_loader, val_loader, test_loader, num_classes

# -------------------------
# Model architecture
# -------------------------
class CustomResNet(nn.Module):
    """Custom ResNet50 with configurable channel widths (no dropout to maintain compatibility with pretrained weights)"""
    def __init__(self, block=Bottleneck, layers=[3,4,6,3], stage_planes=[64,128,256,512], 
                 num_classes=1000, random_init=False):
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def build_baseline_resnet(num_classes):
    """Build baseline ResNet50 and load ImageNet pretrained weights"""
    # Start with pretrained ResNet50
    base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Create custom model with same architecture
    model = CustomResNet(
        block=Bottleneck, 
        layers=[3,4,6,3], 
        stage_planes=[64,128,256,512],
        num_classes=num_classes, 
        random_init=False
    )
    
    # Copy pretrained weights (everything except FC layer)
    model.conv1.weight.data.copy_(base_model.conv1.weight.data)
    model.bn1.weight.data.copy_(base_model.bn1.weight.data)
    model.bn1.bias.data.copy_(base_model.bn1.bias.data)
    model.bn1.running_mean.copy_(base_model.bn1.running_mean)
    model.bn1.running_var.copy_(base_model.bn1.running_var)
    
    for stage_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        base_stage = getattr(base_model, stage_name)
        model_stage = getattr(model, stage_name)
        model_stage.load_state_dict(base_stage.state_dict())
    
    # Initialize new FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model.to(DEVICE)

def build_pruned_resnet(stage_planes, num_classes):
    """Build a ResNet with custom channel widths (for progressive pruning)
    
    Uses random initialization since pruned architectures don't match 
    pretrained ResNet50 dimensions. Weights are copied from the source 
    model using copy_weights_to_pruned_model().
    """
    stage_planes = [max(1, int(p)) for p in stage_planes]
    return CustomResNet(
        block=Bottleneck, 
        layers=[3,4,6,3], 
        stage_planes=stage_planes,
        num_classes=num_classes, 
        random_init=True  # Must be True for pruned models
    ).to(DEVICE)

def stage_orig_channels(model, stage_name):
    """Get the number of channels in a stage"""
    first_block = next(getattr(model, stage_name).children())
    return first_block.conv1.out_channels

# -------------------------
# Channel importance scoring
# -------------------------
def compute_channel_importance(model, stage_name, calib_loader, max_batches=IMPORTANCE_CAL_BATCHES):
    """
    Compute importance score for each channel in a stage using:
    importance = ||activations|| × ||gradients|| × ||weights||
    
    Returns: numpy array of importance scores per channel
    """
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    orig_planes = first_block.conv1.out_channels
    expansion = 4
    device = DEVICE
    
    # Initialize accumulators
    act_norms = torch.zeros(orig_planes, device=device)
    grad_norms = torch.zeros(orig_planes, device=device)
    weight_l1 = torch.zeros(orig_planes, device=device)
    
    # Compute weight importance (L1 norm per channel)
    for block in stage.children():
        w = block.conv3.weight.detach().abs().cpu().numpy()
        for p in range(orig_planes):
            weight_l1[p] += np.sum(w[p*expansion:(p+1)*expansion])
    weight_l1 = weight_l1.to(device)
    
    # Register forward hook to capture activations
    saved = {}
    def hook_fn(module, inp, out):
        saved['act'] = out
    handle = stage.register_forward_hook(hook_fn)
    
    model.train()
    batch_count = 0
    
    for bidx, (imgs, _) in enumerate(calib_loader, 1):
        imgs = imgs.to(device)
        model.zero_grad()
        
        # Forward pass
        _ = model(imgs)
        
        if 'act' not in saved:
            continue
        
        act = saved['act']
        
        # Compute pseudo-loss for gradient calculation
        loss = (act ** 2).mean()
        loss.backward(retain_graph=True)
        
        # Compute activation norms per channel
        with torch.no_grad():
            Cexp = act.shape[1]  # Should be orig_planes * expansion
            act_flat = act.detach().permute(1,0,2,3).reshape(Cexp, -1)
            for p in range(orig_planes):
                idx0 = p * expansion
                idx1 = (p + 1) * expansion
                part = act_flat[idx0:idx1]
                act_norms[p] += torch.norm(part)
        
        # Compute gradient norms per channel
        for block in stage.children():
            g = block.conv3.weight.grad
            if g is None:
                continue
            g_abs = g.abs()
            g_per_out = g_abs.view(g_abs.shape[0], -1).norm(dim=1)
            for p in range(orig_planes):
                idx0 = p * expansion
                idx1 = (p + 1) * expansion
                grad_norms[p] += g_per_out[idx0:idx1].norm()
        
        batch_count += 1
        if batch_count >= max_batches:
            break
    
    handle.remove()
    
    # Compute final importance scores
    with torch.no_grad():
        importance = act_norms * grad_norms * weight_l1
        importance_np = importance.cpu().numpy()
    
    return importance_np

def select_channels_to_keep(importance_scores, keep_ratio):
    """
    Select top channels based on importance scores.
    
    Args:
        importance_scores: numpy array of importance per channel
        keep_ratio: fraction of channels to keep (e.g., 0.9 for 90%)
    
    Returns: sorted numpy array of channel indices to keep
    """
    num_channels = len(importance_scores)
    num_keep = max(1, int(num_channels * keep_ratio))
    
    if num_keep >= num_channels:
        return np.arange(num_channels)
    
    # Select top channels by importance
    keep_indices = np.argsort(importance_scores)[-num_keep:]
    return np.sort(keep_indices)

# -------------------------
# Weight copying for pruned models
# -------------------------
def copy_weights_to_pruned_model(source_model, target_model, keep_indices):
    """
    Copy weights from source model to pruned target model.
    
    Args:
        source_model: original model
        target_model: new model with fewer channels
        keep_indices: dict mapping stage_name -> channel indices to keep
    """
    # Copy conv1 and bn1 (input stem, not pruned)
    target_model.conv1.weight.data.copy_(source_model.conv1.weight.data)
    target_model.bn1.weight.data.copy_(source_model.bn1.weight.data)
    target_model.bn1.bias.data.copy_(source_model.bn1.bias.data)
    target_model.bn1.running_mean.copy_(source_model.bn1.running_mean)
    target_model.bn1.running_var.copy_(source_model.bn1.running_var)
    
    prev_kept = torch.arange(64, dtype=torch.long, device=DEVICE)
    
    for stage_idx, stage in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
        kept = torch.tensor(keep_indices[stage], dtype=torch.long, device=DEVICE)
        source_stage = getattr(source_model, stage)
        target_stage = getattr(target_model, stage)
        
        print(f"    Copying weights for {stage}: keeping {len(kept)} out of {stage_orig_channels(source_model, stage)} channels")
        
        # Expanded indices (each channel has 4 output channels due to bottleneck)
        expanded_rows = torch.cat([torch.arange(p * 4, (p + 1) * 4, dtype=torch.long, device=DEVICE) for p in kept])
        
        for block_idx, (source_block, target_block) in enumerate(zip(source_stage.children(), target_stage.children())):
            # Input channels for this block
            if block_idx == 0:
                in_idx = prev_kept  # From previous stage
            else:
                in_idx = expanded_rows  # From previous block in same stage
            
            out_idx = kept
            
            # Copy conv1 (1x1 conv)
            target_block.conv1.weight.data.copy_(source_block.conv1.weight.data[out_idx][:, in_idx])
            target_block.bn1.weight.data.copy_(source_block.bn1.weight.data[out_idx])
            target_block.bn1.bias.data.copy_(source_block.bn1.bias.data[out_idx])
            target_block.bn1.running_mean.copy_(source_block.bn1.running_mean[out_idx])
            target_block.bn1.running_var.copy_(source_block.bn1.running_var[out_idx])
            
            # Copy conv2 (3x3 conv)
            target_block.conv2.weight.data.copy_(source_block.conv2.weight.data[out_idx][:, out_idx])
            target_block.bn2.weight.data.copy_(source_block.bn2.weight.data[out_idx])
            target_block.bn2.bias.data.copy_(source_block.bn2.bias.data[out_idx])
            target_block.bn2.running_mean.copy_(source_block.bn2.running_mean[out_idx])
            target_block.bn2.running_var.copy_(source_block.bn2.running_var[out_idx])
            
            # Copy conv3 (1x1 conv, expansion)
            target_block.conv3.weight.data.copy_(source_block.conv3.weight.data[expanded_rows][:, out_idx])
            target_block.bn3.weight.data.copy_(source_block.bn3.weight.data[expanded_rows])
            target_block.bn3.bias.data.copy_(source_block.bn3.bias.data[expanded_rows])
            target_block.bn3.running_mean.copy_(source_block.bn3.running_mean[expanded_rows])
            target_block.bn3.running_var.copy_(source_block.bn3.running_var[expanded_rows])
            
            # Copy downsample if exists
            if source_block.downsample is not None and target_block.downsample is not None:
                downsample_in_idx = prev_kept if block_idx == 0 else expanded_rows
                target_block.downsample[0].weight.data.copy_(source_block.downsample[0].weight.data[expanded_rows][:, downsample_in_idx])
                target_block.downsample[1].weight.data.copy_(source_block.downsample[1].weight.data[expanded_rows])
                target_block.downsample[1].bias.data.copy_(source_block.downsample[1].bias.data[expanded_rows])
                target_block.downsample[1].running_mean.copy_(source_block.downsample[1].running_mean[expanded_rows])
                target_block.downsample[1].running_var.copy_(source_block.downsample[1].running_var[expanded_rows])
        
        prev_kept = expanded_rows
    
    # Copy FC layer
    last_kept = torch.tensor(keep_indices['layer4'], dtype=torch.long, device=DEVICE)
    if last_kept.numel() > 0:
        fc_in_idx = torch.cat([torch.arange(p * 4, (p + 1) * 4, dtype=torch.long, device=DEVICE) for p in last_kept])
        if fc_in_idx.numel() > 0:
            target_model.fc.weight.data.copy_(source_model.fc.weight.data[:, fc_in_idx])
    target_model.fc.bias.data.copy_(source_model.fc.bias.data)
    
    return target_model

# -------------------------
# Training and evaluation
# -------------------------
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, train_loader, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for bidx, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        
        if bidx % LOG_INTERVAL == 0:
            print(f"    Epoch {epoch} Batch {bidx}/{len(train_loader)} - loss {running_loss/total:.4f} acc {correct/total:.4f}")
    
    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc

@torch.no_grad()
def evaluate(model, loader, dataset_name="", phase=""):
    """Evaluate model on a dataset"""
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    probs_list = []
    labels_list = []
    
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss_total += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        
        probs = torch.softmax(outputs, dim=1)
        probs_list.append(probs.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
    
    avg_loss = loss_total / total
    acc = correct / total
    
    # Compute AUC
    try:
        probs = np.concatenate(probs_list)
        labels = np.concatenate(labels_list)
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception as e:
        print(f"    AUC calculation failed: {e}")
        auc = float("nan")
    
    return avg_loss, acc, auc

# -------------------------
# Metrics computation
# -------------------------
def count_parameters(model):
    """Count total parameters in millions"""
    return sum(p.numel() for p in model.parameters()) / 1e6

def model_size_mb(model):
    """Compute model size in MB"""
    fd, tmp = tempfile.mkstemp(suffix=".pth")
    os.close(fd)
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / (1024**2)
    os.remove(tmp)
    return size

def compute_flops(model):
    """Compute FLOPs per image"""
    model.eval()
    try:
        inputs = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        macs = profile_macs(model, inputs)
        flops = macs * 2
        return float(flops)
    except Exception as e:
        print(f"    FLOPs calculation failed: {e}")
        return float("nan")

def inference_time_per_batch(model, loader, warmup=WARMUP_BATCHES, timed=TIMING_BATCHES):
    """Measure inference time per batch"""
    model.eval()
    use_cuda = DEVICE.type == "cuda"
    it = iter(loader)
    
    # Warmup
    try:
        for _ in range(warmup):
            imgs, _ = next(it)
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                _ = model(imgs)
            if use_cuda:
                torch.cuda.synchronize()
    except StopIteration:
        it = iter(loader)
    
    # Timed inference
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
            if use_cuda:
                torch.cuda.synchronize()
            batches_done += 1
            images_processed += imgs.size(0)
    except StopIteration:
        print(f"    Warning: Only {batches_done} batches processed")
    
    elapsed = time.time() - start
    avg_batch = elapsed / max(1, batches_done)
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if use_cuda else count_parameters(model) * 4.0 / (1024**2)
    
    return avg_batch, peak_mb

# -------------------------
# CodeCarbon utilities
# -------------------------
def start_energy_tracker(save_dir, project_name):
    """Start CodeCarbon tracker"""
    if not CODECARBON_AVAILABLE:
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "emissions.csv")
    
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except Exception as e:
            pass  # Silently ignore
    
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=save_dir,
        output_file="emissions.csv",
        measure_power_secs=30,
        save_to_file=True,
        log_level="error"  # Only show errors, not info
    )
    tracker.start()
    return tracker

def stop_energy_tracker(tracker, save_dir, project_name):
    """Stop tracker and extract metrics"""
    if tracker is None:
        return {
            "energy_kwh": float("nan"),
            "emissions_kg": float("nan"),
            "duration_s": float("nan")
        }
    
    try:
        emissions_val = tracker.stop()
    except Exception as e:
        print(f"    Error stopping tracker: {e}")
        emissions_val = None
    
    # Read CSV
    csv_path = os.path.join(save_dir, "emissions.csv")
    if not os.path.exists(csv_path):
        return {
            "energy_kwh": float(emissions_val) if emissions_val else float("nan"),
            "emissions_kg": float(emissions_val) if emissions_val else float("nan"),
            "duration_s": float("nan")
        }
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return {
                "energy_kwh": float("nan"),
                "emissions_kg": float("nan"),
                "duration_s": float("nan")
            }
        
        df_match = df[df["project_name"] == project_name]
        if df_match.shape[0] == 0:
            return {
                "energy_kwh": float("nan"),
                "emissions_kg": float("nan"),
                "duration_s": float("nan")
            }
        
        row = df_match.iloc[-1]
        energy_kwh = float(row.get("energy_consumed", float("nan")))
        emissions_kg = float(row.get("emissions", float("nan")))
        duration_s = float(row.get("duration", float("nan")))
        
        # Clean up CSV
        df = df[df["project_name"] != project_name]
        df.to_csv(csv_path, index=False)
        
        return {
            "energy_kwh": energy_kwh,
            "emissions_kg": emissions_kg,
            "duration_s": duration_s
        }
    except Exception as e:
        print(f"    Error reading emissions CSV: {e}")
        return {
            "energy_kwh": float("nan"),
            "emissions_kg": float("nan"),
            "duration_s": float("nan")
        }

# Removed EarlyStopping class - using fixed epochs for consistency

# -------------------------
# Main training functions
# -------------------------
def train_baseline_with_regularization(dataset_name, train_loader, val_loader, test_loader, 
                                      num_classes, save_dir, trial_num):
    """
    Phase 2: Train baseline model with L2 regularization and batch normalization (TRAINED SECOND)
    All models train for exactly FIXED_EPOCHS epochs
    """
    print("\n" + "="*80)
    print(f"PHASE 2: BASELINE WITH REGULARIZATION - {dataset_name.upper()} - TRIAL {trial_num}/{NUM_TRIALS}")
    print("="*80)
    
    # Build model with pretrained ImageNet weights (batch normalization is built into ResNet)
    model = build_baseline_resnet(num_classes)
    print(f"Model built with L2={WEIGHT_DECAY}, batch_norm=True, pretrained=ImageNet")
    
    # Optimizer with L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    print(f"Optimizer: Adam(lr={INITIAL_LR}, weight_decay={WEIGHT_DECAY})")
    
    # Track metrics and best model
    history = []
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Start energy tracking
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_baseline_training_trial{trial_num}")
    
    print(f"\nTraining for exactly {FIXED_EPOCHS} epochs (fixed schedule)")
    
    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, epoch)
        
        # Validate
        val_loss, val_acc, val_auc = evaluate(model, val_loader, dataset_name, "baseline_val")
        
        # Test (for per-epoch tracking)
        test_loss, test_acc, test_auc = evaluate(model, test_loader, dataset_name, "baseline_test")
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # Track best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"  ✓ New best validation accuracy: {val_acc:.4f}")
        
        # Record metrics for all epochs
        history.append({
            'trial': trial_num,
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_auc': test_auc,
        })
    
    # Stop energy tracking
    energy_metrics = stop_energy_tracker(tracker, save_dir, f"{dataset_name}_baseline_training_trial{trial_num}")
    print(f"\nTraining energy: {energy_metrics['energy_kwh']:.6f} kWh, emissions: {energy_metrics['emissions_kg']:.6f} kg")
    
    # Load best model based on validation accuracy
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch} (val_acc: {best_val_acc:.4f})")
    
    # Final evaluation on test set
    test_loss, test_acc, test_auc = evaluate(model, test_loader, dataset_name, "baseline_test")
    print(f"\nFinal Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
    
    # Compute additional metrics
    params_m = count_parameters(model)
    size_mb = model_size_mb(model)
    flops = compute_flops(model)
    inf_time, peak_ram = inference_time_per_batch(model, test_loader)
    
    print(f"Model size: {size_mb:.2f} MB, Params: {params_m:.2f}M")
    print(f"FLOPs: {flops/1e6:.2f}M, Inference: {inf_time*1000:.2f}ms/batch, Peak RAM: {peak_ram:.2f}MB")
    
    # Save model
    model_path = os.path.join(save_dir, f"{dataset_name}_baseline_trial{trial_num}_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to {model_path}")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(save_dir, f"{dataset_name}_baseline_trial{trial_num}_training_history.csv")
    history_df.to_csv(history_path, index=False)
    
    final_metrics = {
        'method': 'baseline_regularized',
        'trial': trial_num,
        'total_epochs': FIXED_EPOCHS,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'params_m': params_m,
        'model_size_mb': size_mb,
        'flops': flops,
        'flops_m': flops / 1e6,
        'inference_time_ms': inf_time * 1000,
        'peak_ram_mb': peak_ram,
        'training_energy_kwh': energy_metrics['energy_kwh'],
        'training_emissions_kg': energy_metrics['emissions_kg'],
        'training_duration_s': energy_metrics['duration_s'],
        'weight_decay': WEIGHT_DECAY,
        'channels_layer1': 64,
        'channels_layer2': 128,
        'channels_layer3': 256,
        'channels_layer4': 512,
    }
    
    return model, final_metrics

def train_with_progressive_pruning(dataset_name, train_loader, val_loader, test_loader, 
                                  num_classes, save_dir, trial_num):
    """
    Phase 1: Train with progressive channel pruning during training (TRAINED FIRST for easier debugging)
    Uses same regularization as baseline: L2, batch normalization
    Fixed schedule: 15 epochs total, prune at epochs 5, 8, 11, 14 (every 3 epochs after warmup)
    """
    print("\n" + "="*80)
    print(f"PHASE 1: PROGRESSIVE PRUNING DURING TRAINING - {dataset_name.upper()} - TRIAL {trial_num}/{NUM_TRIALS}")
    print("="*80)
    
    # Calculate pruning schedule
    # Warmup: epochs 1-2, then prune at 3, 6, 9, 12
    prune_epochs = [WARMUP_EPOCHS + i * EPOCHS_BETWEEN_PRUNES for i in range(1, NUM_PRUNE_STEPS + 1)]
    # Ensure we don't prune beyond FIXED_EPOCHS
    prune_epochs = [e for e in prune_epochs if e <= FIXED_EPOCHS]
    actual_prune_steps = len(prune_epochs)
    
    print(f"Fixed schedule: {FIXED_EPOCHS} epochs total")
    print(f"Warmup: {WARMUP_EPOCHS} epochs")
    print(f"Pruning at epochs: {prune_epochs} (every {EPOCHS_BETWEEN_PRUNES} epochs)")
    print(f"Total prune steps: {actual_prune_steps}")
    
    # Build initial model with pretrained ImageNet weights (same as baseline)
    model = build_baseline_resnet(num_classes)
    print(f"Model built with L2={WEIGHT_DECAY}, batch_norm=True, pretrained=ImageNet")
    
    # Optimizer with L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    current_lr = INITIAL_LR
    
    # Track current channel counts
    current_channels = {s: ORIGINAL_PLANES[i] for i, s in enumerate(STAGES)}
    
    # Track all metrics (including per-epoch val metrics) and best model
    all_metrics = []
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Start energy tracking
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_progressive_training_trial{trial_num}")
    
    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")
        
        # Check if this is a pruning epoch
        if epoch in prune_epochs:
            prune_step = prune_epochs.index(epoch) + 1
            print(f"\n*** PRUNING STEP {prune_step}/{actual_prune_steps} ***")
            
            # Compute importance for each stage
            print("Computing channel importance...")
            keep_indices = {}
            keep_ratio = 1.0 - PRUNE_PERCENT
            
            for stage_name in STAGES:
                importance = compute_channel_importance(model, stage_name, train_loader, 
                                                       max_batches=IMPORTANCE_CAL_BATCHES)
                keeps = select_channels_to_keep(importance, keep_ratio)
                keep_indices[stage_name] = keeps
                
                orig = current_channels[stage_name]
                kept = len(keeps)
                pruned = orig - kept
                print(f"  {stage_name}: {orig} → {kept} channels ({pruned} pruned, {kept/orig*100:.1f}% kept)")
                
                current_channels[stage_name] = kept
            
            # Build new pruned model (random initialization)
            new_stage_planes = [current_channels[s] for s in STAGES]
            pruned_model = build_pruned_resnet(new_stage_planes, num_classes)
            
            # Copy weights from important channels
            print("Copying weights to pruned model...")
            pruned_model = copy_weights_to_pruned_model(model, pruned_model, keep_indices)
            
            # Replace model
            del model
            model = pruned_model
            cleanup_memory()
            
            # Reduce learning rate
            current_lr *= LR_REDUCTION_AFTER_PRUNE
            optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=WEIGHT_DECAY)
            print(f"Learning rate reduced to {current_lr:.6f}")
            
            # Evaluate post-pruning
            val_loss, val_acc, val_auc = evaluate(model, val_loader, dataset_name, "post_prune")
            test_loss, test_acc, test_auc = evaluate(model, test_loader, dataset_name, "post_prune")
            
            print(f"Post-prune validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
            print(f"Post-prune test       - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
            
            # Save checkpoint
            ckpt_path = os.path.join(save_dir, f"{dataset_name}_progressive_trial{trial_num}_epoch{epoch}_postprune.pth")
            torch.save(model.state_dict(), ckpt_path)
            
            # Compute metrics
            params_m = count_parameters(model)
            size_mb = model_size_mb(model)
            flops = compute_flops(model)
            inf_time, peak_ram = inference_time_per_batch(model, test_loader)
            
            all_metrics.append({
                'method': 'progressive_pruning',
                'trial': trial_num,
                'epoch': epoch,
                'stage': f'post_prune_step_{prune_step}',
                'train_loss': None,
                'train_acc': None,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_auc': test_auc,
                'params_m': params_m,
                'model_size_mb': size_mb,
                'flops': flops,
                'flops_m': flops / 1e6,
                'inference_time_ms': inf_time * 1000,
                'peak_ram_mb': peak_ram,
                'channels_layer1': current_channels['layer1'],
                'channels_layer2': current_channels['layer2'],
                'channels_layer3': current_channels['layer3'],
                'channels_layer4': current_channels['layer4'],
                'learning_rate': current_lr,
            })
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, epoch)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, dataset_name, "progressive_val")
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # Track best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"  ✓ New best validation accuracy: {val_acc:.4f}")
        
        # Record metrics for all epochs (including pruning epochs)
        test_loss, test_acc, test_auc = evaluate(model, test_loader, dataset_name, "progressive_test")
        params_m = count_parameters(model)
        size_mb = model_size_mb(model)
        flops = compute_flops(model)
        inf_time, peak_ram = inference_time_per_batch(model, test_loader)
        
        all_metrics.append({
            'method': 'progressive_pruning',
            'trial': trial_num,
            'epoch': epoch,
            'stage': 'after_training' if epoch in prune_epochs else 'training',
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'params_m': params_m,
            'model_size_mb': size_mb,
            'flops': flops,
            'flops_m': flops / 1e6,
            'inference_time_ms': inf_time * 1000,
            'peak_ram_mb': peak_ram,
            'channels_layer1': current_channels['layer1'],
            'channels_layer2': current_channels['layer2'],
            'channels_layer3': current_channels['layer3'],
            'channels_layer4': current_channels['layer4'],
            'learning_rate': current_lr,
        })
    
    # Stop energy tracking
    energy_metrics = stop_energy_tracker(tracker, save_dir, f"{dataset_name}_progressive_training_trial{trial_num}")
    print(f"\nTraining energy: {energy_metrics['energy_kwh']:.6f} kWh, emissions: {energy_metrics['emissions_kg']:.6f} kg")
    
    # Load best model based on validation accuracy
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch} (val_acc: {best_val_acc:.4f})")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    test_loss, test_acc, test_auc = evaluate(model, test_loader, dataset_name, "progressive_final")
    print(f"Final Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
    
    params_m = count_parameters(model)
    size_mb = model_size_mb(model)
    flops = compute_flops(model)
    inf_time, peak_ram = inference_time_per_batch(model, test_loader)
    
    print(f"Model size: {size_mb:.2f} MB, Params: {params_m:.2f}M")
    print(f"FLOPs: {flops/1e6:.2f}M, Inference: {inf_time*1000:.2f}ms/batch, Peak RAM: {peak_ram:.2f}MB")
    print(f"Final channel counts: L1={current_channels['layer1']}, L2={current_channels['layer2']}, "
          f"L3={current_channels['layer3']}, L4={current_channels['layer4']}")
    
    # Save final model
    model_path = os.path.join(save_dir, f"{dataset_name}_progressive_trial{trial_num}_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved final model to {model_path}")
    
    # Add final metrics with energy
    all_metrics.append({
        'method': 'progressive_pruning',
        'trial': trial_num,
        'total_epochs': FIXED_EPOCHS,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'stage': 'final',
        'train_loss': None,
        'train_acc': None,
        'val_loss': None,
        'val_acc': None,
        'val_auc': None,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'params_m': params_m,
        'model_size_mb': size_mb,
        'flops': flops,
        'flops_m': flops / 1e6,
        'inference_time_ms': inf_time * 1000,
        'peak_ram_mb': peak_ram,
        'training_energy_kwh': energy_metrics['energy_kwh'],
        'training_emissions_kg': energy_metrics['emissions_kg'],
        'training_duration_s': energy_metrics['duration_s'],
        'channels_layer1': current_channels['layer1'],
        'channels_layer2': current_channels['layer2'],
        'channels_layer3': current_channels['layer3'],
        'channels_layer4': current_channels['layer4'],
        'total_prune_steps': actual_prune_steps,
        'epochs_between_prunes': EPOCHS_BETWEEN_PRUNES,
        'weight_decay': WEIGHT_DECAY,
    })
    
    # Save all metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(save_dir, f"{dataset_name}_progressive_trial{trial_num}_all_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved progressive metrics to {metrics_path}")
    
    return model, all_metrics[-1]

def train_prune_then_finetune(dataset_name, train_loader, val_loader, test_loader, 
                               num_classes, save_dir, trial_num):
    """
    Phase 3: Prune ONCE at initialization, then train for 15 epochs (PRUNE-THEN-TRAIN)
    - Load pretrained ResNet50
    - Perform one-shot pruning to target dimensions using importance scoring
    - Train pruned model for 15 epochs
    
    This tests: "Is it better to prune once upfront, or progressively during training?"
    """
    print("\n" + "="*80)
    print(f"PHASE 3: PRUNE-THEN-TRAIN - {dataset_name.upper()} - TRIAL {trial_num}/{NUM_TRIALS}")
    print("="*80)
    
    print(f"Fixed schedule: {FIXED_EPOCHS} epochs total")
    print(f"One-shot pruning at initialization to target dimensions:")
    print(f"  Target channels: {TARGET_PRUNED_CHANNELS}")
    
    # Build initial model with pretrained ImageNet weights
    model = build_baseline_resnet(num_classes)
    print(f"Model built with L2={WEIGHT_DECAY}, batch_norm=True, pretrained=ImageNet")
    
    # Start energy tracking (includes pruning time)
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_prune_then_train_trial{trial_num}")
    
    # ONE-SHOT PRUNING at initialization
    print("\n*** ONE-SHOT PRUNING ***")
    print("Computing channel importance for all stages...")
    
    keep_indices = {}
    current_channels = {s: ORIGINAL_PLANES[i] for i, s in enumerate(STAGES)}
    
    for stage_name in STAGES:
        # Compute importance using calibration data
        importance = compute_channel_importance(model, stage_name, train_loader, 
                                               max_batches=IMPORTANCE_CAL_BATCHES)
        
        # Select top K channels to keep (K = target dimension)
        target_k = TARGET_PRUNED_CHANNELS[stage_name]
        keeps = select_channels_to_keep(importance, keep_ratio=target_k/current_channels[stage_name])
        keep_indices[stage_name] = keeps
        
        orig = current_channels[stage_name]
        kept = len(keeps)
        pruned = orig - kept
        print(f"  {stage_name}: {orig} → {kept} channels ({pruned} pruned, {kept/orig*100:.1f}% kept)")
        
        current_channels[stage_name] = kept
    
    # Build pruned model
    new_stage_planes = [current_channels[s] for s in STAGES]
    pruned_model = build_pruned_resnet(new_stage_planes, num_classes)
    
    # Copy weights from important channels
    print("Copying weights to pruned model...")
    pruned_model = copy_weights_to_pruned_model(model, pruned_model, keep_indices)
    
    # Replace model
    del model
    model = pruned_model
    cleanup_memory()
    
    print(f"Pruned model created. Now training for {FIXED_EPOCHS} epochs...")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    
    # Track all metrics and best model
    all_metrics = []
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Train for fixed epochs
    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, epoch)
        
        # Validate
        val_loss, val_acc, val_auc = evaluate(model, val_loader, dataset_name, "prune_then_train_val")
        
        # Test
        test_loss, test_acc, test_auc = evaluate(model, test_loader, dataset_name, "prune_then_train_test")
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # Track best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"  ✓ New best validation accuracy: {val_acc:.4f}")
        
        # Compute metrics
        params_m = count_parameters(model)
        size_mb = model_size_mb(model)
        flops = compute_flops(model)
        inf_time, peak_ram = inference_time_per_batch(model, test_loader)
        
        # Record metrics for all epochs
        all_metrics.append({
            'method': 'prune_then_train',
            'trial': trial_num,
            'epoch': epoch,
            'stage': 'training',
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'params_m': params_m,
            'model_size_mb': size_mb,
            'flops': flops,
            'flops_m': flops / 1e6,
            'inference_time_ms': inf_time * 1000,
            'peak_ram_mb': peak_ram,
            'channels_layer1': current_channels['layer1'],
            'channels_layer2': current_channels['layer2'],
            'channels_layer3': current_channels['layer3'],
            'channels_layer4': current_channels['layer4'],
            'learning_rate': INITIAL_LR,
        })
    
    # Stop energy tracking
    energy_metrics = stop_energy_tracker(tracker, save_dir, f"{dataset_name}_prune_then_train_trial{trial_num}")
    print(f"\nTotal energy (pruning + training): {energy_metrics['energy_kwh']:.6f} kWh, emissions: {energy_metrics['emissions_kg']:.6f} kg")
    
    # Load best model based on validation accuracy
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch} (val_acc: {best_val_acc:.4f})")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    test_loss, test_acc, test_auc = evaluate(model, test_loader, dataset_name, "prune_then_train_final")
    print(f"Final Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
    
    params_m = count_parameters(model)
    size_mb = model_size_mb(model)
    flops = compute_flops(model)
    inf_time, peak_ram = inference_time_per_batch(model, test_loader)
    
    print(f"Model size: {size_mb:.2f} MB, Params: {params_m:.2f}M")
    print(f"FLOPs: {flops/1e6:.2f}M, Inference: {inf_time*1000:.2f}ms/batch, Peak RAM: {peak_ram:.2f}MB")
    print(f"Final channel counts: L1={current_channels['layer1']}, L2={current_channels['layer2']}, "
          f"L3={current_channels['layer3']}, L4={current_channels['layer4']}")
    
    # Save final model
    model_path = os.path.join(save_dir, f"{dataset_name}_prune_then_train_trial{trial_num}_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved final model to {model_path}")
    
    # Add final metrics with energy
    all_metrics.append({
        'method': 'prune_then_train',
        'trial': trial_num,
        'total_epochs': FIXED_EPOCHS,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'stage': 'final',
        'train_loss': None,
        'train_acc': None,
        'val_loss': None,
        'val_acc': None,
        'val_auc': None,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'params_m': params_m,
        'model_size_mb': size_mb,
        'flops': flops,
        'flops_m': flops / 1e6,
        'inference_time_ms': inf_time * 1000,
        'peak_ram_mb': peak_ram,
        'training_energy_kwh': energy_metrics['energy_kwh'],
        'training_emissions_kg': energy_metrics['emissions_kg'],
        'training_duration_s': energy_metrics['duration_s'],
        'channels_layer1': current_channels['layer1'],
        'channels_layer2': current_channels['layer2'],
        'channels_layer3': current_channels['layer3'],
        'channels_layer4': current_channels['layer4'],
        'weight_decay': WEIGHT_DECAY,
    })
    
    # Save all metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(save_dir, f"{dataset_name}_prune_then_train_trial{trial_num}_all_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved prune-then-train metrics to {metrics_path}")
    
    return model, all_metrics[-1]

# -------------------------
# Main execution
# -------------------------
def process_dataset(dataset_name, cfg):
    """Process a single dataset with both training approaches across multiple trials"""
    print("\n" + "#"*100)
    print(f"# PROCESSING DATASET: {dataset_name.upper()}")
    print("#"*100)
    
    # Setup directories
    save_dir = os.path.join(SAVE_DIR_BASE, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading dataset from {cfg['path']}")
    train_loader, val_loader, test_loader, num_classes = make_loaders(cfg['path'], cfg['batch_size'])
    print(f"Number of classes: {num_classes}")
    
    all_baseline_metrics = []
    all_progressive_metrics = []
    all_prune_then_train_metrics = []
    
    # Run multiple trials
    for trial in range(1, NUM_TRIALS + 1):
        print("\n" + "~"*100)
        print(f"~ TRIAL {trial}/{NUM_TRIALS}")
        print("~"*100)
        
        # Set different seed for each trial
        trial_seed = SEED + trial * 100
        set_seed(trial_seed, deterministic=True)
        
        # Phase 1: Progressive pruning (train first for easier debugging)
        progressive_model, progressive_metrics = train_with_progressive_pruning(
            dataset_name, train_loader, val_loader, test_loader, num_classes, save_dir, trial
        )
        all_progressive_metrics.append(progressive_metrics)
        
        # Clean up
        del progressive_model
        cleanup_memory()
        
        # Phase 2: Baseline with regularization
        baseline_model, baseline_metrics = train_baseline_with_regularization(
            dataset_name, train_loader, val_loader, test_loader, num_classes, save_dir, trial
        )
        all_baseline_metrics.append(baseline_metrics)
        
        # Clean up
        del baseline_model
        cleanup_memory()
        
        # Phase 3: Prune-then-train
        prune_then_train_model, prune_then_train_metrics = train_prune_then_finetune(
            dataset_name, train_loader, val_loader, test_loader, num_classes, save_dir, trial
        )
        all_prune_then_train_metrics.append(prune_then_train_metrics)
        
        # Clean up
        del prune_then_train_model
        cleanup_memory()
        
        print(f"\n✓ Trial {trial} completed")
    
    # Combine all trials
    all_metrics_df = pd.DataFrame(all_baseline_metrics + all_progressive_metrics + all_prune_then_train_metrics)
    all_metrics_path = os.path.join(save_dir, f"{dataset_name}_all_trials_metrics.csv")
    all_metrics_df.to_csv(all_metrics_path, index=False)
    print(f"\nSaved all trials metrics to {all_metrics_path}")
    
    # Compute statistics across trials
    baseline_df = pd.DataFrame(all_baseline_metrics)
    progressive_df = pd.DataFrame(all_progressive_metrics)
    prune_then_train_df = pd.DataFrame(all_prune_then_train_metrics)
    
    # Create summary with mean and std
    summary_rows = []
    
    # Baseline summary
    baseline_summary = {
        'method': 'baseline_regularized',
        'dataset': dataset_name,
        'num_trials': NUM_TRIALS,
        'fixed_epochs': FIXED_EPOCHS,
    }
    for col in ['test_acc', 'test_auc', 'test_loss', 'params_m', 'model_size_mb', 'flops_m', 
                'inference_time_ms', 'training_energy_kwh', 'training_emissions_kg', 'best_epoch', 'best_val_acc']:
        if col in baseline_df.columns:
            baseline_summary[f'{col}_mean'] = baseline_df[col].mean()
            baseline_summary[f'{col}_std'] = baseline_df[col].std()
            baseline_summary[f'{col}_min'] = baseline_df[col].min()
            baseline_summary[f'{col}_max'] = baseline_df[col].max()
    summary_rows.append(baseline_summary)
    
    # Progressive summary
    progressive_summary = {
        'method': 'progressive_pruning',
        'dataset': dataset_name,
        'num_trials': NUM_TRIALS,
        'fixed_epochs': FIXED_EPOCHS,
    }
    for col in ['test_acc', 'test_auc', 'test_loss', 'params_m', 'model_size_mb', 'flops_m', 
                'inference_time_ms', 'training_energy_kwh', 'training_emissions_kg', 'best_epoch', 'best_val_acc',
                'channels_layer1', 'channels_layer2', 'channels_layer3', 'channels_layer4']:
        if col in progressive_df.columns:
            progressive_summary[f'{col}_mean'] = progressive_df[col].mean()
            progressive_summary[f'{col}_std'] = progressive_df[col].std()
            progressive_summary[f'{col}_min'] = progressive_df[col].min()
            progressive_summary[f'{col}_max'] = progressive_df[col].max()
    summary_rows.append(progressive_summary)
    
    # Prune-then-train summary
    prune_then_train_summary = {
        'method': 'prune_then_train',
        'dataset': dataset_name,
        'num_trials': NUM_TRIALS,
        'fixed_epochs': FIXED_EPOCHS,
    }
    for col in ['test_acc', 'test_auc', 'test_loss', 'params_m', 'model_size_mb', 'flops_m', 
                'inference_time_ms', 'training_energy_kwh', 'training_emissions_kg', 'best_epoch', 'best_val_acc',
                'channels_layer1', 'channels_layer2', 'channels_layer3', 'channels_layer4']:
        if col in prune_then_train_df.columns:
            prune_then_train_summary[f'{col}_mean'] = prune_then_train_df[col].mean()
            prune_then_train_summary[f'{col}_std'] = prune_then_train_df[col].std()
            prune_then_train_summary[f'{col}_min'] = prune_then_train_df[col].min()
            prune_then_train_summary[f'{col}_max'] = prune_then_train_df[col].max()
    summary_rows.append(prune_then_train_summary)
    
    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(save_dir, f"{dataset_name}_summary_statistics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary statistics to {summary_path}")
    
    # Print comparison table
    print("\n" + "="*140)
    print(f"SUMMARY COMPARISON - {dataset_name.upper()} ({NUM_TRIALS} trials, {FIXED_EPOCHS} epochs each)")
    print("="*140)
    print(f"{'Metric':<30} {'Baseline (Mean±Std)':<35} {'Progressive (Mean±Std)':<35} {'Prune-Then-Train (Mean±Std)':<35}")
    print("-"*140)
    
    metrics_to_compare = [
        ('test_acc', 'Test Accuracy', '{:.4f}', True),
        ('test_auc', 'Test AUC', '{:.4f}', True),
        ('test_loss', 'Test Loss', '{:.4f}', False),
        ('params_m', 'Parameters (M)', '{:.2f}', False),
        ('model_size_mb', 'Model Size (MB)', '{:.2f}', False),
        ('flops_m', 'FLOPs (M)', '{:.2f}', False),
        ('inference_time_ms', 'Inference Time (ms)', '{:.2f}', False),
        ('training_energy_kwh', 'Training Energy (kWh)', '{:.6f}', False),
        ('training_emissions_kg', 'Training Emissions (kg)', '{:.6f}', False),
    ]
    
    for key, name, fmt, higher_better in metrics_to_compare:
        base_mean = baseline_summary.get(f'{key}_mean', float('nan'))
        base_std = baseline_summary.get(f'{key}_std', float('nan'))
        prog_mean = progressive_summary.get(f'{key}_mean', float('nan'))
        prog_std = progressive_summary.get(f'{key}_std', float('nan'))
        ptt_mean = prune_then_train_summary.get(f'{key}_mean', float('nan'))
        ptt_std = prune_then_train_summary.get(f'{key}_std', float('nan'))
        
        base_str = f"{fmt.format(base_mean)} ± {fmt.format(base_std)}"
        prog_str = f"{fmt.format(prog_mean)} ± {fmt.format(prog_std)}"
        ptt_str = f"{fmt.format(ptt_mean)} ± {fmt.format(ptt_std)}"
        
        print(f"{name:<30} {base_str:<35} {prog_str:<35} {ptt_str:<35}")
    
    # Channel reduction info
    print(f"\n{'Final Channel Counts':<30}")
    print(f"{'Stage':<15} {'Original':<15} {'Progressive':<25} {'Prune-Then-Train':<25}")
    print("-"*80)
    for i, stage in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
        orig = ORIGINAL_PLANES[i]
        prog_mean = progressive_summary.get(f'channels_{stage}_mean', float('nan'))
        prog_std = progressive_summary.get(f'channels_{stage}_std', float('nan'))
        ptt_mean = prune_then_train_summary.get(f'channels_{stage}_mean', float('nan'))
        ptt_std = prune_then_train_summary.get(f'channels_{stage}_std', float('nan'))
        
        prog_reduction = ((orig - prog_mean) / orig * 100) if not math.isnan(prog_mean) else 0
        ptt_reduction = ((orig - ptt_mean) / orig * 100) if not math.isnan(ptt_mean) else 0
        
        prog_str = f"{prog_mean:.1f}±{prog_std:.1f} ({prog_reduction:.1f}% reduced)"
        ptt_str = f"{ptt_mean:.1f}±{ptt_std:.1f} ({ptt_reduction:.1f}% reduced)"
        
        print(f"{stage:<15} {orig:<15} {prog_str:<25} {ptt_str:<25}")
    
    print("="*140)
    
    return summary_df

def main():
    """Main entry point"""
    set_seed(SEED, deterministic=True)
    
    print("="*100)
    print("PROGRESSIVE PRUNING DURING TRAINING EXPERIMENT")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {SEED}")
    print(f"  Number of Trials: {NUM_TRIALS}")
    print(f"\n  Training Configuration (SAME for all methods):")
    print(f"    Fixed Epochs: {FIXED_EPOCHS}")
    print(f"    Initial LR: {INITIAL_LR}")
    print(f"    Weight Decay (L2): {WEIGHT_DECAY}")
    print(f"    Batch Normalization: True (built-in to ResNet)")
    print(f"    Pretrained: ImageNet weights")
    print(f"\n  Method 1: Baseline (No Pruning)")
    print(f"    Full ResNet50 architecture")
    print(f"    Channels: [64, 128, 256, 512]")
    print(f"\n  Method 2: Progressive Pruning (During Training)")
    print(f"    Warmup Epochs: {WARMUP_EPOCHS}")
    print(f"    Epochs Between Prunes: {EPOCHS_BETWEEN_PRUNES}")
    print(f"    Number of Prune Steps: {NUM_PRUNE_STEPS}")
    print(f"    Prune Percent per Step: {PRUNE_PERCENT*100}%")
    print(f"    Pruning Schedule: epochs {[WARMUP_EPOCHS + i * EPOCHS_BETWEEN_PRUNES for i in range(1, NUM_PRUNE_STEPS + 1)]}")
    print(f"    LR Reduction After Prune: {LR_REDUCTION_AFTER_PRUNE}")
    print(f"    Final Target Channels: ~{[TARGET_PRUNED_CHANNELS[s] for s in STAGES]}")
    print(f"\n  Method 3: Prune-Then-Train (One-Shot Pruning)")
    print(f"    One-shot pruning at initialization")
    print(f"    Target Channels: {[TARGET_PRUNED_CHANNELS[s] for s in STAGES]}")
    print(f"    Train pruned model for {FIXED_EPOCHS} epochs")
    print(f"\n  Importance Calibration: {IMPORTANCE_CAL_BATCHES} batches")
    print(f"  Datasets: {list(DATASETS.keys())}")
    print(f"  Save Directory: {SAVE_DIR_BASE}")
    
    if not CODECARBON_AVAILABLE:
        print("\n  WARNING: CodeCarbon not available - energy metrics will be NaN")
    
    os.makedirs(SAVE_DIR_BASE, exist_ok=True)
    
    # Process each dataset
    all_summaries = []
    for dataset_name, cfg in DATASETS.items():
        try:
            summary = process_dataset(dataset_name, cfg)
            all_summaries.append(summary)
            print(f"\n✓ Successfully completed {dataset_name}")
        except Exception as e:
            print(f"\n✗ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined summaries
    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_path = os.path.join(SAVE_DIR_BASE, "all_datasets_summary.csv")
        combined_summary.to_csv(combined_path, index=False)
        print(f"\n\nSaved combined summary to {combined_path}")
    
    print("\n" + "="*100)
    print("EXPERIMENT COMPLETED")
    print("="*100)
    print("\nKey Takeaways:")
    print("  - All three methods trained for exactly {} epochs".format(FIXED_EPOCHS))
    print("  - All methods used same regularization: L2={}, batch_norm=True, pretrained=ImageNet".format(WEIGHT_DECAY))
    print("  - All methods saved BEST validation accuracy model (not last epoch)")
    print("\n  Method 1 - Baseline:")
    print("    Full ResNet50, no pruning")
    print("\n  Method 2 - Progressive Pruning:")
    print("    Pruned {} times at epochs {}".format(
        NUM_PRUNE_STEPS, 
        [WARMUP_EPOCHS + i * EPOCHS_BETWEEN_PRUNES for i in range(1, NUM_PRUNE_STEPS + 1)]
    ))
    print("    Gradual adaptation during training")
    print("\n  Method 3 - Prune-Then-Train:")
    print("    One-shot pruning at initialization")
    print("    Target architecture: {}".format([TARGET_PRUNED_CHANNELS[s] for s in STAGES]))
    print("\n  Results are averaged over {} trials for statistical reliability".format(NUM_TRIALS))
    print("="*100)

if __name__ == "__main__":
    main()