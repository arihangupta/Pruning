#!/usr/bin/env python3
"""
Four-method pruning comparison with minimal output
"""

import os, time, math, random, tempfile, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import Bottleneck
from torchvision import models, transforms as T
from sklearn.metrics import roc_auc_score
from torchprofile import profile_macs
import psutil, gc

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False

SAVE_DIR_BASE = "/home/arihangupta/Pruning/dinov2/Pruning/progressive_pruning_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224

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

INITIAL_LR = 1e-3
L1_LAMBDA = 1e-4
FIXED_EPOCHS = 15
WARMUP_EPOCHS = 2
EPOCHS_BETWEEN_PRUNES = 3
NUM_PRUNE_STEPS = 4
PRUNE_PERCENT = 0.10
LR_REDUCTION_AFTER_PRUNE = 0.5
NUM_TRIALS = 1
IMPORTANCE_CAL_BATCHES = 50
PRUNE_THEN_TRAIN_TARGETS = {"layer1": 42, "layer2": 84, "layer3": 168, "layer4": 336}
TIMING_BATCHES = 100
WARMUP_BATCHES = 5
STAGES = ["layer1", "layer2", "layer3", "layer4"]
ORIGINAL_PLANES = [64, 128, 256, 512]

def set_seed(s=SEED, deterministic=True):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED, deterministic=True)

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

class NumpyMemmapDataset(Dataset):
    def __init__(self, imgs_np, labels_np, img_size=224):
        self.imgs = imgs_np
        self.labels = labels_np
        self.img_size = img_size
        self.base_tfms = T.Compose([T.ToPILImage(), T.Resize((img_size, img_size)), T.ToTensor()])
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
    
    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE)
    val_ds = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE)
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    num_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    return train_loader, val_loader, test_loader, num_classes

class CustomResNet(nn.Module):
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
    base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = CustomResNet(block=Bottleneck, layers=[3,4,6,3], stage_planes=[64,128,256,512],
                        num_classes=num_classes, random_init=False)
    
    model.conv1.weight.data.copy_(base_model.conv1.weight.data)
    model.bn1.weight.data.copy_(base_model.bn1.weight.data)
    model.bn1.bias.data.copy_(base_model.bn1.bias.data)
    model.bn1.running_mean.copy_(base_model.bn1.running_mean)
    model.bn1.running_var.copy_(base_model.bn1.running_var)
    
    for stage_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        base_stage = getattr(base_model, stage_name)
        model_stage = getattr(model, stage_name)
        model_stage.load_state_dict(base_stage.state_dict())
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

def build_pruned_resnet(stage_planes, num_classes):
    stage_planes = [max(1, int(p)) for p in stage_planes]
    return CustomResNet(block=Bottleneck, layers=[3,4,6,3], stage_planes=stage_planes,
                       num_classes=num_classes, random_init=True).to(DEVICE)

def stage_orig_channels(model, stage_name):
    first_block = next(getattr(model, stage_name).children())
    return first_block.conv1.out_channels

def compute_channel_importance(model, stage_name, calib_loader, max_batches=IMPORTANCE_CAL_BATCHES):
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
                idx0 = p * expansion
                idx1 = (p + 1) * expansion
                part = act_flat[idx0:idx1]
                act_norms[p] += torch.norm(part)
        
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
    
    with torch.no_grad():
        importance = act_norms * grad_norms * weight_l1
        importance_np = importance.cpu().numpy()
    
    return importance_np

def select_channels_to_keep(importance_scores, keep_ratio=None, target_channels=None):
    num_channels = len(importance_scores)
    
    if target_channels is not None:
        num_keep = max(1, min(target_channels, num_channels))
    elif keep_ratio is not None:
        num_keep = max(1, int(num_channels * keep_ratio))
    else:
        raise ValueError("Must specify either keep_ratio or target_channels")
    
    if num_keep >= num_channels:
        return np.arange(num_channels)
    
    keep_indices = np.argsort(importance_scores)[-num_keep:]
    return np.sort(keep_indices)

def copy_weights_to_pruned_model(source_model, target_model, keep_indices):
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
        
        expanded_rows = torch.cat([torch.arange(p * 4, (p + 1) * 4, dtype=torch.long, device=DEVICE) for p in kept])
        
        for block_idx, (source_block, target_block) in enumerate(zip(source_stage.children(), target_stage.children())):
            if block_idx == 0:
                in_idx = prev_kept
            else:
                in_idx = expanded_rows
            
            out_idx = kept
            
            target_block.conv1.weight.data.copy_(source_block.conv1.weight.data[out_idx][:, in_idx])
            target_block.bn1.weight.data.copy_(source_block.bn1.weight.data[out_idx])
            target_block.bn1.bias.data.copy_(source_block.bn1.bias.data[out_idx])
            target_block.bn1.running_mean.copy_(source_block.bn1.running_mean[out_idx])
            target_block.bn1.running_var.copy_(source_block.bn1.running_var[out_idx])
            
            target_block.conv2.weight.data.copy_(source_block.conv2.weight.data[out_idx][:, out_idx])
            target_block.bn2.weight.data.copy_(source_block.bn2.weight.data[out_idx])
            target_block.bn2.bias.data.copy_(source_block.bn2.bias.data[out_idx])
            target_block.bn2.running_mean.copy_(source_block.bn2.running_mean[out_idx])
            target_block.bn2.running_var.copy_(source_block.bn2.running_var[out_idx])
            
            target_block.conv3.weight.data.copy_(source_block.conv3.weight.data[expanded_rows][:, out_idx])
            target_block.bn3.weight.data.copy_(source_block.bn3.weight.data[expanded_rows])
            target_block.bn3.bias.data.copy_(source_block.bn3.bias.data[expanded_rows])
            target_block.bn3.running_mean.copy_(source_block.bn3.running_mean[expanded_rows])
            target_block.bn3.running_var.copy_(source_block.bn3.running_var[expanded_rows])
            
            if source_block.downsample is not None and target_block.downsample is not None:
                downsample_in_idx = prev_kept if block_idx == 0 else expanded_rows
                target_block.downsample[0].weight.data.copy_(source_block.downsample[0].weight.data[expanded_rows][:, downsample_in_idx])
                target_block.downsample[1].weight.data.copy_(source_block.downsample[1].weight.data[expanded_rows])
                target_block.downsample[1].bias.data.copy_(source_block.downsample[1].bias.data[expanded_rows])
                target_block.downsample[1].running_mean.copy_(source_block.downsample[1].running_mean[expanded_rows])
                target_block.downsample[1].running_var.copy_(source_block.downsample[1].running_var[expanded_rows])
        
        prev_kept = expanded_rows
    
    last_kept = torch.tensor(keep_indices['layer4'], dtype=torch.long, device=DEVICE)
    if last_kept.numel() > 0:
        fc_in_idx = torch.cat([torch.arange(p * 4, (p + 1) * 4, dtype=torch.long, device=DEVICE) for p in last_kept])
        if fc_in_idx.numel() > 0:
            target_model.fc.weight.data.copy_(source_model.fc.weight.data[:, fc_in_idx])
    target_model.fc.bias.data.copy_(source_model.fc.bias.data)
    
    return target_model

criterion = nn.CrossEntropyLoss()

def compute_l1_loss(model, lambda_l1):
    l1_loss = 0.0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

def train_one_epoch(model, train_loader, optimizer, epoch, use_l1_reg=True, lambda_l1=L1_LAMBDA):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for bidx, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        if use_l1_reg:
            loss = loss + compute_l1_loss(model, lambda_l1)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    
    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc

@torch.no_grad()
def evaluate(model, loader, dataset_name="", phase=""):
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
    
    try:
        probs = np.concatenate(probs_list)
        labels = np.concatenate(labels_list)
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except:
        auc = float("nan")
    
    return avg_loss, acc, auc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def model_size_mb(model):
    fd, tmp = tempfile.mkstemp(suffix=".pth")
    os.close(fd)
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / (1024**2)
    os.remove(tmp)
    return size

def compute_flops(model):
    model.eval()
    try:
        inputs = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        macs = profile_macs(model, inputs)
        flops = macs * 2
        return float(flops)
    except:
        return float("nan")

def inference_time_per_batch(model, loader, warmup=WARMUP_BATCHES, timed=TIMING_BATCHES):
    model.eval()
    use_cuda = DEVICE.type == "cuda"
    it = iter(loader)
    
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
    
    elapsed = time.time() - start
    avg_batch = elapsed / max(1, batches_done)
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if use_cuda else count_parameters(model) * 4.0 / (1024**2)
    
    return avg_batch, peak_mb

def start_energy_tracker(save_dir, project_name):
    if not CODECARBON_AVAILABLE:
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "emissions.csv")
    
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except:
            pass
    
    tracker = EmissionsTracker(project_name=project_name, output_dir=save_dir, output_file="emissions.csv",
                             measure_power_secs=30, save_to_file=True, log_level="error")
    tracker.start()
    return tracker

def stop_energy_tracker(tracker, save_dir, project_name):
    if tracker is None:
        return {"energy_kwh": float("nan"), "emissions_kg": float("nan"), "duration_s": float("nan")}
    
    try:
        emissions_val = tracker.stop()
    except:
        emissions_val = None
    
    csv_path = os.path.join(save_dir, "emissions.csv")
    if not os.path.exists(csv_path):
        return {"energy_kwh": float(emissions_val) if emissions_val else float("nan"),
                "emissions_kg": float(emissions_val) if emissions_val else float("nan"),
                "duration_s": float("nan")}
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return {"energy_kwh": float("nan"), "emissions_kg": float("nan"), "duration_s": float("nan")}
        
        df_match = df[df["project_name"] == project_name]
        if df_match.shape[0] == 0:
            return {"energy_kwh": float("nan"), "emissions_kg": float("nan"), "duration_s": float("nan")}
        
        row = df_match.iloc[-1]
        energy_kwh = float(row.get("energy_consumed", float("nan")))
        emissions_kg = float(row.get("emissions", float("nan")))
        duration_s = float(row.get("duration", float("nan")))
        
        df = df[df["project_name"] != project_name]
        df.to_csv(csv_path, index=False)
        
        return {"energy_kwh": energy_kwh, "emissions_kg": emissions_kg, "duration_s": duration_s}
    except:
        return {"energy_kwh": float("nan"), "emissions_kg": float("nan"), "duration_s": float("nan")}

def train_baseline_with_regularization(dataset_name, train_loader, val_loader, test_loader, 
                                      num_classes, save_dir, trial_num):
    print(f"\nMethod 1: Baseline+L1 | {dataset_name} trial {trial_num}")
    
    model = build_baseline_resnet(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=0.0)
    history = []
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_baseline_training_trial{trial_num}")
    
    for epoch in range(1, FIXED_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, epoch, True, L1_LAMBDA)
        val_loss, val_acc, val_auc = evaluate(model, val_loader)
        test_loss, test_acc, test_auc = evaluate(model, test_loader)
        
        print(f"  Epoch {epoch}: val_acc={val_acc:.4f} val_auc={val_auc:.4f} | test_acc={test_acc:.4f} test_auc={test_auc:.4f}")
        
        history.append({'trial': trial_num, 'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                       'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc,
                       'test_loss': test_loss, 'test_acc': test_acc, 'test_auc': test_auc})
        
        # Save incrementally for live monitoring
        pd.DataFrame(history).to_csv(os.path.join(save_dir, f"{dataset_name}_baseline_trial{trial_num}_training_history.csv"), index=False)
    
    energy_metrics = stop_energy_tracker(tracker, save_dir, f"{dataset_name}_baseline_training_trial{trial_num}")
    test_loss, test_acc, test_auc = evaluate(model, test_loader)
    params_m = count_parameters(model)
    size_mb = model_size_mb(model)
    flops = compute_flops(model)
    inf_time, peak_ram = inference_time_per_batch(model, test_loader)
    
    print(f"  Final: acc={test_acc:.4f} auc={test_auc:.4f} params={params_m:.2f}M size={size_mb:.2f}MB")
    
    torch.save(model.state_dict(), os.path.join(save_dir, f"{dataset_name}_baseline_trial{trial_num}_final.pth"))
    
    return model, {
        'method': 'baseline_l1_regularized', 'trial': trial_num, 'total_epochs': FIXED_EPOCHS,
        'test_loss': test_loss, 'test_acc': test_acc, 'test_auc': test_auc,
        'params_m': params_m, 'model_size_mb': size_mb, 'flops': flops, 'flops_m': flops/1e6,
        'inference_time_ms': inf_time*1000, 'peak_ram_mb': peak_ram,
        'training_energy_kwh': energy_metrics['energy_kwh'], 'training_emissions_kg': energy_metrics['emissions_kg'],
        'training_duration_s': energy_metrics['duration_s'], 'l1_lambda': L1_LAMBDA, 'use_regularization': True,
        'channels_layer1': 64, 'channels_layer2': 128, 'channels_layer3': 256, 'channels_layer4': 512
    }

def train_prune_then_train(dataset_name, train_loader, val_loader, test_loader, num_classes, save_dir, trial_num):
    print(f"\nMethod 2: Prune-Then-Train+L1 | {dataset_name} trial {trial_num}")
    
    initial_model = build_baseline_resnet(num_classes)
    keep_indices = {}
    
    for stage_name in STAGES:
        target_count = PRUNE_THEN_TRAIN_TARGETS[stage_name]
        importance = compute_channel_importance(initial_model, stage_name, train_loader, IMPORTANCE_CAL_BATCHES)
        keeps = select_channels_to_keep(importance, target_channels=target_count)
        keep_indices[stage_name] = keeps
    
    target_planes = [PRUNE_THEN_TRAIN_TARGETS[s] for s in STAGES]
    pruned_model = build_pruned_resnet(target_planes, num_classes)
    pruned_model = copy_weights_to_pruned_model(initial_model, pruned_model, keep_indices)
    
    del initial_model
    cleanup_memory()
    
    pre_train_loss, pre_train_acc, pre_train_auc = evaluate(pruned_model, test_loader)
    print(f"  Pre-training: test_acc={pre_train_acc:.4f}")
    
    optimizer = optim.Adam(pruned_model.parameters(), lr=INITIAL_LR, weight_decay=0.0)
    history = [{'trial': trial_num, 'epoch': 0, 'stage': 'pre_training',
                'test_loss': pre_train_loss, 'test_acc': pre_train_acc, 'test_auc': pre_train_auc}]
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_prune_then_train_trial{trial_num}")
    
    for epoch in range(1, FIXED_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(pruned_model, train_loader, optimizer, epoch, True, L1_LAMBDA)
        val_loss, val_acc, val_auc = evaluate(pruned_model, val_loader)
        test_loss, test_acc, test_auc = evaluate(pruned_model, test_loader)
        
        print(f"  Epoch {epoch}: val_acc={val_acc:.4f} val_auc={val_auc:.4f} | test_acc={test_acc:.4f} test_auc={test_auc:.4f}")
        
        history.append({'trial': trial_num, 'epoch': epoch, 'stage': 'training',
                       'train_loss': train_loss, 'train_acc': train_acc,
                       'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc,
                       'test_loss': test_loss, 'test_acc': test_acc, 'test_auc': test_auc})
        
        # Save incrementally for live monitoring
        pd.DataFrame(history).to_csv(os.path.join(save_dir, f"{dataset_name}_prune_then_train_trial{trial_num}_history.csv"), index=False)
    
    energy_metrics = stop_energy_tracker(tracker, save_dir, f"{dataset_name}_prune_then_train_trial{trial_num}")
    test_loss, test_acc, test_auc = evaluate(pruned_model, test_loader)
    params_m = count_parameters(pruned_model)
    size_mb = model_size_mb(pruned_model)
    flops = compute_flops(pruned_model)
    inf_time, peak_ram = inference_time_per_batch(pruned_model, test_loader)
    
    print(f"  Final: acc={test_acc:.4f} auc={test_auc:.4f} params={params_m:.2f}M size={size_mb:.2f}MB")
    
    torch.save(pruned_model.state_dict(), os.path.join(save_dir, f"{dataset_name}_prune_then_train_trial{trial_num}_final.pth"))
    
    return pruned_model, {
        'method': 'prune_then_train', 'trial': trial_num, 'total_epochs': FIXED_EPOCHS,
        'test_loss': test_loss, 'test_acc': test_acc, 'test_auc': test_auc,
        'params_m': params_m, 'model_size_mb': size_mb, 'flops': flops, 'flops_m': flops/1e6,
        'inference_time_ms': inf_time*1000, 'peak_ram_mb': peak_ram,
        'training_energy_kwh': energy_metrics['energy_kwh'], 'training_emissions_kg': energy_metrics['emissions_kg'],
        'training_duration_s': energy_metrics['duration_s'], 'l1_lambda': L1_LAMBDA, 'use_regularization': True,
        'channels_layer1': target_planes[0], 'channels_layer2': target_planes[1],
        'channels_layer3': target_planes[2], 'channels_layer4': target_planes[3]
    }

def train_progressive(dataset_name, train_loader, val_loader, test_loader, num_classes, save_dir, trial_num, use_l1):
    method_str = "Progressive+L1" if use_l1 else "Progressive NoReg"
    print(f"\nMethod {'3' if use_l1 else '4'}: {method_str} | {dataset_name} trial {trial_num}")
    
    prune_epochs = [WARMUP_EPOCHS + i * EPOCHS_BETWEEN_PRUNES for i in range(1, NUM_PRUNE_STEPS + 1)]
    prune_epochs = [e for e in prune_epochs if e <= FIXED_EPOCHS]
    
    model = build_baseline_resnet(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=0.0)
    current_lr = INITIAL_LR
    current_channels = {s: ORIGINAL_PLANES[i] for i, s in enumerate(STAGES)}
    all_metrics = []
    
    suffix = "with_reg" if use_l1 else "no_reg"
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_progressive_{suffix}_trial{trial_num}")
    
    for epoch in range(1, FIXED_EPOCHS + 1):
        if epoch in prune_epochs:
            prune_step = prune_epochs.index(epoch) + 1
            keep_indices = {}
            keep_ratio = 1.0 - PRUNE_PERCENT
            
            for stage_name in STAGES:
                importance = compute_channel_importance(model, stage_name, train_loader, IMPORTANCE_CAL_BATCHES)
                keeps = select_channels_to_keep(importance, keep_ratio=keep_ratio)
                keep_indices[stage_name] = keeps
                current_channels[stage_name] = len(keeps)
            
            new_stage_planes = [current_channels[s] for s in STAGES]
            pruned_model = build_pruned_resnet(new_stage_planes, num_classes)
            pruned_model = copy_weights_to_pruned_model(model, pruned_model, keep_indices)
            
            del model
            model = pruned_model
            cleanup_memory()
            
            current_lr *= LR_REDUCTION_AFTER_PRUNE
            optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=0.0)
            
            val_loss, val_acc, val_auc = evaluate(model, val_loader)
            test_loss, test_acc, test_auc = evaluate(model, test_loader)
            print(f"  Prune step {prune_step}: channels={list(current_channels.values())} | val_acc={val_acc:.4f} test_acc={test_acc:.4f}")
            
            torch.save(model.state_dict(), os.path.join(save_dir, f"{dataset_name}_progressive_{suffix}_trial{trial_num}_epoch{epoch}_postprune.pth"))
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, epoch, use_l1, L1_LAMBDA)
        val_loss, val_acc, val_auc = evaluate(model, val_loader)
        test_loss, test_acc, test_auc = evaluate(model, test_loader)
        
        print(f"  Epoch {epoch}: val_acc={val_acc:.4f} val_auc={val_auc:.4f} | test_acc={test_acc:.4f} test_auc={test_auc:.4f}")
        
        all_metrics.append({
            'method': f'progressive_pruning_{"with_reg" if use_l1 else "no_reg"}',
            'trial': trial_num, 'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc,
            'test_loss': test_loss, 'test_acc': test_acc, 'test_auc': test_auc,
            'channels_layer1': current_channels['layer1'], 'channels_layer2': current_channels['layer2'],
            'channels_layer3': current_channels['layer3'], 'channels_layer4': current_channels['layer4'],
            'learning_rate': current_lr
        })
        
        # Save incrementally for live monitoring
        pd.DataFrame(all_metrics).to_csv(os.path.join(save_dir, f"{dataset_name}_progressive_{suffix}_trial{trial_num}_all_metrics.csv"), index=False)
    
    energy_metrics = stop_energy_tracker(tracker, save_dir, f"{dataset_name}_progressive_{suffix}_trial{trial_num}")
    test_loss, test_acc, test_auc = evaluate(model, test_loader)
    params_m = count_parameters(model)
    size_mb = model_size_mb(model)
    flops = compute_flops(model)
    inf_time, peak_ram = inference_time_per_batch(model, test_loader)
    
    print(f"  Final: acc={test_acc:.4f} auc={test_auc:.4f} params={params_m:.2f}M size={size_mb:.2f}MB")
    
    torch.save(model.state_dict(), os.path.join(save_dir, f"{dataset_name}_progressive_{suffix}_trial{trial_num}_final.pth"))
    
    return model, {
        'method': f'progressive_pruning_{"with_reg" if use_l1 else "no_reg"}',
        'trial': trial_num, 'total_epochs': FIXED_EPOCHS,
        'test_loss': test_loss, 'test_acc': test_acc, 'test_auc': test_auc,
        'params_m': params_m, 'model_size_mb': size_mb, 'flops': flops, 'flops_m': flops/1e6,
        'inference_time_ms': inf_time*1000, 'peak_ram_mb': peak_ram,
        'training_energy_kwh': energy_metrics['energy_kwh'], 'training_emissions_kg': energy_metrics['emissions_kg'],
        'training_duration_s': energy_metrics['duration_s'],
        'channels_layer1': current_channels['layer1'], 'channels_layer2': current_channels['layer2'],
        'channels_layer3': current_channels['layer3'], 'channels_layer4': current_channels['layer4'],
        'l1_lambda': L1_LAMBDA if use_l1 else 0.0, 'use_regularization': use_l1
    }

def process_dataset(dataset_name, cfg):
    print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
    
    save_dir = os.path.join(SAVE_DIR_BASE, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader, num_classes = make_loaders(cfg['path'], cfg['batch_size'])
    
    all_baseline = []
    all_ptt = []
    all_prog_reg = []
    all_prog_noreg = []
    
    for trial in range(1, NUM_TRIALS + 1):
        print(f"\n--- Trial {trial}/{NUM_TRIALS} ---")
        trial_seed = SEED + trial * 100
        set_seed(trial_seed, deterministic=True)
        
        m, metrics = train_baseline_with_regularization(dataset_name, train_loader, val_loader, test_loader, num_classes, save_dir, trial)
        all_baseline.append(metrics)
        del m
        cleanup_memory()
        
        m, metrics = train_prune_then_train(dataset_name, train_loader, val_loader, test_loader, num_classes, save_dir, trial)
        all_ptt.append(metrics)
        del m
        cleanup_memory()
        
        m, metrics = train_progressive(dataset_name, train_loader, val_loader, test_loader, num_classes, save_dir, trial, use_l1=True)
        all_prog_reg.append(metrics)
        del m
        cleanup_memory()
        
        m, metrics = train_progressive(dataset_name, train_loader, val_loader, test_loader, num_classes, save_dir, trial, use_l1=False)
        all_prog_noreg.append(metrics)
        del m
        cleanup_memory()
    
    all_metrics = pd.DataFrame(all_baseline + all_ptt + all_prog_reg + all_prog_noreg)
    all_metrics.to_csv(os.path.join(save_dir, f"{dataset_name}_all_trials_metrics.csv"), index=False)
    
    def summarize(df, method_name):
        summary = {'method': method_name, 'dataset': dataset_name, 'num_trials': NUM_TRIALS, 'fixed_epochs': FIXED_EPOCHS}
        for col in ['test_acc', 'test_auc', 'params_m', 'model_size_mb', 'flops_m', 'inference_time_ms', 
                    'training_energy_kwh', 'channels_layer1', 'channels_layer2', 'channels_layer3', 'channels_layer4']:
            if col in df.columns:
                summary[f'{col}_mean'] = df[col].mean()
                summary[f'{col}_std'] = df[col].std()
        return summary
    
    summary_rows = [
        summarize(pd.DataFrame(all_baseline), 'baseline_l1'),
        summarize(pd.DataFrame(all_ptt), 'prune_then_train'),
        summarize(pd.DataFrame(all_prog_reg), 'progressive_with_reg'),
        summarize(pd.DataFrame(all_prog_noreg), 'progressive_no_reg')
    ]
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(save_dir, f"{dataset_name}_summary_statistics.csv"), index=False)
    
    print(f"\n{dataset_name.upper()} SUMMARY:")
    print(f"{'Method':<25} {'Test Acc':<20} {'Params (M)':<15} {'Size (MB)':<15}")
    for row in summary_rows:
        acc_str = f"{row['test_acc_mean']:.4f}±{row['test_acc_std']:.4f}"
        params_str = f"{row['params_m_mean']:.2f}±{row['params_m_std']:.2f}"
        size_str = f"{row['model_size_mb_mean']:.2f}±{row['model_size_mb_std']:.2f}"
        print(f"{row['method']:<25} {acc_str:<20} {params_str:<15} {size_str:<15}")
    
    return summary_df

def main():
    print("Four-Method Pruning Comparison")
    print(f"Device: {DEVICE} | Trials: {NUM_TRIALS} | Epochs: {FIXED_EPOCHS} | L1: {L1_LAMBDA}")
    
    os.makedirs(SAVE_DIR_BASE, exist_ok=True)
    all_summaries = []
    
    for dataset_name, cfg in DATASETS.items():
        try:
            summary = process_dataset(dataset_name, cfg)
            all_summaries.append(summary)
        except Exception as e:
            print(f"Error on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(os.path.join(SAVE_DIR_BASE, "all_datasets_summary.csv"), index=False)
        print(f"\nSaved combined summary")
    
    print("\nExperiment complete")

if __name__ == "__main__":
    main()