#!/usr/bin/env python3
"""
Calculates AUC for final models (baseline, quantization, slim_kd, slim_kd_amp, regional_gradients, regional_gradients_amp)
for bloodmnist, dermamnist, and pathmnist datasets. Updates existing CSVs with AUC values for final models.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
from sklearn.metrics import roc_auc_score
import psutil
from torchvision.models.resnet import Bottleneck

# -------------------------
# Config
# -------------------------
SAVE_DIR_BASE = "/home/arihangupta/Pruning/dinov2/Pruning/trial_1/pruned_models"
DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
SEED = 42
KEEP_RATIO = 0.5  # For 50% compression
ORIGINAL_PLANES = [64, 128, 256, 512]

DATASET_BATCH_SIZES = {
    "bloodmnist": 32,
    "dermamnist": 32,
    "pathmnist": 16,
}

DATASETS = {
    "bloodmnist": {"path": os.path.join(DATASET_DIR, "bloodmnist_224.npz")},
    "dermamnist": {"path": os.path.join(DATASET_DIR, "dermamnist_224.npz")},
    "pathmnist": {"path": os.path.join(DATASET_DIR, "pathmnist_224.npz")},
}

FINAL_MODELS = [
    {"variant": "baseline", "file": "baseline.pth", "stage": "baseline", "is_amp": False},
    {"variant": "quantization", "file": "quantization_r50compressed_final.pth", "stage": "after_global_finetune", "is_amp": True},
    {"variant": "slim_kd", "file": "slim_kd_r50compressed_final.pth", "stage": "after_global_finetune", "is_amp": False},
    {"variant": "slim_kd_fp16", "file": "slim_kd_amp_r50compressed_final_amp.pth", "stage": "after_global_finetune_amp", "is_amp": True},
    {"variant": "regional_gradients", "file": "pgto_regional_gradients_r50compressed_final.pth", "stage": "after_global_finetune", "is_amp": False},
    {"variant": "regional_gradients_fp16", "file": "pgto_regional_gradients_amp_r50compressed_final_amp.pth", "stage": "after_global_finetune_amp", "is_amp": True},
]

# -------------------------
# Repro
# -------------------------
def set_seed(s=SEED, deterministic=True):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED, deterministic=True)

# -------------------------
# Memory monitoring
# -------------------------
def log_memory_usage(prefix=""):
    process = psutil.Process()
    mem_info = process.memory_info()
    gpu_mem = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    print(f"{prefix}Memory Usage: RSS={mem_info.rss/(1024**2):.2f}MB, GPU={gpu_mem:.2f}MB")

def cleanup_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

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
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset file not found: {npz_path}")
    data = np.load(npz_path, mmap_mode="r")
    X_test, y_test = data["test_images"], data["test_labels"].flatten()
    n_test = len(y_test)
    print(f"Test dataset size: {n_test}")
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    num_classes = int(len(np.unique(y_test)))
    return test_loader, num_classes

# -------------------------
# Models
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

def build_pruned_or_slim_resnet(stage_planes=None, num_classes=1000, random_init=False):
    if stage_planes is None:
        raise ValueError("Provide stage_planes")
    stage_planes = [max(1, int(p)) for p in stage_planes]
    return CustomResNet(block=Bottleneck, layers=[3,4,6,3], stage_planes=stage_planes,
                        num_classes=num_classes, random_init=random_init).to(DEVICE)

# -------------------------
# Metrics
# -------------------------
criterion = nn.CrossEntropyLoss()

def evaluate_model_basic(model, loader):
    model.eval()
    loss_total = 0.0; correct = 0; total = 0
    probs_list = []; labels_list = []
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = DEVICE
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(model_device)
            labels = labels.to(model_device)
            if model_dtype == torch.half:
                images = images.half()
            outputs = model(images)
            if outputs.device != labels.device:
                outputs = outputs.to(labels.device)
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

# -------------------------
# Main processing
# -------------------------
def process_dataset(dataset_name, cfg):
    try:
        print(f"\n=== Processing {dataset_name.upper()} ===")
        log_memory_usage(f"Before loading {dataset_name}: ")

        SAVE_DIR = os.path.join(SAVE_DIR_BASE, dataset_name)
        csv_path = os.path.join(SAVE_DIR, f"{dataset_name}_combined_pruning_kd_metrics_with_energy.csv")
        if not os.path.exists(SAVE_DIR):
            raise FileNotFoundError(f"Save directory not found: {SAVE_DIR}")

        # Load test data
        batch_size = DATASET_BATCH_SIZES.get(dataset_name, 32)
        test_loader, num_classes = make_loaders(cfg["path"], batch_size)
        print(f"Loaded test data for {dataset_name}. Num classes={num_classes}, batch_size={batch_size}")
        log_memory_usage(f"After loading data for {dataset_name}: ")

        # Load or create CSV
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            rows = df.to_dict('records')
        else:
            rows = []

        # Process each final model
        stage_planes = [max(1, int(p * KEEP_RATIO)) for p in ORIGINAL_PLANES]
        for model_info in FINAL_MODELS:
            variant = model_info["variant"]
            model_file = model_info["file"]
            stage = model_info["stage"]
            is_amp = model_info["is_amp"]
            model_path = os.path.join(SAVE_DIR, model_file)

            if not os.path.exists(model_path):
                print(f"  Skipping {variant}: Model not found at {model_path}")
                continue

            print(f"  Processing {variant} ({model_file})")
            
            # Load model
            if variant == "baseline" or variant == "quantization":
                model = build_resnet50_for_load(num_classes)
                if is_amp:
                    model = model.half()
            else:
                model = build_pruned_or_slim_resnet(stage_planes=stage_planes, num_classes=num_classes, random_init=False)
                if is_amp:
                    model = model.half()
            
            try:
                model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
                model = model.to(DEVICE).eval()
                print(f"    Loaded {model_path}")
            except Exception as e:
                print(f"    Error loading {model_path}: {e}")
                continue

            log_memory_usage(f"    After loading {variant}: ")

            # Compute AUC
            _, acc, auc = evaluate_model_basic(model, test_loader)
            print(f"    {variant} AUC: {auc:.4f}, Accuracy: {acc:.4f}")

            # Update or append row
            row = {
                "Variant": variant,
                "Stage": stage,
                "Ratio": KEEP_RATIO if variant != "baseline" else 1.0,
                "AUC": auc,
                "Acc": acc,
                "ModelPath": model_path
            }
            found = False
            for i, r in enumerate(rows):
                if r["Variant"] == variant and r["Stage"] == stage and r["Ratio"] == (KEEP_RATIO if variant != "baseline" else 1.0):
                    rows[i].update(row)
                    found = True
                    break
            if not found:
                rows.append(row)

            del model
            cleanup_memory()
            log_memory_usage(f"    After processing {variant}: ")

        # Save updated CSV
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Saved/updated CSV: {csv_path}")

        del test_loader
        cleanup_memory()
        log_memory_usage(f"After completing {dataset_name}: ")
        return True

    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        return False

# -------------------------
# Main loop
# -------------------------
if __name__ == "__main__":
    for dataset_name, cfg in DATASETS.items():
        success = process_dataset(dataset_name, cfg)
        if not success:
            print(f"Failed to process {dataset_name}. Continuing...")
    print("All datasets processed.")
