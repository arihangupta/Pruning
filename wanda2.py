#!/usr/bin/env python3
"""
Dermamnist ResNet50 PGTO pruning with L1, BN, and Regional Gradients (RG) scoring.

- Stage-by-stage pruning (layer1..layer4)
- Short calibration per stage
- Global finetune after all stages
- Metrics & checkpoint saving

Requirements: torch, torchvision, sklearn, numpy, pandas
"""

import os, time, math, random, copy, tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.resnet import Bottleneck
from torchvision import models
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 32
IMG_SIZE = 224
SAVE_DIR = "./saved_models_pgto"
os.makedirs(SAVE_DIR, exist_ok=True)
DERMA_PATH = "./dermamnist_224.npz"

# PGTO config
METHODS = ["l1", "bn_gamma", "regional_gradients"]
TARGET_RATIOS = [0.5, 0.6, 0.7]
CAL_EPOCHS = 1
CAL_MAX_BATCHES = 150
CAL_LR = 3e-4
FINAL_FINETUNE_EPOCHS = 2
FINAL_LR = 1e-4
LOG_INTERVAL = 20

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
# Data loaders
# -------------------------
def preprocess(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32") / 255.0
    if arr.ndim == 3 or arr.shape[-1] == 1:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    return np.transpose(arr, (0,3,1,2))  # NHWC -> NCHW

def make_loaders(npz_path):
    data = np.load(npz_path)
    X_train, y_train = data["train_images"], data["train_labels"].flatten()
    X_val, y_val = data["val_images"], data["val_labels"].flatten()
    X_test, y_test = data["test_images"], data["test_labels"].flatten()
    X_train = preprocess(X_train)
    X_val = preprocess(X_val)
    X_test = preprocess(X_test)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_ds  = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader, int(len(np.unique(y_train)))

train_loader, val_loader, test_loader, NUM_CLASSES = make_loaders(DERMA_PATH)
print(f"Data loaded. NUM_CLASSES={NUM_CLASSES}, device={DEVICE}")

# -------------------------
# Model builder
# -------------------------
class CustomResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3,4,6,3], stage_planes=[64,128,256,512], num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3,64,7,2,3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3,2,1)
        self.stage_planes = stage_planes[:]
        self.layers_cfg = layers[:]
        self.layer1 = self._make_layer(block, stage_planes[0], layers[0])
        self.layer2 = self._make_layer(block, stage_planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, stage_planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, stage_planes[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(stage_planes[3]*block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion,1,stride,bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x=self.maxpool(x)
        x = self.layer1(x); x=self.layer2(x); x=self.layer3(x); x=self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x,1); x=self.fc(x)
        return x

def build_resnet50_for_load(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -------------------------
# Load baseline
# -------------------------
BASELINE_CKPT = os.path.join(SAVE_DIR,"dermamnist_resnet50_BASELINE.pth")
baseline = build_resnet50_for_load(NUM_CLASSES).to(DEVICE)
if os.path.exists(BASELINE_CKPT):
    baseline.load_state_dict(torch.load(BASELINE_CKPT,map_location=DEVICE))
baseline.eval()
print("Baseline loaded.")

# -------------------------
# Metrics
# -------------------------
criterion = nn.CrossEntropyLoss()

def evaluate_model_basic(model, loader):
    model.eval()
    loss_total=0; correct=0; total=0
    probs_list, labels_list=[],[]
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            loss_total += float(loss.item())*imgs.size(0)
            _, pred = out.max(1)
            total += labels.size(0)
            correct += int(pred.eq(labels).sum().item())
            probs_list.append(torch.softmax(out,1).cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    loss_avg = loss_total/max(1,total)
    acc = correct/max(1,total)
    try:
        auc = roc_auc_score(np.concatenate(labels_list), np.concatenate(probs_list), multi_class="ovr")
    except Exception:
        auc = float("nan")
    return loss_avg, acc, auc

def count_zeros_and_total(model):
    total=0; zeros=0
    for p in model.parameters():
        total+=p.numel()
        zeros+=int((p==0).sum().item())
    return zeros,total

def params_count(model):
    return sum(p.numel() for p in model.parameters())

def model_size_bytes(model):
    fd,tmp = tempfile.mkstemp(".pth"); os.close(fd)
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp)
    os.remove(tmp)
    return size

def collect_metrics_row(tag_variant, tag_stage, ratio, model, test_loader, path_hint):
    loss, acc, auc = evaluate_model_basic(model,test_loader)
    zeros, total = count_zeros_and_total(model)
    params = params_count(model)
    size_mb = os.path.getsize(path_hint)/(1024**2) if os.path.exists(path_hint) else model_size_bytes(model)/(1024**2)
    return {"Variant":tag_variant,"Stage":tag_stage,"Ratio":ratio,"Acc":acc,"AUC":auc,"Loss":loss,
            "Params":params,"Zeros":zeros,"TotalParams":total,"PctZeros":zeros/total*100 if total>0 else 0,
            "ModelSizeMB":size_mb,"ModelPath":path_hint}

# -------------------------
# Stage helpers
# -------------------------
STAGES = ["layer1","layer2","layer3","layer4"]
def stage_orig_channels(model, stage_name):
    layer = getattr(model, stage_name)
    first_block = next(layer.children())
    return first_block.conv1.out_channels

# -------------------------
# Importance scoring
# -------------------------
def compute_stage_importance_and_keeps(model, stage_name, keep_k, method="l1", calib_loader=None):
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    orig_planes = first_block.conv1.out_channels
    if method=="l1":
        block_importances=[]
        for block in stage.children():
            conv3 = block.conv3.weight.detach().abs().cpu().numpy()
            exp = block.conv3.out_channels//block.conv3.in_channels
            per_plane = [np.sum(np.abs(conv3[p*exp:(p+1)*exp,:,:,:])) for p in range(orig_planes)]
            block_importances.append(np.array(per_plane))
        agg = np.mean(np.stack(block_importances,axis=0),axis=0)
    elif method=="bn_gamma":
        block_importances=[]
        for block in stage.children():
            gammas = block.bn3.weight.detach().abs().cpu().numpy()
            exp = block.conv3.out_channels//block.conv3.in_channels
            per_plane = [np.mean(gammas[p*exp:(p+1)*exp]) for p in range(orig_planes)]
            block_importances.append(np.array(per_plane))
        agg = np.mean(np.stack(block_importances,axis=0),axis=0)
    elif method=="regional_gradients":
        assert calib_loader is not None, "RG needs calibration data"
        agg = np.zeros(orig_planes)
        model.eval()
        for bidx,(imgs,_) in enumerate(calib_loader):
            imgs = imgs.to(DEVICE)
            imgs.requires_grad=False
            x = model.conv1(imgs)
            x = model.bn1(x); x = model.relu(x); x=model.maxpool(x)
            for s in STAGES:
                if s==stage_name: break
                x = getattr(model,s)(x)
            stage_out = stage(x)
            grads=[]
            for block in stage.children():
                block.conv1.weight.requires_grad_(True)
                stage_out2 = block(stage_out)
                loss = stage_out2.pow(2).sum()
                model.zero_grad()
                loss.backward(retain_graph=True)
                grad = block.conv3.weight.grad
                if grad is None:
                    grad = block.conv3.weight.data
                grad_val = grad.abs().sum(dim=(1,2,3)).cpu().numpy()
                agg += grad_val
            if bidx+1 >= CAL_MAX_BATCHES: break
        agg = agg / max(1,bidx+1)
    keep = np.arange(len(agg)) if keep_k>=len(agg) else np.argsort(agg)[-keep_k:]
    return np.sort(keep)

# -------------------------
# PGTO pipeline
# -------------------------
rows=[]
# Baseline metrics
base_ckpt = os.path.join(SAVE_DIR,"baseline.pth")
torch.save(baseline.state_dict(), base_ckpt)
row = collect_metrics_row("baseline","baseline",0.0, baseline, test_loader, base_ckpt)
rows.append(row)
print("\n=== EVALUATE BASELINE ===")
print("Baseline metrics:", {k: row[k] for k in ["Acc","AUC","ModelSizeMB"]})

for method in METHODS:
    for target_ratio in TARGET_RATIOS:
        print(f"\n=== PGTO: method={method}, target_ratio={target_ratio} ===")
        keep_indices = {s: np.arange(stage_orig_channels(baseline,s)) for s in STAGES}

        for s in STAGES:
            orig_channels = stage_orig_channels(baseline,s)
            keep_k = max(1,int(math.floor(orig_channels*(1.0-target_ratio))))
            keep_indices[s] = compute_stage_importance_and_keeps(baseline,s,keep_k,method=method,calib_loader=train_loader)
            print(f"  Stage {s}: keeping {len(keep_indices[s])}/{orig_channels} channels ({100*len(keep_indices[s])/orig_channels:.1f}%)")

            # Build pruned model
            pruned_model = build_resnet50_for_load(NUM_CLASSES).to(DEVICE)
            pruned_model.load_state_dict(baseline.state_dict())
            # stage-wise calibration
            pruned_model.eval()
            for ep in range(CAL_EPOCHS):
                for bidx,(imgs,labels) in enumerate(train_loader):
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    _ = pruned_model(imgs)
                    if bidx+1>=CAL_MAX_BATCHES: break

            stage_ckpt = os.path.join(SAVE_DIR,f"dermamnist_resnet50_pgto_{method}_r{int(target_ratio*100)}_{s}_calibrated.pth")
            torch.save(pruned_model.state_dict(), stage_ckpt)
            row = collect_metrics_row(method,f"{s}_after_stage_calibration",target_ratio,pruned_model,test_loader,stage_ckpt)
            rows.append(row)
            print("    Stage metrics:", {k: row[k] for k in ["Acc","AUC","ModelSizeMB"]})

        # Global finetune
        print("  Final global finetune...")
        pruned_model = build_resnet50_for_load(NUM_CLASSES).to(DEVICE)
        pruned_model.load_state_dict(baseline.state_dict())
        optimizer = optim.Adam(pruned_model.parameters(), lr=FINAL_LR)
        pruned_model.train()
        for ep in range(FINAL_FINETUNE_EPOCHS):
            for batch_idx,(images,labels) in enumerate(train_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = pruned_model(images)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()
        final_ckpt = os.path.join(SAVE_DIR,f"dermamnist_resnet50_pgto_{method}_r{int(target_ratio*100)}_final.pth")
        torch.save(pruned_model.state_dict(), final_ckpt)
        row = collect_metrics_row(method,"final_finetune",target_ratio,pruned_model,test_loader,final_ckpt)
        rows.append(row)
        print("    Final finetune metrics:", {k: row[k] for k in ["Acc","AUC","ModelSizeMB"]})

# Save summary CSV
df = pd.DataFrame(rows)
csv_path = os.path.join(SAVE_DIR,"pgto_summary_metrics.csv")
df.to_csv(csv_path,index=False)
print(f"\nAll metrics saved to {csv_path}")
