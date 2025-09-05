#!/usr/bin/env python3
"""
Dermamnist ResNet50 model-surgery pruning (PGTO: stage-by-stage) with metrics.

Uses regional gradient scoring per stage for channel importance.
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

TARGET_RATIOS = [0.5, 0.7]     
CAL_EPOCHS = 1                 
CAL_MAX_BATCHES = 150          
CAL_LR = 3e-4
FINAL_FINETUNE_EPOCHS = 2      
FINAL_LR = 1e-4
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
    return np.transpose(arr, (0, 3, 1, 2))  

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
# CustomResNet
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
# Gradient-based importance
# -------------------------
criterion = nn.CrossEntropyLoss()

def compute_stage_importance_and_keeps_grad(model: nn.Module, stage_name: str, keep_k: int):
    model.eval()
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    orig_planes = first_block.conv1.out_channels

    imgs, labels = next(iter(train_loader))
    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
    imgs.requires_grad = True

    outputs = model(imgs)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()

    grad_scores = np.zeros(orig_planes, dtype=np.float32)
    for block in stage.children():
        g = block.conv3.weight.grad
        if g is not None:
            exp = block.conv3.out_channels // block.conv3.in_channels
            per_plane = g.abs().detach().cpu().numpy()
            per_plane = [np.mean(per_plane[p*exp:(p+1)*exp,:,:,:]) for p in range(orig_planes)]
            grad_scores += np.array(per_plane)

    keep = np.arange(len(grad_scores)) if keep_k >= len(grad_scores) else np.argsort(grad_scores)[-keep_k:]
    return np.sort(keep)

# -------------------------
# Build pruned model & copy weights
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
    """
    Build a CustomResNet with given keep_indices (per stage) and copy weights from base_model.
    Correctly handles Bottleneck: conv1/conv2/conv3, downsample, fc.
    """
    new_model = build_pruned_resnet(keep_indices, num_classes=num_classes).to(DEVICE)
    expansion = 4
    STAGES = ["layer1", "layer2", "layer3", "layer4"]

    prev_stage_keep = None  # input channels from previous stage

    for stage_name in STAGES:
        old_stage = getattr(base_model, stage_name)
        new_stage = getattr(new_model, stage_name)
        stage_keep = keep_indices[stage_name]

        for block_idx, (old_block, new_block) in enumerate(zip(old_stage, new_stage)):
            # input channels for this block
            if stage_name == "layer1" and block_idx == 0:
                prev_keep = torch.arange(old_block.conv1.in_channels)  # full input
            else:
                prev_keep = prev_stage_keep

            # convert to torch long tensors
            out_idx = torch.tensor(stage_keep, dtype=torch.long)
            in_idx  = torch.tensor(prev_keep, dtype=torch.long)

            # --- conv1 ---
            new_block.conv1.weight.data.copy_(old_block.conv1.weight.data[out_idx][:, in_idx, :, :])
            if old_block.conv1.bias is not None:
                new_block.conv1.bias.data.copy_(old_block.conv1.bias.data[out_idx])
            new_block.bn1.weight.data.copy_(old_block.bn1.weight.data[out_idx])
            new_block.bn1.bias.data.copy_(old_block.bn1.bias.data[out_idx])
            new_block.bn1.running_mean.data.copy_(old_block.bn1.running_mean.data[out_idx])
            new_block.bn1.running_var.data.copy_(old_block.bn1.running_var.data[out_idx])

            # --- conv2 ---
            new_block.conv2.weight.data.copy_(old_block.conv2.weight.data[out_idx][:, out_idx, :, :])
            if old_block.conv2.bias is not None:
                new_block.conv2.bias.data.copy_(old_block.conv2.bias.data[out_idx])
            new_block.bn2.weight.data.copy_(old_block.bn2.weight.data[out_idx])
            new_block.bn2.bias.data.copy_(old_block.bn2.bias.data[out_idx])
            new_block.bn2.running_mean.data.copy_(old_block.bn2.running_mean.data[out_idx])
            new_block.bn2.running_var.data.copy_(old_block.bn2.running_var.data[out_idx])

            # --- conv3 ---
            expanded_idx = np.repeat(stage_keep, expansion)
            out_idx3 = torch.tensor(expanded_idx, dtype=torch.long)
            in_idx3 = out_idx  # input channels from conv2
            new_block.conv3.weight.data.copy_(old_block.conv3.weight.data[out_idx3][:, in_idx3, :, :])
            if old_block.conv3.bias is not None:
                new_block.conv3.bias.data.copy_(old_block.conv3.bias.data[out_idx3])
            new_block.bn3.weight.data.copy_(old_block.bn3.weight.data[out_idx3])
            new_block.bn3.bias.data.copy_(old_block.bn3.bias.data[out_idx3])
            new_block.bn3.running_mean.data.copy_(old_block.bn3.running_mean.data[out_idx3])
            new_block.bn3.running_var.data.copy_(old_block.bn3.running_var.data[out_idx3])

            # --- downsample ---
            if old_block.downsample is not None:
                ds_conv_w = old_block.downsample[0].weight.data
                new_block.downsample[0].weight.data.copy_(ds_conv_w[out_idx3][:, in_idx, :, :])
                new_block.downsample[1].weight.data.copy_(old_block.downsample[1].weight.data[out_idx3])
                new_block.downsample[1].bias.data.copy_(old_block.downsample[1].bias.data[out_idx3])
                new_block.downsample[1].running_mean.data.copy_(old_block.downsample[1].running_mean.data[out_idx3])
                new_block.downsample[1].running_var.data.copy_(old_block.downsample[1].running_var.data[out_idx3])

            # update prev_keep for next block
            prev_keep = stage_keep

        # update prev_stage_keep for next stage
        prev_stage_keep = stage_keep

    # --- fc layer ---
    new_model.fc.weight.data.copy_(base_model.fc.weight.data[:, prev_stage_keep])
    new_model.fc.bias.data.copy_(base_model.fc.bias.data)

    return new_model



# -------------------------
# Calibration helpers
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
            running_loss += float(loss.item())*imgs.size(0)
            _, preds = out.max(1)
            total += labels.size(0); correct += int(preds.eq(labels).sum().item())
            steps += 1
            if bidx % LOG_INTERVAL == 0:
                print(f"      Calib {stage_name} ep{ep+1} batch{bidx} - loss {running_loss/max(1,total):.4f}, acc {correct/max(1,total):.4f}")
            if steps >= max_batches:
                return

# -------------------------
# PGTO loop
# -------------------------
def stage_orig_channels(model, stage_name):
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    return first_block.conv1.out_channels

STAGES = ["layer1","layer2","layer3","layer4"]
rows = []

def evaluate_model(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds.append(out.cpu())
            targets.append(y.cpu())
    preds = torch.cat(preds); targets = torch.cat(targets)
    pred_labels = preds.argmax(1)
    acc = (pred_labels == targets).float().mean().item()
    try:
        auc = roc_auc_score(targets.numpy(), nn.functional.softmax(preds,1).numpy(), multi_class='ovr')
    except:
        auc = float('nan')
    return acc, auc

# Baseline metrics
acc, auc = evaluate_model(baseline, test_loader)
rows.append({"Variant":"baseline","Stage":"baseline","Ratio":0.0,"Acc":acc,"AUC":auc,"Params":sum(p.numel() for p in baseline.parameters()),"Zeros":0})

# PGTO pipeline
for target_ratio in TARGET_RATIOS:
    print(f"\n=== PGTO: regional gradient scoring, target_ratio={target_ratio} ===")
    keep_indices = {s: np.arange(stage_orig_channels(baseline, s)) for s in STAGES}
    for s in STAGES:
        orig = stage_orig_channels(baseline, s)
        keep_k = max(1, int(math.floor(orig*(1.0-target_ratio))))
        keep_indices[s] = compute_stage_importance_and_keeps_grad(baseline, s, keep_k)
        print(f"  -> stage {s}: keep {len(keep_indices[s])}/{orig} ({100*len(keep_indices[s])/orig:.1f}% kept)")
        pruned_model = build_pruned_resnet_and_copy_weights(baseline, keep_indices, NUM_CLASSES).to(DEVICE)
        calibrate_stage(pruned_model, s, train_loader)
        acc, auc = evaluate_model(pruned_model, test_loader)
        rows.append({"Variant":"PGTO_grad","Stage":s,"Ratio":1-len(keep_indices[s])/orig,"Acc":acc,"AUC":auc,"Params":sum(p.numel() for p in pruned_model.parameters()),"Zeros":0})

    # Final global finetune
    print("-> Global finetune")
    freeze_all(pruned_model)
    for p in pruned_model.parameters(): p.requires_grad = True
    opt = optim.Adam(pruned_model.parameters(), lr=FINAL_LR)
    for ep in range(FINAL_FINETUNE_EPOCHS):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = criterion(pruned_model(imgs), labels)
            loss.backward(); opt.step()
    acc, auc = evaluate_model(pruned_model, test_loader)
    rows.append({"Variant":"PGTO_grad","Stage":"finetuned","Ratio":target_ratio,"Acc":acc,"AUC":auc,"Params":sum(p.numel() for p in pruned_model.parameters()),"Zeros":0})

# Save metrics
df = pd.DataFrame(rows)
df.to_csv(os.path.join(SAVE_DIR,"pgto_reggrad_results.csv"), index=False)
print(f"PGTO complete. Metrics saved to {SAVE_DIR}/pgto_reggrad_results.csv")
