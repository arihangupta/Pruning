#!/usr/bin/env python3
"""
Dermamnist ResNet50 stage-wise structured pruning with Wanda++ scoring.

Features:
- Wanda++ importance: activation L2 × gradient from regional L2 loss
- Stage-wise pruning with weight surgery
- Local calibration using L2 output discrepancy
- Short global finetune after all stages
- Metrics logging: Acc, AUC, Loss, Params, Zeros, FLOPs, InferenceTime, PeakRAM, PowerProxy
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 32
IMG_SIZE = 224
CAL_EPOCHS = 1
CAL_MAX_BATCHES = 150
CAL_LR = 3e-4
FINAL_FINETUNE_EPOCHS = 2
FINAL_LR = 1e-4
LOG_INTERVAL = 20
WARMUP = 5
TIMING_BATCHES = 30

DERMA_PATH = "/home/arihangupta/Pruning/dinov2/Pruning/datasets/dermamnist_224.npz"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/saved_models_pgto"
BASELINE_CKPT = os.path.join("/home/arihangupta/Pruning/dinov2/Pruning/saved_models",
                             "dermamnist_resnet50_BASELINE.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# Reproducibility
# -------------------------
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# -------------------------
# Data helpers
# -------------------------
def preprocess(arr):
    arr = arr.astype("float32") / 255.0
    if arr.ndim == 3 or arr.shape[-1] == 1:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    return np.transpose(arr, (0,3,1,2))

def make_loaders(npz_path):
    data = np.load(npz_path)
    X_train, y_train = preprocess(data["train_images"]), data["train_labels"].flatten()
    X_val, y_val = preprocess(data["val_images"]), data["val_labels"].flatten()
    X_test, y_test = preprocess(data["test_images"]), data["test_labels"].flatten()
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader, int(len(np.unique(y_train)))

train_loader, val_loader, test_loader, NUM_CLASSES = make_loaders(DERMA_PATH)
print(f"Data loaded. NUM_CLASSES={NUM_CLASSES}, device={DEVICE}")

# -------------------------
# Model
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks): layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x=self.conv1(x); x=self.bn1(x); x=self.relu(x); x=self.maxpool(x)
        x=self.layer1(x); x=self.layer2(x); x=self.layer3(x); x=self.layer4(x)
        x=self.avgpool(x); x=torch.flatten(x,1); x=self.fc(x)
        return x

# -------------------------
# Baseline
# -------------------------
def build_resnet50_for_load(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_baseline_ckpt(path=BASELINE_CKPT):
    model = build_resnet50_for_load(NUM_CLASSES)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()

baseline = load_baseline_ckpt()
criterion = nn.CrossEntropyLoss()

# -------------------------
# Wanda++ Importance
# -------------------------
def collect_activations(model, loader, stage_name, max_batches=CAL_MAX_BATCHES):
    model.eval(); acts = []; hooks = []
    def hook_fn(m, inp, out): acts.append(out.detach().cpu())
    for block in getattr(model, stage_name).children(): hooks.append(block.register_forward_hook(hook_fn))
    for i, (imgs, _) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        _ = model(imgs)
        if i+1 >= max_batches: break
    for h in hooks: h.remove()
    return torch.cat(acts, dim=0)

def wanda_importance(model, baseline, loader, stage_name):
    acts = collect_activations(model, loader, stage_name)
    act_norm = acts.flatten(2).norm(2, dim=2).mean(0)
    # Gradients from regional L2 loss
    model.zero_grad()
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        out_pruned = model(imgs)
        with torch.no_grad(): out_base = baseline(imgs)
        loss = ((out_pruned - out_base)**2).mean()
        loss.backward(); break
    grads = []
    for name, param in model.named_parameters():
        if stage_name in name and "conv" in name and param.grad is not None:
            g = param.grad.detach().abs().mean(dim=[1,2,3])
            grads.append(g)
    grads = torch.cat(grads)
    importance = act_norm * grads[:act_norm.shape[0]]
    return importance.cpu().numpy()

def compute_stage_importance_and_keeps(model, baseline, stage_name, keep_k):
    imp = wanda_importance(model, baseline, train_loader, stage_name)
    keep = np.arange(len(imp)) if keep_k>=len(imp) else np.argsort(imp)[-keep_k:]
    return np.sort(keep)

# -------------------------
# Pruned model builder and weight surgery
# -------------------------
def build_pruned_resnet(keep_indices, num_classes):
    stage_planes = [len(keep_indices[s]) for s in STAGES]
    layers=[3,4,6,3]
    return CustomResNet(Bottleneck, layers, stage_planes, num_classes)

def build_pruned_resnet_and_copy_weights(base_model, keep_indices, num_classes):
    new_model = build_pruned_resnet(keep_indices, num_classes).to(DEVICE)
    expansion = 4
    prev_expanded_idxs = None
    for stage_name in STAGES:
        old_layer = getattr(base_model, stage_name)
        new_layer = getattr(new_model, stage_name)
        kept = keep_indices[stage_name]
        for block_idx, (old_block, new_block) in enumerate(zip(old_layer,new_layer)):
            # conv1
            if stage_name=="layer1" and block_idx==0:
                new_block.conv1.weight.data.copy_(old_block.conv1.weight.data[kept])
                prev_expanded_idxs_safe = torch.arange(old_block.conv1.weight.shape[1])
            else:
                old_w = old_block.conv1.weight.data
                prev_expanded_idxs_safe = torch.arange(old_w.shape[1]) if prev_expanded_idxs is None else prev_expanded_idxs
                new_block.conv1.weight.data.copy_(old_w[kept][:, prev_expanded_idxs_safe,:,:])
            if old_block.conv1.bias is not None:
                new_block.conv1.bias.data.copy_(old_block.conv1.bias.data[kept])
            # bn1
            new_block.bn1.weight.data.copy_(old_block.bn1.weight.data[kept])
            new_block.bn1.bias.data.copy_(old_block.bn1.bias.data[kept])
            new_block.bn1.running_mean.data.copy_(old_block.bn1.running_mean.data[kept])
            new_block.bn1.running_var.data.copy_(old_block.bn1.running_var.data[kept])
            # conv2
            old_w = old_block.conv2.weight.data
            new_block.conv2.weight.data.copy_(old_w[kept][:,kept,:,:])
            if old_block.conv2.bias is not None: new_block.conv2.bias.data.copy_(old_block.conv2.bias.data[kept])
            # bn2
            new_block.bn2.weight.data.copy_(old_block.bn2.weight.data[kept])
            new_block.bn2.bias.data.copy_(old_block.bn2.bias.data[kept])
            new_block.bn2.running_mean.data.copy_(old_block.bn2.running_mean.data[kept])
            new_block.bn2.running_var.data.copy_(old_block.bn2.running_var.data[kept])
            # conv3
            old_w = old_block.conv3.weight.data
            expanded_idx = np.repeat(kept, expansion)
            old_idx = torch.tensor(expanded_idx,dtype=torch.long)
            new_block.conv3.weight.data.copy_(old_w[old_idx][:,kept,:,:])
            if old_block.conv3.bias is not None: new_block.conv3.bias.data.copy_(old_block.conv3.bias.data[old_idx])
            # bn3
            new_block.bn3.weight.data.copy_(old_block.bn3.weight.data[old_idx])
            new_block.bn3.bias.data.copy_(old_block.bn3.bias.data[old_idx])
            new_block.bn3.running_mean.data.copy_(old_block.bn3.running_mean.data[old_idx])
            new_block.bn3.running_var.data.copy_(old_block.bn3.running_var.data[old_idx])
            prev_expanded_idxs = kept
    # FC
    fc_idx = keep_indices["layer4"]
    new_model.fc.weight.data.copy_(base_model.fc.weight.data[:,fc_idx])
    new_model.fc.bias.data.copy_(base_model.fc.bias.data)
    return new_model

# -------------------------
# Freeze / calibrate
# -------------------------
def freeze_all(model):
    for p in model.parameters(): p.requires_grad=False
def unfreeze_stage(model, stage_name):
    for n,p in model.named_parameters():
        if n.startswith(stage_name) or n.startswith("fc.") or n.startswith("bn1."):
            p.requires_grad=True
def calibrate_stage_wanda(model, baseline, stage_name, loader, epochs=CAL_EPOCHS, max_batches=CAL_MAX_BATCHES, lr=CAL_LR):
    freeze_all(model)
    unfreeze_stage(model, stage_name)
    opt = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=lr)
    model.train()
    steps=0
    for ep in range(epochs):
        for imgs,_ in loader:
            imgs=imgs.to(DEVICE)
            opt.zero_grad()
            out_pruned=model(imgs)
            with torch.no_grad(): out_base=baseline(imgs)
            loss=((out_pruned-out_base)**2).mean()
            loss.backward(); opt.step()
            steps+=1
            if steps>=max_batches: return

# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval(); all_logits=[]; all_labels=[]; total_loss=0.0
    for imgs,labels in loader:
        imgs=imgs.to(DEVICE); labels=labels.to(DEVICE)
        logits=model(imgs)
        all_logits.append(logits.cpu()); all_labels.append(labels.cpu())
        total_loss+=criterion(logits,labels).item()*len(labels)
    all_logits=torch.cat(all_logits); all_labels=torch.cat(all_labels)
    pred_labels=all_logits.argmax(1)
    acc=(pred_labels==all_labels).float().mean().item()
    try: auc=roc_auc_score(all_labels.numpy(), all_logits.numpy(), multi_class="ovr")
    except: auc=float('nan')
    loss=total_loss/len(all_labels)
    return acc, auc, loss

# -------------------------
# Main PGTO Wanda++ loop
# -------------------------
STAGES=["layer1","layer2","layer3","layer4"]
TARGET_RATIOS=[0.5,0.7]

rows=[]
for target_ratio in TARGET_RATIOS:
    print(f"\n=== Wanda++ PGTO target_ratio={target_ratio} ===")
    keep_indices = {s: np.arange(next(getattr(baseline,s).children()).conv1.out_channels) for s in STAGES}
    for s in STAGES:
        orig = next(getattr(baseline,s).children()).conv1.out_channels
        keep_k = max(1, int(math.floor(orig*(1.0-target_ratio))))
        keep_indices[s] = compute_stage_importance_and_keeps(baseline, baseline, s, keep_k)
        print(f"  Stage {s}: keeping {len(keep_indices[s])}/{orig} channels")
        pruned_model = build_pruned_resnet_and_copy_weights(baseline, keep_indices, NUM_CLASSES).to(DEVICE)
        calibrate_stage_wanda(pruned_model, baseline, s, train_loader)
    # Final finetune
    freeze_all(pruned_model); unfreeze_stage(pruned_model,"layer4"); opt=optim.Adam(filter(lambda p:p.requires_grad, pruned_model.parameters()), lr=FINAL_LR)
    for ep in range(FINAL_FINETUNE_EPOCHS):
        for imgs,_ in train_loader:
            imgs=imgs.to(DEVICE); opt.zero_grad()
            loss=criterion(pruned_model(imgs),_.to(DEVICE))
            loss.backward(); opt.step()
    # Evaluate
    acc, auc, loss = evaluate(pruned_model, test_loader, criterion)
    # Params, zeros
    params=sum(p.numel() for p in pruned_model.parameters())
    zeros=sum((p==0).sum().item() for p in pruned_model.parameters())
    # Save
    ckpt_path=os.path.join(SAVE_DIR,f"dermamnist_resnet50_WANDA_pp_ratio{target_ratio:.2f}.pth")
    torch.save(pruned_model.state_dict(), ckpt_path)
    print(f"Saved pruned model to {ckpt_path}")
    rows.append({"Variant":"Wanda++","Stage":"final","Ratio":target_ratio,"Acc":acc,"AUC":auc,"Loss":loss,"Params":params,"Zeros":zeros})

# CSV log
df=pd.DataFrame(rows)
csv_path=os.path.join(SAVE_DIR,"metrics_pgto_wanda_pp.csv")
df.to_csv(csv_path,index=False)
print(f"Metrics saved to {csv_path}")
