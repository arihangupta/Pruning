#!/usr/bin/env python3
"""
Dermamnist ResNet50 pruning + knowledge distillation with efficiency metrics.
Tracks FLOPs, Params, Model Size, Inference time, Peak RAM, Power proxy, Loss, Acc, AUC.
"""

import os, time, math, random, copy, csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from torchvision.models.resnet import Bottleneck
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# -------------------------
# CONFIG
# -------------------------
DERMA_PATH = "/home/arihangupta/Pruning/dinov2/Pruning/datasets/dermamnist_224.npz"
SAVE_DIR   = "/home/arihangupta/Pruning/dinov2/Pruning/saved_models_distillation"
BASELINE_CKPT = "/home/arihangupta/Pruning/dinov2/Pruning/saved_models/dermamnist_resnet50_BASELINE.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 32
FINETUNE_EPOCHS = 5
LEARNING_RATE = 1e-4
PRUNE_RATIOS = [0.5, 0.7]
LOG_INTERVAL = 20
DISTILL_ALPHA = 0.7
DISTILL_TEMP = 4.0
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# SEED
# -------------------------
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)

# -------------------------
# DATA HELPERS
# -------------------------
def preprocess(arr):
    arr = arr.astype("float32") / 255.0
    if arr.ndim == 3 or arr.shape[-1] == 1:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    return np.transpose(arr, (0, 3, 1, 2))

def make_loaders(npz_path):
    data = np.load(npz_path)
    X_train, y_train = data["train_images"], data["train_labels"].flatten()
    X_val, y_val = data["val_images"], data["val_labels"].flatten()
    X_test, y_test = data["test_images"], data["test_labels"].flatten()
    X_train, X_val, X_test = preprocess(X_train), preprocess(X_val), preprocess(X_test)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader, int(len(np.unique(y_train)))

train_loader, val_loader, test_loader, NUM_CLASSES = make_loaders(DERMA_PATH)
print(f"Data loaded. NUM_CLASSES={NUM_CLASSES}, device={DEVICE}")

# -------------------------
# CUSTOM RESNET
# -------------------------
class CustomResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3,4,6,3], stage_planes=[64,128,256,512], num_classes=NUM_CLASSES):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3,2,1)
        self.layer1 = self._make_layer(block, stage_planes[0], layers[0])
        self.layer2 = self._make_layer(block, stage_planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, stage_planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, stage_planes[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(stage_planes[3]*block.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        return self.fc(x)

# -------------------------
# BASELINE LOAD
# -------------------------
def build_resnet50(num_classes=NUM_CLASSES):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

baseline = build_resnet50().to(DEVICE)
baseline.load_state_dict(torch.load(BASELINE_CKPT, map_location=DEVICE, weights_only=True))
baseline.eval()
for param in baseline.parameters():
    param.requires_grad = False
print("Baseline loaded.")

# -------------------------
# PRUNING UTILITIES
# -------------------------
def compute_stage_importance_and_keeps(model, stage_name, keep_k, method="l1"):
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    orig_planes = first_block.conv1.out_channels
    block_importances = []
    for block in stage.children():
        if method == "bn_gamma":
            gammas = block.bn3.weight.detach().abs().cpu().numpy()
            exp = block.conv3.out_channels // block.conv3.in_channels
            per_plane = [np.mean(gammas[p*exp:(p+1)*exp]) for p in range(orig_planes)]
        else:
            conv3 = block.conv3.weight.detach().abs().cpu().numpy()
            exp = block.conv3.out_channels // block.conv3.in_channels
            per_plane = [np.sum(np.abs(conv3[p*exp:(p+1)*exp,:,:,:])) for p in range(orig_planes)]
        block_importances.append(np.array(per_plane))
    agg = np.mean(np.stack(block_importances), axis=0)
    if keep_k >= len(agg):
        keep = np.arange(len(agg))
    else:
        keep = np.argsort(agg)[-keep_k:]
    return np.sort(keep)

def build_pruned_resnet(stage_keeps, num_classes=NUM_CLASSES):
    planes = [len(stage_keeps['layer1']), len(stage_keeps['layer2']),
              len(stage_keeps['layer3']), len(stage_keeps['layer4'])]
    return CustomResNet(stage_planes=planes, num_classes=num_classes).to(DEVICE)

# -------------------------
# DISTILLATION LOSS
# -------------------------
def distillation_loss(student_logits, teacher_logits, labels, T=DISTILL_TEMP, alpha=DISTILL_ALPHA):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T*T)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1-alpha) * hard_loss

# -------------------------
# EVALUATION
# -------------------------
criterion = nn.CrossEntropyLoss()
def evaluate_model_basic(model, loader):
    model.eval()
    loss_total=0; correct=0; total=0
    probs_list=[]; labels_list=[]
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss_total += float(criterion(outputs, labels).item())*imgs.size(0)
            _, preds = outputs.max(1)
            correct += int(preds.eq(labels).sum().item())
            total += labels.size(0)
            probs_list.append(torch.softmax(outputs, dim=1).cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    loss_avg = loss_total / total
    acc = correct / total
    try:
        auc = roc_auc_score(np.concatenate(labels_list), np.concatenate(probs_list), multi_class="ovr")
    except:
        auc = 0.0
    return loss_avg, acc, auc

# -------------------------
# TRAIN STUDENT WITH KD
# -------------------------
def train_student_kd(student, teacher, loader, val_loader, epochs=FINETUNE_EPOCHS):
    student.train()
    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE)
    for ep in range(epochs):
        running_loss = 0; total=0; correct=0
        for bidx, (imgs, labels) in enumerate(loader,1):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                teacher_logits = teacher(imgs)
            student_logits = student(imgs)
            loss = distillation_loss(student_logits, teacher_logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running_loss += float(loss.item())*imgs.size(0)
            _, preds = student_logits.max(1)
            total += labels.size(0); correct += int(preds.eq(labels).sum().item())
            if bidx % LOG_INTERVAL == 0:
                print(f"KD Epoch {ep+1} batch {bidx} - loss {running_loss/total:.4f}, acc {correct/total:.4f}")
        loss_val, acc_val, auc_val = evaluate_model_basic(student, val_loader)
        print(f"Epoch {ep+1} val: Loss {loss_val:.4f}, Acc {acc_val:.4f}, AUC {auc_val:.4f}")
    return student

# -------------------------
# PROFILING UTILITIES
# -------------------------
def profile_model(model, input_size=(1,3,224,224)):
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(input_size).to(DEVICE)
        flops = FlopCountAnalysis(model, dummy).total()
    except:
        flops = 0
    params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel()*4 for p in model.parameters())/1e6
    return flops, params, model_size_mb

def benchmark_model(model, loader):
    model.eval()
    start = time.time()
    peak_ram = 0
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        with torch.no_grad():
            _ = model(imgs)
        peak_ram = max(peak_ram, torch.cuda.max_memory_allocated(DEVICE)/1e6)
    elapsed = time.time() - start
    time_per_batch32 = elapsed / len(loader)
    return time_per_batch32, peak_ram

# -------------------------
# MAIN LOOP: PRUNE + KD + METRICS
# -------------------------
methods = ["l1", "bn_gamma"]
results = []

for method in methods:
    for ratio in PRUNE_RATIOS:
        print(f"\n=== Pruning method={method}, ratio={ratio} ===")
        # compute keeps per stage
        stage_keeps = {}
        for stage in ["layer1","layer2","layer3","layer4"]:
            stage_mod = getattr(baseline, stage)
            num_channels = sum([b.conv3.out_channels//b.conv3.in_channels for b in stage_mod.children()])
            keep_k = max(1, int(num_channels*(1-ratio)))
            keeps = compute_stage_importance_and_keeps(baseline, stage, keep_k, method=method)
            stage_keeps[stage] = keeps
        # build student
        student = build_pruned_resnet(stage_keeps)
        # finetune with KD
        student = train_student_kd(student, baseline, train_loader, val_loader)
        # evaluate on test
        test_loss, test_acc, test_auc = evaluate_model_basic(student, test_loader)
        # compute efficiency metrics
        flops, params, model_size_mb = profile_model(student)
        runtime, peak_ram = benchmark_model(student, test_loader)
        power_proxy = flops / (peak_ram*1e6) if peak_ram>0 else 0
        print(f"Metrics -> FLOPs:{flops}, Params:{params}, ModelSize:{model_size_mb:.2f}MB, Time/batch:{runtime:.4f}s, PeakRAM:{peak_ram:.2f}MB, Power proxy:{power_proxy:.2f}MFLOPs")
        # save model
        save_path = os.path.join(SAVE_DIR, f"dermamnist_resnet50_{method}_r{int(ratio*100)}_kd.pth")
        torch.save(student.state_dict(), save_path)
        # save results
        results.append({
            "method": method,
            "prune_ratio": ratio,
            "loss": test_loss,
            "acc": test_acc,
            "auc": test_auc,
            "flops": flops,
            "params": params,
            "model_size_MB": model_size_mb,
            "inference_time_s": runtime,
            "peak_RAM_MB": peak_ram,
            "power_proxy_MFLOPs": power_proxy,
            "model_path": save_path
        })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(SAVE_DIR, "pruned_kd_metrics.csv"), index=False)
print("All metrics saved to CSV.")
