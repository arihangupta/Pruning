import os
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

# ================
# Config
# ================
DATA_PATH = "/home/arihangupta/Pruning/dinov2/Pruning/datasets/dermamnist_224.npz"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/saved_models"
BASELINE_PATH = os.path.join(SAVE_DIR, "dermamnist_resnet50_BASELINE.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
BASE_EPOCHS = 5
FINETUNE_EPOCHS = 3
LEARNING_RATE = 1e-4
IMG_SIZE = 224
RATIOS = [0.1, 0.3, 0.5]   # pruning ratios
SEED = 42

# reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)

# ================
# Data
# ================
print("📦 Loading Dermamnist dataset...")
data = np.load(DATA_PATH)
X_train, y_train = data["train_images"], data["train_labels"].flatten()
X_val, y_val     = data["val_images"], data["val_labels"].flatten()
X_test, y_test   = data["test_images"], data["test_labels"].flatten()

def preprocess(arr):
    arr = arr.astype("float32") / 255.0
    if arr.ndim == 3 or arr.shape[-1] == 1:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    arr = np.transpose(arr, (0, 3, 1, 2))  # NHWC -> NCHW
    return arr

X_train, X_val, X_test = preprocess(X_train), preprocess(X_val), preprocess(X_test)

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.long))
val_ds   = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                         torch.tensor(y_val, dtype=torch.long))
test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                         torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(np.unique(y_train))
print(f"✅ Data loaded. num_classes={num_classes}, train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

# ================
# Model Utils
# ================
def build_resnet50(num_classes):
    model = models.resnet50(weights=None)  # no pretrained, will load weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_baseline():
    model = build_resnet50(num_classes)
    ckpt = torch.load(BASELINE_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt)
    return model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

def evaluate(model, loader):
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    acc = correct / total
    loss_avg = loss_total / total
    probs = np.vstack(all_probs)
    labels = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr")
    except:
        auc = np.nan
    return loss_avg, acc, auc

def train_one_model(model, train_loader, val_loader, epochs=5, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        total, correct, run_loss = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        val_loss, val_acc, val_auc = evaluate(model, val_loader)
        print(f"  Epoch {ep+1}/{epochs}: Train Loss {run_loss/total:.4f}, "
              f"Train Acc {correct/total:.4f}, Val Acc {val_acc:.4f}, Val AUC {val_auc:.4f}")
    return model

# ================
# Pruning Methods
# ================
def get_l1_importance(conv_layer):
    w = conv_layer.weight.data.abs().mean(dim=(1,2,3))
    return w

def get_bn_importance(bn_layer):
    return bn_layer.weight.data.abs()

def structured_prune(model, ratio, method="l1"):
    new_model = build_resnet50(num_classes).to(DEVICE)
    new_state = new_model.state_dict()
    old_state = model.state_dict()

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and ("layer" in name and "conv1" in name):
            if method == "l1":
                imp = get_l1_importance(module)
            else:
                bn_name = name.replace("conv1", "bn1")
                bn_layer = dict(model.named_modules())[bn_name]
                imp = get_bn_importance(bn_layer)
            n_keep = max(1, int((1-ratio)*len(imp)))
            keep_idx = torch.topk(imp, n_keep, largest=True).indices
            mask = torch.zeros_like(imp, dtype=torch.bool)
            mask[keep_idx] = True
            pruned_w = module.weight.data[mask]
            new_state[name+".weight"][:n_keep] = pruned_w
            if module.bias is not None:
                new_state[name+".bias"][:n_keep] = module.bias.data[mask]
        else:
            if name+".weight" in new_state:
                new_state[name+".weight"] = old_state[name+".weight"]
            if name+".bias" in new_state:
                new_state[name+".bias"] = old_state[name+".bias"]

    new_model.load_state_dict(new_state, strict=False)
    return new_model

# ================
# Main Run
# ================
results = []
for method in ["l1", "bn"]:
    for r in RATIOS:
        print(f"\n=== 🔪 Structured pruning method={method}, ratio={r} ===")
        base = load_baseline()
        pruned = structured_prune(base, ratio=r, method=method)

        # before finetuning
        test_loss, test_acc, test_auc = evaluate(pruned, test_loader)
        print(f"Before finetune: Acc={test_acc:.4f}, AUC={test_auc:.4f}")

        # finetune
        pruned = train_one_model(pruned, train_loader, val_loader,
                                 epochs=FINETUNE_EPOCHS, lr=LEARNING_RATE)
        ft_loss, ft_acc, ft_auc = evaluate(pruned, test_loader)
        print(f"After finetune: Acc={ft_acc:.4f}, AUC={ft_auc:.4f}")

        # save models
        tag = f"dermamnist_resnet50_structured_{method}_r{int(r*100)}"
        before_path = os.path.join(SAVE_DIR, tag+"_before.pth")
        after_path  = os.path.join(SAVE_DIR, tag+"_finetuned.pth")
        torch.save(pruned.state_dict(), after_path)
        print(f"💾 Saved pruned model to {after_path}")

        results.append({
            "Method": method,
            "Ratio": r,
            "Acc_before": test_acc,
            "AUC_before": test_auc,
            "Acc_after": ft_acc,
            "AUC_after": ft_auc,
            "Path_before": before_path,
            "Path_after": after_path
        })

# ================
# Save summary
# ================
df = pd.DataFrame(results)
csv_path = os.path.join(SAVE_DIR, "dermamnist_structured_summary.csv")
df.to_csv(csv_path, index=False)
print(f"\n📝 Summary saved to {csv_path}")
print(df)
