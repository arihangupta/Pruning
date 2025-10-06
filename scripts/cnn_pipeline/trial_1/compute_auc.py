#!/usr/bin/env python3
import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH = "/home/arihangupta/Pruning/dinov2/Pruning"
DATASET_DIR = os.path.join(BASE_PATH, "datasets_balanced")
MODEL_DIR = os.path.join(BASE_PATH, "trial_1/pruned_models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

# The datasets and corresponding model folders
DATASETS = ["bloodmnist", "dermamnist", "pathmnist"]

FINAL_MODELS = [
    "baseline.pth",
    "quantization_r50compressed_final.pth",
    "pgto_regional_gradients_r50compressed_final.pth",
    "pgto_regional_gradients_amp_r50compressed_final_amp.pth",
    "slim_kd_r50compressed_final.pth",
    "slim_kd_amp_r50compressed_final_amp.pth",
]

# ============================================================
# DATA LOADING
# ============================================================
def load_npz_dataset(npz_path):
    data = np.load(npz_path)
    x_test, y_test = data["x_test"], data["y_test"]

    # Ensure float tensors with correct shape (N, C, H, W)
    if x_test.ndim == 3:  # missing channel dim
        x_test = np.expand_dims(x_test, axis=1)
    x_test = torch.tensor(x_test).permute(0, 3, 1, 2).float() / 255.0
    y_test = torch.tensor(y_test).long()

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader, len(np.unique(y_test.numpy()))

# ============================================================
# MODEL LOADING (ResNet)
# ============================================================
# NOTE: Replace this stub with your CustomResNet definition from training.
from your_module_or_script import CustomResNet, build_resnet50_for_load

def load_model(model_path, num_classes):
    # Try to infer architecture from checkpoint
    model = build_resnet50_for_load(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

# ============================================================
# EVALUATION (AUC)
# ============================================================
def compute_auc(model, loader, num_classes):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(DEVICE)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Handle multi-class case
    if num_classes > 2:
        y_true = np.eye(num_classes)[all_labels]
        auc = roc_auc_score(y_true, all_probs, multi_class="ovr")
    else:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    return auc

# ============================================================
# MAIN LOOP
# ============================================================
results = []

for dataset_name in DATASETS:
    npz_path = os.path.join(DATASET_DIR, f"{dataset_name}_224.npz")
    test_loader, num_classes = load_npz_dataset(npz_path)
    dataset_results = {}

    print(f"\n=== Evaluating on {dataset_name} ===")
    for model_file in FINAL_MODELS:
        model_path = os.path.join(MODEL_DIR, dataset_name, model_file)
        if not os.path.exists(model_path):
            print(f"Skipping {model_file} — not found.")
            continue

        print(f"-> {model_file}")
        model = load_model(model_path, num_classes)
        auc = compute_auc(model, test_loader, num_classes)
        dataset_results[model_file] = auc
        print(f"   AUC: {auc:.4f}")

    results.append((dataset_name, dataset_results))

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\n\n===== FINAL AUC RESULTS =====")
for dataset, res in results:
    print(f"\n{dataset}:")
    for model_name, auc in res.items():
        print(f"  {model_name:<50} {auc:.4f}")
