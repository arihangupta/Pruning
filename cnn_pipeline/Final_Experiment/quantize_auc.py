#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import resnet50
from sklearn.metrics import roc_auc_score
import os

# Config
MODEL_PATHS = {
    "pathmnist": "/home/arihangupta/Pruning/dinov2/Pruning/CNN_prune/pathmnist/quantization_r50compressed_final.pth",
    "dermamnist": "/home/arihangupta/Pruning/dinov2/Pruning/CNN_prune/dermamnist/quantization_r50compressed_final.pth",
    "bloodmnist": "/home/arihangupta/Pruning/dinov2/Pruning/CNN_prune/bloodmnist/quantization_r50compressed_final.pth"
}
DATASET_PATHS = {
    "pathmnist": "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced/pathmnist_224.npz",
    "dermamnist": "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced/dermamnist_224.npz",
    "bloodmnist": "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced/bloodmnist_224.npz"
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = {
    "pathmnist": 9,
    "dermamnist": 7,
    "bloodmnist": 8
}

class NumpyMemmapDataset(Dataset):
    def __init__(self, imgs_np, labels_np, img_size=IMG_SIZE):
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
        if idx < 5:
            print(f"Sample {idx}: img min={img.min()}, max={img.max()}, normalized min={x.min()}, max={x.max()}")
        return x, label

def load_model(model_path, num_classes):
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    try:
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
        dtype = next(iter(state_dict.values())).dtype
        print(f"State dictionary dtype: {dtype}")
        if dtype == torch.float16:
            model = model.half()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    model = model.to(DEVICE)
    model.eval()
    # Check for invalid weights
    for name, param in model.named_parameters():
        if torch.any(torch.isnan(param)) or torch.any(torch.isinf(param)):
            print(f"Warning: {name} contains NaN or Inf")
    return model

def evaluate_model_basic(model, loader, num_classes, dataset_name):
    model.eval()
    probs_list = []
    labels_list = []
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, 1):
            images = images.to(model_device)
            labels = labels.to(model_device)
            
            if model_dtype == torch.half:
                images = images.half()
            
            outputs = model(images)
            
            if outputs.device != labels.device:
                outputs = outputs.to(labels.device)
            
            # Clamp logits and compute softmax
            outputs = torch.clamp(outputs, min=-1000, max=1000)
            probs = torch.softmax(outputs, dim=1)
            probs_np = probs.cpu().numpy()
            
            # Check for invalid probabilities
            if np.any(np.isnan(probs_np)) or np.any(np.isinf(probs_np)):
                print(f"Warning: Batch {batch_idx} in {dataset_name} has invalid probabilities (NaN={np.any(np.isnan(probs_np))}, Inf={np.any(np.isinf(probs_np))})")
            # Check probability sums
            prob_sums = np.sum(probs_np, axis=1)
            if not np.allclose(prob_sums, 1.0, rtol=1e-3):
                print(f"Warning: Batch {batch_idx} in {dataset_name} has probability sums not close to 1.0: min={prob_sums.min()}, max={prob_sums.max()}")
                # Normalize probabilities
                probs_np = probs_np / np.sum(probs_np, axis=1, keepdims=True)
                # Re-check for NaN after normalization
                if np.any(np.isnan(probs_np)):
                    print(f"Warning: Batch {batch_idx} in {dataset_name} still has NaN after normalization")
            
            probs_list.append(probs_np)
            labels_list.append(labels.cpu().numpy())
    
    all_labels = np.concatenate(labels_list)
    all_probs = np.concatenate(probs_list)
    
    # Verify shapes
    print(f"{dataset_name}: Labels shape={all_labels.shape}, Probs shape={all_probs.shape}")
    
    # Check label distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    print(f"{dataset_name} test set label distribution: {label_dist}")
    
    auc = float("nan")
    try:
        if len(unique_labels) == num_classes:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovo")
            print(f"{dataset_name}: AUC calculated successfully: {auc:.4f}")
        else:
            print(f"{dataset_name}: Cannot compute AUC: Found {len(unique_labels)}/{num_classes} classes")
    except Exception as e:
        print(f"{dataset_name}: AUC calculation failed: {e}")
    return auc

results = {}
for dataset_name in ["pathmnist", "dermamnist", "bloodmnist"]:
    print(f"\nProcessing {dataset_name}")
    data_path = DATASET_PATHS[dataset_name]
    data = np.load(data_path)
    X_test, y_test = data["test_images"], data["test_labels"].flatten()
    test_ds = NumpyMemmapDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    model = load_model(MODEL_PATHS[dataset_name], NUM_CLASSES[dataset_name])
    auc = evaluate_model_basic(model, test_loader, NUM_CLASSES[dataset_name], dataset_name)
    results[dataset_name] = auc

print("\nFinal Results:")
for dataset, auc in results.items():
    print(f"{dataset}: AUC = {auc:.4f}")