#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        return x, label

def load_model(model_path, num_classes):
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    try:
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    model = model.half().to(DEVICE)
    model.eval()
    return model

def evaluate_model_basic(model, loader):
    model.eval()
    probs_list = []; labels_list = []
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
            
            probs = torch.softmax(torch.clamp(outputs, min=-100, max=100), dim=1).cpu().numpy()
            if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                print(f"Warning: Batch {batch_idx} has invalid probabilities")
            probs_list.append(probs)
            labels_list.append(labels.cpu().numpy())
    
    all_labels = np.concatenate(labels_list)
    all_probs = np.concatenate(probs_list)
    
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    print(f"Test set label distribution: {label_dist}")
    
    auc = float("nan")
    try:
        num_unique = len(unique_labels)
        if num_unique > 1:  # Need at least 2 classes for AUC
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovo")
            print(f"AUC calculated successfully: {auc}")
        else:
            print(f"Cannot compute AUC: Only {num_unique} unique class(es) found")
    except Exception as e:
        print(f"AUC calculation failed: {e}")
    return auc

results = {}
for dataset_name in ["pathmnist", "dermamnist", "bloodmnist"]:
    print(f"\nProcessing {dataset_name}")
    data_path = DATASET_PATHS[dataset_name]
    data = np.load(data_path)
    y_train = data['train_labels'].flatten()
    y_val = data['val_labels'].flatten()
    y_test = data['test_labels'].flatten()
    num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    print(f"Num classes: {num_classes}")
    
    X_test, y_test = data["test_images"], data["test_labels"].flatten()
    test_ds = NumpyMemmapDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    model_path = MODEL_PATHS[dataset_name]
    model = load_model(model_path, num_classes)
    
    auc = evaluate_model_basic(model, test_loader)
    results[dataset_name] = auc

print("\nFinal Results:")
for dataset, auc in results.items():
    print(f"{dataset}: AUC = {auc}")