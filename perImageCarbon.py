#!/usr/bin/env python3
import os
import glob
import random
import csv
import math  # Added math import
import numpy as np
import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
from torchvision import transforms, models
from torchvision.models.resnet import Bottleneck

# ---------- CONFIG ----------
DATASETS_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets"
EXPERIMENT_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/experiment2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
# ----------------------------

class CustomResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], stage_planes=[64, 128, 256, 512], num_classes=1000, in_channels=3):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage_planes = stage_planes[:]
        self.layers_cfg = layers[:]
        self.layer1 = self._make_layer(block, stage_planes[0], layers[0])
        self.layer2 = self._make_layer(block, stage_planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, stage_planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, stage_planes[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_planes[3] * block.expansion, num_classes)

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

def build_model(num_classes: int, in_channels: int = 3, pruning_ratio=None):
    """
    Build a ResNet-50 model with the right classifier head, input channels, and optional pruning.
    pruning_ratio: None for baseline, or 0.5, 0.6, 0.7 for pruned models.
    """
    if pruning_ratio is None:
        # Baseline ResNet-50
        stage_planes = [64, 128, 256, 512]
    else:
        # Pruned model with reduced channels
        stage_planes = [
            max(1, int(math.floor(64 * (1.0 - pruning_ratio)))),
            max(1, int(math.floor(128 * (1.0 - pruning_ratio)))),
            max(1, int(math.floor(256 * (1.0 - pruning_ratio)))),
            max(1, int(math.floor(512 * (1.0 - pruning_ratio))))
        ]
    model = CustomResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        stage_planes=stage_planes,
        num_classes=num_classes,
        in_channels=in_channels
    )
    return model

def load_one_image(npz_path):
    try:
        data = np.load(npz_path)
        images, labels = data["test_images"], data["test_labels"]
        idx = random.randint(0, len(images) - 1)
        img, label = images[idx], labels[idx]

        # Ensure channel dimension
        if img.ndim == 2:  # (H, W)
            img = np.expand_dims(img, -1)

        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0  # (C, H, W)
        in_channels = img.shape[0]

        # If grayscale, keep 1 channel for model conv1, but duplicate for normalization
        if in_channels == 1:
            norm_img = img.repeat(3, 1, 1)
        else:
            norm_img = img

        # Use same normalization as pruning code
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        norm_img = transform(norm_img)

        # Ensure label is a Python int
        label = int(label.item()) if isinstance(label, np.ndarray) else int(label)

        return img.unsqueeze(0), norm_img.unsqueeze(0), label, in_channels
    except Exception as e:
        print(f"Error loading image from {npz_path}: {e}")
        raise

def predict_with_energy(model, img, model_path, emissions_dir):
    try:
        os.makedirs(emissions_dir, exist_ok=True)
        tracker = EmissionsTracker(
            project_name=os.path.basename(model_path),
            output_dir=emissions_dir,
            output_file="emissions.csv",
            log_level="error"
        )
        tracker.start()
        model.eval()
        with torch.no_grad():
            out = model(img.to(DEVICE))
            pred = torch.argmax(out, dim=1).item()
        energy_kwh = tracker.stop()  # returns float
        return pred, energy_kwh
    except Exception as e:
        print(f"Error in predict_with_energy for {model_path}: {e}")
        raise
    finally:
        if 'tracker' in locals():
            tracker.stop()  # Ensure tracker is stopped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory

def main():
    datasets = ["bloodmnist", "dermamnist", "octmnist", "pathmnist", "tissuemnist"]
    for dset in datasets:
        print(f"\n=== {dset.upper()} ===")
        npz_path = os.path.join(DATASETS_DIR, f"{dset}_224.npz")

        try:
            raw_img, norm_img, true_label, in_channels = load_one_image(npz_path)
        except Exception as e:
            print(f"Skipping {dset} due to error: {e}")
            continue

        try:
            labels = np.load(npz_path)["test_labels"]
            num_classes = len(np.unique(labels))
        except Exception as e:
            print(f"Error loading labels for {dset}: {e}")
            continue

        model_files = [os.path.join(EXPERIMENT_DIR, dset, "baseline.pth")]
        model_files += sorted(glob.glob(os.path.join(EXPERIMENT_DIR, dset, "*_final.pth")))

        results = []
        for mpath in model_files:
            try:
                # Determine pruning ratio from model file name
                pruning_ratio = None
                if "r50" in mpath:
                    pruning_ratio = 0.5
                elif "r60" in mpath:
                    pruning_ratio = 0.6
                elif "r70" in mpath:
                    pruning_ratio = 0.7

                model = build_model(num_classes, in_channels, pruning_ratio).to(DEVICE)

                # Load weights with weights_only=True
                state_dict = torch.load(mpath, map_location=DEVICE, weights_only=True)
                model.load_state_dict(state_dict)

                # Use normalized image for prediction
                pred, energy_kwh = predict_with_energy(model, norm_img, mpath, os.path.join(EXPERIMENT_DIR, dset))

                print(f"Model: {os.path.basename(mpath):40s} | True: {true_label} | Pred: {pred} | kWh: {energy_kwh:.6f}")
                results.append({
                    "model": os.path.basename(mpath),
                    "true_label": true_label,
                    "predicted_label": pred,
                    "energy_kWh": energy_kwh,
                })
            except Exception as e:
                print(f"Error processing model {mpath}: {e}")
                continue
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear GPU memory after each model

        # Write one CSV per dataset
        if results:
            try:
                out_csv = os.path.join(EXPERIMENT_DIR, dset, "predictions_with_energy.csv")
                with open(out_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
                print(f"→ Saved aggregated results to {out_csv}")
            except Exception as e:
                print(f"Error writing CSV for {dset}: {e}")

if __name__ == "__main__":
    main()