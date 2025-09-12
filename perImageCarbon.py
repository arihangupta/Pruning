#!/usr/bin/env python3
import os
import glob
import random
import csv
import math
import numpy as np
import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
from torchvision import transforms, models
from torchvision.models.resnet import Bottleneck

# ---------- CONFIG ----------
DATASETS_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets"
EXPERIMENT_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/experiment4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
NUM_IMAGES_LIST = [10] + list(range(50, 1001, 50))  # [10, 50, 100, 150, ..., 1000]
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

def build_model(num_classes: int, pruning_ratio=None):
    """
    Build a ResNet-50 model with the right classifier head and optional pruning.
    Always use in_channels=3 to match pruning code's expectation.
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
        in_channels=3  # Always 3 channels to match pruning code
    )
    return model

def load_images(npz_path, num_images):
    """
    Load num_images random test images and their labels from the dataset.
    Returns batched images, normalized images, and labels.
    """
    try:
        data = np.load(npz_path)
        images, labels = data["test_images"], data["test_labels"]
        available_images = len(images)
        num_images = min(num_images, available_images)  # Use available images if fewer than requested
        
        # Select random indices
        indices = random.sample(range(available_images), num_images)
        selected_images = images[indices]
        selected_labels = labels[indices]

        # Process images
        raw_imgs = []
        norm_imgs = []
        in_channels_list = []
        
        for img in selected_images:
            # Ensure channel dimension
            if img.ndim == 2:  # (H, W)
                img = np.expand_dims(img, -1)
            
            img = torch.tensor(img).permute(2, 0, 1).float() / 255.0  # (C, H, W)
            in_channels = img.shape[0]
            in_channels_list.append(in_channels)
            
            # Always convert to 3 channels to match pruning code's expectation
            if in_channels == 1:
                norm_img = img.repeat(3, 1, 1)
            else:
                norm_img = img
            
            # Apply normalization
            transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            norm_img = transform(norm_img)
            
            raw_imgs.append(img)
            norm_imgs.append(norm_img)
        
        # Stack images and labels
        raw_imgs = torch.stack(raw_imgs)  # (num_images, C, H, W)
        norm_imgs = torch.stack(norm_imgs)  # (num_images, 3, H, W)
        labels = torch.tensor([int(label.item()) if isinstance(label, np.ndarray) else int(label) 
                              for label in selected_labels])
        
        print(f"Loaded {num_images} images from {npz_path}: original channels={set(in_channels_list)}")
        return raw_imgs, norm_imgs, labels
    except Exception as e:
        print(f"Error loading images from {npz_path}: {e}")
        raise

def predict_with_energy(model, images, labels, model_path, emissions_dir):
    """
    Predict on a batch of images and measure energy consumption.
    Returns predictions, accuracy, number of correct predictions, and total energy consumption.
    """
    try:
        os.makedirs(emissions_dir, exist_ok=True)
        tracker = EmissionsTracker(
            project_name=os.path.basename(model_path),
            output_dir=emissions_dir,
            output_file=f"emissions_{len(labels)}.csv",  # Unique file per num_images
            log_level="error"
        )
        tracker.start()
        model.eval()
        correct = 0
        total = len(labels)
        
        with torch.no_grad():
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            correct = (preds.cpu() == labels).sum().item()
        
        energy_kwh = tracker.stop()  # Returns float
        accuracy = correct / total if total > 0 else 0.0
        
        return preds.cpu().tolist(), accuracy, correct, energy_kwh
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
            labels_np = np.load(npz_path)["test_labels"]
            num_classes = len(np.unique(labels_np))
        except Exception as e:
            print(f"Error loading labels for {dset}: {e}")
            continue

        model_files = [os.path.join(EXPERIMENT_DIR, dset, "baseline.pth")]
        model_files += sorted(glob.glob(os.path.join(EXPERIMENT_DIR, dset, "*_final.pth")))

        for num_images in NUM_IMAGES_LIST:
            print(f"\nProcessing {num_images} images")
            try:
                raw_imgs, norm_imgs, labels = load_images(npz_path, num_images)
            except Exception as e:
                print(f"Skipping {dset} for {num_images} images due to error: {e}")
                continue

            results = []
            total = len(labels)  # Actual number of images loaded
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

                    model = build_model(num_classes, pruning_ratio).to(DEVICE)

                    # Load weights with weights_only=True
                    state_dict = torch.load(mpath, map_location=DEVICE, weights_only=True)
                    model.load_state_dict(state_dict)

                    # Predict on batch of images
                    preds, accuracy, correct, energy_kwh = predict_with_energy(
                        model, norm_imgs, labels, mpath, os.path.join(EXPERIMENT_DIR, dset)
                    )

                    print(f"Model: {os.path.basename(mpath):40s} | "
                          f"Accuracy: {accuracy:.4f} ({correct}/{total}) | "
                          f"Energy kWh: {energy_kwh:.6f}")
                    results.append({
                        "model": os.path.basename(mpath),
                        "num_images": total,
                        "accuracy": accuracy,
                        "correct_predictions": correct,
                        "energy_kWh": energy_kwh,
                    })
                except Exception as e:
                    print(f"Error processing model {mpath} for {num_images} images: {e}")
                    continue
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # Clear GPU memory after each model

            # Write aggregated results for this num_images
            if results:
                try:
                    out_csv = os.path.join(EXPERIMENT_DIR, dset, f"predictions_with_energy_{num_images}.csv")
                    with open(out_csv, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=results[0].keys())
                        writer.writeheader()
                        writer.writerows(results)
                    print(f"→ Saved aggregated results to {out_csv}")
                    
                    # Optionally save individual predictions
                    individual_csv = os.path.join(EXPERIMENT_DIR, dset, f"individual_predictions_{num_images}.csv")
                    with open(individual_csv, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["model", "image_index", "true_label", "predicted_label"])
                        for mpath in model_files:
                            try:
                                model_name = os.path.basename(mpath)
                                pruning_ratio = None
                                if "r50" in mpath:
                                    pruning_ratio = 0.5
                                elif "r60" in mpath:
                                    pruning_ratio = 0.6
                                elif "r70" in mpath:
                                    pruning_ratio = 0.7
                                model = build_model(num_classes, pruning_ratio).to(DEVICE)
                                state_dict = torch.load(mpath, map_location=DEVICE, weights_only=True)
                                model.load_state_dict(state_dict)
                                preds, _, _, _ = predict_with_energy(
                                    model, norm_imgs, labels, mpath, os.path.join(EXPERIMENT_DIR, dset)
                                )
                                for idx, (true, pred) in enumerate(zip(labels.tolist(), preds)):
                                    writer.writerow([model_name, idx, true, pred])
                            except Exception as e:
                                print(f"Error writing individual predictions for {mpath} ({num_images} images): {e}")
                                continue
                    print(f"→ Saved individual predictions to {individual_csv}")
                except Exception as e:
                    print(f"Error writing CSV for {dset} ({num_images} images): {e}")

if __name__ == "__main__":
    main()
