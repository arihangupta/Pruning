#!/usr/bin/env python3
import os
import glob
import random
import csv
import numpy as np
import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
from torchvision import transforms, models

# ---------- CONFIG ----------
DATASETS_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets"
EXPERIMENT_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/experiment2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
# ----------------------------

def build_model(num_classes: int, in_channels: int = 3):
    """
    Build a ResNet-50 model with the right classifier head
    and input channel handling (1 or 3).
    """
    model = models.resnet50(weights=None)

    # Adjust the first conv layer if grayscale
    if in_channels == 1:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias
        )

    # Replace classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_one_image(npz_path):
    data = np.load(npz_path)
    images, labels = data["test_images"], data["test_labels"]
    idx = random.randint(0, len(images) - 1)
    img, label = images[idx], labels[idx]

    # ensure channel dimension
    if img.ndim == 2:  # (H, W)
        img = np.expand_dims(img, -1)

    img = torch.tensor(img).permute(2, 0, 1).float() / 255.0  # (C, H, W)
    in_channels = img.shape[0]

    # if grayscale, keep 1 channel for model conv1,
    # but duplicate for normalization (which expects 3 channels)
    if in_channels == 1:
        norm_img = img.repeat(3, 1, 1)
    else:
        norm_img = img

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    norm_img = transform(norm_img)

    # ensure label is a Python int
    label = int(label.item()) if isinstance(label, np.ndarray) else int(label)

    return img.unsqueeze(0), norm_img.unsqueeze(0), label, in_channels


def predict_with_energy(model, img, model_path, emissions_dir):
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

def main():
    datasets = ["bloodmnist", "dermamnist", "octmnist", "pathmnist", "tissuemnist"]
    for dset in datasets:
        print(f"\n=== {dset.upper()} ===")
        npz_path = os.path.join(DATASETS_DIR, f"{dset}_224.npz")

        raw_img, norm_img, true_label, in_channels = load_one_image(npz_path)

        labels = np.load(npz_path)["test_labels"]
        num_classes = len(np.unique(labels))

        model_files = [os.path.join(EXPERIMENT_DIR, dset, "baseline.pth")]
        model_files += sorted(glob.glob(os.path.join(EXPERIMENT_DIR, dset, "*_final.pth")))

        results = []
        for mpath in model_files:
            model = build_model(num_classes, in_channels).to(DEVICE)

            # load weights
            state_dict = torch.load(mpath, map_location=DEVICE)
            model.load_state_dict(state_dict)

            # use normalized image for prediction
            pred, energy_kwh = predict_with_energy(model, norm_img, mpath, os.path.join(EXPERIMENT_DIR, dset))

            print(f"Model: {os.path.basename(mpath):40s} | True: {true_label} | Pred: {pred} | kWh: {energy_kwh:.6f}")
            results.append({
                "model": os.path.basename(mpath),
                "true_label": true_label,
                "predicted_label": pred,
                "energy_kWh": energy_kwh,
            })

        # write one CSV per dataset
        out_csv = os.path.join(EXPERIMENT_DIR, dset, "predictions_with_energy.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"→ Saved aggregated results to {out_csv}")

if __name__ == "__main__":
    main()
