"""
single_image_benchmark.py

Single-image inference benchmarking for CNN (ResNet50-based) and ViT (DeiT/ViT)
compressed models across MedMNIST datasets.

Design:
  - Load each model once, run 5 warmup passes, then time 10 individual images
    one-at-a-time across 3 independent repetitions.
  - Energy is tracked per image (CodeCarbon start/stop per image).
  - CPU and CUDA devices are both tested for every model.
  - Results are written to:
      single_image_benchmarking/
      ├── CNN/
      │   ├── {dataset}_results.csv
      │   └── {dataset}_per_image_log.csv
      └── ViT/
          ├── {dataset}_results.csv
          └── {dataset}_per_image_log.csv

Usage:
  python single_image_benchmark.py [--model-type CNN] [--model-type ViT]
                                   [--datasets bloodmnist dermamnist pathmnist chestmnist]
                                   [--devices cpu cuda]
                                   [--reps 3]
                                   [--n-images 10]
                                   [--warmup 5]
                                   [--output-dir /path/to/single_image_benchmarking]
"""

import argparse
import csv
import gc
import hashlib
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — adjust to your filesystem layout
# ---------------------------------------------------------------------------
CNN_MODEL_ROOT = Path(
    "/home/arihangupta/Pruning/dinov2/Pruning/PruneAndTrain/CNN/CNN_pruned_models"
)
VIT_BASELINE_ROOT = Path(
    "/home/arihangupta/Pruning/dinov2/Pruning/PruneAndTrain/Vision/rerun/new_baseline"
)
VIT_PRUNED_ROOT = Path(
    "/home/arihangupta/Pruning/dinov2/Pruning/PruneAndTrain/Vision/rerun/pruned_models"
)
MEDMNIST_ROOT = Path("/home/arihangupta/Pruning/dinov2/data")  # contains *.npz / *.npy

DATASETS = ["bloodmnist", "dermamnist", "pathmnist", "chestmnist"]

# Number of classes per dataset (needed for model heads)
NUM_CLASSES = {
    "bloodmnist": 8,
    "dermamnist": 7,
    "pathmnist": 9,
    "chestmnist": 14,
}

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_test_images(dataset: str, n: int, seed: int) -> torch.Tensor:
    """
    Load `n` random test images from the MedMNIST dataset.

    Tries NPZ first (MedMNIST v2 standard), falls back to NPY.
    Returns a list of (original_idx, tensor) tuples where tensor shape is
    (1, C, H, W), normalised to [0, 1] (float32), for one-at-a-time inference.
    """
    npz_path = MEDMNIST_ROOT / f"{dataset}.npz"
    npy_path = MEDMNIST_ROOT / dataset / "test_images.npy"

    if npz_path.exists():
        data = np.load(npz_path)
        imgs = data["test_images"]  # shape: (N, H, W) or (N, H, W, C)
    elif npy_path.exists():
        imgs = np.load(npy_path)
    else:
        raise FileNotFoundError(
            f"Cannot find test data for {dataset}. "
            f"Tried {npz_path} and {npy_path}."
        )

    rng = random.Random(seed)
    indices = rng.sample(range(len(imgs)), min(n, len(imgs)))

    tensors = []
    for idx in indices:
        img = imgs[idx]
        # Grayscale → RGB
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            img = np.concatenate([img, img, img], axis=-1)
        # (H, W, C) → (C, H, W)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        # Resize to 224×224 if needed
        t = torch.from_numpy(img).unsqueeze(0)  # (1, C, H, W)
        if t.shape[-1] != 224 or t.shape[-2] != 224:
            t = torch.nn.functional.interpolate(
                t, size=(224, 224), mode="bilinear", align_corners=False
            )
        tensors.append((idx, t))

    return tensors  # list of (original_idx, tensor)


# ---------------------------------------------------------------------------
# CNN model definitions (from CNN_benchmarking.py)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class CustomResNet(nn.Module):
    """Flexible ResNet that accepts variable stage_planes (supports pruned widths)."""

    def __init__(self, block, layers, stage_planes, num_classes=1000):
        super().__init__()
        self.in_planes = stage_planes[0]
        self.conv1 = nn.Conv2d(3, stage_planes[0], 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(stage_planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, stage_planes[0], layers[0])
        self.layer2 = self._make_layer(block, stage_planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, stage_planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, stage_planes[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_planes[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _infer_stage_planes_from_state_dict(sd: dict, block_expansion: int = 4) -> List[int]:
    """Infer stage_planes from checkpoint keys."""
    planes = []
    for layer_idx in range(1, 5):
        key = f"layer{layer_idx}.0.conv1.weight"
        if key in sd:
            planes.append(sd[key].shape[0])
        else:
            # Bottleneck last conv
            key2 = f"layer{layer_idx}.0.conv3.weight"
            if key2 in sd:
                planes.append(sd[key2].shape[0] // block_expansion)
            else:
                planes.append(64 * (2 ** (layer_idx - 1)))  # fallback
    return planes


def _is_bottleneck(sd: dict) -> bool:
    return any("conv3" in k for k in sd.keys() if "layer" in k)


def load_cnn_model(model_path: Path, num_classes: int) -> nn.Module:
    """Load a CNN (ResNet50-based) checkpoint into CustomResNet."""
    sd = torch.load(model_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # Strip DataParallel prefix
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    block = Bottleneck if _is_bottleneck(sd) else BasicBlock
    stage_planes = _infer_stage_planes_from_state_dict(sd, block.expansion)

    # Infer in_planes (stem) from conv1.weight
    in_planes = sd["conv1.weight"].shape[0] if "conv1.weight" in sd else stage_planes[0]
    # Make sure in_planes matches stage_planes[0] for layer1 construction
    stage_planes[0] = in_planes

    model = CustomResNet(block, [3, 4, 6, 3], stage_planes, num_classes=num_classes)
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys:
        log.warning(f"CNN load_state_dict missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        log.warning(f"CNN load_state_dict unexpected keys: {result.unexpected_keys}")

    # Cast to fp32 in case stored as fp16
    model = model.float()
    return model


# ---------------------------------------------------------------------------
# ViT model definitions (DeployVisionTransformer for oneshot pruned models)
# ---------------------------------------------------------------------------

class DeployMlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class DeployAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DeployBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DeployAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = DeployMlp(dim, mlp_hidden, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DeployVisionTransformer(nn.Module):
    """Physically smaller ViT (post oneshot pruning) that infers dims from checkpoint."""

    def __init__(self, embed_dim, depth, num_heads, mlp_ratio, num_classes, img_size=224, patch_size=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            DeployBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def _infer_vit_deploy_dims(sd: dict) -> Dict[str, int]:
    """Infer embed_dim, depth, num_heads, mlp_ratio from a oneshot checkpoint."""
    embed_dim = sd["patch_embed.weight"].shape[0]
    depth = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".norm1.weight"))
    # num_heads: assume head_dim=64 (standard ViT convention), fall back to 6
    qkv_key = "blocks.0.attn.qkv.weight"
    if qkv_key in sd:
        head_dim = 64
        num_heads = max(1, embed_dim // head_dim)
        if embed_dim % head_dim != 0:
            log.warning(
                f"embed_dim={embed_dim} not divisible by head_dim={head_dim}; "
                f"num_heads={num_heads} may be incorrect for this pruned model."
            )
    else:
        num_heads = 6  # fallback
    mlp_hidden_key = "blocks.0.mlp.fc1.weight"
    if mlp_hidden_key in sd:
        mlp_hidden = sd[mlp_hidden_key].shape[0]
        mlp_ratio = mlp_hidden / embed_dim
    else:
        mlp_ratio = 4.0
    return dict(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)


def load_vit_model(model_path: Path, num_classes: int) -> nn.Module:
    """
    Load a ViT checkpoint. Branches based on filename prefix:
      - _pretrained.pth   → timm standard ViT (fp32)
      - kd_*_final.pth    → timm student ViT (fp32)
      - quantization_*    → timm ViT cast to fp16
      - oneshot_*         → DeployVisionTransformer (custom, pruned dims)
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for ViT loading: pip install timm")

    name = model_path.name
    sd = torch.load(model_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    def _arch_from_name(stem: str) -> str:
        """Extract timm arch name (e.g. vit_tiny_patch16_224) from a filename component."""
        for size in ["vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224"]:
            if size in stem:
                return size
        raise ValueError(f"Cannot infer ViT arch from: {stem}")

    def _load_and_warn(model: nn.Module, sd: dict, label: str) -> nn.Module:
        result = model.load_state_dict(sd, strict=False)
        if result.missing_keys:
            log.warning(f"ViT [{label}] load_state_dict missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            log.warning(f"ViT [{label}] load_state_dict unexpected keys: {result.unexpected_keys}")
        return model

    if name.startswith("oneshot_"):
        dims = _infer_vit_deploy_dims(sd)
        model = DeployVisionTransformer(
            embed_dim=dims["embed_dim"],
            depth=dims["depth"],
            num_heads=dims["num_heads"],
            mlp_ratio=dims["mlp_ratio"],
            num_classes=num_classes,
        )
        _load_and_warn(model, sd, name)

    elif name.startswith("quantization_"):
        arch = _arch_from_name(name)
        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        _load_and_warn(model, sd, name)
        model = model.half()  # fp16, stays fp16 on both CPU and CUDA

    elif name.startswith("kd_"):
        # Student is the smaller model — infer from "to_{arch}" portion
        # e.g. kd_vit_base_patch16_224_to_vit_small_patch16_224_bloodmnist_final.pth
        # student arch appears after "to_"
        after_to = name.split("_to_", 1)[-1]  # "vit_small_patch16_224_bloodmnist_final.pth"
        arch = _arch_from_name(after_to)
        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        _load_and_warn(model, sd, name)

    else:
        # _pretrained.pth or any other standard checkpoint
        arch = _arch_from_name(name)
        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        _load_and_warn(model, sd, name)

    return model


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def _cnn_model_filter(fname: str) -> bool:
    """Accept baseline.pth and the 7 final compressed variants; reject intermediates."""
    if fname == "baseline.pth":
        return True
    if fname.endswith("_final.pth"):
        # Must not be an intermediate step file
        bad = ["pgto_", "_afterKD", "_preKD", "_allpruned", "_calibrated", "_postprune", "_pre_kd"]
        return not any(b in fname for b in bad)
    return False


def discover_cnn_models(dataset: str) -> List[Dict]:
    """Return list of dicts with keys: path, model_name, pruning_method, stored_precision."""
    base = CNN_MODEL_ROOT / dataset
    if not base.exists():
        log.warning(f"CNN dataset dir not found: {base}")
        return []

    models = []
    for f in sorted(base.iterdir()):
        if not _cnn_model_filter(f.name):
            continue

        name = f.stem  # e.g. "hybrid_pruning_fp16_r50compressed_final"
        if name == "baseline":
            method = "baseline"
            precision = "fp32"
        elif "fp16" in name:
            method = name.split("_fp16")[0]
            precision = "fp16"
        elif "quantization" in name:
            method = "quantization"
            precision = "fp16"  # confirmed fp16
        else:
            method = name.replace("_r50compressed_final", "")
            precision = "fp32"

        models.append({
            "path": f,
            "model_name": name,
            "pruning_method": method,
            "stored_precision": precision,
            "model_size_mb": f.stat().st_size / 1e6,
        })

    return models


def _vit_model_filter(fname: str) -> bool:
    """Accept only final compressed models and pretrained baselines."""
    if fname.endswith("_pretrained.pth"):
        return True
    if fname.endswith("_final.pth"):
        return True
    return False


def discover_vit_models(dataset: str) -> List[Dict]:
    """Return list of dicts for all ViT models (baselines + compressed) for a dataset."""
    models = []

    # Baselines
    baseline_dir = VIT_BASELINE_ROOT
    if baseline_dir.exists():
        for f in sorted(baseline_dir.iterdir()):
            if dataset in f.name and f.name.endswith("_pretrained.pth"):
                arch = None
                for size in ["base", "small", "tiny"]:
                    if f"vit_{size}" in f.name:
                        arch = f"vit_{size}_patch16_224"
                        break
                models.append({
                    "path": f,
                    "model_name": f.stem,
                    "pruning_method": "baseline",
                    "stored_precision": "fp32",
                    "architecture": arch,
                    "model_size_mb": f.stat().st_size / 1e6,
                })

    # Compressed models
    pruned_dir = VIT_PRUNED_ROOT / dataset
    if pruned_dir.exists():
        for f in sorted(pruned_dir.iterdir()):
            if not f.name.endswith("_final.pth"):
                continue

            name = f.stem
            arch = None
            for size in ["base", "small", "tiny"]:
                if f"vit_{size}" in name:
                    arch = f"vit_{size}_patch16_224"
                    break

            # For KD, the relevant arch is the student (smaller one)
            if name.startswith("kd_"):
                after_to = name.split("_to_", 1)[-1] if "_to_" in name else name
                for size in ["base", "small", "tiny"]:
                    if f"vit_{size}" in after_to:
                        arch = f"vit_{size}_patch16_224"
                        break

            method = name.split("_")[0] if "_" in name else name
            precision = "fp16" if name.startswith("quantization") else "fp32"

            models.append({
                "path": f,
                "model_name": name,
                "pruning_method": method,
                "stored_precision": precision,
                "architecture": arch,
                "model_size_mb": f.stat().st_size / 1e6,
            })

    return models


# ---------------------------------------------------------------------------
# Benchmarking core
# ---------------------------------------------------------------------------

def _safe_codecarbon():
    """Import CodeCarbon, return None if unavailable."""
    try:
        from codecarbon import EmissionsTracker
        return EmissionsTracker
    except ImportError:
        log.warning("codecarbon not installed — energy metrics will be NaN.")
        return None


def run_single_image_bench(
    model: nn.Module,
    images: List[Tuple[int, torch.Tensor]],
    device: str,
    codecarbon_output_dir: Path,
    warmup: int = 5,
) -> Tuple[List[Dict], Dict]:
    """
    Run single-image benchmarking.

    Args:
        model: Already-loaded, eval-mode model.
        images: List of (original_idx, tensor) tuples. Tensor shape: (1, C, H, W).
        device: "cpu" or "cuda"
        codecarbon_output_dir: Directory for CodeCarbon CSVs.
        warmup: Number of warmup passes.

    Returns:
        (per_image_rows, aggregated_stats)
    """
    EmissionsTracker = _safe_codecarbon()
    use_cuda_sync = device.startswith("cuda") and torch.cuda.is_available()

    # Cast input to fp16 if model is fp16
    is_fp16 = next(model.parameters()).dtype == torch.float16

    # Move model to device
    model = model.to(device)
    model.eval()

    # Warmup
    dummy = torch.randn(1, 3, 224, 224, device=device)
    if is_fp16:
        dummy = dummy.half()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            if use_cuda_sync:
                torch.cuda.synchronize()
    del dummy

    codecarbon_output_dir.mkdir(parents=True, exist_ok=True)

    per_image_rows = []
    latencies_ms = []
    energies_kwh = []
    cpu_powers, gpu_powers, ram_powers = [], [], []

    for img_idx, (orig_idx, img_tensor) in enumerate(images):
        img = img_tensor.to(device)
        if is_fp16:
            img = img.half()

        # Start energy tracker
        if EmissionsTracker is not None:
            tracker = EmissionsTracker(
                project_name="single_img_bench",
                output_dir=str(codecarbon_output_dir),
                log_level="error",
                save_to_file=False,
                measure_power_secs=0.1,
            )
            tracker.start()

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(img)
        if use_cuda_sync:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        energy_kwh = float("nan")
        cpu_power = float("nan")
        gpu_power = float("nan")
        ram_power = float("nan")

        if EmissionsTracker is not None:
            try:
                emissions = tracker.stop()
                # emissions is kgCO2eq; we want energy in kWh
                # tracker.final_emissions_data has energy breakdown
                data = tracker.final_emissions_data
                energy_kwh = float(getattr(data, "energy_consumed", float("nan")))
                cpu_power = float(getattr(data, "cpu_power", float("nan")))
                gpu_power = float(getattr(data, "gpu_power", float("nan")))
                ram_power = float(getattr(data, "ram_power", float("nan")))
            except Exception as e:
                log.debug(f"CodeCarbon read error: {e}")

        latency_ms = (t1 - t0) * 1000.0
        latencies_ms.append(latency_ms)
        energies_kwh.append(energy_kwh)
        cpu_powers.append(cpu_power)
        gpu_powers.append(gpu_power)
        ram_powers.append(ram_power)

        per_image_rows.append({
            "image_idx": img_idx,
            "image_sample_idx": orig_idx,
            "latency_ms": latency_ms,
            "energy_kwh": energy_kwh,
            "cpu_power_w": cpu_power,
            "gpu_power_w": gpu_power,
            "ram_power_w": ram_power,
        })

    arr = np.array(latencies_ms)
    mean_latency = float(np.mean(arr))
    throughput = 1000.0 / mean_latency if mean_latency > 0 else float("nan")

    aggregated = {
        "mean_latency_ms": mean_latency,
        "std_latency_ms": float(np.std(arr)),
        "median_latency_ms": float(np.median(arr)),
        "p25_latency_ms": float(np.percentile(arr, 25)),
        "p75_latency_ms": float(np.percentile(arr, 75)),
        "p90_latency_ms": float(np.percentile(arr, 90)),
        "min_latency_ms": float(np.min(arr)),
        "max_latency_ms": float(np.max(arr)),
        "throughput_imgs_per_s": throughput,
        "mean_energy_kwh_per_image": float(np.nanmean(energies_kwh)),
        "mean_cpu_power_w": float(np.nanmean(cpu_powers)),
        "mean_gpu_power_w": float(np.nanmean(gpu_powers)),
        "mean_ram_power_w": float(np.nanmean(ram_powers)),
    }

    return per_image_rows, aggregated


# ---------------------------------------------------------------------------
# CSV I/O helpers
# ---------------------------------------------------------------------------

RESULTS_COLS = [
    "model_type", "dataset", "model_name", "pruning_method", "stored_precision",
    "architecture", "device", "rep",
    "mean_latency_ms", "std_latency_ms", "median_latency_ms",
    "p25_latency_ms", "p75_latency_ms", "p90_latency_ms",
    "min_latency_ms", "max_latency_ms",
    "throughput_imgs_per_s",
    "mean_energy_kwh_per_image", "mean_cpu_power_w", "mean_gpu_power_w", "mean_ram_power_w",
    "num_params", "model_size_mb", "num_images", "seed", "timestamp",
]

PER_IMAGE_COLS = [
    "model_type", "dataset", "model_name", "device", "rep",
    "image_idx", "image_sample_idx",
    "latency_ms", "energy_kwh", "cpu_power_w", "gpu_power_w", "ram_power_w",
]


def _ensure_csv(path: Path, cols: List[str]):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()


def _append_row(path: Path, cols: List[str], row: Dict):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writerow(row)


def _load_existing_keys(path: Path) -> set:
    """Load set of (model_name, device, rep) tuples already in the results CSV."""
    if not path.exists():
        return set()
    keys = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row.get("model_name"), row.get("device"), str(row.get("rep"))))
    return keys


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def benchmark_model_type(
    model_type: str,
    datasets: List[str],
    devices: List[str],
    n_reps: int,
    n_images: int,
    warmup: int,
    output_root: Path,
):
    out_dir = output_root / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        log.info(f"=== {model_type} / {dataset} ===")
        n_classes = NUM_CLASSES.get(dataset, 10)

        results_csv = out_dir / f"{dataset}_results.csv"
        per_img_csv = out_dir / f"{dataset}_per_image_log.csv"
        _ensure_csv(results_csv, RESULTS_COLS)
        _ensure_csv(per_img_csv, PER_IMAGE_COLS)
        existing_keys = _load_existing_keys(results_csv)

        if model_type == "CNN":
            model_infos = discover_cnn_models(dataset)
        else:
            model_infos = discover_vit_models(dataset)

        if not model_infos:
            log.warning(f"No models found for {model_type}/{dataset}, skipping.")
            continue

        log.info(f"  Found {len(model_infos)} model checkpoints.")

        for info in model_infos:
            model_name = info["model_name"]
            model_path = info["path"]
            model_size_mb = info["model_size_mb"]
            pruning_method = info["pruning_method"]
            stored_precision = info["stored_precision"]
            architecture = info.get("architecture", "resnet50")

            for device in devices:
                # Check CUDA availability
                if device == "cuda" and not torch.cuda.is_available():
                    log.warning("CUDA requested but not available, skipping cuda device.")
                    continue

                for rep in range(n_reps):
                    key = (model_name, device, str(rep))
                    if key in existing_keys:
                        log.info(f"    SKIP (already done): {model_name} | {device} | rep={rep}")
                        continue

                    seed = rep * 1000 + int(hashlib.md5(model_name.encode()).hexdigest(), 16) % 997
                    seed = seed % (2 ** 31)

                    log.info(f"    {model_name} | {device} | rep={rep} | seed={seed}")

                    # Load images
                    try:
                        images = load_test_images(dataset, n_images, seed)
                    except FileNotFoundError as e:
                        log.error(f"      Dataset load failed: {e}")
                        continue

                    # Load model
                    try:
                        if model_type == "CNN":
                            model = load_cnn_model(model_path, n_classes)
                        else:
                            model = load_vit_model(model_path, n_classes)
                    except Exception as e:
                        log.error(f"      Model load failed: {e}")
                        continue

                    num_params = sum(p.numel() for p in model.parameters())

                    # CodeCarbon tmp dir per run
                    cc_dir = out_dir / "codecarbon_tmp" / dataset / model_name / device / f"rep{rep}"

                    try:
                        per_image_rows, agg = run_single_image_bench(
                            model=model,
                            images=images,
                            device=device,
                            codecarbon_output_dir=cc_dir,
                            warmup=warmup,
                        )
                    except Exception as e:
                        log.error(f"      Benchmark failed: {e}", exc_info=True)
                        del model
                        gc.collect()
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        continue

                    ts = datetime.now(timezone.utc).isoformat()

                    # Write per-image rows
                    for row in per_image_rows:
                        row.update({
                            "model_type": model_type,
                            "dataset": dataset,
                            "model_name": model_name,
                            "device": device,
                            "rep": rep,
                        })
                        _append_row(per_img_csv, PER_IMAGE_COLS, row)

                    # Write aggregated row
                    result_row = {
                        "model_type": model_type,
                        "dataset": dataset,
                        "model_name": model_name,
                        "pruning_method": pruning_method,
                        "stored_precision": stored_precision,
                        "architecture": architecture,
                        "device": device,
                        "rep": rep,
                        **agg,
                        "num_params": num_params,
                        "model_size_mb": round(model_size_mb, 3),
                        "num_images": len(per_image_rows),
                        "seed": seed,
                        "timestamp": ts,
                    }
                    _append_row(results_csv, RESULTS_COLS, result_row)
                    existing_keys.add(key)

                    log.info(
                        f"      → mean latency: {agg['mean_latency_ms']:.2f} ms | "
                        f"throughput: {agg['throughput_imgs_per_s']:.1f} img/s | "
                        f"energy: {agg['mean_energy_kwh_per_image']:.2e} kWh"
                    )

                    # Cleanup
                    del model
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()

    log.info(f"=== {model_type} benchmarking complete ===")


# ---------------------------------------------------------------------------
# Analysis / summary (runs after all data collected)
# ---------------------------------------------------------------------------

def generate_analysis(output_root: Path):
    """Aggregate results across datasets, write global_summary.csv, basic plots."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("pandas or matplotlib not installed — skipping analysis.")
        return

    all_rows = []
    for model_type in ["CNN", "ViT"]:
        type_dir = output_root / model_type
        if not type_dir.exists():
            continue
        for csv_path in type_dir.glob("*_results.csv"):
            try:
                df = pd.read_csv(csv_path)
                all_rows.append(df)
            except Exception as e:
                log.warning(f"Could not read {csv_path}: {e}")

    if not all_rows:
        log.warning("No results CSVs found for analysis.")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    # Aggregate over reps: mean of aggregated metrics
    numeric_cols = [
        "mean_latency_ms", "std_latency_ms", "median_latency_ms",
        "p90_latency_ms", "throughput_imgs_per_s",
        "mean_energy_kwh_per_image", "num_params", "model_size_mb",
    ]
    group_cols = ["model_type", "dataset", "model_name", "pruning_method",
                  "stored_precision", "architecture", "device"]

    summary = (
        combined.groupby(group_cols)[numeric_cols]
        .mean()
        .reset_index()
    )

    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(analysis_dir / "global_summary.csv", index=False)
    log.info(f"Wrote global summary: {analysis_dir / 'global_summary.csv'}")

    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Mean latency CPU vs GPU per model (first dataset as example)
    try:
        pivot = summary.pivot_table(
            index=["model_type", "model_name"],
            columns="device",
            values="mean_latency_ms",
            aggfunc="mean",
        ).reset_index()
        fig, ax = plt.subplots(figsize=(14, 6))
        x = range(len(pivot))
        labels = [f"{r.model_type}/{r.model_name[:20]}" for _, r in pivot.iterrows()]
        if "cpu" in pivot.columns:
            ax.bar([i - 0.2 for i in x], pivot["cpu"], width=0.4, label="CPU")
        if "cuda" in pivot.columns:
            ax.bar([i + 0.2 for i in x], pivot["cuda"], width=0.4, label="CUDA")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("Mean Latency (ms)")
        ax.set_title("Single-Image Latency: CPU vs GPU")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plots_dir / "latency_cpu_vs_gpu.png", dpi=150)
        plt.close(fig)
        log.info("Saved latency_cpu_vs_gpu.png")
    except Exception as e:
        log.warning(f"Plot 1 failed: {e}")

    # Plot 2: Mean energy per image
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        sub = summary[summary["device"] == "cuda"] if "cuda" in summary["device"].values else summary
        labels = [f"{r.model_type}/{r.model_name[:20]}" for _, r in sub.iterrows()]
        ax.bar(range(len(sub)), sub["mean_energy_kwh_per_image"])
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("Mean Energy (kWh/image)")
        ax.set_title("Energy per Single Inference")
        plt.tight_layout()
        fig.savefig(plots_dir / "energy_per_image.png", dpi=150)
        plt.close(fig)
        log.info("Saved energy_per_image.png")
    except Exception as e:
        log.warning(f"Plot 2 failed: {e}")

    # Plot 3: Throughput by method
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        sub = summary[summary["device"] == "cuda"] if "cuda" in summary["device"].values else summary
        labels = [f"{r.model_type}/{r.pruning_method}/{r.model_name[:15]}" for _, r in sub.iterrows()]
        ax.bar(range(len(sub)), sub["throughput_imgs_per_s"])
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("Throughput (img/s)")
        ax.set_title("Single-Image Throughput by Compression Method")
        plt.tight_layout()
        fig.savefig(plots_dir / "throughput_by_model.png", dpi=150)
        plt.close(fig)
        log.info("Saved throughput_by_model.png")
    except Exception as e:
        log.warning(f"Plot 3 failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Single-image inference benchmarking")
    parser.add_argument(
        "--model-types", nargs="+", default=["CNN", "ViT"],
        choices=["CNN", "ViT"], help="Model families to benchmark"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        help="MedMNIST datasets to benchmark"
    )
    parser.add_argument(
        "--devices", nargs="+", default=["cpu", "cuda"],
        help="Devices to benchmark on"
    )
    parser.add_argument("--reps", type=int, default=3, help="Number of independent repetitions")
    parser.add_argument("--n-images", type=int, default=10, help="Images per repetition")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup forward passes")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/arihangupta/Pruning/dinov2/Pruning/single_image_benchmarking"),
        help="Root output directory"
    )
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis/plot generation")
    return parser.parse_args()


def main():
    args = parse_args()
    log.info(f"Output root: {args.output_dir}")
    log.info(f"Model types: {args.model_types}")
    log.info(f"Datasets: {args.datasets}")
    log.info(f"Devices: {args.devices}")
    log.info(f"Reps: {args.reps} | Images/rep: {args.n_images} | Warmup: {args.warmup}")

    for model_type in args.model_types:
        benchmark_model_type(
            model_type=model_type,
            datasets=args.datasets,
            devices=args.devices,
            n_reps=args.reps,
            n_images=args.n_images,
            warmup=args.warmup,
            output_root=args.output_dir,
        )

    if not args.skip_analysis:
        log.info("Generating analysis...")
        generate_analysis(args.output_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()