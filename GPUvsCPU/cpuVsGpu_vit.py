"""
cpuVsGpu_vit.py

Single-image inference benchmarking for ViT (DeiT/ViT) compressed models
across MedMNIST datasets.

Design:
  - Load each model once, run 5 warmup passes, then time N individual images
    one-at-a-time across 3 independent repetitions.
  - Energy is tracked per image (CodeCarbon start/stop per image).
  - CPU and CUDA devices are both tested for every model.
  - Results are written to:
      single_image_benchmarking/
      └── ViT/
          ├── {dataset}_results.csv
          └── {dataset}_per_image_log.csv

Usage:
  python cpuVsGpu_vit.py [--datasets bloodmnist dermamnist pathmnist chestmnist]
                         [--devices cpu cuda]
                         [--reps 3]
                         [--n-images 10]
                         [--warmup 5]
                         [--output-dir /path/to/single_image_benchmarking]
"""

import argparse
import csv
import gc
import logging
import os
import random
import sys
import time
from datetime import datetime
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
VIT_BASELINE_ROOT = Path(
    "/home/arihangupta/Pruning/dinov2/Pruning/PruneAndTrain/Vision/new_baseline"
)
VIT_PRUNED_ROOT = Path(
    "/home/arihangupta/Pruning/dinov2/Pruning/PruneAndTrain/Vision/rerun/pruned_models"
)

_DATASETS_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
_NPY_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_npy"

DATASET_PATHS = {
    "bloodmnist": Path(f"{_DATASETS_DIR}/bloodmnist_224.npz"),
    "dermamnist":  Path(f"{_DATASETS_DIR}/dermamnist_224.npz"),
    "pathmnist":   Path(f"{_DATASETS_DIR}/pathmnist_224.npz"),
    "chestmnist":  Path(f"{_NPY_DIR}/chestmnist_224"),
}
MULTI_LABEL_DATASETS = {"chestmnist"}

DATASETS = ["bloodmnist", "dermamnist", "pathmnist", "chestmnist"]

NUM_CLASSES = {
    "bloodmnist": 8,
    "dermamnist": 7,
    "pathmnist": 9,
    "chestmnist": 14,
}

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_test_images(dataset: str, n: int, seed: int) -> list:
    """Load `n` random test images from the MedMNIST dataset."""
    data_path = DATASET_PATHS.get(dataset)
    if data_path is None:
        raise FileNotFoundError(f"No data path configured for dataset: {dataset}")

    if dataset in MULTI_LABEL_DATASETS:
        npy_file = data_path / "test_images.npy"
        if not npy_file.exists():
            raise FileNotFoundError(f"Cannot find chestmnist test images. Expected: {npy_file}")
        imgs = np.load(npy_file)
    else:
        if not data_path.exists():
            raise FileNotFoundError(f"Cannot find test data for {dataset}. Expected: {data_path}")
        data = np.load(data_path)
        if "test_images" in data:
            imgs = data["test_images"]
        elif "test_img" in data:
            imgs = data["test_img"]
        else:
            raise KeyError(
                f"NPZ file {data_path} has no 'test_images' key. "
                f"Available keys: {list(data.keys())}"
            )

    rng = random.Random(seed)
    indices = rng.sample(range(len(imgs)), min(n, len(imgs)))

    tensors = []
    for idx in indices:
        img = imgs[idx]
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            img = np.concatenate([img, img, img], axis=-1)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        t = torch.from_numpy(img).unsqueeze(0)
        if t.shape[-1] != 224 or t.shape[-2] != 224:
            t = torch.nn.functional.interpolate(
                t, size=(224, 224), mode="bilinear", align_corners=False
            )
        tensors.append((idx, t))

    return tensors


# ---------------------------------------------------------------------------
# ViT model definitions
# ---------------------------------------------------------------------------

class DeployMlp(nn.Module):
    """MLP without gates — physically smaller post-pruning."""
    def __init__(self, in_features, hidden_features, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DeployAttention(nn.Module):
    """
    Multi-head attention without gates — supports per-layer head/dim after pruning.

    Handles the edge case where all heads have been pruned (inner_dim == 0):
    in that case the attention is a no-op (returns zeros), matching the
    zero-sized weight tensors stored in the checkpoint.
    """
    def __init__(self, dim, num_heads, head_dim, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads  # may be 0 if all heads pruned
        self.scale = qk_scale or (head_dim ** -0.5 if head_dim > 0 else 1.0)

        # nn.Linear supports 0-dim in/out — creates weight tensor with that shape
        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # When all heads are pruned the projection from zero inner_dim produces zeros
        if self.inner_dim == 0:
            return torch.zeros_like(x)

        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.inner_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DeployBlock(nn.Module):
    """Transformer block without gates — per-layer dims inferred from checkpoint."""
    def __init__(self, dim, num_heads, head_dim, mlp_hidden_dim,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        try:
            from timm.models.layers import DropPath
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        except ImportError:
            self.drop_path = nn.Identity()

        self.norm1 = norm_layer(dim)
        self.attn = DeployAttention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = DeployMlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DeployVisionTransformer(nn.Module):
    """
    Physically smaller ViT for oneshot-pruned checkpoints.

    Per-layer num_heads and mlp_hidden_dim are inferred per-block from the
    checkpoint so the architecture exactly matches the pruned weight tensors.
    Blocks with all heads removed (inner_dim == 0) are handled correctly:
    the attention is treated as a no-op while the MLP and skip connection
    remain active.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=192, depth=12, head_dim=64,
                 per_layer_num_heads=None, per_layer_mlp_dim=None,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        from functools import partial as _partial
        try:
            from timm.models.layers import PatchEmbed, trunc_normal_
        except ImportError:
            raise ImportError("timm is required for DeployVisionTransformer: pip install timm")

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.head_dim = head_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if per_layer_num_heads is None:
            per_layer_num_heads = [max(1, embed_dim // head_dim)] * depth
        if per_layer_mlp_dim is None:
            per_layer_mlp_dim = [embed_dim * 4] * depth

        self.blocks = nn.ModuleList([
            DeployBlock(
                dim=embed_dim,
                num_heads=per_layer_num_heads[i],
                head_dim=head_dim,
                mlp_hidden_dim=per_layer_mlp_dim[i],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=_partial(nn.LayerNorm, eps=1e-6),
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        try:
            from timm.models.layers import trunc_normal_
        except ImportError:
            return
        if isinstance(m, nn.Linear):
            # Skip zero-element tensors (fully-pruned layers)
            if m.weight.numel() > 0:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None and m.bias.numel() > 0:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def _infer_vit_deploy_dims(sd: dict) -> Dict:
    """
    Infer all per-layer dims from a oneshot pruned checkpoint.

    Key fix: blocks with all attention heads pruned have qkv.weight of shape
    [0, embed_dim], giving inner_dim=0 and num_heads=0. We preserve this
    (no max(1, ...) clamping) so DeployAttention gets the correct zero size
    and can load the checkpoint without shape mismatches.
    """
    if "patch_embed.proj.weight" in sd:
        embed_dim = sd["patch_embed.proj.weight"].shape[0]
    elif "patch_embed.weight" in sd:
        embed_dim = sd["patch_embed.weight"].shape[0]
    else:
        raise KeyError(
            "Cannot find patch embedding weight in checkpoint. "
            f"Available keys (first 10): {list(sd.keys())[:10]}"
        )

    depth = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".norm1.weight"))
    head_dim = 64

    per_layer_num_heads = []
    per_layer_mlp_dim = []

    for i in range(depth):
        qkv_key = f"blocks.{i}.attn.qkv.weight"
        if qkv_key in sd:
            inner_dim = sd[qkv_key].shape[0] // 3
            # Do NOT clamp to max(1, ...) — preserve zero for fully-pruned blocks
            num_heads = inner_dim // head_dim if head_dim > 0 else 0
        else:
            num_heads = embed_dim // head_dim
        per_layer_num_heads.append(num_heads)

        mlp_key = f"blocks.{i}.mlp.fc1.weight"
        if mlp_key in sd:
            per_layer_mlp_dim.append(sd[mlp_key].shape[0])
        else:
            per_layer_mlp_dim.append(embed_dim * 4)

    return dict(
        embed_dim=embed_dim,
        depth=depth,
        head_dim=head_dim,
        per_layer_num_heads=per_layer_num_heads,
        per_layer_mlp_dim=per_layer_mlp_dim,
    )


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
    raw_ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    sd = raw_ckpt
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    def _arch_from_name(stem: str) -> str:
        for size in ["vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224"]:
            if size in stem:
                return size
        raise ValueError(f"Cannot infer ViT arch from: {stem}")

    if name.startswith("oneshot_"):
        # Prefer pruning_config stored in checkpoint (exact deploy dims)
        pruning_config = raw_ckpt.get("pruning_config", {}) if isinstance(raw_ckpt, dict) else {}
        if pruning_config:
            log.info(f"  [{name}] using pruning_config from checkpoint")
            model = DeployVisionTransformer(
                num_classes=num_classes,
                embed_dim=pruning_config["deploy_embed_dim"],
                depth=12,
                head_dim=pruning_config.get("head_dim", 64),
                per_layer_num_heads=pruning_config["deploy_per_layer_num_heads"],
                per_layer_mlp_dim=pruning_config["deploy_per_layer_mlp_dim"],
                drop_path_rate=0.1,
            )
        else:
            dims = _infer_vit_deploy_dims(sd)
            pruned_blocks = [i for i, h in enumerate(dims["per_layer_num_heads"]) if h == 0]
            if pruned_blocks:
                log.info(f"  [{name}] blocks with all heads pruned (no-op attention): {pruned_blocks}")
            model = DeployVisionTransformer(
                num_classes=num_classes,
                embed_dim=dims["embed_dim"],
                depth=dims["depth"],
                head_dim=dims["head_dim"],
                per_layer_num_heads=dims["per_layer_num_heads"],
                per_layer_mlp_dim=dims["per_layer_mlp_dim"],
            )
        model.load_state_dict(sd)

    elif name.startswith("quantization_"):
        arch = _arch_from_name(name)
        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        model.load_state_dict(sd, strict=False)
        model = model.half()

    elif name.startswith("kd_"):
        after_to = name.split("_to_", 1)[-1] if "_to_" in name else name
        arch = _arch_from_name(after_to)
        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        model.load_state_dict(sd, strict=False)

    else:
        # _pretrained.pth or any other standard checkpoint
        arch = _arch_from_name(name)
        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        model.load_state_dict(sd, strict=False)

    return model


# ---------------------------------------------------------------------------
# ViT model discovery
# ---------------------------------------------------------------------------

def discover_vit_models(dataset: str) -> List[Dict]:
    """Return list of dicts for all ViT models (baselines + compressed) for a dataset."""
    models = []

    # Baselines
    if VIT_BASELINE_ROOT.exists():
        for f in sorted(VIT_BASELINE_ROOT.iterdir()):
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

            if name.startswith("kd_") and "_to_" in name:
                after_to = name.split("_to_", 1)[-1]
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
    EmissionsTracker = _safe_codecarbon()
    use_cuda_sync = device.startswith("cuda") and torch.cuda.is_available()

    # CPU has no native fp16 compute — running fp16 on CPU is emulated in
    # software and ~100-1000x slower than fp32. Convert to fp32 for CPU runs.
    if device == "cpu" and next(model.parameters()).dtype == torch.float16:
        model = model.float()

    is_fp16 = next(model.parameters()).dtype == torch.float16

    model = model.to(device)
    model.eval()

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
                tracker.stop()
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

def benchmark_vit(
    datasets: List[str],
    devices: List[str],
    n_reps: int,
    n_images: int,
    warmup: int,
    output_root: Path,
):
    out_dir = output_root / "ViT"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        log.info(f"=== ViT / {dataset} ===")
        n_classes = NUM_CLASSES.get(dataset, 10)

        results_csv = out_dir / f"{dataset}_results.csv"
        per_img_csv = out_dir / f"{dataset}_per_image_log.csv"
        _ensure_csv(results_csv, RESULTS_COLS)
        _ensure_csv(per_img_csv, PER_IMAGE_COLS)
        existing_keys = _load_existing_keys(results_csv)

        model_infos = discover_vit_models(dataset)
        if not model_infos:
            log.warning(f"No ViT models found for {dataset}, skipping.")
            continue

        log.info(f"  Found {len(model_infos)} model checkpoints.")

        for info in model_infos:
            model_name = info["model_name"]
            model_path = info["path"]

            for device in devices:
                if device == "cuda" and not torch.cuda.is_available():
                    log.warning("CUDA requested but not available, skipping cuda device.")
                    continue

                for rep in range(n_reps):
                    key = (model_name, device, str(rep))
                    if key in existing_keys:
                        log.info(f"    SKIP (already done): {model_name} | {device} | rep={rep}")
                        continue

                    seed = rep * 1000 + hash(model_name) % 997
                    seed = abs(seed) % (2 ** 31)

                    log.info(f"    {model_name} | {device} | rep={rep} | seed={seed}")

                    try:
                        images = load_test_images(dataset, n_images, seed)
                    except FileNotFoundError as e:
                        log.error(f"      Dataset load failed: {e}")
                        continue

                    try:
                        model = load_vit_model(model_path, n_classes)
                    except Exception as e:
                        log.error(f"      Model load failed: {e}")
                        continue

                    num_params = sum(p.numel() for p in model.parameters())
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

                    ts = datetime.utcnow().isoformat()

                    for row in per_image_rows:
                        row.update({
                            "model_type": "ViT",
                            "dataset": dataset,
                            "model_name": model_name,
                            "device": device,
                            "rep": rep,
                        })
                        _append_row(per_img_csv, PER_IMAGE_COLS, row)

                    result_row = {
                        "model_type": "ViT",
                        "dataset": dataset,
                        "model_name": model_name,
                        "pruning_method": info["pruning_method"],
                        "stored_precision": info["stored_precision"],
                        "architecture": info.get("architecture"),
                        "device": device,
                        "rep": rep,
                        **agg,
                        "num_params": num_params,
                        "model_size_mb": round(info["model_size_mb"], 3),
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

                    del model
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()

    log.info("=== ViT benchmarking complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ViT single-image inference benchmarking")
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
    return parser.parse_args()


def main():
    args = parse_args()
    log.info(f"Output root: {args.output_dir}")
    log.info(f"Datasets: {args.datasets}")
    log.info(f"Devices: {args.devices}")
    log.info(f"Reps: {args.reps} | Images/rep: {args.n_images} | Warmup: {args.warmup}")

    benchmark_vit(
        datasets=args.datasets,
        devices=args.devices,
        n_reps=args.reps,
        n_images=args.n_images,
        warmup=args.warmup,
        output_root=args.output_dir,
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
