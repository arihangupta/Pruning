"""
cpuVsGpu_vit.py

Single-image inference benchmarking for ViT (DeiT/ViT) compressed models
across MedMNIST datasets.

Design:
  - Load each model once, run 5 warmup passes, then time N individual images
    one-at-a-time across 3 independent repetitions.
  - Energy is tracked per image (CodeCarbon start/stop per image).
  - Energy and power are tracked via CodeCarbon start_task/stop_task,
    with one tracker kept alive per rep so its monitoring thread is warm.
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
    def __init__(self, in_features, hidden_features, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class DeployAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads
        self.scale = qk_scale or (head_dim ** -0.5 if head_dim > 0 else 1.0)
        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        if self.inner_dim == 0:
            return torch.zeros_like(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1); attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.inner_dim)
        x = self.proj(x); x = self.proj_drop(x)
        return x


class DeployBlock(nn.Module):
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
        self.attn = DeployAttention(dim, num_heads=num_heads, head_dim=head_dim,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = DeployMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DeployVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=192, depth=12, head_dim=64,
                 per_layer_num_heads=None, per_layer_mlp_dim=None,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        from functools import partial as _partial
        try:
            from timm.models.layers import PatchEmbed, trunc_normal_
        except ImportError:
            raise ImportError("timm is required: pip install timm")

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.head_dim = head_dim

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
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
            DeployBlock(dim=embed_dim, num_heads=per_layer_num_heads[i],
                        head_dim=head_dim, mlp_hidden_dim=per_layer_mlp_dim[i],
                        qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[i], norm_layer=_partial(nn.LayerNorm, eps=1e-6))
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
    if "patch_embed.proj.weight" in sd:
        embed_dim = sd["patch_embed.proj.weight"].shape[0]
    elif "patch_embed.weight" in sd:
        embed_dim = sd["patch_embed.weight"].shape[0]
    else:
        raise KeyError(f"Cannot find patch embedding weight. Keys: {list(sd.keys())[:10]}")

    depth = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".norm1.weight"))
    head_dim = 64
    per_layer_num_heads, per_layer_mlp_dim = [], []

    for i in range(depth):
        qkv_key = f"blocks.{i}.attn.qkv.weight"
        if qkv_key in sd:
            inner_dim = sd[qkv_key].shape[0] // 3
            num_heads = inner_dim // head_dim if head_dim > 0 else 0
        else:
            num_heads = embed_dim // head_dim
        per_layer_num_heads.append(num_heads)

        mlp_key = f"blocks.{i}.mlp.fc1.weight"
        per_layer_mlp_dim.append(sd[mlp_key].shape[0] if mlp_key in sd else embed_dim * 4)

    return dict(embed_dim=embed_dim, depth=depth, head_dim=head_dim,
                per_layer_num_heads=per_layer_num_heads,
                per_layer_mlp_dim=per_layer_mlp_dim)


def load_vit_model(model_path: Path, num_classes: int) -> nn.Module:
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required: pip install timm")

    name = model_path.name
    raw_ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    sd = raw_ckpt
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    def _arch_from_name(stem):
        for size in ["vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224"]:
            if size in stem:
                return size
        raise ValueError(f"Cannot infer ViT arch from: {stem}")

    if name.startswith("oneshot_"):
        pruning_config = raw_ckpt.get("pruning_config", {}) if isinstance(raw_ckpt, dict) else {}
        if pruning_config:
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
                log.info(f"  [{name}] blocks with all heads pruned: {pruned_blocks}")
            model = DeployVisionTransformer(num_classes=num_classes, **dims)
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
        arch = _arch_from_name(name)
        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        model.load_state_dict(sd, strict=False)

    return model


# ---------------------------------------------------------------------------
# ViT model discovery
# ---------------------------------------------------------------------------

def discover_vit_models(dataset: str) -> List[Dict]:
    models = []

    if VIT_BASELINE_ROOT.exists():
        for f in sorted(VIT_BASELINE_ROOT.iterdir()):
            if dataset in f.name and f.name.endswith("_pretrained.pth"):
                arch = next((f"vit_{s}_patch16_224" for s in ["base", "small", "tiny"]
                             if f"vit_{s}" in f.name), None)
                models.append({
                    "path": f, "model_name": f.stem,
                    "pruning_method": "baseline", "stored_precision": "fp32",
                    "architecture": arch, "model_size_mb": f.stat().st_size / 1e6,
                })

    pruned_dir = VIT_PRUNED_ROOT / dataset
    if pruned_dir.exists():
        for f in sorted(pruned_dir.iterdir()):
            if not f.name.endswith("_final.pth"):
                continue
            name = f.stem
            arch = next((f"vit_{s}_patch16_224" for s in ["base", "small", "tiny"]
                         if f"vit_{s}" in name), None)
            if name.startswith("kd_") and "_to_" in name:
                after_to = name.split("_to_", 1)[-1]
                arch = next((f"vit_{s}_patch16_224" for s in ["base", "small", "tiny"]
                             if f"vit_{s}" in after_to), arch)
            if name.startswith("quantization_"):
                continue  # quantization excluded from ViT benchmarking
            method = name.split("_")[0] if "_" in name else name
            models.append({
                "path": f, "model_name": name,
                "pruning_method": method, "stored_precision": "fp32",
                "architecture": arch, "model_size_mb": f.stat().st_size / 1e6,
            })

    return models


# ---------------------------------------------------------------------------
# Power measurement — tiered fallback: RAPL → perf → CodeCarbon TDP
# ---------------------------------------------------------------------------
#
# Tier 1 (best): Intel RAPL via /sys/class/powercap — direct energy counter.
#   Available on bare-metal EC2 instances and local Linux.
#
# Tier 2: `perf stat -e power/energy-pkg/` — requires perf + CAP_PERFMON.
#
# Tier 3 (fallback): CodeCarbon EmissionsTracker — reads CPU TDP from
#   hardware tables, scales by utilization. Consistent across all models,
#   which is what matters for comparing compression methods.
#
# GPU: pynvml.nvmlDeviceGetPowerUsage() always attempted when device=cuda.
# ---------------------------------------------------------------------------

import subprocess as _subprocess
import threading as _threading

try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

_RAPL_PATH = Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")
_POWER_TIER: Optional[str] = None  # set on first call to _detect_power_tier()


def _detect_power_tier() -> str:
    """Detect which CPU power measurement method is available. Runs once."""
    global _POWER_TIER
    if _POWER_TIER is not None:
        return _POWER_TIER

    # Tier 1: RAPL sysfs
    if _RAPL_PATH.exists():
        try:
            float(_RAPL_PATH.read_text().strip())
            _POWER_TIER = "rapl"
            log.info("Power tier: RAPL (/sys/class/powercap)")
            return _POWER_TIER
        except Exception:
            pass

    # Tier 2: perf stat
    try:
        r = _subprocess.run(
            ["perf", "stat", "-e", "power/energy-pkg/", "sleep", "0.05"],
            capture_output=True, text=True, timeout=5,
        )
        if "power/energy-pkg/" in (r.stdout + r.stderr):
            _POWER_TIER = "perf"
            log.info("Power tier: perf stat (energy-pkg)")
            return _POWER_TIER
    except Exception:
        pass

    # Tier 3: CodeCarbon — probe with a short live measurement to confirm it
    # actually returns a non-NaN cpu_power (some environments report 0/NaN)
    try:
        from codecarbon import EmissionsTracker as _ET
        _t = _ET(project_name="_probe", log_level="error",
                 save_to_file=False, measure_power_secs=0.5)
        _t.start()
        import time as _time; _time.sleep(1.5)
        _t.stop()
        _d = _t.final_emissions_data
        _cpu_w = float(getattr(_d, "cpu_power", float("nan")))
        if not (_cpu_w != _cpu_w) and _cpu_w > 0:  # not NaN and positive
            _POWER_TIER = "codecarbon"
            log.info(f"Power tier: CodeCarbon TDP estimation (probe: {_cpu_w:.1f} W)")
            return _POWER_TIER
        else:
            log.warning(f"CodeCarbon probe returned cpu_power={_cpu_w} — treating as unavailable")
    except Exception as e:
        log.debug(f"CodeCarbon probe failed: {e}")

    _POWER_TIER = "none"
    log.warning(
        "No CPU power measurement available (no RAPL, perf, or working CodeCarbon). "
        "Energy metrics will be NaN. Install codecarbon (`pip install codecarbon`) "
        "or use a bare-metal EC2 instance for hardware counters."
    )
    return _POWER_TIER


def _read_rapl_uj() -> Optional[float]:
    try:
        return float(_RAPL_PATH.read_text().strip()) if _RAPL_PATH.exists() else None
    except Exception:
        return None


def _read_gpu_power_w(handle) -> float:
    try:
        return _pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except Exception:
        return float("nan")


def _get_gpu_handle(device: str):
    """Return a pynvml device handle for a CUDA device, or None for CPU."""
    if not device.startswith("cuda") or not _HAS_NVML:
        return None
    try:
        idx = int(device.split(":")[-1]) if ":" in device else 0
        return _pynvml.nvmlDeviceGetHandleByIndex(idx)
    except Exception:
        return None


def _cpu_power_rapl(duration_s: float) -> float:
    """Mean CPU package power via RAPL over duration_s seconds."""
    e0 = _read_rapl_uj()
    time.sleep(duration_s)
    e1 = _read_rapl_uj()
    if e0 is None or e1 is None:
        return float("nan")
    return ((e1 - e0) / 1e6) / duration_s


def _cpu_power_perf(duration_s: float) -> float:
    """Mean CPU package power via perf stat over duration_s seconds."""
    try:
        r = _subprocess.run(
            ["perf", "stat", "-e", "power/energy-pkg/", "sleep", f"{duration_s:.2f}"],
            capture_output=True, text=True, timeout=duration_s + 5,
        )
        for line in (r.stdout + r.stderr).splitlines():
            if "power/energy-pkg/" in line:
                joules = float(line.strip().split()[0].replace(",", ""))
                return joules / duration_s
    except Exception:
        pass
    return float("nan")


def _cpu_power_codecarbon(duration_s: float) -> float:
    """CPU power via CodeCarbon EmissionsTracker over duration_s seconds."""
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(
            project_name="_power_sample",
            log_level="error",
            save_to_file=False,
            measure_power_secs=max(0.5, duration_s / 4),
        )
        tracker.start()
        time.sleep(duration_s)
        tracker.stop()
        data = tracker.final_emissions_data
        return float(getattr(data, "cpu_power", float("nan")))
    except Exception:
        return float("nan")


def measure_power_w(duration_s: float, gpu_handle) -> Dict:
    """
    Measure mean CPU and GPU power over duration_s seconds using the best
    available tier (RAPL > perf > CodeCarbon TDP).
    Returns {"cpu_w": float, "gpu_w": float}.
    """
    tier = _detect_power_tier()

    # Run CPU measurement and GPU polling concurrently
    cpu_result: List[float] = [float("nan")]

    def _measure_cpu():
        if tier == "rapl":
            cpu_result[0] = _cpu_power_rapl(duration_s)
        elif tier == "perf":
            cpu_result[0] = _cpu_power_perf(duration_s)
        elif tier == "codecarbon":
            cpu_result[0] = _cpu_power_codecarbon(duration_s)

    cpu_thread = _threading.Thread(target=_measure_cpu)
    cpu_thread.start()

    gpu_samples: List[float] = []
    _stop_gpu = _threading.Event()

    def _poll_gpu():
        while not _stop_gpu.is_set():
            gpu_samples.append(_read_gpu_power_w(gpu_handle))
            time.sleep(0.05)

    if gpu_handle is not None:
        gpu_thread = _threading.Thread(target=_poll_gpu, daemon=True)
        gpu_thread.start()

    cpu_thread.join(timeout=duration_s + 15)

    if gpu_handle is not None:
        _stop_gpu.set()
        gpu_thread.join(timeout=2.0)

    cpu_w = cpu_result[0]
    gpu_w = float(np.nanmean(gpu_samples)) if gpu_samples else float("nan")
    return {"cpu_w": cpu_w, "gpu_w": gpu_w}


# ---------------------------------------------------------------------------
# Benchmarking core
# ---------------------------------------------------------------------------

def run_single_image_bench(
    model: nn.Module,
    images: List[Tuple[int, torch.Tensor]],
    device: str,
    warmup: int = 5,
    burst_rounds: int = 50,
) -> Tuple[List[Dict], Dict]:
    """
    Benchmark single-image inference using a rotating burst strategy.

    Clinical motivation: In a PACS-integrated AI tool, images arrive
    individually (one chest X-ray at a time), not in batches. Energy and
    latency must reflect the cost of processing a single image in isolation.

    Rotating burst: cycle all N images x burst_rounds, measuring each image
    one-at-a-time (no batching). Total energy / time / N*rounds gives the
    per-image cost. Different images rotate through so cache pressure is
    realistic — not inflated by running the same image repeatedly.

    Power measurement is tier-aware:

      RAPL / perf (bare-metal EC2 or local Linux):
        - Measure idle baseline for 2 s after warmup.
        - Bracket entire burst with RAPL counter (or perf stat).
        - Active power = burst_power - idle_power, floored at 0.

      CodeCarbon (virtualized EC2 — most common case):
        - CodeCarbon reports TDP x utilization; it has no meaningful "idle"
          reading (returns full TDP even at rest).
        - Instead: start one EmissionsTracker task across the burst window,
          read cpu_power directly — no idle subtraction needed or correct.

      none: energy metrics are NaN.
    """
    use_cuda_sync = device.startswith("cuda") and torch.cuda.is_available()

    if device == "cpu" and next(model.parameters()).dtype == torch.float16:
        model = model.float()
    is_fp16 = next(model.parameters()).dtype == torch.float16

    model = model.to(device)
    model.eval()

    # Pre-move all images to device once
    device_images = []
    for orig_idx, img_tensor in images:
        img = img_tensor.to(device)
        if is_fp16:
            img = img.half()
        device_images.append((orig_idx, img))

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

    tier = _detect_power_tier()
    gpu_handle = _get_gpu_handle(device)

    # --- Idle baseline (RAPL / perf only) ---
    idle_cpu_w = 0.0  # default: no subtraction
    idle_gpu_w = 0.0
    if tier in ("rapl", "perf"):
        log.info(f"      Measuring idle power baseline (2 s)...")
        idle = measure_power_w(duration_s=2.0, gpu_handle=gpu_handle)
        idle_cpu_w = idle["cpu_w"] if not np.isnan(idle["cpu_w"]) else 0.0
        idle_gpu_w = idle["gpu_w"] if not np.isnan(idle["gpu_w"]) else 0.0
        log.info(f"      Idle: cpu={idle_cpu_w:.2f} W  gpu={idle_gpu_w:.2f} W")
    elif tier == "codecarbon":
        log.info(f"      Power tier: CodeCarbon (no idle subtraction — TDP x utilization)")
    else:
        log.info(f"      Power tier: none — energy metrics will be NaN")

    import threading as _threading

    n_images_local = len(device_images)
    n_total = n_images_local * burst_rounds
    per_img_latencies: List[List[float]] = [[] for _ in range(n_images_local)]

    # --- GPU power poller (runs across entire burst) ---
    gpu_samples_burst: List[float] = []
    _stop_gpu = _threading.Event()

    def _poll_gpu(handle=gpu_handle, samples=gpu_samples_burst, stop=_stop_gpu):
        while not stop.is_set():
            samples.append(_read_gpu_power_w(handle))
            time.sleep(0.05)

    if gpu_handle is not None:
        gpu_poller = _threading.Thread(target=_poll_gpu, daemon=True)
        gpu_poller.start()

    # --- CPU power measurement setup ---
    # RAPL/perf: bracket burst with energy counter
    # CodeCarbon: wrap burst in a single EmissionsTracker task
    cc_tracker = None
    if tier == "codecarbon":
        try:
            from codecarbon import EmissionsTracker
            cc_tracker = EmissionsTracker(
                project_name="single_img_bench",
                log_level="error",
                save_to_file=False,
                measure_power_secs=max(1, n_total // 20),  # ~20 samples across burst
            )
            cc_tracker.start()
            cc_tracker.start_task("burst")
        except Exception as e:
            log.debug(f"CodeCarbon tracker init failed: {e}")
            cc_tracker = None

    e0 = _read_rapl_uj() if tier == "rapl" else None
    perf_proc = None
    if tier == "perf":
        try:
            perf_proc = _subprocess.Popen(
                ["perf", "stat", "-e", "power/energy-pkg/", "-p", str(os.getpid())],
                stdout=_subprocess.DEVNULL, stderr=_subprocess.PIPE,
            )
        except Exception:
            perf_proc = None

    burst_t0 = time.perf_counter()

    for _round in range(burst_rounds):
        for img_idx, (orig_idx, img) in enumerate(device_images):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(img)
            if use_cuda_sync:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_img_latencies[img_idx].append((t1 - t0) * 1000.0)

    burst_t1 = time.perf_counter()
    e1 = _read_rapl_uj() if tier == "rapl" else None
    total_burst_s = burst_t1 - burst_t0

    # Stop GPU poller
    if gpu_handle is not None:
        _stop_gpu.set()
        gpu_poller.join(timeout=2.0)

    # --- Read CPU power ---
    raw_cpu_w = float("nan")

    if tier == "rapl" and e0 is not None and e1 is not None and total_burst_s > 0:
        raw_cpu_w = ((e1 - e0) / 1e6) / total_burst_s

    elif tier == "perf" and perf_proc is not None:
        try:
            perf_proc.terminate()
            _, stderr = perf_proc.communicate(timeout=5)
            for line in stderr.decode().splitlines():
                if "power/energy-pkg/" in line:
                    joules = float(line.strip().split()[0].replace(",", ""))
                    raw_cpu_w = joules / total_burst_s
                    break
        except Exception:
            pass

    elif tier == "codecarbon" and cc_tracker is not None:
        try:
            task = cc_tracker.stop_task("burst")
            cc_tracker.stop()
            raw_cpu_w = float(getattr(task, "cpu_power", float("nan")))
        except Exception as e:
            log.debug(f"CodeCarbon burst read failed: {e}")

    # --- GPU power ---
    raw_gpu_w = float(np.nanmean(gpu_samples_burst)) if gpu_samples_burst else float("nan")

    # --- Active power (idle subtraction only for RAPL/perf) ---
    if tier in ("rapl", "perf"):
        cpu_power = max(0.0, raw_cpu_w - idle_cpu_w) if not np.isnan(raw_cpu_w) else float("nan")
        gpu_power = max(0.0, raw_gpu_w - idle_gpu_w) if not np.isnan(raw_gpu_w) else float("nan")
    else:
        # CodeCarbon already gives active power (TDP x util); no subtraction
        cpu_power = raw_cpu_w
        gpu_power = max(0.0, raw_gpu_w - idle_gpu_w) if (
            not np.isnan(raw_gpu_w) and not np.isnan(idle_gpu_w)
        ) else raw_gpu_w

    log.info(
        f"      Burst power: cpu={cpu_power:.2f} W  gpu={gpu_power:.2f} W  "
        f"({n_total} inferences in {total_burst_s:.1f} s)"
    )

    active_w = (cpu_power if not np.isnan(cpu_power) else 0.0) +                (gpu_power if not np.isnan(gpu_power) else 0.0)

    per_image_rows = []
    all_mean_latencies = []

    for img_idx, (orig_idx, _img) in enumerate(device_images):
        lats = np.array(per_img_latencies[img_idx])
        mean_lat_ms = float(np.mean(lats))
        all_mean_latencies.append(mean_lat_ms)
        energy_kwh = (mean_lat_ms / 1000.0 * active_w) / 3_600_000.0

        per_image_rows.append({
            "image_idx": img_idx,
            "image_sample_idx": orig_idx,
            "latency_ms": mean_lat_ms,
            "energy_kwh": energy_kwh,
            "cpu_power_w": cpu_power,
            "gpu_power_w": gpu_power,
            "ram_power_w": float("nan"),
        })

    arr = np.array(all_mean_latencies)
    mean_latency = float(np.mean(arr))
    throughput = 1000.0 / mean_latency if mean_latency > 0 else float("nan")
    all_energies = [r["energy_kwh"] for r in per_image_rows]

    aggregated = {
        "mean_latency_ms":          mean_latency,
        "std_latency_ms":           float(np.std(arr)),
        "median_latency_ms":        float(np.median(arr)),
        "p25_latency_ms":           float(np.percentile(arr, 25)),
        "p75_latency_ms":           float(np.percentile(arr, 75)),
        "p90_latency_ms":           float(np.percentile(arr, 90)),
        "min_latency_ms":           float(np.min(arr)),
        "max_latency_ms":           float(np.max(arr)),
        "throughput_imgs_per_s":    throughput,
        "mean_energy_kwh_per_image":float(np.nanmean(all_energies)),
        "mean_cpu_power_w":         cpu_power,
        "mean_gpu_power_w":         gpu_power,
        "mean_ram_power_w":         float("nan"),
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
        csv.DictWriter(f, fieldnames=cols, extrasaction="ignore").writerow(row)


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
    burst_rounds: int,
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
                    log.warning("CUDA requested but not available, skipping.")
                    continue

                for rep in range(n_reps):
                    key = (model_name, device, str(rep))
                    if key in existing_keys:
                        log.info(f"    SKIP: {model_name} | {device} | rep={rep}")
                        continue

                    seed = abs(rep * 1000 + hash(model_name) % 997) % (2 ** 31)
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
                    try:
                        per_image_rows, agg = run_single_image_bench(
                            model=model, images=images, device=device,
                            warmup=warmup, burst_rounds=burst_rounds,
                        )
                    except Exception as e:
                        log.error(f"      Benchmark failed: {e}", exc_info=True)
                        del model; gc.collect()
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        continue

                    ts = datetime.utcnow().isoformat()
                    for row in per_image_rows:
                        row.update({"model_type": "ViT", "dataset": dataset,
                                    "model_name": model_name, "device": device, "rep": rep})
                        _append_row(per_img_csv, PER_IMAGE_COLS, row)

                    result_row = {
                        "model_type": "ViT", "dataset": dataset,
                        "model_name": model_name,
                        "pruning_method": info["pruning_method"],
                        "stored_precision": info["stored_precision"],
                        "architecture": info.get("architecture"),
                        "device": device, "rep": rep,
                        **agg,
                        "num_params": num_params,
                        "model_size_mb": round(info["model_size_mb"], 3),
                        "num_images": len(per_image_rows),
                        "seed": seed, "timestamp": ts,
                    }
                    _append_row(results_csv, RESULTS_COLS, result_row)
                    existing_keys.add(key)

                    log.info(
                        f"      → latency: {agg['mean_latency_ms']:.2f} ms | "
                        f"cpu_power: {agg['mean_cpu_power_w']:.1f} W | "
                        f"gpu_power: {agg['mean_gpu_power_w']:.1f} W"
                    )

                    del model; gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()

    log.info("=== ViT benchmarking complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ViT single-image inference benchmarking")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--burst-rounds", type=int, default=50, help="Rotating burst rounds per image for energy measurement")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/home/arihangupta/Pruning/dinov2/Pruning/single_image_benchmarking"))
    return parser.parse_args()


def main():
    args = parse_args()
    log.info(f"Output root: {args.output_dir}")
    log.info(f"Datasets: {args.datasets} | Devices: {args.devices}")
    log.info(f"Reps: {args.reps} | Images/rep: {args.n_images} | Warmup: {args.warmup}")
    benchmark_vit(datasets=args.datasets, devices=args.devices, n_reps=args.reps,
                  n_images=args.n_images, warmup=args.warmup, burst_rounds=args.burst_rounds, output_root=args.output_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()