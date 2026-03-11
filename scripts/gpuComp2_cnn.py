"""
cpuVsGpu_cnn.py

Single-image inference benchmarking for CNN (ResNet-based) compressed models
across MedMNIST datasets.

Design:
  - Load each model once, run 5 warmup passes, then time N individual images
    one-at-a-time across 3 independent repetitions.
  - Energy is tracked per image (CodeCarbon start/stop per image).
  - CPU and CUDA devices are both tested for every model.
  - Results are written to:
      single_image_benchmarking/
      └── CNN/
          ├── {dataset}_results.csv
          └── {dataset}_per_image_log.csv

Usage:
  python cpuVsGpu_cnn.py [--datasets bloodmnist dermamnist pathmnist chestmnist]
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
from torchvision.models.resnet import Bottleneck

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
# CNN model definitions
# ---------------------------------------------------------------------------

class CustomResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3],
                 stage_planes=[64, 128, 256, 512], num_classes=1000, in_channels=3):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc(x)


def _infer_resnet_depth(sd: dict) -> List[int]:
    layers = []
    for layer_idx in range(1, 5):
        max_block = -1
        for k in sd:
            if k.startswith(f"layer{layer_idx}."):
                parts = k.split(".")
                if len(parts) >= 2:
                    try:
                        max_block = max(max_block, int(parts[1]))
                    except ValueError:
                        pass
        layers.append(max_block + 1 if max_block >= 0 else 3)
    return layers


def _infer_stage_planes(sd: dict) -> List[int]:
    planes = []
    for layer_idx in range(1, 5):
        key = f"layer{layer_idx}.0.conv1.weight"
        if key in sd:
            planes.append(sd[key].shape[0])
        else:
            raise ValueError(f"Cannot find '{key}' in checkpoint.")
    return planes


def load_cnn_model(model_path: Path, num_classes: int) -> nn.Module:
    sd = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    stage_planes = _infer_stage_planes(sd)
    layers = _infer_resnet_depth(sd)
    sample = next(iter(sd.values()))
    if sample.dtype == torch.float16:
        sd = {k: v.float() for k, v in sd.items()}
    model = CustomResNet(block=Bottleneck, layers=layers,
                         stage_planes=stage_planes, num_classes=num_classes)
    model.load_state_dict(sd)
    return model.float()


# ---------------------------------------------------------------------------
# CNN model discovery
# ---------------------------------------------------------------------------

_CNN_SKIP_PATTERNS = {"optimizer", "scheduler", "epoch_", "checkpoint_", "last_", "tmp_"}


def _cnn_model_filter(fname: str) -> bool:
    if not (fname.endswith(".pth") or fname.endswith(".pt")):
        return False
    fname_lower = fname.lower()
    for pat in _CNN_SKIP_PATTERNS:
        if pat in fname_lower:
            return False
    return True


def _infer_cnn_method_precision(name: str) -> Tuple[str, str]:
    name_lower = name.lower()
    if name in ("baseline", "model"):
        return "baseline", "fp32"
    if "fp16" in name_lower:
        method = name.split("_fp16")[0]
        return method, "fp16"
    if "quantization" in name_lower or "quant" in name_lower:
        return "quantization", "fp16"
    if "int8" in name_lower:
        return "quantization_int8", "int8"
    for suffix in ["_r50compressed_final", "_r18compressed_final", "_r34compressed_final",
                   "_compressed_final", "_final", "_best"]:
        if name.endswith(suffix):
            return name[: -len(suffix)], "fp32"
    return name, "fp32"


def discover_cnn_models(dataset: str) -> List[Dict]:
    base = CNN_MODEL_ROOT / dataset
    if not base.exists():
        log.warning(f"CNN dataset dir not found: {base}")
        return []
    models = []
    seen_paths = set()
    for f in sorted(base.rglob("*")):
        if not f.is_file() or not _cnn_model_filter(f.name) or f in seen_paths:
            continue
        seen_paths.add(f)
        name = f.stem
        method, precision = _infer_cnn_method_precision(name)
        models.append({
            "path": f, "model_name": name,
            "pruning_method": method, "stored_precision": precision,
            "architecture": "resnet", "model_size_mb": f.stat().st_size / 1e6,
        })
    return models



try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

_RAPL_PATH = Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")


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


def measure_power_w(duration_s: float, gpu_handle) -> Dict:
    """
    Measure mean CPU package power (RAPL) and GPU power (pynvml) over
    `duration_s` seconds.

    CPU: two RAPL energy counter readings bracketing the interval.
         delta_energy_J / delta_t = package power in Watts.
    GPU: pynvml.nvmlDeviceGetPowerUsage() polled every 50 ms, then averaged.

    Returns {"cpu_w": float, "gpu_w": float}.
    gpu_w is NaN when gpu_handle is None (CPU-only runs).
    """
    e0 = _read_rapl_uj()
    gpu_samples = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        if gpu_handle is not None:
            gpu_samples.append(_read_gpu_power_w(gpu_handle))
        time.sleep(0.05)
    e1 = _read_rapl_uj()

    cpu_w = ((e1 - e0) / 1e6) / duration_s if (e0 is not None and e1 is not None) else float("nan")
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
) -> Tuple[List[Dict], Dict]:
    """
    Benchmark single-image inference.

    Power methodology:
      - After warmup, measure idle system power for 2 s (model loaded, no
        inference). This captures baseline OS + memory + GPU idle draw.
      - Per image: RAPL counter bracketing + concurrent pynvml GPU polling
        measure total system power during the inference window.
      - Active power = total - idle, floored at 0 W.
      - Energy (kWh) = active_power_W * latency_s / 3_600_000.

    CPU-only runs: gpu_handle=None → gpu_power logged as NaN, not included
    in energy calculation.
    CUDA runs: both CPU and GPU active power measured and summed for energy.
    """
    use_cuda_sync = device.startswith("cuda") and torch.cuda.is_available()

    # fp16 on CPU is software-emulated (~100-1000x slower) — convert to fp32
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

    # Measure idle baseline (model loaded, GPU warm if CUDA, no inference)
    gpu_handle = _get_gpu_handle(device)
    log.info(f"      Measuring idle power baseline (2 s)...")
    idle = measure_power_w(duration_s=2.0, gpu_handle=gpu_handle)
    idle_cpu_w = idle["cpu_w"]
    idle_gpu_w = idle["gpu_w"]
    log.info(f"      Idle: cpu={idle_cpu_w:.2f} W  gpu={idle_gpu_w:.2f} W")

    per_image_rows = []
    latencies_ms = []
    cpu_powers, gpu_powers = [], []

    import threading as _threading

    for img_idx, (orig_idx, img_tensor) in enumerate(images):
        img = img_tensor.to(device)
        if is_fp16:
            img = img.half()

        # --- Start concurrent GPU power polling thread ---
        gpu_samples_img: List[float] = []
        _stop_poll = _threading.Event()

        def _gpu_poll(handle=gpu_handle, samples=gpu_samples_img, stop=_stop_poll):
            while not stop.is_set():
                samples.append(_read_gpu_power_w(handle))
                time.sleep(0.05)

        poller = _threading.Thread(target=_gpu_poll, daemon=True)
        poller.start()

        # --- Read CPU energy counter, run inference, read counter again ---
        e0 = _read_rapl_uj()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(img)
        if use_cuda_sync:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        e1 = _read_rapl_uj()

        _stop_poll.set()
        poller.join(timeout=1.0)

        latency_s = t1 - t0
        latency_ms = latency_s * 1000.0

        # CPU power from RAPL delta
        if e0 is not None and e1 is not None and latency_s > 0:
            raw_cpu_w = ((e1 - e0) / 1e6) / latency_s
        else:
            raw_cpu_w = float("nan")

        # GPU power from pynvml samples (NaN for CPU-only runs)
        raw_gpu_w = float(np.nanmean(gpu_samples_img)) if gpu_samples_img else float("nan")

        # Subtract idle baseline, floor at 0
        cpu_power = (
            max(0.0, raw_cpu_w - idle_cpu_w)
            if not (np.isnan(raw_cpu_w) or np.isnan(idle_cpu_w))
            else raw_cpu_w
        )
        gpu_power = (
            max(0.0, raw_gpu_w - idle_gpu_w)
            if not (np.isnan(raw_gpu_w) or np.isnan(idle_gpu_w))
            else raw_gpu_w
        )

        # Energy = active power × time
        # GPU term is 0 for CPU-only runs (NaN treated as 0 for energy sum only)
        active_w = cpu_power + (gpu_power if not np.isnan(gpu_power) else 0.0)
        energy_kwh = (latency_s * active_w) / 3_600_000.0

        latencies_ms.append(latency_ms)
        cpu_powers.append(cpu_power)
        gpu_powers.append(gpu_power)

        per_image_rows.append({
            "image_idx": img_idx,
            "image_sample_idx": orig_idx,
            "latency_ms": latency_ms,
            "energy_kwh": energy_kwh,
            "cpu_power_w": cpu_power,
            "gpu_power_w": gpu_power,
            "ram_power_w": float("nan"),  # not measured directly
        })

    arr = np.array(latencies_ms)
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
        "mean_cpu_power_w":         float(np.nanmean(cpu_powers)),
        "mean_gpu_power_w":         float(np.nanmean(gpu_powers)),
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

def benchmark_cnn(
    datasets: List[str],
    devices: List[str],
    n_reps: int,
    n_images: int,
    warmup: int,
    output_root: Path,
):
    out_dir = output_root / "CNN"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        log.info(f"=== CNN / {dataset} ===")
        n_classes = NUM_CLASSES.get(dataset, 10)

        results_csv = out_dir / f"{dataset}_results.csv"
        per_img_csv = out_dir / f"{dataset}_per_image_log.csv"
        _ensure_csv(results_csv, RESULTS_COLS)
        _ensure_csv(per_img_csv, PER_IMAGE_COLS)
        existing_keys = _load_existing_keys(results_csv)

        model_infos = discover_cnn_models(dataset)
        if not model_infos:
            log.warning(f"No CNN models found for {dataset}, skipping.")
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
                        model = load_cnn_model(model_path, n_classes)
                    except Exception as e:
                        log.error(f"      Model load failed: {e}")
                        continue

                    num_params = sum(p.numel() for p in model.parameters())

                    try:
                        per_image_rows, agg = run_single_image_bench(
                            model=model,
                            images=images,
                            device=device,
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
                            "model_type": "CNN",
                            "dataset": dataset,
                            "model_name": model_name,
                            "device": device,
                            "rep": rep,
                        })
                        _append_row(per_img_csv, PER_IMAGE_COLS, row)

                    result_row = {
                        "model_type": "CNN",
                        "dataset": dataset,
                        "model_name": model_name,
                        "pruning_method": info["pruning_method"],
                        "stored_precision": info["stored_precision"],
                        "architecture": info.get("architecture", "resnet"),
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
                        f"cpu_power: {agg['mean_cpu_power_w']:.1f} W | "
                        f"gpu_power: {agg['mean_gpu_power_w']:.1f} W"
                    )

                    del model
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()

    log.info("=== CNN benchmarking complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CNN single-image inference benchmarking")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/home/arihangupta/Pruning/dinov2/Pruning/single_image_benchmarking"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log.info(f"Output root: {args.output_dir}")
    log.info(f"Datasets: {args.datasets} | Devices: {args.devices}")
    log.info(f"Reps: {args.reps} | Images/rep: {args.n_images} | Warmup: {args.warmup}")
    benchmark_cnn(datasets=args.datasets, devices=args.devices, n_reps=args.reps,
                  n_images=args.n_images, warmup=args.warmup, output_root=args.output_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()