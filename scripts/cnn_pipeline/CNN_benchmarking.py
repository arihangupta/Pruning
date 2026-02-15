#!/usr/bin/env python3
"""
Complete script to run the full factorial benchmarking experiment on available MedMNIST datasets.
Dynamically discovers models ending in '_final.pth' for each dataset.
MODIFIED to properly handle regional_pruning and hybrid_pruning outputs from the training script.
Uses CustomResNet with channel counts inferred from checkpoints to avoid size mismatches.
Handles both RGB and grayscale images by detecting dataset channel count.
Includes CodeCarbon for power utilization (silently in background) and detailed analysis with pruning method and sparsity.
Supports quantized model loading and AMP precision.
Calculates AUC for each run and processes the test dataset three times.
Calculates energy metrics per batch, per image, and for the full test dataset.
Tracks stored precision (fp32/fp16/int8) separately from runtime precision.
"""

import os
import time
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from typing import Dict, Any
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from pathlib import Path
from itertools import product
import glob
import re
import sys
import subprocess as sp
import logging
from sklearn.metrics import roc_auc_score

# For quantization and AMP
import torch.ao.quantization as ao_quant
import torch.amp

# Configure debug logging to file only
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/arihangupta/Pruning/dinov2/Pruning/time_experiment_debug.log'),
    ]
)
logger = logging.getLogger(__name__)

# CodeCarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except Exception:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False

# Suppress CodeCarbon logs
if CODECARBON_AVAILABLE:
    logging.getLogger("codecarbon").setLevel(logging.ERROR)
    for handler in logging.getLogger("codecarbon").handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            logging.getLogger("codecarbon").removeHandler(handler)

# Import the proven Bottleneck class from torchvision
from torchvision.models.resnet import Bottleneck

# Constants
IMG_SIZE = 224
SEED = 42
DATASETS = {
    "bloodmnist": "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced/bloodmnist_224.npz",
    "dermamnist": "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced/dermamnist_224.npz",
    "octmnist": "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced/octmnist_224.npz",
    "pathmnist": "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced/pathmnist_224.npz",
    "tissuemnist": "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced/tissuemnist_224.npz",
}
SAVE_DIR_BASE = "/home/arihangupta/Pruning/dinov2/Pruning/CNN/CNN_pruned_models"
ORIGINAL_PLANES = [64, 128, 256, 512]
DATASET_NUM_CLASSES = {
    "bloodmnist": 8,
    "dermamnist": 7,
    "octmnist": 4,
    "pathmnist": 9,
    "tissuemnist": 8
}
MATRIX_CONFIG = {
    "datasets": ["bloodmnist", "dermamnist", "pathmnist"],
    "log_dir": "/home/arihangupta/Pruning/dinov2/Pruning/CNN/timing_exps",
    "num_passes": 3,  # Number of full passes through the test dataset
    "warmup_batches": 50,
    "seed": 42,
    "num_workers": 4,
    "pin_memory": True,
    "batch_sizes": [8, 32],
    "precisions": ["fp32", "amp"],
    "repeats": 3
}

# CustomResNet class
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

def build_model(num_classes: int, stage_planes=[64, 128, 256, 512]):
    model = CustomResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        stage_planes=stage_planes,
        num_classes=num_classes,
        in_channels=3
    )
    return model

def get_dataset_channels(npz_path):
    try:
        data = np.load(npz_path, mmap_mode="r")
        test_images = data["test_images"]
        sample_img = test_images[0]
        if sample_img.ndim == 3:
            return sample_img.shape[-1]
        elif sample_img.ndim == 2:
            return 1
        else:
            raise ValueError(f"Unexpected image dimensions in {npz_path}: {sample_img.shape}")
    except Exception as e:
        print(f"Error detecting channels for {npz_path}: {e}")
        return 3

def parse_model_name(filename, dataset):
    """
    Enhanced parser to handle all model naming conventions from the training script.
    Supports: baseline, quantization, hybrid_pruning, hybrid_pruning_fp16, 
    slim_kd, slim_kd_fp16, regional_pruning, regional_pruning_fp16
    """
    basename = os.path.basename(filename)
    logger.debug(f"Parsing model file: {basename} for dataset: {dataset}")
    
    # Handle baseline
    if basename == "baseline.pth":
        return {
            "model_name": "baseline",
            "pruning_method": "baseline",
            "sparsity": "0%",
            "pruning_ratio": None,
            "stored_precision": "fp32"
        }
    
    # Only process files ending in _final.pth
    if not basename.endswith("_final.pth"):
        logger.debug(f"Skipping {basename}: does not end with _final.pth")
        return None
    
    # Detect stored precision
    stored_precision = "fp32"  # default
    if "_fp16_" in basename or basename.endswith("_fp16_r50compressed_final.pth"):
        stored_precision = "amp"
    elif "quantization" in basename:
        stored_precision = "amp"  # quantization uses FP16
    
    logger.debug(f"Detected stored_precision={stored_precision} for {basename}")
    
    # Extract pruning method - UPDATED to handle all methods from training script
    method = None
    if "quantization" in basename:
        method = "quantization"
    elif "hybrid_pruning_fp16" in basename:
        method = "hybrid_pruning_fp16"
    elif "hybrid_pruning" in basename:
        method = "hybrid_pruning"
    elif "slim_kd_fp16" in basename:
        method = "slim_kd_fp16"
    elif "slim_kd" in basename:
        method = "slim_kd"
    elif "regional_pruning_fp16" in basename:
        method = "regional_pruning_fp16"
    elif "regional_pruning" in basename:
        method = "regional_pruning"
    
    if method is None:
        logger.debug(f"Skipping {basename}: no recognized pruning method")
        return None

    # Extract sparsity from filename
    # The training script uses format: {method}_r{compress_ratio}compressed_final.pth
    # where compress_ratio is 50 for 50% compression
    sparsity = None
    pruning_ratio = None
    
    # Match r##compressed pattern (e.g., r50compressed)
    compress_match = re.search(r'_r(\d+)compressed', basename)
    if compress_match:
        compress_ratio = int(compress_match.group(1))
        sparsity = f"{compress_ratio}%"
        pruning_ratio = compress_ratio / 100.0
        logger.debug(f"Sparsity extracted from filename: {sparsity}, pruning_ratio={pruning_ratio}")
    else:
        # Default to 50% if pattern not found but method is recognized
        sparsity = "50%"
        pruning_ratio = 0.5
        logger.debug(f"No sparsity pattern found in {basename}, defaulting to sparsity=50%")

    # Try CSV for additional confirmation if available
    metrics_csv = os.path.join(SAVE_DIR_BASE, dataset, f"{dataset}_combined_pruning_kd_metrics_with_energy.csv")
    if os.path.exists(metrics_csv):
        try:
            df = pd.read_csv(metrics_csv)
            logger.debug(f"Loaded metrics CSV: {metrics_csv}, {len(df)} rows")
            model_basename = basename.replace(".pth", "")
            
            # Look for matching rows
            for idx, row in df.iterrows():
                variant = str(row.get("Variant", "")).lower()
                if method in variant and "final" in str(row.get("Stage", "")).lower():
                    csv_ratio = row.get("Ratio", None)
                    logger.debug(f"Match found at row {idx}: Variant={variant}, Ratio={csv_ratio}")
                    if csv_ratio is not None:
                        # Ratio in CSV is keep_ratio (1 - compress_ratio)
                        compress_ratio = int((1 - float(csv_ratio)) * 100)
                        sparsity = f"{compress_ratio}%"
                        pruning_ratio = compress_ratio / 100.0
                        logger.debug(f"Updated from CSV: sparsity={sparsity}, pruning_ratio={pruning_ratio}")
                        break
        except Exception as e:
            logger.debug(f"CSV error: {str(e)}")

    # Create unique model name
    if stored_precision == "amp":
        model_name = f"{method}_{sparsity}_stored_amp"
    else:
        model_name = f"{method}_{sparsity}"
    
    return {
        "model_name": model_name,
        "pruning_method": method,
        "sparsity": sparsity,
        "pruning_ratio": pruning_ratio,
        "stored_precision": stored_precision
    }

def discover_models():
    """
    Discovers all _final.pth models in the dataset directories.
    """
    models = []
    for dataset in MATRIX_CONFIG["datasets"]:
        model_dir = os.path.join(SAVE_DIR_BASE, dataset)
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory for {dataset} does not exist: {model_dir}")
            continue
        
        # Find all _final.pth files and baseline.pth
        model_files = glob.glob(os.path.join(model_dir, "*_final.pth"))
        baseline_path = os.path.join(model_dir, "baseline.pth")
        if os.path.exists(baseline_path):
            model_files.append(baseline_path)
        
        dataset_models = []
        for model_path in model_files:
            parsed = parse_model_name(model_path, dataset)
            if parsed:
                parsed["model_path"] = model_path
                dataset_models.append(parsed)
                print(f"Discovered: {os.path.basename(model_path)} -> {parsed['model_name']}")
            else:
                print(f"Skipping invalid model file: {model_path}")
        
        if dataset_models:
            models.append({"dataset": dataset, "models": dataset_models})
        else:
            print(f"Warning: No valid models found for {dataset}")
    
    return models

def build_model_for_load(model_name, num_classes, model_path, pruning_ratio=None, in_channels=3, precision="fp32"):
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    
    # Infer stage_planes from state_dict
    stage_planes = []
    for i in range(4):
        key = f'layer{i+1}.0.conv1.weight'
        if key in state_dict:
            planes = state_dict[key].shape[0]
            stage_planes.append(planes)
        else:
            raise ValueError(f"Cannot find {key} to infer stage_planes")
    
    model = build_model(num_classes, stage_planes=stage_planes)
    
    # Check dtype
    sample_param = next(iter(state_dict.values()))
    is_fp16 = sample_param.dtype == torch.float16
    is_int8 = hasattr(sample_param, 'dtype') and sample_param.dtype == torch.qint8
    
    logger.debug(f"Model {model_name} at {model_path}: is_fp16={is_fp16}, is_int8={is_int8}, requested_precision={precision}")
    
    # Handle dtype based on requested precision
    if is_fp16 and precision == "fp32":
        logger.debug(f"Converting float16 model to float32 for {model_name}")
        model = model.float()  # Ensure model is in float32
        # Convert state_dict to float32
        state_dict = {k: v.float() if v.dtype == torch.float16 else v for k, v in state_dict.items()}
    elif is_fp16 and precision == "amp":
        model = model.half()
        logger.debug(f"Keeping float16 model for AMP: {model_name}")
    elif is_int8:
        logger.debug(f"INT8 model detected for {model_name}, using CPU")
        return model, 'cpu'
    
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Error loading state_dict: {e}")
    
    return model, None

MATRIX_CONFIG["models"] = discover_models()

@dataclass
class Config:
    experiment: Dict[str, Any]
    log_dir: str

def set_seed(s=SEED, deterministic=True):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

def set_env_threads(omp_threads=4, mkl_threads=4):
    os.environ['OMP_NUM_THREADS'] = str(omp_threads)
    os.environ['MKL_NUM_THREADS'] = str(mkl_threads)
    os.environ['PYTHONHASHSEED'] = '0'

class NumpyMemmapDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_np, labels_np, img_size=224, in_channels=3):
        self.imgs = imgs_np
        self.labels = labels_np
        self.img_size = img_size
        self.in_channels = in_channels
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

def make_test_loader(npz_path, batch_size):
    data = np.load(npz_path, mmap_mode="r")
    X_test, y_test = data["test_images"], data["test_labels"].flatten()
    in_channels = get_dataset_channels(npz_path)
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, in_channels=in_channels)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True), in_channels, len(test_ds)

def load_model(config, num_classes, dataset_name):
    dataset_path = DATASETS.get(dataset_name)
    if not dataset_path:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    in_channels = get_dataset_channels(dataset_path)
    model_path = config.experiment['model_path']
    if not os.path.exists(model_path):
        raise ValueError(f"Model not found: {model_path}")
    
    result = build_model_for_load(
        config.experiment['model_name'],
        num_classes,
        model_path,
        config.experiment['pruning_ratio'],
        in_channels=in_channels,
        precision=config.experiment['precision']
    )
    
    # Handle both tuple and single return value for backward compatibility
    if isinstance(result, tuple):
        model, override_device = result
    else:
        model = result
        override_device = None
    
    device = override_device if override_device else config.experiment['device']
    model = model.to(device).eval()
    
    return model

def get_num_classes(dataset_name, dataset_path):
    info_path = dataset_path.replace('.npz', '_info.csv')
    if os.path.exists(info_path):
        try:
            df = pd.read_csv(info_path)
            return len(df['label'].unique())
        except Exception:
            pass
    return DATASET_NUM_CLASSES.get(dataset_name, 2)

def start_tracker(save_dir: str, project_name: str, output_file: str="emissions.csv", measure_power_secs: int=10):
    if not CODECARBON_AVAILABLE:
        return None
    os.makedirs(save_dir, exist_ok=True)
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=save_dir,
        output_file=output_file,
        measure_power_secs=measure_power_secs,
        save_to_file=True,
        log_level="ERROR"
    )
    tracker.start()
    return tracker

def _read_latest_tracker_row(save_dir: str, project_name: str):
    csv_path = os.path.join(save_dir, "emissions.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    try:
        df_match = df[df["project_name"] == project_name]
        if df_match.shape[0] == 0:
            return None
        last = df_match.iloc[-1].to_dict()
        return last
    except Exception:
        try:
            return df.iloc[-1].to_dict()
        except Exception:
            return None

def stop_tracker_and_get_metrics(tracker, save_dir: str, project_name: str, images_processed: int, num_batches: int):
    if tracker is None:
        return {
            "emissions_kg": float("nan"),
            "energy_kwh_total": float("nan"),
            "energy_kwh_per_batch": float("nan"),
            "energy_kwh_per_image": float("nan"),
            "cpu_power_w": float("nan"),
            "gpu_power_w": float("nan"),
            "ram_power_w": float("nan"),
            "raw_row": None
        }
    try:
        emissions_val = tracker.stop()
    except Exception as e:
        print(f"Error stopping CodeCarbon tracker: {e}")
        emissions_val = None
    raw = _read_latest_tracker_row(save_dir, project_name)
    energy_kwh_total = float(raw.get("energy_consumed", float("nan"))) if raw is not None and "energy_consumed" in raw else float("nan")
    cpu_power = float(raw.get("cpu_power", float("nan"))) if raw is not None and "cpu_power" in raw else float("nan")
    gpu_power = float(raw.get("gpu_power", float("nan"))) if raw is not None and "gpu_power" in raw else float("nan")
    ram_power = float(raw.get("ram_power", float("nan"))) if raw is not None and "cpu_power" in raw else float("nan")
    emissions_kg = float(raw.get("emissions", float("nan"))) if raw is not None and "emissions" in raw else (float(emissions_val) if emissions_val is not None else float("nan"))
    energy_kwh_per_batch = energy_kwh_total / num_batches if num_batches > 0 and not math.isnan(energy_kwh_total) else float("nan")
    energy_kwh_per_image = energy_kwh_total / images_processed if images_processed > 0 and not math.isnan(energy_kwh_total) else float("nan")
    return {
        "emissions_kg": emissions_kg,
        "energy_kwh_total": energy_kwh_total,
        "energy_kwh_per_batch": energy_kwh_per_batch,
        "energy_kwh_per_image": energy_kwh_per_image,
        "cpu_power_w": cpu_power,
        "gpu_power_w": gpu_power,
        "ram_power_w": ram_power,
        "raw_row": raw
    }

def bench_fixed_passes(config):
    set_seed(config.experiment['seed'])
    set_env_threads(omp_threads=4, mkl_threads=4)
    device = torch.device(config.experiment['device'])
    dataset_name = config.experiment['dataset']
    batch_size = config.experiment['batch_size']
    num_passes = config.experiment['num_passes']
    dataset_path = DATASETS.get(dataset_name, None)
    if not dataset_path:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Validate batch size to prevent OOM
    if device.type == 'cuda':
        try:
            available_mem = torch.cuda.get_device_properties(device).total_memory / (1024**2)
            approx_mem_per_image = (224 * 224 * 3 * 4) / (1024**2) * 2
            estimated_mem = batch_size * approx_mem_per_image
            if estimated_mem > available_mem * 0.8:
                print(f"Warning: Batch size {batch_size} may exceed GPU memory. Skipping.")
                return None
        except Exception as e:
            print(f"Error checking GPU memory: {e}")

    test_loader, in_channels, dataset_size = make_test_loader(dataset_path, batch_size)

    num_classes = get_num_classes(dataset_name, dataset_path)
    try:
        model = load_model(config, num_classes, dataset_name)
    except Exception as e:
        print(f"Failed to load model {config.experiment['model_name']}: {e}")
        raise

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{config.experiment['model_name']}_b{batch_size}_{config.experiment['precision']}_r{config.experiment.get('rep', 0)}"
    project_name = f"{run_id}_inference"

    batch_logs = []
    probs_list = []
    labels_list = []
    images_processed = 0
    total_batches = 0

    tracker = start_tracker(str(log_dir), project_name, measure_power_secs=10) if CODECARBON_AVAILABLE else None

    model_dtype = next(model.parameters()).dtype

    model.eval()
    with torch.no_grad():
        for _ in range(config.experiment['warmup_batches']):
            batch = next(iter(test_loader))
            inputs = batch[0].to(device)
            if inputs.dtype != model_dtype:
                inputs = inputs.to(dtype=model_dtype)
            if config.experiment['precision'] == 'amp':
                with torch.amp.autocast('cuda'):
                    _ = model(inputs)
            else:
                _ = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for pass_idx in range(num_passes):
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            batch_start = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.dtype != model_dtype:
                inputs = inputs.to(dtype=model_dtype)
            with torch.no_grad():
                if config.experiment['precision'] == 'amp':
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                # Calculate AUC components
                if outputs.dtype == torch.half:
                    outputs = outputs.float()
                outputs = torch.clamp(outputs, min=-100, max=100)
                probs = torch.softmax(outputs, dim=1)
                probs_np = probs.cpu().numpy()
                prob_sums = np.sum(probs_np, axis=1, keepdims=True)
                probs_np = probs_np / prob_sums
                if np.any(np.isnan(probs_np)) or np.any(np.isinf(probs_np)):
                    continue
                probs_list.append(probs_np)
                labels_list.append(labels.cpu().numpy())
            batch_end = time.time()
            batch_elapsed = (batch_end - batch_start) * 1000
            batch_logs.append({
                'batch_idx': total_batches,
                'batch_size': len(inputs),
                'batch_start_ts': batch_start,
                'batch_end_ts': batch_end,
                'batch_elapsed_ms': batch_elapsed,
                'pass_idx': pass_idx
            })
            images_processed += len(inputs)
            total_batches += 1

    elapsed_s = time.time() - t0
    throughput = images_processed / elapsed_s if elapsed_s > 0 else 0

    auc = float('nan')
    if probs_list:
        all_probs = np.concatenate(probs_list)
        all_labels = np.concatenate(labels_list)
        try:
            auc = roc_auc_score(
                all_labels,
                all_probs,
                multi_class="ovr",
                labels=list(range(num_classes))
            )
        except Exception as e:
            print(f"AUC calculation failed: {e}")

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else 0

    metrics = stop_tracker_and_get_metrics(tracker, str(log_dir), project_name, images_processed, total_batches)
    energy_kwh_total = metrics["energy_kwh_total"]
    energy_kwh_per_batch = metrics["energy_kwh_per_batch"]
    energy_kwh_per_image = metrics["energy_kwh_per_image"]
    emissions_kg = metrics["emissions_kg"]
    cpu_power_w = metrics["cpu_power_w"]
    gpu_power_w = metrics["gpu_power_w"]
    ram_power_w = metrics["ram_power_w"]
    avg_power_w = gpu_power_w if not math.isnan(gpu_power_w) else float('nan')

    batch_df = pd.DataFrame(batch_logs)
    batch_log_file = log_dir / f"per_batch_log_{run_id}.csv"
    batch_df.to_csv(batch_log_file, index=False)

    metadata = {
        'run_id': run_id,
        'model_name': config.experiment['model_name'],
        'pruning_method': config.experiment['pruning_method'],
        'sparsity': config.experiment['sparsity'],
        'stored_precision': config.experiment.get('stored_precision', 'unknown'),
        'batch_size': batch_size,
        'runtime_precision': config.experiment['precision'],
        'precision': config.experiment['precision'],  # Add this line
        'rep': config.experiment.get('rep', 0),
        'num_passes': num_passes,
        'images_processed': images_processed,
        'elapsed_s': elapsed_s,
        'throughput_imgs_per_s': throughput,
        'auc': auc,
        'median_batch_ms': batch_df['batch_elapsed_ms'].median() if not batch_df.empty else float('nan'),
        'p50_ms': batch_df['batch_elapsed_ms'].quantile(0.5) if not batch_df.empty else float('nan'),
        'p90_ms': batch_df['batch_elapsed_ms'].quantile(0.9) if not batch_df.empty else float('nan'),
        'peak_gpu_mem_MB': peak_mem,
        'avg_power_W': avg_power_w,
        'energy_kWh_total': energy_kwh_total,
        'energy_kWh_per_batch': energy_kwh_per_batch,
        'energy_kWh_per_image': energy_kwh_per_image,
        'emissions_kg_total': emissions_kg,
        'cpu_power_w': cpu_power_w,
        'gpu_power_w': gpu_power_w,
        'ram_power_w': ram_power_w,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seed': config.experiment['seed'],
        'git_commit': get_git_commit(),
        'py_version': f"{sys.version_info.major}.{sys.version_info.minor}",
        'torch_version': torch.__version__,
        'dataset': config.experiment['dataset']
    }

    results_file = log_dir / f"results_{run_id}.json"
    with open(results_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    results_csv = log_dir.parent / f"{dataset_name}_results.csv"
    pd.DataFrame([metadata]).to_csv(results_csv, mode='a', header=not os.path.exists(results_csv), index=False)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return metadata

def get_git_commit():
    try:
        return sp.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except:
        return 'unknown'

def run_matrix(matrix_config):
    datasets = matrix_config['datasets']
    log_base = Path(matrix_config['log_dir'])
    log_base.mkdir(parents=True, exist_ok=True)

    batch_sizes = matrix_config['batch_sizes']
    precisions = matrix_config['precisions']
    repeats = matrix_config['repeats']
    num_passes = matrix_config['num_passes']
    warmup_batches = matrix_config['warmup_batches']
    seed = matrix_config['seed']
    num_workers = matrix_config['num_workers']
    pin_memory = matrix_config['pin_memory']

    for dataset_model in matrix_config['models']:
        dataset = dataset_model['dataset']
        dataset_models = dataset_model['models']
        print(f"Starting dataset: {dataset}")
        print(f"Found {len(dataset_models)} models: {[m['model_name'] for m in dataset_models]}")

        # Load existing results to check what has already been benchmarked
        results_csv = log_base / f"{dataset}_results.csv"
        existing_runs = set()
        if os.path.exists(results_csv):
            try:
                existing_df = pd.read_csv(results_csv)
                for _, row in existing_df.iterrows():
                    key = (row['model_name'], row['batch_size'], row['runtime_precision'], row['rep'])
                    existing_runs.add(key)
            except Exception:
                pass

        for rep in range(repeats):
            print(f"Starting repeat {rep+1}/{repeats} for {dataset}")
            random.seed(seed + rep)

            # Create conditions, restricting to amp for FP16 stored models
            conditions = []
            for midx, model_cfg in enumerate(dataset_models):
                # If model is stored as FP16 (amp), only run with amp precision
                if model_cfg.get('stored_precision') == 'amp':
                    model_precisions = ['amp']
                else:
                    model_precisions = precisions

                for bs, prec in product(batch_sizes, model_precisions):
                    conditions.append((midx, bs, prec))
            random.shuffle(conditions)

            for midx, bs, prec in conditions:
                model_cfg = dataset_models[midx]

                # Check if this combination already has results
                run_key = (model_cfg['model_name'], bs, prec, rep)
                if run_key in existing_runs:
                    print(f"Skipping (already exists): {model_cfg['model_name']}, bs={bs}, prec={prec}, rep={rep}")
                    continue

                exp_cfg = {
                    'model_name': model_cfg['model_name'],
                    'pruning_method': model_cfg['pruning_method'],
                    'sparsity': model_cfg['sparsity'],
                    'pruning_ratio': model_cfg['pruning_ratio'],
                    'stored_precision': model_cfg.get('stored_precision', 'unknown'),
                    'model_path': model_cfg['model_path'],
                    'batch_size': bs,
                    'precision': prec,
                    'device': 'cuda:0',
                    'num_passes': num_passes,
                    'warmup_batches': warmup_batches,
                    'seed': seed + rep * 1000,
                    'num_workers': num_workers,
                    'pin_memory': pin_memory,
                    'dataset': dataset,
                    'rep': rep
                }
                config = Config(experiment=exp_cfg, log_dir=str(log_base / f"{dataset}_rep{rep}"))

                print(f"Running: {exp_cfg['model_name']}, bs={bs}, prec={prec}, stored_as={exp_cfg['stored_precision']}, rep={rep}")
                try:
                    result = bench_fixed_passes(config)
                    if result:
                        print(f"  Throughput: {result['throughput_imgs_per_s']:.2f} imgs/s, "
                              f"auc: {result['auc']:.4f}, "
                              f"stored_precision: {result['stored_precision']}, "
                              f"energy_kWh_total={result['energy_kWh_total']:.6f}, "
                              f"energy_kWh_per_batch={result['energy_kWh_per_batch']:.6f}, "
                              f"energy_kWh_per_image={result['energy_kWh_per_image']:.6f}")
                except Exception as e:
                    print(f"  Error: {e}")
                time.sleep(30)

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    if len(data) < 2:
        return np.nan, np.nan
    bootstraps = [np.random.choice(data, len(data), replace=True) for _ in range(n_bootstrap)]
    medians = np.median(bootstraps, axis=1)
    lower = np.percentile(medians, (1 - ci) / 2 * 100)
    upper = np.percentile(medians, (1 + ci) / 2 * 100)
    return lower, upper

def paired_wilcoxon(base, test):
    if len(base) != len(test) or len(base) < 2:
        return np.nan
    return stats.wilcoxon(base, test).pvalue

def analyze_results(results_csv, log_dir, output_dir, dataset):
    df = pd.read_csv(results_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = df.groupby(['pruning_method', 'sparsity', 'batch_size', 'precision'])

    summary = []
    for (method, sparsity, bs, prec), gdf in groups:
        throughputs = gdf['throughput_imgs_per_s'].values
        aucs = gdf['auc'].dropna()
        energies_total = gdf['energy_kWh_total'].dropna()
        energies_per_batch = gdf['energy_kWh_per_batch'].dropna()
        energies_per_image = gdf['energy_kWh_per_image'].dropna()
        emissions = gdf['emissions_kg_total'].dropna()
        powers = gdf['gpu_power_w'].dropna()
        stored_precisions = gdf['stored_precision'].mode()[0] if 'stored_precision' in gdf.columns and len(gdf['stored_precision'].mode()) > 0 else 'unknown'
        
        median_tp = np.median(throughputs)
        iqr_low, iqr_high = np.percentile(throughputs, [25, 75])
        mean_tp = np.mean(throughputs)
        std_tp = np.std(throughputs)
        ci_low, ci_high = bootstrap_ci(throughputs)
        median_auc = np.median(aucs) if len(aucs) > 0 else np.nan
        mean_auc = np.mean(aucs) if len(aucs) > 0 else np.nan
        std_auc = np.std(aucs) if len(aucs) > 0 else np.nan
        auc_ci_low, auc_ci_high = bootstrap_ci(aucs) if len(aucs) > 1 else (np.nan, np.nan)
        median_energy_total = np.median(energies_total) if len(energies_total) > 0 else np.nan
        median_energy_per_batch = np.median(energies_per_batch) if len(energies_per_batch) > 0 else np.nan
        median_energy_per_image = np.median(energies_per_image) if len(energies_per_image) > 0 else np.nan
        median_emissions = np.median(emissions) if len(emissions) > 0 else np.nan
        median_power = np.median(powers) if len(powers) > 0 else np.nan
        tp_per_w = median_tp / median_power if (median_power and not math.isnan(median_power) and median_power > 0) else np.nan
        tp_per_kwh = median_tp / median_energy_total if (median_energy_total and not math.isnan(median_energy_total) and median_energy_total > 0) else np.nan

        summary.append({
            'dataset': dataset,
            'pruning_method': method,
            'sparsity': sparsity,
            'stored_precision': stored_precisions,
            'batch_size': bs,
            'precision': prec,
            'n_runs': len(throughputs),
            'median_throughput': median_tp,
            'mean_throughput': mean_tp,
            'std_throughput': std_tp,
            'iqr_low': iqr_low,
            'iqr_high': iqr_high,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'median_auc': median_auc,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'auc_ci_low': auc_ci_low,
            'auc_ci_high': auc_ci_high,
            'median_power_gpu_w': median_power,
            'median_energy_kWh_total': median_energy_total,
            'median_energy_kWh_per_batch': median_energy_per_batch,
            'median_energy_kWh_per_image': median_energy_per_image,
            'median_emissions_kg': median_emissions,
            'throughput_per_watt': tp_per_w,
            'throughput_per_kWh': tp_per_kwh
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / 'summary.csv', index=False)

    baseline_groups = summary_df[summary_df['pruning_method'] == 'baseline']
    for method in summary_df['pruning_method'].unique():
        if method == 'baseline':
            continue
        for sparsity in summary_df[summary_df['pruning_method'] == method]['sparsity'].unique():
            pruned_groups = summary_df[(summary_df['pruning_method'] == method) & (summary_df['sparsity'] == sparsity)]
            for _, bl_row in baseline_groups.iterrows():
                match_pruned = pruned_groups[(pruned_groups['batch_size'] == bl_row['batch_size']) &
                                            (pruned_groups['precision'] == bl_row['precision'])]
                if len(match_pruned) > 0:
                    pr_row = match_pruned.iloc[0]
                    speedup = pr_row['median_throughput'] / bl_row['median_throughput']
                    pct_more = (speedup - 1) * 100
                    energy_saving = (bl_row['median_energy_kWh_total'] - pr_row['median_energy_kWh_total']) / bl_row['median_energy_kWh_total'] * 100 if not math.isnan(bl_row['median_energy_kWh_total']) else np.nan
                    auc_diff = pr_row['median_auc'] - bl_row['median_auc'] if not (math.isnan(pr_row['median_auc']) or math.isnan(bl_row['median_auc'])) else np.nan
                    bl_tps = df[(df['pruning_method'] == 'baseline') & 
                               (df['batch_size'] == bl_row['batch_size']) & 
                               (df['precision'] == bl_row['precision'])]['throughput_imgs_per_s'].values
                    pr_tps = df[(df['pruning_method'] == method) & 
                               (df['sparsity'] == sparsity) & 
                               (df['batch_size'] == bl_row['batch_size']) & 
                               (df['precision'] == bl_row['precision'])]['throughput_imgs_per_s'].values
                    bl_aucs = df[(df['pruning_method'] == 'baseline') & 
                                (df['batch_size'] == bl_row['batch_size']) & 
                                (df['precision'] == bl_row['precision'])]['auc'].values
                    pr_aucs = df[(df['pruning_method'] == method) & 
                                (df['sparsity'] == sparsity) & 
                                (df['batch_size'] == bl_row['batch_size']) & 
                                (df['precision'] == bl_row['precision'])]['auc'].values
                    if len(bl_tps) == len(pr_tps) and len(bl_tps) > 1:
                        speedup_ratios = pr_tps / bl_tps
                        ci_speedup_low, ci_speedup_high = bootstrap_ci(speedup_ratios)
                        pval = paired_wilcoxon(bl_tps, pr_tps)
                    else:
                        ci_speedup_low, ci_speedup_high = np.nan, np.nan
                        pval = np.nan
                    if len(bl_aucs) == len(pr_aucs) and len(bl_aucs) > 1:
                        auc_pval = paired_wilcoxon(bl_aucs, pr_aucs)
                    else:
                        auc_pval = np.nan
                    print(f"{dataset}: {method} ({sparsity}, stored_as={pr_row['stored_precision']}) vs baseline (bs={bl_row['batch_size']}, prec={bl_row['precision']}): "
                          f"speedup={speedup:.2f} ({pct_more:.1f}%), energy_saving_pct={energy_saving:.1f}%, "
                          f"auc_diff={auc_diff:.4f}, auc_pval={auc_pval:.4f}, "
                          f"throughput_pval={pval:.4f}, 95% CI throughput [{ci_speedup_low:.2f}, {ci_speedup_high:.2f}]")

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=summary_df, x='batch_size', y='median_throughput', hue='pruning_method', style='sparsity', markers=True, dashes=False)
    plt.title(f'Throughput vs Batch Size ({dataset})')
    plt.savefig(output_dir / 'throughput_vs_bs.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    pivot = summary_df.pivot(index='batch_size', columns=['pruning_method', 'sparsity'], values='median_throughput')
    pivot.plot(kind='bar')
    plt.title(f'Throughput by Model and Batch Size ({dataset})')
    plt.savefig(output_dir / 'throughput_bar.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_df, x='batch_size', y='median_energy_kWh_total', hue='pruning_method')
    plt.title(f'Total Energy Consumption (kWh) by Model and Batch Size ({dataset})')
    plt.savefig(output_dir / 'energy_total_bar.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_df, x='batch_size', y='median_energy_kWh_per_batch', hue='pruning_method')
    plt.title(f'Energy per Batch (kWh) by Model and Batch Size ({dataset})')
    plt.savefig(output_dir / 'energy_per_batch_bar.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_df, x='batch_size', y='median_energy_kWh_per_image', hue='pruning_method')
    plt.title(f'Energy per Image (kWh) by Model and Batch Size ({dataset})')
    plt.savefig(output_dir / 'energy_per_image_bar.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=summary_df, x='batch_size', y='median_auc', hue='pruning_method', style='sparsity', markers=True, dashes=False)
    plt.title(f'AUC vs Batch Size ({dataset})')
    plt.ylabel('Median AUC')
    plt.savefig(output_dir / 'auc_vs_bs.png')
    plt.close()

    print(f"Analysis complete for {dataset}. Outputs in: {output_dir}")
    return summary_df

def run_full_analysis(log_base):
    analysis_dir = log_base / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for dataset in MATRIX_CONFIG['datasets']:
        results_csv = log_base / f"{dataset}_results.csv"
        if os.path.exists(results_csv):
            print(f"Analyzing {dataset}...")
            temp_output = analysis_dir / dataset
            temp_output.mkdir(parents=True, exist_ok=True)
            summary_df = analyze_results(str(results_csv), str(log_base), str(temp_output), dataset)
            all_summaries.append(summary_df)
    if all_summaries:
        global_summary = pd.concat(all_summaries, ignore_index=True)
        global_summary.to_csv(analysis_dir / 'global_summary.csv', index=False)
        print("Global summary saved to analysis/global_summary.csv")

if __name__ == "__main__":
    print("Starting full experiment on available datasets...")
    print(f"Looking for models in: {SAVE_DIR_BASE}")
    print(f"Model discovery results:")
    for dataset_info in MATRIX_CONFIG["models"]:
        print(f"  {dataset_info['dataset']}: {len(dataset_info['models'])} models found")
        for model in dataset_info['models']:
            print(f"    - {model['pruning_method']} ({model['sparsity']}, stored as {model['stored_precision']})")
    
    run_matrix(MATRIX_CONFIG)
    print("All runs complete. Running analysis...")
    run_full_analysis(Path(MATRIX_CONFIG['log_dir']))
    print("Full process complete! Check the timing_exps/ directory for outputs.")