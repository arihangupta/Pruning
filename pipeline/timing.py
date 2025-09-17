
#!/usr/bin/env python3
"""
Complete script to run the full factorial benchmarking experiment on all 5 MedMNIST datasets.
Dynamically discovers all models ending in '_final.pth' and 'baseline.pth' for each dataset.
Uses CustomResNet to account for reduced channels/neurons in pruned models.
Includes CodeCarbon for power utilization and detailed analysis with pruning method and sparsity.

Dependencies: torch, torchvision, pyyaml, psutil, codecarbon, scipy, matplotlib, seaborn, pandas
Install: pip install torch torchvision pyyaml psutil codecarbon scipy matplotlib seaborn pandas

Usage: python run_full_experiment.py
Outputs saved to /home/arihangupta/Pruning/dinov2/Pruning/time_experiment_results/
"""
import os
import time
import yaml
import json
import math
import random
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms as T
from collections import defaultdict
from typing import Dict, Any
import psutil
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from pathlib import Path
import hashlib
import subprocess as sp
import sys
from itertools import product
import glob
import re

# CodeCarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except Exception:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    print("Warning: codecarbon not available. Energy/emissions will be NaN.")

# Constants
IMG_SIZE = 224
SEED = 42
DATASETS = {
    "pathmnist": "/home/arihangupta/Pruning/dinov2/Pruning/datasets/pathmnist_224.npz",
    "dermamnist": "/home/arihangupta/Pruning/dinov2/Pruning/datasets/dermamnist_224.npz",
    "bloodmnist": "/home/arihangupta/Pruning/dinov2/Pruning/datasets/bloodmnist_224.npz",
    "octmnist": "/home/arihangupta/Pruning/dinov2/Pruning/datasets/octmnist_224.npz",
    "tissuemnist": "/home/arihangupta/Pruning/dinov2/Pruning/datasets/tissuemnist_224.npz",
}
SAVE_DIR_BASE = "/home/arihangupta/Pruning/dinov2/Pruning/combined_pruning_kd_experiment"
# Standard ResNet50 channel counts for each stage
ORIGINAL_PLANES = [64, 128, 256, 512]
# Known number of classes for MedMNIST datasets
DATASET_NUM_CLASSES = {
    "pathmnist": 9,
    "dermamnist": 7,
    "bloodmnist": 8,
    "octmnist": 4,
    "tissuemnist": 8
}
# Hardcoded matrix config (models will be dynamically populated)
MATRIX_CONFIG = {
    "datasets": ["pathmnist", "dermamnist", "bloodmnist", "octmnist", "tissuemnist"],
    "log_dir": "/home/arihangupta/Pruning/dinov2/Pruning/time_experiment_results",
    "time_budget_s": 300,
    "warmup_batches": 50,
    "seed": 42,
    "num_workers": 4,
    "pin_memory": True,
    "batch_sizes": [1, 8, 32, 128],
    "precisions": ["fp32", "amp"],
    "repeats": 7
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class CustomResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], stage_planes=[64, 128, 256, 512], num_classes=1000):
        super().__init__()
        self.inplanes = stage_planes[0]
        self.conv1 = nn.Conv2d(3, stage_planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(stage_planes[0])
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

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
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

def parse_model_name(filename):
    """Parse model filename to extract pruning method, sparsity, and keep_ratio."""
    basename = os.path.basename(filename)
    if basename == "baseline.pth":
        return {"model_name": "baseline", "pruning_method": "baseline", "sparsity": "0%", "keep_ratio": 1.0}
    # Match files ending in _final.pth
    match = re.match(r"(.+)_r(\d+)pruned_final\.pth$", basename)
    if not match:
        return None
    method, sparsity_str = match.groups()
    sparsity = int(sparsity_str)
    keep_ratio = 1.0 - (sparsity / 100.0)
    model_name = f"{method}_r{sparsity}"
    return {"model_name": model_name, "pruning_method": method, "sparsity": f"{sparsity}%", "keep_ratio": keep_ratio}

def discover_models():
    """Dynamically discover all models ending in _final.pth and baseline.pth for each dataset."""
    models = []
    for dataset in MATRIX_CONFIG["datasets"]:
        model_dir = os.path.join(SAVE_DIR_BASE, dataset)
        model_files = glob.glob(os.path.join(model_dir, "*_final.pth")) + [os.path.join(model_dir, "baseline.pth")]
        dataset_models = []
        for model_path in model_files:
            parsed = parse_model_name(model_path)
            if parsed:
                parsed["model_path"] = model_path
                dataset_models.append(parsed)
        models.append({"dataset": dataset, "models": dataset_models})
    return models

def build_model_for_load(model_name, num_classes, model_path, keep_ratio=1.0):
    """Build model architecture based on model type and load weights."""
    if "baseline" in model_name:
        # Use full planes for baseline
        stage_planes = ORIGINAL_PLANES[:]
    else:
        # For pruned models, compute stage_planes based on keep_ratio
        stage_planes = [max(1, int(p * keep_ratio)) for p in ORIGINAL_PLANES]
    model = CustomResNet(block=Bottleneck, layers=[3, 4, 6, 3], stage_planes=stage_planes, num_classes=num_classes)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
    return model

# Update MATRIX_CONFIG with discovered models
MATRIX_CONFIG["models"] = discover_models()

@dataclass
class Config:
    """Parsed config with separate experiment and log_dir."""
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
    def __init__(self, imgs_np, labels_np, img_size=224):
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

def make_test_loader(npz_path, batch_size):
    data = np.load(npz_path, mmap_mode="r")
    X_test, y_test = data["test_images"], data["test_labels"].flatten()
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

def load_model(config, num_classes, dataset_name):
    model_path = config.experiment['model_path']
    if not os.path.exists(model_path):
        raise ValueError(f"Model not found: {model_path}")
    model = build_model_for_load(
        config.experiment['model_name'],
        num_classes,
        model_path,
        config.experiment['keep_ratio']
    )
    model = model.to(config.experiment['device']).eval()
    if config.experiment['precision'] == 'amp':
        model = model.half()
    return model

def get_num_classes(dataset_name, dataset_path):
    """Dynamically determine number of classes from info.csv or fallback to known values."""
    info_path = dataset_path.replace('.npz', '_info.csv')
    if os.path.exists(info_path):
        try:
            df = pd.read_csv(info_path)
            return len(df['label'].unique())
        except Exception:
            pass
    return DATASET_NUM_CLASSES.get(dataset_name, 2)  # Fallback to 2 if unknown

def start_tracker(save_dir: str, project_name: str, output_file: str="emissions.csv", measure_power_secs: int=10):
    if not CODECARBON_AVAILABLE:
        return None
    os.makedirs(save_dir, exist_ok=True)
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=save_dir,
        output_file=output_file,
        measure_power_secs=measure_power_secs,
        save_to_file=True
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

def stop_tracker_and_get_metrics(tracker, save_dir: str, project_name: str):
    if tracker is None:
        return {
            "emissions_kg": float("nan"),
            "energy_kwh": float("nan"),
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
    energy_kwh = float(raw.get("energy_consumed", float("nan"))) if raw is not None and "energy_consumed" in raw else float("nan")
    cpu_power = float(raw.get("cpu_power", float("nan"))) if raw is not None and "cpu_power" in raw else float("nan")
    gpu_power = float(raw.get("gpu_power", float("nan"))) if raw is not None and "gpu_power" in raw else float("nan")
    ram_power = float(raw.get("ram_power", float("nan"))) if raw is not None and "ram_power" in raw else float("nan")
    emissions_kg = float(raw.get("emissions", float("nan"))) if raw is not None and "emissions" in raw else (float(emissions_val) if emissions_val is not None else float("nan"))
    return {
        "emissions_kg": emissions_kg,
        "energy_kwh": energy_kwh,
        "cpu_power_w": cpu_power,
        "gpu_power_w": gpu_power,
        "ram_power_w": ram_power,
        "raw_row": raw
    }

def bench_fixed_time(config):
    """Run a single benchmark condition."""
    set_seed(config.experiment['seed'])
    set_env_threads(omp_threads=4, mkl_threads=4)
    device = torch.device(config.experiment['device'])
    dataset_name = config.experiment['dataset']
    dataset_path = DATASETS.get(dataset_name, None)
    if not dataset_path:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load data
    test_loader = make_test_loader(dataset_path, config.experiment['batch_size'])

    # Load model
    num_classes = get_num_classes(dataset_name, dataset_path)
    model = load_model(config, num_classes, dataset_name)

    # Logs
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{config.experiment['model_name']}_b{config.experiment['batch_size']}_{config.experiment['precision']}_r{config.experiment.get('rep', 0)}"
    project_name = f"{run_id}_inference"

    # Per-batch logs
    batch_logs = []

    # Start CodeCarbon tracker
    tracker = start_tracker(str(log_dir), project_name, measure_power_secs=10) if CODECARBON_AVAILABLE else None

    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(config.experiment['warmup_batches']):
            batch = next(iter(test_loader))
            inputs = batch[0].to(device)
            if config.experiment['precision'] == 'amp':
                with torch.cuda.amp.autocast():
                    _ = model(inputs)
            else:
                _ = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Clear cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Timed run
    t0 = time.time()
    images_processed = 0
    batch_idx = 0
    loader_iter = iter(test_loader)
    while time.time() - t0 < config.experiment['time_budget_s']:
        batch_start = time.time()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(test_loader)  # Restart dataset
            batch = next(loader_iter)
        inputs = batch[0].to(device)
        if len(inputs) != config.experiment['batch_size']:
            break  # End of dataset
        with torch.no_grad():
            if config.experiment['precision'] == 'amp':
                with torch.cuda.amp.autocast():
                    _ = model(inputs)
            else:
                _ = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        batch_end = time.time()
        batch_elapsed = (batch_end - batch_start) * 1000  # ms
        batch_logs.append({
            'batch_idx': batch_idx,
            'batch_size': len(inputs),
            'batch_start_ts': batch_start,
            'batch_end_ts': batch_end,
            'batch_elapsed_ms': batch_elapsed
        })
        images_processed += len(inputs)
        batch_idx += 1

    elapsed_s = time.time() - t0
    throughput = images_processed / elapsed_s if elapsed_s > 0 else 0

    # Peak mem
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else 0

    # Stop CodeCarbon and get metrics
    metrics = stop_tracker_and_get_metrics(tracker, str(log_dir), project_name)
    energy_kwh = metrics["energy_kwh"]
    emissions_kg = metrics["emissions_kg"]
    cpu_power_w = metrics["cpu_power_w"]
    gpu_power_w = metrics["gpu_power_w"]
    ram_power_w = metrics["ram_power_w"]
    avg_power_w = gpu_power_w if not math.isnan(gpu_power_w) else float('nan')

    # Save batch logs
    batch_df = pd.DataFrame(batch_logs)
    batch_log_file = log_dir / f"per_batch_log_{run_id}.csv"
    batch_df.to_csv(batch_log_file, index=False)

    # Metadata
    metadata = {
        'run_id': run_id,
        'model_name': config.experiment['model_name'],
        'pruning_method': config.experiment['pruning_method'],
        'sparsity': config.experiment['sparsity'],
        'batch_size': config.experiment['batch_size'],
        'precision': config.experiment['precision'],
        'rep': config.experiment.get('rep', 0),
        'time_budget_s': config.experiment['time_budget_s'],
        'images_processed': images_processed,
        'elapsed_s': elapsed_s,
        'throughput_imgs_per_s': throughput,
        'median_batch_ms': batch_df['batch_elapsed_ms'].median() if not batch_df.empty else float('nan'),
        'p50_ms': batch_df['batch_elapsed_ms'].quantile(0.5) if not batch_df.empty else float('nan'),
        'p90_ms': batch_df['batch_elapsed_ms'].quantile(0.9) if not batch_df.empty else float('nan'),
        'peak_gpu_mem_MB': peak_mem,
        'avg_power_W': avg_power_w,
        'energy_kWh_total': energy_kwh,
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

    # Save per-run JSON
    results_file = log_dir / f"results_{run_id}.json"
    with open(results_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Append to dataset-specific CSV
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
    time_budget_s = matrix_config['time_budget_s']
    warmup_batches = matrix_config['warmup_batches']
    seed = matrix_config['seed']
    num_workers = matrix_config['num_workers']
    pin_memory = matrix_config['pin_memory']

    for dataset_model in matrix_config['models']:
        dataset = dataset_model['dataset']
        dataset_models = dataset_model['models']
        print(f"Starting dataset: {dataset}")
        print(f"Found {len(dataset_models)} models: {[m['model_name'] for m in dataset_models]}")

        for rep in range(repeats):
            print(f"Starting repeat {rep+1}/{repeats} for {dataset}")
            random.seed(seed + rep)

            conditions = list(product(range(len(dataset_models)), batch_sizes, precisions))
            random.shuffle(conditions)

            for midx, bs, prec in conditions:
                model_cfg = dataset_models[midx]
                exp_cfg = {
                    'model_name': model_cfg['model_name'],
                    'pruning_method': model_cfg['pruning_method'],
                    'sparsity': model_cfg['sparsity'],
                    'keep_ratio': model_cfg['keep_ratio'],
                    'model_path': model_cfg['model_path'],
                    'batch_size': bs,
                    'precision': prec,
                    'device': 'cuda:0',
                    'time_budget_s': time_budget_s,
                    'warmup_batches': warmup_batches,
                    'seed': seed + rep * 1000,
                    'num_workers': num_workers,
                    'pin_memory': pin_memory,
                    'dataset': dataset,
                    'rep': rep
                }
                config = Config(experiment=exp_cfg, log_dir=str(log_base / f"{dataset}_rep{rep}"))

                print(f"Running: {exp_cfg['model_name']}, bs={bs}, prec={prec}, rep={rep}")
                try:
                    result = bench_fixed_time(config)
                    print(f"  Throughput: {result['throughput_imgs_per_s']:.2f} imgs/s, energy_kWh={result['energy_kWh_total']:.6f}")
                except Exception as e:
                    print(f"  Error: {e}")
                time.sleep(30)  # Stabilize

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
        energies = gdf['energy_kWh_total'].dropna()
        emissions = gdf['emissions_kg_total'].dropna()
        powers = gdf['gpu_power_w'].dropna()
        median_tp = np.median(throughputs)
        iqr_low, iqr_high = np.percentile(throughputs, [25, 75])
        mean_tp = np.mean(throughputs)
        std_tp = np.std(throughputs)
        ci_low, ci_high = bootstrap_ci(throughputs)

        median_energy = np.median(energies) if len(energies) > 0 else np.nan
        median_emissions = np.median(emissions) if len(emissions) > 0 else np.nan
        median_power = np.median(powers) if len(powers) > 0 else np.nan
        tp_per_w = median_tp / median_power if (median_power and not math.isnan(median_power) and median_power > 0) else np.nan
        tp_per_kwh = median_tp / median_energy if (median_energy and not math.isnan(median_energy) and median_energy > 0) else np.nan

        summary.append({
            'dataset': dataset,
            'pruning_method': method,
            'sparsity': sparsity,
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
            'median_power_gpu_w': median_power,
            'median_energy_kWh': median_energy,
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
                    energy_saving = (bl_row['median_energy_kWh'] - pr_row['median_energy_kWh']) / bl_row['median_energy_kWh'] * 100 if not math.isnan(bl_row['median_energy_kWh']) else np.nan
                    bl_tps = df[(df['pruning_method'] == 'baseline') & (df['batch_size'] == bl_row['batch_size']) & (df['precision'] == bl_row['precision'])]['throughput_imgs_per_s']
                    pr_tps = df[(df['pruning_method'] == method) & (df['sparsity'] == sparsity) & (df['batch_size'] == bl_row['batch_size']) & (df['precision'] == bl_row['precision'])]['throughput_imgs_per_s']
                    pval = paired_wilcoxon(bl_tps.values, pr_tps.values)
                    ci_speedup_low, ci_speedup_high = bootstrap_ci(pr_tps.values / bl_tps.values)
                    print(f"{dataset}: {method} ({sparsity}) vs baseline (bs={bl_row['batch_size']}, prec={bl_row['precision']}): speedup={speedup:.2f} ({pct_more:.1f}%), energy_saving_pct={energy_saving:.1f}%, p={pval:.4f}, 95% CI [{ci_speedup_low:.2f}, {ci_speedup_high:.2f}]")

    # Plots
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=summary_df, x='batch_size', y='median_throughput', hue='pruning_method', style='sparsity', markers=True, dashes=False, errorbar=None)
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
    sns.barplot(data=summary_df, x='batch_size', y='median_energy_kWh', hue='pruning_method', style='sparsity')
    plt.title(f'Energy Consumption (kWh) by Model and Batch Size ({dataset})')
    plt.savefig(output_dir / 'energy_bar.png')
    plt.close()

    print(f"Analysis complete for {dataset}. Outputs in: {output_dir}")
    return summary_df

def run_full_analysis(log_base):
    """Run analysis for all datasets."""
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
    print("Starting full experiment on all 5 datasets...")
    run_matrix(MATRIX_CONFIG)
    print("All runs complete. Running analysis...")
    run_full_analysis(Path(MATRIX_CONFIG['log_dir']))
    print("Full process complete! Check time_experiment_results/ for outputs.")
