#!/usr/bin/env python3
"""
Complete script to run the full factorial benchmarking experiment on Vision Transformer models.
Dynamically discovers baseline, quantized (AMP), and knowledge distillation models.
Uses the same structure and output format as the ResNet benchmarking script.
Measures: inference latency, throughput, memory usage, AUC, and energy consumption.
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
from typing import Dict, Any, List
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from pathlib import Path
from itertools import product
import glob
import sys
import subprocess as sp
import logging
from sklearn.metrics import roc_auc_score, accuracy_score
import timm
from tqdm import tqdm

# Configure debug logging to file only
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/arihangupta/Pruning/dinov2/Pruning/Vision/vit_benchmark_debug.log'),
    ]
)
logger = logging.getLogger(__name__)

# CodeCarbon for energy tracking
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

# Constants
IMG_SIZE = 224
SEED = 42
BASELINE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/new_baseline"
MODELS_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/more_epochs"
DATASETS_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
OUTPUT_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/benchmarking_results"

DATASETS = {
    "bloodmnist": f"{DATASETS_DIR}/bloodmnist_224.npz",
    "dermamnist": f"{DATASETS_DIR}/dermamnist_224.npz",
    "pathmnist": f"{DATASETS_DIR}/pathmnist_224.npz",
}

DATASET_NUM_CLASSES = {
    "bloodmnist": 8,
    "dermamnist": 7,
    "pathmnist": 9,
}

# Benchmarking configuration - matching ResNet script structure
MATRIX_CONFIG = {
    "datasets": ["bloodmnist", "dermamnist", "pathmnist"],
    "batch_sizes": [8, 32],
    "precisions": ["fp32", "amp"],
    "num_passes": 3,  # Number of complete passes through test set
    "warmup_batches": 50,
    "repeats": 3,  # Number of times to repeat each configuration
    "num_workers": 4,
    "pin_memory": True,
    "log_dir": OUTPUT_DIR,
    "seed": SEED,
}


@dataclass
class Config:
    experiment: Dict[str, Any]
    log_dir: str


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_env_threads(omp_threads=4, mkl_threads=4):
    """Set environment threading."""
    os.environ['OMP_NUM_THREADS'] = str(omp_threads)
    os.environ['MKL_NUM_THREADS'] = str(mkl_threads)
    os.environ['PYTHONHASHSEED'] = '0'


class NumpyMemmapDataset(torch.utils.data.Dataset):
    """Dataset wrapper for numpy memmap arrays."""
    def __init__(self, imgs_np, labels_np, img_size=224):
        self.imgs = imgs_np
        self.labels = labels_np
        self.img_size = img_size
        self.base_tfms = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])

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
    """Load test dataset."""
    data = np.load(npz_path, mmap_mode="r")
    X_test = data["test_images"]
    y_test = data["test_labels"].flatten()
    
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE)
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=MATRIX_CONFIG["num_workers"],
        pin_memory=MATRIX_CONFIG["pin_memory"]
    )
    
    return test_loader, len(test_ds)


def parse_model_name(model_path, dataset):
    """
    Parse model filename to extract metadata.
    Supports: baseline, quantized (AMP), and knowledge distillation models.
    """
    basename = os.path.basename(model_path)
    logger.debug(f"Parsing model file: {basename} for dataset: {dataset}")
    
    # Handle baseline models
    if "_pretrained.pth" in basename:
        model_name = basename.replace(f"_{dataset}_pretrained.pth", "")
        return {
            "model_name": model_name,
            "full_name": basename.replace(".pth", ""),
            "pruning_method": "baseline",
            "sparsity": "0%",
            "stored_precision": "fp32",
            "teacher_model": None,
            "student_model": None
        }
    
    # Handle quantized models (AMP)
    if "_amp.pth" in basename:
        model_name = basename.replace(f"_{dataset}_amp.pth", "")
        return {
            "model_name": model_name,
            "full_name": basename.replace(".pth", ""),
            "pruning_method": "quantization",
            "sparsity": "50%",  # AMP typically represents ~50% reduction
            "stored_precision": "amp",
            "teacher_model": None,
            "student_model": None
        }
    
    # Handle knowledge distillation models
    if "_kd_from_" in basename:
        # Format: vit_tiny_patch16_224_{dataset}_kd_from_vit_base_patch16_224.pth
        parts = basename.replace(".pth", "").split("_kd_from_")
        student_full = parts[0].replace(f"_{dataset}", "")
        teacher_name = parts[1] if len(parts) > 1 else None
        
        # Estimate sparsity based on student/teacher size
        sparsity_map = {
            ("base", "small"): "33%",
            ("base", "tiny"): "67%",
            ("small", "tiny"): "50%"
        }
        
        student_size = "tiny" if "tiny" in student_full else ("small" if "small" in student_full else "base")
        teacher_size = "tiny" if teacher_name and "tiny" in teacher_name else ("small" if teacher_name and "small" in teacher_name else "base")
        sparsity = sparsity_map.get((teacher_size, student_size), "unknown")
        
        return {
            "model_name": student_full,
            "full_name": basename.replace(".pth", ""),
            "pruning_method": "kd",
            "sparsity": sparsity,
            "stored_precision": "fp32",
            "teacher_model": teacher_name,
            "student_model": student_full
        }
    
    logger.debug(f"Skipping {basename}: unrecognized format")
    return None


def discover_models():
    """
    Discover all available models: baselines, quantized (AMP), and KD models.
    Returns a structured list matching ResNet script format.
    """
    models = []
    
    print("\n" + "="*80)
    print("DISCOVERING MODELS")
    print("="*80)
    
    for dataset in MATRIX_CONFIG["datasets"]:
        dataset_models = []
        
        # Baseline models
        for model_size in ['tiny', 'small', 'base']:
            model_name = f"vit_{model_size}_patch16_224"
            baseline_path = os.path.join(BASELINE_DIR, f"{model_name}_{dataset}_pretrained.pth")
            if os.path.exists(baseline_path):
                parsed = parse_model_name(baseline_path, dataset)
                if parsed:
                    parsed["model_path"] = baseline_path
                    dataset_models.append(parsed)
                    print(f"✓ Found baseline: {model_name} for {dataset}")
        
        # Quantized models (AMP)
        quant_dir = os.path.join(MODELS_DIR, "quantized_amp")
        if os.path.exists(quant_dir):
            for model_size in ['tiny', 'small', 'base']:
                model_name = f"vit_{model_size}_patch16_224"
                quant_path = os.path.join(quant_dir, f"{model_name}_{dataset}_amp.pth")
                if os.path.exists(quant_path):
                    parsed = parse_model_name(quant_path, dataset)
                    if parsed:
                        parsed["model_path"] = quant_path
                        dataset_models.append(parsed)
                        print(f"✓ Found quantized: {model_name}_amp for {dataset}")
        
        # Knowledge distillation models
        kd_dir = os.path.join(MODELS_DIR, "knowledge_distillation")
        if os.path.exists(kd_dir):
            kd_files = glob.glob(os.path.join(kd_dir, f"*_{dataset}_kd_from_*.pth"))
            for kd_path in kd_files:
                parsed = parse_model_name(kd_path, dataset)
                if parsed:
                    parsed["model_path"] = kd_path
                    dataset_models.append(parsed)
                    print(f"✓ Found KD model: {parsed['full_name']} for {dataset}")
        
        if dataset_models:
            models.append({"dataset": dataset, "models": dataset_models})
        else:
            print(f"Warning: No models found for {dataset}")
    
    return models


def load_model(config, num_classes):
    """Load a model from checkpoint."""
    model_path = config.experiment['model_path']
    model_name = config.experiment['model_name']
    precision = config.experiment['precision']
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Create model
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict)
    
    # Handle precision
    if precision == 'amp' or config.experiment.get('stored_precision') == 'amp':
        model = model.half()
    else:
        model = model.float()
    
    return model


def start_tracker(save_dir: str, project_name: str, output_file: str="emissions.csv", measure_power_secs: int=10):
    """Start CodeCarbon emissions tracker."""
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
    """Read latest emissions data from CodeCarbon CSV."""
    csv_path = os.path.join(save_dir, "emissions.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        df_match = df[df["project_name"] == project_name]
        if df_match.shape[0] == 0:
            return None
        return df_match.iloc[-1].to_dict()
    except Exception:
        return None


def stop_tracker_and_get_metrics(tracker, save_dir: str, project_name: str, images_processed: int, num_batches: int):
    """Stop tracker and extract energy metrics."""
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
        logger.error(f"Error stopping CodeCarbon tracker: {e}")
        emissions_val = None
    
    raw = _read_latest_tracker_row(save_dir, project_name)
    
    energy_kwh_total = float(raw.get("energy_consumed", float("nan"))) if raw else float("nan")
    cpu_power = float(raw.get("cpu_power", float("nan"))) if raw else float("nan")
    gpu_power = float(raw.get("gpu_power", float("nan"))) if raw else float("nan")
    ram_power = float(raw.get("ram_power", float("nan"))) if raw else float("nan")
    emissions_kg = float(raw.get("emissions", float("nan"))) if raw else (float(emissions_val) if emissions_val else float("nan"))
    
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
    """
    Benchmark a single model configuration.
    Main benchmarking function matching ResNet script structure.
    """
    set_seed(config.experiment['seed'])
    set_env_threads(omp_threads=4, mkl_threads=4)
    
    device = torch.device(config.experiment['device'])
    dataset_name = config.experiment['dataset']
    batch_size = config.experiment['batch_size']
    num_passes = config.experiment['num_passes']
    
    dataset_path = DATASETS.get(dataset_name)
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
            logger.error(f"Error checking GPU memory: {e}")
    
    # Load dataset
    test_loader, dataset_size = make_test_loader(dataset_path, batch_size)
    num_classes = DATASET_NUM_CLASSES[dataset_name]
    
    # Load model
    try:
        model = load_model(config, num_classes)
        model = model.to(device).eval()
    except Exception as e:
        print(f"Failed to load model {config.experiment['model_name']}: {e}")
        raise
    
    # Setup logging
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{config.experiment['full_name']}_b{batch_size}_{config.experiment['precision']}_r{config.experiment.get('rep', 0)}"
    project_name = f"{run_id}_inference"
    
    batch_logs = []
    probs_list = []
    labels_list = []
    images_processed = 0
    total_batches = 0
    
    # Start energy tracking
    tracker = start_tracker(str(log_dir), project_name, measure_power_secs=10)
    
    model_dtype = next(model.parameters()).dtype
    
    # Warmup
    print(f"Warming up ({config.experiment['warmup_batches']} batches)...")
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= config.experiment['warmup_batches']:
                break
            inputs = inputs.to(device)
            if inputs.dtype != model_dtype:
                inputs = inputs.to(dtype=model_dtype)
            
            if config.experiment['precision'] == 'amp':
                with torch.amp.autocast('cuda'):
                    _ = model(inputs)
            else:
                _ = model(inputs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Reset memory stats
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark inference
    print(f"Running benchmark ({num_passes} passes)...")
    t0 = time.time()
    
    with torch.no_grad():
        for pass_idx in range(num_passes):
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                batch_start = time.time()
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if inputs.dtype != model_dtype:
                    inputs = inputs.to(dtype=model_dtype)
                
                # Inference
                if config.experiment['precision'] == 'amp':
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                batch_end = time.time()
                batch_elapsed = (batch_end - batch_start) * 1000  # ms
                
                # Log batch timing
                batch_logs.append({
                    'batch_idx': total_batches,
                    'batch_size': len(inputs),
                    'batch_start_ts': batch_start,
                    'batch_end_ts': batch_end,
                    'batch_elapsed_ms': batch_elapsed,
                    'pass_idx': pass_idx
                })
                
                # Collect predictions for AUC
                if outputs.dtype == torch.float16:
                    outputs = outputs.float()
                
                outputs = torch.clamp(outputs, min=-100, max=100)
                probs = torch.softmax(outputs, dim=1)
                probs_np = probs.cpu().numpy()
                
                # Normalize probabilities
                probs_sum = probs_np.sum(axis=1, keepdims=True)
                probs_np = probs_np / probs_sum
                
                # Check for invalid values
                if np.any(np.isnan(probs_np)) or np.any(np.isinf(probs_np)):
                    logger.warning(f"Invalid probabilities in batch {batch_idx}, skipping...")
                    continue
                
                probs_list.append(probs_np)
                labels_list.append(labels.cpu().numpy())
                
                images_processed += len(inputs)
                total_batches += 1
    
    elapsed_s = time.time() - t0
    throughput = images_processed / elapsed_s if elapsed_s > 0 else 0
    
    # Calculate AUC
    auc = float('nan')
    if probs_list:
        all_probs = np.concatenate(probs_list)
        all_labels = np.concatenate(labels_list)
        try:
            auc = roc_auc_score(
                all_labels,
                all_probs,
                multi_class="ovr",
                average='weighted',
                labels=list(range(num_classes))
            )
        except Exception as e:
            logger.error(f"AUC calculation failed: {e}")
    
    # Memory usage
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else 0.0
    
    # Stop energy tracking and get metrics
    energy_metrics = stop_tracker_and_get_metrics(tracker, str(log_dir), project_name, images_processed, total_batches)
    
    # Save per-batch logs
    batch_df = pd.DataFrame(batch_logs)
    batch_log_file = log_dir / f"per_batch_log_{run_id}.csv"
    batch_df.to_csv(batch_log_file, index=False)
    
    # Compile metadata
    metadata = {
        'run_id': run_id,
        'model_name': config.experiment['full_name'],
        'pruning_method': config.experiment['pruning_method'],
        'sparsity': config.experiment['sparsity'],
        'stored_precision': config.experiment.get('stored_precision', 'unknown'),
        'batch_size': batch_size,
        'runtime_precision': config.experiment['precision'],
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
        'avg_power_W': energy_metrics['gpu_power_w'],
        'energy_kWh_total': energy_metrics['energy_kwh_total'],
        'energy_kWh_per_batch': energy_metrics['energy_kwh_per_batch'],
        'energy_kWh_per_image': energy_metrics['energy_kwh_per_image'],
        'emissions_kg_total': energy_metrics['emissions_kg'],
        'cpu_power_w': energy_metrics['cpu_power_w'],
        'gpu_power_w': energy_metrics['gpu_power_w'],
        'ram_power_w': energy_metrics['ram_power_w'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seed': config.experiment['seed'],
        'git_commit': get_git_commit(),
        'py_version': f"{sys.version_info.major}.{sys.version_info.minor}",
        'torch_version': torch.__version__,
        'dataset': config.experiment['dataset'],
        'teacher_model': config.experiment.get('teacher_model'),
        'student_model': config.experiment.get('student_model')
    }
    
    # Save results JSON
    results_file = log_dir / f"results_{run_id}.json"
    with open(results_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Append to dataset results CSV
    results_csv = log_dir.parent / f"{dataset_name}_results.csv"
    pd.DataFrame([metadata]).to_csv(
        results_csv, 
        mode='a', 
        header=not os.path.exists(results_csv), 
        index=False
    )
    
    print(f"  Throughput: {throughput:.2f} imgs/s, "
          f"auc: {auc:.4f}, "
          f"stored_precision: {metadata['stored_precision']}, "
          f"energy_kWh_total={energy_metrics['energy_kwh_total']:.6f}, "
          f"energy_kWh_per_batch={energy_metrics['energy_kwh_per_batch']:.6f}, "
          f"energy_kWh_per_image={energy_metrics['energy_kwh_per_image']:.6f}")
    
    # Cleanup
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return metadata


def get_git_commit():
    """Get current git commit hash."""
    try:
        return sp.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except:
        return 'unknown'


def run_matrix(matrix_config):
    """
    Run full factorial experiment across all configurations.
    Main execution function matching ResNet script structure.
    """
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
        print(f"\nStarting dataset: {dataset}")
        print(f"Found {len(dataset_models)} models: {[m['full_name'] for m in dataset_models]}")
        
        for rep in range(repeats):
            print(f"\nStarting repeat {rep+1}/{repeats} for {dataset}")
            random.seed(seed + rep)
            
            # Create conditions list
            conditions = []
            for midx, model_cfg in enumerate(dataset_models):
                # If model is stored as AMP, only run with amp precision
                if model_cfg.get('stored_precision') == 'amp':
                    model_precisions = ['amp']
                else:
                    model_precisions = precisions
                
                for bs, prec in product(batch_sizes, model_precisions):
                    conditions.append((midx, bs, prec))
            
            # Shuffle conditions to avoid systematic biases
            random.shuffle(conditions)
            
            # Run each condition
            for midx, bs, prec in conditions:
                model_cfg = dataset_models[midx]
                exp_cfg = {
                    'model_name': model_cfg['model_name'],
                    'full_name': model_cfg['full_name'],
                    'pruning_method': model_cfg['pruning_method'],
                    'sparsity': model_cfg['sparsity'],
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
                    'rep': rep,
                    'teacher_model': model_cfg.get('teacher_model'),
                    'student_model': model_cfg.get('student_model')
                }
                
                config = Config(
                    experiment=exp_cfg,
                    log_dir=str(log_base / f"{dataset}_rep{rep}")
                )
                
                print(f"\nRunning: {exp_cfg['full_name']}, bs={bs}, prec={prec}, "
                      f"stored_as={exp_cfg['stored_precision']}, rep={rep}")
                
                try:
                    result = bench_fixed_passes(config)
                    if result is None:
                        print("  Skipped due to memory constraints")
                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Cool down between runs
                time.sleep(30)


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Calculate bootstrap confidence intervals."""
    if len(data) < 2:
        return np.nan, np.nan
    bootstraps = [np.random.choice(data, len(data), replace=True) for _ in range(n_bootstrap)]
    medians = np.median(bootstraps, axis=1)
    lower = np.percentile(medians, (1 - ci) / 2 * 100)
    upper = np.percentile(medians, (1 + ci) / 2 * 100)
    return lower, upper


def paired_wilcoxon(base, test):
    """Perform paired Wilcoxon test."""
    if len(base) != len(test) or len(base) < 2:
        return np.nan
    try:
        return stats.wilcoxon(base, test).pvalue
    except:
        return np.nan


def analyze_results(results_csv, log_dir, output_dir, dataset):
    """
    Analyze benchmark results and generate visualizations.
    Matches ResNet script analysis structure.
    """
    df = pd.read_csv(results_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    groups = df.groupby(['pruning_method', 'sparsity', 'batch_size', 'runtime_precision'])
    
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
    
    # Compare with baselines
    baseline_groups = summary_df[summary_df['pruning_method'] == 'baseline']
    
    print(f"\n{'='*80}")
    print(f"COMPARISONS WITH BASELINE - {dataset.upper()}")
    print(f"{'='*80}\n")
    
    for method in summary_df['pruning_method'].unique():
        if method == 'baseline':
            continue
        
        for sparsity in summary_df[summary_df['pruning_method'] == method]['sparsity'].unique():
            pruned_groups = summary_df[(summary_df['pruning_method'] == method) & 
                                      (summary_df['sparsity'] == sparsity)]
            
            for _, bl_row in baseline_groups.iterrows():
                match_pruned = pruned_groups[
                    (pruned_groups['batch_size'] == bl_row['batch_size']) &
                    (pruned_groups['precision'] == bl_row['precision'])
                ]
                
                if len(match_pruned) > 0:
                    pr_row = match_pruned.iloc[0]
                    speedup = pr_row['median_throughput'] / bl_row['median_throughput']
                    pct_more = (speedup - 1) * 100
                    
                    energy_saving = ((bl_row['median_energy_kWh_total'] - pr_row['median_energy_kWh_total']) / 
                                    bl_row['median_energy_kWh_total'] * 100 
                                    if not math.isnan(bl_row['median_energy_kWh_total']) else np.nan)
                    
                    auc_diff = (pr_row['median_auc'] - bl_row['median_auc'] 
                               if not (math.isnan(pr_row['median_auc']) or math.isnan(bl_row['median_auc'])) 
                               else np.nan)
                    
                    # Statistical tests
                    bl_tps = df[(df['pruning_method'] == 'baseline') & 
                               (df['batch_size'] == bl_row['batch_size']) & 
                               (df['runtime_precision'] == bl_row['precision'])]['throughput_imgs_per_s'].values
                    
                    pr_tps = df[(df['pruning_method'] == method) & 
                               (df['sparsity'] == sparsity) & 
                               (df['batch_size'] == bl_row['batch_size']) & 
                               (df['runtime_precision'] == bl_row['precision'])]['throughput_imgs_per_s'].values
                    
                    if len(bl_tps) == len(pr_tps) and len(bl_tps) > 1:
                        speedup_ratios = pr_tps / bl_tps
                        ci_speedup_low, ci_speedup_high = bootstrap_ci(speedup_ratios)
                        pval = paired_wilcoxon(bl_tps, pr_tps)
                    else:
                        ci_speedup_low, ci_speedup_high = np.nan, np.nan
                        pval = np.nan
                    
                    print(f"{method} ({sparsity}, stored_as={pr_row['stored_precision']}) vs baseline "
                          f"(bs={bl_row['batch_size']}, prec={bl_row['precision']}): "
                          f"speedup={speedup:.2f} ({pct_more:.1f}%), "
                          f"energy_saving_pct={energy_saving:.1f}%, "
                          f"auc_diff={auc_diff:.4f}, "
                          f"throughput_pval={pval:.4f}, "
                          f"95% CI throughput [{ci_speedup_low:.2f}, {ci_speedup_high:.2f}]")
    
    # Generate plots
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=summary_df, x='batch_size', y='median_throughput', 
                hue='pruning_method', style='sparsity', markers=True, dashes=False)
    plt.title(f'Throughput vs Batch Size - {dataset}')
    plt.ylabel('Throughput (img/s)')
    plt.xlabel('Batch Size')
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_vs_bs.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_df, x='batch_size', y='median_energy_kWh_total', 
               hue='pruning_method')
    plt.title(f'Total Energy Consumption (kWh) - {dataset}')
    plt.ylabel('Energy (kWh)')
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_total_bar.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_df, x='batch_size', y='median_energy_kWh_per_batch', 
               hue='pruning_method')
    plt.title(f'Energy per Batch (kWh) - {dataset}')
    plt.ylabel('Energy per Batch (kWh)')
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_per_batch_bar.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_df, x='batch_size', y='median_energy_kWh_per_image', 
               hue='pruning_method')
    plt.title(f'Energy per Image (kWh) - {dataset}')
    plt.ylabel('Energy per Image (kWh)')
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_per_image_bar.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=summary_df, x='batch_size', y='median_auc', 
                hue='pruning_method', style='sparsity', markers=True, dashes=False)
    plt.title(f'AUC vs Batch Size - {dataset}')
    plt.ylabel('Median AUC')
    plt.xlabel('Batch Size')
    plt.tight_layout()
    plt.savefig(output_dir / 'auc_vs_bs.png', dpi=300)
    plt.close()
    
    print(f"\nAnalysis complete for {dataset}. Outputs in: {output_dir}")
    return summary_df


def run_full_analysis(log_base):
    """
    Run comprehensive analysis across all datasets.
    Matches ResNet script analysis structure.
    """
    analysis_dir = log_base / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    all_summaries = []
    for dataset in MATRIX_CONFIG['datasets']:
        results_csv = log_base / f"{dataset}_results.csv"
        if os.path.exists(results_csv):
            print(f"\nAnalyzing {dataset}...")
            temp_output = analysis_dir / dataset
            temp_output.mkdir(parents=True, exist_ok=True)
            summary_df = analyze_results(str(results_csv), str(log_base), str(temp_output), dataset)
            all_summaries.append(summary_df)
    
    if all_summaries:
        global_summary = pd.concat(all_summaries, ignore_index=True)
        global_summary.to_csv(analysis_dir / 'global_summary.csv', index=False)
        print(f"\nGlobal summary saved to {analysis_dir / 'global_summary.csv'}")


def main():
    """Main execution function."""
    print(f"\n{'='*80}")
    print("VISION TRANSFORMER MODEL BENCHMARKING")
    print(f"{'='*80}\n")
    
    print("Configuration:")
    print(f"  Datasets: {MATRIX_CONFIG['datasets']}")
    print(f"  Batch sizes: {MATRIX_CONFIG['batch_sizes']}")
    print(f"  Precisions: {MATRIX_CONFIG['precisions']}")
    print(f"  Passes per test: {MATRIX_CONFIG['num_passes']}")
    print(f"  Warmup batches: {MATRIX_CONFIG['warmup_batches']}")
    print(f"  Repeats: {MATRIX_CONFIG['repeats']}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  CodeCarbon available: {CODECARBON_AVAILABLE}")
    
    # Verify paths exist
    if not os.path.exists(BASELINE_DIR):
        print(f"\n✗ Error: Baseline directory not found: {BASELINE_DIR}")
        return
    
    if not os.path.exists(DATASETS_DIR):
        print(f"\n✗ Error: Datasets directory not found: {DATASETS_DIR}")
        return
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n✗ Warning: CUDA not available. Benchmarks may be slower and less accurate.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  PyTorch version: {torch.__version__}")
    
    # Discover models
    print("\nDiscovering models...")
    MATRIX_CONFIG["models"] = discover_models()
    
    if not MATRIX_CONFIG["models"]:
        print("\n✗ No models found. Check your model directories.")
        return
    
    print(f"\nModel discovery results:")
    for dataset_info in MATRIX_CONFIG["models"]:
        print(f"  {dataset_info['dataset']}: {len(dataset_info['models'])} models found")
        for model in dataset_info['models']:
            print(f"    - {model['pruning_method']} ({model['sparsity']}, "
                  f"stored as {model['stored_precision']})")
    
    # Run benchmarks
    print("\nStarting full experiment...")
    run_matrix(MATRIX_CONFIG)
    
    # Run analysis
    print("\n\nAll runs complete. Running analysis...")
    run_full_analysis(Path(MATRIX_CONFIG['log_dir']))
    
    print(f"\n{'='*80}")
    print("BENCHMARKING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nCheck the {OUTPUT_DIR} directory for outputs:")
    print("  - {dataset}_results.csv (raw results for each dataset)")
    print("  - analysis/{dataset}/summary.csv (statistical summaries)")
    print("  - analysis/global_summary.csv (overall summary)")
    print("  - analysis/{dataset}/*.png (visualization plots)")


if __name__ == "__main__":
    main()