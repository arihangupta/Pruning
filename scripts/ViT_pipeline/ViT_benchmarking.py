#!/usr/bin/env python3
"""
Comprehensive benchmarking script for Vision Transformer models.
Benchmarks baseline, quantized (AMP), and knowledge distillation models.
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
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score
import timm
from tqdm import tqdm

# CodeCarbon for energy tracking
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except Exception:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    print("Warning: CodeCarbon not available. Energy metrics will not be collected.")

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

# Benchmarking configuration
BENCHMARK_CONFIG = {
    "datasets": ["bloodmnist", "dermamnist", "pathmnist"],
    "batch_sizes": [8, 32, 64],
    "precisions": ["fp32", "amp"],
    "num_passes": 3,  # Number of complete passes through test set
    "warmup_batches": 50,
    "repeats": 3,  # Number of times to repeat each configuration
    "num_workers": 4,
    "pin_memory": True,
}


@dataclass
class BenchmarkConfig:
    model_name: str
    model_type: str  # 'baseline', 'quantized', 'kd'
    model_path: str
    dataset: str
    batch_size: int
    precision: str
    num_passes: int
    warmup_batches: int
    device: str
    seed: int
    repeat: int
    teacher_model: str = None  # For KD models
    student_model: str = None  # For KD models


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def load_test_dataset(dataset_name: str, batch_size: int):
    """Load test dataset."""
    npz_path = DATASETS[dataset_name]
    data = np.load(npz_path, mmap_mode="r")
    X_test = data["test_images"]
    y_test = data["test_labels"].flatten()
    
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE)
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=BENCHMARK_CONFIG["num_workers"],
        pin_memory=BENCHMARK_CONFIG["pin_memory"]
    )
    
    return test_loader, len(test_ds)


def discover_models():
    """
    Discover all available models: baselines, quantized, and KD models.
    Returns a structured dictionary of all models to benchmark.
    """
    models = {
        'baseline': {},
        'quantized': {},
        'kd': {}
    }
    
    # Discover baseline models
    print("\n" + "="*80)
    print("DISCOVERING MODELS")
    print("="*80)
    
    for dataset in BENCHMARK_CONFIG["datasets"]:
        models['baseline'][dataset] = []
        models['quantized'][dataset] = []
        models['kd'][dataset] = []
        
        # Baseline models
        for model_size in ['tiny', 'small', 'base']:
            model_name = f"vit_{model_size}_patch16_224"
            baseline_path = os.path.join(BASELINE_DIR, f"{model_name}_{dataset}_pretrained.pth")
            if os.path.exists(baseline_path):
                models['baseline'][dataset].append({
                    'model_name': model_name,
                    'model_path': baseline_path,
                    'size': model_size
                })
                print(f"✓ Found baseline: {model_name} for {dataset}")
        
        # Quantized models
        quant_dir = os.path.join(MODELS_DIR, "quantized_amp")
        if os.path.exists(quant_dir):
            for model_size in ['tiny', 'small', 'base']:
                model_name = f"vit_{model_size}_patch16_224"
                quant_path = os.path.join(quant_dir, f"{model_name}_{dataset}_amp.pth")
                if os.path.exists(quant_path):
                    models['quantized'][dataset].append({
                        'model_name': f"{model_name}_amp",
                        'base_model_name': model_name,
                        'model_path': quant_path,
                        'size': model_size
                    })
                    print(f"✓ Found quantized: {model_name}_amp for {dataset}")
        
        # Knowledge distillation models
        kd_dir = os.path.join(MODELS_DIR, "knowledge_distillation")
        if os.path.exists(kd_dir):
            kd_pairs = [
                ('base', 'small'),
                ('base', 'tiny'),
                ('small', 'tiny')
            ]
            for teacher_size, student_size in kd_pairs:
                teacher_name = f"vit_{teacher_size}_patch16_224"
                student_name = f"vit_{student_size}_patch16_224"
                kd_path = os.path.join(kd_dir, f"{student_name}_{dataset}_kd_from_{teacher_name}.pth")
                if os.path.exists(kd_path):
                    models['kd'][dataset].append({
                        'model_name': f"{student_name}_kd_from_{teacher_name}",
                        'student_model': student_name,
                        'teacher_model': teacher_name,
                        'model_path': kd_path,
                        'size': student_size
                    })
                    print(f"✓ Found KD model: {student_name} ← {teacher_name} for {dataset}")
    
    return models


def load_model(model_path: str, model_name: str, num_classes: int, precision: str, device: str):
    """Load a model from checkpoint."""
    # Extract base model architecture name
    if 'kd_from' in model_name:
        # For KD models, extract student model name
        base_name = model_name.split('_kd_from')[0]
    elif model_name.endswith('_amp'):
        # For quantized models
        base_name = model_name.replace('_amp', '')
    else:
        # For baseline models
        base_name = model_name
    
    # Create model
    model = timm.create_model(base_name, pretrained=False, num_classes=num_classes)
    
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
    if precision == 'amp':
        model = model.half()
    else:
        model = model.float()
    
    model = model.to(device)
    model.eval()
    
    return model


def warmup_model(model, test_loader, num_batches, device, precision):
    """Warmup the model to ensure stable timing measurements."""
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            if precision == 'amp':
                with torch.amp.autocast('cuda'):
                    _ = model(inputs)
            else:
                _ = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()


def benchmark_model(config: BenchmarkConfig):
    """
    Benchmark a single model configuration.
    Returns detailed metrics including latency, throughput, memory, AUC, and energy.
    """
    set_seed(config.seed)
    device = torch.device(config.device)
    
    # Load dataset
    test_loader, dataset_size = load_test_dataset(config.dataset, config.batch_size)
    num_classes = DATASET_NUM_CLASSES[config.dataset]
    
    # Load model
    try:
        model = load_model(
            config.model_path,
            config.model_name,
            num_classes,
            config.precision,
            config.device
        )
    except Exception as e:
        print(f"✗ Error loading model {config.model_name}: {e}")
        return None
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Model size
    model_size_mb = os.path.getsize(config.model_path) / (1024 * 1024)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {config.model_name}")
    print(f"Dataset: {config.dataset}, Batch Size: {config.batch_size}, Precision: {config.precision}")
    print(f"Parameters: {total_params:,} ({model_size_mb:.2f} MB)")
    print(f"{'='*80}")
    
    # Warmup
    print(f"Warming up ({config.warmup_batches} batches)...")
    warmup_model(model, test_loader, config.warmup_batches, device, config.precision)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Start energy tracking
    tracker = None
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name=f"{config.model_name}_{config.dataset}_b{config.batch_size}",
            log_level='error',
            save_to_file=False
        )
        tracker.start()
    
    # Benchmark inference
    print(f"Running benchmark ({config.num_passes} passes)...")
    batch_times = []
    all_probs = []
    all_labels = []
    images_processed = 0
    
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for pass_idx in range(config.num_passes):
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                batch_start = time.time()
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Inference
                if config.precision == 'amp':
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                batch_end = time.time()
                batch_times.append((batch_end - batch_start) * 1000)  # ms
                
                # Collect predictions for AUC with proper probability normalization
                # Convert to float32 if needed for numerical stability
                if outputs.dtype == torch.float16:
                    outputs = outputs.float()
                
                # Clip extreme values to prevent overflow
                outputs = torch.clamp(outputs, min=-100, max=100)
                
                # Calculate softmax probabilities
                probs = torch.softmax(outputs, dim=1)
                probs_np = probs.cpu().numpy()
                
                # Normalize to ensure probabilities sum to 1 (handle numerical precision issues)
                probs_sum = probs_np.sum(axis=1, keepdims=True)
                probs_np = probs_np / probs_sum
                
                # Verify no NaN or Inf values
                if np.any(np.isnan(probs_np)) or np.any(np.isinf(probs_np)):
                    print(f"Warning: Invalid probabilities detected in batch {batch_idx}, skipping...")
                    continue
                
                all_probs.append(probs_np)
                all_labels.append(labels.cpu().numpy())
                
                images_processed += len(inputs)
    
    total_time = time.time() - start_time
    
    # Stop energy tracking
    energy_metrics = {}
    if tracker is not None:
        try:
            emissions = tracker.stop()
            energy_kwh = tracker._total_energy.kWh if hasattr(tracker, '_total_energy') else 0.0
            energy_metrics = {
                'energy_kwh_total': energy_kwh,
                'emissions_kg': emissions if emissions else 0.0,
                'energy_kwh_per_image': energy_kwh / images_processed if images_processed > 0 else 0.0,
                'energy_kwh_per_batch': energy_kwh / len(batch_times) if len(batch_times) > 0 else 0.0
            }
        except Exception as e:
            print(f"Warning: Error stopping energy tracker: {e}")
            energy_metrics = {
                'energy_kwh_total': float('nan'),
                'emissions_kg': float('nan'),
                'energy_kwh_per_image': float('nan'),
                'energy_kwh_per_batch': float('nan')
            }
    else:
        energy_metrics = {
            'energy_kwh_total': float('nan'),
            'emissions_kg': float('nan'),
            'energy_kwh_per_image': float('nan'),
            'energy_kwh_per_batch': float('nan')
        }
    
    # Calculate metrics
    if not all_probs:
        print("Error: No valid predictions collected!")
        return None
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Verify probabilities are valid
    print(f"Probability check: min={all_probs.min():.6f}, max={all_probs.max():.6f}")
    print(f"Probability sums: min={all_probs.sum(axis=1).min():.6f}, max={all_probs.sum(axis=1).max():.6f}")
    
    # Final normalization to ensure exact sum to 1.0
    all_probs = all_probs / all_probs.sum(axis=1, keepdims=True)
    
    # AUC - use explicit label handling for multiclass
    try:
        # For multiclass, we need one-hot encoded labels or probability scores
        auc = roc_auc_score(
            all_labels, 
            all_probs, 
            multi_class='ovr',
            average='weighted',
            labels=np.arange(num_classes)
        )
    except Exception as e:
        print(f"Warning: Could not calculate weighted AUC: {e}")
        try:
            # Try macro average
            auc = roc_auc_score(
                all_labels, 
                all_probs, 
                multi_class='ovr',
                average='macro',
                labels=np.arange(num_classes)
            )
        except Exception as e2:
            print(f"Warning: Could not calculate macro AUC: {e2}")
            auc = float('nan')
    
    # Accuracy
    preds = np.argmax(all_probs, axis=1)
    accuracy = accuracy_score(all_labels, preds)
    
    # Timing metrics
    batch_times_array = np.array(batch_times)
    median_batch_time = np.median(batch_times_array)
    mean_batch_time = np.mean(batch_times_array)
    std_batch_time = np.std(batch_times_array)
    p50_batch_time = np.percentile(batch_times_array, 50)
    p90_batch_time = np.percentile(batch_times_array, 90)
    p95_batch_time = np.percentile(batch_times_array, 95)
    p99_batch_time = np.percentile(batch_times_array, 99)
    
    # Throughput
    throughput = images_processed / total_time  # images/sec
    
    # Memory
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == 'cuda' else 0.0
    
    # Compile results
    results = {
        'model_name': config.model_name,
        'model_type': config.model_type,
        'dataset': config.dataset,
        'batch_size': config.batch_size,
        'precision': config.precision,
        'repeat': config.repeat,
        'num_passes': config.num_passes,
        
        # Model info
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        
        # Accuracy metrics
        'accuracy': accuracy,
        'auc': auc,
        
        # Timing metrics
        'total_time_s': total_time,
        'images_processed': images_processed,
        'throughput_img_per_s': throughput,
        'median_batch_time_ms': median_batch_time,
        'mean_batch_time_ms': mean_batch_time,
        'std_batch_time_ms': std_batch_time,
        'p50_batch_time_ms': p50_batch_time,
        'p90_batch_time_ms': p90_batch_time,
        'p95_batch_time_ms': p95_batch_time,
        'p99_batch_time_ms': p99_batch_time,
        'latency_per_image_ms': (total_time * 1000) / images_processed,
        
        # Memory metrics
        'peak_memory_mb': peak_memory_mb,
        
        # Energy metrics
        **energy_metrics,
        
        # KD specific
        'teacher_model': config.teacher_model,
        'student_model': config.student_model,
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Throughput: {throughput:.2f} img/s")
    print(f"Median batch time: {median_batch_time:.2f} ms")
    print(f"Latency per image: {results['latency_per_image_ms']:.2f} ms")
    print(f"Peak memory: {peak_memory_mb:.2f} MB")
    if not math.isnan(energy_metrics['energy_kwh_total']):
        print(f"Energy (total): {energy_metrics['energy_kwh_total']:.6f} kWh")
        print(f"Energy (per image): {energy_metrics['energy_kwh_per_image']:.9f} kWh")
        print(f"CO2 emissions: {energy_metrics['emissions_kg']:.6f} kg")
    print(f"{'='*60}\n")
    
    # Cleanup
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return results


def run_full_benchmark():
    """Run complete benchmarking suite."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Discover all models
    models = discover_models()
    
    # Create benchmark configurations
    all_configs = []
    
    for dataset in BENCHMARK_CONFIG["datasets"]:
        # Baseline models
        for model_info in models['baseline'][dataset]:
            for batch_size in BENCHMARK_CONFIG["batch_sizes"]:
                for precision in BENCHMARK_CONFIG["precisions"]:
                    for repeat in range(BENCHMARK_CONFIG["repeats"]):
                        config = BenchmarkConfig(
                            model_name=model_info['model_name'],
                            model_type='baseline',
                            model_path=model_info['model_path'],
                            dataset=dataset,
                            batch_size=batch_size,
                            precision=precision,
                            num_passes=BENCHMARK_CONFIG["num_passes"],
                            warmup_batches=BENCHMARK_CONFIG["warmup_batches"],
                            device='cuda:0',
                            seed=SEED + repeat,
                            repeat=repeat
                        )
                        all_configs.append(config)
        
        # Quantized models (only AMP precision makes sense)
        for model_info in models['quantized'][dataset]:
            for batch_size in BENCHMARK_CONFIG["batch_sizes"]:
                for repeat in range(BENCHMARK_CONFIG["repeats"]):
                    config = BenchmarkConfig(
                        model_name=model_info['model_name'],
                        model_type='quantized',
                        model_path=model_info['model_path'],
                        dataset=dataset,
                        batch_size=batch_size,
                        precision='amp',  # Quantized models should use AMP
                        num_passes=BENCHMARK_CONFIG["num_passes"],
                        warmup_batches=BENCHMARK_CONFIG["warmup_batches"],
                        device='cuda:0',
                        seed=SEED + repeat,
                        repeat=repeat
                    )
                    all_configs.append(config)
        
        # KD models
        for model_info in models['kd'][dataset]:
            for batch_size in BENCHMARK_CONFIG["batch_sizes"]:
                for precision in BENCHMARK_CONFIG["precisions"]:
                    for repeat in range(BENCHMARK_CONFIG["repeats"]):
                        config = BenchmarkConfig(
                            model_name=model_info['model_name'],
                            model_type='kd',
                            model_path=model_info['model_path'],
                            dataset=dataset,
                            batch_size=batch_size,
                            precision=precision,
                            num_passes=BENCHMARK_CONFIG["num_passes"],
                            warmup_batches=BENCHMARK_CONFIG["warmup_batches"],
                            device='cuda:0',
                            seed=SEED + repeat,
                            repeat=repeat,
                            teacher_model=model_info['teacher_model'],
                            student_model=model_info['student_model']
                        )
                        all_configs.append(config)
    
    # Shuffle configurations to avoid systematic biases
    random.shuffle(all_configs)
    
    print(f"\n{'='*80}")
    print(f"TOTAL BENCHMARK CONFIGURATIONS: {len(all_configs)}")
    print(f"{'='*80}\n")
    
    # Run benchmarks
    all_results = []
    for i, config in enumerate(all_configs):
        print(f"\n[{i+1}/{len(all_configs)}] Starting benchmark...")
        try:
            result = benchmark_model(config)
            if result is not None:
                all_results.append(result)
                
                # Save intermediate results
                results_df = pd.DataFrame(all_results)
                results_df.to_csv(os.path.join(OUTPUT_DIR, 'benchmark_results.csv'), index=False)
        except Exception as e:
            print(f"✗ Error benchmarking {config.model_name}: {e}")
            import traceback
            traceback.print_exc()
        
        # Cool down between runs
        time.sleep(10)
    
    return all_results


def analyze_results(results_df: pd.DataFrame):
    """Analyze benchmark results and generate comparison tables."""
    output_dir = Path(OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("ANALYZING RESULTS")
    print(f"{'='*80}\n")
    
    # Group by dataset
    for dataset in results_df['dataset'].unique():
        dataset_df = results_df[results_df['dataset'] == dataset]
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS FOR {dataset.upper()}")
        print(f"{'='*80}\n")
        
        # Summary statistics
        summary_groups = dataset_df.groupby(['model_name', 'model_type', 'batch_size', 'precision'])
        
        summary_stats = []
        for (model_name, model_type, bs, prec), group in summary_groups:
            stats_dict = {
                'dataset': dataset,
                'model_name': model_name,
                'model_type': model_type,
                'batch_size': bs,
                'precision': prec,
                'n_runs': len(group),
                
                # Accuracy
                'mean_accuracy': group['accuracy'].mean(),
                'std_accuracy': group['accuracy'].std(),
                'mean_auc': group['auc'].mean(),
                'std_auc': group['auc'].std(),
                
                # Throughput
                'mean_throughput': group['throughput_img_per_s'].mean(),
                'std_throughput': group['throughput_img_per_s'].std(),
                'median_throughput': group['throughput_img_per_s'].median(),
                
                # Latency
                'mean_latency_ms': group['latency_per_image_ms'].mean(),
                'std_latency_ms': group['latency_per_image_ms'].std(),
                'median_latency_ms': group['latency_per_image_ms'].median(),
                
                # Memory
                'mean_peak_memory_mb': group['peak_memory_mb'].mean(),
                'std_peak_memory_mb': group['peak_memory_mb'].std(),
                
                # Energy
                'mean_energy_kwh': group['energy_kwh_total'].mean(),
                'mean_energy_per_img': group['energy_kwh_per_image'].mean(),
                'mean_emissions_kg': group['emissions_kg'].mean(),
                
                # Model info
                'model_size_mb': group['model_size_mb'].iloc[0],
                'total_params': group['total_params'].iloc[0],
            }
            summary_stats.append(stats_dict)
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(output_dir / f'{dataset}_summary.csv', index=False)
        
        # Comparisons with baselines
        print(f"\nCOMPARISONS WITH BASELINE MODELS:\n")
        
        for bs in summary_df['batch_size'].unique():
            for prec in summary_df['precision'].unique():
                # Get baseline models for this configuration
                baseline_models = summary_df[
                    (summary_df['model_type'] == 'baseline') &
                    (summary_df['batch_size'] == bs) &
                    (summary_df['precision'] == prec)
                ]
                
                # Compare quantized models
                quant_models = summary_df[
                    (summary_df['model_type'] == 'quantized') &
                    (summary_df['batch_size'] == bs)
                ]
                
                for _, quant in quant_models.iterrows():
                    base_name = quant['model_name'].replace('_amp', '')
                    baseline = baseline_models[baseline_models['model_name'] == base_name]
                    
                    if len(baseline) > 0:
                        baseline = baseline.iloc[0]
                        speedup = quant['mean_throughput'] / baseline['mean_throughput']
                        accuracy_drop = (baseline['mean_accuracy'] - quant['mean_accuracy']) * 100
                        auc_drop = (baseline['mean_auc'] - quant['mean_auc']) * 100
                        size_reduction = (1 - quant['model_size_mb'] / baseline['model_size_mb']) * 100
                        
                        print(f"Quantized {base_name} (bs={bs}):")
                        print(f"  Speedup: {speedup:.2f}x")
                        print(f"  Accuracy drop: {accuracy_drop:.2f}%")
                        print(f"  AUC drop: {auc_drop:.2f}%")
                        print(f"  Size reduction: {size_reduction:.2f}%")
                        print()
                
                # Compare KD models
                kd_models = summary_df[
                    (summary_df['model_type'] == 'kd') &
                    (summary_df['batch_size'] == bs) &
                    (summary_df['precision'] == prec)
                ]
                
                for _, kd in kd_models.iterrows():
                    # Compare with teacher baseline
                    teacher_name = dataset_df[dataset_df['model_name'] == kd['model_name']]['teacher_model'].iloc[0]
                    if teacher_name:
                        teacher_baseline = baseline_models[baseline_models['model_name'] == teacher_name]
                        
                        if len(teacher_baseline) > 0:
                            teacher = teacher_baseline.iloc[0]
                            speedup = kd['mean_throughput'] / teacher['mean_throughput']
                            accuracy_drop = (teacher['mean_accuracy'] - kd['mean_accuracy']) * 100
                            auc_drop = (teacher['mean_auc'] - kd['mean_auc']) * 100
                            param_reduction = (1 - kd['total_params'] / teacher['total_params']) * 100
                            
                            print(f"KD {kd['model_name']} vs teacher {teacher_name} (bs={bs}, prec={prec}):")
                            print(f"  Speedup: {speedup:.2f}x")
                            print(f"  Accuracy drop: {accuracy_drop:.2f}%")
                            print(f"  AUC drop: {auc_drop:.2f}%")
                            print(f"  Param reduction: {param_reduction:.2f}%")
                            print()
        
        # Generate plots
        plt.figure(figsize=(12, 8))
        sns.barplot(data=summary_df, x='model_name', y='mean_throughput', hue='batch_size')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Throughput Comparison - {dataset}')
        plt.ylabel('Throughput (img/s)')
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset}_throughput.png', dpi=300)
        plt.close()
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=summary_df, x='model_name', y='mean_auc', hue='batch_size')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'AUC Comparison - {dataset}')
        plt.ylabel('AUC')
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset}_auc.png', dpi=300)
        plt.close()
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=summary_df, x='model_name', y='mean_latency_ms', hue='batch_size')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Latency Comparison - {dataset}')
        plt.ylabel('Latency per Image (ms)')
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset}_latency.png', dpi=300)
        plt.close()
        
        # Energy comparison (if available)
        if not summary_df['mean_energy_per_img'].isna().all():
            plt.figure(figsize=(12, 8))
            sns.barplot(data=summary_df, x='model_name', y='mean_energy_per_img', hue='batch_size')
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Energy per Image Comparison - {dataset}')
            plt.ylabel('Energy (kWh)')
            plt.tight_layout()
            plt.savefig(output_dir / f'{dataset}_energy.png', dpi=300)
            plt.close()
        
        # Memory comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(data=summary_df, x='model_name', y='mean_peak_memory_mb', hue='batch_size')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Peak Memory Usage - {dataset}')
        plt.ylabel('Memory (MB)')
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset}_memory.png', dpi=300)
        plt.close()
        
        # Efficiency plot: throughput vs model size
        plt.figure(figsize=(10, 8))
        for model_type in summary_df['model_type'].unique():
            type_df = summary_df[summary_df['model_type'] == model_type]
            plt.scatter(type_df['model_size_mb'], type_df['mean_throughput'], 
                       label=model_type, s=100, alpha=0.7)
        plt.xlabel('Model Size (MB)')
        plt.ylabel('Throughput (img/s)')
        plt.title(f'Efficiency: Throughput vs Model Size - {dataset}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset}_efficiency.png', dpi=300)
        plt.close()
        
        # Accuracy-throughput tradeoff
        plt.figure(figsize=(10, 8))
        for model_type in summary_df['model_type'].unique():
            type_df = summary_df[summary_df['model_type'] == model_type]
            plt.scatter(type_df['mean_throughput'], type_df['mean_auc'], 
                       label=model_type, s=100, alpha=0.7)
        plt.xlabel('Throughput (img/s)')
        plt.ylabel('AUC')
        plt.title(f'Accuracy-Throughput Tradeoff - {dataset}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset}_tradeoff.png', dpi=300)
        plt.close()
    
    # Global summary across all datasets
    global_summary = results_df.groupby(['model_name', 'model_type']).agg({
        'accuracy': ['mean', 'std'],
        'auc': ['mean', 'std'],
        'throughput_img_per_s': ['mean', 'std', 'median'],
        'latency_per_image_ms': ['mean', 'std', 'median'],
        'peak_memory_mb': ['mean', 'std'],
        'energy_kwh_per_image': ['mean', 'std'],
        'model_size_mb': 'first',
        'total_params': 'first'
    }).reset_index()
    
    global_summary.to_csv(output_dir / 'global_summary.csv', index=False)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - benchmark_results.csv (raw results)")
    print("  - {dataset}_summary.csv (per-dataset summaries)")
    print("  - global_summary.csv (overall summary)")
    print("  - *.png (visualization plots)")


def generate_latex_table(results_df: pd.DataFrame, output_dir: Path):
    """Generate LaTeX tables for paper inclusion."""
    
    for dataset in results_df['dataset'].unique():
        dataset_df = results_df[results_df['dataset'] == dataset]
        
        # Create comparison table
        summary = dataset_df.groupby(['model_name', 'model_type', 'batch_size']).agg({
            'accuracy': 'mean',
            'auc': 'mean',
            'throughput_img_per_s': 'mean',
            'latency_per_image_ms': 'mean',
            'peak_memory_mb': 'mean',
            'model_size_mb': 'first',
            'total_params': 'first'
        }).reset_index()
        
        # Format for LaTeX
        latex_rows = []
        for _, row in summary.iterrows():
            latex_row = (
                f"{row['model_name']} & "
                f"{row['model_type']} & "
                f"{row['batch_size']} & "
                f"{row['accuracy']:.4f} & "
                f"{row['auc']:.4f} & "
                f"{row['throughput_img_per_s']:.2f} & "
                f"{row['latency_per_image_ms']:.2f} & "
                f"{row['peak_memory_mb']:.2f} & "
                f"{row['model_size_mb']:.2f} & "
                f"{row['total_params']:,} \\\\"
            )
            latex_rows.append(latex_row)
        
        latex_table = "\\begin{table}[h]\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{Benchmark Results for " + dataset + "}\n"
        latex_table += "\\begin{tabular}{llrrrrrrrr}\n"
        latex_table += "\\toprule\n"
        latex_table += "Model & Type & BS & Acc & AUC & Throughput & Latency & Memory & Size & Params \\\\\n"
        latex_table += "      &      &    &     &     & (img/s) & (ms) & (MB) & (MB) &  \\\\\n"
        latex_table += "\\midrule\n"
        latex_table += "\n".join(latex_rows) + "\n"
        latex_table += "\\bottomrule\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\end{table}\n"
        
        with open(output_dir / f'{dataset}_table.tex', 'w') as f:
            f.write(latex_table)
    
    print(f"\n✓ LaTeX tables generated")


def main():
    """Main execution function."""
    print(f"\n{'='*80}")
    print("VISION TRANSFORMER MODEL BENCHMARKING")
    print(f"{'='*80}\n")
    
    print("Configuration:")
    print(f"  Datasets: {BENCHMARK_CONFIG['datasets']}")
    print(f"  Batch sizes: {BENCHMARK_CONFIG['batch_sizes']}")
    print(f"  Precisions: {BENCHMARK_CONFIG['precisions']}")
    print(f"  Passes per test: {BENCHMARK_CONFIG['num_passes']}")
    print(f"  Warmup batches: {BENCHMARK_CONFIG['warmup_batches']}")
    print(f"  Repeats: {BENCHMARK_CONFIG['repeats']}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  CodeCarbon available: {CODECARBON_AVAILABLE}")
    
    # Verify paths exist
    if not os.path.exists(BASELINE_DIR):
        print(f"\n✗ Error: Baseline directory not found: {BASELINE_DIR}")
        return
    
    if not os.path.exists(MODELS_DIR):
        print(f"\n✗ Error: Models directory not found: {MODELS_DIR}")
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
    
    # Run benchmarks
    print("\nStarting benchmarks...")
    all_results = run_full_benchmark()
    
    if not all_results:
        print("\n✗ No results collected. Check for errors above.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save raw results
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'benchmark_results_final.csv'), index=False)
    print(f"\n✓ Raw results saved to {OUTPUT_DIR}/benchmark_results_final.csv")
    
    # Analyze results
    analyze_results(results_df)
    
    # Generate LaTeX tables
    generate_latex_table(results_df, Path(OUTPUT_DIR))
    
    print(f"\n{'='*80}")
    print("BENCHMARKING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nTotal configurations tested: {len(all_results)}")
    print(f"Results directory: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Review the summary CSV files")
    print("  2. Check the generated plots")
    print("  3. Use LaTeX tables for publication")


if __name__ == "__main__":
    main()