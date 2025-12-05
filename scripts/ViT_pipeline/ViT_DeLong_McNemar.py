#!/usr/bin/env python3
"""
Statistical comparison of Vision Transformer models using:
- DeLong's test for AUC comparison
- McNemar's test for accuracy comparison

Compares quantized and knowledge distillation models against baseline ViTs.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import stats
import glob
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import timm

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants (from ViT benchmarking script)
IMG_SIZE = 224
SEED = 42
BASELINE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/new_baseline"
PRUNED_MODELS_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/pruned_models"
DATASETS_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"

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

TARGET_DATASETS = ["bloodmnist", "dermamnist", "pathmnist"]

# ============================================================================
# DeLong's Test Implementation
# ============================================================================

def compute_midrank(x):
    """Compute midranks for tied values."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    Fast implementation of DeLong's method for computing AUC variance.
    
    Args:
        predictions_sorted_transposed: 2D array (n_classifiers, n_samples)
        label_1_count: number of positive samples
    
    Returns:
        aucs: array of AUC values
        covariance_matrix: covariance matrix for AUCs
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    
    return aucs, delongcov

def delong_roc_variance(ground_truth, predictions):
    """
    Compute DeLong variance for a single classifier.
    
    Args:
        ground_truth: array of true labels (0/1)
        predictions: array of predicted probabilities
    
    Returns:
        auc: AUC value
        variance: variance of AUC
    """
    order = np.argsort(ground_truth)
    label_1_count = int(ground_truth.sum())
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    
    return aucs[0], delongcov

def delong_test(ground_truth, predictions_one, predictions_two):
    """
    Perform DeLong's test to compare two ROC curves.
    
    Args:
        ground_truth: array of true labels
        predictions_one: predictions from model 1
        predictions_two: predictions from model 2
    
    Returns:
        z_score: z-score of the test
        p_value: two-tailed p-value
    """
    order = np.argsort(ground_truth)
    label_1_count = int(ground_truth.sum())
    
    predictions_sorted_transposed = np.vstack((
        predictions_one,
        predictions_two
    ))[:, order]
    
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    
    if delongcov.ndim == 0:
        variance = delongcov
    else:
        variance = delongcov[0, 0] - 2 * delongcov[0, 1] + delongcov[1, 1]
    
    if variance < 0:
        variance = 0
    
    z_score = (aucs[0] - aucs[1]) / np.sqrt(variance) if variance > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return z_score, p_value, aucs[0], aucs[1]

def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Perform McNemar's test to compare two classifiers.
    
    Args:
        y_true: true labels
        y_pred1: predictions from model 1
        y_pred2: predictions from model 2
    
    Returns:
        statistic: McNemar statistic
        p_value: p-value
    """
    # Create contingency table
    correct_1 = (y_pred1 == y_true)
    correct_2 = (y_pred2 == y_true)
    
    # McNemar table:
    # n01: model 1 wrong, model 2 correct
    # n10: model 1 correct, model 2 wrong
    n01 = np.sum(~correct_1 & correct_2)
    n10 = np.sum(correct_1 & ~correct_2)
    
    # McNemar statistic with continuity correction
    if (n01 + n10) == 0:
        return 0.0, 1.0
    
    statistic = (abs(n01 - n10) - 1)**2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return statistic, p_value

# ============================================================================
# Dataset Loading
# ============================================================================

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

def make_test_loader(npz_path, batch_size=32):
    """Load test dataset."""
    data = np.load(npz_path, mmap_mode="r")
    X_test = data["test_images"]
    y_test = data["test_labels"].flatten()
    
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE)
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader

# ============================================================================
# Model Discovery and Loading
# ============================================================================

def parse_model_name(model_path, dataset):
    """
    Parse model filename to extract metadata.
    Handles patterns from pruned_models directory and baseline directory.
    """
    basename = os.path.basename(model_path)
    logger.debug(f"Parsing model file: {basename} for dataset: {dataset}")
    
    # Handle baseline models from baseline directory
    if "_pretrained.pth" in basename:
        model_name = basename.replace(f"_{dataset}_pretrained.pth", "")
        return {
            "model_name": model_name,
            "full_name": f"{model_name}_baseline",
            "pruning_method": "baseline",
            "sparsity": "0%",
            "stored_precision": "fp32",
            "teacher_model": None,
            "student_model": None
        }
    
    # Handle quantized models from pruned_models directory
    # Pattern: quantization_vit_{size}_patch16_224_{dataset}_final.pth
    if basename.startswith("quantization_") and "_final.pth" in basename:
        parts = basename.replace("quantization_", "").replace(f"_{dataset}_final.pth", "")
        model_name = parts  # e.g., vit_tiny_patch16_224
        
        return {
            "model_name": model_name,
            "full_name": f"quantization_{model_name}",
            "pruning_method": "quantization",
            "sparsity": "50%",
            "stored_precision": "amp",
            "teacher_model": None,
            "student_model": None
        }
    
    # Handle knowledge distillation models from pruned_models directory
    # Pattern: kd_vit_{teacher_size}_patch16_224_to_vit_{student_size}_patch16_224_{dataset}_final.pth
    if basename.startswith("kd_") and "_final.pth" in basename:
        middle = basename.replace("kd_", "").replace(f"_{dataset}_final.pth", "")
        
        if "_to_" in middle:
            parts = middle.split("_to_")
            teacher_name = parts[0]  # e.g., vit_base_patch16_224
            student_name = parts[1]  # e.g., vit_small_patch16_224
            
            # Estimate sparsity based on student/teacher size
            sparsity_map = {
                ("base", "small"): "33%",
                ("base", "tiny"): "67%",
                ("small", "tiny"): "50%"
            }
            
            teacher_size = "tiny" if "tiny" in teacher_name else ("small" if "small" in teacher_name else "base")
            student_size = "tiny" if "tiny" in student_name else ("small" if "small" in student_name else "base")
            sparsity = sparsity_map.get((teacher_size, student_size), "unknown")
            
            return {
                "model_name": student_name,
                "full_name": f"kd_{teacher_name}_to_{student_name}",
                "pruning_method": "kd",
                "sparsity": sparsity,
                "stored_precision": "fp32",
                "teacher_model": teacher_name,
                "student_model": student_name
            }
    
    logger.debug(f"Skipping {basename}: unrecognized format")
    return None

def discover_models():
    """
    Discover all available models: baselines, quantized, and KD models.
    """
    models = []
    
    logger.info("\n" + "="*80)
    logger.info("DISCOVERING MODELS")
    logger.info("="*80)
    
    for dataset in TARGET_DATASETS:
        dataset_models = []
        
        # 1. Baseline models (from separate baseline directory)
        for model_size in ['tiny', 'small', 'base']:
            model_name = f"vit_{model_size}_patch16_224"
            baseline_path = os.path.join(BASELINE_DIR, f"{model_name}_{dataset}_pretrained.pth")
            if os.path.exists(baseline_path):
                parsed = parse_model_name(baseline_path, dataset)
                if parsed:
                    parsed["model_path"] = baseline_path
                    dataset_models.append(parsed)
                    logger.info(f"✓ Found baseline: {model_name} for {dataset}")
            else:
                logger.debug(f"Baseline not found: {baseline_path}")
        
        # 2. Quantized and KD models (from pruned_models/{dataset}/ subdirectory)
        dataset_pruned_dir = os.path.join(PRUNED_MODELS_DIR, dataset)
        if os.path.exists(dataset_pruned_dir):
            model_files = glob.glob(os.path.join(dataset_pruned_dir, "*_final.pth"))
            
            for model_path in model_files:
                parsed = parse_model_name(model_path, dataset)
                if parsed:
                    parsed["model_path"] = model_path
                    dataset_models.append(parsed)
                    logger.info(f"✓ Found {parsed['pruning_method']}: {parsed['full_name']} for {dataset}")
                else:
                    logger.debug(f"Skipping unrecognized file: {os.path.basename(model_path)}")
        else:
            logger.warning(f"Pruned models directory for {dataset} does not exist: {dataset_pruned_dir}")
        
        if dataset_models:
            models.append({"dataset": dataset, "models": dataset_models})
            logger.info(f"  Total models for {dataset}: {len(dataset_models)}")
        else:
            logger.warning(f"No models found for {dataset}")
    
    return models

def load_model(model_path, model_name, num_classes, stored_precision='fp32', device='cuda:0'):
    """Load a Vision Transformer model from checkpoint."""
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Create model using timm
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
    is_fp16 = False
    if stored_precision == 'amp':
        model = model.half()
        is_fp16 = True
    else:
        model = model.float()
    
    model = model.to(device)
    model.eval()
    
    return model, is_fp16

def get_predictions(model, dataloader, num_classes, device='cuda:0', is_fp16=False):
    """Get predictions and probabilities from a model."""
    all_probs = []
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            if is_fp16:
                inputs = inputs.half()
            
            outputs = model(inputs)
            
            if outputs.dtype == torch.half:
                outputs = outputs.float()
            
            outputs = torch.clamp(outputs, min=-100, max=100)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Normalize probabilities
    prob_sums = np.sum(all_probs, axis=1, keepdims=True)
    all_probs = all_probs / prob_sums
    
    return all_probs, all_preds, all_labels

# ============================================================================
# Model Comparison
# ============================================================================

def compare_models(dataset_name, baseline_info, pruned_info, dataloader, num_classes, device='cuda:0'):
    """Compare a pruned/compressed model to baseline using DeLong and McNemar tests."""
    logger.info(f"\nComparing {pruned_info['full_name']} to {baseline_info['full_name']} for {dataset_name}")
    
    # Load baseline model
    logger.info("Loading baseline model...")
    baseline_model, baseline_fp16 = load_model(
        baseline_info['model_path'],
        baseline_info['model_name'],
        num_classes,
        baseline_info['stored_precision'],
        device=device
    )
    
    # Get baseline predictions
    logger.info("Getting baseline predictions...")
    baseline_probs, baseline_preds, true_labels = get_predictions(
        baseline_model, 
        dataloader, 
        num_classes, 
        device=device,
        is_fp16=baseline_fp16
    )
    
    # Calculate baseline metrics
    baseline_acc = accuracy_score(true_labels, baseline_preds)
    try:
        baseline_auc = roc_auc_score(
            true_labels, 
            baseline_probs, 
            multi_class="ovr",
            average='weighted',
            labels=list(range(num_classes))
        )
    except:
        baseline_auc = np.nan
    
    logger.info(f"Baseline - Accuracy: {baseline_acc:.4f}, AUC: {baseline_auc:.4f}")
    
    # Clean up baseline model
    del baseline_model
    torch.cuda.empty_cache()
    
    # Load pruned/compressed model
    logger.info(f"Loading compressed model: {pruned_info['full_name']}...")
    pruned_model, pruned_fp16 = load_model(
        pruned_info['model_path'],
        pruned_info['model_name'],
        num_classes,
        pruned_info['stored_precision'],
        device=device
    )
    
    # Get pruned predictions
    logger.info("Getting compressed model predictions...")
    pruned_probs, pruned_preds, _ = get_predictions(
        pruned_model, 
        dataloader, 
        num_classes, 
        device=device,
        is_fp16=pruned_fp16
    )
    
    # Calculate pruned metrics
    pruned_acc = accuracy_score(true_labels, pruned_preds)
    try:
        pruned_auc = roc_auc_score(
            true_labels, 
            pruned_probs, 
            multi_class="ovr",
            average='weighted',
            labels=list(range(num_classes))
        )
    except:
        pruned_auc = np.nan
    
    logger.info(f"Compressed - Accuracy: {pruned_acc:.4f}, AUC: {pruned_auc:.4f}")
    
    # Perform statistical tests
    results = {
        'dataset': dataset_name,
        'baseline_model': baseline_info['model_name'],
        'compressed_model': pruned_info['full_name'],
        'pruning_method': pruned_info['pruning_method'],
        'sparsity': pruned_info['sparsity'],
        'stored_precision': pruned_info['stored_precision'],
        'baseline_accuracy': baseline_acc,
        'compressed_accuracy': pruned_acc,
        'accuracy_difference': pruned_acc - baseline_acc,
        'baseline_auc': baseline_auc,
        'compressed_auc': pruned_auc,
        'auc_difference': pruned_auc - baseline_auc,
    }
    
    # Add teacher/student info for KD models
    if pruned_info['teacher_model']:
        results['teacher_model'] = pruned_info['teacher_model']
        results['student_model'] = pruned_info['student_model']
    else:
        results['teacher_model'] = None
        results['student_model'] = None
    
    # McNemar's test
    logger.info("Performing McNemar's test...")
    try:
        mcnemar_stat, mcnemar_p = mcnemar_test(true_labels, baseline_preds, pruned_preds)
        results['mcnemar_statistic'] = mcnemar_stat
        results['mcnemar_p_value'] = mcnemar_p
        results['mcnemar_significant'] = mcnemar_p < 0.05
        logger.info(f"McNemar's test - Statistic: {mcnemar_stat:.4f}, p-value: {mcnemar_p:.4f}")
    except Exception as e:
        logger.error(f"McNemar's test failed: {e}")
        results['mcnemar_statistic'] = np.nan
        results['mcnemar_p_value'] = np.nan
        results['mcnemar_significant'] = False
    
    # DeLong's test (for each class vs rest, then average)
    logger.info("Performing DeLong's test...")
    delong_z_scores = []
    delong_p_values = []
    
    for class_idx in range(num_classes):
        try:
            # Binary labels: class vs rest
            binary_labels = (true_labels == class_idx).astype(int)
            baseline_class_probs = baseline_probs[:, class_idx]
            pruned_class_probs = pruned_probs[:, class_idx]
            
            # Perform DeLong test
            z_score, p_value, auc1, auc2 = delong_test(
                binary_labels, 
                baseline_class_probs, 
                pruned_class_probs
            )
            
            delong_z_scores.append(z_score)
            delong_p_values.append(p_value)
        except Exception as e:
            logger.warning(f"DeLong's test failed for class {class_idx}: {e}")
            delong_z_scores.append(np.nan)
            delong_p_values.append(np.nan)
    
    # Average results across classes
    results['delong_z_score_mean'] = np.nanmean(delong_z_scores) if delong_z_scores else np.nan
    results['delong_p_value_mean'] = np.nanmean(delong_p_values) if delong_p_values else np.nan
    results['delong_p_value_min'] = np.nanmin(delong_p_values) if delong_p_values else np.nan
    results['delong_significant'] = results['delong_p_value_mean'] < 0.05 if not np.isnan(results['delong_p_value_mean']) else False
    
    logger.info(f"DeLong's test - Mean z-score: {results['delong_z_score_mean']:.4f}, Mean p-value: {results['delong_p_value_mean']:.4f}")
    
    # Clean up
    del pruned_model
    torch.cuda.empty_cache()
    
    return results

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to run comparisons."""
    logger.info("="*80)
    logger.info("VISION TRANSFORMER STATISTICAL MODEL COMPARISON")
    logger.info("="*80)
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Comparisons will be slower.")
    
    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Verify paths exist
    if not os.path.exists(BASELINE_DIR):
        logger.error(f"Baseline directory not found: {BASELINE_DIR}")
        return
    
    if not os.path.exists(PRUNED_MODELS_DIR):
        logger.error(f"Pruned models directory not found: {PRUNED_MODELS_DIR}")
        return
    
    if not os.path.exists(DATASETS_DIR):
        logger.error(f"Datasets directory not found: {DATASETS_DIR}")
        return
    
    # Discover models
    logger.info("\nDiscovering models...")
    all_models = discover_models()
    
    if not all_models:
        logger.error("No models found!")
        return
    
    # Store all results
    all_results = []
    
    # Process each dataset
    for dataset_info in all_models:
        dataset_name = dataset_info['dataset']
        models = dataset_info['models']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*80}")
        
        # Load test data
        dataset_path = DATASETS[dataset_name]
        num_classes = DATASET_NUM_CLASSES[dataset_name]
        
        logger.info(f"Loading test data from {dataset_path}...")
        test_loader = make_test_loader(dataset_path, batch_size=32)
        
        # Separate baseline and compressed models
        baseline_models = []
        compressed_models = []
        
        for model in models:
            if model['pruning_method'] == 'baseline':
                baseline_models.append(model)
            else:
                compressed_models.append(model)
        
        if not baseline_models:
            logger.error(f"No baseline models found for {dataset_name}!")
            continue
        
        logger.info(f"Found {len(baseline_models)} baseline models and {len(compressed_models)} compressed models")
        
        # For each compressed model, find the matching baseline
        # For quantization: compare to baseline of same size
        # For KD: compare student to TEACHER (e.g., distilled small to baseline base)
        for compressed_model in compressed_models:
            baseline_model = None
            
            if compressed_model['pruning_method'] == 'kd':
                # For KD models, compare student to teacher
                # Extract teacher size from teacher_model name
                teacher_name = compressed_model['teacher_model']
                teacher_size = None
                if teacher_name:
                    if 'tiny' in teacher_name:
                        teacher_size = 'tiny'
                    elif 'small' in teacher_name:
                        teacher_size = 'small'
                    elif 'base' in teacher_name:
                        teacher_size = 'base'
                    
                    # Find baseline matching teacher size
                    for bl_model in baseline_models:
                        if teacher_size and teacher_size in bl_model['model_name']:
                            baseline_model = bl_model
                            logger.info(f"KD comparison: {compressed_model['full_name']} vs teacher baseline {bl_model['model_name']}")
                            break
            else:
                # For quantization, compare to baseline of same size
                compressed_size = None
                if 'tiny' in compressed_model['model_name']:
                    compressed_size = 'tiny'
                elif 'small' in compressed_model['model_name']:
                    compressed_size = 'small'
                elif 'base' in compressed_model['model_name']:
                    compressed_size = 'base'
                
                # Find matching baseline
                for bl_model in baseline_models:
                    if compressed_size and compressed_size in bl_model['model_name']:
                        baseline_model = bl_model
                        logger.info(f"Quantization comparison: {compressed_model['full_name']} vs baseline {bl_model['model_name']}")
                        break
            
            if baseline_model is None:
                logger.warning(f"No matching baseline found for {compressed_model['full_name']}")
                continue
            
            try:
                results = compare_models(
                    dataset_name,
                    baseline_model,
                    compressed_model,
                    test_loader,
                    num_classes,
                    device=device
                )
                all_results.append(results)
            except Exception as e:
                logger.error(f"Error comparing {compressed_model['full_name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Reorder columns for better readability
        column_order = [
            'dataset', 'baseline_model', 'compressed_model', 
            'pruning_method', 'sparsity', 'stored_precision',
            'teacher_model', 'student_model',
            'baseline_accuracy', 'compressed_accuracy', 'accuracy_difference',
            'mcnemar_statistic', 'mcnemar_p_value', 'mcnemar_significant',
            'baseline_auc', 'compressed_auc', 'auc_difference',
            'delong_z_score_mean', 'delong_p_value_mean', 'delong_p_value_min', 'delong_significant'
        ]
        results_df = results_df[column_order]
        
        # Save to CSV
        output_path = Path(PRUNED_MODELS_DIR).parent / "vit_statistical_comparison_results.csv"
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"{'='*80}")
        
        # Print summary
        logger.info("\nSummary of comparisons:")
        logger.info(f"Total comparisons: {len(results_df)}")
        logger.info(f"Significant accuracy differences (McNemar p < 0.05): {results_df['mcnemar_significant'].sum()}")
        logger.info(f"Significant AUC differences (DeLong p < 0.05): {results_df['delong_significant'].sum()}")
        
        # Print detailed results
        logger.info("\nDetailed Results:")
        for _, row in results_df.iterrows():
            logger.info(f"\n{row['dataset']} - {row['compressed_model']}:")
            logger.info(f"  Baseline: {row['baseline_model']}")
            logger.info(f"  Accuracy: {row['baseline_accuracy']:.4f} → {row['compressed_accuracy']:.4f} (Δ={row['accuracy_difference']:+.4f})")
            logger.info(f"  McNemar: p={row['mcnemar_p_value']:.4f} {'***' if row['mcnemar_significant'] else ''}")
            logger.info(f"  AUC: {row['baseline_auc']:.4f} → {row['compressed_auc']:.4f} (Δ={row['auc_difference']:+.4f})")
            logger.info(f"  DeLong: p={row['delong_p_value_mean']:.4f} {'***' if row['delong_significant'] else ''}")
            if row['teacher_model']:
                logger.info(f"  KD: {row['teacher_model']} → {row['student_model']}")
    else:
        logger.error("No results to save!")

if __name__ == "__main__":
    main()