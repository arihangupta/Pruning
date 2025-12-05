#!/usr/bin/env python3
"""
Statistical comparison of pruned models vs baseline using:
- DeLong's test for AUC comparison
- McNemar's test for accuracy comparison

Uses the same model discovery and loading infrastructure as the benchmarking script.
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
warnings.filterwarnings('ignore')

# Import torchvision Bottleneck
from torchvision.models.resnet import Bottleneck

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants (same as benchmarking script)
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
DATASET_NUM_CLASSES = {
    "bloodmnist": 8,
    "dermamnist": 7,
    "octmnist": 4,
    "pathmnist": 9,
    "tissuemnist": 8
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
# Model Architecture (same as benchmarking script)
# ============================================================================

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

# ============================================================================
# Dataset and Model Loading (same as benchmarking script)
# ============================================================================

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
        logger.error(f"Error detecting channels for {npz_path}: {e}")
        return 3

def make_test_loader(npz_path, batch_size=32):
    data = np.load(npz_path, mmap_mode="r")
    X_test, y_test = data["test_images"], data["test_labels"].flatten()
    in_channels = get_dataset_channels(npz_path)
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, in_channels=in_channels)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True), in_channels

def parse_model_name(filename, dataset):
    """Parse model filename to extract metadata."""
    basename = os.path.basename(filename)
    
    # Handle baseline
    if basename == "baseline.pth":
        return {
            "model_name": "baseline",
            "pruning_method": "baseline",
            "sparsity": "0%",
            "pruning_ratio": None,
            "stored_precision": "fp32"
        }
    
    if not basename.endswith("_final.pth"):
        return None
    
    stored_precision = "fp32"
    if "_fp16_" in basename or basename.endswith("_fp16_r50compressed_final.pth"):
        stored_precision = "amp"
    elif "quantization" in basename:
        stored_precision = "amp"
    
    # Extract pruning method
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
        return None

    # Extract sparsity
    compress_match = re.search(r'_r(\d+)compressed', basename)
    if compress_match:
        compress_ratio = int(compress_match.group(1))
        sparsity = f"{compress_ratio}%"
        pruning_ratio = compress_ratio / 100.0
    else:
        sparsity = "50%"
        pruning_ratio = 0.5

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
    """Discover all models for each dataset."""
    models = []
    for dataset in TARGET_DATASETS:
        model_dir = os.path.join(SAVE_DIR_BASE, dataset)
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory for {dataset} does not exist: {model_dir}")
            continue
        
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
        
        if dataset_models:
            models.append({"dataset": dataset, "models": dataset_models})
            logger.info(f"Found {len(dataset_models)} models for {dataset}")
    
    return models

def load_model(model_path, num_classes, in_channels=3, device='cuda:0'):
    """Load a model from checkpoint."""
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
    
    model = CustomResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        stage_planes=stage_planes,
        num_classes=num_classes,
        in_channels=3
    )
    
    # Handle dtype
    sample_param = next(iter(state_dict.values()))
    is_fp16 = sample_param.dtype == torch.float16
    
    if is_fp16:
        model = model.half()
        state_dict = {k: v.half() if v.dtype == torch.float32 else v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
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

def compare_models(dataset_name, baseline_info, pruned_info, dataloader, num_classes, device='cuda:0'):
    """Compare a pruned model to baseline using DeLong and McNemar tests."""
    logger.info(f"\nComparing {pruned_info['model_name']} to baseline for {dataset_name}")
    
    # Load baseline model
    logger.info("Loading baseline model...")
    baseline_model, baseline_fp16 = load_model(
        baseline_info['model_path'], 
        num_classes, 
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
            labels=list(range(num_classes))
        )
    except:
        baseline_auc = np.nan
    
    logger.info(f"Baseline - Accuracy: {baseline_acc:.4f}, AUC: {baseline_auc:.4f}")
    
    # Clean up baseline model
    del baseline_model
    torch.cuda.empty_cache()
    
    # Load pruned model
    logger.info(f"Loading pruned model: {pruned_info['model_name']}...")
    pruned_model, pruned_fp16 = load_model(
        pruned_info['model_path'], 
        num_classes, 
        device=device
    )
    
    # Get pruned predictions
    logger.info("Getting pruned model predictions...")
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
            labels=list(range(num_classes))
        )
    except:
        pruned_auc = np.nan
    
    logger.info(f"Pruned - Accuracy: {pruned_acc:.4f}, AUC: {pruned_auc:.4f}")
    
    # Perform statistical tests
    results = {
        'dataset': dataset_name,
        'pruning_method': pruned_info['pruning_method'],
        'sparsity': pruned_info['sparsity'],
        'stored_precision': pruned_info['stored_precision'],
        'baseline_accuracy': baseline_acc,
        'pruned_accuracy': pruned_acc,
        'accuracy_difference': pruned_acc - baseline_acc,
        'baseline_auc': baseline_auc,
        'pruned_auc': pruned_auc,
        'auc_difference': pruned_auc - baseline_auc,
    }
    
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

def main():
    """Main function to run comparisons."""
    logger.info("Starting statistical model comparison...")
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Discover models
    logger.info("Discovering models...")
    all_models = discover_models()
    
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
        test_loader, in_channels = make_test_loader(dataset_path, batch_size=32)
        
        # Find baseline model
        baseline_model = None
        pruned_models = []
        
        for model in models:
            if model['pruning_method'] == 'baseline':
                baseline_model = model
            else:
                pruned_models.append(model)
        
        if baseline_model is None:
            logger.error(f"No baseline model found for {dataset_name}!")
            continue
        
        logger.info(f"Found baseline and {len(pruned_models)} pruned models")
        
        # Compare each pruned model to baseline
        for pruned_model in pruned_models:
            try:
                results = compare_models(
                    dataset_name,
                    baseline_model,
                    pruned_model,
                    test_loader,
                    num_classes,
                    device=device
                )
                all_results.append(results)
            except Exception as e:
                logger.error(f"Error comparing {pruned_model['model_name']}: {e}")
                continue
    
    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Reorder columns for better readability
        column_order = [
            'dataset', 'pruning_method', 'sparsity', 'stored_precision',
            'baseline_accuracy', 'pruned_accuracy', 'accuracy_difference',
            'mcnemar_statistic', 'mcnemar_p_value', 'mcnemar_significant',
            'baseline_auc', 'pruned_auc', 'auc_difference',
            'delong_z_score_mean', 'delong_p_value_mean', 'delong_p_value_min', 'delong_significant'
        ]
        results_df = results_df[column_order]
        
        # Save to CSV
        output_path = Path(SAVE_DIR_BASE).parent / "statistical_comparison_results.csv"
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
            logger.info(f"\n{row['dataset']} - {row['pruning_method']} ({row['sparsity']}):")
            logger.info(f"  Accuracy: {row['baseline_accuracy']:.4f} → {row['pruned_accuracy']:.4f} (Δ={row['accuracy_difference']:+.4f})")
            logger.info(f"  McNemar: p={row['mcnemar_p_value']:.4f} {'***' if row['mcnemar_significant'] else ''}")
            logger.info(f"  AUC: {row['baseline_auc']:.4f} → {row['pruned_auc']:.4f} (Δ={row['auc_difference']:+.4f})")
            logger.info(f"  DeLong: p={row['delong_p_value_mean']:.4f} {'***' if row['delong_significant'] else ''}")
    else:
        logger.error("No results to save!")

if __name__ == "__main__":
    main()