#!/usr/bin/env python3
"""
Bootstrap confidence intervals for accuracy and AUC across all discovered ViT models.

Reuses model discovery/loading and dataset utilities from ViT_benchmarking.py.
Unlike the full factorial throughput/energy benchmark, this script runs a single
inference pass per model to collect per-example predictions, then resamples those
predictions with replacement (bootstrap) to estimate confidence intervals for
accuracy and AUC. This is a proper prediction-level bootstrap, as opposed to
bootstrapping over the small number of repeat-run summary statistics.
"""

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from ViT_benchmarking import (
    SEED, MULTI_LABEL_DATASETS, DATASETS, DATASET_NUM_CLASSES,
    BASELINE_DIR, PRUNED_MODELS_DIR, DATASETS_DIR,
    Config, discover_models, load_model, make_test_loader,
    set_seed, set_env_threads, get_git_commit,
)

OUTPUT_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/PruneAndTrain/Vision/rerun/bootstrap_ci_results"
BATCH_SIZE = 32
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95


def collect_predictions(model_cfg, dataset, device):
    """Run one inference pass over the full test set, at the model's stored precision."""
    num_classes = DATASET_NUM_CLASSES[dataset]
    multi_label = dataset in MULTI_LABEL_DATASETS
    dataset_path = DATASETS[dataset]

    test_loader, _ = make_test_loader(dataset, dataset_path, BATCH_SIZE)

    precision = model_cfg.get('stored_precision', 'fp32')
    exp_cfg = {
        'model_name': model_cfg['model_name'],
        'full_name': model_cfg['full_name'],
        'pruning_method': model_cfg['pruning_method'],
        'sparsity': model_cfg['sparsity'],
        'stored_precision': precision,
        'model_path': model_cfg['model_path'],
        'precision': precision,
    }
    config = Config(experiment=exp_cfg, log_dir=str(OUTPUT_DIR))

    model = load_model(config, num_classes)
    model = model.to(device).eval()
    model_dtype = next(model.parameters()).dtype

    probs_list, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            if inputs.dtype != model_dtype:
                inputs = inputs.to(dtype=model_dtype)

            if precision == 'amp' and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            if outputs.dtype == torch.float16:
                outputs = outputs.float()
            outputs = torch.clamp(outputs, min=-100, max=100)

            probs = torch.sigmoid(outputs) if multi_label else torch.softmax(outputs, dim=1)
            probs_list.append(probs.cpu().numpy())
            labels_list.append(labels.numpy())

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    all_probs = np.concatenate(probs_list)
    all_labels = np.concatenate(labels_list)
    return all_probs, all_labels, num_classes, multi_label


def compute_accuracy(probs, labels, multi_label):
    """Standard top-1 accuracy; for multi-label (chestmnist), mean per-label (Hamming) accuracy."""
    if multi_label:
        preds = (probs >= 0.5).astype(int)
        return accuracy_score(labels.ravel(), preds.ravel())
    preds = np.argmax(probs, axis=1)
    return accuracy_score(labels, preds)


def compute_auc(probs, labels, multi_label, num_classes):
    try:
        if multi_label:
            return roc_auc_score(labels, probs, average='macro')
        return roc_auc_score(
            labels, probs, multi_class='ovr', average='weighted',
            labels=list(range(num_classes))
        )
    except Exception:
        return float('nan')


def bootstrap_metric_cis(probs, labels, multi_label, num_classes,
                          n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, seed=SEED):
    """Resample test examples with replacement and compute accuracy/AUC distributions."""
    n = len(labels)
    rng = np.random.default_rng(seed)

    acc_point = compute_accuracy(probs, labels, multi_label)
    auc_point = compute_auc(probs, labels, multi_label, num_classes)

    acc_boot = np.empty(n_bootstrap)
    auc_boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        acc_boot[i] = compute_accuracy(probs[idx], labels[idx], multi_label)
        auc_boot[i] = compute_auc(probs[idx], labels[idx], multi_label, num_classes)

    alpha = (1 - ci) / 2 * 100
    acc_ci_low, acc_ci_high = np.nanpercentile(acc_boot, [alpha, 100 - alpha])
    auc_ci_low, auc_ci_high = np.nanpercentile(auc_boot, [alpha, 100 - alpha])

    return {
        'accuracy': acc_point,
        'accuracy_ci_low': acc_ci_low,
        'accuracy_ci_high': acc_ci_high,
        'auc': auc_point,
        'auc_ci_low': auc_ci_low,
        'auc_ci_high': auc_ci_high,
        'n_test_samples': n,
        'n_bootstrap': n_bootstrap,
    }


def plot_results(results_df, output_dir):
    """Bar plots of accuracy and AUC with bootstrap CI error bars, per dataset."""
    for dataset in results_df['dataset'].unique():
        df = results_df[results_df['dataset'] == dataset].copy()
        df['label'] = df['pruning_method'] + '_' + df['sparsity'].astype(str)

        for metric in ['accuracy', 'auc']:
            fig, ax = plt.subplots(figsize=(10, 6))
            yerr = np.array([
                df[metric] - df[f'{metric}_ci_low'],
                df[f'{metric}_ci_high'] - df[metric],
            ])
            ax.bar(df['label'], df[metric], yerr=yerr, capsize=4)
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} with {int(CI_LEVEL*100)}% Bootstrap CI - {dataset}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / f'{dataset}_{metric}_ci.png', dpi=300)
            plt.close(fig)


def run_bootstrap_analysis():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(SEED)
    set_env_threads(omp_threads=4, mkl_threads=4)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_models = discover_models()

    all_results = []
    for entry in dataset_models:
        dataset = entry['dataset']
        models = entry['models']
        print(f"\n{'='*80}\nDataset: {dataset}\n{'='*80}")

        dataset_results = []
        for model_cfg in models:
            print(f"\nRunning: {model_cfg['full_name']} on {dataset}")
            try:
                probs, labels, num_classes, multi_label = collect_predictions(model_cfg, dataset, device)
                metrics = bootstrap_metric_cis(probs, labels, multi_label, num_classes)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                continue

            result = {
                'dataset': dataset,
                'model_name': model_cfg['full_name'],
                'pruning_method': model_cfg['pruning_method'],
                'sparsity': model_cfg['sparsity'],
                'stored_precision': model_cfg.get('stored_precision', 'unknown'),
                'teacher_model': model_cfg.get('teacher_model'),
                'student_model': model_cfg.get('student_model'),
                **metrics,
                'git_commit': get_git_commit(),
            }
            dataset_results.append(result)
            all_results.append(result)

            print(f"  Accuracy: {metrics['accuracy']:.4f} "
                  f"[{metrics['accuracy_ci_low']:.4f}, {metrics['accuracy_ci_high']:.4f}], "
                  f"AUC: {metrics['auc']:.4f} "
                  f"[{metrics['auc_ci_low']:.4f}, {metrics['auc_ci_high']:.4f}]")

        if dataset_results:
            pd.DataFrame(dataset_results).to_csv(output_dir / f"{dataset}_bootstrap_ci.csv", index=False)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / "all_bootstrap_ci.csv", index=False)
        plot_results(results_df, output_dir)
        print(f"\nResults saved to {output_dir}")

    return all_results


def main():
    print("ViT Bootstrap CI Analysis (Accuracy & AUC)")

    if not os.path.exists(BASELINE_DIR):
        print(f"Error: Baseline directory not found: {BASELINE_DIR}")
        return
    if not os.path.exists(PRUNED_MODELS_DIR):
        print(f"Error: Pruned models directory not found: {PRUNED_MODELS_DIR}")
        return
    if not os.path.exists(DATASETS_DIR):
        print(f"Error: Datasets directory not found: {DATASETS_DIR}")
        return

    if not torch.cuda.is_available():
        print("Warning: CUDA not available.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    run_bootstrap_analysis()
    print("Bootstrap CI analysis complete.")


if __name__ == "__main__":
    main()
