#!/usr/bin/env python3
"""
Bootstrap confidence intervals for accuracy and AUC across all discovered CNN models.

Reuses model discovery/loading and dataset utilities from CNN_benchmarking.py.
Unlike the full factorial throughput/energy benchmark, this script runs a single
inference pass per model to collect per-example predictions, then resamples those
predictions with replacement (bootstrap) to estimate confidence intervals for
accuracy and AUC. This is a proper prediction-level bootstrap, as opposed to
bootstrapping over the small number of repeat-run summary statistics.

Each model is evaluated in its own spawned subprocess. chestmnist's test set is
far larger than the other datasets, and a run there previously got OOM-killed by
the OS mid-script, taking the whole job down with it. Isolating each model in a
subprocess means a kill only takes out that one subprocess (visible as a non-zero
exitcode); the driver catches it, retries with fewer bootstrap resamples, and
otherwise keeps going. Per-model results are checkpointed to CSV immediately so a
rerun resumes instead of starting over.
"""

import os
import gc
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

from CNN_benchmarking import (
    SEED, IMG_SIZE, MULTI_LABEL_DATASETS, DATASETS, SAVE_DIR_BASE,
    Config, NumpyMemmapDataset, discover_models, load_model, get_num_classes,
    get_dataset_channels, set_seed, set_env_threads, get_git_commit,
)

OUTPUT_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/PruneAndTrain/CNN/bootstrap_ci_results"
BATCH_SIZE = 32
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95
MAX_RETRIES = 2  # on subprocess kill, retry with fewer bootstrap resamples


def make_bootstrap_loader(dataset_name, dataset_path, batch_size=BATCH_SIZE):
    """Single-process loader (num_workers=0) to avoid fork-based worker memory duplication."""
    multi_label = dataset_name in MULTI_LABEL_DATASETS
    if multi_label:
        X_test = np.load(os.path.join(dataset_path, "test_images.npy"), mmap_mode="r")
        y_test = np.load(os.path.join(dataset_path, "test_labels.npy"), mmap_mode="r")
        in_channels = X_test[0].shape[-1] if X_test[0].ndim == 3 else 1
    else:
        data = np.load(dataset_path, mmap_mode="r")
        X_test, y_test = data["test_images"], data["test_labels"].flatten()
        in_channels = get_dataset_channels(dataset_path)
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, in_channels=in_channels, multi_label=multi_label)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return loader, len(test_ds)


def collect_predictions(model_cfg, dataset, device):
    """Run one inference pass over the full test set, at the model's stored precision."""
    dataset_path = DATASETS[dataset]
    multi_label = dataset in MULTI_LABEL_DATASETS
    num_classes = get_num_classes(dataset, dataset_path)

    test_loader, _ = make_bootstrap_loader(dataset, dataset_path)

    precision = model_cfg.get('stored_precision', 'fp32')
    exp_cfg = {
        'model_name': model_cfg['model_name'],
        'pruning_method': model_cfg['pruning_method'],
        'sparsity': model_cfg['sparsity'],
        'pruning_ratio': model_cfg['pruning_ratio'],
        'stored_precision': precision,
        'model_path': model_cfg['model_path'],
        'precision': precision,
        'device': device,
    }
    config = Config(experiment=exp_cfg, log_dir=str(OUTPUT_DIR))

    model = load_model(config, num_classes, dataset)
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    probs_list, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(model_device)
            if inputs.dtype != model_dtype:
                inputs = inputs.to(dtype=model_dtype)

            if precision == 'amp' and model_device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            if outputs.dtype == torch.half:
                outputs = outputs.float()
            outputs = torch.clamp(outputs, min=-100, max=100)

            probs = torch.sigmoid(outputs) if multi_label else torch.softmax(outputs, dim=1)
            probs_list.append(probs.cpu().numpy())
            labels_list.append(labels.numpy())

    del model, test_loader
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    all_probs = np.concatenate(probs_list)
    all_labels = np.concatenate(labels_list)
    del probs_list, labels_list
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


def _worker_run_model(model_cfg, dataset, device_str, n_bootstrap, queue):
    """Runs in a spawned subprocess so an OOM kill only takes out this process."""
    try:
        device = torch.device(device_str)
        probs, labels, num_classes, multi_label = collect_predictions(model_cfg, dataset, device)
        metrics = bootstrap_metric_cis(probs, labels, multi_label, num_classes, n_bootstrap=n_bootstrap)
        queue.put(('ok', metrics))
    except Exception as e:
        import traceback
        queue.put(('error', f"{e}\n{traceback.format_exc()}"))


def run_model_isolated(model_cfg, dataset, device_str, n_bootstrap=N_BOOTSTRAP, max_retries=MAX_RETRIES):
    """Run one model's collect+bootstrap in a subprocess; retry with fewer resamples if it gets killed."""
    ctx = mp.get_context('spawn')
    attempt_n = n_bootstrap

    for attempt in range(max_retries + 1):
        queue = ctx.Queue()
        p = ctx.Process(target=_worker_run_model, args=(model_cfg, dataset, device_str, attempt_n, queue))
        p.start()
        p.join()

        if p.exitcode != 0:
            msg = f"Subprocess killed (exit code {p.exitcode}), likely OOM at n_bootstrap={attempt_n}"
            if attempt < max_retries:
                attempt_n = max(attempt_n // 2, 200)
                print(f"  Warning: {msg}. Retrying with n_bootstrap={attempt_n}...")
                continue
            return None, msg

        if queue.empty():
            return None, "Subprocess exited cleanly but returned no result"

        status, payload = queue.get()
        if status == 'ok':
            return payload, None
        return None, payload

    return None, "Exhausted retries"


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
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
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

        dataset_csv = output_dir / f"{dataset}_bootstrap_ci.csv"
        dataset_results = []
        completed = set()
        if dataset_csv.exists():
            existing_df = pd.read_csv(dataset_csv)
            dataset_results = existing_df.to_dict('records')
            completed = set(existing_df['model_name'])
            all_results.extend(dataset_results)
            print(f"Resuming: {len(completed)} model(s) already done for {dataset}, skipping them.")

        for model_cfg in models:
            if model_cfg['model_name'] in completed:
                continue

            print(f"\nRunning: {model_cfg['model_name']} on {dataset}")
            metrics, error = run_model_isolated(model_cfg, dataset, device_str)
            gc.collect()

            if error is not None:
                print(f"  Error: {error}")
                continue

            result = {
                'dataset': dataset,
                'model_name': model_cfg['model_name'],
                'pruning_method': model_cfg['pruning_method'],
                'sparsity': model_cfg['sparsity'],
                'stored_precision': model_cfg.get('stored_precision', 'unknown'),
                **metrics,
                'git_commit': get_git_commit(),
            }
            dataset_results.append(result)
            all_results.append(result)

            # Checkpoint after every model so a crash never loses more than one model's work.
            pd.DataFrame(dataset_results).to_csv(dataset_csv, index=False)

            print(f"  Accuracy: {metrics['accuracy']:.4f} "
                  f"[{metrics['accuracy_ci_low']:.4f}, {metrics['accuracy_ci_high']:.4f}], "
                  f"AUC: {metrics['auc']:.4f} "
                  f"[{metrics['auc_ci_low']:.4f}, {metrics['auc_ci_high']:.4f}]"
                  f"{' (n_bootstrap=' + str(metrics['n_bootstrap']) + ')' if metrics['n_bootstrap'] != N_BOOTSTRAP else ''}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / "all_bootstrap_ci.csv", index=False)
        plot_results(results_df, output_dir)
        print(f"\nResults saved to {output_dir}")

    return all_results


def main():
    print("CNN Bootstrap CI Analysis (Accuracy & AUC)")

    if not os.path.exists(SAVE_DIR_BASE):
        print(f"Error: Pruned models directory not found: {SAVE_DIR_BASE}")
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
