#!/usr/bin/env python3
"""
Organize ViT experiment outputs into folders by dataset.

This script:
1. Scans the output directory for experiment files
2. Creates subdirectories for each dataset
3. Moves files to their appropriate dataset folder
4. Generates a combined summary CSV across all datasets

Usage:
    python organize_vit_outputs.py                    # Use default directory
    python organize_vit_outputs.py --base-dir /path   # Custom directory
    python organize_vit_outputs.py --dry-run          # Preview without moving
"""

import os
import re
import shutil
import argparse
import pandas as pd
from pathlib import Path

# Default base directory
SAVE_DIR_BASE = "/home/arihangupta/Pruning/dinov2/Pruning/multipleTrials/vit"

# Known datasets
DATASETS = ['bloodmnist', 'pathmnist', 'dermamnist']

# File patterns for matching dataset files
# Pattern: {dataset}_{model}_{method/type}...
FILE_PATTERNS = [
    # Model files
    r'^(\w+mnist)_vit_.*\.pth$',
    # CSV files
    r'^(\w+mnist)_vit_.*\.csv$',
    # Emissions files from codecarbon
    r'^emissions.*\.csv$',
]


def get_dataset_from_filename(filename):
    """Extract dataset name from filename."""
    for dataset in DATASETS:
        if filename.startswith(dataset):
            return dataset
    return None


def organize_files(base_dir, dry_run=False):
    """Organize files into dataset folders."""
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    print(f"Organizing files in: {base_dir}")
    if dry_run:
        print("(DRY RUN - no changes will be made)\n")

    # Create dataset directories
    for dataset in DATASETS:
        dataset_dir = os.path.join(base_dir, dataset)
        if not os.path.exists(dataset_dir):
            if dry_run:
                print(f"Would create directory: {dataset}")
            else:
                os.makedirs(dataset_dir, exist_ok=True)
                print(f"Created directory: {dataset}")

    # Track files for summary
    moved_files = {dataset: [] for dataset in DATASETS}
    skipped_files = []
    root_files = []

    # Scan files in base directory (not subdirectories)
    for filename in os.listdir(base_dir):
        filepath = os.path.join(base_dir, filename)

        # Skip directories
        if os.path.isdir(filepath):
            continue

        # Skip files that should stay in root
        if filename in ['all_experiments_combined.csv', 'all_datasets_summary.csv']:
            root_files.append(filename)
            continue

        # Determine target dataset
        dataset = get_dataset_from_filename(filename)

        if dataset:
            target_dir = os.path.join(base_dir, dataset)
            target_path = os.path.join(target_dir, filename)

            if os.path.exists(target_path):
                skipped_files.append((filename, "already exists in target"))
                continue

            if dry_run:
                print(f"  Would move: {filename} -> {dataset}/")
            else:
                shutil.move(filepath, target_path)
                print(f"  Moved: {filename} -> {dataset}/")

            moved_files[dataset].append(filename)
        else:
            # Check if it's an emissions file or other root-level file
            if filename.startswith('emissions'):
                skipped_files.append((filename, "emissions file (keep in root)"))
            else:
                skipped_files.append((filename, "unknown dataset"))

    # Print summary
    print(f"\n{'='*60}")
    print("ORGANIZATION SUMMARY")
    print(f"{'='*60}")

    for dataset in DATASETS:
        count = len(moved_files[dataset])
        if count > 0:
            print(f"  {dataset}: {count} files moved")

    if root_files:
        print(f"\n  Root files (kept): {', '.join(root_files)}")

    if skipped_files:
        print(f"\n  Skipped files:")
        for fname, reason in skipped_files[:10]:  # Show first 10
            print(f"    - {fname}: {reason}")
        if len(skipped_files) > 10:
            print(f"    ... and {len(skipped_files) - 10} more")

    return moved_files


def generate_combined_summary(base_dir, dry_run=False):
    """Generate a combined summary CSV from all dataset summaries."""
    print(f"\n{'='*60}")
    print("GENERATING COMBINED SUMMARY")
    print(f"{'='*60}")

    all_summaries = []

    for dataset in DATASETS:
        dataset_dir = os.path.join(base_dir, dataset)

        if not os.path.exists(dataset_dir):
            # Try looking in base dir instead (files might not be organized yet)
            dataset_dir = base_dir

        # Find summary files for this dataset
        for filename in os.listdir(dataset_dir):
            if filename.endswith('_summary_statistics.csv') and filename.startswith(dataset):
                filepath = os.path.join(dataset_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    all_summaries.append(df)
                    print(f"  Loaded: {filename}")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        output_path = os.path.join(base_dir, "all_datasets_summary.csv")

        if dry_run:
            print(f"\n  Would save combined summary ({len(combined)} rows) to: all_datasets_summary.csv")
        else:
            combined.to_csv(output_path, index=False)
            print(f"\n  Saved combined summary ({len(combined)} rows) to: all_datasets_summary.csv")

        # Print summary table
        print(f"\n{'='*120}")
        print("COMBINED SUMMARY")
        print(f"{'='*120}")

        if 'test_acc_mean' in combined.columns:
            print(f"{'Dataset':<15} {'Model':<25} {'Method':<30} {'Test Acc':<20} {'Params (M)':<15}")
            print("-"*120)
            for _, row in combined.iterrows():
                dataset = row.get('dataset', 'N/A')
                model = row.get('model', 'N/A')
                if isinstance(model, str):
                    model = model.replace('_patch16_224', '')
                method = row.get('method', 'N/A')
                acc_mean = row.get('test_acc_mean', float('nan'))
                acc_std = row.get('test_acc_std', float('nan'))
                params = row.get('params_m_mean', float('nan'))
                print(f"{dataset:<15} {model:<25} {method:<30} {acc_mean:.4f}±{acc_std:.4f}      {params:.2f}")

        return combined
    else:
        print("  No summary files found.")
        return None


def combine_all_trials(base_dir, dry_run=False):
    """Combine all individual trial metrics into one CSV."""
    print(f"\n{'='*60}")
    print("COMBINING ALL TRIAL METRICS")
    print(f"{'='*60}")

    all_trials = []

    for dataset in DATASETS:
        dataset_dir = os.path.join(base_dir, dataset)

        if not os.path.exists(dataset_dir):
            dataset_dir = base_dir

        for filename in os.listdir(dataset_dir):
            if filename.endswith('_all_trials_metrics.csv') and filename.startswith(dataset):
                filepath = os.path.join(dataset_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    all_trials.append(df)
                    print(f"  Loaded: {filename} ({len(df)} rows)")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")

    if all_trials:
        combined = pd.concat(all_trials, ignore_index=True)
        output_path = os.path.join(base_dir, "all_experiments_combined.csv")

        if dry_run:
            print(f"\n  Would save combined trials ({len(combined)} rows) to: all_experiments_combined.csv")
        else:
            combined.to_csv(output_path, index=False)
            print(f"\n  Saved combined trials ({len(combined)} rows) to: all_experiments_combined.csv")

        return combined
    else:
        print("  No trial metrics files found.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Organize ViT experiment outputs into folders by dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python organize_vit_outputs.py                     # Organize with default directory
  python organize_vit_outputs.py --dry-run           # Preview changes
  python organize_vit_outputs.py --base-dir /path    # Custom directory
  python organize_vit_outputs.py --summary-only      # Only generate summary CSVs
        """
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--base-dir', type=str, default=SAVE_DIR_BASE,
                        help=f'Base directory for experiment results (default: {SAVE_DIR_BASE})')
    parser.add_argument('--summary-only', action='store_true',
                        help='Only generate summary CSVs, do not move files')

    args = parser.parse_args()

    base_dir = args.base_dir

    if not args.summary_only:
        organize_files(base_dir, dry_run=args.dry_run)

    generate_combined_summary(base_dir, dry_run=args.dry_run)
    combine_all_trials(base_dir, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
