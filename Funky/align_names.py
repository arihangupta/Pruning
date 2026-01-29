#!/usr/bin/env python3
"""
Aligns file names and method names in CSVs to be consistent.

Standardizes to:
- baseline_l1
- baseline_no_reg
- prune_then_train
- progressive_with_reg
- progressive_no_reg
"""

import os
import re
import pandas as pd
from pathlib import Path

# Base directory where experiment results are saved
SAVE_DIR_BASE = "/home/arihangupta/Pruning/dinov2/Pruning/multipleTrials/CNN"

DATASETS = ["pathmnist", "dermamnist", "bloodmnist"]

# Method name mappings (old -> new)
METHOD_NAME_MAP = {
    # In all_trials CSV
    'baseline_l1_regularized': 'baseline_l1',
    'baseline_no_regularization': 'baseline_no_reg',
    'progressive_pruning_with_reg': 'progressive_with_reg',
    'progressive_pruning_no_reg': 'progressive_no_reg',
    # Already correct, but include for completeness
    'baseline_l1': 'baseline_l1',
    'baseline_no_reg': 'baseline_no_reg',
    'prune_then_train': 'prune_then_train',
    'progressive_with_reg': 'progressive_with_reg',
    'progressive_no_reg': 'progressive_no_reg',
}

# File rename patterns: (old_pattern, new_pattern)
# Uses regex groups for dataset and trial number
FILE_RENAME_PATTERNS = [
    # baseline_trial -> baseline_l1_trial (for .pth and .csv files)
    (r'^(\w+)_baseline_trial(\d+)_(.+)$', r'\1_baseline_l1_trial\2_\3'),
    # baseline_training_trial -> baseline_l1_training_trial (for emissions tracking)
    (r'^(\w+)_baseline_training_trial(\d+)(.*)$', r'\1_baseline_l1_training_trial\2\3'),
]


def update_csv_method_names(csv_path):
    """Update method names in a CSV file."""
    if not os.path.exists(csv_path):
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Error reading {csv_path}: {e}")
        return False

    if 'method' not in df.columns:
        return False

    original_methods = df['method'].unique().tolist()
    df['method'] = df['method'].map(lambda x: METHOD_NAME_MAP.get(x, x))
    new_methods = df['method'].unique().tolist()

    if original_methods != new_methods:
        df.to_csv(csv_path, index=False)
        print(f"  Updated methods in {os.path.basename(csv_path)}: {original_methods} -> {new_methods}")
        return True

    return False


def rename_files(save_dir, dataset_name, dry_run=False):
    """Rename files to use consistent naming."""
    renamed = []

    for filename in os.listdir(save_dir):
        new_filename = filename

        for old_pattern, new_pattern in FILE_RENAME_PATTERNS:
            if re.match(old_pattern, filename):
                new_filename = re.sub(old_pattern, new_pattern, filename)
                break

        if new_filename != filename:
            old_path = os.path.join(save_dir, filename)
            new_path = os.path.join(save_dir, new_filename)

            if os.path.exists(new_path):
                print(f"  Warning: {new_filename} already exists, skipping rename of {filename}")
                continue

            if dry_run:
                print(f"  Would rename: {filename} -> {new_filename}")
            else:
                os.rename(old_path, new_path)
                print(f"  Renamed: {filename} -> {new_filename}")
            renamed.append((filename, new_filename))

    return renamed


def process_dataset(dataset_name, base_dir, dry_run=False):
    """Process a single dataset directory."""
    save_dir = os.path.join(base_dir, dataset_name)

    if not os.path.exists(save_dir):
        print(f"Directory not found: {save_dir}")
        return

    print(f"\nProcessing {dataset_name}...")

    # Update CSV files
    csv_files = [
        f"{dataset_name}_all_trials_metrics.csv",
        f"{dataset_name}_summary_statistics.csv",
    ]

    for csv_file in csv_files:
        csv_path = os.path.join(save_dir, csv_file)
        if os.path.exists(csv_path):
            update_csv_method_names(csv_path)

    # Rename model and history files
    rename_files(save_dir, dataset_name, dry_run)


def update_combined_summary(base_dir, dry_run=False):
    """Update the combined summary CSV if it exists."""
    combined_path = os.path.join(base_dir, "all_datasets_summary.csv")

    if os.path.exists(combined_path):
        print(f"\nProcessing combined summary...")
        update_csv_method_names(combined_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Align file and method names for pruning experiments")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--base-dir", type=str, default=SAVE_DIR_BASE, help="Base directory for experiment results")
    args = parser.parse_args()

    base_dir = args.base_dir

    print(f"Aligning names in: {base_dir}")
    if args.dry_run:
        print("(DRY RUN - no changes will be made)")

    for dataset_name in DATASETS:
        process_dataset(dataset_name, base_dir, dry_run=args.dry_run)

    update_combined_summary(base_dir, dry_run=args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
