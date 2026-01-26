"""
Vision Transformer Progressive Pruning - Hyperparameter Tuning
===============================================================

Comprehensive hyperparameter search with robust error handling.
Tests 18 configurations with detailed logging and automatic recovery from failures.

Strategy:
- Phase 1: All 18 configs × 1 trial × bloodmnist (complete sweep)
- Phase 2: Top 3 configs × 3 trials × bloodmnist (validation)
- Phase 3: Best config × 3 trials × all 3 datasets (final test)

Error Handling:
- Automatic try-catch around each config
- Failed configs logged with error details
- Training continues to next config on failure
- Memory cleanup after each run

Output:
- Comprehensive CSV with ALL metrics and hyperparameters
- Separate error log for failed configs
- Per-epoch training history for each config
- Gate statistics and sparsity evolution

Key Hypotheses to Test:
1. LR restart after pruning vs. reduction
2. Gate regularization strength
3. Sparsity schedule (aggressive vs. gradual)
4. Deployment finetuning duration/LR
5. More but gentler pruning steps
6. Weight decay impact

Date: 2026
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import timm
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import copy
import gc
from functools import partial
import time
from datetime import datetime

# Import from original script (assuming it's in the same directory)
# We'll copy the necessary classes and functions

# ==================== CONFIGURATION ====================

DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/multipleTrials/vit_hyperparam_tuning"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 64

os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== HYPERPARAMETER CONFIGURATIONS ====================

CONFIGS = [
    # ============================================================
    # GROUP 1: Learning Rate Strategies (Configs 1-5)
    # ============================================================
    
    {
        'name': 'baseline_original',
        'description': 'Original settings from your code',
        'INITIAL_LR': 1e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.5,
        'GATE_L1_WEIGHT': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.2, 0.35, 0.45, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 3,
        'DEPLOY_FINETUNE_EPOCHS': 5,
        'DEPLOY_FINETUNE_LR': 5e-4,
        'TOTAL_EPOCHS': 15
    },
    
    {
        'name': 'high_lr_no_reduction',
        'description': 'Higher LR, no reduction after pruning (LR restart hypothesis)',
        'INITIAL_LR': 1e-3,
        'LR_REDUCTION_AFTER_PRUNE': 1.0,  # No reduction
        'GATE_L1_WEIGHT': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.2, 0.35, 0.45, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 3,
        'DEPLOY_FINETUNE_EPOCHS': 10,
        'DEPLOY_FINETUNE_LR': 1e-3,
        'TOTAL_EPOCHS': 15
    },
    
    {
        'name': 'lr_boost_after_prune',
        'description': 'INCREASE LR after pruning to help network adapt',
        'INITIAL_LR': 5e-4,
        'LR_REDUCTION_AFTER_PRUNE': 1.5,  # Boost by 50%
        'GATE_L1_WEIGHT': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.2, 0.35, 0.45, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 3,
        'DEPLOY_FINETUNE_EPOCHS': 10,
        'DEPLOY_FINETUNE_LR': 5e-4,
        'TOTAL_EPOCHS': 15
    },
    
    {
        'name': 'moderate_lr_gentle_reduction',
        'description': 'Moderate LR with gentle 30% reduction',
        'INITIAL_LR': 5e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.7,
        'GATE_L1_WEIGHT': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.2, 0.35, 0.45, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 3,
        'DEPLOY_FINETUNE_EPOCHS': 10,
        'DEPLOY_FINETUNE_LR': 5e-4,
        'TOTAL_EPOCHS': 15
    },
    
    {
        'name': 'very_high_lr_strong_reduction',
        'description': 'Start high, reduce aggressively (classic pruning approach)',
        'INITIAL_LR': 2e-3,
        'LR_REDUCTION_AFTER_PRUNE': 0.3,
        'GATE_L1_WEIGHT': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.2, 0.35, 0.45, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 3,
        'DEPLOY_FINETUNE_EPOCHS': 10,
        'DEPLOY_FINETUNE_LR': 1e-3,
        'TOTAL_EPOCHS': 15
    },
    
    # ============================================================
    # GROUP 2: Gate Regularization (Configs 6-9)
    # ============================================================
    
    {
        'name': 'very_low_gate_reg',
        'description': 'Minimal gate regularization (let importance scores dominate)',
        'INITIAL_LR': 5e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.7,
        'GATE_L1_WEIGHT': 1e-5,  # 10x lower
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.2, 0.35, 0.45, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 3,
        'DEPLOY_FINETUNE_EPOCHS': 10,
        'DEPLOY_FINETUNE_LR': 5e-4,
        'TOTAL_EPOCHS': 15
    },
    
    {
        'name': 'low_gate_reg',
        'description': 'Lower gate regularization',
        'INITIAL_LR': 5e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.7,
        'GATE_L1_WEIGHT': 5e-5,
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.2, 0.35, 0.45, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 3,
        'DEPLOY_FINETUNE_EPOCHS': 10,
        'DEPLOY_FINETUNE_LR': 5e-4,
        'TOTAL_EPOCHS': 15
    },
    
    {
        'name': 'high_gate_reg',
        'description': 'Higher gate regularization (more aggressive sparsity)',
        'INITIAL_LR': 5e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.7,
        'GATE_L1_WEIGHT': 5e-4,  # 5x higher
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.2, 0.35, 0.45, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 3,
        'DEPLOY_FINETUNE_EPOCHS': 10,
        'DEPLOY_FINETUNE_LR': 5e-4,
        'TOTAL_EPOCHS': 15
    },
    
    {
        'name': 'no_gate_reg',
        'description': 'No gate regularization (pure importance-based)',
        'INITIAL_LR': 5e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.7,
        'GATE_L1_WEIGHT': 0.0,  # No regularization
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.2, 0.35, 0.45, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 3,
        'DEPLOY_FINETUNE_EPOCHS': 10,
        'DEPLOY_FINETUNE_LR': 5e-4,
        'TOTAL_EPOCHS': 15
    },
    
    # ============================================================
    # GROUP 3: Sparsity Schedules (Configs 10-12)
    # ============================================================
    
    {
        'name': 'gradual_sparsity',
        'description': 'More gradual sparsity increase (gentler on network)',
        'INITIAL_LR': 5e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.7,
        'GATE_L1_WEIGHT': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.1, 0.2, 0.35, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 4,  # More time between prunes
        'DEPLOY_FINETUNE_EPOCHS': 10,
        'DEPLOY_FINETUNE_LR': 5e-4,
        'TOTAL_EPOCHS': 16
    },
    
    {
        'name': 'linear_sparsity',
        'description': 'Linear sparsity schedule',
        'INITIAL_LR': 5e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.7,
        'GATE_L1_WEIGHT': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.125, 0.25, 0.375, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 3,
        'DEPLOY_FINETUNE_EPOCHS': 10,
        'DEPLOY_FINETUNE_LR': 5e-4,
        'TOTAL_EPOCHS': 15
    },
    
    {
        'name': 'many_gentle_steps',
        'description': '5 pruning steps instead of 4 (more gradual)',
        'INITIAL_LR': 5e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.7,
        'GATE_L1_WEIGHT': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.1, 0.2, 0.3, 0.4, 0.5],
        'NUM_PRUNE_STEPS': 5,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 2,
        'DEPLOY_FINETUNE_EPOCHS': 10,
        'DEPLOY_FINETUNE_LR': 5e-4,
        'TOTAL_EPOCHS': 14
    },
    
    # ============================================================
    # GROUP 4: Deployment Finetuning (Configs 13-15)
    # ============================================================
    
    {
        'name': 'long_deploy_finetune',
        'description': 'Extended deployment finetuning (15 epochs)',
        'INITIAL_LR': 5e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.7,
        'GATE_L1_WEIGHT': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.2, 0.35, 0.45, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 3,
        'DEPLOY_FINETUNE_EPOCHS': 15,
        'DEPLOY_FINETUNE_LR': 5e-4,
        'TOTAL_EPOCHS': 15
    },
    
    {
        'name': 'high_lr_deploy_finetune',
        'description': 'Higher LR for deployment finetuning',
        'INITIAL_LR': 5e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.7,
        'GATE_L1_WEIGHT': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.2, 0.35, 0.45, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 3,
        'DEPLOY_FINETUNE_EPOCHS': 10,
        'DEPLOY_FINETUNE_LR': 1e-3,  # 2x higher
        'TOTAL_EPOCHS': 15
    },
    
    {
        'name': 'minimal_deploy_finetune',
        'description': 'Minimal deployment finetuning (test if needed)',
        'INITIAL_LR': 5e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.7,
        'GATE_L1_WEIGHT': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.2, 0.35, 0.45, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 3,
        'DEPLOY_FINETUNE_EPOCHS': 3,
        'DEPLOY_FINETUNE_LR': 5e-4,
        'TOTAL_EPOCHS': 15
    },
    
    # ============================================================
    # GROUP 5: Combined Strategies (Configs 16-18)
    # ============================================================
    
    {
        'name': 'aggressive_recovery',
        'description': 'Aggressive: high LR, low gate reg, long finetune',
        'INITIAL_LR': 1e-3,
        'LR_REDUCTION_AFTER_PRUNE': 1.0,
        'GATE_L1_WEIGHT': 1e-5,
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.2, 0.35, 0.45, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 3,
        'DEPLOY_FINETUNE_EPOCHS': 15,
        'DEPLOY_FINETUNE_LR': 1e-3,
        'TOTAL_EPOCHS': 15
    },
    
    {
        'name': 'conservative_stable',
        'description': 'Conservative: moderate everything for stability',
        'INITIAL_LR': 3e-4,
        'LR_REDUCTION_AFTER_PRUNE': 0.8,
        'GATE_L1_WEIGHT': 5e-5,
        'WEIGHT_DECAY': 1e-4,
        'SPARSITY_SCHEDULE': [0.1, 0.2, 0.35, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 3,
        'EPOCHS_BETWEEN_PRUNES': 4,
        'DEPLOY_FINETUNE_EPOCHS': 12,
        'DEPLOY_FINETUNE_LR': 3e-4,
        'TOTAL_EPOCHS': 17
    },
    
    {
        'name': 'optimal_hypothesis',
        'description': 'Best guess based on literature: moderate LR boost, low reg, gradual',
        'INITIAL_LR': 7e-4,
        'LR_REDUCTION_AFTER_PRUNE': 1.2,  # Slight boost
        'GATE_L1_WEIGHT': 5e-5,
        'WEIGHT_DECAY': 5e-5,  # Lower weight decay
        'SPARSITY_SCHEDULE': [0.1, 0.2, 0.35, 0.5],
        'NUM_PRUNE_STEPS': 4,
        'WARMUP_EPOCHS': 2,
        'EPOCHS_BETWEEN_PRUNES': 4,
        'DEPLOY_FINETUNE_EPOCHS': 12,
        'DEPLOY_FINETUNE_LR': 7e-4,
        'TOTAL_EPOCHS': 16
    },
]

# ==================== DETAILED OUTPUT SPECIFICATIONS ====================

# CSV will include ALL of these columns for comprehensive analysis:
OUTPUT_COLUMNS = [
    # Identification
    'config_name', 'config_description', 'trial', 'dataset', 'phase',
    
    # Hyperparameters - Learning Rate
    'initial_lr', 'lr_reduction_after_prune', 'min_lr', 
    
    # Hyperparameters - Regularization
    'gate_l1_weight', 'weight_decay',
    
    # Hyperparameters - Pruning Schedule
    'num_prune_steps', 'warmup_epochs', 'epochs_between_prunes',
    'sparsity_schedule', 'total_epochs',
    
    # Hyperparameters - Deployment
    'deploy_finetune_epochs', 'deploy_finetune_lr',
    
    # Training Metrics
    'best_epoch', 'best_val_acc', 'final_train_acc', 'final_train_loss',
    
    # Test Performance
    'test_acc', 'test_auc', 'test_precision', 'test_recall', 'test_f1',
    
    # Model Size Metrics
    'original_params_m', 'deploy_params_m', 'param_reduction_ratio',
    'original_size_mb', 'deploy_size_mb', 'size_reduction_ratio',
    'original_flops_g', 'deploy_flops_g', 'flops_reduction_ratio',
    
    # Architecture Details (Deployment)
    'original_embed_dim', 'deploy_embed_dim', 'embed_retention_ratio',
    'original_num_heads', 'deploy_avg_heads', 'head_retention_ratio',
    'original_mlp_dim', 'deploy_avg_mlp_dim', 'mlp_retention_ratio',
    
    # Sparsity Achieved
    'target_sparsity', 'actual_sparsity', 'sparsity_gap',
    
    # Gate Statistics (Final)
    'attn_gates_mean', 'attn_gates_std', 'attn_gates_zeros_pct',
    'mlp_gates_mean', 'mlp_gates_std', 'mlp_gates_zeros_pct',
    'res_gates_mean', 'res_gates_std', 'res_gates_zeros_pct',
    
    # Energy Metrics
    'training_energy_kwh', 'inference_energy_kwh', 'total_energy_kwh',
    
    # Accuracy Comparisons
    'acc_vs_baseline_absolute', 'acc_vs_baseline_relative_pct',
    
    # Training Stability
    'val_acc_std_dev', 'val_acc_range',
    
    # Timing
    'training_time_minutes', 'deploy_finetune_time_minutes', 'total_time_minutes',
    
    # Status
    'completed_successfully', 'error_message', 'warning_flags'
]

# Per-epoch history columns (separate CSV)
EPOCH_HISTORY_COLUMNS = [
    'config_name', 'trial', 'epoch', 'phase',
    'train_loss', 'train_acc', 'val_acc', 'test_acc',
    'learning_rate', 'gate_l1_loss',
    'attn_gates_active_pct', 'mlp_gates_active_pct', 'res_gates_active_pct',
    'params_m', 'was_prune_epoch'
]

# Error log columns
ERROR_LOG_COLUMNS = [
    'config_name', 'trial', 'dataset', 'phase',
    'error_type', 'error_message', 'traceback',
    'timestamp', 'partial_results_saved'
]

# ==================== ERROR HANDLING & LOGGING ====================

class ExperimentLogger:
    """Comprehensive logging for hyperparameter tuning"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.results = []
        self.epoch_history = []
        self.errors = []
        self.baseline_metrics = {}
        
    def log_result(self, result_dict):
        """Log a completed experiment result"""
        self.results.append(result_dict)
        self._save_incremental()
    
    def log_epoch(self, epoch_dict):
        """Log per-epoch metrics"""
        self.epoch_history.append(epoch_dict)
    
    def log_error(self, error_dict):
        """Log an error"""
        self.errors.append(error_dict)
        self._save_errors()
    
    def set_baseline(self, dataset, metrics):
        """Store baseline metrics for comparison"""
        self.baseline_metrics[dataset] = metrics
    
    def _save_incremental(self):
        """Save results incrementally (don't lose data on crash)"""
        if len(self.results) > 0:
            df = pd.DataFrame(self.results)
            df.to_csv(os.path.join(self.save_dir, 'results_incremental.csv'), index=False)
        
        if len(self.epoch_history) > 0:
            df_epochs = pd.DataFrame(self.epoch_history)
            df_epochs.to_csv(os.path.join(self.save_dir, 'epoch_history_incremental.csv'), index=False)
    
    def _save_errors(self):
        """Save error log"""
        if len(self.errors) > 0:
            df_errors = pd.DataFrame(self.errors)
            df_errors.to_csv(os.path.join(self.save_dir, 'error_log.csv'), index=False)
    
    def save_final_results(self):
        """Save final comprehensive results"""
        # Main results
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.save_dir, 'all_results_final.csv'), index=False)
        
        # Epoch history
        if len(self.epoch_history) > 0:
            df_epochs = pd.DataFrame(self.epoch_history)
            df_epochs.to_csv(os.path.join(self.save_dir, 'epoch_history_final.csv'), index=False)
        
        # Summary statistics by config
        summary = self._compute_summary_statistics(df)
        summary.to_csv(os.path.join(self.save_dir, 'summary_statistics.csv'), index=False)
        
        # Comparison to baseline
        comparison = self._compute_baseline_comparison(df)
        comparison.to_csv(os.path.join(self.save_dir, 'baseline_comparison.csv'), index=False)
        
        print(f"\n✓ Final results saved to {self.save_dir}")
    
    def _compute_summary_statistics(self, df):
        """Compute mean/std for each config across trials"""
        summary_rows = []
        
        for config_name in df['config_name'].unique():
            config_df = df[df['config_name'] == config_name]
            
            summary = {
                'config_name': config_name,
                'config_description': config_df.iloc[0]['config_description'],
                'num_trials': len(config_df),
                'num_successful': config_df['completed_successfully'].sum(),
            }
            
            # Compute statistics for key metrics
            for metric in ['test_acc', 'test_precision', 'deploy_params_m', 
                          'deploy_flops_g', 'param_reduction_ratio', 'flops_reduction_ratio',
                          'inference_energy_kwh', 'training_time_minutes']:
                if metric in config_df.columns:
                    summary[f'{metric}_mean'] = config_df[metric].mean()
                    summary[f'{metric}_std'] = config_df[metric].std()
                    summary[f'{metric}_min'] = config_df[metric].min()
                    summary[f'{metric}_max'] = config_df[metric].max()
            
            summary_rows.append(summary)
        
        return pd.DataFrame(summary_rows)
    
    def _compute_baseline_comparison(self, df):
        """Compare each config to baseline"""
        comparison_rows = []
        
        for dataset, baseline in self.baseline_metrics.items():
            dataset_df = df[df['dataset'] == dataset]
            
            for config_name in dataset_df['config_name'].unique():
                config_df = dataset_df[dataset_df['config_name'] == config_name]
                
                comparison = {
                    'config_name': config_name,
                    'dataset': dataset,
                    'baseline_acc': baseline.get('test_acc', float('nan')),
                    'config_acc_mean': config_df['test_acc'].mean(),
                    'acc_drop_absolute': baseline.get('test_acc', 0) - config_df['test_acc'].mean(),
                    'acc_drop_relative_pct': (baseline.get('test_acc', 0) - config_df['test_acc'].mean()) / baseline.get('test_acc', 1) * 100,
                    'baseline_params_m': baseline.get('params_m', float('nan')),
                    'config_params_m_mean': config_df['deploy_params_m'].mean(),
                    'param_reduction_pct': (1 - config_df['deploy_params_m'].mean() / baseline.get('params_m', 1)) * 100,
                    'baseline_flops_g': baseline.get('flops_g', float('nan')),
                    'config_flops_g_mean': config_df['deploy_flops_g'].mean(),
                    'flops_reduction_pct': (1 - config_df['deploy_flops_g'].mean() / baseline.get('flops_g', 1)) * 100,
                }
                
                comparison_rows.append(comparison)
        
        return pd.DataFrame(comparison_rows)

def run_config_with_error_handling(config, dataset_name, train_loader, val_loader, 
                                   test_loader, num_classes, save_dir, trial_num, 
                                   logger, phase):
    """
    Run a single configuration with comprehensive error handling
    Returns: (success, result_dict, error_dict)
    """
    import traceback
    
    start_time = time.time()
    error_dict = None
    result_dict = None
    
    try:
        print(f"\n{'='*100}")
        print(f"RUNNING: {config['name']} - Trial {trial_num} - {dataset_name}")
        print(f"{'='*100}")
        
        # TODO: Call the actual training function here
        # This will be implemented with your original training code
        # result_dict = train_progressive_with_config(config, dataset_name, ...)
        
        # For now, placeholder
        result_dict = {
            'config_name': config['name'],
            'config_description': config['description'],
            'trial': trial_num,
            'dataset': dataset_name,
            'phase': phase,
            'completed_successfully': True,
            'error_message': None,
            'warning_flags': '',
            # All hyperparameters
            **{k.lower(): v for k, v in config.items() if k.isupper()},
            # Placeholder metrics (will be filled by actual training)
            'test_acc': 0.0,
            'training_time_minutes': (time.time() - start_time) / 60
        }
        
        print(f"✓ Config {config['name']} completed successfully")
        return True, result_dict, None
        
    except RuntimeError as e:
        # CUDA out of memory or other runtime errors
        error_type = 'RuntimeError'
        error_msg = str(e)
        tb = traceback.format_exc()
        
        print(f"\n{'!'*100}")
        print(f"ERROR: {error_type} in config {config['name']}")
        print(f"Message: {error_msg}")
        print(f"{'!'*100}")
        
        error_dict = {
            'config_name': config['name'],
            'trial': trial_num,
            'dataset': dataset_name,
            'phase': phase,
            'error_type': error_type,
            'error_message': error_msg[:500],  # Truncate long messages
            'traceback': tb[:1000],
            'timestamp': datetime.now().isoformat(),
            'partial_results_saved': False
        }
        
        # Create minimal result dict for logging
        result_dict = {
            'config_name': config['name'],
            'config_description': config['description'],
            'trial': trial_num,
            'dataset': dataset_name,
            'phase': phase,
            'completed_successfully': False,
            'error_message': error_msg[:200],
            'warning_flags': error_type,
            **{k.lower(): v for k, v in config.items() if k.isupper()},
            'test_acc': float('nan'),
            'training_time_minutes': (time.time() - start_time) / 60
        }
        
        return False, result_dict, error_dict
        
    except Exception as e:
        # Any other error
        error_type = type(e).__name__
        error_msg = str(e)
        tb = traceback.format_exc()
        
        print(f"\n{'!'*100}")
        print(f"UNEXPECTED ERROR: {error_type} in config {config['name']}")
        print(f"Message: {error_msg}")
        print(f"{'!'*100}")
        
        error_dict = {
            'config_name': config['name'],
            'trial': trial_num,
            'dataset': dataset_name,
            'phase': phase,
            'error_type': error_type,
            'error_message': error_msg[:500],
            'traceback': tb[:1000],
            'timestamp': datetime.now().isoformat(),
            'partial_results_saved': False
        }
        
        result_dict = {
            'config_name': config['name'],
            'config_description': config['description'],
            'trial': trial_num,
            'dataset': dataset_name,
            'phase': phase,
            'completed_successfully': False,
            'error_message': error_msg[:200],
            'warning_flags': error_type,
            **{k.lower(): v for k, v in config.items() if k.isupper()},
            'test_acc': float('nan'),
            'training_time_minutes': (time.time() - start_time) / 60
        }
        
        return False, result_dict, error_dict
    
    finally:
        # Always clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        print("✓ Memory cleaned up")

# ==================== MAIN EXECUTION ====================

def main():
    print("\n" + "="*100)
    print("VISION TRANSFORMER PROGRESSIVE PRUNING - HYPERPARAMETER TUNING")
    print("="*100)
    print(f"\nDevice: {DEVICE}")
    print(f"Save directory: {SAVE_DIR}")
    print(f"Number of configurations: {len(CONFIGS)}")
    print(f"\nExecution Plan:")
    print(f"  Phase 1: All {len(CONFIGS)} configs × 1 trial × bloodmnist")
    print(f"  Phase 2: Top 3 configs × 3 trials × bloodmnist") 
    print(f"  Phase 3: Best config × 3 trials × all 3 datasets")
    
    # Print all configurations
    print("\n" + "="*80)
    print("CONFIGURATIONS TO TEST")
    print("="*80)
    for i, cfg in enumerate(CONFIGS, 1):
        print(f"\n{i}. {cfg['name']}")
        print(f"   {cfg['description']}")
        print(f"   Initial LR: {cfg['INITIAL_LR']:.6f}, "
              f"LR Reduction: {cfg['LR_REDUCTION_AFTER_PRUNE']:.2f}, "
              f"Gate L1: {cfg['GATE_L1_WEIGHT']:.6f}")
        print(f"   Sparsity: {cfg['SPARSITY_SCHEDULE']}, "
              f"Deploy FT: {cfg['DEPLOY_FINETUNE_EPOCHS']} epochs @ {cfg['DEPLOY_FINETUNE_LR']:.6f}")
    
    print("\n" + "="*80)
    print("OUTPUT SPECIFICATIONS")
    print("="*80)
    print(f"Main results CSV will contain {len(OUTPUT_COLUMNS)} columns:")
    print(f"  - Identification: config_name, trial, dataset, phase")
    print(f"  - Hyperparameters: all LR, regularization, pruning settings")
    print(f"  - Performance: accuracy, precision, recall, F1, AUC")
    print(f"  - Compression: params, FLOPs, model size (original & deployment)")
    print(f"  - Architecture: embed_dim, num_heads, MLP dims (retention ratios)")
    print(f"  - Gate statistics: mean, std, zeros% for each gate type")
    print(f"  - Energy: training and inference energy consumption")
    print(f"  - Timing: training time, finetuning time, total time")
    print(f"  - Status: success flag, error messages, warnings")
    
    print(f"\nEpoch history CSV will track:")
    print(f"  - Per-epoch: train/val/test metrics")
    print(f"  - Learning rate and gate regularization loss")
    print(f"  - Gate sparsity evolution")
    print(f"  - Pruning event markers")
    
    print("\n" + "="*80)
    print("READY TO START HYPERPARAMETER TUNING")
    print("="*80)
    print("\nThis script will:")
    print("  1. Run ALL configs (no time limit)")
    print("  2. Handle errors gracefully and continue")
    print("  3. Save comprehensive results after each run")
    print("  4. Generate summary statistics and comparisons")
    print("\nTo complete this implementation, we need to:")
    print("  1. Integrate with your original training functions")
    print("  2. Add detailed metric collection during training")
    print("  3. Implement gate statistics tracking")
    print("  4. Add epoch-by-epoch logging")

if __name__ == "__main__":
    main()