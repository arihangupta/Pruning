"""
Comprehensive visualization script with per-model-size baseline identification
Includes all metrics and break-even analysis
Each model size (Base, Small, Tiny) is compared against its own baseline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_output_directories(base_path):
    """Create output directories for saving plots"""
    output_path = Path(base_path) / 'visualization_outputs_complete'
    datasets = ['bloodmnist', 'dermamnist', 'pathmnist']
    
    output_path.mkdir(exist_ok=True)
    for dataset in datasets:
        (output_path / dataset).mkdir(exist_ok=True)
    
    return output_path


def load_merged_data(merged_dir):
    """Load data from individual merged CSV files"""
    merged_path = Path(merged_dir)
    all_data = []
    
    csv_files = list(merged_path.glob('*_merged_results.csv'))
    print(f"Found CSV files: {[f.name for f in csv_files]}")
    
    for csv_file in csv_files:
        dataset_name = csv_file.stem.replace('_merged_results', '')
        print(f"Loading {dataset_name} data from {csv_file.name}")
        
        df = pd.read_csv(csv_file)
        
        # Ensure numeric columns are properly typed
        numeric_cols = [
            'energy_kWh_per_image', 'energy_kWh_per_batch', 'energy_kWh_total',
            'throughput_imgs_per_s', 'median_batch_ms', 'p50_ms', 'p90_ms',
            'peak_gpu_mem_MB', 'avg_power_W', 'emissions_kg_total',
            'cpu_power_w', 'gpu_power_w', 'ram_power_w',
            'Acc', 'AUC', 'ModelSizeMB', 'FLOPs_per_image', 'FLOPs_M_per_image',
            'TrainingEnergy_kWh', 'InferenceTime_per_batch_s', 'PeakRAM_MB'
        ]
        
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'dataset' not in df.columns:
            df['dataset'] = dataset_name
        all_data.append(df)
        print(f"  Loaded {len(df)} records")
    
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    print(f"\nTotal combined records: {len(combined_df)}")
    return combined_df


def create_method_label(row):
    """
    Create display labels for each method variant.
    Only baseline_vit_base_patch16_224 is the true baseline.
    """
    method = row['pruning_method']
    precision = row['runtime_precision']
    variant = row['Variant']
    
    # THE ONLY TRUE BASELINE
    if variant == 'baseline_vit_base_patch16_224' and precision == 'fp32':
        return 'Baseline (Base)', 'baseline', 0
    
    # Pruned baseline models (treated as optimized methods)
    elif 'baseline_vit_small_patch16_224' in variant and method == 'baseline' and precision == 'fp32':
        return 'Pruned (Small)', 'pruned', 1
    elif 'baseline_vit_tiny_patch16_224' in variant and method == 'baseline' and precision == 'fp32':
        return 'Pruned (Tiny)', 'pruned', 2
    
    # Quantized versions
    elif variant == 'baseline_vit_base_patch16_224' and precision == 'amp':
        return 'Quantized (Base)', 'quantized', 3
    elif 'baseline_vit_small_patch16_224' in variant and method == 'quantization' and precision == 'amp':
        return 'Quantized (Small)', 'quantized', 4
    elif 'baseline_vit_tiny_patch16_224' in variant and method == 'quantization' and precision == 'amp':
        return 'Quantized (Tiny)', 'quantized', 5
    
    # Knowledge Distillation methods
    elif method == 'kd' and precision == 'fp32':
        if 'base' in variant and 'small' in variant:
            return 'KD (Base→Small)', 'kd', 6
        elif 'base' in variant and 'tiny' in variant:
            return 'KD (Base→Tiny)', 'kd', 7
        elif 'small' in variant and 'tiny' in variant:
            return 'KD (Small→Tiny)', 'kd', 8
        else:
            return 'KD', 'kd', 6
    
    # KD with quantization
    elif method == 'kd' and precision == 'amp':
        if 'base' in variant and 'small' in variant:
            return 'KD+Quant (Base→Small)', 'kd_quant', 9
        elif 'base' in variant and 'tiny' in variant:
            return 'KD+Quant (Base→Tiny)', 'kd_quant', 10
        elif 'small' in variant and 'tiny' in variant:
            return 'KD+Quant (Small→Tiny)', 'kd_quant', 11
        else:
            return 'KD+Quant', 'kd_quant', 9
    
    else:
        return f'{method} {precision}', 'other', 99


def get_method_color(label):
    """Get color for each method label"""
    color_map = {
        # THE true baseline - bright red
        'Baseline (Base)': '#FF0000',
        
        # Pruned models - shades of orange
        'Pruned (Small)': '#FF8C00',
        'Pruned (Tiny)': '#FFA500',
        
        # Quantized models - shades of green  
        'Quantized (Base)': '#006400',
        'Quantized (Small)': '#228B22',
        'Quantized (Tiny)': '#32CD32',
        
        # KD methods - shades of blue
        'KD (Base→Small)': '#00008B',
        'KD (Base→Tiny)': '#0000CD',
        'KD (Small→Tiny)': '#4169E1',
        
        # KD+Quantization - shades of purple
        'KD+Quant (Base→Small)': '#4B0082',
        'KD+Quant (Base→Tiny)': '#6A0DAD',
        'KD+Quant (Small→Tiny)': '#9370DB',
    }
    return color_map.get(label, '#333333')


def format_value_for_label(value, metric_name=None):
    """Format value for bar label based on its magnitude"""
    if pd.isna(value):
        return 'N/A'
    
    if abs(value) < 1e-5:
        return f'{value:.2e}'
    elif abs(value) < 1e-3:
        return f'{value:.1e}'
    elif abs(value) < 0.01:
        return f'{value:.4f}'
    elif abs(value) < 1:
        return f'{value:.3f}'
    elif abs(value) < 10:
        return f'{value:.2f}'
    elif abs(value) < 100:
        return f'{value:.1f}'
    elif abs(value) < 10000:
        return f'{value:.0f}'
    else:
        return f'{value:.2e}'


def plot_dual_batch_bars(axes, data_dict_bs8, data_dict_bs32, ylabel, title, 
                         show_values=True, y_limits=None, metric_col=None):
    """Create dual grouped bar plots with the true baseline highlighted"""
    
    legend_handles = []
    legend_labels = []
    
    for idx, (ax, data_dict, batch_size) in enumerate([(axes[0], data_dict_bs8, 8), 
                                                         (axes[1], data_dict_bs32, 32)]):
        if not data_dict:
            ax.set_title(f'Batch Size {batch_size} - No Data')
            if y_limits:
                ax.set_ylim(y_limits)
            continue
        
        # Group methods by category
        groups = {
            'Baseline': [],
            'Pruned': [],
            'Quantized': [],
            'KD': [],
            'KD+Quant': []
        }
        
        for method_label in data_dict.keys():
            if method_label == 'Baseline (Base)':
                groups['Baseline'].append(method_label)
            elif 'Pruned' in method_label:
                groups['Pruned'].append(method_label)
            elif 'Quantized' in method_label and 'KD' not in method_label:
                groups['Quantized'].append(method_label)
            elif 'KD+Quant' in method_label:
                groups['KD+Quant'].append(method_label)
            elif 'KD' in method_label:
                groups['KD'].append(method_label)
        
        x_pos = 0
        x_ticks = []
        x_labels = []
        bar_width = 0.35
        group_spacing = 0.8
        
        for group_name, group_methods in groups.items():
            if not group_methods:
                continue
                
            group_start = x_pos
            
            sorted_methods = sorted(group_methods, key=lambda x: (
                'Base' not in x,
                'Small' not in x,
                x
            ))
            
            for method in sorted_methods:
                value = data_dict[method]
                if value is not None and not pd.isna(value):
                    color = get_method_color(method)
                    
                    alpha = 1.0 if method == 'Baseline (Base)' else 0.85
                    linewidth = 2.0 if method == 'Baseline (Base)' else 0.5
                    
                    bar = ax.bar(x_pos, value, bar_width, color=color, alpha=alpha, 
                                label=method, edgecolor='black', linewidth=linewidth)
                    
                    if idx == 0:
                        legend_handles.append(bar[0])
                        legend_labels.append(method)
                    
                    if show_values:
                        label_text = format_value_for_label(value, metric_col)
                        fontsize = 8 if len(label_text) > 6 else 9
                        y_position = value * 1.02 if value > 0 else 0
                        fontweight = 'bold' if method == 'Baseline (Base)' else 'normal'
                        
                        ax.text(x_pos, y_position, label_text,
                               ha='center', va='bottom', fontsize=fontsize, 
                               fontweight=fontweight)
                    
                    x_pos += bar_width
            
            if x_pos > group_start:
                group_center = group_start + (x_pos - group_start - bar_width) / 2
                x_ticks.append(group_center)
                
                if group_name == 'Baseline':
                    x_labels.append('BASELINE\n(Reference)')
                else:
                    x_labels.append(group_name)
            
            x_pos += group_spacing
        
        # Add baseline reference line
        if 'Baseline (Base)' in data_dict:
            baseline_value = data_dict['Baseline (Base)']
            if baseline_value is not None and not pd.isna(baseline_value):
                ax.axhline(y=baseline_value, color='red', linestyle='--', 
                          alpha=0.3, linewidth=1.5, label='_nolegend_')
        
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'Batch Size {batch_size}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        if y_limits and y_limits[1] < 1e-5:
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        
        if y_limits:
            ax.set_ylim(y_limits)
    
    return legend_handles, legend_labels


def get_y_limits_for_metric(dataset_data, metric_col):
    """Calculate appropriate y-limits for a metric"""
    valid_values = dataset_data[metric_col].dropna()
    if valid_values.empty:
        return None
    
    min_val = valid_values.min()
    max_val = valid_values.max()
    
    if metric_col in ['Acc', 'AUC']:
        min_val = 0.5
        padding = (max_val - min_val) * 0.15
        return (min_val, max_val + padding)
    
    if max_val < 1e-5:
        range_val = max_val - min_val
        if range_val == 0:
            padding = abs(max_val) * 0.2
        else:
            padding = range_val * 0.15
        return (min_val - padding if min_val > 0 else 0, max_val + padding)
    
    range_val = max_val - min_val
    if range_val == 0:
        padding = abs(min_val) * 0.1 if min_val != 0 else 1
        return (min_val - padding, max_val + padding)
    else:
        padding = range_val * 0.15
        return (max(0, min_val - padding), max_val + padding)


def plot_metric_for_dataset(df, dataset, metric_dict, output_path):
    """Plot a metric for a dataset with dual subplots (or single if batch-invariant)"""
    
    dataset_data = df[df['dataset'] == dataset].copy()
    if dataset_data.empty:
        print(f"    No data for {dataset}")
        return
    
    metric_col = metric_dict['column']
    
    if metric_col not in dataset_data.columns:
        print(f"    Column '{metric_col}' not found for {dataset}")
        return
    
    # Add method labels
    dataset_data[['method_label', 'method_group', 'sort_order']] = dataset_data.apply(
        lambda row: pd.Series(create_method_label(row)), axis=1
    )
    
    # Check if metric is batch-invariant
    batch_invariant_metrics = ['Acc', 'AUC', 'ModelSizeMB', 'FLOPs_M_per_image', 'TrainingEnergy_kWh', 'PeakRAM_MB']
    is_batch_invariant = metric_col in batch_invariant_metrics
    
    # Check if values actually differ across batch sizes (with rounding tolerance)
    if is_batch_invariant:
        batch_sizes = sorted(dataset_data['batch_size'].unique())
        if len(batch_sizes) >= 2:
            # For each method, compare values between batch size 8 and 32
            for method_label in dataset_data['method_label'].unique():
                # Get data for this method at each batch size
                bs8_data = dataset_data[(dataset_data['method_label'] == method_label) & 
                                       (dataset_data['batch_size'] == batch_sizes[0])]
                bs32_data = dataset_data[(dataset_data['method_label'] == method_label) & 
                                        (dataset_data['batch_size'] == batch_sizes[1])]
                
                # Get the values
                if len(bs8_data) > 0 and len(bs32_data) > 0:
                    val_bs8 = bs8_data[metric_col].iloc[0]
                    val_bs32 = bs32_data[metric_col].iloc[0]
                    
                    # Skip if either is NaN
                    if pd.notna(val_bs8) and pd.notna(val_bs32):
                        # Round to 5 decimal places and compare
                        rounded_bs8 = round(val_bs8, 5)
                        rounded_bs32 = round(val_bs32, 5)
                        
                        if rounded_bs8 != rounded_bs32:
                            # Values differ across batch sizes for this method
                            print(f"    Note: {metric_col} differs for {method_label}: BS8={val_bs8:.6f}, BS32={val_bs32:.6f} - using dual plot")
                            is_batch_invariant = False
                            break
    
    if is_batch_invariant:
        # Single plot for batch-invariant metrics
        batch_sizes = sorted(dataset_data['batch_size'].unique())
        # Use first batch size (or any, since they should be the same)
        batch_data = dataset_data[dataset_data['batch_size'] == batch_sizes[0]]
        
        data_dict = {}
        for _, row in batch_data.iterrows():
            label = row['method_label']
            value = row[metric_col]
            if not pd.isna(value):
                data_dict[label] = value
        
        if not data_dict:
            print(f"    No data for {metric_col} in {dataset}")
            return
        
        y_limits = get_y_limits_for_metric(dataset_data, metric_col)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Group methods by category
        groups = {
            'Baseline': [],
            'Pruned': [],
            'Quantized': [],
            'KD': [],
            'KD+Quant': []
        }
        
        for method_label in data_dict.keys():
            if method_label == 'Baseline (Base)':
                groups['Baseline'].append(method_label)
            elif 'Pruned' in method_label:
                groups['Pruned'].append(method_label)
            elif 'Quantized' in method_label and 'KD' not in method_label:
                groups['Quantized'].append(method_label)
            elif 'KD+Quant' in method_label:
                groups['KD+Quant'].append(method_label)
            elif 'KD' in method_label:
                groups['KD'].append(method_label)
        
        x_pos = 0
        x_ticks = []
        x_labels = []
        bar_width = 0.5
        group_spacing = 1.0
        
        legend_handles = []
        legend_labels = []
        
        for group_name, group_methods in groups.items():
            if not group_methods:
                continue
                
            group_start = x_pos
            
            sorted_methods = sorted(group_methods, key=lambda x: (
                'Base' not in x,
                'Small' not in x,
                x
            ))
            
            for method in sorted_methods:
                value = data_dict[method]
                if value is not None and not pd.isna(value):
                    color = get_method_color(method)
                    
                    alpha = 1.0 if method == 'Baseline (Base)' else 0.85
                    linewidth = 2.0 if method == 'Baseline (Base)' else 0.5
                    
                    bar = ax.bar(x_pos, value, bar_width, color=color, alpha=alpha, 
                                label=method, edgecolor='black', linewidth=linewidth)
                    
                    legend_handles.append(bar[0])
                    legend_labels.append(method)
                    
                    if metric_dict.get('show_values', True):
                        label_text = format_value_for_label(value, metric_col)
                        fontsize = 8 if len(label_text) > 6 else 9
                        y_position = value * 1.02 if value > 0 else 0
                        fontweight = 'bold' if method == 'Baseline (Base)' else 'normal'
                        
                        ax.text(x_pos, y_position, label_text,
                               ha='center', va='bottom', fontsize=fontsize, 
                               fontweight=fontweight)
                    
                    x_pos += bar_width
            
            if x_pos > group_start:
                group_center = group_start + (x_pos - group_start - bar_width) / 2
                x_ticks.append(group_center)
                
                if group_name == 'Baseline':
                    x_labels.append('BASELINE\n(Reference)')
                else:
                    x_labels.append(group_name)
            
            x_pos += group_spacing
        
        # Add baseline reference line
        if 'Baseline (Base)' in data_dict:
            baseline_value = data_dict['Baseline (Base)']
            if baseline_value is not None and not pd.isna(baseline_value):
                ax.axhline(y=baseline_value, color='red', linestyle='--', 
                          alpha=0.3, linewidth=1.5, label='_nolegend_')
        
        ax.set_ylabel(metric_dict['ylabel'], fontsize=11)
        ax.set_title(f"{dataset.replace('mnist', '').title()} - {metric_dict['title']}", 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        if y_limits and y_limits[1] < 1e-5:
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        
        if y_limits:
            ax.set_ylim(y_limits)
        
        if legend_handles and legend_labels:
            baseline_idx = legend_labels.index('Baseline (Base)') if 'Baseline (Base)' in legend_labels else None
            if baseline_idx is not None:
                legend_handles = [legend_handles[baseline_idx]] + legend_handles[:baseline_idx] + legend_handles[baseline_idx+1:]
                legend_labels = [legend_labels[baseline_idx]] + legend_labels[:baseline_idx] + legend_labels[baseline_idx+1:]
            
            fig.legend(legend_handles, legend_labels, 
                      loc='lower center', 
                      ncol=min(6, len(legend_labels)),
                      fontsize=9,
                      bbox_to_anchor=(0.5, -0.08),
                      frameon=True,
                      fancybox=True,
                      shadow=True)
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        
    else:
        # Dual plot for batch-dependent metrics
        batch_sizes = sorted(dataset_data['batch_size'].unique())
        if len(batch_sizes) < 2:
            batch_sizes = batch_sizes + [None] * (2 - len(batch_sizes))
        
        y_limits = get_y_limits_for_metric(dataset_data, metric_col)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        
        data_dicts = []
        for batch_size in batch_sizes[:2]:
            if batch_size is not None:
                batch_data = dataset_data[dataset_data['batch_size'] == batch_size]
                data_dict = {}
                for _, row in batch_data.iterrows():
                    label = row['method_label']
                    value = row[metric_col]
                    if not pd.isna(value):
                        data_dict[label] = value
                data_dicts.append(data_dict)
            else:
                data_dicts.append({})
        
        legend_handles, legend_labels = None, None
        if data_dicts[0] or data_dicts[1]:
            show_values = metric_dict.get('show_values', True)
            legend_handles, legend_labels = plot_dual_batch_bars(
                axes, data_dicts[0], data_dicts[1], 
                metric_dict['ylabel'], metric_dict['title'],
                show_values=show_values, y_limits=y_limits, metric_col=metric_col
            )
        
        if legend_handles and legend_labels:
            baseline_idx = legend_labels.index('Baseline (Base)') if 'Baseline (Base)' in legend_labels else None
            if baseline_idx is not None:
                legend_handles = [legend_handles[baseline_idx]] + legend_handles[:baseline_idx] + legend_handles[baseline_idx+1:]
                legend_labels = [legend_labels[baseline_idx]] + legend_labels[:baseline_idx] + legend_labels[baseline_idx+1:]
            
            fig.legend(legend_handles, legend_labels, 
                      loc='lower center', 
                      ncol=min(6, len(legend_labels)),
                      fontsize=9,
                      bbox_to_anchor=(0.5, -0.08),
                      frameon=True,
                      fancybox=True,
                      shadow=True)
        
        plt.suptitle(f"{dataset.replace('mnist', '').title()} - {metric_dict['title']}", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    filename = f"{metric_col}_{dataset}.png"
    plt.savefig(output_path / dataset / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename}")


def plot_breakeven_analysis(df, dataset, output_path):
    """Create three separate break-even analyses (one per model size) with baseline slopes"""
    
    print(f"    Creating break-even analyses for {dataset}...")
    
    dataset_data = df[df['dataset'] == dataset].copy()
    
    if dataset_data.empty:
        print(f"    No data for {dataset}")
        return
    
    # Use batch size 8 only
    analysis_data = dataset_data[dataset_data['batch_size'] == 8].copy()
    
    if analysis_data.empty:
        print(f"    No batch size 8 data for {dataset}")
        return
    
    # Add method labels
    analysis_data[['method_label', 'method_group', 'sort_order']] = analysis_data.apply(
        lambda row: pd.Series(create_method_label(row)), axis=1
    )
    
    # Create separate plots for each model size (Base, Small, Tiny)
    model_sizes = ['Base', 'Small', 'Tiny']
    
    for model_size in model_sizes:
        print(f"\n    Processing {model_size} models...")
        
        # Get baseline for this size
        baseline_data = analysis_data[
            (analysis_data['pruning_method'] == 'baseline') & 
            (analysis_data['runtime_precision'] == 'fp32') &
            (analysis_data['Variant'].str.contains(model_size.lower(), case=False))
        ]
        
        if baseline_data.empty:
            print(f"      No baseline {model_size} data, skipping...")
            continue
        
        baseline_energy_per_image = baseline_data['energy_kWh_per_image'].iloc[0]
        if pd.isna(baseline_energy_per_image):
            print(f"      Invalid baseline energy per image, skipping...")
            continue
        
        print(f"      Baseline ({model_size}) energy per image: {baseline_energy_per_image:.8f} kWh")
        
        # Get optimized methods for this size
        size_methods = analysis_data[
            (analysis_data['Variant'].str.contains(model_size.lower(), case=False)) &
            ~((analysis_data['pruning_method'] == 'baseline') & 
              (analysis_data['runtime_precision'] == 'fp32'))
        ]
        
        if size_methods.empty:
            print(f"      No optimized methods for {model_size}, skipping...")
            continue
        
        # Calculate break-even for each method
        break_evens = {}
        max_x = 0
        methods_to_plot = []
        
        for _, row in size_methods.iterrows():
            label = row['method_label']
            energy_per_image = row['energy_kWh_per_image']
            training_energy = row.get('TrainingEnergy_kWh', 0)
            
            if pd.isna(energy_per_image):
                continue
            
            if pd.isna(training_energy):
                training_energy = 0
            
            print(f"      {label}: energy_per_image={energy_per_image:.8f}, training_energy={training_energy:.8f}")
            
            # Calculate break-even
            energy_saved_per_image = baseline_energy_per_image - energy_per_image
            
            if energy_saved_per_image > 0 and training_energy >= 0:
                if training_energy == 0:
                    x_intersect = 0
                else:
                    x_intersect = training_energy / energy_saved_per_image
                
                if x_intersect >= 0:
                    break_evens[label] = round(x_intersect)
                    max_x = max(max_x, x_intersect)
                    methods_to_plot.append(row)
                    print(f"      {label} break-even: {x_intersect:.0f} predictions")
            else:
                print(f"      {label}: uses more energy than baseline, skipping...")
        
        if not break_evens:
            print(f"      No break-even data for {model_size}, skipping...")
            continue
        
        # Set x-range - cap at 10 million for visualization
        MAX_PLOT_PREDICTIONS = 10_000_000
        
        # Find max_x for methods we'll actually plot
        max_x_for_plot = 0
        methods_within_range = []
        methods_beyond_range = []
        
        for label, be_value in break_evens.items():
            if be_value <= MAX_PLOT_PREDICTIONS:
                max_x_for_plot = max(max_x_for_plot, be_value)
                methods_within_range.append(label)
            else:
                methods_beyond_range.append(label)
        
        if max_x_for_plot == 0:
            max_x_for_plot = 10000
        
        # Set x_max to either 1.2x the highest break-even point or 10M, whichever is lower
        x_max = min(max_x_for_plot * 1.2, MAX_PLOT_PREDICTIONS)
        x = np.linspace(0, x_max, 100)
        
        if methods_beyond_range:
            print(f"      Note: {len(methods_beyond_range)} method(s) have break-even > 10M predictions")
            print(f"            These will be in the table but plot will focus on realistic range")
        
        # Create figure with plot and table
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Determine baseline label for this model size
        if model_size == 'Base':
            baseline_label = 'Baseline (Base)'
        elif model_size == 'Small':
            baseline_label = 'Pruned (Small)'
        else:  # Tiny
            baseline_label = 'Pruned (Tiny)'
        
        baseline_color = get_method_color(baseline_label)
        
        # Plot BASELINE slope first (as reference)
        y_baseline = baseline_energy_per_image * x
        ax1.plot(x, y_baseline, label=f'Baseline ({model_size})', 
                color=baseline_color, linewidth=3, linestyle='-', zorder=10)
        
        # Plot optimized method lines
        for row in methods_to_plot:
            label = row['method_label']
            
            if label not in break_evens:
                continue
            
            energy_per_image = row['energy_kWh_per_image']
            training_energy = row.get('TrainingEnergy_kWh', 0)
            
            if pd.isna(energy_per_image):
                continue
            if pd.isna(training_energy):
                training_energy = 0
            
            # Plot line: y = training_energy + (energy_per_image * x)
            y = training_energy + (energy_per_image * x)
            color = get_method_color(label)
            line_style = '--' if 'Quant' in label else '-'
            ax1.plot(x, y, color=color, linewidth=2.5, label=label, linestyle=line_style)
            
            # Mark the break-even point ONLY if it's within the plot range
            be_value = break_evens[label]
            if be_value > 0 and be_value <= x_max:
                x_intersect = be_value
                y_intersect = baseline_energy_per_image * x_intersect
                ax1.plot(x_intersect, y_intersect, 'o', color=color, markersize=10, 
                        markeredgecolor='black', markeredgewidth=1.5, zorder=5)
                
                # Add subtle vertical line at break-even point
                ax1.axvline(x=x_intersect, color=color, linestyle=':', alpha=0.3, linewidth=1)
        
        ax1.set_title(f'{dataset.upper().replace("MNIST", "")} - Total Energy vs Predictions\n{model_size} Models (Batch Size 8)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Predictions', fontsize=12)
        ax1.set_ylabel('Total Cumulative Energy (kWh)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True, shadow=True)
        ax1.grid(axis='both', alpha=0.3, linestyle='--')
        
        # Create table
        ax2.axis('off')
        table_data = [['Method', 'Break-even\nPredictions', 'Energy\nSavings']]
        
        for label in sorted(break_evens.keys(), key=lambda x: break_evens[x]):
            be = break_evens[label]
            
            # Format break-even value
            if be > 1_000_000_000:  # > 1 billion
                be_str = f"{be/1_000_000_000:.2f}B"
            elif be > 1_000_000:  # > 1 million
                be_str = f"{be/1_000_000:.2f}M"
            elif be > 10_000:  # > 10k
                be_str = f"{be/1_000:.1f}K"
            else:
                be_str = f"{be:,}"
            
            # Add indicator if beyond plot range
            if be > MAX_PLOT_PREDICTIONS:
                be_str = f"{be_str} *"
            
            # Calculate energy savings percentage
            method_row = [r for r in methods_to_plot if r['method_label'] == label][0]
            method_energy = method_row['energy_kWh_per_image']
            savings_pct = ((baseline_energy_per_image - method_energy) / baseline_energy_per_image) * 100
            savings_str = f"{savings_pct:.1f}%"
            
            table_data.append([label, be_str, savings_str])
        
        if len(table_data) > 1:
            table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                             cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            # Style table
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            for i in range(1, len(table_data)):
                for j in range(len(table_data[0])):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f1f1f2')
        
        ax2.set_title(f'Break-even Analysis - {model_size} Models\n(vs Baseline {model_size} FP32)', 
                     fontsize=12, fontweight='bold', pad=20)
        
        # Add footnote if any methods are beyond plot range
        if methods_beyond_range:
            footnote = "* Break-even > 10M predictions\n(included in table but beyond plot scale)"
            ax2.text(0.5, -0.05, footnote, transform=ax2.transAxes,
                    fontsize=9, style='italic', ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout()
        filename = f"energy_breakeven_{model_size.lower()}_{dataset}.png"
        plt.savefig(output_path / dataset / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      Saved: {filename}")


def main():
    # Set your data path here
    merged_dir = "/Users/arihangupta/Downloads/pruning_project_data/Vision/merged_results"
    base_path = "/Users/arihangupta/Downloads/pruning_project_data/Vision"
    
    # Create output directories
    output_path = create_output_directories(base_path)
    print(f"Created output directory: {output_path}\n")
    
    # Load data
    print("Loading merged data...")
    df = load_merged_data(merged_dir)
    
    if df.empty:
        print("No data loaded! Check the merged directory path.")
        return
    
    # Verify we have the true baseline
    baseline_check = df[df['Variant'] == 'baseline_vit_base_patch16_224']
    if baseline_check.empty:
        print("WARNING: No baseline_vit_base_patch16_224 found!")
    else:
        print(f"✓ Found {len(baseline_check)} baseline_vit_base_patch16_224 records\n")
    
    datasets = df['dataset'].unique()
    print(f"Datasets found: {datasets}\n")
    
    # Show data summary
    print("Data summary:")
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]
        print(f"  {dataset}: {len(dataset_data)} records")
        print(f"    Batch sizes: {sorted(dataset_data['batch_size'].unique())}")
        print(f"    Methods: {sorted(dataset_data['pruning_method'].unique())}")
        print(f"    Variants: {sorted(dataset_data['Variant'].unique())}")
    
    # Define ALL metrics to plot (comprehensive list)
    metrics_to_plot = [
        {'column': 'throughput_imgs_per_s', 'ylabel': 'Throughput (imgs/s)', 'title': 'Throughput', 'show_values': True},
        {'column': 'median_batch_ms', 'ylabel': 'Median Batch Time (ms)', 'title': 'Median Batch Processing Time', 'show_values': True},
        {'column': 'p50_ms', 'ylabel': 'P50 Time (ms)', 'title': 'P50 Processing Time', 'show_values': True},
        {'column': 'p90_ms', 'ylabel': 'P90 Time (ms)', 'title': 'P90 Processing Time', 'show_values': True},
        {'column': 'peak_gpu_mem_MB', 'ylabel': 'Peak GPU Memory (MB)', 'title': 'Peak GPU Memory Usage', 'show_values': True},
        {'column': 'avg_power_W', 'ylabel': 'Average Power (W)', 'title': 'Average Power Consumption', 'show_values': True},
        {'column': 'energy_kWh_total', 'ylabel': 'Total Energy (kWh)', 'title': 'Total Energy Consumption', 'show_values': True},
        {'column': 'energy_kWh_per_batch', 'ylabel': 'Energy per Batch (kWh)', 'title': 'Energy per Batch', 'show_values': True},
        {'column': 'energy_kWh_per_image', 'ylabel': 'Energy per Image (kWh)', 'title': 'Energy per Image', 'show_values': True},
        {'column': 'emissions_kg_total', 'ylabel': 'Total Emissions (kg CO₂)', 'title': 'Total CO₂ Emissions', 'show_values': True},
        {'column': 'cpu_power_w', 'ylabel': 'CPU Power (W)', 'title': 'CPU Power Consumption', 'show_values': True},
        {'column': 'gpu_power_w', 'ylabel': 'GPU Power (W)', 'title': 'GPU Power Consumption', 'show_values': True},
        {'column': 'ram_power_w', 'ylabel': 'RAM Power (W)', 'title': 'RAM Power Consumption', 'show_values': True},
        {'column': 'Acc', 'ylabel': 'Accuracy', 'title': 'Model Accuracy', 'show_values': True},
        {'column': 'AUC', 'ylabel': 'AUC', 'title': 'Area Under Curve (AUC)', 'show_values': True},
        {'column': 'ModelSizeMB', 'ylabel': 'Model Size (MB)', 'title': 'Model Size', 'show_values': True},
        {'column': 'FLOPs_M_per_image', 'ylabel': 'MFLOPs per Image', 'title': 'Computational Cost (MFLOPs)', 'show_values': True},
        {'column': 'TrainingEnergy_kWh', 'ylabel': 'Training Energy (kWh)', 'title': 'Training Energy Cost', 'show_values': True},
        {'column': 'InferenceTime_per_batch_s', 'ylabel': 'Inference Time (s)', 'title': 'Inference Time per Batch', 'show_values': True},
        {'column': 'PeakRAM_MB', 'ylabel': 'Peak RAM (MB)', 'title': 'Peak RAM Usage', 'show_values': True}
    ]
    
    # Process each dataset
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset.upper()}")
        print(f"{'='*80}")
        
        # Plot all metrics
        for metric in metrics_to_plot:
            plot_metric_for_dataset(df, dataset, metric, output_path)
        
        # Create break-even analysis
        plot_breakeven_analysis(df, dataset, output_path)
    
    # Summary
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"All visualizations saved to: {output_path}")
    print("\nNote: Bar charts show all methods compared against baseline_vit_base_patch16_224")
    print("      Break-even charts show each model size compared to its own baseline:")
    print("      - Base models vs baseline_vit_base_patch16_224")
    print("      - Small models vs baseline_vit_small_patch16_224")
    print("      - Tiny models vs baseline_vit_tiny_patch16_224")
    print("\nGenerated files per dataset:")
    
    for dataset in datasets:
        dataset_dir = output_path / dataset
        if dataset_dir.exists():
            png_files = list(dataset_dir.glob('*.png'))
            print(f"\n{dataset.upper()}/ - {len(png_files)} images created")


if __name__ == "__main__":
    main()