import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_output_directories(base_path):
    """Create output directories for saving plots"""
    output_path = Path(base_path) / 'visualization_outputs'
    datasets = ['bloodmnist', 'dermamnist', 'pathmnist']
    
    # Create main output directory
    output_path.mkdir(exist_ok=True)
    
    # Create subdirectories for each dataset
    for dataset in datasets:
        (output_path / dataset).mkdir(exist_ok=True)
    
    return output_path

def load_merged_data(merged_dir):
    """Load data from individual merged CSV files"""
    merged_path = Path(merged_dir)
    all_data = []
    
    # Look for CSV files in the merged directory
    csv_files = list(merged_path.glob('*_merged.csv'))
    print(f"Found CSV files: {[f.name for f in csv_files]}")
    
    for csv_file in csv_files:
        # Extract dataset name from filename
        dataset_name = csv_file.stem.replace('_merged', '')
        print(f"Loading {dataset_name} data from {csv_file.name}")
        
        df = pd.read_csv(csv_file)
        df['dataset'] = dataset_name
        all_data.append(df)
        print(f"  Loaded {len(df)} records for {dataset_name}")
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    print(f"\nTotal combined records: {len(combined_df)}")
    
    return combined_df

def get_method_color(method):
    """Get color for each method"""
    color_map = {
        'baseline': '#000000',  # Black
        'quantization': '#808080',  # Gray
        'regional_pruning': '#8B0000',  # Dark red
        'regional_pruning_fp16': '#FF6B6B',  # Light red
        'slim_kd': '#000080',  # Dark blue
        'slim_kd_fp16': '#6B6BFF',  # Light blue
        'hybrid_pruning': '#4B0082',  # Dark purple
        'hybrid_pruning_fp16': '#9370DB'  # Light purple
    }
    return color_map.get(method, '#333333')

def get_method_groups():
    """Define method groupings for plotting"""
    return [
        ['baseline', 'quantization'],
        ['slim_kd', 'slim_kd_fp16'], 
        ['regional_pruning', 'regional_pruning_fp16'],
        ['hybrid_pruning', 'hybrid_pruning_fp16']
    ]

def plot_grouped_bars(ax, data_dict, ylabel, title, show_values=False, y_limits=None):
    """Create grouped bar plot with proper spacing"""
    
    method_groups = get_method_groups()
    
    x_pos = 0
    x_ticks = []
    x_labels = []
    bar_width = 0.35
    group_spacing = 0.8
    
    for group in method_groups:
        group_start = x_pos
        group_bars = []
        
        for i, method in enumerate(group):
            if method in data_dict:
                value = data_dict[method]
                if value is not None and not pd.isna(value):
                    color = get_method_color(method)
                    bar = ax.bar(x_pos, value, bar_width, color=color, alpha=0.8)
                    group_bars.append(bar)
                    
                    # Add value labels for specific metrics
                    if show_values:
                        if value < 1:
                            ax.text(x_pos, value, f'{value:.3f}',
                                   ha='center', va='bottom', fontsize=9, fontweight='bold')
                        elif value < 10:
                            ax.text(x_pos, value, f'{value:.2f}',
                                   ha='center', va='bottom', fontsize=9, fontweight='bold')
                        else:
                            ax.text(x_pos, value, f'{value:.1f}',
                                   ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
                    x_pos += bar_width
        
        # Add group label at center
        if group_bars:
            group_center = group_start + (len(group_bars) - 1) * bar_width / 2
            x_ticks.append(group_center)
            
            # Create group label
            if group[0] == 'baseline':
                x_labels.append('Baseline Methods')
            elif group[0] == 'slim_kd':
                x_labels.append('Slim KD')
            elif group[0] == 'regional_pruning':
                x_labels.append('Regional Pruning')
            elif group[0] == 'hybrid_pruning':
                x_labels.append('Hybrid Pruning')
        
        x_pos += group_spacing
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.grid(False)
    
    # Set y-axis limits if provided
    if y_limits:
        ax.set_ylim(y_limits)

def check_if_values_same_across_batch_sizes(df, dataset, metric):
    """Check if metric values are the same across batch sizes for a dataset"""
    dataset_data = df[df['dataset'] == dataset]
    
    # Group by pruning method and check if values are the same across batch sizes
    same_values = True
    for method in dataset_data['pruning_method'].unique():
        method_data = dataset_data[dataset_data['pruning_method'] == method]
        if len(method_data[metric].unique()) > 1:
            same_values = False
            break
    
    return same_values

def get_y_limits_for_metric(dataset_data, metric_col):
    """Calculate appropriate y-limits for a metric across all batch sizes"""
    valid_values = dataset_data[metric_col].dropna()
    if valid_values.empty:
        return None
    
    min_val = valid_values.min()
    max_val = valid_values.max()
    
    # Add some padding (5% on each side)
    range_val = max_val - min_val
    if range_val == 0:
        # If all values are the same, add some padding around the value
        padding = abs(min_val) * 0.1 if min_val != 0 else 1
        return (min_val - padding, max_val + padding)
    else:
        padding = range_val * 0.05
        return (min_val - padding, max_val + padding)

def plot_metric_for_dataset(df, dataset, metric_dict, output_path):
    """Plot a metric for a dataset - single plot if values same across batch sizes, dual if different"""
    
    dataset_data = df[df['dataset'] == dataset].copy()
    if dataset_data.empty:
        print(f"    No data for {dataset}")
        return
    
    metric_col = metric_dict['column']
    
    # Check if metric exists in data
    if metric_col not in dataset_data.columns:
        print(f"    Column '{metric_col}' not found for {dataset}")
        return
    
    # Check if values are the same across batch sizes
    same_across_batches = check_if_values_same_across_batch_sizes(df, dataset, metric_col)
    
    if same_across_batches:
        # Single plot - use batch size 8 data (or first available batch size)
        available_batch_sizes = sorted(dataset_data['batch_size'].unique())
        data_subset = dataset_data[dataset_data['batch_size'] == available_batch_sizes[0]]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create data dictionary
        data_dict = {}
        for _, row in data_subset.iterrows():
            method = row['pruning_method']
            value = row[metric_col]
            if not pd.isna(value):
                data_dict[method] = value
        
        if not data_dict:
            print(f"    No valid data for {metric_col} in {dataset}")
            plt.close()
            return
        
        show_values = metric_dict.get('show_values', False)
        plot_grouped_bars(ax, data_dict, metric_dict['ylabel'], 
                         f"{dataset.replace('mnist', '').title()} - {metric_dict['title']}", 
                         show_values=show_values)
        
        plt.tight_layout()
        plt.savefig(output_path / dataset / f"{metric_col}_{dataset}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved single plot: {metric_col}_{dataset}.png")
        
    else:
        # Dual plot - separate for each batch size with SAME Y-AXIS SCALE
        batch_sizes = sorted(dataset_data['batch_size'].unique())
        
        if len(batch_sizes) < 2:
            print(f"    Only one batch size available for {dataset}, creating single plot")
            # Fall back to single plot
            data_subset = dataset_data[dataset_data['batch_size'] == batch_sizes[0]]
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            data_dict = {}
            for _, row in data_subset.iterrows():
                method = row['pruning_method']
                value = row[metric_col]
                if not pd.isna(value):
                    data_dict[method] = value
            
            if data_dict:
                show_values = metric_dict.get('show_values', False)
                plot_grouped_bars(ax, data_dict, metric_dict['ylabel'], 
                                 f"{dataset.replace('mnist', '').title()} - {metric_dict['title']}", 
                                 show_values=show_values)
                
                plt.tight_layout()
                plt.savefig(output_path / dataset / f"{metric_col}_{dataset}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    Saved single plot: {metric_col}_{dataset}.png")
            else:
                plt.close()
            return
        
        # Calculate y-limits across ALL batch sizes to ensure same scale
        y_limits = get_y_limits_for_metric(dataset_data, metric_col)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, batch_size in enumerate(batch_sizes[:2]):  # Only take first 2 batch sizes
            ax = axes[idx]
            data_subset = dataset_data[dataset_data['batch_size'] == batch_size]
            
            # Create data dictionary
            data_dict = {}
            for _, row in data_subset.iterrows():
                method = row['pruning_method']
                value = row[metric_col]
                if not pd.isna(value):
                    data_dict[method] = value
            
            if data_dict:
                show_values = metric_dict.get('show_values', False)
                plot_grouped_bars(ax, data_dict, metric_dict['ylabel'], 
                                 f"{metric_dict['title']} - Batch Size {batch_size}", 
                                 show_values=show_values, y_limits=y_limits)
            else:
                ax.set_title(f"No data for Batch Size {batch_size}")
                if y_limits:
                    ax.set_ylim(y_limits)
        
        plt.suptitle(f"{dataset.replace('mnist', '').title()} - {metric_dict['title']}", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / dataset / f"{metric_col}_{dataset}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved dual plot with matching scales: {metric_col}_{dataset}.png")

def calculate_adjusted_retrain_energy(df, dataset, method):
    """Calculate adjusted retrain energy for fp16 methods"""
    dataset_data = df[df['dataset'] == dataset]
    
    if method.endswith('_fp16'):
        base_method = method.replace('_fp16', '')
        
        # Get base method retrain energy
        base_data = dataset_data[dataset_data['pruning_method'] == base_method]
        fp16_data = dataset_data[dataset_data['pruning_method'] == method]
        
        if not base_data.empty and not fp16_data.empty:
            base_retrain = base_data['RetrainEnergy_kWh'].iloc[0]
            fp16_retrain = fp16_data['RetrainEnergy_kWh'].iloc[0]
            
            # Handle NaN values
            base_retrain = 0 if pd.isna(base_retrain) else base_retrain
            fp16_retrain = 0 if pd.isna(fp16_retrain) else fp16_retrain
            
            return base_retrain + fp16_retrain
        elif not fp16_data.empty:
            fp16_retrain = fp16_data['RetrainEnergy_kWh'].iloc[0]
            return 0 if pd.isna(fp16_retrain) else fp16_retrain
    else:
        # Regular method
        method_data = dataset_data[dataset_data['pruning_method'] == method]
        if not method_data.empty:
            retrain = method_data['RetrainEnergy_kWh'].iloc[0]
            return 0 if pd.isna(retrain) else retrain
    
    return 0

def plot_breakeven_analysis(df, dataset, output_path):
    """Create break-even analysis using RetrainEnergy_kWh as intercept and energy_kWh_per_image as slope"""
    
    print(f"    Creating break-even analysis for {dataset}...")
    
    # Use batch size 8 data for break-even analysis (or first available batch size)
    dataset_data = df[df['dataset'] == dataset].copy()
    
    if dataset_data.empty:
        print(f"    No data for {dataset}")
        return
    
    # Get available batch sizes and use batch size 8 if available, otherwise use first available
    available_batch_sizes = sorted(dataset_data['batch_size'].unique())
    preferred_batch_size = 8 if 8 in available_batch_sizes else available_batch_sizes[0]
    
    analysis_data = dataset_data[dataset_data['batch_size'] == preferred_batch_size].copy()
    print(f"    Using batch size {preferred_batch_size} for break-even analysis")
    
    # Get baseline energy per image
    baseline_data = analysis_data[analysis_data['pruning_method'] == 'baseline']
    if baseline_data.empty:
        print(f"    No baseline data for {dataset}")
        return
    
    baseline_energy_per_image = baseline_data['energy_kWh_per_image'].iloc[0]
    if pd.isna(baseline_energy_per_image):
        print(f"    Invalid baseline energy per image for {dataset}")
        return
    
    print(f"    Baseline energy per image: {baseline_energy_per_image:.8f} kWh")
    
    # Define variants (excluding baseline)
    variants = ['quantization', 'regional_pruning', 'regional_pruning_fp16', 
                'hybrid_pruning', 'hybrid_pruning_fp16', 'slim_kd', 'slim_kd_fp16']
    
    # Calculate break-even points
    break_evens = {}
    max_x = 0
    
    for variant in variants:
        variant_data = analysis_data[analysis_data['pruning_method'] == variant]
        if variant_data.empty:
            print(f"    No data for {variant}")
            continue
        
        # Get energy per image (slope)
        energy_per_image = variant_data['energy_kWh_per_image'].iloc[0]
        
        # Get adjusted retrain energy (intercept)
        retrain_energy = calculate_adjusted_retrain_energy(df, dataset, variant)
        
        if pd.isna(energy_per_image):
            print(f"    Invalid energy per image for {variant}")
            continue
        
        print(f"    {variant}: energy_per_image={energy_per_image:.8f}, retrain_energy={retrain_energy:.8f}")
        
        # Calculate break-even: retrain_energy / (baseline_energy - variant_energy)
        energy_saved_per_image = baseline_energy_per_image - energy_per_image
        
        if energy_saved_per_image > 0 and retrain_energy >= 0:
            if retrain_energy == 0:
                x_intersect = 0
            else:
                x_intersect = retrain_energy / energy_saved_per_image
            
            if x_intersect >= 0:
                break_evens[variant] = round(x_intersect)
                max_x = max(max_x, x_intersect)
                print(f"    {variant} break-even: {x_intersect:.0f} predictions")
            else:
                break_evens[variant] = 0
                print(f"    {variant} break-even set to 0 (negative)")
        else:
            break_evens[variant] = 0
            print(f"    {variant} break-even set to 0 (no savings or negative retrain)")
    
    if not break_evens:
        print(f"    No break-even data for {dataset}")
        return
    
    # Set x-range
    if max_x == 0:
        max_x = 10000
    x_max = 1.1 * max_x
    x = np.linspace(0, x_max, 100)
    
    # Create figure with plot and table
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot baseline line
    y_baseline = baseline_energy_per_image * x
    ax1.plot(x, y_baseline, label='Baseline', color='#000000', linewidth=3, linestyle='-')
    
    # Plot variant lines
    for variant in variants:
        if variant not in break_evens:
            continue
        
        variant_data = analysis_data[analysis_data['pruning_method'] == variant]
        if variant_data.empty:
            continue
        
        energy_per_image = variant_data['energy_kWh_per_image'].iloc[0]
        retrain_energy = calculate_adjusted_retrain_energy(df, dataset, variant)
        
        if pd.isna(energy_per_image):
            continue
        
        # Plot line: y = retrain_energy + (energy_per_image * x)
        y = retrain_energy + (energy_per_image * x)
        color = get_method_color(variant)
        line_style = '-' if not variant.endswith('_fp16') else '--'
        ax1.plot(x, y, color=color, linewidth=2, 
                label=variant.replace('_', ' ').title(), linestyle=line_style)
        
        # Plot intersection point
        if break_evens[variant] > 0:
            x_intersect = break_evens[variant]
            y_intersect = baseline_energy_per_image * x_intersect
            ax1.plot(x_intersect, y_intersect, 'o', color=color, markersize=8, 
                     markeredgecolor='black', markeredgewidth=1)
    
    # Customize plot
    ax1.set_title(f'{dataset.upper().replace("MNIST", "")} - Total Energy Usage vs Number of Predictions\n(Batch Size {preferred_batch_size})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Predictions', fontsize=12)
    ax1.set_ylabel('Total Energy Usage (kWh)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(False)
    
    # Create table
    ax2.axis('off')
    table_data = [['Variant', 'Break-even Predictions']]
    for variant in variants:
        if variant in break_evens:
            be = break_evens[variant]
            be_str = f"{be:,}" if isinstance(be, (int, float)) else str(be)
            table_data.append([variant.replace('_', ' ').title(), be_str])
    
    if len(table_data) > 1:
        table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style table
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f1f1f2')
    
    ax2.set_title('Break-even Analysis\n(Intersections with Baseline)', 
                  fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path / dataset / f"energy_breakeven_{dataset}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: energy_breakeven_{dataset}.png")

def main():
    # Set your data path here
    merged_dir = "/Users/arihangupta/Downloads/pruning_project_data/updated_exp/merged"
    base_path = "/Users/arihangupta/Downloads/pruning_project_data/updated_exp"
    
    # Create output directories
    output_path = create_output_directories(base_path)
    print(f"Created output directory: {output_path}")
    
    # Load data from merged CSV files
    print("Loading merged data...")
    df = load_merged_data(merged_dir)
    
    if df.empty:
        print("No data loaded! Check the merged directory path.")
        return
    
    datasets = df['dataset'].unique()
    print(f"Datasets: {datasets}")
    
    # Show data summary
    print(f"\nData summary:")
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]
        print(f"  {dataset}: {len(dataset_data)} records")
        print(f"    Batch sizes: {sorted(dataset_data['batch_size'].unique())}")
        print(f"    Methods: {sorted(dataset_data['pruning_method'].unique())}")
    
    # Define metrics to plot
    metrics_to_plot = [
        {'column': 'throughput_imgs_per_s', 'ylabel': 'Throughput (imgs/s)', 'title': 'Throughput'},
        {'column': 'median_batch_ms', 'ylabel': 'Median Batch Time (ms)', 'title': 'Median Batch Processing Time'},
        {'column': 'p50_ms', 'ylabel': 'P50 Time (ms)', 'title': 'P50 Processing Time'},
        {'column': 'p90_ms', 'ylabel': 'P90 Time (ms)', 'title': 'P90 Processing Time'},
        {'column': 'peak_gpu_mem_MB', 'ylabel': 'Peak GPU Memory (MB)', 'title': 'Peak GPU Memory Usage'},
        {'column': 'avg_power_W', 'ylabel': 'Average Power (W)', 'title': 'Average Power Consumption'},
        {'column': 'energy_kWh_total', 'ylabel': 'Total Energy (kWh)', 'title': 'Total Energy Consumption'},
        {'column': 'energy_kWh_per_batch', 'ylabel': 'Energy per Batch (kWh)', 'title': 'Energy per Batch'},
        {'column': 'energy_kWh_per_image', 'ylabel': 'Energy per Image (kWh)', 'title': 'Energy per Image'},
        {'column': 'emissions_kg_total', 'ylabel': 'Total Emissions (kg)', 'title': 'Total Emissions'},
        {'column': 'cpu_power_w', 'ylabel': 'CPU Power (W)', 'title': 'CPU Power Consumption'},
        {'column': 'gpu_power_w', 'ylabel': 'GPU Power (W)', 'title': 'GPU Power Consumption'},
        {'column': 'Acc', 'ylabel': 'Accuracy', 'title': 'Model Accuracy', 'show_values': True},
        {'column': 'AUC', 'ylabel': 'AUC', 'title': 'Area Under Curve', 'show_values': True},
        {'column': 'ModelSizeMB', 'ylabel': 'Model Size (MB)', 'title': 'Model Size', 'show_values': True},
        {'column': 'FLOPs_per_image', 'ylabel': 'FLOPs per Image', 'title': 'Computational Cost (FLOPs)'},
        {'column': 'FLOPs_M_per_image', 'ylabel': 'FLOPs per Image (M)', 'title': 'Computational Cost (M FLOPs)'},
        {'column': 'RetrainEnergy_kWh', 'ylabel': 'Retrain Energy (kWh)', 'title': 'Retraining Energy Cost'}
    ]
    
    # Process each dataset
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Processing {dataset.upper()}")
        print(f"{'='*50}")
        
        # Plot all metrics
        for metric in metrics_to_plot:
            plot_metric_for_dataset(df, dataset, metric, output_path)
        
        # Create break-even analysis
        plot_breakeven_analysis(df, dataset, output_path)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"All visualizations saved to: {output_path}")
    
    for dataset in datasets:
        dataset_dir = output_path / dataset
        if dataset_dir.exists():
            png_files = list(dataset_dir.glob('*.png'))
            print(f"\n{dataset.upper()}/ - {len(png_files)} images created")
            for png_file in sorted(png_files):
                print(f"  - {png_file.name}")

if __name__ == "__main__":
    main()