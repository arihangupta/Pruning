import pandas as pd
import os
import medmnist
from medmnist import INFO

# Define paths
base_dir = "/home/arihangupta/Pruning/dinov2/Pruning/CNN_prune"
dataset_dir = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
datasets = {
    "bloodmnist": "bloodmnist_combined_pruning_kd_metrics_with_energy.csv",
    "dermamnist": "dermamnist_combined_pruning_kd_metrics_with_energy.csv",
    "pathmnist": "pathmnist_combined_pruning_kd_metrics_with_energy.csv"
}

# Energy column name in the CSV (adjust if different)
energy_column = "total_energy"  # Replace with the actual column name if different

# Function to get the number of training images for a MedMNIST dataset
def get_num_training_images(dataset_name):
    try:
        # Map directory names to MedMNIST dataset classes
        dataset_map = {
            "bloodmnist": medmnist.BloodMNIST,
            "dermamnist": medmnist.DermaMNIST,
            "pathmnist": medmnist.PathMNIST
        }
        
        if dataset_name not in dataset_map:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Load the dataset
        dataset_class = dataset_map[dataset_name]
        dataset = dataset_class(split="train", download=False, root=dataset_dir)
        
        # Get the number of training images
        num_images = len(dataset)
        return num_images
    
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {str(e)}")
        return None

# Process each dataset
for dataset, csv_file in datasets.items():
    csv_path = os.path.join(base_dir, dataset, csv_file)
    
    # Get the number of training images
    num_images = get_num_training_images(dataset)
    if num_images is None:
        print(f"Skipping {csv_path} due to dataset loading error.")
        continue
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if the energy column exists
        if energy_column not in df.columns:
            print(f"Error: Column '{energy_column}' not found in {csv_path}. Available columns: {df.columns}")
            continue
        
        # Calculate energy per image
        df["energy_per_image"] = df[energy_column] / num_images
        
        # Save the updated CSV
        output_path = csv_path
        df.to_csv(output_path, index=False)
        print(f"Updated {csv_path} with 'energy_per_image' column. Number of training images: {num_images}")
    
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")

print("Processing complete.")