import pandas as pd
import numpy as np
import os

# Define paths
base_dir = "/home/arihangupta/Pruning/dinov2/Pruning/CNN_prune"
dataset_dir = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
datasets = {
    "bloodmnist": {"csv_file": "bloodmnist_combined_pruning_kd_metrics_with_energy.csv", "npz_file": "bloodmnist_224.npz"},
    "dermamnist": {"csv_file": "dermamnist_combined_pruning_kd_metrics_with_energy.csv", "npz_file": "dermamnist_224.npz"},
    "pathmnist": {"csv_file": "pathmnist_combined_pruning_kd_metrics_with_energy.csv", "npz_file": "pathmnist_224.npz"}
}

# Energy column name in the CSV (adjust if different)
energy_column = "RetrainEnergy_kWh"  # Replace with the actual column name if different

# Function to get the number of training images from an .npz file
def get_num_training_images(npz_path):
    try:
        # Load the .npz file
        data = np.load(npz_path)
        
        # Check for the 'train_images' key
        if "train_images" not in data:
            raise ValueError(f"'train_images' key not found in {npz_path}. Available keys: {list(data.keys())}")
        
        # Get the number of training images
        num_images = len(data["train_images"])
        return num_images
    
    except Exception as e:
        print(f"Error loading {npz_path}: {str(e)}")
        return None

# Process each dataset
for dataset, info in datasets.items():
    csv_path = os.path.join(base_dir, dataset, info["csv_file"])
    npz_path = os.path.join(dataset_dir, info["npz_file"])
    
    # Get the number of training images
    num_images = get_num_training_images(npz_path)
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