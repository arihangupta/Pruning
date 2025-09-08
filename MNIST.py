import medmnist
from medmnist import DermaMNIST, PathMNIST, OCTMNIST, BloodMNIST, TissueMNIST
import numpy as np
import os
import shutil
from tqdm import tqdm
import tempfile
import gc

# -------------------------
# Config
# -------------------------
IMG_SIZE = 224
ROOT_RAW = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_raw"
ROOT_TRIM = "/home/arihangupta/Pruning/dinov2/Pruning/datasets"
os.makedirs(ROOT_RAW, exist_ok=True)
os.makedirs(ROOT_TRIM, exist_ok=True)
MAX_TOTAL = 20000
TRIM_SPLITS = {"train": 14000, "val": 2000, "test": 4000}
RNG_SEED = 42

# -------------------------
# Dataset registry
# -------------------------
datasets = {
    "DermaMNIST": DermaMNIST,
    "PathMNIST": PathMNIST,
    "OCTMNIST": OCTMNIST,
    "BloodMNIST": BloodMNIST,
    "TissueMNIST": TissueMNIST
}

# -------------------------
# Memory-efficient helper functions
# -------------------------
def efficient_random_indices(n_total, n_keep, rng):
    """Generate random indices more efficiently for large datasets."""
    if n_keep >= n_total:
        return list(range(n_total))
    
    # Use numpy's choice for better performance and memory efficiency
    return rng.choice(n_total, size=min(n_keep, n_total), replace=False).tolist()

def trim_split_chunked(images, labels, n_keep, chunk_size=1000, desc=""):
    """Trim a split in chunks to minimize peak memory usage."""
    indices = efficient_random_indices(len(images), n_keep, np.random.default_rng(RNG_SEED))
    
    # Sort indices for better memory access patterns
    indices = sorted(indices)
    
    # Pre-allocate result arrays
    result_images = np.empty((len(indices), *images.shape[1:]), dtype=images.dtype)
    result_labels = np.empty((len(indices),), dtype=labels.dtype)
    
    # Process in chunks
    for i in tqdm(range(0, len(indices), chunk_size), desc=f"Trimming {desc}", unit="chunk"):
        chunk_end = min(i + chunk_size, len(indices))
        chunk_indices = indices[i:chunk_end]
        
        # Load chunk from source
        chunk_images = images[chunk_indices]
        chunk_labels = labels[chunk_indices]
        
        # Copy to result
        result_images[i:chunk_end] = chunk_images
        result_labels[i:chunk_end] = chunk_labels
        
        # Force garbage collection for large chunks
        if chunk_size > 500:
            del chunk_images, chunk_labels
            gc.collect()
    
    return result_images, result_labels

def trim_split_streaming(images, labels, n_keep, desc=""):
    """Most memory-efficient: stream individual samples."""
    indices = efficient_random_indices(len(images), n_keep, np.random.default_rng(RNG_SEED))
    
    # Create temporary file for very large datasets
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
        tmp_images = []
        tmp_labels = []
        
        for idx in tqdm(indices, desc=f"Trimming {desc}", unit="img"):
            tmp_images.append(images[idx])
            tmp_labels.append(labels[idx])
            
            # Save to temp file in batches to avoid memory buildup
            if len(tmp_images) >= 1000:
                if not hasattr(trim_split_streaming, 'batch_count'):
                    trim_split_streaming.batch_count = 0
                
                # Convert to numpy arrays
                batch_images = np.array(tmp_images)
                batch_labels = np.array(tmp_labels)
                
                # Save batch (you could implement incremental saving here)
                # For now, we'll keep in memory but this shows the pattern
                if trim_split_streaming.batch_count == 0:
                    all_images = batch_images
                    all_labels = batch_labels
                else:
                    all_images = np.concatenate([all_images, batch_images])
                    all_labels = np.concatenate([all_labels, batch_labels])
                
                trim_split_streaming.batch_count += 1
                
                # Clear batch
                tmp_images.clear()
                tmp_labels.clear()
                del batch_images, batch_labels
                gc.collect()
        
        # Handle remaining samples
        if tmp_images:
            batch_images = np.array(tmp_images)
            batch_labels = np.array(tmp_labels)
            
            if 'all_images' in locals():
                all_images = np.concatenate([all_images, batch_images])
                all_labels = np.concatenate([all_labels, batch_labels])
            else:
                all_images = batch_images
                all_labels = batch_labels
    
    # Reset batch counter
    if hasattr(trim_split_streaming, 'batch_count'):
        delattr(trim_split_streaming, 'batch_count')
    
    return all_images, all_labels

def maybe_trim_dataset_efficient(name: str, npz_path: str, method="chunked"):
    """Memory-efficient dataset trimming with multiple strategies."""
    # Always copy original to raw folder
    raw_copy = os.path.join(ROOT_RAW, os.path.basename(npz_path))
    if not os.path.exists(raw_copy):
        shutil.copy(npz_path, raw_copy)
    
    print(f"Processing {name} at {npz_path} ...")
    
    # Use memory mapping to avoid loading everything at once
    data = np.load(npz_path, mmap_mode="r")
    
    train_images, train_labels = data["train_images"], data["train_labels"]
    val_images, val_labels = data["val_images"], data["val_labels"]
    test_images, test_labels = data["test_images"], data["test_labels"]
    
    total = len(train_images) + len(val_images) + len(test_images)
    print(f"   Original sizes → Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)} (Total={total})")
    
    final_path = os.path.join(ROOT_TRIM, os.path.basename(npz_path))
    
    if total <= MAX_TOTAL:
        print("   Dataset under limit, copying to trimmed folder.")
        if not os.path.exists(final_path):
            shutil.copy(npz_path, final_path)
        return final_path
    
    print(f"   Dataset too large, trimming to {MAX_TOTAL} samples using {method} method...")
    
    # Choose trimming method based on dataset size and available memory
    if method == "streaming":
        trim_func = trim_split_streaming
    else:  # chunked (default)
        trim_func = lambda imgs, lbls, n, desc: trim_split_chunked(imgs, lbls, n, chunk_size=500, desc=desc)
    
    # Process each split separately and save immediately to reduce peak memory
    temp_files = {}
    
    try:
        # Trim train split
        print("   Processing training split...")
        train_images_trim, train_labels_trim = trim_func(train_images, train_labels, TRIM_SPLITS["train"], "train")
        temp_files['train'] = (train_images_trim, train_labels_trim)
        
        # Force garbage collection
        gc.collect()
        
        # Trim validation split
        print("   Processing validation split...")
        val_images_trim, val_labels_trim = trim_func(val_images, val_labels, TRIM_SPLITS["val"], "val")
        temp_files['val'] = (val_images_trim, val_labels_trim)
        
        # Force garbage collection
        gc.collect()
        
        # Trim test split
        print("   Processing test split...")
        test_images_trim, test_labels_trim = trim_func(test_images, test_labels, TRIM_SPLITS["test"], "test")
        temp_files['test'] = (test_images_trim, test_labels_trim)
        
        # Save all splits to final file
        print("   Saving trimmed dataset...")
        np.savez_compressed(
            final_path,
            train_images=temp_files['train'][0], train_labels=temp_files['train'][1],
            val_images=temp_files['val'][0], val_labels=temp_files['val'][1],
            test_images=test_images_trim, test_labels=test_labels_trim
        )
        
        print(f"   Trimmed dataset saved to {final_path}")
        
    finally:
        # Clean up temporary data
        del temp_files
        gc.collect()
        
    return final_path

def get_memory_usage():
    """Get current memory usage (requires psutil: pip install psutil)."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        return "psutil not installed"

# -------------------------
# Main loop with memory monitoring
# -------------------------
print(f"Starting memory usage: {get_memory_usage():.1f} MB")

for name, cls in datasets.items():
    print(f"\nDownloading {name} (size={IMG_SIZE}) ...")
    print(f"Memory before processing: {get_memory_usage():.1f} MB")
    
    # Trigger medmnist download (ensures file exists)
    for split in ["train", "val", "test"]:
        cls(split=split, download=True, size=IMG_SIZE, root=ROOT_RAW)
    
    npz_path = os.path.join(ROOT_RAW, f"{name.lower()}_{IMG_SIZE}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Expected {npz_path} not found!")
    
    # Choose method based on your memory constraints
    # method="chunked" for moderate memory reduction
    # method="streaming" for maximum memory efficiency (slower)
    final_path = maybe_trim_dataset_efficient(name, npz_path, method="chunked")
    
    # Quick verification
    data = np.load(final_path, mmap_mode="r")
    img, label = data["train_images"][0], data["train_labels"][0]
    print(f"   Sample image shape: {img.shape}, Label: {label}")
    print(f"   Memory after processing: {get_memory_usage():.1f} MB")
    
    # Force cleanup between datasets
    del data
    gc.collect()

print(f"\nAll datasets processed. Final memory usage: {get_memory_usage():.1f} MB")