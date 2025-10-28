import os
import sys
import time
import math
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import timm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import csv
import shutil
from torchprofile import profile_macs
import psutil
import gc

# CodeCarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except Exception:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    print("Warning: codecarbon not available. Energy/emissions will be NaN.")

# -------------------------
# Config
# -------------------------
DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
BASELINE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/new_baseline"
SAVE_DIR_BASE = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/more_epochs"
EPOCHS_KD = 50  # Epochs for knowledge distillation
BATCH_SIZE = 64
LR_KD = 0.0005  # Learning rate for student
MIN_LR = 1e-6
WEIGHT_DECAY = 0.01
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EARLY_STOP_PATIENCE = 10

# Knowledge Distillation Parameters
TEMPERATURE = 4.0  # Temperature for softening probabilities
ALPHA = 0.7  # Weight for distillation loss (1-alpha for student loss)

# Benchmarking Parameters
TIMING_BATCHES = 100
WARMUP = 5
NUM_BASELINE_RUNS = 3
PREDICTION_IMAGES = 50

os.makedirs(SAVE_DIR_BASE, exist_ok=True)


# -------------------------
# Early Stopping Class
# -------------------------
class EarlyStopping:
    """Early stops training if validation accuracy doesn't improve after patience epochs."""
    def __init__(self, patience=10, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_acc = 0
        
    def __call__(self, val_acc):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.best_acc = val_acc
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_acc = val_acc
            self.counter = 0


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Memory monitoring
# -------------------------
def log_memory_usage(prefix=""):
    process = psutil.Process()
    mem_info = process.memory_info()
    gpu_mem = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    print(f"{prefix}Memory Usage: RSS={mem_info.rss/(1024**2):.2f}MB, GPU={gpu_mem:.2f}MB")

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# -------------------------
# Dataset utilities
# -------------------------
class NumpyMemmapDataset(Dataset):
    """Wraps a numpy array (H,W[,C]) and a label array."""
    def __init__(self, imgs_np, labels_np, img_size=224, is_train=False):
        self.imgs = imgs_np
        self.labels = labels_np
        self.img_size = img_size
        self.is_train = is_train

        if is_train:
            self.base_tfms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            self.base_tfms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = int(self.labels[idx])
        x = self.base_tfms(img)
        
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        
        x = self.normalize(x)
        return x, label


def load_dataset(npz_path: str):
    """Load dataset from NPZ file."""
    print(f"Loading {npz_path} ...")
    data = np.load(npz_path, mmap_mode="r")

    X_train = data["train_images"]
    y_train = data["train_labels"].flatten()
    X_val = data["val_images"]
    y_val = data["val_labels"].flatten()
    X_test = data["test_images"]
    y_test = data["test_labels"].flatten()

    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    print(f"Dataset sizes: train={n_train}, val={n_val}, test={n_test}")

    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE, is_train=True)
    val_ds = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE, is_train=False)
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    num_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    dataset_name = os.path.splitext(os.path.basename(npz_path))[0].replace('_224', '')
    
    return train_loader, val_loader, test_loader, num_classes, dataset_name


def specificity_per_class(conf_matrix):
    """Calculates specificity for each class."""
    specificity = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        if (tn + fp) > 0:
            specificity.append(tn / (tn + fp))
        else:
            specificity.append(0.0)
    return specificity


def evaluate_model(net, test_loader, device, use_amp=False):
    """Evaluate the model."""
    net.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        val_bar = tqdm(test_loader, file=sys.stdout, desc="Evaluating", leave=False)
        for inputs, targets in val_bar:
            inputs = inputs.to(device)
            
            if use_amp:
                with autocast():
                    outputs = net(inputs)
            else:
                outputs = net(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            predict_y = torch.max(probs, dim=1)[1]
            
            all_preds.extend(predict_y.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = sum([1 for i in range(len(all_preds)) if all_preds[i] == all_labels[i]]) / len(all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    specificity = specificity_per_class(conf_matrix)
    avg_specificity = sum(specificity) / len(specificity) if specificity else 0.0
    
    n_classes = len(conf_matrix)
    all_labels_one_hot = label_binarize(all_labels, classes=list(range(n_classes)))
    
    try:
        auc = roc_auc_score(all_labels_one_hot, all_probs, multi_class='ovr')
    except ValueError:
        auc = float('nan')
    
    metrics = {
        'acc': acc,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'specificity': avg_specificity,
        'f1': f1
    }
    
    return metrics


# -------------------------
# Benchmarking utilities
# -------------------------
def params_count(model):
    return sum(p.numel() for p in model.parameters())

def model_size_bytes(model):
    fd, tmp = tempfile.mkstemp(suffix=".pth")
    os.close(fd)
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp)
    os.remove(tmp)
    return size

def compute_flops(model):
    model.eval()
    try:
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
    except StopIteration:
        model_dtype = torch.float32
        model_device = DEVICE
    
    try:
        inputs = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(model_device)
        if model_dtype == torch.half:
            inputs = inputs.half()
        macs = profile_macs(model, inputs)
        flops = macs * 2
        return float(flops)
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        return float("nan")

def inference_time_per_batch(model, loader, warmup=WARMUP, timed=TIMING_BATCHES):
    model.eval()
    try:
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_device = DEVICE
        model_dtype = torch.float32
    
    use_cuda = model_device.type == "cuda"
    it = iter(loader)
    
    # Warmup
    try:
        for _ in range(warmup):
            imgs, _ = next(it)
            imgs = imgs.to(model_device)
            if model_dtype == torch.half:
                imgs = imgs.half()
            with torch.no_grad():
                _ = model(imgs)
            if use_cuda:
                torch.cuda.synchronize()
    except StopIteration:
        print("    Warning: Iterator exhausted during warmup, resetting iterator")
        it = iter(loader)
    
    # Timed run
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    batches_done = 0
    images_processed = 0
    try:
        for _ in range(timed):
            imgs, _ = next(it)
            imgs = imgs.to(model_device)
            if model_dtype == torch.half:
                imgs = imgs.half()
            with torch.no_grad():
                _ = model(imgs)
            if use_cuda:
                torch.cuda.synchronize()
            batches_done += 1
            images_processed += imgs.size(0)
    except StopIteration:
        print(f"    Warning: Only {batches_done} batches processed due to dataset size")
    
    elapsed = time.time() - start
    avg_batch = elapsed / max(1, batches_done)
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if use_cuda else params_count(model)*4.0/(1024**2)
    return avg_batch, peak_mb, images_processed


# -------------------------
# CodeCarbon helpers
# -------------------------
def start_tracker(save_dir: str, project_name: str, output_file: str="emissions.csv", measure_power_secs: int=30):
    if not CODECARBON_AVAILABLE:
        return None
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, output_file)
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
            print(f"  Removed existing {csv_path} to ensure fresh data")
        except Exception as e:
            print(f"  Warning: Failed to remove {csv_path}: {e}")
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=save_dir,
        output_file=output_file,
        measure_power_secs=measure_power_secs,
        save_to_file=True
    )
    tracker.start()
    return tracker

def _read_latest_tracker_row(save_dir: str, project_name: str):
    csv_path = os.path.join(save_dir, "emissions.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        df_match = df[df["project_name"] == project_name]
        if df_match.shape[0] == 0:
            return None
        return df_match.iloc[-1].to_dict()
    except Exception:
        return None

def stop_tracker_and_get_metrics(tracker, save_dir: str, project_name: str):
    if tracker is None:
        return {
            "emissions_kg": float("nan"),
            "energy_kwh": float("nan"),
            "cpu_power_w": float("nan"),
            "gpu_power_w": float("nan"),
            "ram_power_w": float("nan"),
            "raw_row": None
        }
    try:
        emissions_val = tracker.stop()
    except Exception as e:
        print(f"  Error stopping CodeCarbon tracker: {e}")
        emissions_val = None
    
    raw = _read_latest_tracker_row(save_dir, project_name)
    
    if raw is None:
        print(f"  Warning: No valid tracker data for {project_name}")
        return {
            "emissions_kg": float(emissions_val) if emissions_val is not None else float("nan"),
            "energy_kwh": float("nan"),
            "cpu_power_w": float("nan"),
            "gpu_power_w": float("nan"),
            "ram_power_w": float("nan"),
            "raw_row": None
        }
    
    energy_kwh = float(raw.get("energy_consumed", float("nan")))
    cpu_power = float(raw.get("cpu_power", float("nan")))
    gpu_power = float(raw.get("gpu_power", float("nan")))
    ram_power = float(raw.get("ram_power", float("nan")))
    emissions_kg = float(raw.get("emissions", float("nan"))) if raw.get("emissions") is not None else (
        float(emissions_val) if emissions_val is not None else float("nan")
    )
    
    # Clean up CSV
    csv_path = os.path.join(save_dir, "emissions.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df = df[df["project_name"] != project_name]
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"  Error cleaning emissions.csv: {e}")
    
    return {
        "emissions_kg": emissions_kg,
        "energy_kwh": energy_kwh,
        "cpu_power_w": cpu_power,
        "gpu_power_w": gpu_power,
        "ram_power_w": ram_power,
        "raw_row": raw
    }

def measure_prediction_energy(model, test_loader, save_dir, project_name, num_images=PREDICTION_IMAGES):
    if not CODECARBON_AVAILABLE:
        return float("nan"), float("nan")
    
    tracker = start_tracker(save_dir, project_name, measure_power_secs=30)
    model.eval()
    images_processed = 0
    it = iter(test_loader)
    
    try:
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_device = DEVICE
        model_dtype = torch.float32
    
    with torch.no_grad():
        while images_processed < num_images:
            try:
                imgs, _ = next(it)
                imgs = imgs.to(model_device)
                if model_dtype == torch.half:
                    imgs = imgs.half()
                batch_size = imgs.size(0)
                if images_processed + batch_size > num_images:
                    imgs = imgs[:num_images - images_processed]
                _ = model(imgs)
                if model_device.type == "cuda":
                    torch.cuda.synchronize()
                images_processed += imgs.size(0)
            except StopIteration:
                break
    
    metrics = stop_tracker_and_get_metrics(tracker, save_dir, project_name)
    energy_kwh = metrics["energy_kwh"]
    emissions_kg = metrics["emissions_kg"]
    energy_per_image_kwh = energy_kwh / images_processed if images_processed > 0 and not math.isnan(energy_kwh) else float("nan")
    
    print(f"  Prediction energy for {images_processed} images: total_kWh={energy_kwh}, per_image_kWh={energy_per_image_kwh}, emissions_kg={emissions_kg}")
    return energy_per_image_kwh, emissions_kg


def calculate_break_even_safe(retrain_energy_kwh, baseline_energy_per_pred_kwh, pruned_energy_per_pred_kwh):
    if (math.isnan(retrain_energy_kwh) or 
        math.isnan(baseline_energy_per_pred_kwh) or 
        math.isnan(pruned_energy_per_pred_kwh)):
        return float("nan")
    
    delta = baseline_energy_per_pred_kwh - pruned_energy_per_pred_kwh
    
    if delta <= 0:
        return float("inf")
    elif retrain_energy_kwh <= 0:
        return 0.0
    else:
        return retrain_energy_kwh / delta


def measure_baseline_energy_averaged(baseline, test_loader, save_dir, model_name, dataset_name):
    """Measure baseline energy with multiple runs and averaging."""
    energies_total = []
    energies_per_pred = []
    images_per_run = []
    emissions_per_run = []
    
    for run in range(NUM_BASELINE_RUNS):
        proj = f"{dataset_name}_{model_name}_baseline_inference_run{run}"
        tracker = start_tracker(save_dir, proj, measure_power_secs=30) if CODECARBON_AVAILABLE else None
        avg_time, _, images = inference_time_per_batch(baseline, test_loader, timed=TIMING_BATCHES)
        metrics = stop_tracker_and_get_metrics(tracker, save_dir, proj)
        energy_kwh = metrics["energy_kwh"]
        emissions_kg = metrics["emissions_kg"]
        
        if images > 0 and not math.isnan(energy_kwh):
            energies_total.append(energy_kwh)
            energies_per_pred.append(energy_kwh / images)
        if not math.isnan(emissions_kg):
            emissions_per_run.append(emissions_kg)
        images_per_run.append(images)
    
    avg_images = np.mean(images_per_run)
    baseline_energy_kwh = np.mean(energies_total) if len(energies_total) > 0 else float("nan")
    baseline_emissions_kg = np.mean(emissions_per_run) if len(emissions_per_run) > 0 else float("nan")
    baseline_energy_per_pred_kwh = np.mean(energies_per_pred) if len(energies_per_pred) > 0 else float("nan")
    
    print(f"Averaged baseline ({NUM_BASELINE_RUNS} runs): images={avg_images:.0f}, energy_kWh={baseline_energy_kwh}, emissions_kg={baseline_emissions_kg}, per_pred={baseline_energy_per_pred_kwh}")
    
    # Measure prediction energy
    baseline_pred_energy_per_image_kwh, baseline_pred_emissions_kg = measure_prediction_energy(
        baseline, test_loader, save_dir, f"{dataset_name}_{model_name}_baseline_pred_50images"
    )
    
    return baseline_energy_kwh, baseline_emissions_kg, baseline_energy_per_pred_kwh, float(avg_images), baseline_pred_energy_per_image_kwh


# -------------------------
# Quantization with AMP
# -------------------------
def quantize_model_amp(dataset_name, model_name, baseline_path, baseline_model, test_loader, num_classes, save_dir):
    """
    Apply Automatic Mixed Precision (AMP) quantization with full benchmarking.
    """
    print(f"\n{'='*100}")
    print(f"Quantizing {model_name} on {dataset_name} using AMP")
    print(f"{'='*100}")
    
    # Check if quantized model already exists
    save_path = os.path.join(save_dir, f'quantization_{model_name}_{dataset_name}_final.pth')
    if os.path.exists(save_path):
        print(f"⊗ Quantized model already exists at {save_path}")
        print(f"⊗ Skipping quantization for {model_name} on {dataset_name}")
        return None
    
    # Start conversion energy tracking
    conversion_proj = f"{dataset_name}_quantization_{model_name}_conversion"
    conversion_tracker = start_tracker(save_dir, conversion_proj, measure_power_secs=30)
    
    # Create quantized model (FP16)
    quantized_model = timm.create_model(model_name, pretrained=False, num_classes=num_classes).to(DEVICE)
    checkpoint = torch.load(baseline_path, map_location=DEVICE)
    quantized_model.load_state_dict(checkpoint['model'])
    quantized_model = quantized_model.half()  # Convert to FP16
    
    # Stop conversion tracking
    conversion_metrics = stop_tracker_and_get_metrics(conversion_tracker, save_dir, conversion_proj)
    conversion_energy_kwh = conversion_metrics["energy_kwh"]
    conversion_emissions_kg = conversion_metrics["emissions_kg"]
    
    print(f"⚡ Conversion energy: {conversion_energy_kwh:.6f} kWh")
    print(f"🌍 Conversion CO2: {conversion_emissions_kg:.6f} kg")
    
    # Evaluate quantized model
    print("Evaluating quantized model (FP16)...")
    quantized_metrics = evaluate_model(quantized_model, test_loader, DEVICE, use_amp=True)
    
    # Get baseline metrics for comparison
    print("Measuring baseline metrics...")
    baseline_metrics = evaluate_model(baseline_model, test_loader, DEVICE, use_amp=False)
    
    # Measure baseline energy (averaged)
    baseline_energy_kwh, baseline_emissions_kg, baseline_energy_per_pred_kwh, baseline_images, baseline_pred_energy_per_image_kwh = measure_baseline_energy_averaged(
        baseline_model, test_loader, save_dir, model_name, dataset_name
    )
    
    # Measure inference energy for quantized model
    quantized_inf_proj = f"{dataset_name}_quantization_{model_name}_inference"
    quantized_tracker = start_tracker(save_dir, quantized_inf_proj, measure_power_secs=30)
    inf_time, peak_ram, quantized_images = inference_time_per_batch(quantized_model, test_loader, timed=TIMING_BATCHES)
    quantized_inf_metrics = stop_tracker_and_get_metrics(quantized_tracker, save_dir, quantized_inf_proj)
    quantized_energy_kwh = quantized_inf_metrics["energy_kwh"]
    quantized_emissions_kg = quantized_inf_metrics["emissions_kg"]
    quantized_energy_per_pred_kwh = quantized_energy_kwh / quantized_images if quantized_images > 0 and not math.isnan(quantized_energy_kwh) else float("nan")
    
    print(f"⚡ Quantized inference: images={quantized_images}, energy_kWh={quantized_energy_kwh}, emissions_kg={quantized_emissions_kg}")
    
    # Measure prediction energy (50 images)
    pred_energy_proj = f"{dataset_name}_quantization_{model_name}_pred_50images"
    pred_energy_per_image_kwh, pred_emissions_kg = measure_prediction_energy(
        quantized_model, test_loader, save_dir, pred_energy_proj
    )
    
    # Calculate additional metrics
    baseline_params = params_count(baseline_model)
    quantized_params = params_count(quantized_model)
    
    baseline_size = os.path.getsize(baseline_path) / (1024 * 1024) if os.path.exists(baseline_path) else model_size_bytes(baseline_model) / (1024 * 1024)
    
    # Save quantized model
    state = {
        'model': quantized_model.state_dict(),
        'acc': quantized_metrics['acc'],
        'auc': quantized_metrics['auc'],
        'model_name': model_name,
        'dataset': dataset_name,
        'pruning_method': 'quantization'
    }
    torch.save(state, save_path)
    print(f"✓ Quantized model saved to {save_path}")
    
    quantized_size = os.path.getsize(save_path) / (1024 * 1024)
    compression_ratio = baseline_size / quantized_size if quantized_size > 0 else 1.0
    
    # FLOPs calculation
    flops = compute_flops(quantized_model)
    flops_m = flops / 1e6 if not math.isnan(flops) else float("nan")
    
    # Break-even calculation
    break_even = calculate_break_even_safe(conversion_energy_kwh, baseline_energy_per_pred_kwh, quantized_energy_per_pred_kwh)
    
    # Calculate drops
    acc_drop = (baseline_metrics['acc'] - quantized_metrics['acc']) * 100
    auc_drop = (baseline_metrics['auc'] - quantized_metrics['auc']) * 100
    
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS:")
    print(f"{'='*60}")
    print(f"Baseline  - Acc: {baseline_metrics['acc']:.4f}, AUC: {baseline_metrics['auc']:.4f}")
    print(f"Quantized - Acc: {quantized_metrics['acc']:.4f}, AUC: {quantized_metrics['auc']:.4f}")
    print(f"Drop      - Acc: {acc_drop:.2f}%, AUC: {auc_drop:.2f}%")
    print(f"Size      - Baseline: {baseline_size:.2f} MB, Quantized: {quantized_size:.2f} MB")
    print(f"Compression: {compression_ratio:.2f}x")
    print(f"Break-even: {break_even:.0f} predictions" if not math.isinf(break_even) else "Break-even: Never (no energy savings)")
    print(f"{'='*60}")
    
    # Prepare results row (single row with all metrics)
    results = {
        'Dataset': dataset_name,
        'Variant': f'quantization_{model_name}',
        'TeacherModel': model_name,
        'StudentModel': f'{model_name}_fp16',
        'Stage': 'final',
        'KeepRatio': 1.0,
        
        # Performance metrics
        'Acc': quantized_metrics['acc'],
        'AUC': quantized_metrics['auc'],
        'Precision': quantized_metrics['precision'],
        'Recall': quantized_metrics['recall'],
        'Specificity': quantized_metrics['specificity'],
        'F1': quantized_metrics['f1'],
        
        # Comparison metrics
        'BaselineAcc': baseline_metrics['acc'],
        'BaselineAUC': baseline_metrics['auc'],
        'AccDrop_percent': acc_drop,
        'AucDrop_percent': auc_drop,
        
        # Model characteristics
        'Params': quantized_params,
        'BaselineParams': baseline_params,
        'ParamReduction_percent': ((baseline_params - quantized_params) / baseline_params * 100) if baseline_params > 0 else 0,
        'ModelSizeMB': quantized_size,
        'BaselineSizeMB': baseline_size,
        'CompressionRatio': compression_ratio,
        'FLOPs_per_image': flops,
        'FLOPs_M_per_image': flops_m,
        
        # Timing & Resources
        'InferenceTime_per_batch_s': inf_time,
        'PeakRAM_MB': peak_ram,
        'ImagesProcessedDuringTiming': quantized_images,
        
        # Energy metrics - Conversion
        'ConversionEnergy_kWh': conversion_energy_kwh,
        'ConversionEmissions_kg': conversion_emissions_kg,
        
        # Energy metrics - Baseline
        'BaselineInferenceEnergy_kWh_total': baseline_energy_kwh,
        'BaselineEnergy_per_pred_kWh': baseline_energy_per_pred_kwh,
        'BaselineEmissions_kg_total': baseline_emissions_kg,
        'BaselinePredictionEnergy_per_image_kWh': baseline_pred_energy_per_image_kwh,
        
        # Energy metrics - Quantized
        'InferenceEnergy_kWh_total': quantized_energy_kwh,
        'Energy_per_pred_kWh': quantized_energy_per_pred_kwh,
        'Emissions_kg_total': quantized_emissions_kg,
        'PredictionEnergy_per_image_kWh': pred_energy_per_image_kwh,
        
        # Break-even
        'BreakEvenPredictions': break_even,
        
        # Path
        'ModelPath': save_path
    }
    
    return results


# -------------------------
# Knowledge Distillation
# -------------------------
class DistillationLoss(nn.Module):
    """
    Distillation loss combines:
    1. Distillation loss (soft targets from teacher)
    2. Student loss (hard targets from ground truth)
    """
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets (distillation loss)
        soft_targets = torch.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = torch.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = self.kl_loss(soft_student, soft_targets) * (self.temperature ** 2)
        
        # Hard targets (student loss)
        student_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        return total_loss


def train_epoch_kd(student, teacher, train_loader, optimizer, scheduler, criterion, device):
    """Train student for one epoch with knowledge distillation."""
    student.train()
    teacher.eval()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout, desc="Training (KD)", leave=False)
    
    for step, (images, labels) in enumerate(train_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Get teacher predictions (no gradient)
        with torch.no_grad():
            teacher_logits = teacher(images)
        
        # Get student predictions
        student_logits = student(images)
        
        # Calculate distillation loss
        loss = criterion(student_logits, teacher_logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.3f}")
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def knowledge_distillation(dataset_name, teacher_model_name, student_model_name, 
                          teacher_path, teacher_model, train_loader, val_loader, test_loader, num_classes, save_dir):
    """
    Perform knowledge distillation from teacher to student with full benchmarking.
    """
    print(f"\n{'='*100}")
    print(f"Knowledge Distillation: {teacher_model_name} → {student_model_name} on {dataset_name}")
    print(f"{'='*100}")
    
    # Check if KD model already exists
    save_path = os.path.join(save_dir, f'kd_{teacher_model_name}_to_{student_model_name}_{dataset_name}_final.pth')
    if os.path.exists(save_path):
        print(f"⊗ KD model already exists at {save_path}")
        print(f"⊗ Skipping knowledge distillation for {teacher_model_name} → {student_model_name} on {dataset_name}")
        return None
    
    # Start training energy tracking
    training_proj = f"{dataset_name}_kd_{teacher_model_name}_to_{student_model_name}_training"
    training_tracker = start_tracker(save_dir, training_proj, measure_power_secs=30)
    
    # Load teacher model
    print(f"Loading teacher model: {teacher_model_name}")
    teacher = timm.create_model(teacher_model_name, pretrained=False, num_classes=num_classes).to(DEVICE)
    checkpoint = torch.load(teacher_path, map_location=DEVICE)
    teacher.load_state_dict(checkpoint['model'])
    teacher.eval()
    
    # Evaluate teacher
    print("Evaluating teacher model...")
    teacher_metrics = evaluate_model(teacher, test_loader, DEVICE)
    print(f"Teacher Test Accuracy: {teacher_metrics['acc']:.4f}, AUC: {teacher_metrics['auc']:.4f}")
    
    # Measure teacher energy (averaged)
    teacher_energy_kwh, teacher_emissions_kg, teacher_energy_per_pred_kwh, teacher_images, teacher_pred_energy_per_image_kwh = measure_baseline_energy_averaged(
        teacher, test_loader, save_dir, teacher_model_name, dataset_name
    )
    
    # Create student model (random initialization)
    print(f"Creating student model: {student_model_name} (random weights)")
    student = timm.create_model(student_model_name, pretrained=False, num_classes=num_classes).to(DEVICE)
    
    teacher_params = params_count(teacher)
    student_params = params_count(student)
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Parameter reduction: {(1 - student_params/teacher_params)*100:.2f}%")
    
    # Setup training
    criterion = DistillationLoss(temperature=TEMPERATURE, alpha=ALPHA)
    optimizer = optim.AdamW(student.parameters(), lr=LR_KD, weight_decay=WEIGHT_DECAY)
    
    total_steps = EPOCHS_KD * len(train_loader)
    warmup_steps = 3 * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, verbose=True)
    best_acc, best_auc = 0.0, 0.0
    
    # Training loop
    print(f"\nStarting knowledge distillation training for {EPOCHS_KD} epochs...")
    for epoch in range(EPOCHS_KD):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS_KD}]")
        train_loss = train_epoch_kd(student, teacher, train_loader, optimizer, scheduler, criterion, DEVICE)
        metrics = evaluate_model(student, val_loader, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val AUC: {metrics['auc']:.4f}, Val Acc: {metrics['acc']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        
        if metrics['acc'] > best_acc:
            print(f"\n✓ New best accuracy: {metrics['acc']:.4f} (previous: {best_acc:.4f})")
            best_acc, best_auc = metrics['acc'], metrics['auc']
            state = {
                'model': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'auc': best_auc,
                'epoch': epoch,
                'student_model': student_model_name,
                'teacher_model': teacher_model_name,
                'dataset': dataset_name,
                'pruning_method': 'knowledge_distillation'
            }
            torch.save(state, save_path)
            print(f"Model saved to {save_path}")
        
        early_stopping(metrics['acc'])
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    # Stop training energy tracking
    training_metrics = stop_tracker_and_get_metrics(training_tracker, save_dir, training_proj)
    training_energy_kwh = training_metrics["energy_kwh"]
    training_emissions_kg = training_metrics["emissions_kg"]
    
    print(f"\n⚡ Training energy consumed: {training_energy_kwh:.6f} kWh")
    print(f"🌍 Training CO2 emissions: {training_emissions_kg:.6f} kg")
    
    print(f"\nTraining completed! Best Acc: {best_acc:.4f}, AUC: {best_auc:.4f}")
    
    # Test set evaluation
    print(f"\nEvaluating student model on test set...")
    checkpoint = torch.load(save_path, map_location=DEVICE)
    student.load_state_dict(checkpoint['model'])
    test_metrics = evaluate_model(student, test_loader, DEVICE)
    
    # Measure inference energy for student
    student_inf_proj = f"{dataset_name}_kd_{teacher_model_name}_to_{student_model_name}_inference"
    student_tracker = start_tracker(save_dir, student_inf_proj, measure_power_secs=30)
    inf_time, peak_ram, student_images = inference_time_per_batch(student, test_loader, timed=TIMING_BATCHES)
    student_inf_metrics = stop_tracker_and_get_metrics(student_tracker, save_dir, student_inf_proj)
    student_energy_kwh = student_inf_metrics["energy_kwh"]
    student_emissions_kg = student_inf_metrics["emissions_kg"]
    student_energy_per_pred_kwh = student_energy_kwh / student_images if student_images > 0 and not math.isnan(student_energy_kwh) else float("nan")
    
    print(f"⚡ Student inference: images={student_images}, energy_kWh={student_energy_kwh}, emissions_kg={student_emissions_kg}")
    
    # Measure prediction energy (50 images)
    pred_energy_proj = f"{dataset_name}_kd_{teacher_model_name}_to_{student_model_name}_pred_50images"
    pred_energy_per_image_kwh, pred_emissions_kg = measure_prediction_energy(
        student, test_loader, save_dir, pred_energy_proj
    )
    
    # Calculate additional metrics
    teacher_size = os.path.getsize(teacher_path) / (1024 * 1024) if os.path.exists(teacher_path) else model_size_bytes(teacher) / (1024 * 1024)
    student_size = os.path.getsize(save_path) / (1024 * 1024)
    compression_ratio = teacher_size / student_size if student_size > 0 else 1.0
    
    # FLOPs calculation
    flops = compute_flops(student)
    flops_m = flops / 1e6 if not math.isnan(flops) else float("nan")
    
    # Break-even calculation
    break_even = calculate_break_even_safe(training_energy_kwh, teacher_energy_per_pred_kwh, student_energy_per_pred_kwh)
    
    # Calculate drops
    acc_drop = (teacher_metrics['acc'] - test_metrics['acc']) * 100
    auc_drop = (teacher_metrics['auc'] - test_metrics['auc']) * 100
    
    print(f"\nFinal Results:")
    print(f"Teacher Test Acc: {teacher_metrics['acc']:.4f}, AUC: {teacher_metrics['auc']:.4f}")
    print(f"Student Test Acc: {test_metrics['acc']:.4f}, AUC: {test_metrics['auc']:.4f}")
    print(f"Accuracy drop: {acc_drop:.2f}%")
    print(f"Break-even: {break_even:.0f} predictions" if not math.isinf(break_even) else "Break-even: Never (no energy savings)")
    
    # Prepare results row (single row with all metrics)
    results = {
        'Dataset': dataset_name,
        'Variant': f'kd_{teacher_model_name}_to_{student_model_name}',
        'TeacherModel': teacher_model_name,
        'StudentModel': student_model_name,
        'Stage': 'final',
        'KeepRatio': student_params / teacher_params if teacher_params > 0 else 1.0,
        
        # Performance metrics
        'Acc': test_metrics['acc'],
        'AUC': test_metrics['auc'],
        'Precision': test_metrics['precision'],
        'Recall': test_metrics['recall'],
        'Specificity': test_metrics['specificity'],
        'F1': test_metrics['f1'],
        
        # Teacher comparison
        'TeacherAcc': teacher_metrics['acc'],
        'TeacherAUC': teacher_metrics['auc'],
        'AccDrop_percent': acc_drop,
        'AucDrop_percent': auc_drop,
        
        # Model characteristics
        'Params': student_params,
        'TeacherParams': teacher_params,
        'ParamReduction_percent': (1 - student_params/teacher_params)*100,
        'ModelSizeMB': student_size,
        'TeacherSizeMB': teacher_size,
        'CompressionRatio': compression_ratio,
        'FLOPs_per_image': flops,
        'FLOPs_M_per_image': flops_m,
        
        # Timing & Resources
        'InferenceTime_per_batch_s': inf_time,
        'PeakRAM_MB': peak_ram,
        'ImagesProcessedDuringTiming': student_images,
        
        # Energy metrics - Training
        'TrainingEnergy_kWh': training_energy_kwh,
        'TrainingEmissions_kg': training_emissions_kg,
        
        # Energy metrics - Teacher
        'TeacherInferenceEnergy_kWh_total': teacher_energy_kwh,
        'TeacherEnergy_per_pred_kWh': teacher_energy_per_pred_kwh,
        'TeacherEmissions_kg_total': teacher_emissions_kg,
        'TeacherPredictionEnergy_per_image_kWh': teacher_pred_energy_per_image_kwh,
        
        # Energy metrics - Student
        'InferenceEnergy_kWh_total': student_energy_kwh,
        'Energy_per_pred_kWh': student_energy_per_pred_kwh,
        'Emissions_kg_total': student_emissions_kg,
        'PredictionEnergy_per_image_kWh': pred_energy_per_image_kwh,
        
        # Break-even
        'BreakEvenPredictions': break_even,
        
        # Path
        'ModelPath': save_path
    }
    
    return results


def evaluate_baseline_model(model_name, baseline_path, test_loader, num_classes, dataset_name, save_dir):
    """
    Evaluate a baseline model with full benchmarking.
    """
    print(f"\n{'='*100}")
    print(f"Evaluating Baseline: {model_name} on {dataset_name}")
    print(f"{'='*100}")
    
    # Load baseline model
    print(f"Loading baseline model from {baseline_path}")
    net = timm.create_model(model_name, pretrained=False, num_classes=num_classes).to(DEVICE)
    checkpoint = torch.load(baseline_path, map_location=DEVICE)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    
    # Evaluate
    print("Evaluating baseline model...")
    metrics = evaluate_model(net, test_loader, DEVICE, use_amp=False)
    
    # Measure energy (averaged)
    baseline_energy_kwh, baseline_emissions_kg, baseline_energy_per_pred_kwh, baseline_images, baseline_pred_energy_per_image_kwh = measure_baseline_energy_averaged(
        net, test_loader, save_dir, model_name, dataset_name
    )
    
    # Additional metrics
    params = params_count(net)
    model_size = os.path.getsize(baseline_path) / (1024 * 1024) if os.path.exists(baseline_path) else model_size_bytes(net) / (1024 * 1024)
    flops = compute_flops(net)
    flops_m = flops / 1e6 if not math.isnan(flops) else float("nan")
    
    # Timing
    inf_time, peak_ram, images = inference_time_per_batch(net, test_loader, timed=TIMING_BATCHES)
    
    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS:")
    print(f"{'='*60}")
    print(f"Acc: {metrics['acc']:.4f}, AUC: {metrics['auc']:.4f}")
    print(f"Params: {params:,}")
    print(f"Size: {model_size:.2f} MB")
    print(f"FLOPs: {flops_m:.2f}M per image")
    print(f"{'='*60}")
    
    # Prepare results row
    results = {
        'Dataset': dataset_name,
        'Variant': f'baseline_{model_name}',
        'TeacherModel': model_name,
        'StudentModel': model_name,
        'Stage': 'baseline',
        'KeepRatio': 1.0,
        
        # Performance metrics
        'Acc': metrics['acc'],
        'AUC': metrics['auc'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'Specificity': metrics['specificity'],
        'F1': metrics['f1'],
        
        # No comparison for baseline
        'TeacherAcc': metrics['acc'],
        'TeacherAUC': metrics['auc'],
        'AccDrop_percent': 0.0,
        'AucDrop_percent': 0.0,
        
        # Model characteristics
        'Params': params,
        'TeacherParams': params,
        'ParamReduction_percent': 0.0,
        'ModelSizeMB': model_size,
        'TeacherSizeMB': model_size,
        'CompressionRatio': 1.0,
        'FLOPs_per_image': flops,
        'FLOPs_M_per_image': flops_m,
        
        # Timing & Resources
        'InferenceTime_per_batch_s': inf_time,
        'PeakRAM_MB': peak_ram,
        'ImagesProcessedDuringTiming': images,
        
        # Energy metrics - No training for baseline
        'TrainingEnergy_kWh': 0.0,
        'TrainingEmissions_kg': 0.0,
        
        # Energy metrics - Baseline is also teacher
        'TeacherInferenceEnergy_kWh_total': baseline_energy_kwh,
        'TeacherEnergy_per_pred_kWh': baseline_energy_per_pred_kwh,
        'TeacherEmissions_kg_total': baseline_emissions_kg,
        'TeacherPredictionEnergy_per_image_kWh': baseline_pred_energy_per_image_kwh,
        
        # Energy metrics - Same as teacher for baseline
        'InferenceEnergy_kWh_total': baseline_energy_kwh,
        'Energy_per_pred_kWh': baseline_energy_per_pred_kwh,
        'Emissions_kg_total': baseline_emissions_kg,
        'PredictionEnergy_per_image_kWh': baseline_pred_energy_per_image_kwh,
        
        # No break-even for baseline
        'BreakEvenPredictions': float('nan'),
        
        # Path
        'ModelPath': baseline_path
    }
    
    return results, net


def save_results_to_csv(results_list, dataset_name, save_dir):
    """Save results to CSV file."""
    csv_path = os.path.join(save_dir, f"{dataset_name}_pruning_results.csv")
    
    if len(results_list) == 0:
        print(f"Warning: No results to save for {dataset_name}")
        return
    
    df = pd.DataFrame(results_list)
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to {csv_path}")


def main():
    set_seed(SEED)
    print(f"Using {DEVICE} device.")
    
    # Validate directories
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found!")
        sys.exit(1)
    
    if not os.path.exists(BASELINE_DIR):
        print(f"Error: Baseline directory '{BASELINE_DIR}' not found!")
        sys.exit(1)
    
    # Define datasets and models
    datasets = ['bloodmnist', 'pathmnist', 'dermamnist']
    
    # Baseline models
    baseline_models = ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224']
    
    # Model hierarchy for knowledge distillation (teacher -> student)
    kd_pairs = [
        ('vit_base_patch16_224', 'vit_small_patch16_224'),
        ('vit_base_patch16_224', 'vit_tiny_patch16_224'),
        ('vit_small_patch16_224', 'vit_tiny_patch16_224')
    ]
    
    models_for_quantization = ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224']
    
    print("=" * 100)
    print("MODEL PRUNING: QUANTIZATION & KNOWLEDGE DISTILLATION WITH BENCHMARKING")
    print("=" * 100)
    print(f"\nDatasets: {datasets}")
    print(f"Baseline models: {baseline_models}")
    print(f"Models for quantization: {models_for_quantization}")
    print(f"Knowledge distillation pairs: {kd_pairs}")
    print(f"\nParameters:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - KD epochs: {EPOCHS_KD}")
    print(f"  - KD learning rate: {LR_KD}")
    print(f"  - Temperature: {TEMPERATURE}")
    print(f"  - Alpha (distillation weight): {ALPHA}")
    print(f"  - Timing batches: {TIMING_BATCHES}")
    print(f"  - Baseline runs: {NUM_BASELINE_RUNS}")
    print(f"  - Prediction images: {PREDICTION_IMAGES}")
    print("\n" + "=" * 100)
    
    # Process each dataset
    for dataset in datasets:
        print(f"\n{'#'*100}")
        print(f"# PROCESSING DATASET: {dataset.upper()}")
        print(f"{'#'*100}")
        
        # Create save directory
        save_dir = os.path.join(SAVE_DIR_BASE, dataset)
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if results already exist
        csv_path = os.path.join(save_dir, f"{dataset}_pruning_results.csv")
        if os.path.exists(csv_path):
            print(f"⊗ Results already exist at {csv_path}")
            print(f"⊗ Skipping dataset {dataset}")
            continue
        
        # Load dataset
        npz_path = os.path.join(DATASET_DIR, f"{dataset}_224.npz")
        if not os.path.exists(npz_path):
            print(f"✗ Error: Dataset file not found: {npz_path}")
            continue
        
        try:
            train_loader, val_loader, test_loader, num_classes, dataset_name = load_dataset(npz_path)
            print(f"Loaded {dataset_name}: {num_classes} classes")
            log_memory_usage(f"After loading {dataset_name}: ")
        except Exception as e:
            print(f"✗ Error loading dataset {dataset}: {str(e)}")
            continue
        
        # Store all results
        all_results = []
        
        # Cache for baseline models
        baseline_cache = {}
        
        # -------------------------
        # 1. EVALUATE BASELINES
        # -------------------------
        print(f"\n{'='*100}")
        print(f"STEP 1: EVALUATE BASELINE MODELS")
        print(f"{'='*100}")
        
        for model_name in baseline_models:
            baseline_path = os.path.join(BASELINE_DIR, f'{model_name}_{dataset_name}_pretrained.pth')
            
            if not os.path.exists(baseline_path):
                print(f"✗ Baseline model not found: {baseline_path}")
                continue
            
            try:
                results, baseline_model = evaluate_baseline_model(
                    model_name, baseline_path, test_loader, num_classes, dataset_name, save_dir
                )
                all_results.append(results)
                baseline_cache[model_name] = (baseline_model, baseline_path)
                print(f"✓ Completed baseline evaluation for {model_name} on {dataset_name}")
                cleanup_memory()
            except Exception as e:
                print(f"✗ Error evaluating baseline {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # -------------------------
        # 2. QUANTIZATION (AMP)
        # -------------------------
        print(f"\n{'='*100}")
        print(f"STEP 2: QUANTIZATION WITH AMP")
        print(f"{'='*100}")
        
        for model_name in models_for_quantization:
            if model_name not in baseline_cache:
                print(f"✗ Baseline model {model_name} not in cache, skipping quantization")
                continue
            
            baseline_model, baseline_path = baseline_cache[model_name]
            
            try:
                results = quantize_model_amp(
                    dataset_name, model_name, baseline_path, baseline_model, 
                    test_loader, num_classes, save_dir
                )
                if results is not None:
                    all_results.append(results)
                    print(f"✓ Completed quantization for {model_name} on {dataset_name}")
                else:
                    print(f"⊗ Skipped quantization for {model_name} on {dataset_name} (already exists)")
                cleanup_memory()
            except Exception as e:
                print(f"✗ Error quantizing {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # -------------------------
        # 3. KNOWLEDGE DISTILLATION
        # -------------------------
        print(f"\n{'='*100}")
        print(f"STEP 3: KNOWLEDGE DISTILLATION")
        print(f"{'='*100}")
        
        for teacher_model, student_model in kd_pairs:
            if teacher_model not in baseline_cache:
                print(f"✗ Teacher model {teacher_model} not in cache, skipping KD")
                continue
            
            teacher_baseline, teacher_path = baseline_cache[teacher_model]
            
            try:
                results = knowledge_distillation(
                    dataset_name, teacher_model, student_model,
                    teacher_path, teacher_baseline, train_loader, val_loader, test_loader, num_classes, save_dir
                )
                if results is not None:
                    all_results.append(results)
                    print(f"✓ Completed KD: {teacher_model} → {student_model} on {dataset_name}")
                else:
                    print(f"⊗ Skipped KD for {teacher_model} → {student_model} on {dataset_name} (already exists)")
                cleanup_memory()
            except Exception as e:
                print(f"✗ Error in KD {teacher_model} → {student_model}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Save all results to CSV
        save_results_to_csv(all_results, dataset_name, save_dir)
        
        # Cleanup
        del train_loader, val_loader, test_loader, baseline_cache
        cleanup_memory()
        print(f"\nFinished processing dataset: {dataset.upper()}\n")

    print("\n" + "="*100)
    print("ALL DATASETS PROCESSED SUCCESSFULLY!")
    print("="*100)


if __name__ == "__main__":
    # Optional: Set environment variables for performance
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nCritical error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)