import os
import sys
import numpy as np
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
from codecarbon import EmissionsTracker

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
# Quantization with AMP
# -------------------------
def quantize_model_amp(dataset_name, model_name, baseline_path, test_loader, num_classes):
    """
    Apply Automatic Mixed Precision (AMP) quantization.
    This converts the model to use float16 where beneficial while keeping float32 where needed.
    """
    print(f"\n{'='*100}")
    print(f"Quantizing {model_name} on {dataset_name} using AMP")
    print(f"{'='*100}")
    
    # Create save directory
    save_dir = os.path.join(SAVE_DIR_BASE, "quantized_amp")
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if quantized model already exists
    save_path = os.path.join(save_dir, f'{model_name}_{dataset_name}_amp.pth')
    if os.path.exists(save_path):
        print(f"⊗ Quantized model already exists at {save_path}")
        print(f"⊗ Skipping quantization for {model_name} on {dataset_name}")
        return None
    
    # Start energy tracking
    tracker = EmissionsTracker(
        project_name=f"quantization_{model_name}_{dataset_name}",
        log_level='error',
        save_to_file=False
    )
    tracker.start()
    
    # Load baseline model
    print(f"Loading baseline model from {baseline_path}")
    net = timm.create_model(model_name, pretrained=False, num_classes=num_classes).to(DEVICE)
    checkpoint = torch.load(baseline_path, map_location=DEVICE)
    net.load_state_dict(checkpoint['model'])
    
    # Evaluate baseline (without AMP)
    print("Evaluating baseline model (float32)...")
    baseline_metrics = evaluate_model(net, test_loader, DEVICE, use_amp=False)
    
    # Evaluate with AMP (float16)
    print("Evaluating with AMP (float16)...")
    quantized_metrics = evaluate_model(net, test_loader, DEVICE, use_amp=True)
    
    # Stop energy tracking
    emissions = tracker.stop()
    energy_kwh = tracker._total_energy.kWh if hasattr(tracker, '_total_energy') else 0.0
    co2_kg = emissions if emissions else 0.0
    
    print(f"\n⚡ Energy consumed: {energy_kwh:.6f} kWh")
    print(f"🌍 CO2 emissions: {co2_kg:.6f} kg")
    
    # Save quantized model
    state = {
        'model': net.state_dict(),
        'acc': quantized_metrics['acc'],
        'auc': quantized_metrics['auc'],
        'model_name': model_name,
        'dataset': dataset_name,
        'pruning_method': 'amp_quantization'
    }
    torch.save(state, save_path)
    print(f"✓ Quantized model saved to {save_path}")
    
    # Calculate drops
    acc_drop = (baseline_metrics['acc'] - quantized_metrics['acc']) * 100
    auc_drop = (baseline_metrics['auc'] - quantized_metrics['auc']) * 100
    
    # Calculate model size
    baseline_size = os.path.getsize(baseline_path) / (1024 * 1024)  # MB
    quantized_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
    compression_ratio = baseline_size / quantized_size if quantized_size > 0 else 1.0
    
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS:")
    print(f"{'='*60}")
    print(f"Baseline  - Acc: {baseline_metrics['acc']:.4f}, AUC: {baseline_metrics['auc']:.4f}")
    print(f"Quantized - Acc: {quantized_metrics['acc']:.4f}, AUC: {quantized_metrics['auc']:.4f}")
    print(f"Drop      - Acc: {acc_drop:.2f}%, AUC: {auc_drop:.2f}%")
    print(f"Size      - Baseline: {baseline_size:.2f} MB, Quantized: {quantized_size:.2f} MB")
    print(f"Compression: {compression_ratio:.2f}x")
    print(f"{'='*60}")
    
    # Prepare results
    results = {
        'dataset': dataset_name,
        'model': model_name,
        'pruning_method': 'amp_quantization',
        'baseline_acc': baseline_metrics['acc'],
        'baseline_auc': baseline_metrics['auc'],
        'test_acc': quantized_metrics['acc'],
        'test_auc': quantized_metrics['auc'],
        'acc_drop_percent': acc_drop,
        'auc_drop_percent': auc_drop,
        'test_precision': quantized_metrics['precision'],
        'test_recall': quantized_metrics['recall'],
        'test_specificity': quantized_metrics['specificity'],
        'test_f1': quantized_metrics['f1'],
        'baseline_size_mb': baseline_size,
        'quantized_size_mb': quantized_size,
        'compression_ratio': compression_ratio,
        'energy_kwh': energy_kwh,
        'co2_emissions_kg': co2_kg
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
                          teacher_path, train_loader, val_loader, test_loader, num_classes):
    """
    Perform knowledge distillation from teacher to student.
    """
    print(f"\n{'='*100}")
    print(f"Knowledge Distillation: {teacher_model_name} → {student_model_name} on {dataset_name}")
    print(f"{'='*100}")
    
    # Create save directory
    save_dir = os.path.join(SAVE_DIR_BASE, "knowledge_distillation")
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if KD model already exists
    save_path = os.path.join(save_dir, f'{student_model_name}_{dataset_name}_kd_from_{teacher_model_name}.pth')
    if os.path.exists(save_path):
        print(f"⊗ KD model already exists at {save_path}")
        print(f"⊗ Skipping knowledge distillation for {teacher_model_name} → {student_model_name} on {dataset_name}")
        return None
    
    # Start energy tracking for training
    training_tracker = EmissionsTracker(
        project_name=f"kd_training_{teacher_model_name}_to_{student_model_name}_{dataset_name}",
        log_level='error',
        save_to_file=False
    )
    training_tracker.start()
    
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
    
    # Create student model (random initialization)
    print(f"Creating student model: {student_model_name} (random weights)")
    student = timm.create_model(student_model_name, pretrained=False, num_classes=num_classes).to(DEVICE)
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
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
    training_emissions = training_tracker.stop()
    training_energy_kwh = training_tracker._total_energy.kWh if hasattr(training_tracker, '_total_energy') else 0.0
    training_co2_kg = training_emissions if training_emissions else 0.0
    
    print(f"\n⚡ Training energy consumed: {training_energy_kwh:.6f} kWh")
    print(f"🌍 Training CO2 emissions: {training_co2_kg:.6f} kg")
    
    print(f"\nTraining completed! Best Acc: {best_acc:.4f}, AUC: {best_auc:.4f}")
    
    # Test set evaluation with energy tracking
    print(f"\nEvaluating student model on test set...")
    eval_tracker = EmissionsTracker(
        project_name=f"kd_evaluation_{student_model_name}_{dataset_name}",
        log_level='error',
        save_to_file=False
    )
    eval_tracker.start()
    
    checkpoint = torch.load(save_path, map_location=DEVICE)
    student.load_state_dict(checkpoint['model'])
    test_metrics = evaluate_model(student, test_loader, DEVICE)
    
    eval_emissions = eval_tracker.stop()
    eval_energy_kwh = eval_tracker._total_energy.kWh if hasattr(eval_tracker, '_total_energy') else 0.0
    eval_co2_kg = eval_emissions if eval_emissions else 0.0
    
    print(f"\n⚡ Evaluation energy consumed: {eval_energy_kwh:.6f} kWh")
    print(f"🌍 Evaluation CO2 emissions: {eval_co2_kg:.6f} kg")
    
    # Total energy
    total_energy_kwh = training_energy_kwh + eval_energy_kwh
    total_co2_kg = training_co2_kg + eval_co2_kg
    
    print(f"\n⚡ TOTAL energy consumed: {total_energy_kwh:.6f} kWh")
    print(f"🌍 TOTAL CO2 emissions: {total_co2_kg:.6f} kg")
    
    print(f"\nFinal Results:")
    print(f"Teacher Test Acc: {teacher_metrics['acc']:.4f}, AUC: {teacher_metrics['auc']:.4f}")
    print(f"Student Test Acc: {test_metrics['acc']:.4f}, AUC: {test_metrics['auc']:.4f}")
    print(f"Accuracy drop: {(teacher_metrics['acc'] - test_metrics['acc'])*100:.2f}%")
    
    # Prepare results
    results = {
        'dataset': dataset_name,
        'teacher_model': teacher_model_name,
        'student_model': student_model_name,
        'pruning_method': 'knowledge_distillation',
        'teacher_test_acc': teacher_metrics['acc'],
        'teacher_test_auc': teacher_metrics['auc'],
        'student_test_acc': test_metrics['acc'],
        'student_test_auc': test_metrics['auc'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_specificity': test_metrics['specificity'],
        'test_f1': test_metrics['f1'],
        'param_reduction_percent': (1 - student_params/teacher_params)*100,
        'training_energy_kwh': training_energy_kwh,
        'training_co2_kg': training_co2_kg,
        'eval_energy_kwh': eval_energy_kwh,
        'eval_co2_kg': eval_co2_kg,
        'total_energy_kwh': total_energy_kwh,
        'total_co2_kg': total_co2_kg
    }
    
    return results


def save_results_to_csv(results, method_name):
    """Save results to CSV file."""
    csv_path = os.path.join(SAVE_DIR_BASE, f"{method_name}_results.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    
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
    
    # Model hierarchy for knowledge distillation (teacher -> student)
    kd_pairs = [
        ('vit_base_patch16_224', 'vit_small_patch16_224'),
        ('vit_base_patch16_224', 'vit_tiny_patch16_224'),
        ('vit_small_patch16_224', 'vit_tiny_patch16_224')
    ]
    
    models_for_quantization = ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224']
    
    print("=" * 100)
    print("MODEL PRUNING: QUANTIZATION (AMP) & KNOWLEDGE DISTILLATION")
    print("=" * 100)
    print(f"\nDatasets: {datasets}")
    print(f"Models for quantization: {models_for_quantization}")
    print(f"Knowledge distillation pairs: {kd_pairs}")
    print(f"\nParameters:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - KD epochs: {EPOCHS_KD}")
    print(f"  - KD learning rate: {LR_KD}")
    print(f"  - Temperature: {TEMPERATURE}")
    print(f"  - Alpha (distillation weight): {ALPHA}")
    print("\n" + "=" * 100)
    
    # Process each dataset
    for dataset in datasets:
        print(f"\n{'#'*100}")
        print(f"# PROCESSING DATASET: {dataset.upper()}")
        print(f"{'#'*100}")
        
        # Load dataset
        npz_path = os.path.join(DATASET_DIR, f"{dataset}_224.npz")
        if not os.path.exists(npz_path):
            print(f"✗ Error: Dataset file not found: {npz_path}")
            continue
        
        try:
            train_loader, val_loader, test_loader, num_classes, dataset_name = load_dataset(npz_path)
            print(f"Loaded {dataset_name}: {num_classes} classes")
        except Exception as e:
            print(f"✗ Error loading dataset {dataset}: {str(e)}")
            continue
        
        # -------------------------
        # 1. QUANTIZATION (AMP)
        # -------------------------
        print(f"\n{'='*100}")
        print(f"STEP 1: QUANTIZATION WITH AMP")
        print(f"{'='*100}")
        
        for model_name in models_for_quantization:
            baseline_path = os.path.join(BASELINE_DIR, f'{model_name}_{dataset_name}_pretrained.pth')
            
            if not os.path.exists(baseline_path):
                print(f"✗ Baseline model not found: {baseline_path}")
                continue
            
            try:
                results = quantize_model_amp(dataset_name, model_name, baseline_path, test_loader, num_classes)
                if results is not None:
                    save_results_to_csv(results, "quantization_amp")
                    print(f"✓ Completed quantization for {model_name} on {dataset_name}")
                else:
                    print(f"⊗ Skipped quantization for {model_name} on {dataset_name} (already exists)")
            except Exception as e:
                print(f"✗ Error quantizing {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # -------------------------
        # 2. KNOWLEDGE DISTILLATION
        # -------------------------
        print(f"\n{'='*100}")
        print(f"STEP 2: KNOWLEDGE DISTILLATION")
        print(f"{'='*100}")
        
        for teacher_model, student_model in kd_pairs:
            teacher_path = os.path.join(BASELINE_DIR, f'{teacher_model}_{dataset_name}_pretrained.pth')
            
            if not os.path.exists(teacher_path):
                print(f"✗ Teacher model not found: {teacher_path}")
                continue
            
            try:
                results = knowledge_distillation(
                    dataset_name, teacher_model, student_model,
                    teacher_path, train_loader, val_loader, test_loader, num_classes
                )
                if results is not None:
                    save_results_to_csv(results, "knowledge_distillation")
                    print(f"✓ Completed KD: {teacher_model} → {student_model} on {dataset_name}")
                else:
                    print(f"⊗ Skipped KD for {teacher_model} → {student_model} on {dataset_name} (already exists)")
            except Exception as e:
                print(f"✗ Error in KD {teacher_model} → {student_model}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 100)
    print("ALL PRUNING COMPLETED!")
    print("=" * 100)
    print(f"\nResults saved in:")
    print(f"  - Quantized models: {os.path.join(SAVE_DIR_BASE, 'quantized_amp')}")
    print(f"  - KD models: {os.path.join(SAVE_DIR_BASE, 'knowledge_distillation')}")
    print(f"  - CSV results: {SAVE_DIR_BASE}")


if __name__ == '__main__':
    main()