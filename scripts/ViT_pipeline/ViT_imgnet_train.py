import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import timm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import csv

# -------------------------
# Config
# -------------------------
DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
NPY_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_npy"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/PruneAndTrain/Vision/new_baseline"
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.001
MIN_LR = 1e-6
WEIGHT_DECAY = 0.05
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EARLY_STOP_PATIENCE = 10

# Training strategy: 'pretrained', 'scratch_enhanced', or 'scratch_original'
TRAINING_STRATEGY = 'pretrained'  # Change this to experiment

# Enhanced regularization for from-scratch training
DROP_PATH_RATE = 0.1  # Stochastic depth
LABEL_SMOOTHING = 0.1  # Label smoothing
MIXUP_ALPHA = 0.2  # Mixup augmentation

# Multi-label datasets (use BCEWithLogitsLoss + sigmoid instead of CrossEntropy + softmax)
MULTI_LABEL_DATASETS = {'chestmnist'}

os.makedirs(SAVE_DIR, exist_ok=True)


# -------------------------
# Mixup Implementation
# -------------------------
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# -------------------------
# Early Stopping Class
# -------------------------
class EarlyStopping:
    """Early stops training if validation metric doesn't improve after patience epochs."""
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
    """
    Wraps a numpy array (H,W[,C]) and a label array.
    Auto-detects grayscale vs RGB and normalizes accordingly.
    Includes data augmentation for training.
    Supports both multi-class (single int label) and multi-label (float vector).
    """
    def __init__(self, imgs_np, labels_np, img_size=224, is_train=False, use_strong_aug=False, multi_label=False):
        self.imgs = imgs_np
        self.labels = labels_np
        self.img_size = img_size
        self.is_train = is_train
        self.multi_label = multi_label

        if is_train:
            if use_strong_aug:
                # Stronger augmentation for from-scratch training
                self.base_tfms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomRotation(20),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
                    transforms.ToTensor(),
                ])
            else:
                # Standard augmentation
                self.base_tfms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                ])
        else:
            # Test/validation transforms without augmentation
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
        x = self.base_tfms(img)

        # If grayscale → expand to RGB
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)

        x = self.normalize(x)

        if self.multi_label:
            label = torch.FloatTensor(np.array(self.labels[idx]))
        else:
            label = int(self.labels[idx])

        return x, label


def load_dataset(dataset_name: str, use_strong_aug=False):
    """Load dataset from NPZ (datasets_balanced) or NPY dir (datasets_npy)."""
    multi_label = dataset_name in MULTI_LABEL_DATASETS

    if multi_label:
        # Load from .npy directory
        npy_dir = os.path.join(NPY_DIR, f"{dataset_name}_{IMG_SIZE}")
        print(f"Loading {npy_dir} (multi-label) ...")
        X_train = np.load(os.path.join(npy_dir, "train_images.npy"), mmap_mode="r")
        y_train = np.load(os.path.join(npy_dir, "train_labels.npy"), mmap_mode="r")
        X_val = np.load(os.path.join(npy_dir, "val_images.npy"), mmap_mode="r")
        y_val = np.load(os.path.join(npy_dir, "val_labels.npy"), mmap_mode="r")
        X_test = np.load(os.path.join(npy_dir, "test_images.npy"), mmap_mode="r")
        y_test = np.load(os.path.join(npy_dir, "test_labels.npy"), mmap_mode="r")
        num_classes = y_train.shape[1]
    else:
        # Load from .npz file
        npz_path = os.path.join(DATASET_DIR, f"{dataset_name}_224.npz")
        print(f"Loading {npz_path} ...")
        data = np.load(npz_path, mmap_mode="r")
        X_train = data["train_images"]
        y_train = data["train_labels"].flatten()
        X_val = data["val_images"]
        y_val = data["val_labels"].flatten()
        X_test = data["test_images"]
        y_test = data["test_labels"].flatten()
        num_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))

    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    print(f"Dataset sizes: train={n_train}, val={n_val}, test={n_test}")
    print(f"Num classes: {num_classes}, Multi-label: {multi_label}")

    # Create datasets
    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE, is_train=True,
                                  use_strong_aug=use_strong_aug, multi_label=multi_label)
    val_ds = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE, is_train=False, multi_label=multi_label)
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, is_train=False, multi_label=multi_label)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

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


def overall_accuracy(conf_matrix):
    """Calculates overall accuracy for multi-class."""
    tp_tn_sum = conf_matrix.trace()
    total_sum = conf_matrix.sum()
    return tp_tn_sum / total_sum if total_sum > 0 else 0.0


def train_epoch(net, train_loader, optimizer, scheduler, loss_function, device, use_mixup=False, multi_label=False):
    """Train for one epoch."""
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout, desc="Training")

    for step, (images, labels) in enumerate(train_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_mixup and np.random.rand() < 0.5:
            # Apply mixup
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=MIXUP_ALPHA)
            outputs = net(images)
            loss = mixup_criterion(loss_function, outputs, targets_a, targets_b, lam)
        else:
            outputs = net(images)
            loss = loss_function(outputs, labels)

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        train_bar.set_postfix(loss=f"{loss.item():.3f}")

    avg_loss = running_loss / len(train_loader)
    return avg_loss


def evaluate_model(net, test_loader, device, multi_label=False):
    """Evaluate the model (multi-class)."""
    net.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        val_bar = tqdm(test_loader, file=sys.stdout, desc="Evaluating")
        for inputs, targets in val_bar:
            inputs = inputs.to(device)
            outputs = net(inputs)

            if multi_label:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
            else:
                probs = torch.softmax(outputs, dim=1)
                predict_y = torch.max(probs, dim=1)[1]
                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

    if multi_label:
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)

        num_labels = all_labels.shape[1]

        # Per-label accuracy
        per_label_acc_array = (all_preds == all_labels).mean(axis=0)  # array of length num_labels
        per_label_acc = float(per_label_acc_array.mean())  # scalar average
        # Exact match ratio (subset accuracy)
        exact_match = (all_preds == all_labels).all(axis=1).mean()

        # Per-label AUC
        per_label_auc = []
        for i in range(num_labels):
            try:
                per_label_auc.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
            except ValueError:
                per_label_auc.append(float('nan'))

        try:
            auc = roc_auc_score(all_labels, all_probs, average='macro')
        except ValueError:
            auc = float('nan')

        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # Per-label specificity: TN / (TN + FP) for each label
        per_label_specificity = []
        per_label_precision = []
        per_label_recall = []
        per_label_f1 = []
        for i in range(num_labels):
            tn = ((all_preds[:, i] == 0) & (all_labels[:, i] == 0)).sum()
            fp = ((all_preds[:, i] == 1) & (all_labels[:, i] == 0)).sum()
            tp = ((all_preds[:, i] == 1) & (all_labels[:, i] == 1)).sum()
            fn = ((all_preds[:, i] == 0) & (all_labels[:, i] == 1)).sum()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_i = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_label_specificity.append(spec)
            per_label_precision.append(prec)
            per_label_recall.append(rec)
            per_label_f1.append(f1_i)

        avg_specificity = float(np.mean(per_label_specificity))

        metrics = {
            'acc': float(per_label_acc),
            'exact_match': float(exact_match),
            'auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': avg_specificity,
            'f1': float(f1),
            'per_label_auc': per_label_auc,
            'per_label_acc': per_label_acc_array.tolist(),
            'per_label_precision': per_label_precision,
            'per_label_recall': per_label_recall,
            'per_label_f1': per_label_f1,
            'per_label_specificity': per_label_specificity,
        }
        return metrics

    # Multi-class path (unchanged)
    acc = sum([1 for i in range(len(all_preds)) if all_preds[i] == all_labels[i]]) / len(all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    specificity = specificity_per_class(conf_matrix)
    avg_specificity = sum(specificity) / len(specificity) if specificity else 0.0

    # Calculate AUC
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




def train(dataset_name, model_name, train_loader, val_loader, test_loader, num_classes, strategy='pretrained'):
    """Main training function with different strategies."""
    multi_label = dataset_name in MULTI_LABEL_DATASETS

    print(f"\n{'='*100}")
    print(f"Training {model_name} on {dataset_name}")
    print(f"Strategy: {strategy}, Multi-label: {multi_label}")
    print(f"{'='*100}")

    # Determine strategy details
    use_mixup = (strategy == 'scratch_enhanced')
    use_strong_aug = (strategy == 'scratch_enhanced')
    use_pretrained = (strategy == 'pretrained')

    # Create model
    print(f"Creating model: {model_name}")
    if use_pretrained:
        print("Loading ImageNet pretrained weights...")
        net = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            drop_path_rate=0.0
        ).to(DEVICE)
    else:
        print("Training from scratch...")
        if strategy == 'scratch_enhanced':
            print(f"Using enhanced regularization: DropPath={DROP_PATH_RATE}, Mixup={MIXUP_ALPHA}, LabelSmoothing={LABEL_SMOOTHING}")
        net = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            drop_path_rate=DROP_PATH_RATE if strategy == 'scratch_enhanced' else 0.0
        ).to(DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Loss function
    if multi_label:
        loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING if strategy == 'scratch_enhanced' else 0.0)

    # Optimizer
    if use_pretrained:
        actual_lr = LR * 0.1
        actual_weight_decay = WEIGHT_DECAY * 0.1
        print(f"Fine-tuning with LR={actual_lr}, WD={actual_weight_decay}")
    else:
        actual_lr = LR
        actual_weight_decay = WEIGHT_DECAY

    optimizer = optim.AdamW(net.parameters(), lr=actual_lr, betas=(0.9, 0.999), weight_decay=actual_weight_decay)

    # Scheduler
    total_steps = EPOCHS * len(train_loader)
    if use_pretrained:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)
    else:
        warmup_steps = 5 * len(train_loader)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Early stopping setup
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, verbose=True)
    best_acc, best_auc = 0.0, 0.0
    strategy_suffix = 'pretrained' if use_pretrained else ('scratch_enhanced' if strategy == 'scratch_enhanced' else 'scratch')
    save_path = os.path.join(SAVE_DIR, f'{model_name}_{dataset_name}_{strategy_suffix}.pth')

    # ------------------ Training Loop ------------------
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        train_loss = train_epoch(net, train_loader, optimizer, scheduler, loss_function, DEVICE,
                                 use_mixup=use_mixup, multi_label=multi_label)
        metrics = evaluate_model(net, val_loader, DEVICE, multi_label=multi_label)

        print(f"Train Loss: {train_loss:.4f}")
        if multi_label:
            print(f"Val AUC: {metrics['auc']:.4f}, Val Per-Label Acc: {metrics['acc']:.4f}, Val Exact-Match: {metrics['exact_match']:.4f}")
        else:
            print(f"Val AUC: {metrics['auc']:.4f}, Val Acc: {metrics['acc']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

        if metrics['auc'] > best_auc:
            best_acc, best_auc = metrics['acc'], metrics['auc']
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'auc': best_auc,
                'epoch': epoch,
                'model_name': model_name,
                'dataset': dataset_name,
                'strategy': strategy,
                'multi_label': multi_label
            }
            torch.save(state, save_path)
            print(f"Model saved to {save_path}")

        early_stopping(metrics['auc'] if multi_label else metrics['acc'])
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    print(f"\nTraining completed! Best Acc: {best_acc:.4f}, AUC: {best_auc:.4f}")

    # ------------------ Test Set Evaluation ------------------
    print(f"\nEvaluating best model on test set for {dataset_name}...")
    checkpoint = torch.load(save_path, map_location=DEVICE)
    net.load_state_dict(checkpoint['model'])
    test_metrics = evaluate_model(net, test_loader, DEVICE, multi_label=multi_label)

    # Prepare results
    results = {
        'dataset': dataset_name,
        'model': model_name,
        'strategy': strategy,
        'multi_label': multi_label,
        'test_acc': test_metrics['acc'],
        'test_auc': test_metrics['auc'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_specificity': test_metrics['specificity'],
        'test_f1': test_metrics['f1']
    }

    # Save to CSV
    csv_path = os.path.join(SAVE_DIR, "test_results.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    print(f"Test results saved to {csv_path}")

    # Save per-label metrics for multi-label datasets
    if multi_label and 'per_label_auc' in test_metrics:
        per_label_csv_path = os.path.join(SAVE_DIR, "test_results_per_label.csv")
        per_label_exists = os.path.isfile(per_label_csv_path)
        num_labels = len(test_metrics['per_label_auc'])

        with open(per_label_csv_path, mode='a', newline='') as csv_file:
            fieldnames = ['dataset', 'model', 'strategy', 'label_idx',
                          'auc', 'acc', 'precision', 'recall', 'f1', 'specificity']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not per_label_exists:
                writer.writeheader()
            for i in range(num_labels):
                writer.writerow({
                    'dataset': dataset_name,
                    'model': model_name,
                    'strategy': strategy,
                    'label_idx': i,
                    'auc': test_metrics['per_label_auc'][i],
                    'acc': test_metrics['per_label_acc'][i],
                    'precision': test_metrics['per_label_precision'][i],
                    'recall': test_metrics['per_label_recall'][i],
                    'f1': test_metrics['per_label_f1'][i],
                    'specificity': test_metrics['per_label_specificity'][i],
                })

        print(f"Per-label metrics saved to {per_label_csv_path}")
        print(f"Per-label AUC: {[f'{a:.4f}' for a in test_metrics['per_label_auc']]}")



def main():
    set_seed(SEED)
    print(f"Using {DEVICE} device.")
    print(f"Training Strategy: {TRAINING_STRATEGY}")

    # Define datasets and models to train
    datasets = ['bloodmnist', 'pathmnist', 'dermamnist', 'chestmnist']
    model_names = ['vit_tiny_patch16_224', 'vit_small_patch16_224',
              'vit_base_patch16_224']

    strategy_suffix = 'pretrained' if TRAINING_STRATEGY == 'pretrained' else (
        'scratch_enhanced' if TRAINING_STRATEGY == 'scratch_enhanced' else 'scratch')

    print("=" * 100)
    print("TRAINING VISION TRANSFORMERS ON MEDMNIST DATASETS")
    print("=" * 100)
    print(f"\nDatasets: {datasets}")
    print(f"Models: {model_names}")
    print(f"Total combinations: {len(datasets) * len(model_names)}")
    print(f"\nHyperparameters:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Max epochs: {EPOCHS}")
    print(f"  - Early stopping patience: {EARLY_STOP_PATIENCE}")
    print(f"  - Learning rate: {LR}")
    print(f"  - Weight decay: {WEIGHT_DECAY}")
    print(f"  - Training strategy: {TRAINING_STRATEGY}")
    if TRAINING_STRATEGY == 'scratch_enhanced':
        print(f"  - Drop path rate: {DROP_PATH_RATE}")
        print(f"  - Label smoothing: {LABEL_SMOOTHING}")
        print(f"  - Mixup alpha: {MIXUP_ALPHA}")
    print("\n" + "=" * 100)

    # Determine if we need strong augmentation
    use_strong_aug = (TRAINING_STRATEGY == 'scratch_enhanced')

    # Train all combinations
    total_combos = len(datasets) * len(model_names)
    current = 0

    for dataset in datasets:
        # Check if ALL models for this dataset already exist → skip loading entirely
        all_exist = all(
            os.path.exists(os.path.join(SAVE_DIR, f'{m}_{dataset}_{strategy_suffix}.pth'))
            for m in model_names
        )
        if all_exist:
            for m in model_names:
                current += 1
            print(f"\nSkipping {dataset}: all {len(model_names)} model baselines already exist.")
            continue

        try:
            train_loader, val_loader, test_loader, num_classes, dataset_name = load_dataset(dataset, use_strong_aug=use_strong_aug)
        except Exception as e:
            print(f"Error loading dataset {dataset}: {str(e)}")
            for m in model_names:
                current += 1
            continue

        for model_name in model_names:
            current += 1
            save_path = os.path.join(SAVE_DIR, f'{model_name}_{dataset}_{strategy_suffix}.pth')
            if os.path.exists(save_path):
                print(f"\nSkipping {model_name} on {dataset} ({current}/{total_combos}): baseline already exists at {save_path}")
                continue

            print(f"\nTraining {model_name} on {dataset} ({current}/{total_combos})")

            try:
                train(dataset_name, model_name, train_loader, val_loader, test_loader, num_classes, strategy=TRAINING_STRATEGY)
            except Exception as e:
                print(f"Error training {model_name} on {dataset_name}: {str(e)}")
                continue

    print("Training complete.")


if __name__ == '__main__':
    main()
