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


# -------------------------
# Config
# -------------------------
DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/new_baseline"
EPOCHS = 100
BATCH_SIZE = 64
LR = 0.001
MIN_LR = 1e-6
WEIGHT_DECAY = 0.05
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EARLY_STOP_PATIENCE = 10

os.makedirs(SAVE_DIR, exist_ok=True)


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
    """
    Wraps a numpy array (H,W[,C]) and a label array.
    Auto-detects grayscale vs RGB and normalizes accordingly.
    Includes data augmentation for training.
    """
    def __init__(self, imgs_np, labels_np, img_size=224, is_train=False):
        self.imgs = imgs_np
        self.labels = labels_np
        self.img_size = img_size
        self.is_train = is_train

        if is_train:
            # Training transforms with augmentation
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
        label = int(self.labels[idx])
        x = self.base_tfms(img)
        
        # If grayscale → expand to RGB
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

    # Create datasets
    train_ds = NumpyMemmapDataset(X_train, y_train, img_size=IMG_SIZE, is_train=True)
    val_ds = NumpyMemmapDataset(X_val, y_val, img_size=IMG_SIZE, is_train=False)
    test_ds = NumpyMemmapDataset(X_test, y_test, img_size=IMG_SIZE, is_train=False)

    # Create data loaders
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


def overall_accuracy(conf_matrix):
    """Calculates overall accuracy for multi-class."""
    tp_tn_sum = conf_matrix.trace()
    total_sum = conf_matrix.sum()
    return tp_tn_sum / total_sum if total_sum > 0 else 0.0


def train_epoch(net, train_loader, optimizer, scheduler, loss_function, device):
    """Train for one epoch."""
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout, desc="Training")
    
    for step, (images, labels) in enumerate(train_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        
        train_bar.set_postfix(loss=f"{loss.item():.3f}")
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def evaluate_model(net, test_loader, device):
    """Evaluate the model."""
    net.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        val_bar = tqdm(test_loader, file=sys.stdout, desc="Evaluating")
        for inputs, targets in val_bar:
            inputs = inputs.to(device)
            outputs = net(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            predict_y = torch.max(probs, dim=1)[1]
            
            all_preds.extend(predict_y.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
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


def train(dataset_name, model_name, train_loader, val_loader, num_classes):
    """Main training function."""
    print(f"\n{'='*100}")
    print(f"Training {model_name} on {dataset_name}")
    print(f"{'='*100}")
    
    # Create model (from scratch, no pretrained weights)
    print(f"Creating model: {model_name}")
    net = timm.create_model(
        model_name,
        pretrained=False,  # Train from scratch
        num_classes=num_classes
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Loss function
    loss_function = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        net.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    total_steps = EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=MIN_LR
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, verbose=True)
    
    # Training loop
    best_acc = 0.0
    best_auc = 0.0
    save_path = os.path.join(SAVE_DIR, f'{model_name}_{dataset_name}_scratch.pth')
    
    print("\nStarting training...")
    print(f"Early stopping enabled with patience: {EARLY_STOP_PATIENCE}")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        
        # Train
        train_loss = train_epoch(net, train_loader, optimizer, scheduler, loss_function, DEVICE)
        
        # Evaluate
        metrics = evaluate_model(net, val_loader, DEVICE)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val AUC: {metrics['auc']:.4f}, Val Acc: {metrics['acc']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}, F1: {metrics['f1']:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.8f}")
        
        # Save best model
        if metrics['acc'] > best_acc:
            print(f"\n✓ New best accuracy: {metrics['acc']:.4f} (previous: {best_acc:.4f})")
            best_acc = metrics['acc']
            best_auc = metrics['auc']
            
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'auc': best_auc,
                'epoch': epoch,
                'model_name': model_name,
                'dataset': dataset_name
            }
            torch.save(state, save_path)
            print(f"Model saved to {save_path}")
        
        # Check early stopping
        early_stopping(metrics['acc'])
        
        if early_stopping.early_stop:
            print(f"\n{'='*100}")
            print(f"Early stopping triggered at epoch {epoch + 1}")
            print(f"No improvement in validation accuracy for {EARLY_STOP_PATIENCE} consecutive epochs")
            print(f"{'='*100}")
            break
    
    print(f"\n{'='*100}")
    print("Training completed!")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Total epochs trained: {epoch + 1}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*100}")


def main():
    set_seed(SEED)
    print(f"Using {DEVICE} device.")
    
    # Validate dataset path
    if not os.path.exists(DATASET_DIR):
        print(f"Error: '{DATASET_DIR}' directory not found!")
        print("Please ensure your balanced datasets are in this directory.")
        sys.exit(1)
    
    # Define datasets and models to train
    datasets = ['bloodmnist', 'pathmnist', 'dermamnist']
    models = ['vit_tiny_patch16_224', 'vit_small_patch16_224', 
              'vit_base_patch16_224', 'vit_large_patch16_224']
    
    print("=" * 100)
    print("TRAINING VISION TRANSFORMERS ON MEDMNIST DATASETS")
    print("=" * 100)
    print(f"\nDatasets: {datasets}")
    print(f"Models: {models}")
    print(f"Total combinations: {len(datasets) * len(models)}")
    print(f"\nHyperparameters:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Max epochs: {EPOCHS}")
    print(f"  - Early stopping patience: {EARLY_STOP_PATIENCE}")
    print(f"  - Learning rate: {LR}")
    print(f"  - Weight decay: {WEIGHT_DECAY}")
    print("\n" + "=" * 100)
    
    # Train all combinations
    total_models = len(datasets) * len(models)
    current_model = 0
    
    for dataset in datasets:
        # Load dataset NPZ file
        npz_path = os.path.join(DATASET_DIR, f"{dataset}_224.npz")
        
        if not os.path.exists(npz_path):
            print(f"\n✗ Error: Dataset file not found: {npz_path}")
            print("Skipping this dataset...")
            continue
        
        try:
            train_loader, val_loader, test_loader, num_classes, dataset_name = load_dataset(npz_path)
            print(f"Loaded {dataset_name}: {num_classes} classes")
        except Exception as e:
            print(f"\n✗ Error loading dataset {dataset}: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Skipping this dataset...")
            continue
        
        for model_name in models:
            current_model += 1
            
            print("\n" + "=" * 100)
            print(f"TRAINING MODEL {current_model}/{total_models}")
            print(f"Dataset: {dataset_name} | Model: {model_name}")
            print("=" * 100)
            
            try:
                train(dataset_name, model_name, train_loader, val_loader, num_classes)
                print(f"\n✓ Successfully completed training for {model_name} on {dataset_name}")
            except Exception as e:
                print(f"\n✗ Error training {model_name} on {dataset_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                print("Continuing to next model...")
                continue
    
    print("\n" + "=" * 100)
    print("ALL TRAINING COMPLETED!")
    print("=" * 100)


if __name__ == '__main__':
    main()