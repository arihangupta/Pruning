"""
Progressive Vision Transformer Pruning for MedMNIST Datasets

Three-method comparison:
1. Baseline - No pruning
2. Progressive Pruning - Prune gradually during training
3. Prune-Then-Train - One-shot pruning at initialization

Matches experimental setup from ResNet progressive pruning script.
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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import pandas as pd
import copy
import gc
import math

# Try codecarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("CodeCarbon not available - energy metrics will be NaN")

# ==================== CONFIGURATION ====================

DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/vit_progressive_pruning"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224

# Training configuration (MATCHING CNN SCRIPT)
FIXED_EPOCHS = 15
BATCH_SIZE = 64
INITIAL_LR = 1e-4  # Lower LR for finetuning pretrained models
WEIGHT_DECAY = 1e-4  # L2 regularization
MIN_LR = 1e-6

# Progressive pruning configuration (MATCHING CNN SCRIPT)
WARMUP_EPOCHS = 2
EPOCHS_BETWEEN_PRUNES = 3
NUM_PRUNE_STEPS = 4  # Prune at epochs 3, 6, 9, 12
PRUNE_PERCENT = 0.10  # Remove 10% each time
LR_REDUCTION_AFTER_PRUNE = 0.5
L1_LAMBDA = 1e-4  # Sparsity penalty
IMPORTANCE_CAL_BATCHES = 50

# Prune-then-train configuration (target dimensions after 4 prunes)
# These will be calculated dynamically based on model architecture

# Experimental configuration (MATCHING CNN SCRIPT)
NUM_TRIALS = 1

os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== UTILITIES ====================

def set_seed(seed=SEED):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup_memory():
    """Clean up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def count_parameters(model):
    """Count trainable parameters in millions"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def model_size_mb(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2

def compute_flops_vit(model, num_patches=196, seq_len=197):
    """Estimate FLOPs for Vision Transformer"""
    if hasattr(model, 'base_model'):
        h = model.num_heads
        d = model.hidden_dim
        L = model.num_layers
        # Get MLP dimension
        mlp_dim = model.base_model.blocks[0].mlp.fc1.out_features
    else:
        h = model.blocks[0].attn.num_heads
        d = model.embed_dim
        L = len(model.blocks)
        mlp_dim = model.blocks[0].mlp.fc1.out_features
    
    n = seq_len
    
    # MHSA FLOPs per layer
    mhsa_flops = (3 * n * d * d) + (2 * n * n * d) + (n * d * d)
    
    # FFN FLOPs per layer
    ffn_flops = 2 * n * d * mlp_dim
    
    # Total
    total_flops = L * (mhsa_flops + ffn_flops)
    
    return total_flops

# ==================== ENERGY TRACKING ====================

def start_energy_tracker(save_dir, project_name):
    """Start energy tracking"""
    if not CODECARBON_AVAILABLE:
        return None
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=save_dir,
        log_level="error"
    )
    tracker.start()
    return tracker

def stop_energy_tracker(tracker, save_dir, project_name):
    """Stop energy tracking"""
    if tracker is None:
        return {
            'energy_kwh': float('nan'),
            'emissions_kg': float('nan'),
            'duration_s': float('nan')
        }
    emissions = tracker.stop()
    return {
        'energy_kwh': emissions,
        'emissions_kg': emissions * 0.475,
        'duration_s': tracker._total_duration.total_seconds() if hasattr(tracker, '_total_duration') else 0
    }

# ==================== DATASET ====================

class NumpyMemmapDataset(Dataset):
    """Dataset wrapper for NPZ files"""
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
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

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

def load_dataset(npz_path):
    """Load dataset from NPZ file"""
    print(f"Loading {npz_path}...")
    data = np.load(npz_path, mmap_mode="r")

    X_train = data["train_images"]
    y_train = data["train_labels"].flatten()
    X_val = data["val_images"]
    y_val = data["val_labels"].flatten()
    X_test = data["test_images"]
    y_test = data["test_labels"].flatten()

    print(f"Dataset sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

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

# ==================== METRICS ====================

def evaluate_model(net, test_loader, device):
    """Evaluate model"""
    net.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
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
    
    # Calculate AUC
    conf_matrix = confusion_matrix(all_labels, all_preds)
    n_classes = len(conf_matrix)
    all_labels_one_hot = label_binarize(all_labels, classes=list(range(n_classes)))
    
    try:
        auc = roc_auc_score(all_labels_one_hot, all_probs, multi_class='ovr')
    except ValueError:
        auc = float('nan')
    
    return acc, auc, precision, recall, f1

# ==================== PRUNABLE VIT WITH TIMM ====================

class PrunableViTWrapper(nn.Module):
    """Wrapper around TIMM ViT models to add learnable importance scores"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Extract ViT configuration
        self.num_layers = len(base_model.blocks)
        self.hidden_dim = base_model.embed_dim
        self.num_heads = base_model.blocks[0].attn.num_heads
        
        # Add learnable importance scores
        self.head_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.num_heads))
            for _ in range(self.num_layers)
        ])
        
        self.embed_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.hidden_dim))
            for _ in range(self.num_layers)
        ])
        
        # For FFN (MLP) importance
        mlp_hidden_dim = base_model.blocks[0].mlp.fc1.out_features
        self.mlp_importance = nn.ParameterList([
            nn.Parameter(torch.ones(mlp_hidden_dim))
            for _ in range(self.num_layers)
        ])
        
        # Register hooks
        self._register_importance_hooks()
    
    def _register_importance_hooks(self):
        """Register forward hooks to apply importance scores"""
        for i, block in enumerate(self.base_model.blocks):
            # Hook for attention
            def make_attn_hook(layer_idx):
                def hook(module, input, output):
                    B, N, C = output.shape
                    num_heads = self.num_heads
                    head_dim = C // num_heads
                    
                    output_reshaped = output.view(B, N, num_heads, head_dim)
                    head_weights = self.head_importance[layer_idx].view(1, 1, num_heads, 1)
                    output_weighted = output_reshaped * head_weights
                    
                    return output_weighted.view(B, N, C)
                return hook
            
            # Hook for MLP
            def make_mlp_hook(layer_idx):
                def hook(module, input, output):
                    return output * self.mlp_importance[layer_idx].view(1, 1, -1)
                return hook
            
            block.attn.proj.register_forward_hook(make_attn_hook(i))
            block.mlp.fc2.register_forward_hook(make_mlp_hook(i))
    
    def forward(self, x):
        return self.base_model(x)
    
    def get_importance_scores(self):
        """Collect all importance scores"""
        return {
            'heads': [imp.data.clone() for imp in self.head_importance],
            'mlp': [imp.data.clone() for imp in self.mlp_importance],
            'embed': [imp.data.clone() for imp in self.embed_importance]
        }
    
    def apply_mask(self, mask_dict):
        """Apply binary masks to prune dimensions"""
        if 'heads' in mask_dict:
            for i, mask in enumerate(mask_dict['heads']):
                self.head_importance[i].data *= mask
        if 'mlp' in mask_dict:
            for i, mask in enumerate(mask_dict['mlp']):
                self.mlp_importance[i].data *= mask
        if 'embed' in mask_dict:
            for i, mask in enumerate(mask_dict['embed']):
                self.embed_importance[i].data *= mask

def create_prunable_vit(model_name, num_classes, pretrained=True):
    """Create a prunable ViT model using TIMM"""
    print(f"Creating {model_name} (pretrained={pretrained})...")
    
    base_model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    model = PrunableViTWrapper(base_model)
    
    return model

# ==================== IMPORTANCE COMPUTATION ====================

def compute_importance_taylor(model, data_loader, max_batches=50):
    """Compute importance using Taylor expansion"""
    model.eval()
    
    num_layers = model.num_layers
    num_heads = model.num_heads
    hidden_dim = model.hidden_dim
    mlp_dim = model.mlp_importance[0].shape[0]
    
    # Initialize accumulators
    importance = {
        'heads': [torch.zeros(num_heads).to(DEVICE) for _ in range(num_layers)],
        'mlp': [torch.zeros(mlp_dim).to(DEVICE) for _ in range(num_layers)],
        'embed': [torch.zeros(hidden_dim).to(DEVICE) for _ in range(num_layers)]
    }
    
    # Compute gradients
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        
        # Accumulate importance = |weight × gradient|
        for i in range(num_layers):
            if model.head_importance[i].grad is not None:
                importance['heads'][i] += torch.abs(
                    model.head_importance[i].data * model.head_importance[i].grad.data
                )
            if model.mlp_importance[i].grad is not None:
                importance['mlp'][i] += torch.abs(
                    model.mlp_importance[i].data * model.mlp_importance[i].grad.data
                )
            if model.embed_importance[i].grad is not None:
                importance['embed'][i] += torch.abs(
                    model.embed_importance[i].data * model.embed_importance[i].grad.data
                )
    
    # Average
    for key in importance:
        importance[key] = [imp / max_batches for imp in importance[key]]
    
    return importance

def prune_model_progressive(importance, prune_ratio):
    """Generate pruning masks based on importance scores"""
    masks = {
        'heads': [],
        'mlp': [],
        'embed': []
    }
    
    # Prune each component
    for layer_importance in importance['heads']:
        threshold = torch.quantile(layer_importance, prune_ratio)
        mask = (layer_importance >= threshold).float()
        masks['heads'].append(mask)
    
    for layer_importance in importance['mlp']:
        threshold = torch.quantile(layer_importance, prune_ratio)
        mask = (layer_importance >= threshold).float()
        masks['mlp'].append(mask)
    
    # For embedding, use global threshold
    all_embed = torch.cat([imp for imp in importance['embed']])
    threshold = torch.quantile(all_embed, prune_ratio)
    for layer_importance in importance['embed']:
        mask = (layer_importance >= threshold).float()
        masks['embed'].append(mask)
    
    return masks

# ==================== TRAINING ====================

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, l1_lambda=0):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Cross-entropy loss
        ce_loss = nn.functional.cross_entropy(outputs, labels)
        
        # L1 regularization on importance scores
        l1_loss = 0
        if l1_lambda > 0:
            for imp in model.head_importance:
                l1_loss += torch.sum(torch.abs(imp))
            for imp in model.mlp_importance:
                l1_loss += torch.sum(torch.abs(imp))
            for imp in model.embed_importance:
                l1_loss += torch.sum(torch.abs(imp))
        
        loss = ce_loss + l1_lambda * l1_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), correct / total

# ==================== MAIN TRAINING FUNCTIONS ====================

def train_baseline_vit(dataset_name, model_name, train_loader, val_loader, test_loader, 
                       num_classes, save_dir, trial_num):
    """
    METHOD 1: Baseline ViT without pruning
    Saves BEST validation accuracy model
    """
    print("\n" + "="*80)
    print(f"METHOD 1: BASELINE - {dataset_name.upper()} - {model_name} - TRIAL {trial_num}")
    print("="*80)
    
    # Create model
    model = create_prunable_vit(model_name, num_classes, pretrained=True).to(DEVICE)
    print(f"Parameters: {count_parameters(model):.2f}M")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    
    # Scheduler
    total_steps = FIXED_EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)
    
    # Track metrics
    history = []
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Start energy tracking
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_baseline_trial{trial_num}")
    
    print(f"\nTraining for exactly {FIXED_EPOCHS} epochs")
    
    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, DEVICE)
        
        # Test (for per-epoch tracking)
        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, DEVICE)
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Track best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"  ✓ New best validation accuracy: {val_acc:.4f}")
        
        # Record metrics
        history.append({
            'trial': trial_num,
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })
    
    # Stop energy tracking
    energy_metrics = stop_energy_tracker(tracker, save_dir, 
                                         f"{dataset_name}_{model_name}_baseline_trial{trial_num}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch} (val_acc: {best_val_acc:.4f})")
    
    # Save best model
    model_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_baseline_trial{trial_num}_final.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_baseline_trial{trial_num}_history.csv")
    history_df.to_csv(history_path, index=False)
    
    # Final test evaluation
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, DEVICE)
    
    params_m = count_parameters(model)
    size_mb = model_size_mb(model)
    flops = compute_flops_vit(model)
    
    final_metrics = {
        'method': 'baseline',
        'dataset': dataset_name,
        'model': model_name,
        'trial': trial_num,
        'total_epochs': FIXED_EPOCHS,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'params_m': params_m,
        'model_size_mb': size_mb,
        'flops_g': flops / 1e9,
        'training_energy_kwh': energy_metrics['energy_kwh'],
        'training_emissions_kg': energy_metrics['emissions_kg'],
        'num_heads': model.num_heads,
        'hidden_dim': model.hidden_dim
    }
    
    print(f"\nFinal Test - Acc: {test_acc:.4f}, Precision: {test_precision:.4f}")
    
    return model, final_metrics

def train_progressive_pruning_vit(dataset_name, model_name, train_loader, val_loader, test_loader,
                                  num_classes, save_dir, trial_num):
    """
    METHOD 2: Progressive Pruning - Prune during training
    Saves BEST validation accuracy model
    """
    print("\n" + "="*80)
    print(f"METHOD 2: PROGRESSIVE PRUNING - {dataset_name.upper()} - {model_name} - TRIAL {trial_num}")
    print("="*80)
    
    # Pruning schedule
    prune_epochs = [WARMUP_EPOCHS + i * EPOCHS_BETWEEN_PRUNES for i in range(1, NUM_PRUNE_STEPS + 1)]
    print(f"Fixed schedule: {FIXED_EPOCHS} epochs total")
    print(f"Pruning at epochs: {prune_epochs}")
    
    # Create model
    model = create_prunable_vit(model_name, num_classes, pretrained=True).to(DEVICE)
    print(f"Initial parameters: {count_parameters(model):.2f}M")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    current_lr = INITIAL_LR
    
    # Track metrics
    all_metrics = []
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Start energy tracking
    tracker = start_energy_tracker(save_dir, 
                                   f"{dataset_name}_{model_name}_progressive_trial{trial_num}")
    
    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")
        
        # Check if pruning epoch
        if epoch in prune_epochs:
            prune_step = prune_epochs.index(epoch) + 1
            print(f"\n*** PRUNING STEP {prune_step}/{NUM_PRUNE_STEPS} ***")
            
            # Compute importance
            print("Computing importance scores...")
            importance = compute_importance_taylor(model, train_loader, max_batches=IMPORTANCE_CAL_BATCHES)
            
            # Generate masks
            masks = prune_model_progressive(importance, PRUNE_PERCENT)
            
            # Apply masks
            model.apply_mask(masks)
            
            # Report
            for i in range(min(3, model.num_layers)):  # Show first 3 layers
                heads_kept = masks['heads'][i].sum().item()
                mlp_kept = masks['mlp'][i].sum().item()
                embed_kept = masks['embed'][i].sum().item()
                print(f"  Layer {i}: heads={heads_kept:.0f}/{model.num_heads}, "
                      f"mlp={mlp_kept:.0f}/{model.mlp_importance[0].shape[0]}, "
                      f"embed={embed_kept:.0f}/{model.hidden_dim}")
            
            # Reduce LR
            current_lr *= LR_REDUCTION_AFTER_PRUNE
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"  LR reduced to {current_lr:.6f}")
        
        # Setup scheduler
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch, eta_min=MIN_LR)
        
        # Train with L1
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, l1_lambda=L1_LAMBDA)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, DEVICE)
        
        # Test
        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, DEVICE)
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"  ✓ New best validation accuracy: {val_acc:.4f}")
        
        # Record metrics
        params_m = count_parameters(model)
        all_metrics.append({
            'trial': trial_num,
            'epoch': epoch,
            'stage': 'after_training' if epoch in prune_epochs else 'training',
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'params_m': params_m,
            'lr': current_lr
        })
    
    # Stop energy tracking
    energy_metrics = stop_energy_tracker(tracker, save_dir,
                                         f"{dataset_name}_{model_name}_progressive_trial{trial_num}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch} (val_acc: {best_val_acc:.4f})")
    
    # Save best model
    model_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_progressive_trial{trial_num}_final.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_progressive_trial{trial_num}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # Final test
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, DEVICE)
    
    # Get effective dimensions
    importance_scores = model.get_importance_scores()
    effective_heads = sum((imp > 0).sum().item() for imp in importance_scores['heads']) / model.num_layers
    effective_mlp = sum((imp > 0).sum().item() for imp in importance_scores['mlp']) / model.num_layers
    effective_embed = sum((imp > 0).sum().item() for imp in importance_scores['embed']) / model.num_layers
    
    params_m = count_parameters(model)
    size_mb = model_size_mb(model)
    flops = compute_flops_vit(model)
    
    final_metrics = {
        'method': 'progressive_pruning',
        'dataset': dataset_name,
        'model': model_name,
        'trial': trial_num,
        'total_epochs': FIXED_EPOCHS,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'params_m': params_m,
        'model_size_mb': size_mb,
        'flops_g': flops / 1e9,
        'training_energy_kwh': energy_metrics['energy_kwh'],
        'training_emissions_kg': energy_metrics['emissions_kg'],
        'effective_heads': effective_heads,
        'effective_mlp': effective_mlp,
        'effective_embed': effective_embed,
        'total_prune_steps': NUM_PRUNE_STEPS
    }
    
    print(f"\nFinal Test - Acc: {test_acc:.4f}, Precision: {test_precision:.4f}")
    print(f"Effective: heads={effective_heads:.1f}, mlp={effective_mlp:.0f}, embed={effective_embed:.0f}")
    
    return model, final_metrics

def train_prune_then_finetune_vit(dataset_name, model_name, train_loader, val_loader, test_loader,
                                  num_classes, save_dir, trial_num, target_dims=None):
    """
    METHOD 3: Prune-Then-Train - One-shot pruning at initialization
    Saves BEST validation accuracy model
    """
    print("\n" + "="*80)
    print(f"METHOD 3: PRUNE-THEN-TRAIN - {dataset_name.upper()} - {model_name} - TRIAL {trial_num}")
    print("="*80)
    
    print(f"Fixed schedule: {FIXED_EPOCHS} epochs total")
    print(f"One-shot pruning at initialization")
    
    # Create model
    model = create_prunable_vit(model_name, num_classes, pretrained=True).to(DEVICE)
    print(f"Initial parameters: {count_parameters(model):.2f}M")
    
    # Start energy tracking (includes pruning time)
    tracker = start_energy_tracker(save_dir,
                                   f"{dataset_name}_{model_name}_prune_then_train_trial{trial_num}")
    
    # ONE-SHOT PRUNING
    print("\n*** ONE-SHOT PRUNING ***")
    print("Computing importance scores...")
    
    importance = compute_importance_taylor(model, train_loader, max_batches=IMPORTANCE_CAL_BATCHES)
    
    # Calculate cumulative prune ratio (4 steps of 10% each = 1 - 0.9^4 = 0.3439)
    cumulative_prune_ratio = 1 - (1 - PRUNE_PERCENT) ** NUM_PRUNE_STEPS
    
    # Generate masks for one-shot pruning
    masks = prune_model_progressive(importance, cumulative_prune_ratio)
    
    # Apply masks
    model.apply_mask(masks)
    
    # Report
    for i in range(min(3, model.num_layers)):
        heads_kept = masks['heads'][i].sum().item()
        mlp_kept = masks['mlp'][i].sum().item()
        embed_kept = masks['embed'][i].sum().item()
        print(f"  Layer {i}: heads={heads_kept:.0f}/{model.num_heads}, "
              f"mlp={mlp_kept:.0f}/{model.mlp_importance[0].shape[0]}, "
              f"embed={embed_kept:.0f}/{model.hidden_dim}")
    
    print(f"\nPruned model created. Now training for {FIXED_EPOCHS} epochs...")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    
    # Scheduler
    total_steps = FIXED_EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)
    
    # Track metrics
    all_metrics = []
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, DEVICE)
        
        # Test
        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, DEVICE)
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"  ✓ New best validation accuracy: {val_acc:.4f}")
        
        # Record metrics
        params_m = count_parameters(model)
        all_metrics.append({
            'trial': trial_num,
            'epoch': epoch,
            'stage': 'training',
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'params_m': params_m
        })
    
    # Stop energy tracking
    energy_metrics = stop_energy_tracker(tracker, save_dir,
                                         f"{dataset_name}_{model_name}_prune_then_train_trial{trial_num}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch} (val_acc: {best_val_acc:.4f})")
    
    # Save best model
    model_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_prune_then_train_trial{trial_num}_final.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_prune_then_train_trial{trial_num}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # Final test
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, DEVICE)
    
    # Get effective dimensions
    importance_scores = model.get_importance_scores()
    effective_heads = sum((imp > 0).sum().item() for imp in importance_scores['heads']) / model.num_layers
    effective_mlp = sum((imp > 0).sum().item() for imp in importance_scores['mlp']) / model.num_layers
    effective_embed = sum((imp > 0).sum().item() for imp in importance_scores['embed']) / model.num_layers
    
    params_m = count_parameters(model)
    size_mb = model_size_mb(model)
    flops = compute_flops_vit(model)
    
    final_metrics = {
        'method': 'prune_then_train',
        'dataset': dataset_name,
        'model': model_name,
        'trial': trial_num,
        'total_epochs': FIXED_EPOCHS,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'params_m': params_m,
        'model_size_mb': size_mb,
        'flops_g': flops / 1e9,
        'training_energy_kwh': energy_metrics['energy_kwh'],
        'training_emissions_kg': energy_metrics['emissions_kg'],
        'effective_heads': effective_heads,
        'effective_mlp': effective_mlp,
        'effective_embed': effective_embed
    }
    
    print(f"\nFinal Test - Acc: {test_acc:.4f}, Precision: {test_precision:.4f}")
    print(f"Effective: heads={effective_heads:.1f}, mlp={effective_mlp:.0f}, embed={effective_embed:.0f}")
    
    return model, final_metrics

# ==================== PROCESS DATASET ====================

def process_dataset(dataset_name, train_loader, val_loader, test_loader, num_classes, save_dir):
    """
    Process one dataset through all three methods and all trials
    MATCHING CNN script structure
    """
    print("\n" + "="*100)
    print(f"PROCESSING DATASET: {dataset_name.upper()}")
    print("="*100)
    
    all_baseline_metrics = []
    all_progressive_metrics = []
    all_prune_then_train_metrics = []
    
    # Run multiple trials
    for trial in range(1, NUM_TRIALS + 1):
        print(f"\n{'~'*100}")
        print(f"~ TRIAL {trial}/{NUM_TRIALS}")
        print(f"{'~'*100}")
        
        # Set different seed for each trial
        trial_seed = SEED + trial * 100
        set_seed(trial_seed)
        
        # Phase 1: Progressive pruning
        _, progressive_metrics = train_progressive_pruning_vit(
            dataset_name, 'vit_tiny_patch16_224', train_loader, val_loader, test_loader,
            num_classes, save_dir, trial
        )
        all_progressive_metrics.append(progressive_metrics)
        cleanup_memory()
        
        # Phase 2: Baseline
        _, baseline_metrics = train_baseline_vit(
            dataset_name, 'vit_tiny_patch16_224', train_loader, val_loader, test_loader,
            num_classes, save_dir, trial
        )
        all_baseline_metrics.append(baseline_metrics)
        cleanup_memory()
        
        # Phase 3: Prune-then-train
        _, prune_then_train_metrics = train_prune_then_finetune_vit(
            dataset_name, 'vit_tiny_patch16_224', train_loader, val_loader, test_loader,
            num_classes, save_dir, trial
        )
        all_prune_then_train_metrics.append(prune_then_train_metrics)
        cleanup_memory()
        
        print(f"\n✓ Trial {trial} completed")
    
    # Save combined results
    all_metrics_df = pd.DataFrame(all_baseline_metrics + all_progressive_metrics + all_prune_then_train_metrics)
    all_metrics_path = os.path.join(save_dir, f"{dataset_name}_all_trials_metrics.csv")
    all_metrics_df.to_csv(all_metrics_path, index=False)
    print(f"\nSaved all trials metrics to {all_metrics_path}")
    
    # Compute summary statistics
    baseline_df = pd.DataFrame(all_baseline_metrics)
    progressive_df = pd.DataFrame(all_progressive_metrics)
    prune_then_train_df = pd.DataFrame(all_prune_then_train_metrics)
    
    summary_rows = []
    
    # Baseline summary
    baseline_summary = {
        'method': 'baseline',
        'dataset': dataset_name,
        'num_trials': NUM_TRIALS,
        'fixed_epochs': FIXED_EPOCHS
    }
    for col in ['test_acc', 'test_loss', 'test_precision', 'test_recall', 'test_f1',
                'params_m', 'model_size_mb', 'flops_g', 'training_energy_kwh']:
        if col in baseline_df.columns:
            baseline_summary[f'{col}_mean'] = baseline_df[col].mean()
            baseline_summary[f'{col}_std'] = baseline_df[col].std()
    summary_rows.append(baseline_summary)
    
    # Progressive summary
    progressive_summary = {
        'method': 'progressive_pruning',
        'dataset': dataset_name,
        'num_trials': NUM_TRIALS,
        'fixed_epochs': FIXED_EPOCHS
    }
    for col in ['test_acc', 'test_loss', 'test_precision', 'test_recall', 'test_f1',
                'params_m', 'model_size_mb', 'flops_g', 'training_energy_kwh',
                'effective_heads', 'effective_mlp', 'effective_embed']:
        if col in progressive_df.columns:
            progressive_summary[f'{col}_mean'] = progressive_df[col].mean()
            progressive_summary[f'{col}_std'] = progressive_df[col].std()
    summary_rows.append(progressive_summary)
    
    # Prune-then-train summary
    ptt_summary = {
        'method': 'prune_then_train',
        'dataset': dataset_name,
        'num_trials': NUM_TRIALS,
        'fixed_epochs': FIXED_EPOCHS
    }
    for col in ['test_acc', 'test_loss', 'test_precision', 'test_recall', 'test_f1',
                'params_m', 'model_size_mb', 'flops_g', 'training_energy_kwh',
                'effective_heads', 'effective_mlp', 'effective_embed']:
        if col in prune_then_train_df.columns:
            ptt_summary[f'{col}_mean'] = prune_then_train_df[col].mean()
            ptt_summary[f'{col}_std'] = prune_then_train_df[col].std()
    summary_rows.append(ptt_summary)
    
    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(save_dir, f"{dataset_name}_summary_statistics.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Print comparison
    print("\n" + "="*140)
    print(f"SUMMARY - {dataset_name.upper()} ({NUM_TRIALS} trials, {FIXED_EPOCHS} epochs each)")
    print("="*140)
    print(f"{'Metric':<30} {'Baseline (Mean±Std)':<35} {'Progressive (Mean±Std)':<35} {'Prune-Then-Train (Mean±Std)':<35}")
    print("-"*140)
    
    for col in ['test_acc', 'test_precision', 'params_m', 'flops_g', 'training_energy_kwh']:
        base_mean = baseline_summary.get(f'{col}_mean', float('nan'))
        base_std = baseline_summary.get(f'{col}_std', float('nan'))
        prog_mean = progressive_summary.get(f'{col}_mean', float('nan'))
        prog_std = progressive_summary.get(f'{col}_std', float('nan'))
        ptt_mean = ptt_summary.get(f'{col}_mean', float('nan'))
        ptt_std = ptt_summary.get(f'{col}_std', float('nan'))
        
        base_str = f"{base_mean:.4f} ± {base_std:.4f}"
        prog_str = f"{prog_mean:.4f} ± {prog_std:.4f}"
        ptt_str = f"{ptt_mean:.4f} ± {ptt_std:.4f}"
        
        print(f"{col:<30} {base_str:<35} {prog_str:<35} {ptt_str:<35}")
    
    print("="*140)

# ==================== MAIN ====================

def main():
    """Main entry point - MATCHING CNN script structure"""
    set_seed(SEED)
    
    print("="*100)
    print("PROGRESSIVE VISION TRANSFORMER PRUNING")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Fixed Epochs: {FIXED_EPOCHS}")
    print(f"  Pruning Schedule: {[WARMUP_EPOCHS + i * EPOCHS_BETWEEN_PRUNES for i in range(1, NUM_PRUNE_STEPS + 1)]}")
    print(f"  Number of Trials: {NUM_TRIALS}")
    print(f"\n  Method 1: Baseline (No pruning)")
    print(f"  Method 2: Progressive Pruning (Prune at epochs 3, 6, 9, 12)")
    print(f"  Method 3: Prune-Then-Train (One-shot pruning at start)")
    
    # Datasets
    datasets = ['bloodmnist', 'pathmnist', 'dermamnist']
    
    for dataset in datasets:
        npz_path = os.path.join(DATASET_DIR, f"{dataset}_224.npz")
        
        if not os.path.exists(npz_path):
            print(f"\nDataset not found: {npz_path}")
            continue
        
        # Load dataset
        train_loader, val_loader, test_loader, num_classes, dataset_name = load_dataset(npz_path)
        
        # Process through all methods
        process_dataset(dataset_name, train_loader, val_loader, test_loader, num_classes, SAVE_DIR)
    
    print("\n" + "="*100)
    print("EXPERIMENT COMPLETED")
    print("="*100)
    print(f"\nKey Takeaways:")
    print(f"  - All three methods trained for exactly {FIXED_EPOCHS} epochs")
    print(f"  - All methods used same regularization and pretrained weights")
    print(f"  - All methods saved BEST validation accuracy model")
    print(f"  - Results averaged over {NUM_TRIALS} trials for statistical reliability")
    print("="*100)

if __name__ == "__main__":
    main()


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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import pandas as pd
import copy
import gc

# Try codecarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("CodeCarbon not available - energy metrics will be NaN")

# ==================== CONFIGURATION ====================

DATASET_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets_balanced"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/Vision/vit_progressive_pruning"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMG_SIZE = 224

# Training configuration
FIXED_EPOCHS = 15
BATCH_SIZE = 64
INITIAL_LR = 1e-4  # Lower LR for finetuning pretrained models
WEIGHT_DECAY = 0.05
MIN_LR = 1e-6
EARLY_STOP_PATIENCE = 10

# Progressive pruning configuration
WARMUP_EPOCHS = 2
EPOCHS_BETWEEN_PRUNES = 3
NUM_PRUNE_STEPS = 4  # Prune at epochs 3, 6, 9, 12
PRUNE_PERCENT = 0.10  # Remove 10% each time
LR_REDUCTION_AFTER_PRUNE = 0.5
L1_LAMBDA = 1e-4  # Sparsity penalty
IMPORTANCE_CAL_BATCHES = 50

# Experimental configuration
NUM_TRIALS = 3

os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== UTILITIES ====================

def set_seed(seed=SEED):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup_memory():
    """Clean up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def count_parameters(model):
    """Count trainable parameters in millions"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def model_size_mb(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2

# ==================== ENERGY TRACKING ====================

def start_energy_tracker(save_dir, project_name):
    """Start energy tracking"""
    if not CODECARBON_AVAILABLE:
        return None
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=save_dir,
        log_level="error"
    )
    tracker.start()
    return tracker

def stop_energy_tracker(tracker, save_dir, project_name):
    """Stop energy tracking"""
    if tracker is None:
        return {
            'energy_kwh': float('nan'),
            'emissions_kg': float('nan'),
            'duration_s': float('nan')
        }
    emissions = tracker.stop()
    return {
        'energy_kwh': emissions,
        'emissions_kg': emissions * 0.475,
        'duration_s': tracker._total_duration.total_seconds() if hasattr(tracker, '_total_duration') else 0
    }

# ==================== DATASET ====================

class NumpyMemmapDataset(Dataset):
    """Dataset wrapper for NPZ files (from your baseline code)"""
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
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

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

def load_dataset(npz_path):
    """Load dataset from NPZ file"""
    print(f"Loading {npz_path}...")
    data = np.load(npz_path, mmap_mode="r")

    X_train = data["train_images"]
    y_train = data["train_labels"].flatten()
    X_val = data["val_images"]
    y_val = data["val_labels"].flatten()
    X_test = data["test_images"]
    y_test = data["test_labels"].flatten()

    print(f"Dataset sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

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

# ==================== METRICS ====================

def evaluate_model(net, test_loader, device):
    """Evaluate model (from your baseline code)"""
    net.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
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
    
    # Calculate AUC
    conf_matrix = confusion_matrix(all_labels, all_preds)
    n_classes = len(conf_matrix)
    all_labels_one_hot = label_binarize(all_labels, classes=list(range(n_classes)))
    
    try:
        auc = roc_auc_score(all_labels_one_hot, all_probs, multi_class='ovr')
    except ValueError:
        auc = float('nan')
    
    return {
        'acc': acc,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ==================== PRUNABLE VIT WITH TIMM ====================

class PrunableViTWrapper(nn.Module):
    """
    Wrapper around TIMM ViT models to add learnable importance scores
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Extract ViT configuration
        self.num_layers = len(base_model.blocks)
        self.hidden_dim = base_model.embed_dim
        self.num_heads = base_model.blocks[0].attn.num_heads
        
        # Add learnable importance scores for each transformer block
        self.head_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.num_heads))
            for _ in range(self.num_layers)
        ])
        
        self.embed_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.hidden_dim))
            for _ in range(self.num_layers)
        ])
        
        # For FFN (MLP) importance
        mlp_hidden_dim = base_model.blocks[0].mlp.fc1.out_features
        self.mlp_importance = nn.ParameterList([
            nn.Parameter(torch.ones(mlp_hidden_dim))
            for _ in range(self.num_layers)
        ])
        
        # Register hooks to apply importance during forward pass
        self._register_importance_hooks()
    
    def _register_importance_hooks(self):
        """Register forward hooks to apply importance scores"""
        for i, block in enumerate(self.base_model.blocks):
            # Hook for attention
            def make_attn_hook(layer_idx):
                def hook(module, input, output):
                    # Apply head importance
                    B, N, C = output.shape
                    num_heads = self.num_heads
                    head_dim = C // num_heads
                    
                    # Reshape to separate heads
                    output_reshaped = output.view(B, N, num_heads, head_dim)
                    
                    # Apply head importance
                    head_weights = self.head_importance[layer_idx].view(1, 1, num_heads, 1)
                    output_weighted = output_reshaped * head_weights
                    
                    # Reshape back
                    return output_weighted.view(B, N, C)
                return hook
            
            # Hook for MLP
            def make_mlp_hook(layer_idx):
                def hook(module, input, output):
                    # Apply MLP hidden dimension importance
                    return output * self.mlp_importance[layer_idx].view(1, 1, -1)
                return hook
            
            # Register hooks
            block.attn.proj.register_forward_hook(make_attn_hook(i))
            block.mlp.fc2.register_forward_hook(make_mlp_hook(i))
    
    def forward(self, x):
        return self.base_model(x)
    
    def get_importance_scores(self):
        """Collect all importance scores"""
        return {
            'heads': [imp.data.clone() for imp in self.head_importance],
            'mlp': [imp.data.clone() for imp in self.mlp_importance],
            'embed': [imp.data.clone() for imp in self.embed_importance]
        }
    
    def apply_mask(self, mask_dict):
        """Apply binary masks to prune dimensions"""
        if 'heads' in mask_dict:
            for i, mask in enumerate(mask_dict['heads']):
                self.head_importance[i].data *= mask
        if 'mlp' in mask_dict:
            for i, mask in enumerate(mask_dict['mlp']):
                self.mlp_importance[i].data *= mask
        if 'embed' in mask_dict:
            for i, mask in enumerate(mask_dict['embed']):
                self.embed_importance[i].data *= mask

def create_prunable_vit(model_name, num_classes, pretrained=True):
    """Create a prunable ViT model using TIMM"""
    print(f"Creating {model_name} (pretrained={pretrained})...")
    
    # Load base model from TIMM
    base_model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    # Wrap with prunable version
    model = PrunableViTWrapper(base_model)
    
    return model

# ==================== IMPORTANCE COMPUTATION ====================

def compute_importance_taylor(model, data_loader, max_batches=50):
    """
    Compute importance using Taylor expansion (gradient × weight)
    """
    model.eval()
    
    num_layers = model.num_layers
    num_heads = model.num_heads
    hidden_dim = model.hidden_dim
    mlp_dim = model.mlp_importance[0].shape[0]
    
    # Initialize accumulators
    importance = {
        'heads': [torch.zeros(num_heads).to(DEVICE) for _ in range(num_layers)],
        'mlp': [torch.zeros(mlp_dim).to(DEVICE) for _ in range(num_layers)],
        'embed': [torch.zeros(hidden_dim).to(DEVICE) for _ in range(num_layers)]
    }
    
    # Compute gradients
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        
        # Accumulate importance = |weight × gradient|
        for i in range(num_layers):
            if model.head_importance[i].grad is not None:
                importance['heads'][i] += torch.abs(
                    model.head_importance[i].data * model.head_importance[i].grad.data
                )
            if model.mlp_importance[i].grad is not None:
                importance['mlp'][i] += torch.abs(
                    model.mlp_importance[i].data * model.mlp_importance[i].grad.data
                )
            if model.embed_importance[i].grad is not None:
                importance['embed'][i] += torch.abs(
                    model.embed_importance[i].data * model.embed_importance[i].grad.data
                )
    
    # Average
    for key in importance:
        importance[key] = [imp / max_batches for imp in importance[key]]
    
    return importance

def prune_model_progressive(importance, prune_ratio):
    """Generate pruning masks based on importance scores"""
    masks = {
        'heads': [],
        'mlp': [],
        'embed': []
    }
    
    # Prune each component
    for layer_importance in importance['heads']:
        threshold = torch.quantile(layer_importance, prune_ratio)
        mask = (layer_importance >= threshold).float()
        masks['heads'].append(mask)
    
    for layer_importance in importance['mlp']:
        threshold = torch.quantile(layer_importance, prune_ratio)
        mask = (layer_importance >= threshold).float()
        masks['mlp'].append(mask)
    
    # For embedding, use global threshold across all layers
    all_embed = torch.cat([imp for imp in importance['embed']])
    threshold = torch.quantile(all_embed, prune_ratio)
    for layer_importance in importance['embed']:
        mask = (layer_importance >= threshold).float()
        masks['embed'].append(mask)
    
    return masks

# ==================== TRAINING ====================

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, l1_lambda=0):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    train_bar = tqdm(train_loader, file=sys.stdout, desc=f"Epoch {epoch}")
    
    for images, labels in train_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Cross-entropy loss
        ce_loss = nn.functional.cross_entropy(outputs, labels)
        
        # L1 regularization on importance scores
        l1_loss = 0
        if l1_lambda > 0:
            for imp in model.head_importance:
                l1_loss += torch.sum(torch.abs(imp))
            for imp in model.mlp_importance:
                l1_loss += torch.sum(torch.abs(imp))
            for imp in model.embed_importance:
                l1_loss += torch.sum(torch.abs(imp))
        
        loss = ce_loss + l1_lambda * l1_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.3f}")
    
    return running_loss / len(train_loader)

# ==================== MAIN TRAINING FUNCTIONS ====================

def train_baseline_vit(dataset_name, model_name, train_loader, val_loader, test_loader, 
                       num_classes, save_dir, trial_num):
    """Train baseline ViT without pruning"""
    print("\n" + "="*80)
    print(f"BASELINE VIT - {dataset_name.upper()} - {model_name} - TRIAL {trial_num}")
    print("="*80)
    
    # Create model
    model = create_prunable_vit(model_name, num_classes, pretrained=True).to(DEVICE)
    print(f"Parameters: {count_parameters(model):.2f}M")
    
    # Optimizer (lower LR for finetuning)
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    total_steps = FIXED_EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)
    
    # Track metrics
    history = []
    
    # Start energy tracking
    tracker = start_energy_tracker(save_dir, f"{dataset_name}_{model_name}_baseline_trial{trial_num}")
    
    print(f"\nTraining for {FIXED_EPOCHS} epochs...")
    
    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
        
        # Evaluate
        metrics = evaluate_model(model, val_loader, DEVICE)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Acc: {metrics['acc']:.4f}, AUC: {metrics['auc']:.4f}")
        
        history.append({
            'trial': trial_num,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_acc': metrics['acc'],
            'val_auc': metrics['auc']
        })
    
    # Stop energy tracking
    energy_metrics = stop_energy_tracker(tracker, save_dir, 
                                         f"{dataset_name}_{model_name}_baseline_trial{trial_num}")
    
    # Save model
    model_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_baseline_trial{trial_num}.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_baseline_trial{trial_num}_history.csv")
    history_df.to_csv(history_path, index=False)
    
    # Final test
    test_metrics = evaluate_model(model, test_loader, DEVICE)
    
    final_metrics = {
        'method': 'baseline',
        'dataset': dataset_name,
        'model': model_name,
        'trial': trial_num,
        'test_acc': test_metrics['acc'],
        'test_auc': test_metrics['auc'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'params_m': count_parameters(model),
        'model_size_mb': model_size_mb(model),
        'training_energy_kwh': energy_metrics['energy_kwh']
    }
    
    print(f"\nFinal Test - Acc: {test_metrics['acc']:.4f}, AUC: {test_metrics['auc']:.4f}")
    
    return model, final_metrics

def train_progressive_pruning_vit(dataset_name, model_name, train_loader, val_loader, test_loader,
                                  num_classes, save_dir, trial_num):
    """Train ViT with progressive pruning"""
    print("\n" + "="*80)
    print(f"PROGRESSIVE PRUNING - {dataset_name.upper()} - {model_name} - TRIAL {trial_num}")
    print("="*80)
    
    # Pruning schedule
    prune_epochs = [WARMUP_EPOCHS + i * EPOCHS_BETWEEN_PRUNES for i in range(1, NUM_PRUNE_STEPS + 1)]
    print(f"Pruning at epochs: {prune_epochs}")
    
    # Create model
    model = create_prunable_vit(model_name, num_classes, pretrained=True).to(DEVICE)
    print(f"Initial parameters: {count_parameters(model):.2f}M")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    current_lr = INITIAL_LR
    
    # Track metrics
    all_metrics = []
    
    # Start energy tracking
    tracker = start_energy_tracker(save_dir, 
                                   f"{dataset_name}_{model_name}_progressive_trial{trial_num}")
    
    for epoch in range(1, FIXED_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{FIXED_EPOCHS} ---")
        
        # Check if pruning epoch
        if epoch in prune_epochs:
            prune_step = prune_epochs.index(epoch) + 1
            print(f"\n*** PRUNING STEP {prune_step}/{NUM_PRUNE_STEPS} ***")
            
            # Compute importance
            print("Computing importance scores...")
            importance = compute_importance_taylor(model, train_loader, max_batches=IMPORTANCE_CAL_BATCHES)
            
            # Generate masks
            masks = prune_model_progressive(importance, PRUNE_PERCENT)
            
            # Apply masks
            model.apply_mask(masks)
            
            # Report
            for i in range(model.num_layers):
                heads_kept = masks['heads'][i].sum().item()
                mlp_kept = masks['mlp'][i].sum().item()
                print(f"  Layer {i}: heads={heads_kept:.0f}/{model.num_heads}, "
                      f"mlp={mlp_kept:.0f}/{model.mlp_importance[0].shape[0]}")
            
            # Reduce LR
            current_lr *= LR_REDUCTION_AFTER_PRUNE
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"  LR reduced to {current_lr:.6f}")
        
        # Setup scheduler for this epoch
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch, eta_min=MIN_LR)
        
        # Train with L1 regularization
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, l1_lambda=L1_LAMBDA)
        
        # Evaluate
        metrics = evaluate_model(model, val_loader, DEVICE)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Acc: {metrics['acc']:.4f}, AUC: {metrics['auc']:.4f}")
        
        all_metrics.append({
            'trial': trial_num,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_acc': metrics['acc'],
            'val_auc': metrics['auc'],
            'params_m': count_parameters(model),
            'lr': current_lr
        })
    
    # Stop energy tracking
    energy_metrics = stop_energy_tracker(tracker, save_dir,
                                         f"{dataset_name}_{model_name}_progressive_trial{trial_num}")
    
    # Save model
    model_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_progressive_trial{trial_num}.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(save_dir, f"{dataset_name}_{model_name}_progressive_trial{trial_num}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # Final test
    test_metrics = evaluate_model(model, test_loader, DEVICE)
    
    # Get effective dimensions
    importance_scores = model.get_importance_scores()
    effective_heads = sum((imp > 0).sum().item() for imp in importance_scores['heads']) / model.num_layers
    effective_mlp = sum((imp > 0).sum().item() for imp in importance_scores['mlp']) / model.num_layers
    
    final_metrics = {
        'method': 'progressive_pruning',
        'dataset': dataset_name,
        'model': model_name,
        'trial': trial_num,
        'test_acc': test_metrics['acc'],
        'test_auc': test_metrics['auc'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'params_m': count_parameters(model),
        'model_size_mb': model_size_mb(model),
        'training_energy_kwh': energy_metrics['energy_kwh'],
        'effective_heads': effective_heads,
        'effective_mlp': effective_mlp
    }
    
    print(f"\nFinal Test - Acc: {test_metrics['acc']:.4f}, AUC: {test_metrics['auc']:.4f}")
    print(f"Effective heads: {effective_heads:.1f}, MLP: {effective_mlp:.0f}")
    
    return model, final_metrics

# ==================== MAIN ====================

def main():
    """Main entry point"""
    set_seed(SEED)
    
    print("="*100)
    print("PROGRESSIVE VISION TRANSFORMER PRUNING ON MEDMNIST")
    print("="*100)
    print(f"\nDevice: {DEVICE}")
    print(f"Pruning schedule: {[WARMUP_EPOCHS + i * EPOCHS_BETWEEN_PRUNES for i in range(1, NUM_PRUNE_STEPS + 1)]}")
    print(f"Number of trials: {NUM_TRIALS}")
    
    # Datasets and models
    datasets = ['bloodmnist', 'pathmnist', 'dermamnist']
    models = ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224']
    
    print(f"\nDatasets: {datasets}")
    print(f"Models: {models}")
    
    all_results = []
    
    for dataset in datasets:
        npz_path = os.path.join(DATASET_DIR, f"{dataset}_224.npz")
        
        if not os.path.exists(npz_path):
            print(f"\nDataset not found: {npz_path}")
            continue
        
        # Load dataset
        train_loader, val_loader, test_loader, num_classes, dataset_name = load_dataset(npz_path)
        
        for model_name in models:
            print(f"\n{'='*100}")
            print(f"Dataset: {dataset_name} | Model: {model_name}")
            print(f"{'='*100}")
            
            for trial in range(1, NUM_TRIALS + 1):
                trial_seed = SEED + trial * 100
                set_seed(trial_seed)
                
                # Baseline
                try:
                    _, baseline_metrics = train_baseline_vit(
                        dataset_name, model_name, train_loader, val_loader, test_loader,
                        num_classes, SAVE_DIR, trial
                    )
                    all_results.append(baseline_metrics)
                except Exception as e:
                    print(f"Error in baseline: {e}")
                
                cleanup_memory()
                
                # Progressive pruning
                try:
                    _, progressive_metrics = train_progressive_pruning_vit(
                        dataset_name, model_name, train_loader, val_loader, test_loader,
                        num_classes, SAVE_DIR, trial
                    )
                    all_results.append(progressive_metrics)
                except Exception as e:
                    print(f"Error in progressive: {e}")
                
                cleanup_memory()
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(SAVE_DIR, "all_results.csv")
    results_df.to_csv(results_path, index=False)
    
    print("\n" + "="*100)
    print("EXPERIMENT COMPLETED!")
    print(f"Results saved to: {results_path}")
    print("="*100)

if __name__ == "__main__":
    main()