import os
import sys
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
import timm
import medmnist
from medmnist import INFO, Evaluator
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize


def build_dataset(args):
    """Build dataset from local balanced datasets."""
    dataset_name = args.dataset
    
    if not dataset_name.endswith('mnist'):
        raise ValueError(f"This script only supports MedMNIST datasets. Got: {dataset_name}")
    
    info = INFO[dataset_name]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # Test transforms without augmentation
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # Load from local balanced datasets
    train_dataset = DataClass(
        split='train',
        transform=train_transform,
        download=False,
        root='./datasets_balanced'
    )
    
    test_dataset = DataClass(
        split='test',
        transform=test_transform,
        download=False,
        root='./datasets_balanced'
    )
    
    return train_dataset, test_dataset, n_classes, task


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


def train_epoch(net, train_loader, optimizer, scheduler, loss_function, device, task):
    """Train for one epoch."""
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout, desc="Training")
    
    for step, (images, labels) in enumerate(train_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(images)
        
        if task == 'multi-label, binary-class':
            labels = labels.to(torch.float32)
            loss = loss_function(outputs, labels)
        else:
            labels = labels.squeeze().long()
            loss = loss_function(outputs, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        
        train_bar.set_postfix(loss=f"{loss.item():.3f}")
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def evaluate_model(net, test_loader, device, task, data_flag):
    """Evaluate the model."""
    net.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    y_score = torch.tensor([])
    
    with torch.no_grad():
        val_bar = tqdm(test_loader, file=sys.stdout, desc="Evaluating")
        for inputs, targets in val_bar:
            inputs = inputs.to(device)
            outputs = net(inputs)
            
            if task == 'multi-label, binary-class':
                probs = torch.sigmoid(outputs)
                y_score = torch.cat((y_score, probs.cpu()), 0)
                targets_np = targets.numpy()
                all_labels.extend(targets_np)
                all_probs.extend(probs.cpu().numpy())
            else:
                probs = torch.softmax(outputs, dim=1)
                predict_y = torch.max(probs, dim=1)[1]
                
                targets = targets.squeeze().long()
                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # For MedMNIST evaluator
                targets_float = targets.float().resize_(len(targets), 1)
                y_score = torch.cat((y_score, probs.cpu()), 0)
    
    # Calculate MedMNIST metrics
    y_score_np = y_score.detach().numpy()
    evaluator = Evaluator(data_flag, 'test', size=224, root='./datasets_balanced')
    auc, acc = evaluator.evaluate(y_score_np)
    
    # Calculate additional metrics for multi-class
    if task != 'multi-label, binary-class':
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        conf_matrix = confusion_matrix(all_labels, all_preds)
        specificity = specificity_per_class(conf_matrix)
        avg_specificity = sum(specificity) / len(specificity) if specificity else 0.0
        
        metrics = {
            'auc': auc,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'specificity': avg_specificity,
            'f1': f1
        }
    else:
        metrics = {
            'auc': auc,
            'acc': acc
        }
    
    return metrics


def train(args):
    """Main training function."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    
    # Build dataset
    print(f"Loading dataset: {args.dataset}")
    train_dataset, test_dataset, nb_classes, task = build_dataset(args)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}, Classes: {nb_classes}")
    
    # Create data loaders
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model (from scratch, no pretrained weights)
    print(f"Creating model: {args.model_name}")
    net = timm.create_model(
        args.model_name,
        pretrained=False,  # Train from scratch
        num_classes=nb_classes
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Loss function
    if task == 'multi-label, binary-class':
        loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = nn.CrossEntropyLoss()
    
    # Optimizer with warmup-friendly settings
    optimizer = optim.AdamW(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.min_lr
    )
    
    # Training loop
    best_acc = 0.0
    best_auc = 0.0
    save_path = f'./{args.model_name}_{args.dataset}_scratch.pth'
    
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        
        # Train
        train_loss = train_epoch(net, train_loader, optimizer, scheduler, loss_function, device, task)
        
        # Evaluate
        metrics = evaluate_model(net, test_loader, device, task, args.dataset)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val AUC: {metrics['auc']:.4f}, Val Acc: {metrics['acc']:.4f}")
        
        if task != 'multi-label, binary-class':
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
                'args': vars(args)
            }
            torch.save(state, save_path)
            print(f"Model saved to {save_path}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Model saved to: {save_path}")


def main():
    # Validate dataset path
    if not os.path.exists('./datasets_balanced'):
        print("Error: './datasets_balanced' directory not found!")
        print("Please ensure your balanced datasets are in this directory.")
        sys.exit(1)
    
    # Define datasets and models to train
    datasets = ['bloodmnist', 'pathmnist', 'dermamnist']
    models = ['vit_tiny_patch16_224', 'vit_small_patch16_224', 
              'vit_base_patch16_224', 'vit_large_patch16_224']
    
    # Training hyperparameters
    batch_size = 64
    epochs = 100
    lr = 0.001
    min_lr = 1e-6
    weight_decay = 0.05
    
    print("=" * 100)
    print("TRAINING VISION TRANSFORMERS ON MEDMNIST DATASETS")
    print("=" * 100)
    print(f"\nDatasets: {datasets}")
    print(f"Models: {models}")
    print(f"Total combinations: {len(datasets) * len(models)}")
    print(f"\nHyperparameters:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Weight decay: {weight_decay}")
    print("\n" + "=" * 100)
    
    # Train all combinations
    total_models = len(datasets) * len(models)
    current_model = 0
    
    for dataset in datasets:
        for model_name in models:
            current_model += 1
            
            print("\n" + "=" * 100)
            print(f"TRAINING MODEL {current_model}/{total_models}")
            print(f"Dataset: {dataset} | Model: {model_name}")
            print("=" * 100)
            
            # Create args object
            class Args:
                pass
            
            args = Args()
            args.dataset = dataset
            args.model_name = model_name
            args.batch_size = batch_size
            args.epochs = epochs
            args.lr = lr
            args.min_lr = min_lr
            args.weight_decay = weight_decay
            
            try:
                train(args)
                print(f"\n✓ Successfully completed training for {model_name} on {dataset}")
            except Exception as e:
                print(f"\n✗ Error training {model_name} on {dataset}: {str(e)}")
                import traceback
                traceback.print_exc()
                print("Continuing to next model...")
                continue
            
            print("\n" + "=" * 100)
    
    print("\n" + "=" * 100)
    print("ALL TRAINING COMPLETED!")
    print("=" * 100)


if __name__ == '__main__':
    main()