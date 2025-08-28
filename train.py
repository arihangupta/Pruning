import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

# ================
# Config
# ================
DATA_PATH = "/home/arihangupta/Pruning/dinov2/Pruning/datasets/dermamnist_224.npz"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/saved_models"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 20  # print batch progress every 20 batches

os.makedirs(SAVE_DIR, exist_ok=True)

# ================
# Data
# ================
print("📦 Loading dataset...")
data = np.load(DATA_PATH)
X_train, y_train = data["train_images"], data["train_labels"].flatten()
X_val, y_val     = data["val_images"], data["val_labels"].flatten()
X_test, y_test   = data["test_images"], data["test_labels"].flatten()
print(f"✅ Data loaded: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

# Normalize and convert to 3 channels, NCHW
def preprocess(arr):
    arr = arr.astype("float32") / 255.0
    if arr.ndim == 3 or arr.shape[-1] == 1:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    arr = np.transpose(arr, (0, 3, 1, 2))  # NHWC -> NCHW
    return arr

print("🔄 Preprocessing dataset...")
X_train = preprocess(X_train)
X_val   = preprocess(X_val)
X_test  = preprocess(X_test)
print("✅ Preprocessing done.")

# Convert to tensors
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.long))
val_ds   = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                         torch.tensor(y_val, dtype=torch.long))
test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                         torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(np.unique(y_train))
print(f"🔢 Number of classes: {num_classes}")

# ================
# Model
# ================
print("🚀 Building ResNet50 model...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)
print("✅ Model built and moved to device:", DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ================
# Training & Evaluation
# ================
def evaluate_model(model, loader):
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return loss_total / total, correct / total

def train_model(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % LOG_INTERVAL == 0 or batch_idx == len(train_loader):
                print(f"  Batch {batch_idx}/{len(train_loader)} - "
                      f"Loss: {running_loss/total:.4f}, Acc: {correct/total:.4f}")

        val_loss, val_acc = evaluate_model(model, val_loader)
        epoch_time = time.time() - start_time
        print(f"📈 Epoch {epoch+1}/{epochs} completed in {epoch_time:.1f}s - "
              f"Train Loss: {running_loss/total:.4f}, Train Acc: {correct/total:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ================
# Run Training
# ================
print("\n🎯 Starting training...")
train_model(model, train_loader, val_loader, EPOCHS)

# ================
# Evaluate on Test Set
# ================
print("\n🧪 Evaluating on test set...")
test_loss, test_acc = evaluate_model(model, test_loader)
print(f"✅ Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# ================
# Save Model
# ================
model_path = os.path.join(SAVE_DIR, "resnet50_dermamnist_224.pth")
torch.save(model.state_dict(), model_path)
print(f"\n💾 Model saved to {model_path}")
