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
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# ================
# Data
# ================
data = np.load(DATA_PATH)
X_train, y_train = data["train_images"], data["train_labels"].flatten()
X_val, y_val     = data["val_images"], data["val_labels"].flatten()
X_test, y_test   = data["test_images"], data["test_labels"].flatten()

# Normalize and convert to 3 channels
def preprocess(arr):
    arr = arr.astype("float32") / 255.0
    if arr.ndim == 3 or arr.shape[-1] == 1:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    arr = np.transpose(arr, (0, 3, 1, 2))  # NHWC -> NCHW
    return arr

X_train, X_val, X_test = preprocess(X_train), preprocess(X_val), preprocess(X_test)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_ds   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
test_ds  = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(np.unique(y_train))

# ================
# Model
# ================
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ================
# Training & Evaluation
# ================
def train_model(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
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

        val_loss, val_acc = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {running_loss/total:.4f}, Train Acc: {correct/total:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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

# ================
# Run Training
# ================
print("\n🚀 Training ResNet50 on Dermamnist (224x224)...")
train_model(model, train_loader, val_loader, EPOCHS)

# ================
# Evaluate on Test Set
# ================
test_loss, test_acc = evaluate_model(model, test_loader)
print(f"\n✅ Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# ================
# Save Model
# ================
model_path = os.path.join(SAVE_DIR, "resnet50_dermamnist_224.pth")
torch.save(model.state_dict(), model_path)
print(f"\n📝 Model saved to {model_path}")
