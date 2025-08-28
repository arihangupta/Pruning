import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
from dinov2.models.vision_transformer import vit_small
from medmnist import DermaMNIST
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import pandas as pd

# ================
# Config
# ================
ROOT_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/datasets"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/saved_models"
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
IMG_SIZE = 224  # DinoV2 expects 224x224

# Reproducibility
tf.keras.utils.set_random_seed(42)
np.random.seed(42)
torch.manual_seed(42)

os.makedirs(SAVE_DIR, exist_ok=True)

# ================
# DinoV2 Setup
# ================
dinov2 = vit_small(patch_size=14, img_size=224, init_values=1.0)
dinov2.load_state_dict(torch.hub.load_state_dict_from_url(
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    map_location="cpu"
))
dinov2.eval()
if torch.cuda.is_available():
    dinov2 = dinov2.cuda()

# Preprocess transform for DinoV2
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract DinoV2 features
def extract_dinov2_features(dataset, num_samples=None):
    features, labels = [], []
    for i, (img, label) in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
        img = transform(img).unsqueeze(0)  # Add batch dim
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            out = dinov2(img)  # forward pass
            feat = out["x_norm_clstoken"].cpu().numpy()  # [1, 384]
        features.append(feat.flatten())
        labels.append(int(label))
    return np.array(features), np.array(labels)

# ================
# Data
# ================
derma_train = DermaMNIST(split="train", download=False, size=IMG_SIZE, root=ROOT_DIR)
derma_val = DermaMNIST(split="val", download=False, size=IMG_SIZE, root=ROOT_DIR)
derma_test = DermaMNIST(split="test", download=False, size=IMG_SIZE, root=ROOT_DIR)

print("🚀 Extracting DinoV2 features for DermaMNIST...")
X_train, y_train = extract_dinov2_features(derma_train)
X_val, y_val = extract_dinov2_features(derma_val)
X_test, y_test = extract_dinov2_features(derma_test)

num_classes = len(np.unique(y_train))

# tf.data pipelines
def make_dataset(features, labels, batch_size=BATCH_SIZE, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(X_train, y_train, shuffle=True)
val_ds = make_dataset(X_val, y_val, shuffle=False)
test_ds = make_dataset(X_test, y_test, shuffle=False)

# ================
# Model
# ================
def build_classifier(input_shape, num_classes, learning_rate=LEARNING_RATE):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

class TrainingProgress(tf.keras.callbacks.Callback):
    def __init__(self, test_ds, y_test, save_dir):
        super().__init__()
        self.test_ds = test_ds
        self.y_test = y_test
        self.save_dir = save_dir
        self.class_names = [str(i) for i in range(num_classes)]

    def on_epoch_end(self, epoch, logs=None):
        print(f"✅ Epoch {epoch+1}: "
              f"loss={logs['loss']:.4f}, "
              f"val_loss={logs['val_loss']:.4f}, "
              f"acc={logs['accuracy']:.4f}, "
              f"val_acc={logs['val_accuracy']:.4f}")

        y_pred = self.model.predict(self.test_ds, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(self.y_test, y_pred_classes)

        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        cm_path = os.path.join(self.save_dir, f"confusion_matrix_epoch_{epoch+1}.csv")
        cm_df.to_csv(cm_path)
        print(f"📊 Confusion matrix saved to {cm_path}")

# ================
# Train
# ================
print("\n🚀 Training classifier on DermaMNIST with DinoV2 features...")
input_shape = (X_train.shape[1],)  # should be (384,)
model = build_classifier(input_shape, num_classes)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[TrainingProgress(test_ds, y_test, SAVE_DIR)],
    verbose=0
)

_, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\n📦 Final Test Accuracy: {test_acc:.4f}")

model_path = os.path.join(SAVE_DIR, "dermamnist_dinov2.h5")
model.save(model_path)
print(f"📝 Model saved to {model_path}")
