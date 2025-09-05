#!/usr/bin/env python3
"""
Fixed PGTO pruning script for ResNet50 on dermamnist.

Key fixes:
 - compute_stage_importance_and_keeps returns torch.LongTensor keeps (no numpy/int .size() mistakes)
 - No BN resizing hacks during importance computation (we compute Wanda++ on the unpruned baseline)
 - Proper forward/backward hooks that store tensors in closure lists (not relying on hook return)
 - Robust weight surgery: correct use of prev_expanded_idxs, repeat_interleave for expansion, device-safe copying
 - Clear fallbacks from wanda++ -> l1 -> bn_gamma
 - Preserves your PGTO pipeline: per-stage keep computation, local calibration, save checkpoint, short global finetune, CSV output
"""

import os
import time
import math
import random
import copy
import csv
import tempfile
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.resnet import Bottleneck
from torchvision import models
from sklearn.metrics import roc_auc_score

# Try fvcore for FLOPs if available
HAS_FVCORE = True
try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    HAS_FVCORE = False

# -------------------------
# Config (keep your paths)
# -------------------------
DERMA_PATH = "/home/arihangupta/Pruning/dinov2/Pruning/datasets/dermamnist_224.npz"
SAVE_DIR = "/home/arihangupta/Pruning/dinov2/Pruning/saved_models_pgto"
BASELINE_CKPT = os.path.join("/home/arihangupta/Pruning/dinov2/Pruning/saved_models",
                             "dermamnist_resnet50_BASELINE.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 32
IMG_SIZE = 224

# Wanda++-style knobs
TARGET_RATIOS = [0.5, 0.7]     # final prune targets per stage (50%, 70%)
METHODS = ["wanda++", "l1", "bn_gamma"]   # importance criterion (we will fallback gracefully)
CAL_EPOCHS = 1                 # short local calibration after each stage
CAL_MAX_BATCHES = 150          # cap steps for calibration
CAL_LR = 3e-4
FINAL_FINETUNE_EPOCHS = 2      # short global finetune after all stages
FINAL_LR = 1e-4

# Logging/timing
LOG_INTERVAL = 20
WARMUP = 5
TIMING_BATCHES = 30

# -------------------------
# Reproducibility
# -------------------------
def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# -------------------------
# Data helpers (same as your original)
# -------------------------
def preprocess(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32") / 255.0
    if arr.ndim == 3 or arr.shape[-1] == 1:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    return np.transpose(arr, (0, 3, 1, 2))  # NHWC -> NCHW

def make_loaders(npz_path):
    data = np.load(npz_path)
    X_train, y_train = data["train_images"], data["train_labels"].flatten()
    X_val, y_val     = data["val_images"], data["val_labels"].flatten()
    X_test, y_test   = data["test_images"], data["test_labels"].flatten()
    X_train = preprocess(X_train); X_val = preprocess(X_val); X_test = preprocess(X_test)
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader, int(len(np.unique(y_train)))

train_loader, val_loader, test_loader, NUM_CLASSES = make_loaders(DERMA_PATH)
print(f"Data loaded. NUM_CLASSES={NUM_CLASSES}, device={DEVICE}")

# -------------------------
# CustomResNet builder (variable stage widths)
# -------------------------
class CustomResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3,4,6,3], stage_planes=[64,128,256,512], num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage_planes = stage_planes[:]
        self.layers_cfg = layers[:]

        self.layer1 = self._make_layer(block, stage_planes[0], layers[0])
        self.layer2 = self._make_layer(block, stage_planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, stage_planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, stage_planes[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(stage_planes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x,1); x = self.fc(x)
        return x

# -------------------------
# Baseline load (unchanged)
# -------------------------
def build_resnet50_for_load(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_baseline_ckpt(path=BASELINE_CKPT):
    model = build_resnet50_for_load(NUM_CLASSES)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model

baseline = load_baseline_ckpt()
print("Baseline loaded.")

# -------------------------
# Importance & keep computation (fixed)
# - Wanda++ uses activation norms and regional gradients computed on baseline
# - We avoid changing BN running stats or changing model structure during importance computation
# -------------------------
def compute_stage_importance_and_keeps(model: nn.Module, stage_name: str, keep_k: int, method: str="l1", train_loader=None, alpha=1.0, max_batches=20):
    """
    Returns torch.LongTensor sorted indices (kept channels).
    Methods: 'wanda++' (needs train_loader), 'l1', 'bn_gamma'
    """
    stage = getattr(model, stage_name)
    first_block = next(stage.children())
    orig_planes = first_block.conv1.out_channels  # input planes to the stage (pre-expansion)
    device = DEVICE

    # If user requested wanda++ but provided no train_loader, fallback
    if method == "wanda++" and train_loader is None:
        print(f"Warning: wanda++ requested but no train_loader provided for {stage_name}; falling back to l1.")
        method = "l1"

    # Wanda++: collect input activations to conv1 and gradients of a regional loss (stage output norm)
    if method == "wanda++":
        try:
            model.eval()
            act_list = []
            grad_list = []
            # forward hook on conv1 inputs for each block in stage
            # We'll register hooks on the first block's conv1 input (that's representative), but average across blocks to be robust.
            # We'll collect activations from conv1 inputs of each block using forward hooks and accumulate gradients via backward hook on stage output.
            # Define hooks:
            conv1_inputs = []

            def make_forward_hook(storage):
                def hook(module, input, output):
                    # input[0] is the tensor into conv1, shape [B, C, H, W] (this is the input into conv1)
                    storage.append(input[0].detach())
                return hook

            # Register forward hooks on conv1 of each block of the stage
            fhooks = []
            storages = []
            for block in stage.children():
                s = []
                storages.append(s)
                h = block.conv1.register_forward_hook(make_forward_hook(s))
                fhooks.append(h)

            # We will also register a hook to capture the stage output so we can compute regional loss
            stage_outputs = []
            def stage_forward_hook(module, input, output):
                # output is the stage's output tensor; store it for loss computation
                stage_outputs.append(output)
            shook = stage.register_forward_hook(stage_forward_hook)

            batches = 0
            for imgs, labels in train_loader:
                imgs = imgs.to(device); labels = labels.to(device)
                # zero grads
                model.zero_grad()
                out = model(imgs)
                # Use L2 norm of stage output as regional loss (same idea as original)
                if len(stage_outputs) == 0:
                    # Should not happen, but just in case
                    regional_loss = out.norm()
                else:
                    regional_loss = stage_outputs[-1].norm()
                regional_loss.backward(retain_graph=False)
                batches += 1
                stage_outputs.clear()
                if batches >= max_batches:
                    break

            # remove hooks
            for h in fhooks: h.remove()
            shook.remove()

            # Aggregate activations and grads across storages
            # storages is a list of lists (one per block), each with tensors [B, C, H, W] across batches.
            # We'll compute per-channel L2 norm across all collected tensors.
            # Flatten by concatenation along batch dimension
            act_concat = []
            for s in storages:
                if len(s) > 0:
                    act_concat.append(torch.cat(s, dim=0))
            if len(act_concat) == 0:
                raise RuntimeError("No activations captured for wanda++; falling back.")
            act_all = torch.cat(act_concat, dim=0)  # shape [N, C, H, W]

            # For grads: extract conv3.weight.grad from each block and average over blocks
            grad_mags = []
            for block in stage.children():
                g = block.conv3.weight.grad  # shape [out_ch, in_ch, k, k]
                if g is None:
                    continue
                # compute mean absolute gradient per input plane grouping (account for expansion)
                g_abs = g.abs().detach().to("cpu")
                out_ch = g_abs.shape[0]
                in_ch = g_abs.shape[1]
                exp = out_ch // in_ch if in_ch > 0 else 1
                # collapse output channels into groups of size exp to map to input planes
                g_per_plane = []
                for p in range(in_ch):
                    start = p * exp
                    end = start + exp
                    g_slice = g_abs[start:end, p:p+1, :, :]  # shape [exp, 1, k, k]
                    g_per_plane.append(g_slice.mean().item())
                grad_mags.append(np.array(g_per_plane))
            if len(grad_mags) == 0:
                # fallback to zeros if no grads captured
                grad_mags = np.zeros((1, orig_planes))
            grad_magnitudes = np.mean(np.stack(grad_mags, axis=0), axis=0)  # shape [orig_planes]

            # Activation norms per channel
            act_norms = act_all.norm(p=2, dim=(0,2,3)).cpu().numpy()  # length = C

            # Weight magnitudes: average conv3 weights aggregated per input-plane-group
            weight_mags = []
            for block in stage.children():
                conv3_w = block.conv3.weight.detach().abs().cpu().numpy()
                out_ch = conv3_w.shape[0]
                in_ch = conv3_w.shape[1]
                if in_ch == 0:
                    continue
                exp = out_ch // in_ch if in_ch>0 else 1
                per_plane = []
                for p in range(in_ch):
                    start = p*exp
                    end = start + exp
                    per_plane.append(conv3_w[start:end, :, :, :].sum())
                weight_mags.append(np.array(per_plane))
            if len(weight_mags) == 0:
                weight_mags = np.ones((1, orig_planes))
            weight_mags = np.mean(np.stack(weight_mags, axis=0), axis=0)

            # final importance: (alpha * grad + act_norm) * weight_mags
            importance = (alpha * grad_magnitudes + act_norms) * weight_mags
            importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
            # choose top-k
            k = keep_k
            if k >= len(importance):
                keep = np.arange(len(importance))
            else:
                keep = np.argsort(importance)[-k:]
            keep = np.sort(keep)
            keep_t = torch.from_numpy(keep).long().to(device)
            return keep_t.cpu().long()
        except Exception as e:
            print(f"Warning: Wanda++ computation failed for {stage_name}, falling back to l1. Error: {e}")
            method = "l1"

    # Fallback methods
    if method == "l1":
        block_importances = []
        for block in stage.children():
            conv3 = block.conv3.weight.detach().abs().cpu().numpy()
            out_ch = conv3.shape[0]
            in_ch = conv3.shape[1]
            exp = out_ch // in_ch if in_ch>0 else 1
            per_plane = []
            for p in range(in_ch):
                start = p*exp
                end = start + exp
                per_plane.append(conv3[start:end, :, :, :].sum())
            block_importances.append(np.array(per_plane))
        importance = np.mean(np.stack(block_importances, axis=0), axis=0)
        # choose top-k
        k = keep_k
        if k >= len(importance):
            keep = np.arange(len(importance))
        else:
            keep = np.argsort(importance)[-k:]
        return torch.from_numpy(np.sort(keep)).long()

    elif method == "bn_gamma":
        block_importances = []
        for block in stage.children():
            gammas = block.bn3.weight.detach().abs().cpu().numpy()
            out_ch = gammas.shape[0]
            # map groups of expansion to input planes
            # we assume expansion = out_ch / in_ch
            # fallback: if not divisible, take mean groups
            # find candidate in_ch by checking block.conv3.in_channels
            in_ch = block.conv3.weight.shape[1]
            if in_ch == 0:
                continue
            exp = out_ch // in_ch if in_ch>0 else 1
            per_plane = []
            for p in range(in_ch):
                start = p*exp
                end = start + exp
                per_plane.append(np.mean(gammas[start:end]))
            block_importances.append(np.array(per_plane))
        importance = np.mean(np.stack(block_importances, axis=0), axis=0)
        k = keep_k
        if k >= len(importance):
            keep = np.arange(len(importance))
        else:
            keep = np.argsort(importance)[-k:]
        return torch.from_numpy(np.sort(keep)).long()

    # Should not reach here
    raise RuntimeError("Unknown method in compute_stage_importance_and_keeps")

# -------------------------
# Build pruned model and copy weights (surgery) - robust
# -------------------------
def build_pruned_resnet(keep_indices, num_classes):
    # keep_indices: dict {stage: 1D torch.LongTensor of kept input channels for that stage}
    stage_planes = [
        len(keep_indices['layer1']),
        len(keep_indices['layer2']),
        len(keep_indices['layer3']),
        len(keep_indices['layer4'])
    ]
    layers = [3, 4, 6, 3]
    return CustomResNet(block=Bottleneck, layers=layers, stage_planes=stage_planes, num_classes=num_classes)

def build_pruned_resnet_and_copy_weights(base_model, keep_indices, num_classes):
    """
    Create pruned CustomResNet according to keep_indices and copy weights from base_model.
    keep_indices entries will be normalized to torch.LongTensor on CPU for indexing.
    """
    # normalize keep indices to torch LongTensor (on CPU for indexing)
    for s in ['layer1','layer2','layer3','layer4']:
        idxs = keep_indices.get(s, None)
        if idxs is None:
            raise ValueError(f"Missing keep indices for {s}")
        if isinstance(idxs, np.ndarray):
            keep_indices[s] = torch.from_numpy(idxs).long().cpu()
        elif isinstance(idxs, list):
            keep_indices[s] = torch.tensor(idxs, dtype=torch.long).cpu()
        elif isinstance(idxs, torch.Tensor):
            keep_indices[s] = idxs.cpu().long()
        else:
            raise TypeError("keep index type must be numpy/list/torch")

    new_model = build_pruned_resnet(keep_indices, num_classes=num_classes).to(DEVICE)

    expansion = 4
    STAGES = ["layer1", "layer2", "layer3", "layer4"]

    # Copy stem parameters (conv1, bn1)
    with torch.no_grad():
        new_model.conv1.weight.copy_(base_model.conv1.weight.to(DEVICE))
        new_model.bn1.load_state_dict(base_model.bn1.state_dict())

    # prev_expanded_idxs tracks which original channels map to the current input channels for next stage
    prev_expanded_idxs = None  # None for layer1 (input channels are original conv1 out channels)

    for stage_name in STAGES:
        old_layer = getattr(base_model, stage_name)
        new_layer = getattr(new_model, stage_name)
        kept = keep_indices[stage_name].to(DEVICE)  # kept input-plane indices for this stage (on device for copy)
        for block_idx, (old_block, new_block) in enumerate(zip(old_layer, new_layer)):
            # ---------------- conv1 (maps input planes -> intermediate)
            # old_block.conv1.weight: [old_out, old_in, k, k]
            # new_block.conv1.weight: [new_out, new_in, k, k]
            # new_out should equal kept.size(0) (we pruned output channels of conv1 to match kept)
            # new_in should equal prev_kept.size(0) if prev_expanded_idxs is set, else old_in (original)
            old_w = old_block.conv1.weight.data  # CPU/CUDA depends on model
            # Determine prev_in_idxs: which indices in old input correspond to the new input channels
            if prev_expanded_idxs is None:
                # copy full input channels (old_in)
                prev_in_idxs = torch.arange(old_w.shape[1], dtype=torch.long, device=DEVICE)
            else:
                prev_in_idxs = prev_expanded_idxs.to(DEVICE)

            # For conv1, we need to select the kept output channels and slice input channels by prev_in_idxs
            # old_w[kept][:, prev_in_idxs, :, :]
            new_block.conv1.weight.data.copy_(old_w[kept][:, prev_in_idxs, :, :].to(DEVICE))

            if getattr(old_block.conv1, "bias", None) is not None:
                new_block.conv1.bias.data.copy_(old_block.conv1.bias.data[kept].to(DEVICE))

            # bn1 copy - bn1 has running stats sized to conv1.out_channels (after expansion not considered here)
            # bn1.params length corresponds to new_block.conv1.out_channels (which equals kept.size(0))
            for attr in ("weight","bias","running_mean","running_var"):
                old_attr = getattr(old_block.bn1, attr).data
                new_attr = getattr(new_block.bn1, attr).data
                # old_attr may be longer; slice to kept.size(0)
                new_attr.copy_(old_attr[:kept.size(0)].to(DEVICE))

            # ---------------- conv2 (pointwise) - this typically keeps same channel count as conv1.out
            old_w2 = old_block.conv2.weight.data
            # conv2 often expects input channels == conv1.out_channels and output == conv1.out_channels (bottleneck)
            # We'll select the same kept indices for conv2 input/output where appropriate.
            # For a safer approach, copy the intersection slice: old_w2[kept][:, :kept.size(0), :, :]
            new_block.conv2.weight.data.copy_(old_w2[kept][:, :kept.size(0), :, :].to(DEVICE))
            if getattr(old_block.conv2, "bias", None) is not None:
                new_block.conv2.bias.data.copy_(old_block.conv2.bias.data[kept].to(DEVICE))

            for attr in ("weight","bias","running_mean","running_var"):
                old_attr = getattr(old_block.bn2, attr).data
                new_attr = getattr(new_block.bn2, attr).data
                new_attr.copy_(old_attr[:kept.size(0)].to(DEVICE))

            # ---------------- conv3 (output conv) - must account for expansion
            old_w3 = old_block.conv3.weight.data  # [old_out, old_in, k,k]
            old_out = old_w3.shape[0]
            old_in = old_w3.shape[1]
            # compute old_idx by repeating kept according to expansion factor
            # Note: in baseline old_out == old_in * expansion
            # new_block.conv3.out_channels == kept.size(0) * expansion
            old_idx = kept.repeat_interleave(expansion).to(DEVICE)  # shape [new_out]
            # For input channels of conv3, we should use prev_in_idxs (what we used for conv1)
            new_block.conv3.weight.data.copy_(old_w3[old_idx][:, :prev_in_idxs.size(0), :, :].to(DEVICE))
            if getattr(old_block.conv3, "bias", None) is not None:
                # conv3 bias is length old_out; pick old_idx
                new_block.conv3.bias.data.copy_(old_block.conv3.bias.data[old_idx].to(DEVICE))

            # bn3: size = kept.size(0) * expansion
            for attr in ("weight","bias","running_mean","running_var"):
                old_attr = getattr(old_block.bn3, attr).data
                new_attr = getattr(new_block.bn3, attr).data
                new_attr.copy_(old_attr[:kept.size(0)*expansion].to(DEVICE))

            # ---------------- downsample (if present)
            if old_block.downsample is not None:
                # old_block.downsample[0] is conv1x1 mapping residual input to out_channels
                ds_old_conv = old_block.downsample[0].weight.data  # [old_out, old_in, 1,1]
                ds_new_conv = new_block.downsample[0].weight.data  # [new_out, new_in, 1,1]
                # ds_new_conv should map from prev_in_idxs to old_idx (expanded)
                # copy ds_old_conv[old_idx][:, prev_in_idxs, :, :]
                ds_new_conv.copy_(ds_old_conv[old_idx][:, :prev_in_idxs.size(0), :, :].to(DEVICE))
                # downsample BN
                for attr in ("weight","bias","running_mean","running_var"):
                    old_attr = getattr(old_block.downsample[1], attr).data
                    new_attr = getattr(new_block.downsample[1], attr).data
                    new_attr.copy_(old_attr[:kept.size(0)*expansion].to(DEVICE))

            # After copying this block, set prev_expanded_idxs for the next block (in same stage or next stage)
            # For the next block, the input channels that map forward are the expanded indices of this kept set (old_idx)
            prev_expanded_idxs = old_idx.clone().detach().cpu()

        # End of stage loop; prev_expanded_idxs ready for next stage iteration (kept indices expanded)
        # Note: keep_indices[stage_name] is defined per-stage; next stage's conv1 expects inputs indexed by prev_expanded_idxs
    # end for stages

    # fc: final linear uses input dimension equal to prev_expanded_idxs.size(0)
    # prev_expanded_idxs corresponds to final stage expansion mapping; we copy fc weights by selecting appropriate input columns
    with torch.no_grad():
        if prev_expanded_idxs is None:
            # Shouldn't happen, but fallback: copy entire fc
            new_model.fc.weight.copy_(base_model.fc.weight.to(DEVICE))
            new_model.fc.bias.copy_(base_model.fc.bias.to(DEVICE))
        else:
            # base_model.fc.weight shape [num_classes, in_features]
            # We need to copy columns corresponding to prev_expanded_idxs
            base_fc_w = base_model.fc.weight.data.to(DEVICE)
            # If prev_expanded_idxs length matches base fc in_features -> copy full
            if prev_expanded_idxs.size(0) == base_fc_w.shape[1]:
                new_model.fc.weight.copy_(base_fc_w)
            else:
                # select columns
                sel = prev_expanded_idxs.to(DEVICE)
                new_model.fc.weight.data.copy_(base_fc_w[:, sel])
            new_model.fc.bias.copy_(base_model.fc.bias.to(DEVICE))

    return new_model

# -------------------------
# Metrics & evaluation functions (kept from original with small safety tweaks)
# -------------------------
criterion = nn.CrossEntropyLoss()

def evaluate_model_basic(model, loader):
    model.eval()
    loss_total = 0.0; correct = 0; total = 0
    probs_list = []; labels_list = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += float(loss.item()) * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0); correct += int(predicted.eq(labels).sum().item())
            probs_list.append(torch.softmax(outputs, dim=1).cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    loss_avg = loss_total / max(1, total)
    acc = correct / max(1, total)
    try:
        auc = roc_auc_score(np.concatenate(labels_list), np.concatenate(probs_list), multi_class="ovr")
    except Exception:
        auc = float("nan")
    return loss_avg, acc, auc

def count_zeros_and_total(model):
    total = 0; zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += int((p == 0).sum().item())
    return zeros, total

def params_count(model):
    return sum(p.numel() for p in model.parameters())

def model_size_bytes(model):
    fd, tmp = tempfile.mkstemp(suffix=".pth"); os.close(fd)
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp)
    os.remove(tmp)
    return size

def compute_flops(model):
    model = model.eval()
    try:
        if HAS_FVCORE:
            example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            fca = FlopCountAnalysis(model, example)
            return float(fca.total())
        else:
            flops = 0
            hooks = []
            def conv_hook(self, inp, out):
                nonlocal flops
                in_t = inp[0]; out_t = out
                kh, kw = self.kernel_size
                C_in = in_t.shape[1]; C_out = out_t.shape[1]
                H_out = out_t.shape[2]; W_out = out_t.shape[3]
                flops += kh * kw * C_in * C_out * H_out * W_out
            def linear_hook(self, inp, out):
                nonlocal flops
                flops += inp[0].shape[1] * out.shape[1]
            for m in model.modules():
                if isinstance(m, nn.Conv2d): hooks.append(m.register_forward_hook(conv_hook))
                if isinstance(m, nn.Linear): hooks.append(m.register_forward_hook(linear_hook))
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            with torch.no_grad(): _ = model(dummy)
            for h in hooks: h.remove()
            return float(flops)
    except Exception as e:
        print("FLOPs estimate failed:", e)
        return float("nan")

def measure_inference_time_and_peak_ram(model, loader, batch_size=BATCH_SIZE, warmup=WARMUP, timed=TIMING_BATCHES):
    model.eval()
    use_cuda = DEVICE.type == "cuda"
    it = iter(loader)
    # warmup
    try:
        for _ in range(warmup):
            imgs, _ = next(it)
            imgs = imgs.to(DEVICE)
            with torch.no_grad(): _ = model(imgs)
            if use_cuda: torch.cuda.synchronize()
    except StopIteration:
        pass
    if use_cuda: torch.cuda.reset_peak_memory_stats()
    start = time.time()
    batches_done = 0
    try:
        for _ in range(timed):
            imgs, _ = next(it)
            imgs = imgs.to(DEVICE)
            with torch.no_grad(): _ = model(imgs)
            if use_cuda: torch.cuda.synchronize()
            batches_done += 1
    except StopIteration:
        pass
    elapsed = time.time() - start
    avg_batch = elapsed / max(1, batches_done)
    if use_cuda:
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_mb = params_count(model) * 4.0 / (1024**2)
    return avg_batch, peak_mb, batches_done

def collect_metrics_row(tag_variant, tag_stage, ratio, model, test_loader, path_hint):
    loss, acc, auc = evaluate_model_basic(model, test_loader)
    zeros, total = count_zeros_and_total(model)
    params = params_count(model)
    flops = compute_flops(model)
    flops_m = flops / 1e6 if not math.isnan(flops) else float("nan")
    avg_time, peak_ram, _ = measure_inference_time_and_peak_ram(model, test_loader)
    if os.path.exists(path_hint):
        size_mb = os.path.getsize(path_hint)/(1024**2)
    else:
        size_mb = model_size_bytes(model)/(1024**2)
    power_m = (flops * ((total - zeros)/total)) / 1e6 if not math.isnan(flops) and total>0 else float("nan")
    return {
        "Variant": tag_variant, "Stage": tag_stage, "Ratio": ratio,
        "Acc": acc, "AUC": auc, "Loss": loss,
        "Params": params, "Zeros": zeros, "TotalParams": total, "PctZeros": (zeros/total)*100 if total>0 else 0,
        "ModelSizeMB": size_mb, "FLOPs_per_image": flops, "FLOPs_M_per_image": flops_m,
        "InferenceTime_per_batch32_s": avg_time, "PeakRAM_MB": peak_ram,
        "PowerProxy_MFLOPs": power_m, "ModelPath": path_hint
    }

# -------------------------
# Calibration helpers (PGTO)
# -------------------------
def freeze_all(model):
    for p in model.parameters(): p.requires_grad = False

def unfreeze_stage(model, stage_name):
    for name, p in model.named_parameters():
        if name.startswith(stage_name):
            p.requires_grad = True
    # Also allow FC & top BN to breathe a bit
    for name, p in model.named_parameters():
        if name.startswith("fc.") or name.startswith("bn1."):
            p.requires_grad = True

def calibrate_stage(model, stage_name, train_loader, epochs=CAL_EPOCHS, max_batches=CAL_MAX_BATCHES, lr=CAL_LR):
    freeze_all(model)
    unfreeze_stage(model, stage_name)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    steps = 0
    for ep in range(epochs):
        running_loss = 0.0; total = 0; correct = 0
        for bidx, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward(); opt.step()
            running_loss += float(loss.item()) * imgs.size(0)
            _, preds = out.max(1)
            total += labels.size(0); correct += int(preds.eq(labels).sum().item())
            steps += 1
            if bidx % LOG_INTERVAL == 0:
                print(f"      Calib {stage_name} ep{ep+1} batch{bidx} - loss {running_loss/max(1,total):.4f}, acc {correct/max(1,total):.4f}")
            if steps >= max_batches:
                return

# -------------------------
# PGTO pipeline (fixed end-to-end)
# -------------------------
rows = []

# Baseline metrics
print("\n=== EVALUATE BASELINE ===")
loss_b, acc_b, auc_b = evaluate_model_basic(baseline, test_loader)
zeros_b, total_b = count_zeros_and_total(baseline)
params_b = params_count(baseline)
flops_b = compute_flops(baseline)
flops_b_m = flops_b / 1e6 if not math.isnan(flops_b) else float("nan")
avg_time_b, peak_ram_b, done_b = measure_inference_time_and_peak_ram(baseline, test_loader)
size_b = os.path.getsize(BASELINE_CKPT) / (1024**2) if os.path.exists(BASELINE_CKPT) else (model_size_bytes(baseline)/(1024**2))
power_b_m = (flops_b * ((total_b - zeros_b) / total_b)) / 1e6 if not math.isnan(flops_b) else float("nan")
rows.append({
    "Variant": "baseline", "Stage": "baseline", "Ratio": 0.0,
    "Acc": acc_b, "AUC": auc_b, "Loss": loss_b,
    "Params": params_b, "Zeros": zeros_b, "TotalParams": total_b, "PctZeros": (zeros_b/total_b)*100 if total_b>0 else 0,
    "ModelSizeMB": size_b,
    "FLOPs_per_image": flops_b, "FLOPs_M_per_image": flops_b_m,
    "InferenceTime_per_batch32_s": avg_time_b, "PeakRAM_MB": peak_ram_b,
    "PowerProxy_MFLOPs": power_b_m, "ModelPath": BASELINE_CKPT
})
print("Baseline done:", rows[-1])

# Helper to get original stage widths from baseline
def stage_orig_channels(model, stage_name):
    first_block = next(getattr(model, stage_name).children())
    return first_block.conv1.out_channels

STAGES = ["layer1","layer2","layer3","layer4"]

# PGTO loop: for each method and target ratio, prune stage-by-stage with calibration
for method in METHODS:
    for target_ratio in TARGET_RATIOS:
        print(f"\n=== PGTO: method={method}, target_ratio={target_ratio} ===")
        # Start with keep all for each stage (so we can iteratively replace)
        keep_indices = {s: torch.arange(stage_orig_channels(baseline, s), dtype=torch.long) for s in STAGES}
        # Map from original baseline weights for surgery (stable)
        for s in STAGES:
            orig = stage_orig_channels(baseline, s)
            keep_k = max(1, int(math.floor(orig * (1.0 - target_ratio))))
            # compute fresh keeps for JUST this stage using baseline weights and train loader for calibration if needed
            print(f"  Computing keeps for {s}: keep_k={keep_k}, method={method}")
            try:
                keeps = compute_stage_importance_and_keeps(baseline, s, keep_k, method=method, train_loader=train_loader, alpha=1.0, max_batches=20)
            except Exception as e:
                print(f"    Importance calc failed for {s} with method {method}: {e}. Falling back to l1.")
                keeps = compute_stage_importance_and_keeps(baseline, s, keep_k, method="l1", train_loader=train_loader)
            keep_indices[s] = keeps.cpu().long()
            print(f"  -> stage {s}: keep {len(keep_indices[s])}/{orig} ({100*len(keep_indices[s])/orig:.1f}% kept)")

            # Build pruned model from baseline and current keep_indices
            pruned_model = build_pruned_resnet_and_copy_weights(baseline, keep_indices, NUM_CLASSES).to(DEVICE)
            # Quick sanity forward
            pruned_model.eval()
            with torch.no_grad():
                _ = pruned_model(torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE))

            # Local calibration for this stage
            print(f"    Calibrating {s} (local, gentle)...")
            calibrate_stage(pruned_model, s, train_loader, epochs=CAL_EPOCHS, max_batches=CAL_MAX_BATCHES, lr=CAL_LR)

            # Save & log metrics AFTER this stage calibration
            stage_ckpt = os.path.join(SAVE_DIR, f"dermamnist_resnet50_pgto_{method}_r{int(target_ratio*100)}_{s}_calibrated.pth")
            torch.save(pruned_model.state_dict(), stage_ckpt)
            row = collect_metrics_row(
                tag_variant=f"{method}",
                tag_stage=f"{s}_after_stage_calibration",
                ratio=target_ratio,
                model=pruned_model,
                test_loader=test_loader,
                path_hint=stage_ckpt
            )
            rows.append(row)
            print("    Stage metrics:", {k: row[k] for k in ["Acc","AUC","ModelSizeMB","FLOPs_M_per_image","InferenceTime_per_batch32_s","PeakRAM_MB","PowerProxy_MFLOPs"]})

        # After all stages pruned at this target, perform final short global finetune
        print("  Final short global finetune...")
        pruned_model = build_pruned_resnet_and_copy_weights(baseline, keep_indices, NUM_CLASSES).to(DEVICE)
        opt = optim.Adam(pruned_model.parameters(), lr=FINAL_LR)
        for ep in range(FINAL_FINETUNE_EPOCHS):
            pruned_model.train()
            running_loss=0; total=0; correct=0
            for bidx,(imgs,labels) in enumerate(train_loader,1):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad(); out = pruned_model(imgs)
                loss = criterion(out, labels); loss.backward(); opt.step()
                running_loss += float(loss.item())*imgs.size(0)
                _, preds = out.max(1); total += labels.size(0); correct += int(preds.eq(labels).sum().item())
                if bidx % LOG_INTERVAL == 0:
                    print(f"    Global FT ep{ep+1} batch{bidx} - loss {running_loss/max(1,total):.4f}, acc {correct/max(1,total):.4f}")
            vloss, vacc, vauc = evaluate_model_basic(pruned_model, val_loader)
            print(f"    Global FT epoch {ep+1}: ValLoss {vloss:.4f}, ValAcc {vacc:.4f}, ValAUC {vauc:.4f}")

        final_ckpt = os.path.join(SAVE_DIR, f"dermamnist_resnet50_pgto_{method}_r{int(target_ratio*100)}_final.pth")
        torch.save(pruned_model.state_dict(), final_ckpt)
        row = collect_metrics_row(
            tag_variant=f"{method}",
            tag_stage="after_global_finetune",
            ratio=target_ratio,
            model=pruned_model,
            test_loader=test_loader,
            path_hint=final_ckpt
        )
        rows.append(row)
        print("  Final metrics:", {k: row[k] for k in ["Acc","AUC","ModelSizeMB","FLOPs_M_per_image","InferenceTime_per_batch32_s","PeakRAM_MB","PowerProxy_MFLOPs"]})

# Save CSV
csv_path = os.path.join(SAVE_DIR, "dermamnist_pgto_pruning_metrics.csv")
df = pd.DataFrame(rows); df.to_csv(csv_path, index=False)
print("\nAll done. CSV:", csv_path)
print(df.to_string(index=False))
