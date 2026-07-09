"""
percv_cv.cnn — Task 4: CNN model training and evaluation wrapper.

Supports ResNet18 (active model) and MobileNetV2 (baseline).
"""

from __future__ import annotations
import os
import time
import io
import cv2
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Confirmed model and preprocessing constants
CLASS_NAMES = ["buildings", "forest", "mountain", "street"]
INPUT_SIZE = (128, 128)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(backbone: str, num_classes: int) -> torch.nn.Module:
    """
    Construct a ResNet18 or MobileNetV2 classification model with replaced classifier head.
    Matches the architecture defined in the notebook.
    """
    backbone_type = backbone.lower()
    if backbone_type == "mobilenetv2":
        try:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        except AttributeError:
            model = models.mobilenet_v2(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    else:
        # Default: resnet18
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def load_checkpoint(model: torch.nn.Module, path: str, device: str) -> torch.nn.Module:
    """Loads state_dict checkpoint from disk (bare or wrapped), moves to device, sets to eval mode."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint model path does not exist: {path}")

    loaded = torch.load(path, map_location=torch.device(device))
    if isinstance(loaded, dict) and "state_dict" in loaded:
        state_dict = loaded["state_dict"]
    else:
        state_dict = loaded

    # Strip 'module.' prefix if present (from DataParallel training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def load_model(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """Builds the architecture and loads the checkpoint from the path (supporting both rich & bare formats)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    loaded = torch.load(checkpoint_path, map_location=torch.device(device))
    
    # Determine backbone type (default to resnet18 if not metadata-wrapped)
    backbone = "resnet18"
    if isinstance(loaded, dict) and "backbone" in loaded:
        backbone = loaded["backbone"]

    model = build_model(backbone, len(CLASS_NAMES))
    
    if isinstance(loaded, dict) and "state_dict" in loaded:
        state_dict = loaded["state_dict"]
    else:
        state_dict = loaded

    # Strip 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def predict(model: torch.nn.Module, image: np.ndarray, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Predict the class and probabilities for a single BGR image.

    Args:
        model: Trained PyTorch model in eval mode.
        image: BGR image (H, W, 3).
        class_names: List of class labels (ignored in favor of fixed CLASS_NAMES).

    Returns:
        Dict containing predicted label, confidence, and full probs mapping.
    """
    device = next(model.parameters()).device
    h_in, w_in = INPUT_SIZE

    # Preprocessing must match training exactly
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (w_in, h_in))
    img_norm = img_resized.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img_norm = (img_norm - mean) / std
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)

    # Ensure eval mode
    model.eval()

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    probs_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return {
        "label": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": probs_dict,
        "probs_list": probs.tolist()
    }


def evaluate(model: torch.nn.Module, dataloader: Any, class_names: List[str]) -> Dict[str, Any]:
    """
    Runs evaluation on the dataset split loader, returning metrics and confusion matrix.
    """
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Computations
    total = len(all_targets)
    correct = int(np.sum(all_preds == all_targets))
    accuracy = correct / total if total > 0 else 0.0

    # Calculate metrics manually to match sklearn behavior without depending on scikit-learn
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(all_targets, all_preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion[t, p] += 1

    per_class = {}
    f1_scores = []

    for i, cls in enumerate(class_names):
        tp = confusion[i, i]
        fp = np.sum(confusion[:, i]) - tp
        fn = np.sum(confusion[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_scores.append(f1)
        per_class[cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist()
    }


def benchmark_backbone(model: torch.nn.Module, sample_input: torch.Tensor, n_runs: int = 50) -> Dict[str, Any]:
    """
    Returns parameter counts, size, and inference throughput (FPS).
    """
    params = sum(p.numel() for p in model.parameters())
    params_millions = params / 1e6

    # Model serialized size estimation
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = len(buffer.getvalue()) / (1024.0 * 1024.0)

    # Measure FPS
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample_input)

    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(sample_input)
    duration = time.time() - start
    fps = (n_runs * sample_input.size(0)) / duration

    return {
        "params_millions": float(params_millions),
        "size_mb": float(size_mb),
        "fps": float(fps)
    }
