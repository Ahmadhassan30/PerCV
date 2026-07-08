"""
percv_cv.cnn — Task 4: CNN transfer learning (model load / predict / eval).

Supports ResNet18 (active) and MobileNetV2 (baseline comparison).
Handles model construction, head replacement, training loop, and
inference evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Dict, Optional


def build_model(backbone: str = "resnet18",
                num_classes: int = 4,
                pretrained: bool = True) -> "object":
    """Construct a classification model with a replaced head.

    Supported backbones: 'resnet18', 'mobilenetv2'.
    """
    raise NotImplementedError("Algorithm logic deferred to Prompt 2+")


def train_model(model: "object", train_loader: "object",
                val_loader: "object", config: "Dict") -> "Dict":
    """Fine-tune with Adam, return epoch-wise metrics dict."""
    raise NotImplementedError("Algorithm logic deferred to Prompt 2+")


def evaluate_model(model: "object", test_loader: "object") -> "Dict":
    """Run inference on the test split, return accuracy/F1/FPS."""
    raise NotImplementedError("Algorithm logic deferred to Prompt 2+")


def load_checkpoint(path: "Path", backbone: str = "resnet18",
                    num_classes: int = 4) -> "object":
    """Load a saved .pt checkpoint and return the model in eval mode."""
    raise NotImplementedError("Algorithm logic deferred to Prompt 2+")
