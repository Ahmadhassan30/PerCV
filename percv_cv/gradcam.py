"""
percv_cv.gradcam — Task 4: Hook-based Grad-CAM activation maps.

Manually registers forward hooks on the final convolutional layer to extract
activations and gradients, computing class-specific overlays.
"""

from __future__ import annotations
import cv2
import torch
import numpy as np
from typing import Tuple, Optional


class HookGradCAM:
    """Manually registers a forward hook on a target layer to compute activation maps."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activation = None
        self.hook = self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activation = output
        output.retain_grad()

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> Tuple[np.ndarray, int]:
        self.model.zero_grad()
        outputs = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(outputs, dim=1).item()
        score = outputs[0, class_idx]
        score.backward()

        gradients = self.activation.grad[0].detach().cpu().numpy()
        activations = self.activation[0].detach().cpu().numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        mx = np.max(cam)
        if mx > 0:
            cam = cam / mx
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx

    def release(self):
        self.hook.remove()


def register_hooks(model: torch.nn.Module, target_layer_name: str) -> HookGradCAM:
    """
    Manual forward hook registration on the final conv layer.
    """
    modules = dict(model.named_modules())
    if target_layer_name not in modules:
        # Fallback helper to search by name suffix/token if named module not exactly matched
        matched = None
        for name, mod in model.named_modules():
            if name.endswith(target_layer_name):
                matched = mod
                break
        if matched is None:
            raise KeyError(f"Target convolutional layer name '{target_layer_name}' not resolved in model modules.")
        target_layer = matched
    else:
        target_layer = modules[target_layer_name]

    return HookGradCAM(model, target_layer)


def generate_gradcam(model: torch.nn.Module, image: np.ndarray,
                     target_class: int | None, target_layer_name: str) -> np.ndarray:
    """
    Full forward+backward pass, weighted activation map, upsampled and
    overlaid as a heatmap on the original image (BGR).

    Args:
        model: PyTorch model.
        image: Original BGR image (H, W, 3) in [0, 255] or [0, 1].
        target_class: Integer index of target class, or None.
        target_layer_name: String identifier of final conv layer.

    Returns:
        Overlaid image of shape (H, W, 3), scaled to [0, 255] (uint8).
    """
    device = next(model.parameters()).device
    h_in, w_in = 128, 128

    # Preprocess image for inference (BGR to normalized PyTorch tensor)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (w_in, h_in))
    img_norm = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    # Register hook
    extractor = register_hooks(model, target_layer_name)

    try:
        # Run forward & backward passes
        cam, pred_class = extractor.generate(img_tensor, class_idx=target_class)
    finally:
        # Guarantee hook removal
        extractor.release()

    # Normalize heatmap visualization
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    h_o, w_o = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap_rgb, (w_o, h_o))

    # Scale input image to [0, 1] for blending
    img_norm_blend = image.astype(np.float32)
    if np.max(img_norm_blend) > 1.0:
        img_norm_blend /= 255.0

    overlay = img_norm_blend + (heatmap_resized.astype(np.float32) / 255.0) * 0.40
    overlay = overlay / np.max(overlay)

    # Return as BGR image [0, 255]
    overlay_bgr = cv2.cvtColor((overlay * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return overlay_bgr
