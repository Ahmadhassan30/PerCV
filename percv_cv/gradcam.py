"""
percv_cv.gradcam — Task 4: Hook-based Grad-CAM activation maps.

Manually registers forward/backward hooks on the final convolutional
layer to extract activations and gradients, then computes class-specific
spatial heatmaps overlaid on input images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class GradCAMExtractor:
    """Hook-based Grad-CAM for a given CNN model and target layer."""

    def __init__(self, model: "object", target_layer: str = "layer4"):
        self.model = model
        self.target_layer = target_layer
        self._activation = None
        self._gradient = None

    def _register_hooks(self) -> None:
        """Attach forward/backward hooks to the target layer."""
        raise NotImplementedError("Algorithm logic deferred to Prompt 2+")

    def generate(self, input_tensor: "object",
                 class_idx: int | None = None) -> "np.ndarray":
        """Compute Grad-CAM heatmap for a single input tensor.

        Returns the heatmap as a numpy array (H, W) in [0, 1].
        """
        raise NotImplementedError("Algorithm logic deferred to Prompt 2+")

    def overlay(self, image: "np.ndarray",
                heatmap: "np.ndarray",
                alpha: float = 0.5) -> "np.ndarray":
        """Blend a Grad-CAM heatmap onto the original image."""
        raise NotImplementedError("Algorithm logic deferred to Prompt 2+")
