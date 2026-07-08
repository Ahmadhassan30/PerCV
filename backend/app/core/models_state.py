import os
import logging
from typing import Dict, Any

logger = logging.getLogger("percv-backend")

# Try to import torch and cnn modules
try:
    import torch
    from percv_cv.cnn import build_model, load_checkpoint
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch is not available. Classification routes will operate in stub/mock mode.")

LOADED_MODELS: Dict[str, Any] = {}
AVAILABLE_BACKBONES = []


def init_models():
    """Load CNN checkpoints once at startup."""
    global LOADED_MODELS, AVAILABLE_BACKBONES
    if not TORCH_AVAILABLE:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    
    # Paths to search for checkpoints relative to project root
    resnet_candidates = [
        project_root / "model_best.pt",
        project_root / "outputs/experiments/experiment_001/models/model_best.pt",
        project_root / "artifacts/model_best.pt"
    ]
    mobilenet_candidates = [
        project_root / "model_mobilenetv2.pt",
        project_root / "outputs/experiments/experiment_001/models/model_mobilenetv2.pt",
        project_root / "artifacts/model_mobilenetv2.pt"
    ]

    # 1. Try to load ResNet18
    resnet_path = None
    for p in resnet_candidates:
        if os.path.exists(p):
            resnet_path = p
            break

    if resnet_path:
        try:
            model = build_model("resnet18", num_classes=4)
            model = load_checkpoint(model, resnet_path, device)
            LOADED_MODELS["resnet18"] = model
            AVAILABLE_BACKBONES.append("resnet18")
            logger.info(f"Loaded ResNet18 checkpoint from: {resnet_path}")
        except Exception as e:
            logger.warning(f"Failed to load ResNet18 checkpoint: {e}")
    else:
        logger.warning("ResNet18 checkpoint 'model_best.pt' not found at startup.")

    # 2. Try to load MobileNetV2
    mobilenet_path = None
    for p in mobilenet_candidates:
        if os.path.exists(p):
            mobilenet_path = p
            break

    if mobilenet_path:
        try:
            model = build_model("mobilenetv2", num_classes=4)
            model = load_checkpoint(model, mobilenet_path, device)
            LOADED_MODELS["mobilenetv2"] = model
            AVAILABLE_BACKBONES.append("mobilenetv2")
            logger.info(f"Loaded MobileNetV2 checkpoint from: {mobilenet_path}")
        except Exception as e:
            logger.warning(f"Failed to load MobileNetV2 checkpoint: {e}")
    else:
        logger.warning("MobileNetV2 checkpoint 'model_mobilenetv2.pt' not found at startup.")

    # Log warnings if only one or none is loaded
    if not LOADED_MODELS:
        logger.warning("No CNN checkpoints loaded. Classification queries will fall back to initialized weights.")
    elif len(LOADED_MODELS) < 2:
        loaded = list(LOADED_MODELS.keys())[0]
        missing = "mobilenetv2" if loaded == "resnet18" else "resnet18"
        logger.warning(f"Only '{loaded}' checkpoint loaded. Backbone option '{missing}' is disabled.")
