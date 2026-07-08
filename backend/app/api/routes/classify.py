"""CNN Classification & Grad-CAM routes — Task 4."""

import base64
from pathlib import Path
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any

from app.core.models_state import TORCH_AVAILABLE, LOADED_MODELS, AVAILABLE_BACKBONES

router = APIRouter()


class ClassifyResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    gradcam_base64: str


@router.post("", response_model=ClassifyResponse)
async def classify_image(
    file: UploadFile = File(...),
    backbone: str = Query("resnet18", description="CNN backbone: resnet18 or mobilenetv2")
):
    """
    Accepts one image.
    Runs CNN transfer learning prediction + hooks into final conv layer for Grad-CAM.
    Returns prediction, probs dict, and base64 Grad-CAM activation heatmap overlay.
    """
    allowed_exts = {".jpg", ".jpeg", ".png"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.filename}. Only JPG, JPEG, and PNG are allowed."
        )

    # Validate backbone selection
    backbone_type = backbone.lower()
    if backbone_type not in ["resnet18", "mobilenetv2"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid backbone option: '{backbone}'. Must be 'resnet18' or 'mobilenetv2'."
        )

    # Read image contents
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to decode uploaded image."
        )

    class_names = ["buildings", "forest", "mountain", "street"]

    # 1. Run actual prediction if PyTorch is available and checkpoint loaded
    if TORCH_AVAILABLE and backbone_type in LOADED_MODELS:
        try:
            from percv_cv.cnn import predict
            from percv_cv.gradcam import generate_gradcam

            model = LOADED_MODELS[backbone_type]
            
            # Predict
            pred_res = predict(model, img, class_names)
            predicted_class = pred_res["label"]
            confidence = pred_res["confidence"]
            probs_dict = pred_res["probs"]

            # Select target layer name depending on backbone
            target_layer_name = "layer4.1.conv2" if backbone_type == "resnet18" else "features.18.0"
            
            # Retrieve predicted label index
            pred_idx = class_names.index(predicted_class)
            
            # Run Grad-CAM
            overlay = generate_gradcam(model, img, target_class=pred_idx, target_layer_name=target_layer_name)

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference pipeline execution error: {e}"
            )
    else:
        # Stub fallback mode for local testing without checkpoints/PyTorch
        # Check if the backbone was requested but missing (when torch is available, it means checkpoint is absent)
        if TORCH_AVAILABLE and backbone_type not in AVAILABLE_BACKBONES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Backbone checkpoint '{backbone_type}' is disabled/unavailable. Please load model weights first."
            )

        # Mock results
        predicted_class = "street"
        confidence = 0.9277
        probs_dict = {
            "buildings": 0.0512,
            "forest": 0.0102,
            "mountain": 0.0109,
            "street": 0.9277
        }
        # Create a simple mock overlay (colored rectangle)
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 20), (100, 100), (0, 0, 255), 3)

    # Encode overlay
    _, encoded_img = cv2.imencode(".png", overlay)
    overlay_base64 = base64.b64encode(encoded_img).decode("utf-8")

    return ClassifyResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probs_dict,
        gradcam_base64=overlay_base64
    )
