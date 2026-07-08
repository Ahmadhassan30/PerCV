import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn
    from percv_cv.cnn import build_model, predict
    from percv_cv.gradcam import generate_gradcam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed in the local environment")
def test_build_model_shape():
    # Build models for 4 classes
    resnet = build_model("resnet18", num_classes=4)
    mobilenet = build_model("mobilenetv2", num_classes=4)

    dummy_batch = torch.zeros((2, 3, 128, 128))

    # Evaluate outputs
    out_resnet = resnet(dummy_batch)
    out_mobilenet = mobilenet(dummy_batch)

    assert out_resnet.shape == (2, 4), f"Expected ResNet shape (2, 4), got {out_resnet.shape}"
    assert out_mobilenet.shape == (2, 4), f"Expected MobileNet shape (2, 4), got {out_mobilenet.shape}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed in the local environment")
def test_predict_probabilities():
    resnet = build_model("resnet18", num_classes=4)
    resnet.eval()

    # Create dummy BGR image (128 x 128 x 3)
    dummy_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    class_names = ["buildings", "forest", "mountain", "street"]

    result = predict(resnet, dummy_img, class_names)

    # Check keys
    assert "label" in result
    assert "confidence" in result
    assert "probs" in result

    # Check sum of probabilities equals 1.0 (with precision tolerance)
    probs_sum = sum(result["probs"].values())
    assert pytest.approx(probs_sum, abs=1e-5) == 1.0
    assert result["label"] in class_names
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed in the local environment")
def test_generate_gradcam_output():
    resnet = build_model("resnet18", num_classes=4)
    resnet.eval()

    # Create dummy BGR image (128 x 128 x 3)
    dummy_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    # ResNet18 last conv layer module name
    target_layer_name = "layer4.1.conv2"

    overlay = generate_gradcam(resnet, dummy_img, target_class=0, target_layer_name=target_layer_name)

    # Output overlaid image should have same dimensions as input BGR image
    assert overlay.shape == dummy_img.shape
    assert overlay.dtype == np.uint8
    # Assert values are scaled within normal image bounds
    assert 0 <= np.min(overlay) <= 255
    assert 0 <= np.max(overlay) <= 255
