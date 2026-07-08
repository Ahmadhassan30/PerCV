"""
Run CNN classification pipeline (Task 4) and compare results against documented baseline.
"""

import sys
import os
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from percv_cv.config import CONFIG
from percv_cv.cnn import build_model, load_checkpoint, predict, evaluate, benchmark_backbone
from percv_cv.gradcam import generate_gradcam


def discover_intel_data(data_dir_arg: str | None) -> Path:
    """Resolve Intel dataset test directory dynamically or via arguments."""
    if data_dir_arg:
        path = Path(data_dir_arg)
        if path.exists():
            return path
        raise FileNotFoundError(f"Data directory does not exist: {data_dir_arg}")

    # Fallback search path in CONFIG input_root or /kaggle/input or local
    search_paths = [
        Path(CONFIG.input_root),
        Path("/kaggle/input"),
        Path("./datasets"),
        Path("scratch/dummy_intel")
    ]

    for base_path in search_paths:
        if not base_path.exists():
            continue
        # Check if dummy matching folder exists first
        if (base_path / "test").exists():
            return base_path / "test"

        # Search for Intel Classification folder
        for p in base_path.rglob("*"):
            if "intel" in p.name.lower() and p.is_dir():
                for sub in p.rglob("seg_test"):
                    if sub.is_dir():
                        return sub
                return p

    raise FileNotFoundError(
        "Could not dynamically discover Intel Classification dataset. Please run dummy generator or specify path via --data-dir."
    )


def save_confusion_matrix_plot(cm: list[list[int]], class_names: list[str], path: Path):
    """Draw a basic text-based visual representation of confusion matrix or write the image using OpenCV."""
    h_c = 400
    w_c = 400
    canvas = np.ones((h_c, w_c, 3), dtype=np.uint8) * 255
    
    # Draw simple grid and values
    grid_size = len(class_names)
    cell_w = w_c // grid_size
    cell_h = h_c // grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            val = cm[i][j]
            cv2.rectangle(canvas, (j*cell_w, i*cell_h), ((j+1)*cell_w, (i+1)*cell_h), (200, 200, 200), 1)
            cv2.putText(canvas, str(val), (j*cell_w + 30, i*cell_h + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Print class labels on axis edges
            if i == 0:
                cv2.putText(canvas, class_names[j][:4], (j*cell_w + 10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 0, 0), 1)
            if j == 0:
                cv2.putText(canvas, class_names[i][:4], (10, i*cell_h + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 0, 0), 1)

    cv2.imwrite(str(path), canvas)


def main():
    parser = argparse.ArgumentParser(description="Task 4 CNN Classification pipeline.")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "mobilenetv2"], help="CNN backbone architecture.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weight checkpoint .pt file.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to Intel test/eval dataset directory.")
    parser.add_argument("--out", type=str, default="outputs/classify", help="Path to save outputs.")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Resolve Dataset
    try:
        test_dir = discover_intel_data(args.data_dir)
        print(f"Intel test dataset resolved at: {test_dir}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    class_names = ["buildings", "forest", "mountain", "street"]
    
    # Setup test transform matching val_tx
    val_tx = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        test_dataset = ImageFolder(root=str(test_dir), transform=val_tx)
        # Verify classes matches expected targets
        loaded_classes = [c.lower() for c in test_dataset.classes]
        if not all(c in loaded_classes for c in class_names):
            print(f"WARNING: ImageFolder classes {loaded_classes} do not match expected targets {class_names}.")
    except Exception as e:
        print(f"ERROR: Failed to load dataset from {test_dir}: {e}")
        sys.exit(1)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 2. Build Model
    model = build_model(args.backbone, len(class_names))

    # 3. Load Checkpoint weights
    ckpt_path = args.checkpoint
    if not ckpt_path:
        # Check fallback options
        candidate_paths = ["model_best.pt", "outputs/experiments/experiment_001/models/model_best.pt"]
        for p in candidate_paths:
            if os.path.exists(p):
                ckpt_path = p
                break

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            model = load_checkpoint(model, ckpt_path, device)
            print(f"Checkpoint weights loaded successfully from: {ckpt_path}")
        except Exception as e:
            print(f"WARNING: Failed to load checkpoint state dict: {e}. Proceeding with initialized parameters.")
            model.to(device)
            model.eval()
    else:
        print("WARNING: No model checkpoint found. Running evaluation on default/random weights.")
        model.to(device)
        model.eval()

    # 4. Evaluate Pipeline
    metrics = evaluate(model, test_loader, class_names)

    # Save detailed metrics JSON
    with open(output_dir / f"metrics_{args.backbone}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save Confusion Matrix Visual
    save_confusion_matrix_plot(metrics["confusion_matrix"], class_names, output_dir / f"confusion_matrix_{args.backbone}.png")

    # 5. Benchmark Performance
    sample_tensor = torch.zeros((1, 3, 128, 128))
    bench = benchmark_backbone(model, sample_tensor, n_runs=10)
    print(f"\nThroughput Benchmark:\n  Params: {bench['params_millions']:.2f} M\n  Size: {bench['size_mb']:.2f} MB\n  Speed: {bench['fps']:.2f} FPS")

    # 6. Generate Grad-CAM overlays (Task 4 visual mapping)
    gradcam_dir = output_dir / "gradcam" / args.backbone
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    # Select target layer name depending on backbone
    target_layer_name = "layer4.1.conv2" if args.backbone == "resnet18" else "features.18.0"

    # Find one sample for building and one for street to demonstrate classification failure visual overlays
    street_idx = -1
    building_idx = -1
    for idx, (path, label_idx) in enumerate(test_dataset.samples):
        cls_label = test_dataset.classes[label_idx].lower()
        if cls_label == "street" and street_idx == -1:
            street_idx = idx
        if cls_label == "buildings" and building_idx == -1:
            building_idx = idx
        if street_idx != -1 and building_idx != -1:
            break

    # Run Grad-CAM on selected samples
    for name, idx in [("street", street_idx), ("building", building_idx)]:
        if idx == -1:
            continue
        # Load raw image directly from path for backdrop
        img_path, label_idx = test_dataset.samples[idx]
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            continue
        
        try:
            overlay = generate_gradcam(model, raw_img, target_class=label_idx, target_layer_name=target_layer_name)
            cv2.imwrite(str(gradcam_dir / f"gradcam_sample_{name}.png"), overlay)
            print(f"Grad-CAM overlay saved for '{name}' sample to: {gradcam_dir}")
        except Exception as e:
            print(f"WARNING: Grad-CAM generation failed for '{name}': {e}")

    # 7. Write comparison baseline dashboard summary if matching baseline json exists
    baseline_metrics_path = Path("artifacts/baseline_metrics.json")
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path) as f:
            base_data = json.load(f)
        
        # Load backbone timings
        resnet_train_time = base_data["task4"]["resnet18"]["train_time_sec"]
        mobilenet_train_time = base_data["task4"]["mobilenetv2"]["train_time_sec"]

        # If resnet18 is evaluated, compile row
        # Load Mobilenet metrics from baseline JSON as comparison targets
        comparison = {
            "resnet18": {
                "accuracy": metrics["accuracy"] if args.backbone == "resnet18" else base_data["task4"]["resnet18"]["accuracy"],
                "f1_score": metrics["macro_f1"] if args.backbone == "resnet18" else base_data["task4"]["resnet18"]["f1_score"],
                "params_m": bench["params_millions"] if args.backbone == "resnet18" else base_data["task4"]["resnet18"]["params_m"],
                "size_mb": bench["size_mb"] if args.backbone == "resnet18" else base_data["task4"]["resnet18"]["size_mb"],
                "speed_fps": bench["fps"] if args.backbone == "resnet18" else base_data["task4"]["resnet18"]["speed_fps"],
                "train_time_sec": resnet_train_time
            },
            "mobilenetv2": {
                "accuracy": metrics["accuracy"] if args.backbone == "mobilenetv2" else base_data["task4"]["mobilenetv2"]["accuracy"],
                "f1_score": metrics["macro_f1"] if args.backbone == "mobilenetv2" else base_data["task4"]["mobilenetv2"]["f1_score"],
                "params_m": bench["params_millions"] if args.backbone == "mobilenetv2" else base_data["task4"]["mobilenetv2"]["params_m"],
                "size_mb": bench["size_mb"] if args.backbone == "mobilenetv2" else base_data["task4"]["mobilenetv2"]["size_mb"],
                "speed_fps": bench["fps"] if args.backbone == "mobilenetv2" else base_data["task4"]["mobilenetv2"]["speed_fps"],
                "train_time_sec": mobilenet_train_time
            }
        }
        with open(output_dir / "backbone_comparison.json", "w") as f:
            json.dump(comparison, f, indent=4)
        print(f"Comparison dashboard JSON written to: {output_dir / 'backbone_comparison.json'}")

    # Compare with documented baseline (resnet18 test accuracy: 0.9277, macro F1: 0.9262)
    baseline_acc = 0.9277
    baseline_f1 = 0.9262
    
    print("\n=== Baseline Comparison ===")
    print(f"Backbone: {args.backbone.upper()}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f} (Baseline: {baseline_acc:.4f})")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f} (Baseline: {baseline_f1:.4f})")
    
    # Delta check if evaluating ResNet18 using the real BDD100K/Intel test split
    if args.backbone == "resnet18" and test_dir.name != "test" and os.path.exists("model_best.pt"):
        acc_delta = abs(metrics["accuracy"] - baseline_acc)
        if acc_delta > 0.05:
            print("WARNING: Deviation from documented classification baseline. Check checkpoint file or test dataloader.")
        else:
            print("SUCCESS: Performance matches documented baseline within accepted tolerance.")


if __name__ == "__main__":
    main()
