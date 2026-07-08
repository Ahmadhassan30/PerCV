# PerCV: Computational Computer Vision & Neural Classification Benchmarking Pipeline

This directory contains the core Kaggle computational notebook `percv_kaggle.ipynb` for the **PerCV** project. It is structured as a reproducible end-to-end computer vision experimentation and deep learning pipeline designed to benchmark classical image processing, descriptors, warping transforms, and deep neural backbones.

---

## 1. Overview & Architecture

PerCV integrates classical computer vision techniques with modern deep learning paradigms. It targets four critical CV benchmarks and runs on Kaggle using GPU hardware acceleration.

```text
            [ Input Scene Images ]
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
   [ Task 1 ]     [ Task 2 ]     [ Task 3 ]
  Edge & Line        SIFT           SIFT
   Detection       Matching      Matching
       │              │              │
       │              ▼              ▼
       │         [ Evaluation ]  Homography
       │          Lowe's Ratio    (RANSAC)
       │         [0.6,0.75,0.9]      │
       │              │              ▼
       │              │          Perspective
       │              │            Warping
       │              │              │
       │              │              ▼
       │              │          [ Panorama ]
       │              │         Alpha Blended
       │              │
       │              └──────────────┐
       ▼                             ▼
 [ Visual plots ]               [ Task 4 ]
  & CSV Outputs             ResNet / MobileNet
                             (Fine-Tuning)
                                     │
                                     ▼
                               [ Evaluation ]
                              Metrics & Heatmap
                                 (Grad-CAM)
```

---

## 2. Pipeline Stages

The pipeline consists of the following progressive modules:
1. **Task 1 (Edge & Line Tracking)**: Applies Gaussian smoothing to filter pavement textures. Computes multi-threshold Canny edge configurations and fits probabilistic Hough lines to extract highway line segments.
2. **Task 2 (SIFT Matching)**: Detects scale and rotation invariant keypoints. Explores the matching trade-offs under varying Lowe's ratio test thresholds ($0.60, 0.75, 0.90$) to assess the precision-recall envelope.
3. **Task 3 (Panorama Assembly)**: progressive pairwise feature alignment. Solves homographies using RANSAC. Implements translation coordinate adjustments and manual distance-transform alpha blending to stitch 3 images without seam lines.
4. **Task 4 (Neural Category Fine-Tuning)**: Loads a pretrained convolutional backbone (ResNet18 or MobileNetV2), locks backbone representations, and trains a linear classifier head. Runs Grad-CAM visualizations using PyTorch gradient retention hooks to audit model activation attention.

---

## 3. Benchmarking Datasets

Every dataset in this pipeline represents the industry-standard benchmark for its specific computer vision task. Organize your inputs in the following directory layout:

```text
datasets/
├── road_images/              # Task 1: BDD100K / TuSimple / CULane road samples (e.g. road_view.jpg)
├── feature_pairs/            # Task 2: HPatches image pairs (e.g. scene_left.jpg, scene_right.jpg)
├── panorama/                 # Task 3: OpenPano or custom overlapping photos (3 panels)
└── scene_classification/     # Task 4: Intel Image Classification Dataset (subfolders: buildings, forest, mountain, street)
```

---

## 4. Installation & Setup

1. **Upload Datasets**: Upload the benchmark directories to Kaggle (e.g., as private datasets or using the Kaggle API).
2. **Configure Accelerators**: Open your Kaggle Notebook editor, go to **Settings** -> **Accelerator**, and choose **GPU T4 x2** or **GPU P100**.
3. **Import Notebook**: Download `percv_kaggle.ipynb` from this repository and upload it using **File** -> **Import Notebook** in Kaggle.

---

## 5. Configuration (`CONFIG`)

The execution parameters are defined centrally in the `CONFIG` dictionary at the top of the notebook:

```python
CONFIG = {
    "experiment_id": "experiment_001",
    "seed": 42,
    "output_root": "outputs",
    
    # Task Parameters
    "gaussian_ksize": (5, 5),
    "gaussian_sigma": 1.0,
    "canny_threshold_pairs": [
        {"low": 35, "high": 95, "label": "sensitive"},
        {"low": 75, "high": 155, "label": "balanced"},
        {"low": 125, "high": 245, "label": "strict"}
    ],
    "lowe_ratios": [0.60, 0.75, 0.90],
    "default_lowe_ratio": 0.75,
    "ransac_threshold": 5.0,
    "stitching_min_matches": 10,
    "distortion_det_threshold": 0.1,
    
    # CNN Parameters
    "backbone": "resnet18",  # Options: 'resnet18' or 'mobilenetv2'
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    
    # Dataset Paths
    "dataset_paths": {
        "road_images": "/kaggle/input/percv-road-images/road_view.jpg",
        "feature_pairs": {
            "img1": "/kaggle/input/percv-sift-images/scene_left.jpg",
            "img2": "/kaggle/input/percv-sift-images/scene_right.jpg"
        },
        "panorama": [
            "/kaggle/input/percv-stitch-images/view_left.jpg",
            "/kaggle/input/percv-stitch-images/view_middle.jpg",
            "/kaggle/input/percv-stitch-images/view_right.jpg"
        ],
        "scene_classification": "/kaggle/input/intel-image-classification/scene_dataset"
    }
}
```

*Note: If any dataset file or folder is missing at the specified paths during execution, the notebook raises a descriptive `FileNotFoundError` explaining what is expected.*

---

## 6. Output & Experiment Structure

The notebook logs all parameters, weights, metrics, and plots under an experiment directory:

```text
outputs/
├── experiments/
│   └── experiment_001/
│       ├── config.json         # Dump of CONFIG
│       ├── metrics.json        # Compiled metrics & stage-wise timing
│       ├── run_info.json       # Host, GPU, OS, random seed, library versions
│       ├── plots/              # Figures (Canny, SIFT, Panorama, Confusion Matrix, Grad-CAM overlays)
│       ├── models/             # Best model checkpoint weights (model_best.pt)
│       └── reports/            # Detailed experiment json logs
└── percv_artifacts.zip         # Compressed zip of all logs and plots
```

---

## 7. Results & Performance Dashboard

The final cell prints a pipeline execution dashboard containing:
- Execution times for all pipeline modules and total duration.
- Test accuracy, precision, recall, and F1-score for the active model configuration.
- A **Neural Backbone Benchmark Dashboard** comparing accuracy, F1, parameters (M), model size (MB), speed (FPS), and training times between ResNet18 and MobileNetV2.

---

## 8. Future Work & Extensions
- **Learned Features**: Benchmark SIFT against deep feature matching networks (e.g. SuperPoint + SuperGlue).
- **Stitching Envelopes**: Implement cylindrical/spherical warping canvas steps to handle non-planar parallax.
- **Explainability**: Integrate causal explainability modules such as Integrated Gradients or RISE.

---

## 9. References
- Balntas, V., et al. (2017). *HPatches: A benchmark and evaluation of local descriptors.* CVPR.
- Yu, F., et al. (2020). *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning.* CVPR.
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
