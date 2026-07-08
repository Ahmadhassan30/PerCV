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
1. **Task 1 (Edge & Line Tracking)**: Loads BDD100K road images, applies Gaussian smoothing to filter pavement textures. Computes Canny edge comparisons and fits probabilistic Hough lines to extract highway line segments. Evaluates quality scores across a batch of 20 images to output mean/std stats.
2. **Task 2 (SIFT Matching)**: Evaluates matching trade-offs under varying Lowe's ratio test thresholds ($0.60, 0.75, 0.90$) using SIFT descriptors extracted from HPatches sequences to assess the precision-recall envelope.
3. **Task 3 (Panorama Assembly)**: progressive pairwise feature alignment on 3 overlapping images from the panorama scene directory (sorted alphabetically). Solves homographies using RANSAC. Implements translation adjustments and manual distance-transform alpha blending to stitch frames without seams.
4. **Task 4 (Neural Category Fine-Tuning)**: Loads training/testing directories dynamically from the Intel dataset. Filters subfolders to target categories (`buildings`, `forest`, `mountain`, `street`) and fine-tunes a ResNet18/MobileNetV2 classifier head. Runs Grad-CAM visualizations on correct and incorrect/low-confidence test samples using PyTorch hooks.

---

## 3. Benchmarking Datasets

Every dataset in this pipeline represents the industry-standard benchmark for its specific computer vision task. Attach these datasets in your Kaggle notebook sidebar:

- **BDD100K** (Road images)
- **HPatches Sequence Release** (Feature matching pairs)
- **panorama** (Image stitching sequence containing `back/`, `front/`, and `room/` scenes)
- **Intel Image Classification** (Multi-class scene categories)

---

## 4. Installation & Setup

1. **Attach Datasets**: Under the **Data** pane in the right panel of the Kaggle notebook, click **+ Add Data** and search/attach the four datasets listed above.
2. **Configure Accelerators**: Under **Settings** -> **Accelerator**, choose **GPU T4 x2** or **GPU P100**.
3. **Import Notebook**: Download `percv_kaggle.ipynb` from this repository and upload it using **File** -> **Import Notebook** in Kaggle.

---

## 5. Dataset Path Discovery (`DatasetManager`)

The notebook includes a dedicated **Dataset Discovery & Validation** cell containing a reusable `DatasetManager` class. 

```python
dm = DatasetManager(CONFIG.get("input_root", "/kaggle/input"))
dm.print_dataset_tree(max_depth=2)
dm.validate_and_report()
```

- **Dynamic Discovery**: Scans `/kaggle/input` recursively for keyword matches (`bdd`, `hpatches`, `panorama`, `intel`), mapping mounted folders automatically.
- **Strict Validation**: Checks that all 4 datasets are attached. If any is missing, it raises a descriptive `FileNotFoundError` explaining which dataset is expected and how to attach it.
- **Path Resolution**: Provides pathlib.Path objects for files, matching target parameters automatically (e.g. sequence retrieval, scene selection, and class filtering).
- **Summary Dashboard**: Automatically compiles and displays a markdown table reporting loaded image counts, sequences, and scenes for confirmation.

---

## 6. Configuration Options (`CONFIG`)

The pipeline execution parameters are defined centrally in the `CONFIG` dictionary at the top of the notebook:

```python
CONFIG = {
    "experiment_id": "experiment_001",
    "seed": 42,
    "output_root": "outputs",
    
    # Task Parameters
    "task1": {
        "num_images": 20,  # Number of BDD100K images to load and benchmark
        "gaussian_ksize": (5, 5),
        "gaussian_sigma": 1.0,
        "canny_threshold_pairs": [...]
    },
    "task2": {
        "lowe_ratios": [0.60, 0.75, 0.90],
        "default_lowe_ratio": 0.75
    },
    "task3": {
        "scene": "room",  # Switch between 'room', 'front', and 'back' to load different sequences
        "ransac_threshold": 5.0
    },
    "task4": {
        "backbone": "resnet18",  # Options: 'resnet18' or 'mobilenetv2'
        "batch_size": 32,
        "epochs": 5
    }
}
```

Changing `CONFIG["task3"]["scene"]` automatically updates the path mapping to load the corresponding panorama frames from `/kaggle/input/panorama/<scene>/*.jpg` sorted alphabetically.

---

## 7. Output & Experiment Structure

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

## 8. Results & Performance Dashboard

The final cell prints a pipeline execution dashboard containing:
- Execution times for all pipeline modules and total duration.
- Test accuracy, precision, recall, and F1-score for the active model configuration.
- A **Neural Backbone Benchmark Dashboard** comparing accuracy, F1, parameters (M), model size (MB), speed (FPS), and training times between ResNet18 and MobileNetV2.
