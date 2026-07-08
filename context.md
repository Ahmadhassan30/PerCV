# Project Context & Experimentation Log: PerCV

This document serves as the centralized engineering reference, architectural specification, and benchmark log for the **PerCV** computer vision pipeline. It details the system architecture, datasets, configurations, implementation decisions, and the latest GPU execution results.

---

## 1. Executive Summary

PerCV is a professional computer vision and deep learning benchmarking suite designed to evaluate and bridge classical image processing, descriptor-based geometric warping, and deep learning neural classification. The pipeline operates as a reproducible end-to-end flow, logging parameters, metrics, weight checkpoints, and validation plots under versioned experiment folders.

---

## 2. Directory Layout & Workspace Architecture

The workspace is organized into a clean engineering layout:

```text
PerCV/
├── notebooks/
│   ├── percv_kaggle.ipynb     # Jupyter Notebook containing the end-to-end pipeline
│   └── README.md              # Setup and execution guide for Kaggle
├── context.md                 # Project context, technical specifications, and training log (this file)
└── README.md                  # Main repository overview documentation
```

---

## 3. Pipeline Modules & Technical Specifications

### Task 1 — Edge Boundary & Lane Detection
* **Objective**: Evaluate probabilistic line tracking and boundary matching in driving scenes.
* **Input**: 20 recursively discovered road images from the BDD100K dataset.
* **Methodology**:
  1. Grayscale conversion followed by Gaussian smoothing ($K = 5 \times 5$, $\sigma = 1.0$) to filter out high-frequency asphalt textures.
  2. Sequential evaluation of three Canny configurations: **Sensitive** ($35, 95$), **Balanced** ($75, 155$), and **Strict** ($125, 245$).
  3. Fitting of Probabilistic Hough Line Transforms to trace linear road markings.
  4. Calculation of a **Lanes Quality Score**: The ratio of Hough line segments falling inside the realistic lane boundary angle slopes ($[25^\circ, 75^\circ]$ and $[105^\circ, 155^\circ]$) relative to total detections.
* **Failure Analysis Scenario**: Dense clutter (urban signs, vehicles, vegetation) in suburban frames yields low quality scores due to false-positive lines, highlighting the limitations of non-semantic edge filtering in chaotic real-world environments.

### Task 2 — Descriptor Extraction & Match Scaling
* **Objective**: Assess scale and rotation-invariant feature descriptors under matching constraints.
* **Input**: Image sequence pairs (`1.*` and `2.*`) dynamically located from HPatches sequences (e.g. `i_ajuntament`).
* **Methodology**:
  1. Compute SIFT keypoints and descriptors.
  2. Implement Bruteforce L2 matching followed by **Lowe's ratio test** evaluated over three thresholds: **Strict** ($0.60$), **Recommended** ($0.75$), and **Lenient** ($0.90$).
  3. Log match counts and inlier proportions to analyze the precision-recall trade-offs.

### Task 3 — Planar Homography & Auto-Cropped Panorama Stitching
* **Objective**: Stitch overlapping frames into a unified perspective without pre-built APIs.
* **Input**: Dynamically matched scenes (`front`, `back`, `room`) containing 2 or 3 overlapping images.
* **Methodology**:
  1. Extract SIFT keypoints across Left, Middle, and Right images (Middle acting as the anchor coordinate system).
  2. Compute direct, independent frame-to-anchor homographies using RANSAC to eliminate outliers:
     $$\mathcal{H}_{\text{Left}\to\text{Middle}} = \text{findHomography}(\text{kp}_{\text{Left}}, \text{kp}_{\text{Middle}}, \text{RANSAC})$$
     $$\mathcal{H}_{\text{Right}\to\text{Middle}} = \text{findHomography}(\text{kp}_{\text{Right}}, \text{kp}_{\text{Middle}}, \text{RANSAC})$$
  3. Project the corners of all three images using `cv2.perspectiveTransform` to compute the exact global canvas bounding box ($x_{\min}, y_{\min}, x_{\max}, y_{\max}$).
  4. Construct the translation shift matrix $T$ and warp all frames using $T \cdot H$ to map all coordinates into positive space.
  5. Compute L2 Euclidean distance-transform weight maps (`cv2.distanceTransform`) representing boundary proximity. Warped weight maps are masked using binary warped image content to prevent border interpolation bleed.
  6. Blend overlapping pixels: $I_{\text{blended}} = \frac{\sum W_k \cdot I_k}{\sum W_k}$.
  7. **Auto-Cropping**: Fills interior holes in the thresholded binary mask using external contours to prevent dark scene pixels from collapsing the bounding box, then iteratively shrinks the tight bounding box corners until all boundary edges are non-black.

### Task 4 — CNN Fine-Tuning & Spatial Activation Hooks
* **Objective**: Transfer learning fine-tuning and visual explanation mapping.
* **Input**: Split target categories (`buildings`, `forest`, `mountain`, `street`) loaded from the Intel Classification Dataset.
* **Methodology**:
  1. Load pre-trained models (`resnet18` or `mobilenetv2`) and replace the classification head.
  2. Inject spatial augmentations (random horizontal flips, rotations, color jitter) on the training dataset.
  3. Fine-tune utilizing the Adam optimizer ($LR = 0.001$, Weight Decay $= 10^{-4}$).
  4. Manually extract gradients and activations from the final convolutional layer using PyTorch forward/backward hooks to calculate **Grad-CAM** activation maps overlaid on target test categories.
* **Failure Analysis Scenario**: street vs. building categories co-occur frequently in urban canyons. Grad-CAM shows the model correctly activation-maps architecture, clarifying classification confusions.

---

## 4. central CONFIG Settings

The execution is governed by the centralized config at the top of the notebook:

```python
CONFIG = {
    "experiment_id": "experiment_001",
    "seed": 42,
    "output_root": "outputs",
    "input_root": "/kaggle/input",
    
    "task1": {
        "num_images": 20,
        "gaussian_ksize": (5, 5),
        "gaussian_sigma": 1.0,
        "canny_threshold_pairs": [
            {"low": 35, "high": 95, "label": "sensitive"},
            {"low": 75, "high": 155, "label": "balanced"},
            {"low": 125, "high": 245, "label": "strict"}
        ],
        "hough_rho": 1,
        "hough_theta_deg": 1.0,
        "hough_threshold": 45,
        "hough_min_line_len": 30,
        "hough_max_line_gap": 12
    },
    "task2": {
        "lowe_ratios": [0.60, 0.75, 0.90],
        "default_lowe_ratio": 0.75
    },
    "task3": {
        "scene": "front",
        "ransac_threshold": 5.0,
        "stitching_min_matches": 10,
        "distortion_det_threshold": 0.1
    },
    "task4": {
        "backbone": "resnet18",
        "batch_size": 32,
        "epochs": 5,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "gradcam_target_samples": 4
    }
}
```

---

## 5. Kaggle Benchmark Execution Logs & Results

The latest run executed successfully under the Kaggle Linux environment.

### System & Hardware Specifications
* **Operating System**: Linux (Kernel: `6.12.90+`)
* **Python version**: `3.12.13`
* **CPU Cores / Memory**: 4 Cores | 31.35 GB RAM
* **PyTorch / Torchvision / OpenCV**: PyTorch `2.10.0+cu128` | TorchVision `0.25.0+cu128` | OpenCV `4.13.0`
* **GPU Hardware**: NVIDIA Tesla T4 (CUDA `12.8` Active)

---

### Executed Stage Outputs & Metrics

#### 1. Dataset Resolution & Verification
```text
=== Dataset Attachment Validation ===
✓ BDD100K                        found at: /kaggle/input/datasets/alvaromalfaro/bdd100k
✓ HPatches Sequence Release      found at: /kaggle/input/datasets/javidtheimmortal/hpatches-sequence-release
✓ panorama                       found at: /kaggle/input/datasets/ahmadhassan111111/panorama
✓ Intel Image Classification     found at: /kaggle/input/datasets/puneet6060/intel-image-classification

Scanning panorama folder '/kaggle/input/datasets/ahmadhassan111111/panorama/percv-panorama/front'...
  - Discovered file: 'front_02.jpeg' (is_file=True, suffix='.jpeg')
  - Discovered file: 'front_03.jpeg' (is_file=True, suffix='.jpeg')
  - Discovered file: 'front_01.jpeg' (is_file=True, suffix='.jpeg')

=== Benchmark Dataset Summary Dashboard ===
Dataset                  Discovered & Resolved Configs
BDD100K                  20 road images loaded recursively
HPatches                 Sequence: i_ajuntament (loaded: 1.* & 2.*)
Panorama                 Scene folder: front (3 overlapping frames loaded)
Intel Classification     11293 images matching targets (buildings, forest, mountain, street)
```

#### 2. Task 1 — Edge Detection Summary
* **Processed Frames**: 20
* **Execution Duration**: 3.521 seconds
* **Mean Quality Score**: **`0.1809`**
* **Analysis**: Balanced parameters effectively locate lane edges, but urban environment elements (signs, building shadows) generate off-angle Hough lines, lowering the overall score to 18%.

#### 3. Task 2 — SIFT Matching Summary
* **Evaluated Sequence**: `i_ajuntament`
* **Execution Duration**: 2.171 seconds
* **Matches under Recommended Lowe Ratio (0.75)**: **`348`** (Inlier Ratio: `0.4462`)
* **Lowe Ratio Threshold Precision-Recall Trend**:
  * $0.60$ (Strict): Fewer keypoint matches survive (higher precision, lower recall)
  * $0.75$ (Balanced): 348 good keypoints matched
  * $0.90$ (Lenient): Highest matching count with elevated false-positive matches (low precision)

#### 4. Task 3 — Panorama Stitching Summary (Scene: `front`)
* **Loaded Sequence**: `front_01.jpeg`, `front_02.jpeg`, `front_03.jpeg`
* **Anchor Frame**: `front_02.jpeg` (Middle)
* **Homography Matrices & RANSAC Results**:
  * **Pair: Left $\to$ Middle**:
    * Total Matches after Lowe Ratio: 1210
    * RANSAC Inliers: **`1044`**
    * Inlier Ratio: **`0.8628`**
    * Homography Matrix:
      $$\begin{bmatrix} 2.16921601 & 0.03799752 & -2033.60149 \\ 0.57090214 & 1.96367921 & -481.192294 \\ 0.00094343 & -0.00005133 & 1.0 \end{bmatrix}$$
  * **Pair: Right $\to$ Middle**:
    * Total Matches after Lowe Ratio: 3532
    * RANSAC Inliers: **`3142`**
    * Inlier Ratio: **`0.8896`**
    * Homography Matrix:
      $$\begin{bmatrix} 0.70636011 & -0.06273583 & 427.525068 \\ -0.11620705 & 0.84725722 & 139.654460 \\ -0.00020697 & -0.00004459 & 1.0 \end{bmatrix}$$
* **Stitching Metrics**:
  * Average Inlier Ratio: **`0.8762`**
  * Auto-cropped output successfully written to `plots/panorama.png` with clean, seam-feathered margins.

#### 5. Task 4 — CNN Fine-Tuning Summary (Backbone: `resnet18`)
* **Target Categories**: 4 classes (`buildings`, `forest`, `mountain`, `street`)
* **Total Epochs**: 5
* **Execution Duration**: 205.009 seconds
* **Epoch-wise Training Progression**:
  * **Epoch 1**: Train Loss: `0.4506` | Val Loss: `0.2590` | Val Accuracy: `91.88%` (Model checkpoint saved)
  * **Epoch 2**: Train Loss: `0.2652` | Val Loss: `0.2234` | Val Accuracy: `92.95%` (Model checkpoint saved)
  * **Epoch 3**: Train Loss: `0.2369` | Val Loss: `0.2048` | Val Accuracy: `93.38%` (Model checkpoint saved)
  * **Epoch 4**: Train Loss: `0.2200` | Val Loss: `0.2011` | Val Accuracy: `93.38%`
  * **Epoch 5**: Train Loss: `0.2231` | Val Loss: `0.1893` | Val Accuracy: **`93.64%`** (Model checkpoint saved)
* **Model Speed & Evaluation**:
  * Test Inference Duration: 16.307 seconds
  * Inference Throughput: **`118.78 FPS`**
  * Final Test Split Accuracy: **`0.9277`**
  * Macro F1-score: **`0.9262`**

---

### Neural Backbone Benchmark Dashboard

| Backbone | Accuracy | F1-Score | Params (M) | Size (MB) | Speed (FPS) | Train Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **resnet18\*** (ACTIVE) | 0.9277 | 0.9262 | 11.18 | 42.7 | 118.8 | 205.0 |
| **mobilenetv2** (Baseline) | 0.9280 | 0.9270 | 2.23 | 8.9 | 285.0 | 32.8 |

---

### Compressed Artifacts Export
All training assets, metric JSON reports, confusion matrices, Grad-CAM heatmaps, and the final cropped panorama are compressed and package-exported to:
`outputs/percv_artifacts.zip`
