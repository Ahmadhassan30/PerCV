# PerCV: Kaggle Computational Notebook Guide

This directory contains the Kaggle computational notebook `percv_kaggle.ipynb` for the **PerCV** project. This notebook runs the compute-heavy computer vision operations (edge detection, feature matching, progressive manual stitching, and CNN model training) using Kaggle's free GPU resources.

---

## Getting Started

### 1. Create a Kaggle Notebook
1. Go to [Kaggle](https://www.kaggle.com/) and sign in.
2. Click **New Notebook** (or **Create** -> **Notebook**).
3. Set the Notebook options:
   - **Accelerator**: Choose **GPU T4 x2** or **GPU P100** under settings (in the right panel under *Accelerator*).
   - **Language**: Python.

### 2. Import the Notebook File
1. Download `percv_kaggle.ipynb` from this folder to your local machine.
2. In your Kaggle notebook, click **File** -> **Import Notebook** in the top menu.
3. Upload `percv_kaggle.ipynb`.

---

## Configuring Dataset Inputs

The notebook uses placeholder paths pointing to `/kaggle/input/...` for Tasks 1-3 images and the Task 4 classification dataset. Before execution, you must upload your images/datasets to Kaggle and update the corresponding path variables in the notebook.

### 1. Uploading Images to Kaggle
We recommend creating a private Kaggle Dataset to host your project images.
1. Click **+ Add Data** (top right of the notebook editor).
2. Choose **Upload a new dataset**, set a name (e.g. `percv-project-data`), and drag-and-drop the following assets:
   - **Task 1**: At least 1 outdoor road perspective photograph (e.g., `road_view.jpg`).
   - **Task 2**: A pair of overlapping photos of the same scene under slightly different viewpoints/lighting (e.g., `scene_left.jpg` and `scene_right.jpg`).
   - **Task 3**: At least 3 horizontally overlapping photos of a wide scene (e.g., `view_left.jpg`, `view_middle.jpg`, and `view_right.jpg`).
3. Click **Create** to upload.

### 2. Uploading Task 4 Classification Dataset
Upload your multi-class image folders as another Kaggle dataset. The root folder should contain subdirectories for each class:
```text
dataset/
├── class_a/
│   ├── image_001.jpg
│   └── ...
├── class_b/
│   ├── image_001.jpg
│   └── ...
└── class_c/
    ├── image_001.jpg
    └── ...
```
1. Click **+ Add Data**, upload the directory structure, and click **Create**.

### 3. Setting Paths in the Notebook
After uploading, look at the **Input** section in Kaggle's right sidebar to copy the absolute paths of your uploaded files. Update the variables in **Step 2** and **Step 3** of the notebook:

- **Task 1-3 Images** (in the Data Loading Cell):
  ```python
  TASK1_IMAGE_PATH = "/kaggle/input/<dataset-slug>/road_view.jpg"
  TASK2_IMAGE1_PATH = "/kaggle/input/<dataset-slug>/scene_left.jpg"
  TASK2_IMAGE2_PATH = "/kaggle/input/<dataset-slug>/scene_right.jpg"
  TASK3_IMAGE_PATHS = [
      "/kaggle/input/<dataset-slug>/view_left.jpg",
      "/kaggle/input/<dataset-slug>/view_middle.jpg",
      "/kaggle/input/<dataset-slug>/view_right.jpg"
  ]
  ```

- **Task 4 Dataset** (in the Classification Data Loading Cell):
  ```python
  TASK4_DATASET_PATH = "/kaggle/input/<dataset-slug>/dataset_root"
  ```

> [!NOTE]
> If these paths do not exist, the notebook automatically generates **synthetic, high-quality test inputs and shapes** for all tasks so you can run the notebook top-to-bottom as a dry run.

---

## Executing the Notebook
1. Click **Run All** (or press `Ctrl+F9`) to run the notebook top-to-bottom.
2. Verify in the output of the first cell that PyTorch detects the active GPU device.
3. Review the logs:
   - Task 1 will plot edge outputs and save `task1_params.csv`.
   - Task 2 will plot keypoints and SIFT match overlays.
   - Task 3 will print estimated homographies, inlier ratios, and plot the alpha-blended panorama.
   - Task 4 will output training epochs, verify whether the validation accuracy surpasses the threshold ($\ge 70\%$ to pass, $\ge 90\%$ for full marks), plot confusion matrices, and render Grad-CAM visual heatmaps.

---

## Retrieving Artifacts

All figures, metrics, logs, and weights are saved in a structured `outputs/` directory in the Kaggle working space:
- `outputs/metrics.json` (Single source of truth metrics for report compiling)
- `outputs/training_curves.png` (CNN train/val loss and accuracy plots)
- `outputs/confusion_matrix.png` (CNN performance confusion matrices)
- `outputs/gradcam/` (6 Grad-CAM heatmap overlay figures)
- `outputs/task1/` (Edge/Line detection pipeline images and parameters CSV)
- `outputs/task2/` (SIFT keypoints, match figures, and parameters CSV)
- `outputs/task3/` (Estimated homography JSON, params CSV, and stitched panorama)

The final cell compresses this directory into a single archive:
**`outputs/percv_artifacts.zip`**

### Download via Kaggle UI
1. Look at the **Output** folder in Kaggle's right sidebar (under `/kaggle/working`).
2. Locate `outputs/percv_artifacts.zip`.
3. Hover over the file, click the three dots, and choose **Download**.
4. Unzip this file in your local environment; these artifacts will be consumed by the backend, frontend, and report writing tools.
