# PerCV Pipeline Manifest

This manifest maps each task module to its corresponding configurations, expected output files, and baseline benchmark metrics.

| Task ID | Task Description | Config Keys | Output Files | Headline Metric (Baseline) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 1** | Edge-based Lane Detection | `task1` (num_images, Canny thresholds, Hough parameters) | `outputs/experiments/experiment_001/plots/lanes_sensitive.png`<br>`outputs/experiments/experiment_001/plots/lanes_balanced.png`<br>`outputs/experiments/experiment_001/plots/lanes_strict.png` | Mean Quality Score: **`0.1809`** | Missing |
| **Task 2** | SIFT Descriptor Matching | `task2` (lowe_ratios, default_lowe_ratio) | `outputs/experiments/experiment_001/plots/sift_matches.png`<br>`outputs/experiments/experiment_001/plots/sift_ratio_bar.png` | 348 matches (Ratio: **`0.4462`**) at Lowe threshold 0.75 | Missing |
| **Task 3** | From-Scratch Panorama Stitching | `task3` (scene, ransac_threshold, stitching_min_matches, distortion_det_threshold) | `outputs/experiments/experiment_001/plots/panorama.png`<br>`outputs/experiments/experiment_001/plots/panorama_visualization.png`<br>`outputs/experiments/experiment_001/metrics/task3_homographies.json`<br>`outputs/experiments/experiment_001/metrics/task3_params.csv` | Average Inlier Ratio: **`0.8762`** | Missing |
| **Task 4** | CNN Transfer Learning & Grad-CAM | `task4` (backbone, batch_size, epochs, learning_rate, weight_decay, gradcam_target_samples) | `outputs/experiments/experiment_001/plots/training_curves.png`<br>`outputs/experiments/experiment_001/plots/confusion_matrix.png`<br>`outputs/experiments/experiment_001/plots/gradcam_overlaid.png`<br>`outputs/experiments/experiment_001/models/model_best.pt` | Test Accuracy: **`0.9277`** (F1: **`0.9262`**) | Missing |
| **All** | Consolidated Zip Export | N/A | `outputs/percv_artifacts.zip` | N/A | Missing |
