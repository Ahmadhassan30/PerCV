# Critical Analysis Report

> **Instruction to User**: This document is a pre-filled skeleton detailing methodology, results, and critical prompts-to-self based on the baseline metrics. Fill in each section with your own technical analysis, observations, and findings.

---

## 1. Methodology
<!--
Describe the mathematical and algorithmic implementations:
- Lane Detection: Gaussian blur kernel size (5x5) and standard deviation (1.0), Canny hysteresis threshold bounds (35/95, 75/155, 125/245), and Probabilistic Hough parameters (rho=1, theta=1°, threshold=45, min_line_len=30, max_line_gap=12).
- Feature Matching: SIFT keypoint extraction, BruteForce L2 kNN distance matcher, and Lowe's ratio test thresholds (0.60, 0.75, 0.90).
- Panorama Stitching: SIFT description, RANSAC homography estimation, translation projection, L2 Euclidean distance-transform blending weight maps, and contour-filled bounding box auto-cropping.
- CNN Classification: ResNet18 and MobileNetV2 classifiers with frozen backbones and replaced fully connected/classifier heads trained on Intel Scene Dataset. Hook-based Grad-CAM gradients extraction.
-->

---

## 2. Results
<!--
Analyze the quantitative baseline metrics compiled during evaluation:
- Lane Detection: Balanced preset mean quality score = 0.1809 across 20 frames.
- Feature Matching: Matches @ 0.75 ratio = 348 with RANSAC inlier ratio of 0.4462 (sequence i_ajuntament).
- Panorama Stitching: Average inlier ratio = 0.8762 (Left->Middle: 0.8628, Right->Middle: 0.8896).
- CNN Classification: ResNet18 test accuracy = 0.9277 (macro F1: 0.9262) at 118.78 FPS vs MobileNetV2 test accuracy = 0.9280 (macro F1: 0.9270) at 285.0 FPS.
-->

---

## 3. Critical Analysis
> [!NOTE]
> **Prompt-to-self**: Lane detection scored a low 0.1809 mean quality score. Explain how the non-semantic nature of Canny/Hough pipelines causes structural clutter (trees, signage, dynamic vehicles) to project false-positive lines outside realistic lane boundary slopes. Discuss if a semantic segmentation approach (e.g. U-Net, LaneNet) would eliminate these geometric outliers.

> [!NOTE]
> **Prompt-to-self**: Feature matching shows a direct trade-off when sweeping the Lowe ratio from 0.60 to 0.90. Analyze why matches increase but RANSAC inlier ratio declines, identifying the precision-recall boundary. Reflect on SIFT's robustness against extreme illumination and scale shifts.

---

## 4. Limitations
> [!NOTE]
> **Prompt-to-self**: Panorama warp stitching assumes a purely planar scene and a single center of projection. Discuss how parallax violations (due to camera translation) and dynamic object movement (ghosting) degrade stitching seams. Reflect on the limitations of contour-based auto-cropping on highly skewed canvas bounds.

> [!NOTE]
> **Prompt-to-self**: CNN classification suffers from street vs. building confusion in urban scene images. Reflect on how Grad-CAM activation heatmaps reveal the model's focus (e.g. vertical facades vs horizontal asphalt) and how this interpretability validates the choice of active model.

---

## 5. Conclusion
<!--
Summarize the comparative pipeline analysis. Highlight the performance/computational footprint trade-offs between ResNet18 and MobileNetV2, and outline future directions for pipeline scaling and optimization.
-->
