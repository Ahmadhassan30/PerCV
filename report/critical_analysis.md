# Critical Analysis Report

> This document is a **skeleton** for human-written critical analysis content.
> Fill in each section with your own observations, failure-mode discussions,
> and rubric-aligned commentary.

---

## Task 1 — Lane Detection

### What worked
<!-- Discuss Canny sensitivity trade-offs, Hough parameter tuning -->

### Failure analysis
<!-- Mean quality score of 0.1809: urban clutter, off-angle false positives -->

---

## Task 2 — SIFT Feature Matching

### What worked
<!-- Lowe ratio sweep precision-recall curve behaviour -->

### Failure analysis
<!-- Sequence-dependent descriptor density, illumination invariance limits -->

---

## Task 3 — Panorama Stitching

### What worked
<!-- Independent homographies, distance-transform blending, auto-crop -->

### Failure analysis
<!-- Parallax violations, dynamic object ghosting, narrow overlap edge cases -->

---

## Task 4 — CNN Classification & Grad-CAM

### What worked
<!-- 93.64% val accuracy, ResNet18 vs MobileNetV2 comparison -->

### Failure analysis
<!-- Street vs. building confusion in urban canyons, Grad-CAM spatial focus -->

---

## Cross-Task Reflections

<!-- Overall pipeline design decisions, reproducibility, what you would change -->
