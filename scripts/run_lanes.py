"""
Run lane detection pipeline (Task 1) and compare results against documented baseline.
"""

import sys
import os
import argparse
import json
import csv
from pathlib import Path
import cv2
import numpy as np

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from percv_cv.config import CONFIG
from percv_cv.lanes import preprocess, detect_edges, detect_lines, lanes_quality_score, run_lane_detection


def discover_bdd100k(input_dir_arg: str | None) -> Path:
    """Find BDD100K images folder dynamically or via arguments."""
    if input_dir_arg:
        path = Path(input_dir_arg)
        if path.exists():
            return path
        raise FileNotFoundError(f"Input directory does not exist: {input_dir_arg}")

    # Fallback search path in CONFIG input_root or /kaggle/input
    search_paths = [
        Path(CONFIG.input_root),
        Path("/kaggle/input"),
        Path("./datasets"),
        Path(".")
    ]

    for base_path in search_paths:
        if not base_path.exists():
            continue
        # Search for bdd100k folders recursively up to 3 levels
        for p in base_path.rglob("*"):
            if "bdd100k" in p.name.lower() and p.is_dir():
                # Look for an images folder inside
                for sub in p.rglob("images"):
                    if sub.is_dir():
                        return sub
                return p

    raise FileNotFoundError(
        "Could not dynamically discover BDD100K dataset. Please mount it or specify path via --input-dir."
    )


def save_overlays(image_paths: list[Path], output_dir: Path, canny_pair: dict):
    """Draw overlay visualizations for a handful of samples (up to 3)."""
    overlay_dir = output_dir / "sample_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    samples = image_paths[:3]
    for idx, path in enumerate(samples):
        img = cv2.imread(str(path))
        if img is None:
            continue

        gray = preprocess(img, CONFIG.task1.gaussian_ksize, CONFIG.task1.gaussian_sigma)
        edges = detect_edges(gray, canny_pair["low"], canny_pair["high"])
        lines = detect_lines(
            edges,
            rho=CONFIG.task1.hough_rho,
            theta_deg=CONFIG.task1.hough_theta_deg,
            threshold=CONFIG.task1.hough_threshold,
            min_line_len=CONFIG.task1.hough_min_line_len,
            max_line_gap=CONFIG.task1.hough_max_line_gap
        )

        overlay = img.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = x2 - x1, y2 - y1
                angle = np.abs(np.arctan2(dy, dx) * 180.0 / np.pi)
                if angle > 180.0:
                    angle -= 180.0
                
                # Check if it falls inside expected lane slopes
                if (25.0 <= angle <= 75.0) or (105.0 <= angle <= 155.0):
                    color = (0, 255, 0)  # Green: passed
                else:
                    color = (0, 0, 255)  # Red: failed
                
                cv2.line(overlay, (x1, y1), (x2, y2), color, 2)

        out_name = f"sample_{idx+1:02d}_{canny_pair['label']}.png"
        cv2.imwrite(str(overlay_dir / out_name), overlay)


def main():
    parser = argparse.ArgumentParser(description="Task 1 Lane Detection pipeline.")
    parser.add_argument("--input-dir", type=str, default=None, help="Path to BDD100K images directory.")
    parser.add_argument("--out", type=str, default="outputs/lanes", help="Path to save outputs.")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        bdd_dir = discover_bdd100k(args.input_dir)
        print(f"BDD100K dataset resolved at: {bdd_dir}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Collect images recursively (limit to config's count of 20)
    allowed_exts = {".jpg", ".jpeg", ".png"}
    image_paths = []
    for p in bdd_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed_exts:
            image_paths.append(p)
            if len(image_paths) >= CONFIG.task1.num_images:
                break

    if not image_paths:
        print("ERROR: No images found in BDD100K directory.")
        sys.exit(1)

    print(f"Collected {len(image_paths)} images for lane detection evaluation.")

    # Run detection
    results = run_lane_detection(image_paths, CONFIG.task1)

    # Save detailed CSV using built-in csv module
    csv_path = output_dir / "quality_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "preset", "lines_detected", "quality_score"])
        for preset_label, details in results["results_by_preset"].items():
            for d in details:
                writer.writerow([
                    Path(d["image_path"]).name,
                    preset_label,
                    d["num_lines"],
                    d["quality_score"]
                ])

    # Save summary.json
    summary = {
        "mean_quality_scores": results["preset_mean_scores"],
        "processed_frames": len(image_paths)
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    # Save overlay samples for each preset configuration
    for pair in CONFIG.task1.canny_threshold_pairs:
        save_overlays(image_paths, output_dir, {
            "low": pair.low,
            "high": pair.high,
            "label": pair.label
        })

    # Compare with documented baseline (0.1809)
    baseline_score = 0.1809
    detected_score = results["overall_mean_quality_score"]
    delta = abs(detected_score - baseline_score)

    print("\n=== Lane Detection Module Executed ===")
    print(f"Output files saved to: {output_dir.resolve()}")
    for preset_label, score in results["preset_mean_scores"].items():
        print(f"Preset '{preset_label}': Mean Quality Score = {score:.4f}")
    
    print(f"Overall Quality Score (Balanced Canny Preset): {detected_score:.4f}")
    print(f"Baseline Quality Score (Documented Reference): {baseline_score:.4f}")
    print(f"Absolute Score Delta: {delta:.4f}")
    
    if delta > 0.05:
        print("WARNING: Large deviation from documented baseline. Inspect data files and parameters.")
    else:
        print("SUCCESS: Performance matches documented baseline within accepted tolerance.")


if __name__ == "__main__":
    main()
