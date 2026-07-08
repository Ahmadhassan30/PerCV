"""
Run panorama stitching pipeline (Task 3) and compare results against documented baseline.
"""

import sys
import os
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from percv_cv.config import CONFIG
from percv_cv.panorama import stitch_panorama


def discover_panorama_scene(scene_dir_arg: str | None) -> Path:
    """Resolve panorama scene directory dynamically or via arguments."""
    if scene_dir_arg:
        path = Path(scene_dir_arg)
        if path.exists():
            return path
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir_arg}")

    # Fallback search path in CONFIG input_root or /kaggle/input
    search_paths = [
        Path(CONFIG.input_root),
        Path("/kaggle/input"),
        Path("./datasets"),
        Path("scratch/dummy_panorama")
    ]

    for base_path in search_paths:
        if not base_path.exists():
            continue
        # Check if dummy matching scene exists first
        if (base_path / "middle.png").exists():
            return base_path

        # Search for panorama scene folder
        for p in base_path.rglob("*"):
            if "panorama" in p.name.lower() and p.is_dir():
                # Check for scene folder 'front' inside it
                for scene_sub in p.rglob(CONFIG.task3.scene):
                    if scene_sub.is_dir():
                        return scene_sub
                return p

    raise FileNotFoundError(
        "Could not dynamically discover panorama dataset. Please run dummy generator or specify path via --scene-dir."
    )


def main():
    parser = argparse.ArgumentParser(description="Task 3 Panorama Stitching pipeline.")
    parser.add_argument("--scene-dir", type=str, default=None, help="Path to scene images directory.")
    parser.add_argument("--anchor", type=str, default="middle", help="Filename token or key acting as the anchor frame.")
    parser.add_argument("--out", type=str, default="outputs/panorama", help="Path to save outputs.")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        scene_dir = discover_panorama_scene(args.scene_dir)
        print(f"Panorama scene resolved at: {scene_dir}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Collect images
    allowed_exts = {".jpg", ".jpeg", ".png"}
    img_paths = sorted([p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts])

    if not img_paths:
        print("ERROR: No images found in resolved scene directory.")
        sys.exit(1)

    # Label them dynamically
    images = {}
    anchor_key = "middle"

    if len(img_paths) == 2:
        # 2-image panorama
        images = {
            "left": cv2.imread(str(img_paths[0])),
            "right": cv2.imread(str(img_paths[1]))
        }
        anchor_key = "right" if "right" in args.anchor.lower() or "right" in img_paths[1].name.lower() else "left"
        print(f"Loaded 2 images for stitching. Anchor: '{anchor_key}'")
    else:
        # 3-image panorama (default)
        images = {
            "left": cv2.imread(str(img_paths[0])),
            "middle": cv2.imread(str(img_paths[1])),
            "right": cv2.imread(str(img_paths[2]))
        }
        anchor_key = "middle"
        print(f"Loaded 3 images for stitching. Anchor: '{anchor_key}'")

    # Run panorama stitching pipeline
    results = stitch_panorama(images, anchor_key, CONFIG.task3)

    # Save output panorama image
    cv2.imwrite(str(output_dir / "panorama.png"), results["panorama"])

    # Save weight maps for inspection
    weight_dir = output_dir / "weight_maps"
    weight_dir.mkdir(parents=True, exist_ok=True)
    for idx, w_map in enumerate(results["weight_maps"]):
        # Scale to [0, 255] for saving
        scaled_w = (w_map * 255).astype(np.uint8)
        cv2.imwrite(str(weight_dir / f"weight_map_{idx+1:02d}.png"), scaled_w)

    # Save pair_metrics.json
    with open(output_dir / "pair_metrics.json", "w") as f:
        json.dump(results["pair_metrics"], f, indent=4)

    # Compare with documented baseline (Left->Middle: 0.8628, Right->Middle: 0.8896, Avg: 0.8762)
    baseline_lm = 0.8628
    baseline_rm = 0.8896
    baseline_avg = 0.8762

    print("\n=== Panorama Stitching Executed ===")
    print(f"Output files saved to: {output_dir.resolve()}")
    
    for pair, stats in results["pair_metrics"].items():
        print(f"Pair '{pair}': Matches = {stats['total_matches']}, RANSAC Inliers = {stats['inliers']}, Inlier Ratio = {stats['inlier_ratio']:.4f}")
    
    detected_avg = results["average_inlier_ratio"]
    print(f"Average Inlier Ratio: {detected_avg:.4f} (Baseline: {baseline_avg:.4f})")

    # If it is a 3-image stitch of the 'front' scene, check deltas
    is_front = scene_dir.name == "front" or (scene_dir / "left.png").exists()
    if len(img_paths) == 3 and is_front:
        avg_delta = abs(detected_avg - baseline_avg)
        if avg_delta > 0.15:
            print("WARNING: Large deviation from documented baseline. Inspect keypoint features and homographies.")
        else:
            print("SUCCESS: Performance matches documented baseline within accepted tolerance.")


if __name__ == "__main__":
    main()
