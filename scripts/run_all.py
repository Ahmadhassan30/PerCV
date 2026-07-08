"""Run all four pipelines and regenerate artifacts/*.json for diffing against baseline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lanes import main as run_lanes
from scripts.run_matching import main as run_matching
from scripts.run_panorama import main as run_panorama
from scripts.run_classify import main as run_classify


def main():
    print("=== PerCV Full Pipeline Run ===")
    for name, fn in [
        ("Lane Detection", run_lanes),
        ("SIFT Matching", run_matching),
        ("Panorama Stitching", run_panorama),
        ("CNN Classification", run_classify),
    ]:
        print(f"\n--- {name} ---")
        try:
            fn()
        except NotImplementedError as e:
            print(f"  [SKIPPED] {e}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
