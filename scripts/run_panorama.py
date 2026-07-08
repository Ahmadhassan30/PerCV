"""Run panorama stitching pipeline (Task 3) and write results to artifacts/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from percv_cv.config import CONFIG


def main():
    print(f"[run_panorama] Config: {CONFIG.task3}")
    raise NotImplementedError("Algorithm logic deferred to Prompt 2+")


if __name__ == "__main__":
    main()
