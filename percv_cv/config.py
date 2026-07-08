from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

@dataclass(frozen=True)
class CannyThresholdPair:
    low: int
    high: int
    label: str

@dataclass(frozen=True)
class Task1Config:
    num_images: int = 20
    gaussian_ksize: Tuple[int, int] = (5, 5)
    gaussian_sigma: float = 1.0
    canny_threshold_pairs: List[CannyThresholdPair] = field(default_factory=lambda: [
        CannyThresholdPair(low=35, high=95, label="sensitive"),
        CannyThresholdPair(low=75, high=155, label="balanced"),
        CannyThresholdPair(low=125, high=245, label="strict")
    ])
    hough_rho: int = 1
    hough_theta_deg: float = 1.0
    hough_threshold: int = 45
    hough_min_line_len: int = 30
    hough_max_line_gap: int = 12

@dataclass(frozen=True)
class Task2Config:
    lowe_ratios: List[float] = field(default_factory=lambda: [0.60, 0.75, 0.90])
    default_lowe_ratio: float = 0.75

@dataclass(frozen=True)
class Task3Config:
    scene: str = "front"
    ransac_threshold: float = 5.0
    stitching_min_matches: int = 10
    distortion_det_threshold: float = 0.1

@dataclass(frozen=True)
class Task4Config:
    backbone: str = "resnet18"
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradcam_target_samples: int = 4

@dataclass(frozen=True)
class PipelineConfig:
    experiment_id: str = "experiment_001"
    seed: int = 42
    output_root: str = "outputs"
    input_root: str = "/kaggle/input"
    task1: Task1Config = Task1Config()
    task2: Task2Config = Task2Config()
    task3: Task3Config = Task3Config()
    task4: Task4Config = Task4Config()

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass structure to a standard dictionary format matching CONFIG."""
        return {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "output_root": self.output_root,
            "input_root": self.input_root,
            "task1": {
                "num_images": self.task1.num_images,
                "gaussian_ksize": self.task1.gaussian_ksize,
                "gaussian_sigma": self.task1.gaussian_sigma,
                "canny_threshold_pairs": [
                    {"low": p.low, "high": p.high, "label": p.label}
                    for p in self.task1.canny_threshold_pairs
                ],
                "hough_rho": self.task1.hough_rho,
                "hough_theta_deg": self.task1.hough_theta_deg,
                "hough_threshold": self.task1.hough_threshold,
                "hough_min_line_len": self.task1.hough_min_line_len,
                "hough_max_line_gap": self.task1.hough_max_line_gap,
            },
            "task2": {
                "lowe_ratios": self.task2.lowe_ratios,
                "default_lowe_ratio": self.task2.default_lowe_ratio,
            },
            "task3": {
                "scene": self.task3.scene,
                "ransac_threshold": self.task3.ransac_threshold,
                "stitching_min_matches": self.task3.stitching_min_matches,
                "distortion_det_threshold": self.task3.distortion_det_threshold,
            },
            "task4": {
                "backbone": self.task4.backbone,
                "batch_size": self.task4.batch_size,
                "epochs": self.task4.epochs,
                "learning_rate": self.task4.learning_rate,
                "weight_decay": self.task4.weight_decay,
                "gradcam_target_samples": self.task4.gradcam_target_samples,
            }
        }

# Instantiate default configuration matching documented parameters
CONFIG = PipelineConfig()
