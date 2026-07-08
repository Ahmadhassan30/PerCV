import numpy as np
import pytest
from percv_cv.lanes import lanes_quality_score


def create_line_segment(x1, y1, x2, y2):
    """Utility to create line segments matching OpenCV HoughLinesP output format."""
    return np.array([[[x1, y1, x2, y2]]], dtype=np.int32)


def test_empty_or_none():
    assert lanes_quality_score(None) == 0.0
    assert lanes_quality_score(np.array([])) == 0.0


def test_specific_angles():
    # 45 degrees: passes (25 <= 45 <= 75)
    line_45 = create_line_segment(0, 0, 10, 10)
    assert lanes_quality_score(line_45) == 1.0

    # 90 degrees (vertical): fails (neither bucket contains 90)
    line_90 = create_line_segment(0, 0, 0, 10)
    assert lanes_quality_score(line_90) == 0.0

    # 0 degrees (horizontal): fails
    line_0 = create_line_segment(0, 0, 10, 0)
    assert lanes_quality_score(line_0) == 0.0

    # 135 degrees: passes (105 <= 135 <= 155)
    line_135 = create_line_segment(0, 10, 10, 0)
    assert lanes_quality_score(line_135) == 1.0


def test_boundary_inclusive():
    # 25 degrees boundary: passes
    # tan(25 deg) = 0.466307658
    # y = 4664, x = 10000 -> angle is approx 25.004 degrees
    line_25 = create_line_segment(0, 0, 10000, 4664)
    assert lanes_quality_score(line_25) == 1.0

    # 75 degrees boundary: passes
    # tan(75 deg) = 3.7320508
    # y = 37319, x = 10000 -> angle is approx 74.999 degrees
    line_75 = create_line_segment(0, 0, 10000, 37319)
    assert lanes_quality_score(line_75) == 1.0

    # 105 degrees boundary: passes
    # tan(105 deg) = -3.7320508
    # y = 37320, x = -10000 -> angle is approx 105.0007 degrees
    line_105 = create_line_segment(0, 0, -10000, 37320)
    assert lanes_quality_score(line_105) == 1.0

    # 155 degrees boundary: passes
    # tan(155 deg) = -0.466307658
    # y = 4664, x = -10000 -> angle is approx 154.998 degrees
    line_155 = create_line_segment(0, 0, -10000, 4664)
    assert lanes_quality_score(line_155) == 1.0


def test_ratio_calculation():
    # 4 lines total: 2 pass, 2 fail -> score should be 0.5
    lines = np.array([
        [[0, 0, 10, 10]],   # 45 deg (pass)
        [[0, 10, 10, 0]],   # 135 deg (pass)
        [[0, 0, 0, 10]],    # 90 deg (fail)
        [[0, 0, 10, 0]]     # 0 deg (fail)
    ], dtype=np.int32)
    assert lanes_quality_score(lines) == 0.5
