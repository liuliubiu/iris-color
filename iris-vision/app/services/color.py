"""取色与高光剔除（含调试用 mask）。"""

from dataclasses import dataclass
from typing import Tuple

import colorspacious
import cv2
import numpy as np


@dataclass
class LabResult:
    """Lab 中位数结果。"""

    L: float
    a: float
    b: float
    sample_pixel_count: int


@dataclass
class SamplingMasks:
    """虹膜环带内各阶段像素 mask（bool 数组）。"""

    ring: np.ndarray
    highlight_in_ring: np.ndarray
    dark_in_ring: np.ndarray
    bright_in_ring: np.ndarray
    valid: np.ndarray


def compute_sampling_masks(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    highlight_v_threshold: int = 240,
) -> SamplingMasks:
    """计算环带、环内高光/极暗/过亮像素、最终有效采样区域。"""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    ring = mask > 0
    highlight_in_ring = ring & (v_channel >= highlight_v_threshold)
    dark_in_ring = ring & (v_channel <= 8)
    bright_cutoff = max(min(highlight_v_threshold - 15, 235), 220)
    bright_in_ring = ring & (v_channel >= bright_cutoff)
    valid = ring & ~highlight_in_ring & ~dark_in_ring & ~bright_in_ring
    return SamplingMasks(
        ring=ring,
        highlight_in_ring=highlight_in_ring,
        dark_in_ring=dark_in_ring,
        bright_in_ring=bright_in_ring,
        valid=valid,
    )


def extract_iris_lab_median(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    highlight_v_threshold: int = 240,
) -> LabResult:
    """在 mask 区域内取色，剔除高光，返回 CIELAB 中位数。"""
    masks = compute_sampling_masks(image_bgr, mask, highlight_v_threshold)
    pixels_bgr = image_bgr[masks.valid]

    if len(pixels_bgr) == 0:
        raise ValueError("no_valid_pixels_after_highlight_removal")

    pixels_rgb = pixels_bgr[:, ::-1].astype(np.float64) / 255.0
    lab_array = colorspacious.cspace_convert(pixels_rgb, "sRGB1", "CIELab")

    return LabResult(
        L=float(np.median(lab_array[:, 0])),
        a=float(np.median(lab_array[:, 1])),
        b=float(np.median(lab_array[:, 2])),
        sample_pixel_count=int(len(pixels_bgr)),
    )
