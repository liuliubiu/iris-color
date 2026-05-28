"""取色与高光剔除（含调试用 mask）。"""

from dataclasses import dataclass

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
class IrisColorResult:
    """基于 CIELAB 的基础虹膜颜色判断。"""

    code: str
    label: str
    confidence: float
    reason: str


@dataclass
class SamplingMasks:
    """虹膜环带内各阶段像素 mask（bool 数组）。"""

    ring: np.ndarray
    highlight_in_ring: np.ndarray
    dark_in_ring: np.ndarray
    bright_in_ring: np.ndarray
    valid: np.ndarray


_COLOR_LABELS = {
    "light_blue": "浅蓝色",
    "darker_blue": "深蓝色",
    "green": "绿色",
    "brown": "棕色",
    "dark_brown": "深棕色",
}


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


def classify_iris_color(lab: LabResult, config: dict | None = None) -> IrisColorResult:
    """
    用 CIELAB 近似判断基础虹膜颜色。

    当前只覆盖论文色类中的纯色子集：浅蓝、深蓝、绿色、棕色、深棕色。
    复合色环类型需要完整虹膜分区后再加入。
    """
    thresholds = (config or {}).get("color_classification", {})
    light_blue_l_min = thresholds.get("light_blue_l_min", 55.0)
    blue_b_max = thresholds.get("blue_b_max", -8.0)
    green_a_max = thresholds.get("green_a_max", -3.0)
    green_b_min = thresholds.get("green_b_min", -5.0)
    green_b_max = thresholds.get("green_b_max", 35.0)
    dark_brown_l_max = thresholds.get("dark_brown_l_max", 36.0)

    l_star = lab.L
    a_star = lab.a
    b_star = lab.b

    if b_star <= blue_b_max and a_star <= 12.0:
        if l_star >= light_blue_l_min:
            confidence = min(0.95, 0.55 + (l_star - light_blue_l_min) / 35 + abs(b_star - blue_b_max) / 50)
            return IrisColorResult("light_blue", _COLOR_LABELS["light_blue"], round(confidence, 2), "b* 为负且 L* 较高")
        confidence = min(0.95, 0.55 + abs(b_star - blue_b_max) / 45 + (light_blue_l_min - l_star) / 45)
        return IrisColorResult("darker_blue", _COLOR_LABELS["darker_blue"], round(confidence, 2), "b* 为负且 L* 偏低")

    if a_star <= green_a_max and green_b_min <= b_star <= green_b_max:
        confidence = min(0.95, 0.55 + abs(a_star - green_a_max) / 25)
        return IrisColorResult("green", _COLOR_LABELS["green"], round(confidence, 2), "a* 偏负，符合绿色轴特征")

    if l_star <= dark_brown_l_max:
        confidence = min(0.95, 0.55 + (dark_brown_l_max - l_star) / 35 + max(b_star, 0) / 80)
        return IrisColorResult("dark_brown", _COLOR_LABELS["dark_brown"], round(confidence, 2), "L* 较低，整体偏深")

    confidence = min(0.9, 0.50 + max(b_star, 0) / 80 + max(a_star, 0) / 80)
    return IrisColorResult("brown", _COLOR_LABELS["brown"], round(confidence, 2), "未满足蓝/绿条件，按棕色系归类")
