"""取色与高光剔除（含调试用 mask）。"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    hue: str = ""
    depth: str = ""


@dataclass
class SamplingMasks:
    """虹膜环带内各阶段像素 mask（bool 数组）。"""

    ring: np.ndarray
    highlight_in_ring: np.ndarray
    dark_in_ring: np.ndarray
    bright_in_ring: np.ndarray
    valid: np.ndarray


_HUE_NAMES = {
    "blue": "蓝",
    "green": "绿",
    "brown": "棕",
}

_HUE_COLOR_SUFFIX = {
    "blue": "蓝色",
    "green": "绿色",
    "brown": "棕色",
}

_DEPTH_CODES = {
    ("blue", "light"): "light_blue",
    ("blue", "medium"): "blue",
    ("blue", "dark"): "dark_blue",
    ("green", "light"): "light_green",
    ("green", "medium"): "green",
    ("green", "dark"): "dark_green",
    ("brown", "light"): "light_brown",
    ("brown", "medium"): "brown",
    ("brown", "dark"): "dark_brown",
}

_DEPTH_LABEL_PREFIX = {
    "light": "浅",
    "medium": "",
    "dark": "深",
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


def filter_ring_mask_sectors(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    center: Tuple[int, int],
    sector_count: int = 36,
    v_dev_max: float = 45.0,
    min_keep_ratio: float = 0.40,
) -> Tuple[np.ndarray, int]:
    """
    按角度把环带分成扇区，剔除亮度中位数偏离全环中位数过多的扇区。

    实拍图虹膜上部常被眼皮/睫毛遮挡，环带完整圆环会采到皮肤（偏亮）或
    睫毛丛（偏暗）；异常扇区整块剔除比逐像素过滤更稳。保留扇区不足
    min_keep_ratio 时视为过滤失败，返回原 mask（kept=0 表示未过滤）。
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask, 0

    hsv_v = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    values = hsv_v[ys, xs].astype(np.float32)

    cx, cy = float(center[0]), float(center[1])
    angles = np.arctan2(ys.astype(np.float32) - cy, xs.astype(np.float32) - cx)
    sector_idx = ((angles + np.pi) / (2 * np.pi) * sector_count).astype(np.int32)
    sector_idx = np.clip(sector_idx, 0, sector_count - 1)

    medians = np.full(sector_count, np.nan, dtype=np.float32)
    for s in range(sector_count):
        sel = sector_idx == s
        if np.any(sel):
            medians[s] = float(np.median(values[sel]))

    present = ~np.isnan(medians)
    if not np.any(present):
        return mask, 0
    global_median = float(np.median(medians[present]))
    keep_sector = present & (np.abs(medians - global_median) <= v_dev_max)

    kept = int(np.count_nonzero(keep_sector))
    if kept < max(int(sector_count * min_keep_ratio), 1):
        return mask, 0

    keep_pixel = keep_sector[sector_idx]
    filtered = np.zeros_like(mask)
    filtered[ys[keep_pixel], xs[keep_pixel]] = 255
    return filtered, kept


def extract_iris_lab_median(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    highlight_v_threshold: int = 240,
    sample_cap: int = 0,
    masks: Optional[SamplingMasks] = None,
    mad_trim: float = 0.0,
) -> LabResult:
    """在 mask 区域内取色，剔除高光，返回 CIELAB 中位数。

    masks 可传入已算好的 SamplingMasks，避免重复的全图 HSV 转换。
    sample_cap > 0 且有效像素超过该上限时，固定随机种子抽样后再做 CIELAB 转换：
    中位数对抽样稳健，大图可省去十几万像素的 colorspacious 转换，显著提速。
    mad_trim > 0 时按 L* 中位数 ± mad_trim×MAD 修剪离群像素（睫毛暗像素、
    残余高光）后再取中位数。
    """
    if masks is None:
        masks = compute_sampling_masks(image_bgr, mask, highlight_v_threshold)
    pixels_bgr = image_bgr[masks.valid]

    total = len(pixels_bgr)
    if total == 0:
        raise ValueError("no_valid_pixels_after_highlight_removal")

    if sample_cap and total > sample_cap:
        rng = np.random.default_rng(12345)
        idx = rng.choice(total, size=sample_cap, replace=False)
        pixels_bgr = pixels_bgr[idx]

    pixels_rgb = pixels_bgr[:, ::-1].astype(np.float64) / 255.0
    lab_array = colorspacious.cspace_convert(pixels_rgb, "sRGB1", "CIELab")

    if mad_trim > 0 and len(lab_array) >= 100:
        l_channel = lab_array[:, 0]
        median_l = float(np.median(l_channel))
        mad = float(np.median(np.abs(l_channel - median_l)))
        if mad > 1e-6:
            inliers = np.abs(l_channel - median_l) <= mad_trim * mad
            # 修剪过狠说明分布双峰（遮挡未除净），保守起见不修剪
            if int(inliers.sum()) >= max(int(len(lab_array) * 0.5), 50):
                lab_array = lab_array[inliers]

    return LabResult(
        L=float(np.median(lab_array[:, 0])),
        a=float(np.median(lab_array[:, 1])),
        b=float(np.median(lab_array[:, 2])),
        sample_pixel_count=int(total),
    )


def _classify_hue(l_star: float, a_star: float, b_star: float, hue_cfg: dict) -> Tuple[str, float, str]:
    """根据 a*/b* 判定色相，返回 (hue, confidence, reason)。"""
    blue_b_max = hue_cfg.get("blue_b_max", -8.0)
    blue_a_max = hue_cfg.get("blue_a_max", 12.0)
    green_a_max = hue_cfg.get("green_a_max", -3.0)
    green_b_min = hue_cfg.get("green_b_min", -5.0)
    green_b_max = hue_cfg.get("green_b_max", 35.0)

    if b_star <= blue_b_max and a_star <= blue_a_max:
        confidence = min(0.95, 0.55 + abs(b_star - blue_b_max) / 50 + max(blue_a_max - a_star, 0) / 40)
        return "blue", round(confidence, 2), f"b*≤{blue_b_max} 且 a*≤{blue_a_max}，判定为蓝色系"

    if a_star <= green_a_max and green_b_min <= b_star <= green_b_max:
        confidence = min(0.95, 0.55 + abs(a_star - green_a_max) / 25)
        return "green", round(confidence, 2), f"a*≤{green_a_max} 且 {green_b_min}≤b*≤{green_b_max}，判定为绿色系"

    confidence = min(0.9, 0.50 + max(b_star, 0) / 80 + max(a_star, 0) / 80)
    return "brown", round(confidence, 2), "未满足蓝/绿条件，按棕色系归类"


def map_l_star_to_depth(l_star: float, boundaries: List[float]) -> Tuple[str, float]:
    """
    将 L* 映射到 light / medium / dark 三档。

    boundaries 为 2 个递减上界：[b1, b2]
    - light:  L* > b1
    - medium: b2 < L* <= b1
    - dark:   L* <= b2

    恰在边界 → 取较深档。
    """
    b1, b2 = boundaries

    if l_star > b1:
        depth = "light"
        distance = l_star - b1
    elif l_star > b2:
        depth = "medium"
        distance = min(abs(l_star - b1), abs(l_star - b2))
    else:
        depth = "dark"
        distance = b2 - l_star

    max_span = max(boundaries[0] - boundaries[-1], 1.0)
    confidence = min(1.0, max(0.3, distance / (max_span / 2)))
    return depth, round(confidence, 2)


def _build_color_label(hue: str, depth: str) -> str:
    prefix = _DEPTH_LABEL_PREFIX[depth]
    suffix = _HUE_COLOR_SUFFIX[hue]
    return f"{prefix}{suffix}"


def classify_iris_color(lab: LabResult, config: dict | None = None) -> IrisColorResult:
    """
    用 CIELAB 判断虹膜颜色：先定色相（a*/b*），再定深浅（L* 三档）。

    色相与深浅阈值均在 config/grade_thresholds.yaml 的 color_classification 段配置，
    与 Pan Grade 1–5 的 grade.boundaries 相互独立。
    """
    color_cfg = (config or {}).get("color_classification", {})
    hue_cfg = color_cfg.get("hue", {})
    depth_cfg = color_cfg.get("depth", {})
    depth_boundaries = depth_cfg.get("boundaries", [21.0, 13.0])

    l_star = lab.L
    a_star = lab.a
    b_star = lab.b

    hue, hue_conf, hue_reason = _classify_hue(l_star, a_star, b_star, hue_cfg)
    depth, depth_conf = map_l_star_to_depth(l_star, depth_boundaries)

    code = _DEPTH_CODES[(hue, depth)]
    label = _build_color_label(hue, depth)
    confidence = round(min(hue_conf, depth_conf), 2)

    b1, b2 = depth_boundaries
    depth_desc = {"light": f"L*>{b1}", "medium": f"{b2}<L*≤{b1}", "dark": f"L*≤{b2}"}[depth]
    reason = f"{hue_reason}；{depth_desc}（{_DEPTH_LABEL_PREFIX[depth] or '中'}{_HUE_NAMES[hue]}）"

    return IrisColorResult(
        code=code,
        label=label,
        confidence=confidence,
        reason=reason,
        hue=hue,
        depth=depth,
    )
